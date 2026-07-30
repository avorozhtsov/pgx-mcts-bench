from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax
import numpy as np
from pgx.go import Go

from pgx_mcts_bench.config import AnyGameConfig, BraidGameConfig, GameConfig, pick_stage


@dataclass(frozen=True)
class Transition:
    state: Any
    observation: np.ndarray
    legal_actions: np.ndarray
    reward: float
    terminated: bool
    player: int
    # Search-visible metadata. MuZero's learned-rules path needs to know how far
    # a game has run and whether it is drifting toward a pass-pass end, and it
    # cannot read those off a latent state. Every adapter fills them in so that
    # `search.py` never has to reach into a game-specific Pgx state.
    move_count: int = 0
    consecutive_passes: int = 0


class GameAdapter(Protocol):
    """What `NeuralMCTS` and the training loop require of a game.

    Deliberately small: exact transitions, an observation in the side-to-move
    frame, a legality mask, and terminal rewards. Anything game-specific beyond
    that belongs in the adapter, not in search or training.
    """

    config: AnyGameConfig

    def reset(self, seed: int) -> Transition: ...

    def step(self, state: Any, action: int) -> Transition: ...

    def final_rewards(self, state: Any) -> np.ndarray: ...

    def state_info(self, state: Any) -> dict[str, int]:
        """Search-visible scalars for a state, without building an observation.

        Returns `player`, `move_count` and `consecutive_passes` as a kwargs dict
        for `Node`.
        """
        ...

    def first_role_player(self, state: Any) -> int:
        """Player id holding the asymmetric first role (Black / Scrambler).

        Pgx randomises external player ids, so the arena has to ask rather than
        assume that player 0 holds the role.
        """
        ...

    def unwrap(self, state: Any) -> Any:
        """The underlying Pgx state. Wrappers that carry extra state override it."""
        return state


class Go6x6:
    """Thin non-jitted Pgx adapter with observations in the side-to-move frame."""

    def __init__(self, config: GameConfig):
        if config.board_size != 6:
            raise ValueError("This benchmark is intentionally pinned to 6x6 Go")
        self.config = config
        self.env = Go(
            size=config.board_size,
            komi=config.komi,
            history_length=config.history_length,
            max_terminal_steps=config.max_moves,
        )
        # AlphaZero invokes the exact transition inside every simulation. Without
        # JIT this Python-loop benchmark spends nearly all its time dispatching
        # the many small JAX operations that implement one Go move.
        self._init = jax.jit(self.env.init)
        self._step = jax.jit(self.env.step)

    def reset(self, seed: int) -> Transition:
        state = self._init(jax.random.PRNGKey(seed))
        return self._view(state, reward=0.0)

    def step(self, state: Any, action: int) -> Transition:
        if not bool(self._legal_actions(state)[action]):
            raise ValueError(f"Illegal action {action}")
        actor = int(np.asarray(state.current_player))
        next_state = self._step(state, np.int32(action))
        rewards = np.asarray(next_state.rewards, dtype=np.float32)
        return self._view(next_state, reward=float(rewards[actor]))

    def _view(self, state: Any, reward: float) -> Transition:
        observation = np.asarray(state.observation, dtype=np.float32)
        size = self.config.board_size
        consecutive_passes = int(np.asarray(state._x.consecutive_pass_count))
        move_count = int(np.asarray(state._x.step_count))
        previous_pass = float(consecutive_passes > 0)
        move_fraction = min(float(move_count) / self.config.max_moves, 1.0)
        metadata = np.stack(
            [
                np.full((size, size), previous_pass, dtype=np.float32),
                np.full((size, size), move_fraction, dtype=np.float32),
            ],
            axis=-1,
        )
        return Transition(
            state=state,
            observation=np.concatenate([observation, metadata], axis=-1),
            legal_actions=self._legal_actions(state),
            reward=reward,
            terminated=bool(np.asarray(state.terminated)),
            player=int(np.asarray(state.current_player)),
            move_count=move_count,
            consecutive_passes=consecutive_passes,
        )

    def final_rewards(self, state: Any) -> np.ndarray:
        return np.asarray(state.rewards, dtype=np.float32)

    def state_info(self, state: Any) -> dict[str, int]:
        return {
            "player": int(np.asarray(state.current_player)),
            "move_count": int(np.asarray(state._x.step_count)),
            "consecutive_passes": int(np.asarray(state._x.consecutive_pass_count)),
        }

    def first_role_player(self, state: Any) -> int:
        # current_player at reset is Black, so use the preserved mapping rather
        # than assuming player 0 is Black.
        return int(np.asarray(state._player_order[0]))

    def _legal_actions(self, state: Any) -> np.ndarray:
        legal = np.asarray(state.legal_action_mask, dtype=bool).copy()
        move_count = int(np.asarray(state._x.step_count))
        if move_count < self.config.min_moves_before_pass:
            legal[-1] = False
        return legal


class BraidUnknotGame:
    """Adapter for the `rf_knots` Scrambler vs. Simplifier braid environment.

    Two things differ from Go and both matter to search:

    1. Players do **not** alternate. The Scrambler moves `K` times in a row, then
       the Simplifier moves up to `M` times. `Transition.player` is therefore the
       only reliable source of whose value a node holds, and `search.py` flips
       the backed-up sign on player change rather than on every ply.
    2. The observation is a word, not a board. It is presented as a `1 x L`
       image with one-hot letter channels so the existing convolutional stack
       applies unchanged.
    """

    def __init__(self, config: BraidGameConfig):
        from rf_knots.env import BraidUnknot

        self.config = config
        self.env = BraidUnknot(config.to_braid_config())
        self._init = jax.jit(self.env.init)
        self._step = jax.jit(self.env.step)
        self.generator = None
        if config.generator_max_crossings or config.generator_max_scramble:
            from rf_knots.generator import GradedGenerator

            self.generator = GradedGenerator(
                config.to_braid_config(),
                max_crossings=config.generator_max_crossings,
                positive_braids=config.generator_positive_braids,
                positive_seed=config.generator_positive_seed,
            )

    def reset(self, seed: int) -> Transition:
        if self.generator is None:
            state = self._init(jax.random.PRNGKey(seed))
            return self._view(state, reward=0.0)
        return self._view(self._generated(seed)[0], reward=0.0)

    def _generated(self, seed: int):
        """An instance from the graded generator, with log(A/B) sampled.

        There is no Scrambler phase here: the instance is *given*, the solver
        starts immediately, and the source knot's unknotting number is known.
        """
        rng = np.random.default_rng(seed)
        source, moves = pick_stage(self.config, self.generator, rng)
        instance = self.generator.generate(source, moves, rng)
        low, high = self.config.log_ratio_range
        log_ratio = float(rng.uniform(low, high)) if high > low else low
        state = self.env.init_from_word(
            list(instance.word), instance.strands, log_ratio=log_ratio
        )
        return state, source

    def step(self, state: Any, action: int) -> Transition:
        if not bool(np.asarray(state.legal_action_mask)[action]):
            raise ValueError(f"Illegal action {action}")
        actor = int(np.asarray(state.current_player))
        next_state = self._step(state, np.int32(action))
        rewards = np.asarray(next_state.rewards, dtype=np.float32)
        return self._view(next_state, reward=float(rewards[actor]))

    def from_word(
        self, word: list[int], strands: int, log_ratio: float = 0.0
    ) -> Transition:
        """Start at an externally supplied instance, with the solver to move.

        Same signature on every adapter, so evaluation code never has to know
        whether it is talking to the parallel or the moving-window formulation.
        """
        return self._view(
            self.env.init_from_word(word, strands, log_ratio=log_ratio), reward=0.0
        )

    def _view(self, state: Any, reward: float) -> Transition:
        observation = np.asarray(state.observation, dtype=np.float32)
        # (L, C) -> (1, L, C): a one-row image, so the shared Conv2d stack works.
        observation = observation.reshape(1, self.config.max_len, self.config.observation_channels)
        move_count = int(np.asarray(state._step_count))
        return Transition(
            state=state,
            observation=observation,
            legal_actions=np.asarray(state.legal_action_mask, dtype=bool).copy(),
            reward=reward,
            terminated=bool(np.asarray(state.terminated)),
            player=int(np.asarray(state.current_player)),
            move_count=move_count,
            # There is no pass-pass ending here; the phase budget is the clock.
            consecutive_passes=0,
        )

    def final_rewards(self, state: Any) -> np.ndarray:
        return np.asarray(state.rewards, dtype=np.float32)

    def state_info(self, state: Any) -> dict[str, int]:
        return {
            "player": int(np.asarray(state.current_player)),
            "move_count": int(np.asarray(state._step_count)),
            "consecutive_passes": 0,
        }

    def first_role_player(self, state: Any) -> int:
        """The Scrambler, which is this game's analogue of Black."""
        return int(np.asarray(state._scrambler))

    def unwrap(self, state: Any) -> Any:
        return state


def make_game(config: AnyGameConfig) -> GameAdapter:
    if isinstance(config, BraidGameConfig):
        if config.serial_window:
            from pgx_mcts_bench.serial_braid import SerialBraidGame

            return SerialBraidGame(config)
        return BraidUnknotGame(config)
    if isinstance(config, GameConfig):
        return Go6x6(config)
    raise ValueError(f"Unknown game configuration: {config!r}")
