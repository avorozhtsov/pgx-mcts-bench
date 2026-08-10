from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax
import numpy as np
from pgx.go import Go

from pgx_mcts_bench.config import AnyGameConfig, BraidGameConfig, GameConfig


def sample_log_ratio(config: BraidGameConfig, rng: np.random.Generator) -> float:
    """Sample either an explicit objective or the historical continuous range."""
    if config.objective_ratio_choices:
        choices = np.asarray(config.objective_ratio_choices, dtype=np.float64)
        if np.any(choices <= 0):
            raise ValueError("objective ratio choices must be positive")
        if config.objective_ratio_weights:
            weights = np.asarray(config.objective_ratio_weights, dtype=np.float64)
            if weights.shape != choices.shape or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("objective ratio weights must be non-negative and aligned")
            probabilities = weights / weights.sum()
        else:
            probabilities = None
        return float(np.log(rng.choice(choices, p=probabilities)))
    low, high = config.log_ratio_range
    return float(rng.uniform(low, high)) if high > low else low


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
    # Empty while an episode is live. Budget-conditioned wrappers distinguish
    # objective censoring from the environment's ordinary move-clock failure.
    termination_reason: str = ""


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

    def value_potential(self, state: Any, player: int) -> float:
        """Player-perspective potential used by optional reward shaping."""
        ...

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

    def semantic_move_count(self, state: Any) -> int:
        """Portable solution moves accrued in the shared environment."""
        ...

    def native_ply_count(self, state: Any) -> int:
        """All controller/environment plies consumed by this scientist."""
        ...

    def internal_ply_count(self, state: Any) -> int:
        """Scientist-specific plies that do not enter the semantic witness."""
        ...


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

    def value_potential(self, state: Any, player: int) -> float:
        del state, player
        return 0.0

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

    def semantic_move_count(self, state: Any) -> int:
        return int(np.asarray(state._x.step_count))

    def native_ply_count(self, state: Any) -> int:
        return int(np.asarray(state._x.step_count))

    def internal_ply_count(self, state: Any) -> int:
        del state
        return 0

    def _legal_actions(self, state: Any) -> np.ndarray:
        legal = np.asarray(state.legal_action_mask, dtype=bool).copy()
        move_count = int(np.asarray(state._x.step_count))
        if move_count < self.config.min_moves_before_pass:
            legal[-1] = False
        return legal


def make_game(config: AnyGameConfig) -> GameAdapter:
    if isinstance(config, BraidGameConfig):
        from pgx_mcts_bench.serial_braid import SerialBraidGame

        return SerialBraidGame(config)
    if isinstance(config, GameConfig):
        return Go6x6(config)
    raise ValueError(f"Unknown game configuration: {config!r}")
