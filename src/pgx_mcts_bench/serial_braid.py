"""Serial (moving-window) view of the braid environment.

The parallel formulation scores every position at once: the policy head emits a
logit per (position, move-type), so the action space is `O(L)` and a decision
costs one forward pass over the whole word. That is the right trade while a
full-length pass is affordable.

This is the other formulation. A **head** points at one position; the agent sees
only a window of width `w` around it and may either act at the head or shift the
head by `w/2`. The consequences:

* the action space is `2N + 8` -- **independent of L**. At N=5 that is 18
  against 388, and it does not grow when the word does;
* the network never sees the whole word, so nothing in it depends on `L` either;
* but reaching a distant site costs *actions*, and in MCTS an action is a ply of
  search depth. That is the price, and measuring it is the point.

Shifts consume budget like any other move. The alternative -- free shifts -- makes
the game non-terminating, and it would also hide exactly the cost we want to
measure. Serial variants compensate with a larger `simplify_budget`.

The word is cyclic, so the head wraps and a window may show the same letter more
than once when the word is shorter than `w`. That is correct rather than a
degenerate case: the necklace really does repeat.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

from pgx_mcts_bench.config import BraidGameConfig
from pgx_mcts_bench.game import Transition

# Serial action layout, with G = max_strands - 1 generators.
#   0                REDUCE at head
#   1                COMMUTE at head
#   2                BRAID at head
#   3 .. 3+2G-1      INSERT at head, (generator, sign)
#   +0               DESTABILIZE          (position-free)
#   +1               STABILIZE_POS
#   +2               STABILIZE_NEG
#   +3               PASS
#   +4               CROSSING_CHANGE at head
#   +5               SHIFT_LEFT   by w/2
#   +6               SHIFT_RIGHT  by w/2
SERIAL_REDUCE = 0
SERIAL_COMMUTE = 1
SERIAL_BRAID = 2
SERIAL_INSERT = 3


def serial_action_size(max_strands: int, act_width: int = 1) -> int:
    """Actions for a window agent.

    `act_width` is how many window offsets the agent may act at. 1 means "only at
    the head" -- the position lives entirely in the state and reaching a
    neighbour costs a shift. Larger values let one ply act anywhere in the
    visible window, which costs a wider head but no plies.

    Either way the count is independent of L, which is the whole point.
    """
    positional = 3 + 2 * (max_strands - 1) + 1  # reduce/commute/braid, inserts, crossing
    return act_width * positional + 6           # + 4 singletons + 2 shifts


def serial_action_names(max_strands: int) -> list[str]:
    names = ["REDUCE", "COMMUTE", "BRAID"]
    for generator in range(1, max_strands):
        for sign in ("+", "-"):
            names.append(f"INSERT(s{generator}{sign})")
    names += [
        "DESTABILIZE",
        "STABILIZE_POS",
        "STABILIZE_NEG",
        "PASS",
        "CROSSING_CHANGE",
        "SHIFT_LEFT",
        "SHIFT_RIGHT",
    ]
    return names


class SerialBraidGame:
    """Adapter presenting the braid environment through a moving window.

    State is `(pgx_state, head)`. `GameAdapter` treats state opaquely, so search
    and training need no changes -- this drops into the existing sweep as another
    variant, scored on the same anchors.
    """

    def __init__(self, config: BraidGameConfig):
        from rf_knots.actions import (
            BRAID,
            COMMUTE,
            CROSSING_CHANGE,
            DESTABILIZE,
            INSERT,
            PASS,
            REDUCE,
            STABILIZE_NEG,
            STABILIZE_POS,
        )
        from rf_knots.env import BraidUnknot

        if config.serial_window < 3:
            raise ValueError("serial_window must be at least 3")
        self.config = config
        self.window = config.serial_window
        self.act_width = min(max(config.serial_act_width, 1), config.serial_window)
        self.stride = max(1, config.serial_window // 2)
        self.env = BraidUnknot(config.to_braid_config())
        self.spec = self.env.spec
        self._init = jax.jit(self.env.init)
        self._step = jax.jit(self.env.step)
        self.generator = None
        if config.generator_max_crossings or config.generator_max_scramble:
            from rf_knots.generator import GradedGenerator

            self.generator = GradedGenerator(
                config.to_braid_config(), max_crossings=config.generator_max_crossings
            )

        generators = config.max_strands - 1
        self._positional_kinds = [REDUCE, COMMUTE, BRAID]
        self._inserts = [
            (generator, sign)
            for generator in range(1, generators + 1)
            for sign in (1, -1)
        ]
        self._singletons = [DESTABILIZE, STABILIZE_POS, STABILIZE_NEG, PASS]
        self._crossing = CROSSING_CHANGE
        self._insert_kind = INSERT
        self.num_actions = serial_action_size(config.max_strands, self.act_width)
        self._per_offset = 3 + 2 * generators + 1
        self._singleton_base = self.act_width * self._per_offset
        self._shift_left = self.num_actions - 2
        self._shift_right = self.num_actions - 1

    # -- action translation ---------------------------------------------------

    def underlying_action(self, action: int, head: int, length: int) -> int | None:
        """Map a serial action to an action of the wrapped environment.

        `None` means a head shift, which the wrapped environment knows nothing
        about. Offsets wrap: the word is a necklace.
        """
        if action >= self._shift_left:
            return None
        if action >= self._singleton_base:
            return self.spec.encode(self._singletons[action - self._singleton_base])
        offset, within = divmod(action, self._per_offset)
        position = (head + offset) % max(length, 1)
        if within < SERIAL_INSERT:
            return self.spec.encode(self._positional_kinds[within], position=position)
        insert_end = SERIAL_INSERT + len(self._inserts)
        if within < insert_end:
            generator, sign = self._inserts[within - SERIAL_INSERT]
            return self.spec.encode(
                self._insert_kind, position=position, generator=generator, sign=sign
            )
        return self.spec.encode(self._crossing, position=position)

    def describe(self, action: int) -> str:
        if action >= self._shift_left:
            return "SHIFT_LEFT" if action == self._shift_left else "SHIFT_RIGHT"
        if action >= self._singleton_base:
            return ["DESTABILIZE", "STABILIZE_POS", "STABILIZE_NEG", "PASS"][
                action - self._singleton_base
            ]
        offset, within = divmod(action, self._per_offset)
        names = ["REDUCE", "COMMUTE", "BRAID"]
        for generator in range(1, self.config.max_strands):
            for sign in ("+", "-"):
                names.append(f"INSERT(s{generator}{sign})")
        names.append("CROSSING_CHANGE")
        return f"{names[within]}@+{offset}"

    # -- environment ----------------------------------------------------------

    def reset(self, seed: int) -> Transition:
        if self.generator is None:
            state = self._init(jax.random.PRNGKey(seed))
            return self._view(state, 0, reward=0.0)
        return self._view(self._generated(seed), 0, reward=0.0)

    def _generated(self, seed: int):
        """An instance from the graded generator, with log(A/B) sampled."""
        rng = np.random.default_rng(seed)
        if self.config.stage_source:
            source = next(
                s for s in self.generator.sources if s.name == self.config.stage_source
            )
            moves = self.config.stage_scramble
        else:
            levels = self.generator.levels(self.config.generator_max_scramble)
            source, moves = levels[int(rng.integers(len(levels)))]
        instance = self.generator.generate(source, moves, rng)
        low, high = self.config.log_ratio_range
        log_ratio = float(rng.uniform(low, high)) if high > low else low
        return self.env.init_from_word(
            list(instance.word), instance.strands, log_ratio=log_ratio
        )

    def from_word(
        self, word: list[int], strands: int, log_ratio: float = 0.0
    ) -> Transition:
        return self._view(
            self.env.init_from_word(word, strands, log_ratio=log_ratio), 0, reward=0.0
        )

    def step(self, state: Any, action: int) -> Transition:
        pgx_state, head = state
        mask = self._legal(pgx_state, head)
        if not mask[action]:
            raise ValueError(f"Illegal serial action {self.describe(action)}")

        if action >= self._shift_left:
            offset = -self.stride if action == self._shift_left else self.stride
            length = max(int(np.asarray(pgx_state._word).astype(bool).sum()), 1)
            new_head = (head + offset) % length
            advanced = self._charge_budget(pgx_state)
            return self._view(advanced, new_head, reward=0.0)

        actor = int(np.asarray(pgx_state.current_player))
        length_before = int(np.asarray(pgx_state._word).astype(bool).sum())
        next_state = self._step(
            pgx_state, np.int32(self.underlying_action(action, head, length_before))
        )
        rewards = np.asarray(next_state.rewards, dtype=np.float32)
        length = max(int(np.asarray(next_state._word).astype(bool).sum()), 1)
        return self._view(next_state, head % length, reward=float(rewards[actor]))

    def _charge_budget(self, pgx_state: Any):
        """Spend one ply without touching the word.

        A shift has to cost something or the game never ends, and it has to cost
        the *same* thing an edit costs, or the measurement is rigged in the
        serial agent's favour.
        """
        import jax.numpy as jnp

        budget = pgx_state._budget - 1
        switching = (pgx_state._phase == 0) & (budget <= 0)
        phase = jnp.where(switching, 1, pgx_state._phase).astype(jnp.int32)
        budget = jnp.where(
            switching, self.config.simplify_budget, budget
        ).astype(jnp.int32)
        exhausted = (phase == 1) & (budget <= 0)
        simplifier = 1 - pgx_state._scrambler
        rewards = jnp.zeros(2, dtype=jnp.float32)
        rewards = rewards.at[simplifier].set(-1.0).at[pgx_state._scrambler].set(1.0)
        rewards = jnp.where(exhausted, rewards, jnp.zeros(2, dtype=jnp.float32))
        current = jnp.where(phase == 0, pgx_state._scrambler, simplifier).astype(jnp.int32)
        return pgx_state.replace(
            current_player=current,
            rewards=rewards,
            terminated=exhausted,
            _phase=phase,
            _budget=budget,
            _step_count=pgx_state._step_count + 1,
        )

    def final_rewards(self, state: Any) -> np.ndarray:
        pgx_state, _ = state
        return np.asarray(pgx_state.rewards, dtype=np.float32)

    def state_info(self, state: Any) -> dict[str, int]:
        pgx_state, _ = state
        return {
            "player": int(np.asarray(pgx_state.current_player)),
            "move_count": int(np.asarray(pgx_state._step_count)),
            "consecutive_passes": 0,
        }

    def first_role_player(self, state: Any) -> int:
        pgx_state, _ = state
        return int(np.asarray(pgx_state._scrambler))

    def unwrap(self, state: Any) -> Any:
        """Drop the head; callers that inspect the word want the Pgx state."""
        pgx_state, _ = state
        return pgx_state

    # -- observation and legality --------------------------------------------

    def _legal(self, pgx_state: Any, head: int) -> np.ndarray:
        full = np.asarray(pgx_state.legal_action_mask, dtype=bool)
        mask = np.zeros(self.num_actions, dtype=bool)
        length = int(np.asarray(pgx_state._word).astype(bool).sum())
        for action in range(self.num_actions - 2):
            mask[action] = full[self.underlying_action(action, head, length)]
        # Shifting is pointless on a word too short to move within, and it must
        # never be the only option -- PASS remains the guaranteed fallback.
        if length > 1 and not bool(np.asarray(pgx_state.terminated)):
            mask[self._shift_left] = True
            mask[self._shift_right] = True
        if not mask.any():
            mask[self._singleton_base + 3] = True
        return mask

    def _view(self, pgx_state: Any, head: int, reward: float) -> Transition:
        observation = np.asarray(pgx_state.observation, dtype=np.float32)  # (L, C)
        word = np.asarray(pgx_state._word)
        length = int((word != 0).sum())
        # The word is cyclic: gather the window with wraparound. When the word is
        # shorter than the window the same positions repeat, which is what the
        # necklace actually looks like.
        if length > 0:
            indexes = (head + np.arange(self.window)) % length
        else:
            indexes = np.zeros(self.window, dtype=int)
        window = observation[indexes]
        return Transition(
            state=(pgx_state, head),
            observation=window.reshape(1, self.window, observation.shape[1]),
            legal_actions=self._legal(pgx_state, head),
            reward=reward,
            terminated=bool(np.asarray(pgx_state.terminated)),
            player=int(np.asarray(pgx_state.current_player)),
            move_count=int(np.asarray(pgx_state._step_count)),
            consecutive_passes=0,
        )
