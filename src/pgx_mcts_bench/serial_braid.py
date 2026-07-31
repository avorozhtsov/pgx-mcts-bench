"""Serial (moving-window) view of the braid environment.

The parallel formulation scores every position at once: the policy head emits a
logit per (position, move-type), so the action space is `O(L)` and a decision
costs one forward pass over the whole word. That is the right trade while a
full-length pass is affordable.

This is the other formulation. A **head** points at one position; the agent sees
only a window of width `w` *centred* on it and may either act inside that window
or shift the head. The window is centred rather than forward-looking so that the
two shift directions are equally informed -- with `[head, head+w)` a left shift
walks into tape the agent has never seen, and the first ladder run showed the
head-only agent taking 118 blind left shifts against 2 right ones in the
episodes it failed. The consequences:

* the action space is `W * (2N + 2) + 4 + 2S` -- **independent of L**. At N=5,
  W=1 and five strides that is 22 against 388, and it does not grow when the
  word does;
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

from typing import Any, NamedTuple

import jax
import numpy as np

from pgx_mcts_bench.config import BraidGameConfig, pick_stage
from pgx_mcts_bench.game import Transition


class SerialState(NamedTuple):
    """What the serial agent carries between plies.

    A tuple so `search.py` keeps treating it opaquely and every existing
    `state[0]` / `state[1]` access still works, named so that adding a field does
    not silently repoint the ones already there.

    `colours[h]` is the colour of the strand *currently at height h under the
    head*, not of a fixed thread: the head moves through crossings, and a crossing
    swaps two strands, so the colours swap with them. That is what makes a colour
    follow its strand without depending on the seam.
    """

    pgx: object
    head: int
    registers: np.ndarray
    colours: np.ndarray
    colour: int = 0

# Serial action layout, with G = max_strands - 1 generators and W = act_width.
# One block of `per_offset = 3 + 2G + 1` actions per actionable offset, then the
# position-free singletons, then two shift actions per stride:
#   offset j, j = 0 .. W-1, acting at head + j - W//2
#     +0             REDUCE
#     +1             COMMUTE
#     +2             BRAID
#     +3 .. +3+2G-1  INSERT (generator, sign)
#     +3+2G          CROSSING_CHANGE
#   W*per_offset + 0 DESTABILIZE          (position-free)
#                +1  STABILIZE_POS
#                +2  STABILIZE_NEG
#                +3  PASS
#                +4  SHIFT by -strides[0]
#                +5  SHIFT by +strides[0]
#                ... one pair per stride
SERIAL_REDUCE = 0
SERIAL_COMMUTE = 1
SERIAL_BRAID = 2
SERIAL_INSERT = 3

# Strides available to the head, in letters. A **fixed** set of powers of two:
# fixed so the action count stays exactly independent of L (deriving it from
# `max_len` would make the space grow like log L, giving up the property the
# formulation exists for), powers of two so the head reaches a site by binary
# decomposition instead of stepping. This matters more than it looks -- each ply
# of repositioning is a ply of MCTS search depth, and depth is exponentially
# expensive, so a head that can only step puts distant sites past the horizon
# entirely. Measured on the first ladder: in the episodes `serial-w7-head`
# failed, it spent 118 plies stepping one way round the necklace and 2 the
# other, and arrived nowhere. At max_len = 48 this set repositions anywhere in
# at most two plies.
DEFAULT_STRIDES: tuple[int, ...] = (1, 2, 4, 8, 16)


def shift_strides(
    window: int, max_len: int, strides: tuple[int, ...] = ()
) -> tuple[int, ...]:
    """Head strides. `strides` overrides; `(w // 2,)` reproduces the old tape."""
    del window, max_len  # deliberately not a function of the word capacity
    return tuple(strides) if strides else DEFAULT_STRIDES


def serial_action_size(
    max_strands: int,
    act_width: int = 1,
    n_strides: int = 1,
    registers: int = 0,
    colours: int = 0,
) -> int:
    """Actions for a window agent.

    `act_width` is how many window offsets the agent may act at. 1 means "only at
    the head" -- the position lives entirely in the state and reaching a
    neighbour costs a shift. Larger values let one ply act anywhere in the
    visible window, which costs a wider head but no plies.

    Either way the count is independent of L, which is the whole point: the
    stride set is fixed by `max_len` at *construction*, not by the current word.
    """
    positional = 3 + 2 * (max_strands - 1) + 1  # reduce/commute/braid, inserts, crossing
    # Colours cost three actions however many colours there are: PAINT_LOW,
    # PAINT_HIGH, CYCLE. Paint-per-(strand, colour) would be 20 dead actions at
    # N=5, C=4, and dead actions are what sank the register arm.
    return act_width * positional + 4 + 2 * n_strides + registers + (3 if colours else 0)


def serial_action_names(max_strands: int, strides: tuple[int, ...] = (1,)) -> list[str]:
    names = ["REDUCE", "COMMUTE", "BRAID"]
    for generator in range(1, max_strands):
        for sign in ("+", "-"):
            names.append(f"INSERT(s{generator}{sign})")
    names += [
        "CROSSING_CHANGE",
        "DESTABILIZE",
        "STABILIZE_POS",
        "STABILIZE_NEG",
        "PASS",
    ]
    for stride in strides:
        names += [f"SHIFT_LEFT({stride})", f"SHIFT_RIGHT({stride})"]
    return names


class SerialBraidGame:
    """Adapter presenting the braid environment through a moving window.

    State is `(pgx_state, head, registers)`. `GameAdapter` treats state opaquely,
    so search and training need no changes -- this drops into the existing sweep as
    another variant, scored on the same anchors. The registers ride in the state
    and are broadcast into the observation, which is why nothing in `search.py`,
    `data.py` or `training.py` has to know they exist: the register value is part
    of the observation that gets stored in the replay buffer.
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
        self.strides = shift_strides(
            config.serial_window, config.max_len, config.serial_shift_strides
        )
        self.env = BraidUnknot(config.to_braid_config())
        self.spec = self.env.spec
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
                random_crossings=config.generator_random_crossings,
                random_per_grade=config.generator_random_per_grade,
                random_seed=config.generator_random_seed,
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
        self.num_actions = serial_action_size(
            config.max_strands,
            self.act_width,
            len(self.strides),
            max(config.serial_registers, 0),
            max(config.serial_colours, 0),
        )
        self._per_offset = 3 + 2 * generators + 1
        self._singleton_base = self.act_width * self._per_offset
        self._shift_base = self._singleton_base + 4
        self.registers = max(config.serial_registers, 0)
        self._register_base = self._shift_base + 2 * len(self.strides)
        self.colours = max(config.serial_colours, 0)
        self._colour_base = self._register_base + self.registers
        self._paint_low = self._colour_base
        self._paint_high = self._colour_base + 1
        self._cycle = self._colour_base + 2
        # Offsets are centred on the head, matching the centred window: action
        # block j acts at head + j - act_width//2. At act_width = 1 that is the
        # head itself.
        self._act_origin = self.act_width // 2
        self._window_origin = self.window // 2

    # -- action translation ---------------------------------------------------

    def underlying_action(self, action: int, head: int, length: int) -> int | None:
        """Map a serial action to an action of the wrapped environment.

        `None` means a head shift, which the wrapped environment knows nothing
        about. Offsets wrap: the word is a necklace.
        """
        if action >= self._shift_base:
            return None
        if action >= self._singleton_base:
            return self.spec.encode(self._singletons[action - self._singleton_base])
        offset, within = divmod(action, self._per_offset)
        position = (head + offset - self._act_origin) % max(length, 1)
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
        if self.colours and action >= self._colour_base:
            return ["PAINT_LOW", "PAINT_HIGH", "CYCLE"][action - self._colour_base]
        if action >= self._register_base:
            return f"TOGGLE(r{action - self._register_base})"
        if action >= self._shift_base:
            index, direction = divmod(action - self._shift_base, 2)
            side = "RIGHT" if direction else "LEFT"
            return f"SHIFT_{side}({self.strides[index]})"
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
        return f"{names[within]}@{offset - self._act_origin:+d}"

    def shift_of(self, action: int) -> int | None:
        """Signed head displacement of a shift action, or `None` if it is not one."""
        if not self._shift_base <= action < self._register_base:
            return None
        index, direction = divmod(action - self._shift_base, 2)
        return self.strides[index] if direction else -self.strides[index]

    def register_of(self, action: int) -> int | None:
        """Index of the register a TOGGLE action flips, or `None`."""
        if not self._register_base <= action < self._colour_base:
            return None
        return action - self._register_base

    def colour_action(self, action: int) -> str | None:
        """`"low"`, `"high"`, `"cycle"`, or `None` if this is not a colour action."""
        if not self.colours or action < self._colour_base:
            return None
        return ("low", "high", "cycle")[action - self._colour_base]

    def _crossing_heights(self, pgx_state: object, head: int) -> tuple[int, int]:
        """The two heights the letter under the head swaps, as 0-based indices."""
        word = np.asarray(pgx_state._word)
        length = max(int((word != 0).sum()), 1)
        generator = abs(int(word[head % length]))
        low = max(generator - 1, 0)
        return low, min(low + 1, self.config.max_strands - 1)

    def _transport(
        self, pgx_state: object, head: int, displacement: int, colours: np.ndarray
    ) -> np.ndarray:
        """Carry the colours through every crossing the head moves across.

        A crossing swaps the two strands it joins, so a colour attached to a height
        has to swap with them or it stops describing the strand it was painted on.
        Moving right across position p applies p's transposition; moving left off
        position p-1 undoes it, and a transposition is its own inverse, so both
        directions are the same swap.
        """
        if not self.colours or displacement == 0:
            return colours
        word = np.asarray(pgx_state._word)
        length = max(int((word != 0).sum()), 1)
        moved = colours.copy()
        if displacement > 0:
            positions = [(head + i) % length for i in range(displacement)]
        else:
            positions = [(head - i) % length for i in range(1, -displacement + 1)]
        for position in positions:
            generator = abs(int(word[position]))
            if 1 <= generator < self.config.max_strands:
                low = generator - 1
                moved[low], moved[low + 1] = moved[low + 1], moved[low]
        return moved

    # -- environment ----------------------------------------------------------

    def reset(self, seed: int) -> Transition:
        if self.generator is None:
            state = self._init(jax.random.PRNGKey(seed))
            return self._view(state, 0, self._no_registers(), self._no_colours(), 0, reward=0.0)
        return self._view(
            self._generated(seed), 0, self._no_registers(), self._no_colours(), 0,
            reward=0.0,
        )

    def _generated(self, seed: int):
        """An instance from the graded generator, with log(A/B) sampled."""
        rng = np.random.default_rng(seed)
        source, moves = pick_stage(self.config, self.generator, rng)
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
            self.env.init_from_word(word, strands, log_ratio=log_ratio),
            0,
            self._no_registers(),
            self._no_colours(),
            0,
            reward=0.0,
        )

    def step(self, state: Any, action: int) -> Transition:
        pgx_state, head, registers, colours, colour = state
        mask = self._legal(pgx_state, head)
        if not mask[action]:
            raise ValueError(f"Illegal serial action {self.describe(action)}")

        # A toggle costs a ply, like every other action. Free writes would let the
        # agent set up an arbitrary control state between two edits at no cost,
        # which is not a machine -- it is an oracle.
        slot = self.register_of(action)
        if slot is not None:
            flipped = registers.copy()
            flipped[slot] = 1.0 - flipped[slot]
            return self._view(
                self._charge_budget(pgx_state), head, flipped, colours, colour,
                reward=0.0,
            )

        # Painting costs a ply like everything else, and so does cycling: a free
        # palette would let the agent set an arbitrary control state between two
        # edits, which is an oracle rather than a machine.
        paint = self.colour_action(action)
        if paint is not None:
            painted, held = colours.copy(), colour
            if paint == "cycle":
                held = (colour + 1) % self.colours
            else:
                low, high = self._crossing_heights(pgx_state, head)
                painted[low if paint == "low" else high] = colour + 1
            return self._view(
                self._charge_budget(pgx_state), head, registers, painted, held,
                reward=0.0,
            )

        displacement = self.shift_of(action)
        if displacement is not None:
            length = max(int(np.asarray(pgx_state._word).astype(bool).sum()), 1)
            new_head = (head + displacement) % length
            carried = self._transport(pgx_state, head, displacement, colours)
            advanced = self._charge_budget(pgx_state)
            return self._view(advanced, new_head, registers, carried, colour, reward=0.0)

        actor = int(np.asarray(pgx_state.current_player))
        length_before = int(np.asarray(pgx_state._word).astype(bool).sum())
        next_state = self._step(
            pgx_state, np.int32(self.underlying_action(action, head, length_before))
        )
        rewards = np.asarray(next_state.rewards, dtype=np.float32)
        length = max(int(np.asarray(next_state._word).astype(bool).sum()), 1)
        return self._view(
            next_state, head % length, registers, colours, colour,
            reward=float(rewards[actor]),
        )

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
        pgx_state = state[0]
        return np.asarray(pgx_state.rewards, dtype=np.float32)

    def state_info(self, state: Any) -> dict[str, int]:
        pgx_state = state[0]
        return {
            "player": int(np.asarray(pgx_state.current_player)),
            "move_count": int(np.asarray(pgx_state._step_count)),
            "consecutive_passes": 0,
        }

    def first_role_player(self, state: Any) -> int:
        pgx_state = state[0]
        return int(np.asarray(pgx_state._scrambler))

    def unwrap(self, state: Any) -> Any:
        """Drop the head and registers; callers inspecting the word want Pgx."""
        pgx_state = state[0]
        return pgx_state

    # -- observation and legality --------------------------------------------

    def _legal(self, pgx_state: Any, head: int) -> np.ndarray:
        full = np.asarray(pgx_state.legal_action_mask, dtype=bool)
        mask = np.zeros(self.num_actions, dtype=bool)
        length = int(np.asarray(pgx_state._word).astype(bool).sum())
        for action in range(self._shift_base):
            mask[action] = full[self.underlying_action(action, head, length)]
        # Toggles are always available while the episode runs: the control state is
        # the agent's own, not a function of the word.
        if not bool(np.asarray(pgx_state.terminated)):
            mask[self._register_base : self._colour_base] = True
            if self.colours:
                mask[self._colour_base :] = True
        # Shifting is pointless on a word too short to move within, and it must
        # never be the only option -- PASS remains the guaranteed fallback. A
        # stride that is a multiple of the current length is a no-op on the
        # necklace, so it is masked out rather than offered as a wasted ply.
        if length > 1 and not bool(np.asarray(pgx_state.terminated)):
            for index, stride in enumerate(self.strides):
                if stride % length:
                    mask[self._shift_base + 2 * index] = True
                    mask[self._shift_base + 2 * index + 1] = True
        if not mask.any():
            mask[self._singleton_base + 3] = True
        return mask

    def _no_registers(self) -> np.ndarray:
        return np.zeros(self.registers, dtype=np.float32)

    def _no_colours(self) -> np.ndarray:
        # 0 means unpainted; a painted strand carries 1..colours.
        return np.zeros(self.config.max_strands, dtype=np.int64)

    def _colour_planes(self, colours: np.ndarray, colour: int) -> np.ndarray:
        """One-hot the colour at each height, then the colour being held.

        One-hot rather than a scalar per height: colour ids are labels, and a
        scalar would tell the network colour 1 is nearer 2 than 3.
        """
        heights = np.zeros((self.config.max_strands, self.colours), dtype=np.float32)
        for height, value in enumerate(colours[: self.config.max_strands]):
            if 1 <= int(value) <= self.colours:
                heights[height, int(value) - 1] = 1.0
        held = np.zeros(self.colours, dtype=np.float32)
        held[colour % self.colours] = 1.0
        return np.concatenate([heights.reshape(-1), held])

    def _view(
        self,
        pgx_state: Any,
        head: int,
        registers: np.ndarray,
        colours: np.ndarray,
        colour: int,
        reward: float,
    ) -> Transition:
        observation = np.asarray(pgx_state.observation, dtype=np.float32)  # (L, C)
        word = np.asarray(pgx_state._word)
        length = int((word != 0).sum())
        # The word is cyclic: gather the window with wraparound. When the word is
        # shorter than the window the same positions repeat, which is what the
        # necklace actually looks like.
        if length > 0 and self.config.serial_encoder:
            # A head-relative complete scan: occupied letters first, then the
            # environment's padding slots. The action semantics remain local and
            # O(1); only the candidate encoder changes.
            occupied = (head + np.arange(length)) % length
            indexes = np.concatenate(
                [occupied, np.arange(length, self.config.max_len)]
            )
        elif length > 0:
            indexes = (head + np.arange(self.window) - self._window_origin) % length
        else:
            width = self.config.max_len if self.config.serial_encoder else self.window
            indexes = np.zeros(width, dtype=int)
        window = observation[indexes]
        observed_width = window.shape[0]
        if self.registers:
            # Broadcast along the window, like the environment's own scalar planes.
            # Carrying the control state in the observation rather than in a
            # separate channel to the network is what keeps search, the replay
            # buffer and the training step untouched.
            planes = np.broadcast_to(
                registers[None, :], (observed_width, self.registers)
            )
            window = np.concatenate([window, planes], axis=1)
        if self.colours:
            palette = self._colour_planes(colours, colour)
            planes = np.broadcast_to(
                palette[None, :], (observed_width, palette.shape[0])
            )
            window = np.concatenate([window, planes], axis=1)
        return Transition(
            state=SerialState(pgx_state, head, registers, colours, colour),
            observation=window.reshape(1, observed_width, window.shape[1]),
            legal_actions=self._legal(pgx_state, head),
            reward=reward,
            terminated=bool(np.asarray(pgx_state.terminated)),
            player=int(np.asarray(pgx_state.current_player)),
            move_count=int(np.asarray(pgx_state._step_count)),
            consecutive_passes=0,
        )
