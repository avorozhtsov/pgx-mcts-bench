"""Bounded moving-window view of the braid environment.

A **head** points at one position; the agent sees
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
* but reaching a distant site costs *native plies*, and in MCTS a ply is search
  depth. That controller/search cost is reported separately from semantic `L`.

Shifts consume the native episode budget like any other ply. The alternative --
free shifts -- makes the game non-terminating. They do not consume the semantic
solution objective. Serial variants compensate with a larger `simplify_budget`.

The word is cyclic, so the head wraps and a window may show the same letter more
than once when the word is shorter than `w`. That is correct rather than a
degenerate case: the necklace really does repeat.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import numpy as np

from pgx_mcts_bench.config import BraidGameConfig, pick_stage
from pgx_mcts_bench.game import Transition, sample_log_ratio


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
    colour: int
    tape: np.ndarray
    internal_steps: int
    # Portable braid-state changes since the start of the simplifier episode.
    # Appended with a default so old seven-field pickles remain loadable.
    semantic_moves: int = 0


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


def shift_strides(window: int, max_len: int, strides: tuple[int, ...] = ()) -> tuple[int, ...]:
    """Head strides. `strides` overrides; `(w // 2,)` reproduces the old tape."""
    del window, max_len  # deliberately not a function of the word capacity
    return tuple(strides) if strides else DEFAULT_STRIDES


def serial_action_size(
    max_strands: int,
    act_width: int = 1,
    n_strides: int = 1,
    registers: int = 0,
    colours: int = 0,
    tape_symbols: int = 0,
    tape_preserve_shift: bool = False,
    cyclic_band_generators: bool = False,
) -> int:
    """Actions for a window agent.

    `act_width` is how many window offsets the agent may act at. 1 means "only at
    the head" -- the position lives entirely in the state and reaching a
    neighbour costs a shift. Larger values let one ply act anywhere in the
    visible window, which costs a wider head but no plies.

    Either way the count is independent of L, which is the whole point: the
    stride set is fixed by `max_len` at *construction*, not by the current word.
    """
    generators = max_strands - 1 + int(cyclic_band_generators)
    positional = 3 + 2 * generators + 1  # reduce/commute/braid, inserts, crossing
    # Colours cost three actions however many colours there are: PAINT_LOW,
    # PAINT_HIGH, CYCLE. Paint-per-(strand, colour) would be 20 dead actions at
    # N=5, C=4, and dead actions are what sank the register arm.
    tape_variants = max(tape_symbols + int(tape_preserve_shift), 1)
    return (
        act_width * positional
        + 4
        + 2 * n_strides * tape_variants
        + registers
        + (3 if colours else 0)
    )


def serial_action_names(
    max_strands: int,
    strides: tuple[int, ...] = (1,),
    cyclic_band_generators: bool = False,
) -> list[str]:
    names = ["REDUCE", "COMMUTE", "BRAID"]
    capacity = max_strands - 1 + int(cyclic_band_generators)
    for generator in range(1, capacity + 1):
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
        self._step = jax.jit(self.env.step)
        self._charge_budget_step = jax.jit(self._charge_budget_impl)
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

        generators = config.generator_capacity
        self._positional_kinds = [REDUCE, COMMUTE, BRAID]
        self._inserts = [
            (generator, sign) for generator in range(1, generators + 1) for sign in (1, -1)
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
            max(config.serial_tape_symbols, 0),
            config.serial_tape_preserve_shift,
            config.cyclic_band_generators,
        )
        self._per_offset = 3 + 2 * generators + 1
        self._singleton_base = self.act_width * self._per_offset
        self._shift_base = self._singleton_base + 4
        self.tape_symbols = max(config.serial_tape_symbols, 0)
        self.tape_preserve_shift = bool(config.serial_tape_preserve_shift)
        self._tape_variants = max(self.tape_symbols + int(self.tape_preserve_shift), 1)
        self.registers = max(config.serial_registers, 0)
        self._register_base = self._shift_base + 2 * len(self.strides) * self._tape_variants
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
        # Legal masks are requested for every expanded MCTS leaf.  The mapping
        # from local serial actions to the wrapped braid action space depends
        # only on (head, current word length), so cache the integer gather map
        # instead of decoding every action in Python at every leaf.
        self._underlying_action_cache: dict[tuple[int, int], np.ndarray] = {}
        self.certified_value_stats = {
            "evaluations": 0,
            "informative": 0,
            "binding": 0,
            "by_method": {},
        }

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
            shift, variant = divmod(action - self._shift_base, self._tape_variants)
            index, direction = divmod(shift, 2)
            side = "RIGHT" if direction else "LEFT"
            if self.tape_preserve_shift and variant == 0:
                write = ",PRESERVE"
            elif self.tape_symbols:
                symbol = variant - int(self.tape_preserve_shift)
                write = f",WRITE({symbol})"
            else:
                write = ""
            return f"SHIFT_{side}({self.strides[index]}{write})"
        if action >= self._singleton_base:
            return ["DESTABILIZE", "STABILIZE_POS", "STABILIZE_NEG", "PASS"][
                action - self._singleton_base
            ]
        offset, within = divmod(action, self._per_offset)
        names = ["REDUCE", "COMMUTE", "BRAID"]
        for generator in range(1, self.config.generator_capacity + 1):
            for sign in ("+", "-"):
                names.append(f"INSERT(s{generator}{sign})")
        names.append("CROSSING_CHANGE")
        return f"{names[within]}@{offset - self._act_origin:+d}"

    def shift_of(self, action: int) -> int | None:
        """Signed head displacement of a shift action, or `None` if it is not one."""
        if not self._shift_base <= action < self._register_base:
            return None
        shift, _ = divmod(action - self._shift_base, self._tape_variants)
        index, direction = divmod(shift, 2)
        return self.strides[index] if direction else -self.strides[index]

    def tape_write_of(self, action: int) -> int | None:
        """Symbol written at the old head by a shift, or ``None`` without tape."""
        if not self.tape_symbols or not self._shift_base <= action < self._register_base:
            return None
        variant = (action - self._shift_base) % self._tape_variants
        if self.tape_preserve_shift and variant == 0:
            return None
        return variant - int(self.tape_preserve_shift)

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
        strands = int(np.asarray(pgx_state._n))
        if self.config.cyclic_band_generators and generator == strands:
            return strands - 1, 0
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
            strands = int(np.asarray(pgx_state._n))
            if self.config.cyclic_band_generators and generator == strands:
                moved[0], moved[strands - 1] = moved[strands - 1], moved[0]
            elif 1 <= generator < self.config.max_strands:
                low = generator - 1
                moved[low], moved[low + 1] = moved[low + 1], moved[low]
        return moved

    # -- environment ----------------------------------------------------------

    def reset(self, seed: int) -> Transition:
        if self.generator is None:
            raise RuntimeError(
                "SerialBraidGame.reset requires a configured graded generator; "
                "use from_word() for an externally supplied representation"
            )
        return self._view(
            self._generated(seed),
            0,
            self._no_registers(),
            self._no_colours(),
            0,
            self._no_tape(),
            reward=0.0,
        )

    def _generated(self, seed: int):
        """An instance from the graded generator, with log(A/B) sampled."""
        rng = np.random.default_rng(seed)
        source, moves = pick_stage(self.config, self.generator, rng)
        instance = self.generator.generate(source, moves, rng)
        log_ratio = sample_log_ratio(self.config, rng)
        word, strands = self._initial_representation(list(instance.word), instance.strands)
        return self.env.init_from_word(word, strands, log_ratio=log_ratio)

    def _initial_representation(self, word: list[int], strands: int) -> tuple[list[int], int]:
        """Apply configured Markov stabilizations without charging solve cost."""
        count = self.config.serial_initial_markov_stabilizations
        sign = self.config.serial_initial_markov_sign
        if count < 0:
            raise ValueError("initial Markov stabilization count must be non-negative")
        if sign not in {-1, 1}:
            raise ValueError("initial Markov stabilization sign must be -1 or 1")
        expanded = list(word)
        height = int(strands)
        if height + count > self.config.max_strands:
            raise ValueError("initial Markov stabilizations exceed strand capacity")
        if len(expanded) + count > self.config.max_len:
            raise ValueError("initial Markov stabilizations exceed word capacity")
        for _ in range(count):
            expanded.append(sign * height)
            height += 1
        return expanded, height

    def from_word(self, word: list[int], strands: int, log_ratio: float = 0.0) -> Transition:
        word, strands = self._initial_representation(word, strands)
        return self._view(
            self.env.init_from_word(word, strands, log_ratio=log_ratio),
            0,
            self._no_registers(),
            self._no_colours(),
            0,
            self._no_tape(),
            reward=0.0,
        )

    def assessment_scan(
        self,
        word: list[int],
        strands: int,
        log_ratio: float = 0.0,
        *,
        steps: int = 5,
    ) -> tuple[list[np.ndarray], float]:
        """Deterministic read-only head sweep used only by task assessment.

        It performs no semantic edit and consumes no solver episode budget. Each
        view starts with a full internal budget, so the scheduling assessor is
        invariant to an arbitrary controller route. Local windows are spaced
        evenly around the cyclic word; full-scan encoders receive the same
        rotations through this common interface.
        """
        if steps < 1:
            raise ValueError("assessment scan steps must be positive")
        word, strands = self._initial_representation(word, strands)
        pgx_state = self.env.init_from_word(word, strands, log_ratio=log_ratio)
        length = max(int(np.count_nonzero(np.asarray(pgx_state._word))), 1)
        heads = (np.arange(steps, dtype=np.int64) * length // steps).tolist()
        views = [
            self._view(
                pgx_state,
                int(head),
                self._no_registers(),
                self._no_colours(),
                0,
                self._no_tape(),
                reward=0.0,
            ).observation
            for head in heads
        ]
        if (
            self.config.serial_encoder
            or self.config.serial_ensemble
            or self.config.serial_raster == "scalable"
        ):
            coverage = 1.0
        else:
            radius = self.window // 2
            covered = {
                (int(head) + offset) % length
                for head in heads
                for offset in range(-radius, radius + 1)
            }
            coverage = min(len(covered) / length, 1.0)
        return views, coverage

    def step(self, state: Any, action: int) -> Transition:
        pgx_state = state.pgx
        head = state.head
        registers = state.registers
        colours = state.colours
        colour = state.colour
        tape = state.tape
        internal_steps = state.internal_steps
        semantic_moves = int(getattr(state, "semantic_moves", 0))
        mask = self._legal(pgx_state, head, internal_steps)
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
                self._charge_budget(pgx_state),
                head,
                flipped,
                colours,
                colour,
                tape,
                reward=0.0,
                internal_steps=internal_steps + 1,
                semantic_moves=semantic_moves,
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
                self._charge_budget(pgx_state),
                head,
                registers,
                painted,
                held,
                tape,
                reward=0.0,
                internal_steps=internal_steps + 1,
                semantic_moves=semantic_moves,
            )

        displacement = self.shift_of(action)
        if displacement is not None:
            length = max(int(np.asarray(pgx_state._word).astype(bool).sum()), 1)
            new_head = (head + displacement) % length
            carried = self._transport(pgx_state, head, displacement, colours)
            written = tape.copy()
            symbol = self.tape_write_of(action)
            if symbol is not None and length > 0:
                written[head % length] = symbol
            advanced = self._charge_budget(pgx_state)
            return self._view(
                advanced,
                new_head,
                registers,
                carried,
                colour,
                written,
                reward=0.0,
                internal_steps=internal_steps + 1,
                semantic_moves=semantic_moves,
            )

        actor = int(np.asarray(pgx_state.current_player))
        length_before = int(np.asarray(pgx_state._word).astype(bool).sum())
        next_state = self._step(
            pgx_state, np.int32(self.underlying_action(action, head, length_before))
        )
        moved_tape = self._rewrite_tape(pgx_state, action, head, tape)
        rewards = np.asarray(next_state.rewards, dtype=np.float32)
        length = max(int(np.asarray(next_state._word).astype(bool).sum()), 1)
        before_braid = (
            tuple(int(value) for value in np.asarray(pgx_state._word) if int(value)),
            int(np.asarray(pgx_state._n)),
        )
        after_braid = (
            tuple(int(value) for value in np.asarray(next_state._word) if int(value)),
            int(np.asarray(next_state._n)),
        )
        return self._view(
            next_state,
            head % length,
            registers,
            colours,
            colour,
            moved_tape,
            reward=float(rewards[actor]),
            internal_steps=0,
            semantic_moves=semantic_moves + int(after_braid != before_braid),
        )

    def _rewrite_tape(self, pgx_state: Any, action: int, head: int, tape: np.ndarray) -> np.ndarray:
        """Apply the braid rewrite's positional transport to the annotation tape."""
        if not self.tape_symbols:
            return tape
        from rf_knots.actions import (
            BRAID,
            COMMUTE,
            DESTABILIZE,
            INSERT,
            REDUCE,
            STABILIZE_NEG,
            STABILIZE_POS,
        )

        underlying = self.underlying_action(
            action, head, int(np.asarray(pgx_state._word).astype(bool).sum())
        )
        assert underlying is not None
        kind, position, _, _ = self.spec.decode(underlying)
        word = np.asarray(pgx_state._word)
        length = int((word != 0).sum())
        out = tape.copy()

        def delete(positions: set[int]) -> np.ndarray:
            kept = [out[i] for i in range(length) if i not in positions]
            return np.asarray(kept + [0] * (self.config.max_len - len(kept)), dtype=np.int64)

        if kind == REDUCE:
            return delete({position % length, (position + 1) % length})
        if kind == COMMUTE:
            left, right = position % length, (position + 1) % length
            out[left], out[right] = out[right], out[left]
        elif kind == BRAID:
            indexes = [(position + offset) % length for offset in range(3)]
            values = [out[index] for index in indexes][::-1]
            for index, value in zip(indexes, values, strict=True):
                out[index] = value
        elif kind == INSERT:
            values = list(out[:length])
            values[position:position] = [0, 0]
            out = np.asarray(values + [0] * (self.config.max_len - len(values)), dtype=np.int64)
        elif kind == DESTABILIZE:
            top = int(np.asarray(pgx_state._n)) - 1
            position = int(np.flatnonzero(np.abs(word[:length]) == top)[0])
            return delete({position})
        elif kind in (STABILIZE_POS, STABILIZE_NEG):
            out[length] = 0
        return out

    def _charge_budget(self, pgx_state: Any):
        """Spend one ply without touching the word.

        A shift has to consume the native episode clock or the game never ends.
        This clock is deliberately separate from the semantic solution cost.
        """
        return self._charge_budget_step(pgx_state)

    def _charge_budget_impl(self, pgx_state: Any):
        import jax.numpy as jnp

        budget = pgx_state._budget - 1
        switching = (pgx_state._phase == 0) & (budget <= 0)
        phase = jnp.where(switching, 1, pgx_state._phase).astype(jnp.int32)
        budget = jnp.where(switching, self.config.simplify_budget, budget).astype(jnp.int32)
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
        return self._semantic_final_rewards(state.pgx, self.semantic_move_count(state))

    def _semantic_final_rewards(self, pgx_state: Any, semantic_moves: int) -> np.ndarray:
        """Recompute serial terminal payoff from the portable scientific cost.

        The underlying environment's move clock includes head shifts and memory
        writes. Those plies bound computation but must not leak into L_AB.
        """
        rewards = np.asarray(pgx_state.rewards, dtype=np.float32).copy()
        if not self.config.multi_objective or not bool(np.asarray(pgx_state.terminated)):
            return rewards
        solved = int(np.asarray(pgx_state._n)) == 1
        ratio = float(np.exp(float(np.asarray(pgx_state._log_ratio))))
        crossings = int(np.asarray(pgx_state._crossing_changes))
        worst = (ratio + 1.0) * self.config.simplify_budget
        cost = ratio * crossings + int(semantic_moves)
        payoff = 1.0 - 2.0 * min(max(cost / max(worst, 1.0), 0.0), 1.0) if solved else -1.0
        simplifier = 1 - int(np.asarray(pgx_state._scrambler))
        rewards[simplifier] = payoff
        rewards[1 - simplifier] = -payoff
        return rewards

    def value_potential(self, state: Any, player: int) -> float:
        pgx_state = state.pgx
        if not self.config.multi_objective or bool(np.asarray(pgx_state.terminated)):
            return 0.0
        ratio = float(np.exp(float(np.asarray(pgx_state._log_ratio))))
        crossings = int(np.asarray(pgx_state._crossing_changes))
        moves = int(getattr(state, "semantic_moves", 0))
        worst = (ratio + 1.0) * self.config.simplify_budget
        simplifier_potential = -2.0 * (ratio * crossings + moves) / worst
        simplifier = 1 - int(np.asarray(pgx_state._scrambler))
        return simplifier_potential if player == simplifier else -simplifier_potential

    def certified_value(self, state: Any, predicted: float) -> float:
        """Clamp a search value to the optional theorem-certified cost floor."""
        if not self.config.certified_value_floor:
            return float(predicted)
        from rf_knots.certified_value import certified_floor, clamp_value

        pgx_state = state.pgx
        if bool(np.asarray(pgx_state.terminated)):
            return float(predicted)
        word = tuple(int(value) for value in np.asarray(pgx_state._word) if int(value))
        strands = int(np.asarray(pgx_state._n))
        ratio = float(np.exp(float(np.asarray(pgx_state._log_ratio))))
        floor = certified_floor(word, strands, ratio=ratio)
        crossings = int(np.asarray(pgx_state._crossing_changes))
        semantic_moves = int(getattr(state, "semantic_moves", 0))
        spent = ratio * crossings + semantic_moves
        cap = (ratio + 1.0) * self.config.simplify_budget
        simplifier = 1 - int(np.asarray(pgx_state._scrambler))
        solver_to_move = int(np.asarray(pgx_state.current_player)) == simplifier
        clamped = clamp_value(
            predicted,
            floor,
            cap,
            spent_cost=spent,
            solver_to_move=solver_to_move,
        )
        stats = self.certified_value_stats
        stats["evaluations"] += 1
        if floor.informative:
            stats["informative"] += 1
            methods = stats["by_method"]
            methods[floor.method] = methods.get(floor.method, 0) + 1
        if clamped != float(predicted):
            stats["binding"] += 1
        return clamped

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

    def semantic_move_count(self, state: Any) -> int:
        return int(getattr(state, "semantic_moves", 0))

    def native_ply_count(self, state: Any) -> int:
        return max(self.config.simplify_budget - int(np.asarray(state.pgx._budget)), 0)

    def internal_ply_count(self, state: Any) -> int:
        return max(self.native_ply_count(state) - self.semantic_move_count(state), 0)

    # -- observation and legality --------------------------------------------

    def _legal(self, pgx_state: Any, head: int, internal_steps: int = 0) -> np.ndarray:
        full = np.asarray(pgx_state.legal_action_mask, dtype=bool)
        mask = np.zeros(self.num_actions, dtype=bool)
        length = int(np.asarray(pgx_state._word).astype(bool).sum())
        cache_key = (head % max(length, 1), length)
        action_map = self._underlying_action_cache.get(cache_key)
        if action_map is None:
            action_map = np.fromiter(
                (
                    int(self.underlying_action(action, head, length))
                    for action in range(self._shift_base)
                ),
                dtype=np.int64,
                count=self._shift_base,
            )
            self._underlying_action_cache[cache_key] = action_map
        mask[: self._shift_base] = full[action_map]
        # Toggles are always available while the episode runs: the control state is
        # the agent's own, not a function of the word.
        internal_allowed = internal_steps < self.config.serial_internal_horizon
        if not bool(np.asarray(pgx_state.terminated)) and internal_allowed:
            mask[self._register_base : self._colour_base] = True
            if self.colours:
                mask[self._colour_base :] = True
        # Shifting is pointless on a word too short to move within, and it must
        # never be the only option -- PASS remains the guaranteed fallback. A
        # stride that is a multiple of the current length is a no-op on the
        # necklace, so it is masked out rather than offered as a wasted ply.
        if length > 1 and not bool(np.asarray(pgx_state.terminated)) and internal_allowed:
            for index, stride in enumerate(self.strides):
                if stride % length:
                    start = self._shift_base + 2 * index * self._tape_variants
                    mask[start : start + 2 * self._tape_variants] = True
        if not mask.any():
            mask[self._singleton_base + 3] = True
        return mask

    def _no_registers(self) -> np.ndarray:
        return np.zeros(self.registers, dtype=np.float32)

    def _no_colours(self) -> np.ndarray:
        # 0 means unpainted; a painted strand carries 1..colours.
        return np.zeros(self.config.max_strands, dtype=np.int64)

    def _no_tape(self) -> np.ndarray:
        return np.zeros(self.config.max_len, dtype=np.int64)

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

    @staticmethod
    def _strand_graph_planes(word: np.ndarray, strands: int, observed_width: int) -> np.ndarray:
        """Compile the closed braid into four crossing-neighbour pointers.

        Every crossing has two incidences, one on each physical strand.  The
        planes give previous/next crossing positions along incidence 0 and then
        incidence 1.  Following the closure matters: the strand leaving bottom
        height ``h`` continues at top height ``h``, which may be a differently
        labelled open-braid strand.
        """
        length = len(word)
        planes = np.zeros((observed_width, 4), dtype=np.float32)
        if not length:
            return planes

        at_height = list(range(strands))
        occurrences: list[list[tuple[int, int]]] = [[] for _ in range(strands)]
        for position, letter in enumerate(word):
            generator = abs(int(letter)) - 1
            seam = generator == strands - 1
            if not 0 <= generator < strands - 1 and not seam:
                raise ValueError(f"generator {generator + 1} is invalid for {strands} strands")
            left, right = (strands - 1, 0) if seam else (generator, generator + 1)
            lower, upper = at_height[left], at_height[right]
            occurrences[lower].append((position, 0))
            occurrences[upper].append((position, 1))
            at_height[left], at_height[right] = upper, lower

        # An open strand label ending at bottom height h continues, through the
        # closure arc, as the strand whose top label is h.
        successor = [0] * strands
        for bottom_height, label in enumerate(at_height):
            successor[label] = bottom_height

        next_incidence: dict[tuple[int, int], tuple[int, int]] = {}
        for label, path in enumerate(occurrences):
            if not path:
                continue
            for current, following in zip(path, path[1:], strict=False):
                next_incidence[current] = following
            next_label = successor[label]
            for _ in range(strands):
                if occurrences[next_label]:
                    next_incidence[path[-1]] = occurrences[next_label][0]
                    break
                next_label = successor[next_label]
            else:  # pragma: no cover - current path itself is non-empty
                raise RuntimeError("closed strand component has no crossing incidence")

        previous_incidence = {following: current for current, following in next_incidence.items()}
        scale = float(max(observed_width - 1, 1))
        for position in range(length):
            for incidence in (0, 1):
                current = (position, incidence)
                previous = previous_incidence[current][0]
                following = next_incidence[current][0]
                planes[position, 2 * incidence] = previous / scale
                planes[position, 2 * incidence + 1] = following / scale
        return planes

    @staticmethod
    def braid_raster_planes(
        word: np.ndarray,
        strands: int,
        max_strands: int,
        cyclic_band_generators: bool = False,
    ) -> np.ndarray:
        """Encode an Artin word as ``(columns, rows, [route bits, active])``.

        For positive ``sigma_i`` the strand entering row ``i`` goes right and
        over (011), while the strand entering row ``i+1`` goes left and under
        (100); a negative letter reverses over/under.  Non-participating active
        strands go straight (010).  Inactive capacity rows are 000 with mask 0.

        The active mask is not optional: without it an absent strand and a
        padding cell both look like 000.  Keeping this compiler deterministic
        also prevents the network from seeing locally inconsistent crossing
        pairs which cannot represent a braid.
        """
        if not 1 <= strands <= max_strands:
            raise ValueError(f"strands={strands} outside 1..{max_strands}")
        raster = np.zeros((len(word), max_strands, 4), dtype=np.float32)
        raster[:, :strands, 1] = 1.0  # 010: an active straight strand
        raster[:, :strands, 3] = 1.0  # explicit active-row mask
        for column, raw_letter in enumerate(word):
            letter = int(raw_letter)
            generator = abs(letter)
            seam = cyclic_band_generators and generator == strands
            if not 1 <= generator < strands and not seam:
                raise ValueError(f"generator {generator} is invalid for {strands} strands")
            upper, lower = (strands - 1, 0) if seam else (generator - 1, generator)
            if letter > 0:
                raster[column, upper, :3] = (0.0, 1.0, 1.0)  # right, over
                raster[column, lower, :3] = (1.0, 0.0, 0.0)  # left, under
            else:
                raster[column, upper, :3] = (0.0, 0.0, 1.0)  # right, under
                raster[column, lower, :3] = (1.0, 1.0, 0.0)  # left, over
        return raster

    @staticmethod
    def word_from_braid_raster(
        raster: np.ndarray,
        strands: int,
        cyclic_band_generators: bool = False,
    ) -> np.ndarray:
        """Strict inverse used by representation tests and data validation."""
        if raster.ndim != 3 or raster.shape[2] != 4:
            raise ValueError("raster must have shape (columns, rows, 4)")
        letters: list[int] = []
        straight = np.asarray((0, 1, 0), dtype=np.float32)
        for column in raster:
            active = column[:, 3] > 0.5
            if not np.array_equal(active, np.arange(len(active)) < strands):
                raise ValueError("active mask is not a contiguous strand prefix")
            changed = [
                row for row in range(strands) if not np.array_equal(column[row, :3], straight)
            ]
            # An all-straight column is an explicit identity slice.  It has no
            # Artin letter and therefore disappears when decoding to the compact
            # word, exactly as intended for safe column insertion.
            if not changed:
                continue
            is_seam = cyclic_band_generators and changed == [0, strands - 1]
            if len(changed) != 2 or (changed[1] != changed[0] + 1 and not is_seam):
                raise ValueError("column is not one paired Artin crossing")
            upper, lower = (strands - 1, 0) if is_seam else tuple(changed)
            pair = (tuple(column[upper, :3]), tuple(column[lower, :3]))
            if pair == ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)):
                sign = 1
            elif pair == ((0.0, 0.0, 1.0), (1.0, 1.0, 0.0)):
                sign = -1
            else:
                raise ValueError("crossing halves disagree on direction or sign")
            letters.append(sign * (strands if is_seam else upper + 1))
        return np.asarray(letters, dtype=np.int32)

    def _view(
        self,
        pgx_state: Any,
        head: int,
        registers: np.ndarray,
        colours: np.ndarray,
        colour: int,
        tape: np.ndarray,
        reward: float,
        internal_steps: int = 0,
        semantic_moves: int = 0,
    ) -> Transition:
        observation = np.asarray(pgx_state.observation, dtype=np.float32)  # (L, C)
        objective_remaining = None
        if self.config.objective_budget_channel:
            ratio = float(np.exp(float(np.asarray(pgx_state._log_ratio))))
            spent = ratio * int(np.asarray(pgx_state._crossing_changes)) + int(semantic_moves)
            cap = (ratio + 1.0) * self.config.simplify_budget
            remaining = np.clip((cap - spent) / max(cap, 1.0), -1.0, 1.0)
            objective_remaining = remaining
        word = np.asarray(pgx_state._word)
        length = int((word != 0).sum())
        # The word is cyclic: gather the window with wraparound. When the word is
        # shorter than the window the same positions repeat, which is what the
        # necklace actually looks like.
        full_scan = bool(
            self.config.serial_encoder
            or self.config.serial_ensemble
            or self.config.serial_raster == "scalable"
        )
        if length > 0 and full_scan:
            # A head-relative complete scan: occupied letters first, then the
            # environment's padding slots. The action semantics remain local and
            # O(1); only the candidate encoder changes.
            occupied = (head + np.arange(length)) % length
            indexes = np.concatenate([occupied, np.arange(length, self.config.max_len)])
        elif length > 0:
            indexes = (head + np.arange(self.window) - self._window_origin) % length
        else:
            width = self.config.max_len if full_scan else self.window
            indexes = np.zeros(width, dtype=int)
        window = observation[indexes]
        observed_width = window.shape[0]
        if self.registers:
            # Broadcast along the window, like the environment's own scalar planes.
            # Carrying the control state in the observation rather than in a
            # separate channel to the network is what keeps search, the replay
            # buffer and the training step untouched.
            planes = np.broadcast_to(registers[None, :], (observed_width, self.registers))
            window = np.concatenate([window, planes], axis=1)
        if self.colours:
            palette = self._colour_planes(colours, colour)
            planes = np.broadcast_to(palette[None, :], (observed_width, palette.shape[0]))
            window = np.concatenate([window, planes], axis=1)
        if self.tape_symbols:
            tape_window = tape[indexes]
            one_hot = np.eye(self.tape_symbols, dtype=np.float32)[tape_window]
            window = np.concatenate([window, one_hot], axis=1)
        if self.config.serial_encoder.startswith("strand-graph"):
            scanned_word = np.asarray(word[indexes[:length]], dtype=np.int32)
            graph = self._strand_graph_planes(
                scanned_word,
                int(np.asarray(pgx_state._n)),
                observed_width,
            )
            window = np.concatenate([window, graph], axis=1)
        if self.config.serial_raster:
            selected_word = np.asarray(word[indexes], dtype=np.int32)
            # A zero can only occur in a full-word padded scan.  It is encoded
            # as a completely inactive column; ordinary serial windows contain
            # occupied letters and therefore remain exactly invertible.
            raster = np.zeros((observed_width, self.config.max_strands, 4), dtype=np.float32)
            if self.config.serial_raster_identity_padding:
                strands = int(np.asarray(pgx_state._n))
                raster[:, :strands, 1] = 1.0
                raster[:, :strands, 3] = 1.0
            occupied_columns = selected_word != 0
            if occupied_columns.any():
                raster[occupied_columns] = self.braid_raster_planes(
                    selected_word[occupied_columns],
                    int(np.asarray(pgx_state._n)),
                    self.config.max_strands,
                    self.config.cyclic_band_generators,
                )
            window = np.concatenate([window, raster.reshape(observed_width, -1)], axis=1)
        fraction = min(internal_steps / self.config.serial_internal_horizon, 1.0)
        if self.config.serial_internal_budget_remaining:
            fraction = 1.0 - fraction
        plane = np.full((observed_width, 1), fraction, dtype=np.float32)
        window = np.concatenate([window, plane], axis=1)
        if objective_remaining is not None:
            # Append after registers, colours and tape so every historical
            # channel keeps its index. Checkpoint migration then only has to
            # zero one genuinely new final input, including for tape agents.
            plane = np.full((observed_width, 1), objective_remaining, dtype=np.float32)
            window = np.concatenate([window, plane], axis=1)
        terminal_rewards = self._semantic_final_rewards(pgx_state, semantic_moves)
        if bool(np.asarray(pgx_state.terminated)):
            reward = float(terminal_rewards[int(np.asarray(pgx_state.current_player))])
        return Transition(
            state=SerialState(
                pgx_state,
                head,
                registers,
                colours,
                colour,
                tape,
                internal_steps,
                semantic_moves,
            ),
            observation=window.reshape(1, observed_width, window.shape[1]),
            legal_actions=self._legal(pgx_state, head, internal_steps),
            reward=reward,
            terminated=bool(np.asarray(pgx_state.terminated)),
            player=int(np.asarray(pgx_state.current_player)),
            move_count=int(np.asarray(pgx_state._step_count)),
            consecutive_passes=0,
            termination_reason=(
                "solved"
                if bool(np.asarray(pgx_state.terminated)) and int(np.asarray(pgx_state._n)) == 1
                else "move_budget_exhausted"
                if bool(np.asarray(pgx_state.terminated))
                else ""
            ),
        )
