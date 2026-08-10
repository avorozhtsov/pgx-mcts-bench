"""Pure replay and verification for portable braid solutions.

This module deliberately has no policy network, search tree, or game clock.  It
is the architecture-independent boundary between a scientist's native action
space and a portable sequence of semantic braid operations.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from rf_knots import reference
from rf_knots.actions import ActionSpec
from rf_knots.evidence import BraidState, UnknotWitness


@dataclass(frozen=True)
class SemanticBraidVerifier:
    """Apply and certify semantic actions without a trainable parallel game."""

    spec: ActionSpec

    @classmethod
    def from_config(cls, config: Any) -> SemanticBraidVerifier:
        return cls(
            ActionSpec(
                max_len=int(config.max_len),
                max_strands=int(config.max_strands),
                cyclic_band_generators=bool(config.cyclic_band_generators),
            )
        )

    def state(self, word: Iterable[int], strands: int) -> BraidState:
        return BraidState(
            tuple(int(value) for value in word if int(value)),
            int(strands),
            self.spec.cyclic_band_generators,
        )

    def legal_actions(
        self, state: BraidState, *, allow_crossing_change: bool = True
    ) -> tuple[int, ...]:
        return tuple(
            action
            for action in range(self.spec.num_actions)
            if reference.is_legal(
                self.spec,
                state.word,
                state.strands,
                action,
                allow_crossing_change,
            )
        )

    def apply(
        self,
        state: BraidState,
        action: int,
        *,
        allow_crossing_change: bool = True,
    ) -> BraidState:
        action = int(action)
        if not reference.is_legal(
            self.spec,
            state.word,
            state.strands,
            action,
            allow_crossing_change,
        ):
            raise ValueError(f"illegal semantic action: {self.spec.describe(action)}")
        word, strands = reference.apply(self.spec, state.word, state.strands, action)
        return BraidState(word, strands, self.spec.cyclic_band_generators)

    def verify_actions(
        self,
        word: Iterable[int],
        strands: int,
        actions: Iterable[int],
    ) -> UnknotWitness:
        return UnknotWitness.from_actions(word, strands, self.spec, actions)

    def verify_states(self, states: Iterable[BraidState]) -> UnknotWitness:
        return UnknotWitness.from_states(states, self.spec)
