from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pgx_mcts_bench.positive_acquisition import _evaluation_aligned_rng


class _Generator:
    sources = (SimpleNamespace(name="fixed"),)

    def generate(self, source, moves, rng):
        del source, moves
        rng.integers(0, 100, size=5)


def test_acquisition_rng_matches_evaluation_after_representation_generation() -> None:
    game = SimpleNamespace(
        generator=_Generator(),
        config=SimpleNamespace(
            stage_mix=(),
            stage_source="fixed",
            stage_scramble=0,
            log_ratio_range=(0.0, 0.0),
        ),
    )
    expected = np.random.default_rng(37)
    game.generator.generate(game.generator.sources[0], 0, expected)

    actual = _evaluation_aligned_rng(game, 37)

    assert actual.integers(0, 1_000_000) == expected.integers(0, 1_000_000)
