from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import (
    FixedWordGame,
    KnotItem,
    choose_group_proposal,
    choose_proposals,
    simplicity,
    smallest_crossing_pool,
    softmax,
)
from pgx_mcts_bench.training import upper_bound_cost_loss


def test_smallest_crossing_pool_is_frozen_sorted_and_compatible() -> None:
    pool = smallest_crossing_pool(200)
    assert len(pool) == 200
    assert [(k.crossings, k.name) for k in pool] == sorted(
        (k.crossings, k.name) for k in pool
    )
    assert all(k.crossings > 0 and k.strands <= 5 and len(k.word) <= 48 for k in pool)
    assert pool[-1].crossings == 11


def test_simplicity_is_exact_proposed_formula() -> None:
    np.testing.assert_allclose(
        simplicity(np.array([0.5, 0.8]), np.array([4.0, 10.0])),
        np.array([8.0, 8.0]),
    )


def test_softmax_and_proposals_are_seed_replayable() -> None:
    np.testing.assert_allclose(softmax(np.array([1000.0, 1001.0]), 1.0).sum(), 1.0)
    rows = [np.array([1.0, 3.0]), np.array([4.0, 2.0])]
    assert choose_proposals(rows, np.random.default_rng(7), 0.0) == [1, 0]


def test_starvation_overrides_group_priority_with_scientists_best_knot() -> None:
    rows = [np.array([100.0, 0.0]), np.array([1.0, 2.0])]
    scientist, knot, reason = choose_group_proposal(
        rows,
        proposals=[0, 0],
        ignored_rounds=[0, 4],
        rng=np.random.default_rng(0),
        alpha=100.0,
        temperature=1.0,
        starvation_rounds=4,
    )
    assert (scientist, knot, reason) == (1, 1, "starvation")


def test_shared_cost_target_is_an_upper_bound_not_an_equality() -> None:
    prediction = torch.tensor([3.0, 7.0, 3.0])
    target = torch.tensor([5.0, 5.0, 5.0])
    shared = torch.tensor([True, True, False])
    loss = upper_bound_cost_loss(prediction, target, shared)
    assert loss[0].item() == 0.0
    assert loss[1].item() > 0.0
    assert loss[2].item() > 0.0


def test_fixed_word_game_delegates_value_potential() -> None:
    base = SimpleNamespace(
        config=object(),
        value_potential=lambda state, player: float(state + player),
    )
    fixed = FixedWordGame(base, KnotItem("test", 1, (1,), 2))
    assert fixed.value_potential(3, 2) == 5.0
