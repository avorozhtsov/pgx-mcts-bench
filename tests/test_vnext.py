from __future__ import annotations

import numpy as np
import torch

from pgx_mcts_bench.game import make_game, sample_log_ratio
from pgx_mcts_bench.ladder import STAGES, _config, vnext_arms
from pgx_mcts_bench.networks import make_braid_network
from pgx_mcts_bench.vnext import (
    AdaptiveState,
    BlockMetrics,
    capped_portfolio_cost,
    decide_block,
    donation_is_eligible,
    registered_manifest,
    resolved_schedule,
    semantic_cost,
)


def test_vnext_names_are_stable_and_compute_free() -> None:
    arms = vnext_arms()
    assert [arm.name for arm in arms] == [
        "window-local",
        "raster-axial",
        "cyclic-memory",
        "strand-graph",
    ]
    assert all("64" not in arm.name and "128" not in arm.name for arm in arms)
    assert all(arm.objective_ratios == (1000.0, 10.0) for arm in arms)
    assert all(arm.objective_ratio_weights == (3.0, 1.0) for arm in arms)
    assert all(arm.objective_budget_channel for arm in arms)
    assert all(arm.serial_internal_budget_remaining for arm in arms)
    assert all(arm.serial_internal_horizon == 5 for arm in arms)
    assert all(arm.auxiliary_solve_backprop_to_encoder for arm in arms)


def test_every_vnext_scientist_runs_from_a_fresh_initialization() -> None:
    for index, arm in enumerate(vnext_arms()):
        torch.manual_seed(index)
        config = _config(arm, STAGES[0], index, "cpu", selfplay_games=1)
        game = make_game(config.game)
        network = make_braid_network(config.game, config.model)
        transition = game.from_word([1, -2, 3, -1], strands=4)
        observation = torch.from_numpy(transition.observation[None]).permute(0, 3, 1, 2)
        policy, value = network(observation.float())
        assert policy.shape == (1, config.game.action_size)
        assert value.shape == (1,)


def test_manifest_forbids_checkpoint_inheritance() -> None:
    manifest = registered_manifest(seed=19)
    assert manifest["initialization"] == "from-scratch"
    assert manifest["initial_schedule"] == {
        "F_native": 5,
        "F_old": 1,
        "simulations_per_move": 64,
        "donation_fraction": 0.0,
    }
    assert not any("checkpoint" in scientist for scientist in manifest["scientists"])


def test_vnext_samples_only_registered_objectives_with_l1000_bias() -> None:
    arm = vnext_arms()[0]
    config = _config(arm, STAGES[0], 0, "cpu").game
    sampled = np.exp(
        [sample_log_ratio(config, np.random.default_rng(seed)) for seed in range(200)]
    )
    assert set(np.round(sampled)) == {10.0, 1000.0}
    assert np.count_nonzero(sampled > 100) > 2 * np.count_nonzero(sampled < 100)


def test_strict_donation_filter() -> None:
    assert donation_is_eligible(
        donor_cost=12,
        receiver_native_cost=13,
        donor_verified=True,
        same_representation=True,
        same_objective=True,
    )
    assert not donation_is_eligible(
        donor_cost=12,
        receiver_native_cost=12,
        donor_verified=True,
        same_representation=True,
        same_objective=True,
    )
    assert not donation_is_eligible(
        donor_cost=11,
        receiver_native_cost=12,
        donor_verified=False,
        same_representation=True,
        same_objective=True,
    )
    assert semantic_cost(1, 2, 10) == 12


def test_weak_block_raises_native_search_and_rehearsal_and_rolls_back() -> None:
    metrics = BlockMetrics(
        acquisition_rate=0.5,
        evaluation_rate=0.6,
        retention_rate=0.7,
        eligible_donations=8,
        solved_before=frozenset({"a", "b"}),
        solved_after=frozenset({"b"}),
        capped_cost_before=40,
        capped_cost_after=50,
    )
    decision = decide_block(AdaptiveState(donation_index=2), metrics)
    assert not decision.accept
    assert decision.lost == ("a",)
    assert resolved_schedule(decision.state) == {
        "F_native": 8,
        "F_old": 2,
        "simulations_per_move": 128,
        "donation_fraction": 0.10,
    }


def test_healthy_block_can_raise_donation_dose() -> None:
    metrics = BlockMetrics(
        acquisition_rate=0.9,
        evaluation_rate=0.8,
        retention_rate=0.9,
        eligible_donations=3,
        solved_before=frozenset({"a"}),
        solved_after=frozenset({"a", "b"}),
        capped_cost_before=40,
        capped_cost_after=30,
    )
    decision = decide_block(AdaptiveState(), metrics)
    assert decision.accept
    assert decision.gained == ("b",)
    assert resolved_schedule(decision.state)["donation_fraction"] == 0.10


def test_capped_portfolio_cost_uses_best_or_failure_cap() -> None:
    assert capped_portfolio_cost({"a": 12, "b": None, "c": 300}, 100) == 212
