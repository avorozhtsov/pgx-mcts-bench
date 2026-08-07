from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pgx_mcts_bench.joint_pretraining import (
    _portfolio_retention_decision,
    joint_pretraining_candidates,
    migration_equivalence,
    run_joint_pretraining,
)
from pgx_mcts_bench.ladder import STAGES, _config, serial_arms
from pgx_mcts_bench.networks import make_braid_network


def _historical_checkpoint(path: Path) -> None:
    candidate = next(candidate for candidate in serial_arms() if candidate.name == "s-window-128")
    config = _config(candidate, STAGES[0], 0, "cpu", selfplay_games=1)
    network = make_braid_network(config.game, config.model)
    torch.save({"network": network.state_dict(), "candidate": candidate.name}, path)


def test_joint_candidates_keep_capacity_fixed_and_separate_h5() -> None:
    candidates = joint_pretraining_candidates(simulations=16, train_steps=3)

    for candidate in candidates.values():
        assert candidate.channels == 32
        assert candidate.residual_blocks == 2
        assert candidate.objective_budget_channel
        assert not candidate.auxiliary_backprop_to_encoder
        assert candidate.auxiliary_solve_backprop_to_encoder
        assert candidate.auxiliary_budget_conditioning
        assert candidate.simulations == 16
        assert candidate.train_steps == 3
        assert not candidate.use_auxiliary_value
    assert candidates["warm"].serial_internal_horizon == 0
    assert candidates["scratch"].serial_internal_horizon == 0
    assert candidates["h5"].serial_internal_horizon == 5
    assert candidates["h5"].serial_internal_budget_remaining


def test_portfolio_progress_is_primary_over_exact_cell_retention() -> None:
    before = {"solved_attempts": 10, "capped_loss": 100.0, "rows": [{}]}
    after = {
        "solved_attempts": 11,
        "capped_loss": 90.0,
        "rows": [{}],
        "minimum_solve_rate": 0.25,
    }

    decision = _portfolio_retention_decision(before, after)

    assert decision["portfolio_noninferior"]
    assert not decision["exact_cell_retention_secondary"]


def test_appended_budget_channels_are_bit_exact_at_migration(tmp_path: Path) -> None:
    checkpoint = tmp_path / "historical.pt"
    _historical_checkpoint(checkpoint)
    candidates = joint_pretraining_candidates(simulations=2, train_steps=1)

    warm = migration_equivalence(checkpoint, candidates["warm"], seed=7, device="cpu")
    h5 = migration_equivalence(checkpoint, candidates["h5"], seed=7, device="cpu")

    assert warm["migrated"]
    assert h5["migrated"]
    assert warm["bit_exact"]
    assert h5["bit_exact"]
    assert warm["target_observation_channels"] == warm["source_observation_channels"] + 1
    assert h5["target_observation_channels"] == h5["source_observation_channels"] + 2


def test_warm_checkpoint_cannot_silently_rewind_to_easy_rungs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "promoted.pt"
    _historical_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["stage_result"] = {"stage": 21}
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="refusing to rewind"):
        run_joint_pretraining(
            checkpoint,
            tmp_path / "run",
            smoke_stop_after=4,
            continue_stop_after=4,
            include_scratch=False,
            include_h5=False,
        )
