from __future__ import annotations

import pytest
import torch

from pgx_mcts_bench.mastery_v3_distill import certified_budget_labels, ordinal_targets
from pgx_mcts_bench.mastery_v3_pretrain import _rotations
from pgx_mcts_bench.mastery_v3_screening import (
    _proof_distillation_gate,
    _retention,
    _upper,
)


def test_pretraining_views_are_replayable_cyclic_rotations() -> None:
    word = [1, -2, 3, -1]
    views = _rotations(word, 4, 17)
    assert len(views) == 4
    assert all(sorted(view) == sorted(word) for view in views)
    assert len({tuple(view) for view in views}) == 4


def test_screening_upper_uses_registered_certified_bound() -> None:
    assert _upper({"certified_unknotting_upper_bound": 3}) == 3
    assert _upper({"known_unknotting_number": 2}) == 2


def test_retention_is_fail_closed_when_control_has_no_solved_strand() -> None:
    control = [
        {"representation_id": "a", "seed": 1, "strands": 6, "solved": True},
        {"representation_id": "b", "seed": 1, "strands": 7, "solved": False},
    ]
    candidate = [
        {"representation_id": "a", "seed": 1, "strands": 6, "solved": True},
        {"representation_id": "b", "seed": 1, "strands": 7, "solved": True},
    ]
    overall, by_strand = _retention(control, candidate)
    assert overall == 1.0
    assert by_strand["6"] == 1.0
    assert by_strand["7"] is None


def test_proof_labels_mask_unknown_interval_and_never_cross_lower_bound() -> None:
    import numpy as np

    row = {
        "id": "k",
        "representation_id": "braid:k",
        "certified_unknotting_lower_bound": 2,
        "certified_unknotting_upper_bound": 4,
    }
    witness = {"crossing_changes": 4, "moves": 17, "evidence_id": "verified"}
    labels = certified_budget_labels(
        row,
        1000,
        np.random.default_rng(7),
        samples_per_side=20,
        witness=witness,
    )
    negatives = [label for label in labels if not label.feasible]
    positives = [label for label in labels if label.feasible]
    assert negatives and max(label.budget for label in negatives) < 2000
    assert positives and min(label.budget for label in positives) >= 4017
    targets, mask = ordinal_targets(2, 4)
    assert targets[:2] == [0.0, 0.0]
    assert mask[2:4] == [0.0, 0.0]
    assert targets[4:] == [1.0] * 9


def test_screening_requires_passed_proof_distillation(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    report = {
        "schema": "mastery-v3-proof-distillation-v1",
        "status": "passed",
        "candidate": "cyclic-memory-deep-v3",
        "curriculum_sha256": "curriculum",
        "evidence_snapshot_sha256": "evidence",
        "unsafe_labels": 0,
        "training_dose_complete": True,
        "operational_p_solve_trained": False,
    }
    torch.save({"mastery_v3_proof_distillation": report}, checkpoint)
    assert (
        _proof_distillation_gate(
            checkpoint, "cyclic-memory-deep-v3", "curriculum"
        )
        == report
    )
    report["status"] = "failed"
    torch.save({"mastery_v3_proof_distillation": report}, checkpoint)
    with pytest.raises(ValueError, match="did not pass"):
        _proof_distillation_gate(checkpoint, "cyclic-memory-deep-v3", "curriculum")
