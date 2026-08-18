from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pgx_mcts_bench.mastery_v3_distill import (
    certified_budget_labels,
    masked_policy_kl,
    ordinal_targets,
)
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


def test_policy_preservation_ignores_logits_for_illegal_actions() -> None:
    parent = torch.tensor([[2.0, 0.0, -1.0, 17.0]])
    child = torch.tensor([[2.0, 0.0, -1.0, -23.0]])
    legal = torch.tensor([[True, True, True, False]])
    assert masked_policy_kl(child, parent, legal).item() == pytest.approx(0.0, abs=1e-6)
    raw = torch.nn.functional.kl_div(
        torch.log_softmax(child, dim=1),
        torch.softmax(parent, dim=1),
        reduction="batchmean",
    )
    assert raw.item() > 1.0


def test_policy_preservation_fails_closed_without_legal_action() -> None:
    logits = torch.zeros((1, 3))
    with pytest.raises(ValueError, match="at least one legal action"):
        masked_policy_kl(logits, logits, torch.zeros_like(logits, dtype=torch.bool))


def test_local_q4000_audit_uses_semantic_bank_hash(tmp_path) -> None:
    group = {
        "name": "q1",
        "size": 1,
        "skip_policy": {"maximum_skips": 0},
        "rows": [{"id": "k", "strands": 6}],
    }
    group_path = tmp_path / "group.json"
    group_path.write_text(json.dumps(group, indent=2) + "\n")
    output = tmp_path / "output"
    (output / "native-events").mkdir(parents=True)
    (output / "native-events" / "000.json").write_text("{}\n")
    (output / "state.pt.gz").write_bytes(b"state")
    report = {
        "bank_sha256": hashlib.sha256(
            json.dumps(group, sort_keys=True).encode()
        ).hexdigest(),
        "completed_rungs": 1,
        "processed": ["k"],
        "curriculum_skips": {"scientist": 0},
        "events": [
            {
                "selected": "k",
                "scientists": {
                    "scientist": {
                        "curriculum_skip": None,
                        "iterations": [{"selfplay_solved": 1}],
                    }
                },
            }
        ],
        "block_reports": [
            {
                "scientists": {
                    "scientist": {
                        "retention_after": {"attempts": 1, "solved": 1}
                    }
                }
            }
        ],
    }
    (output / "report.json").write_text(json.dumps(report) + "\n")
    script = Path(__file__).parents[1] / "scripts" / "audit_local_q4000_group.py"
    result = subprocess.run(
        [sys.executable, str(script), "--group", str(group_path), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_local_q4000_audit_applies_skip_limit_per_scientist(tmp_path) -> None:
    group = {
        "name": "q1",
        "size": 1,
        "skip_policy": {"maximum_skips": 1},
        "rows": [{"id": "k", "strands": 6}],
    }
    group_path = tmp_path / "group.json"
    group_path.write_text(json.dumps(group) + "\n")
    output = tmp_path / "output"
    (output / "native-events").mkdir(parents=True)
    (output / "native-events" / "000.json").write_text("{}\n")
    (output / "state.pt.gz").write_bytes(b"state")
    report = {
        "bank_sha256": hashlib.sha256(
            json.dumps(group, sort_keys=True).encode()
        ).hexdigest(),
        "completed_rungs": 1,
        "processed": ["k"],
        "curriculum_skips": {"a": 1, "b": 1},
        "events": [
            {
                "selected": "k",
                "scientists": {
                    "a": {
                        "curriculum_skip": {"reason": "budget_exhausted"},
                        "iterations": [{"selfplay_solved": 1}],
                    },
                    "b": {
                        "curriculum_skip": {"reason": "budget_exhausted"},
                        "iterations": [{"selfplay_solved": 0}],
                    },
                },
            }
        ],
        "block_reports": [
            {
                "scientists": {
                    "a": {"retention_after": {"attempts": 1, "solved": 1}},
                    "b": {"retention_after": {"attempts": 1, "solved": 1}},
                }
            }
        ],
    }
    (output / "report.json").write_text(json.dumps(report) + "\n")
    script = Path(__file__).parents[1] / "scripts" / "audit_local_q4000_group.py"
    result = subprocess.run(
        [sys.executable, str(script), "--group", str(group_path), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    audit = json.loads((output / "local-group-audit.json").read_text())
    assert audit["skip_count"] == 2
    assert audit["skip_counts"] == {"a": 1, "b": 1}
