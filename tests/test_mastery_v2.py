from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from rf_knots.actions import ActionSpec
from rf_knots.evidence import BraidState, UnknotWitness

from pgx_mcts_bench.mastery_v2 import (
    AttemptDeadlineExceeded,
    CertificationEngine,
    DoseCalibrationTable,
    EvidenceSnapshot,
    admissible_negative,
    hard_deadline,
    outcome_class,
    parameter_capacity_report,
)


def test_outcome_classes_do_not_mix_relaxed_and_scientific_successes():
    assert (
        outcome_class(solved=True, reason="verified", attempt_target_u=2, scientific_target_u=2)
        == "strict_challenge_success"
    )
    assert (
        outcome_class(solved=True, reason="verified", attempt_target_u=3, scientific_target_u=2)
        == "relaxed_training_success"
    )
    assert (
        outcome_class(
            solved=False, reason="hard_timeout", attempt_target_u=2, scientific_target_u=2
        )
        == "hard_timeout"
    )
    assert (
        outcome_class(
            solved=False,
            reason="unsupported_capacity",
            attempt_target_u=2,
            scientific_target_u=2,
        )
        == "unsupported_capacity"
    )
    assert admissible_negative("objective_budget_exhausted") is True
    assert admissible_negative("hard_timeout") is False
    assert admissible_negative("unsupported_capacity") is False


def test_hard_deadline_interrupts_and_restores_alarm_handler():
    with pytest.raises(AttemptDeadlineExceeded):
        with hard_deadline(0.01):
            time.sleep(0.1)
    with hard_deadline(0.05):
        pass


def test_dose_calibration_is_conditioned_on_simulation_count():
    table = DoseCalibrationTable(bins=5, prior_strength=1.0)
    for _ in range(4):
        table.observe(32, 0.6, False, 1.0)
        table.observe(128, 0.6, True, 4.0)
    low, low_n, _ = table.calibrate(32, 0.6)
    high, high_n, _ = table.calibrate(128, 0.6)
    assert low_n == high_n == 4
    assert high > low
    restored = DoseCalibrationTable.from_dict(table.to_dict())
    assert restored.calibrate(128, 0.6) == table.calibrate(128, 0.6)


def test_evidence_snapshot_is_hash_pinned_and_supplies_known_upper(tmp_path: Path):
    path = tmp_path / "best-solutions-pool.json"
    path.write_text(
        json.dumps(
            {
                "schema": "q-r-skm-evidence-catalog-v2",
                "verified": {
                    "best_by_representation": {"instance": {"evidence_id": "e"}},
                    "best_by_knot": {"K": {"crossing_changes": 2}},
                },
            }
        )
    )
    snapshot = EvidenceSnapshot.load(path)
    assert len(snapshot.sha256) == 64
    assert snapshot.best_upper("K", 4) == 2
    assert snapshot.best_upper("missing", 4) == 4


def test_first_improvement_certificate_never_overclaims_exactness(tmp_path: Path):
    witness = UnknotWitness.from_actions((), 1, ActionSpec(16, 12), [])
    report = CertificationEngine(tmp_path).certify(
        evidence_id="evidence",
        knot_name="not-in-table",
        root=BraidState((), 1),
        witness=witness,
        declared_lower_bound=0,
    )
    assert report["witness_replay_verified"] is True
    assert report["status"] in {"upper-bound-certified", "exact-certified"}
    assert (tmp_path / "evidence.json").is_file()


def test_capacity_audit_counts_network_weights_without_claiming_sufficiency(tmp_path: Path):
    small = tmp_path / "small.pt"
    large = tmp_path / "large.pt"
    torch.save({"network": {"weight": torch.zeros(2, 3)}}, small)
    torch.save({"network": {"weight": torch.zeros(8, 3)}}, large)
    report = parameter_capacity_report(
        [("small", small), ("large", large)], {"small": 0.5, "large": 0.9}
    )
    assert report["scientists"][0]["network_parameters"] == 6
    assert report["scientists"][0]["interpretation"].startswith("capacity-risk")
    assert "cannot establish" in report["warning"]
