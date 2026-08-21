from __future__ import annotations

import json
from pathlib import Path

from rf_knots.actions import ActionSpec
from rf_knots.evidence import UnknotWitness, braid_instance_id

from pgx_mcts_bench.dkt72_v2 import (
    audit_training_leakage,
    certify_improvement,
    freeze_protocol,
    preflight,
    register_checkpoint_eligibility,
    select_two_checkpoint_systems,
)


def _write_panel(path: Path) -> Path:
    instances = []
    for index in range(72):
        word = [1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        instances.append(
            {
                "source_id": f"test-{index:03d}",
                "instance_id": braid_instance_id(word, 12),
                "payload": {"word": word, "strands": 12},
            }
        )
    path.write_text(json.dumps({"instances": instances}))
    return path


def _metadata(checkpoint: Path, scientist: str, *, cyclic: bool = False) -> dict:
    return {
        "scientist": scientist,
        "architecture": scientist,
        "solver_version": "test-v1",
        "checkpoint": str(checkpoint),
        "max_strands": 12,
        "cyclic_band_generators": cyclic,
        "q_gate_passed": True,
        "training_strand_counts": {str(strand): 10 for strand in range(6, 13)},
        "retention_by_strand": {str(strand): 0.9 for strand in range(6, 13)},
        "native_high_strand_successes": 2,
        "capacity_probe_supported": 72,
        "capacity_probe_total": 72,
        "training_representation_ids": [],
        "training_knot_ids": [],
        "selection_metrics": {
            "strict_high_strand_successes": 2,
            "capped_l1000": 2000.0,
            "network_evaluations": 100,
        },
    }


def test_eligibility_is_fail_closed_and_audits_both_leakage_levels(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "panel.json")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    good = register_checkpoint_eligibility(
        _metadata(checkpoint, "ordinary"), panel, tmp_path / "good.json"
    )
    assert good["eligible"] is True
    assert good["capacity_probe"] == {"supported": 72, "total": 72}

    bad_metadata = _metadata(checkpoint, "bad")
    bad_metadata["retention_by_strand"]["9"] = 0.79
    bad_metadata["training_representation_ids"] = [
        json.loads(panel.read_text())["instances"][0]["instance_id"]
    ]
    bad_metadata["training_knot_ids"] = ["test-001"]
    bad = register_checkpoint_eligibility(bad_metadata, panel, tmp_path / "bad.json")
    assert bad["eligible"] is False
    assert "retention_below_0.80:9" in bad["ineligibility_reasons"]
    assert "training_representation_overlap=1" in bad["ineligibility_reasons"]
    assert "training_knot_overlap=1" in bad["ineligibility_reasons"]


def test_leakage_audit_separates_representation_and_knot_identity(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "panel.json")
    panel_row = json.loads(panel.read_text())["instances"][0]
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id": panel_row["source_id"],
                        "name": panel_row["source_id"],
                        "instance_id": "different-exact-representation",
                    }
                ]
            }
        )
    )
    report = audit_training_leakage(panel, [bank], tmp_path / "leakage.json")
    assert report["exact_representation_overlap"] == []
    assert report["knot_identity_overlap"] == [panel_row["source_id"]]
    assert report["eligible"] is False


def test_selection_exposes_cyclic_band_system_as_explicit_comparison(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "panel.json")
    reports = []
    for name, cyclic in (("ordinary", False), ("bstar", True), ("ordinary-2", False)):
        checkpoint = tmp_path / f"{name}.pt"
        checkpoint.write_bytes(name.encode())
        report = tmp_path / f"{name}.json"
        register_checkpoint_eligibility(_metadata(checkpoint, name, cyclic=cyclic), panel, report)
        reports.append(report)
    selection = select_two_checkpoint_systems(reports, tmp_path / "selection.json")
    assert selection["status"] == "selected"
    assert selection["selection_policy"] == "best-ordinary-plus-best-cyclic-band"
    assert {row["cyclic_band_generators"] for row in selection["selected"]} == {False, True}


def test_frozen_protocol_and_preflight_pin_all_inputs(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "panel.json")
    reports = []
    for name in ("a", "b"):
        checkpoint = tmp_path / f"{name}.pt"
        checkpoint.write_bytes(name.encode())
        report = tmp_path / f"{name}.json"
        register_checkpoint_eligibility(_metadata(checkpoint, name), panel, report)
        reports.append(report)
    selection_path = tmp_path / "selection.json"
    select_two_checkpoint_systems(reports, selection_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = freeze_protocol(
        panel=panel,
        selection=selection_path,
        output=protocol_path,
        simulations=256,
        attempts=4,
        action_horizon=128,
        seed=20267202,
    )
    assert protocol["learning"] is False
    assert protocol["primary_denominator"] == 72
    assert preflight(protocol_path, tmp_path / "preflight.json")["status"] == "eligible-to-launch"

    Path(protocol["scientists"][0]["checkpoint"]).write_bytes(b"changed")
    failed = preflight(protocol_path, tmp_path / "failed.json")
    assert failed["status"] == "blocked"
    assert failed["failures"][0].startswith("checkpoint_changed:")


def test_strict_improvement_is_recorded_and_certified(tmp_path: Path) -> None:
    witness = UnknotWitness.from_actions((), 1, ActionSpec(16, 12), [])
    witness_path = tmp_path / "witness.json"
    witness_path.write_text(json.dumps(witness.to_dict()))
    result = certify_improvement(
        witness_path=witness_path,
        inventory_root=tmp_path / "inventory",
        certificate_root=tmp_path / "certificates",
        knot_name="not-in-table",
        representation_id="test-000",
        scientist="scientist-v2",
        previous_upper_bound=1,
        declared_lower_bound=0,
        solver_metadata={"checkpoint_sha256": "abc", "solver_version": "v2"},
        search_parameters={"simulations": 256, "action_horizon": 128},
    )
    assert result["certificate"]["witness_replay_verified"] is True
    assert (tmp_path / "inventory" / "witnesses" / f"{result['evidence_id']}.json").is_file()
    assert (tmp_path / "certificates" / f"{result['evidence_id']}.json").is_file()
