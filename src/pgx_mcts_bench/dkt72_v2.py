"""Fail-closed registration and certification gates for DKT72-PD-v2.

The DKT72 panel is a final frozen evaluation set, not a training curriculum.
This module deliberately contains no evaluator and no service launcher.  It
only produces immutable, hash-bearing decisions that an evaluator must check
before it is allowed to run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rf_knots.evidence import BraidState, UnknotWitness, braid_instance_id

from pgx_mcts_bench.mastery_program import EvidenceInventory
from pgx_mcts_bench.mastery_v2 import CertificationEngine

ELIGIBILITY_SCHEMA = "dkt72-pd-v2-checkpoint-eligibility-v1"
SELECTION_SCHEMA = "dkt72-pd-v2-selection-v1"
PROTOCOL_SCHEMA = "dkt72-pd-v2-frozen-protocol-v1"
PREFLIGHT_SCHEMA = "dkt72-pd-v2-preflight-v1"
EXPECTED_PANEL_SIZE = 72
MIN_RETENTION = 0.80
REQUIRED_TRAINING_STRANDS = tuple(range(6, 13))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _panel_rows(panel: Path) -> list[dict[str, Any]]:
    payload = json.loads(panel.read_text())
    rows = payload.get("instances")
    if not isinstance(rows, list):
        raise ValueError("DKT72 panel must contain an instances list")
    return [dict(row) for row in rows]


def panel_identities(panel: Path) -> tuple[set[str], set[str]]:
    """Return exact braid-instance ids and declared knot ids from the panel."""

    exact: set[str] = set()
    knots: set[str] = set()
    for row in _panel_rows(panel):
        braid = row.get("payload") or {}
        instance_id = row.get("instance_id")
        if not instance_id:
            state = BraidState(
                tuple(int(value) for value in braid["word"]),
                int(braid["strands"]),
                bool(braid.get("cyclic_band_generators", False)),
            )
            instance_id = braid_instance_id(state.word, state.strands)
        exact.add(str(instance_id))
        knots.add(str(row["source_id"]))
    return exact, knots


def audit_training_leakage(
    panel: Path, training_banks: Iterable[Path], output: Path
) -> dict[str, Any]:
    """Audit exact-representation and knot-identity overlap per training bank."""

    panel_exact, panel_knots = panel_identities(panel)
    all_exact: set[str] = set()
    all_knots: set[str] = set()
    bank_reports = []
    for bank in training_banks:
        payload = json.loads(bank.read_text())
        rows = payload.get("rows", payload if isinstance(payload, list) else [])
        exact: set[str] = set()
        knots: set[str] = set()
        for row in rows:
            braid = row.get("payload", row)
            instance_id = row.get("instance_id")
            if instance_id is None and "word" in braid and "strands" in braid:
                instance_id = braid_instance_id(braid["word"], int(braid["strands"]))
            if instance_id is not None:
                exact.add(str(instance_id))
            name = next(
                (
                    row.get(key)
                    for key in ("canonical_name", "knot_name", "name", "source_id", "id")
                    if row.get(key) is not None
                ),
                None,
            )
            if name is not None:
                knots.add(str(name).split("::", 1)[0])
        all_exact.update(exact)
        all_knots.update(knots)
        bank_reports.append(
            {
                "bank": str(bank.resolve()),
                "bank_sha256": sha256_file(bank),
                "rows": len(rows),
                "exact_representation_overlap": sorted(panel_exact & exact),
                "knot_identity_overlap": sorted(panel_knots & knots),
            }
        )
    report = {
        "schema": "dkt72-pd-v2-training-leakage-audit-v1",
        "created_at_utc": _utc_now(),
        "panel": str(panel.resolve()),
        "panel_sha256": sha256_file(panel),
        "panel_rows": len(_panel_rows(panel)),
        "training_banks": bank_reports,
        "exact_representation_overlap": sorted(panel_exact & all_exact),
        "knot_identity_overlap": sorted(panel_knots & all_knots),
    }
    report["eligible"] = (
        not report["exact_representation_overlap"] and not report["knot_identity_overlap"]
    )
    report["report_sha256"] = _json_digest(report)
    _atomic_json(output, report)
    return report


def register_checkpoint_eligibility(
    metadata: dict[str, Any], panel: Path, output: Path
) -> dict[str, Any]:
    """Register a checkpoint without inspecting any DKT72 solver outcomes.

    ``metadata`` is produced by the training/capacity pipeline.  Requiring it
    as an explicit input keeps this gate independent from changing artifact
    layouts and makes missing evidence a failed prerequisite rather than an
    inferred success.
    """

    checkpoint = Path(metadata["checkpoint"])
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    rows = _panel_rows(panel)
    exact_panel, knot_panel = panel_identities(panel)
    trained_exact = {str(value) for value in metadata.get("training_representation_ids", [])}
    trained_knots = {str(value) for value in metadata.get("training_knot_ids", [])}
    exact_overlap = sorted(exact_panel & trained_exact)
    knot_overlap = sorted(knot_panel & trained_knots)
    training_counts = {
        int(key): int(value) for key, value in metadata.get("training_strand_counts", {}).items()
    }
    retention_by_strand = {
        int(key): float(value) for key, value in metadata.get("retention_by_strand", {}).items()
    }
    reasons: list[str] = []
    if len(rows) != EXPECTED_PANEL_SIZE:
        reasons.append(f"panel_size={len(rows)}!=72")
    if not bool(metadata.get("q_gate_passed", False)):
        reasons.append("q_gate_not_passed")
    if int(metadata.get("max_strands", 0)) < 12:
        reasons.append("max_strands_below_12")
    absent_training = [
        strand for strand in REQUIRED_TRAINING_STRANDS if training_counts.get(strand, 0) < 1
    ]
    if absent_training:
        reasons.append("missing_training_strands:" + ",".join(map(str, absent_training)))
    missing_retention = [
        strand for strand in REQUIRED_TRAINING_STRANDS if strand not in retention_by_strand
    ]
    if missing_retention:
        reasons.append("missing_retention_strands:" + ",".join(map(str, missing_retention)))
    failed_retention = [
        strand
        for strand in REQUIRED_TRAINING_STRANDS
        if retention_by_strand.get(strand, 0.0) < MIN_RETENTION
    ]
    if failed_retention:
        reasons.append("retention_below_0.80:" + ",".join(map(str, failed_retention)))
    if int(metadata.get("native_high_strand_successes", 0)) < 1:
        reasons.append("no_native_success_on_6plus_strands")
    supported = int(metadata.get("capacity_probe_supported", 0))
    total = int(metadata.get("capacity_probe_total", 0))
    if supported != EXPECTED_PANEL_SIZE or total != EXPECTED_PANEL_SIZE:
        reasons.append(f"capacity_probe={supported}/{total}!=72/72")
    if exact_overlap:
        reasons.append(f"training_representation_overlap={len(exact_overlap)}")
    if knot_overlap:
        reasons.append(f"training_knot_overlap={len(knot_overlap)}")
    if not metadata.get("solver_version"):
        reasons.append("missing_solver_version")

    report = {
        "schema": ELIGIBILITY_SCHEMA,
        "created_at_utc": _utc_now(),
        "scientist": str(metadata["scientist"]),
        "architecture": str(metadata.get("architecture", metadata["scientist"])),
        "solver_version": metadata.get("solver_version"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "max_strands": int(metadata.get("max_strands", 0)),
        "cyclic_band_generators": bool(metadata.get("cyclic_band_generators", False)),
        "q_gate_passed": bool(metadata.get("q_gate_passed", False)),
        "training_strand_counts": {
            str(key): value for key, value in sorted(training_counts.items())
        },
        "retention_by_strand": {
            str(key): value for key, value in sorted(retention_by_strand.items())
        },
        "native_high_strand_successes": int(metadata.get("native_high_strand_successes", 0)),
        "capacity_probe": {"supported": supported, "total": total},
        "panel": str(panel.resolve()),
        "panel_sha256": sha256_file(panel),
        "panel_rows": len(rows),
        "training_representation_overlap": exact_overlap,
        "training_knot_overlap": knot_overlap,
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
        "selection_metrics": dict(metadata.get("selection_metrics", {})),
        "source_reports": dict(metadata.get("source_reports", {})),
    }
    report["report_sha256"] = _json_digest(report)
    _atomic_json(output, report)
    return report


def _selection_rank(report: dict[str, Any]) -> tuple[Any, ...]:
    metrics = report.get("selection_metrics", {})
    retention = min(float(value) for value in report["retention_by_strand"].values())
    training = sum(int(value) for value in report["training_strand_counts"].values())
    return (
        -int(metrics.get("strict_high_strand_successes", report["native_high_strand_successes"])),
        -retention,
        -training,
        float(metrics.get("capped_l1000", float("inf"))),
        int(metrics.get("network_evaluations", 2**63 - 1)),
        str(report["scientist"]),
    )


def select_two_checkpoint_systems(reports: Iterable[Path], output: Path) -> dict[str, Any]:
    """Select at most two systems using only preregistered non-DKT metrics.

    When both alphabets have eligible systems, one ordinary Artin and one B*
    system are selected.  This makes the extra cyclic-band freedom an explicit
    system-level comparison rather than an unreported confound.
    """

    loaded = [json.loads(path.read_text()) for path in reports]
    eligible = sorted((row for row in loaded if row.get("eligible")), key=_selection_rank)
    ordinary = [row for row in eligible if not row["cyclic_band_generators"]]
    cyclic = [row for row in eligible if row["cyclic_band_generators"]]
    if ordinary and cyclic:
        selected = [ordinary[0], cyclic[0]]
        policy = "best-ordinary-plus-best-cyclic-band"
    else:
        selected = eligible[:2]
        policy = "best-two-independent-q-metrics"
    selected.sort(key=lambda row: str(row["scientist"]))
    result = {
        "schema": SELECTION_SCHEMA,
        "created_at_utc": _utc_now(),
        "status": "selected" if len(selected) == 2 else "blocked",
        "selection_policy": policy,
        "outcome_blinding": "DKT72 search outcomes were not read for eligibility or selection",
        "eligible_candidates": [row["scientist"] for row in eligible],
        "selected": [
            {
                "scientist": row["scientist"],
                "architecture": row["architecture"],
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "eligibility_report_sha256": row["report_sha256"],
                "max_strands": row["max_strands"],
                "cyclic_band_generators": row["cyclic_band_generators"],
            }
            for row in selected
        ],
        "blocking_reason": None
        if len(selected) == 2
        else f"only_{len(selected)}_eligible_candidates",
    }
    result["selection_sha256"] = _json_digest(result)
    _atomic_json(output, result)
    return result


def freeze_protocol(
    *,
    panel: Path,
    selection: Path,
    output: Path,
    simulations: int,
    attempts: int,
    action_horizon: int,
    seed: int,
) -> dict[str, Any]:
    """Freeze the complete no-learning evaluation protocol before outcomes."""

    chosen = json.loads(selection.read_text())
    rows = _panel_rows(panel)
    if chosen.get("status") != "selected" or len(chosen.get("selected", [])) != 2:
        raise ValueError("exactly two eligible checkpoint systems must be selected first")
    if len(rows) != EXPECTED_PANEL_SIZE:
        raise ValueError("DKT72-v2 requires the complete 72-row panel")
    if min(simulations, attempts, action_horizon) < 1:
        raise ValueError("compute dose, attempts, and horizon must be positive")
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "created_at_utc": _utc_now(),
        "status": "frozen",
        "panel": str(panel.resolve()),
        "panel_sha256": sha256_file(panel),
        "panel_rows": len(rows),
        "selection": str(selection.resolve()),
        "selection_sha256": sha256_file(selection),
        "scientists": chosen["selected"],
        "objective": "L1000",
        "simulations": int(simulations),
        "attempts_per_representation": int(attempts),
        "action_horizon": int(action_horizon),
        "base_seed": int(seed),
        "paired_representation_attempt_seeds": True,
        "root_noise": True,
        "learning": False,
        "primary_denominator": 72,
        "supported_search_failure_cap_l1000": 20_128,
        "capacity_exception_policy": "invalidates eligibility; never becomes a search failure",
        "outcome_classes": [
            "strict_current_upper_bound_improvement",
            "verified_non_improving_solution",
            "supported_search_failure",
            "hard_timeout",
            "unsupported_capacity",
            "invalid_witness",
        ],
        "certification": {
            "trigger": "every strict current upper-bound improvement",
            "witness_replay_required": True,
            "independent_lower_bound_check_required": True,
            "exact_label_requires_matching_independent_lower_bound": True,
            "evidence_inventory_ingest_required": True,
        },
    }
    protocol["protocol_sha256"] = _json_digest(protocol)
    _atomic_json(output, protocol)
    return protocol


def preflight(protocol_path: Path, output: Path) -> dict[str, Any]:
    """Recheck immutable inputs immediately before any evaluator starts."""

    protocol = json.loads(protocol_path.read_text())
    failures: list[str] = []
    panel = Path(protocol["panel"])
    selection = Path(protocol["selection"])
    if sha256_file(panel) != protocol["panel_sha256"]:
        failures.append("panel_hash_changed")
    if sha256_file(selection) != protocol["selection_sha256"]:
        failures.append("selection_hash_changed")
    if len(_panel_rows(panel)) != EXPECTED_PANEL_SIZE:
        failures.append("panel_is_not_72_rows")
    for scientist in protocol.get("scientists", []):
        checkpoint = Path(scientist["checkpoint"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != scientist["checkpoint_sha256"]:
            failures.append(f"checkpoint_changed:{scientist['scientist']}")
        if int(scientist.get("max_strands", 0)) < 12:
            failures.append(f"capacity_below_12:{scientist['scientist']}")
    if len(protocol.get("scientists", [])) != 2:
        failures.append("scientist_count_is_not_2")
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "created_at_utc": _utc_now(),
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "status": "eligible-to-launch" if not failures else "blocked",
        "failures": failures,
    }
    _atomic_json(output, report)
    return report


def certify_improvement(
    *,
    witness_path: Path,
    inventory_root: Path,
    certificate_root: Path,
    knot_name: str,
    representation_id: str,
    scientist: str,
    previous_upper_bound: int,
    declared_lower_bound: int,
    solver_metadata: dict[str, Any],
    search_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Replay, record, and certify one strict DKT72-v2 improvement."""

    payload = json.loads(witness_path.read_text())
    witness = UnknotWitness.from_dict(payload.get("witness", payload))
    witness.verify()
    if witness.crossing_changes >= int(previous_upper_bound):
        raise ValueError("candidate is not a strict current upper-bound improvement")
    inventory = EvidenceInventory(inventory_root)
    evidence_id = inventory.record(
        witness=witness,
        knot_name=knot_name,
        representation_id=representation_id,
        scientist=scientist,
        sequence_name="DKT72-PD-v2",
        challenge_id=representation_id,
        previous_upper_bound=previous_upper_bound,
        solver_metadata=solver_metadata,
        search_parameters=search_parameters,
    )
    certificate = CertificationEngine(certificate_root).certify(
        evidence_id=evidence_id,
        knot_name=knot_name,
        root=witness.start,
        witness=witness,
        declared_lower_bound=declared_lower_bound,
    )
    return {"evidence_id": evidence_id, "certificate": certificate}
