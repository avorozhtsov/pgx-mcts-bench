"""Deterministic, fail-closed registration for the mastery-v3 curriculum."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

CLEAN_Q500_GROUPS = ("q20", "q40-1", "q40-2", "q200-1", "q200-2")
REQUIRED_CAPACITY_STRANDS = tuple(range(6, 13))
PAIRED_SCREENING_SEEDS = (2026081701, 2026081702)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def representation_id(row: Mapping[str, Any]) -> str:
    payload = {"strands": int(row["strands"]), "word": [int(x) for x in row["word"]]}
    return f"braid:{canonical_sha256(payload)}"


def _identity(row: Mapping[str, Any]) -> str:
    return str(row.get("name") or row.get("id"))


def _load_q_rows(q_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = q_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows: list[dict[str, Any]] = []
    source_groups = []
    for index, group in enumerate(manifest.get("groups", [])):
        bank_path = q_root / str(group["bank"])
        observed = file_sha256(bank_path)
        expected = str(group["bank_sha256"])
        if observed != expected:
            raise ValueError(f"Q bank hash mismatch for {bank_path.name}: {observed} != {expected}")
        bank = json.loads(bank_path.read_text())
        if str(bank.get("name")) != str(group["name"]):
            raise ValueError(f"Q bank name mismatch for {bank_path.name}")
        if len(bank.get("rows", [])) != int(group["size"]):
            raise ValueError(f"Q bank size mismatch for {bank_path.name}")
        for source_index, source in enumerate(bank["rows"]):
            row = deepcopy(source)
            row["source_q_group"] = str(group["name"])
            row["source_q_group_index"] = index
            row["source_row_index"] = source_index
            row["representation_id"] = representation_id(row)
            rows.append(row)
        source_groups.append(
            {
                "name": group["name"],
                "bank": group["bank"],
                "bank_sha256": expected,
                "size": group["size"],
            }
        )
    return rows, {
        "manifest_sha256": file_sha256(manifest_path),
        "groups": source_groups,
        "total_rows": len(rows),
    }


def _load_dkt_exclusions(panel_path: Path) -> tuple[set[str], set[str], dict[str, Any]]:
    panel = json.loads(panel_path.read_text())
    instances = panel.get("instances", panel.get("rows", []))
    identities: set[str] = set()
    representations: set[str] = set()
    for item in instances:
        for key in ("source_id", "name", "knot_name", "canonical_name"):
            if item.get(key) is not None:
                identities.add(str(item[key]))
        payload = item.get("payload", item)
        if payload.get("strands") is not None and payload.get("word") is not None:
            representations.add(representation_id(payload))
        if item.get("instance_id") is not None:
            representations.add(str(item["instance_id"]))
    return (
        identities,
        representations,
        {
            "path": str(panel_path),
            "sha256": file_sha256(panel_path),
            "rows": len(instances),
        },
    )


def _stable_order(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[str, str]:
        token = f"{seed}:{_identity(row)}:{row['representation_id']}"
        return hashlib.sha256(token.encode()).hexdigest(), str(row["representation_id"])

    return sorted(rows, key=key)


def _markov_stabilize_to_eight(row: Mapping[str, Any], seed: int) -> dict[str, Any]:
    if int(row["strands"]) != 7:
        raise ValueError("8-strand synthesis requires a 7-strand parent")
    result = deepcopy(dict(row))
    parent_representation = str(row["representation_id"])
    bit = int(hashlib.sha256(f"{seed}:{parent_representation}".encode()).hexdigest(), 16) & 1
    generator = 7 if bit == 0 else -7
    result["strands"] = 8
    result["word"] = [int(x) for x in row["word"]] + [generator]
    result["native_presentation"] = False
    result["representation_variant"] = "deterministic-markov-stabilization-to-8"
    result["transformation"] = {
        "kind": "markov-stabilization",
        "parent_representation_id": parent_representation,
        "appended_generator": generator,
        "seed": seed,
        "preserves_knot_identity": True,
    }
    result["representation_id"] = representation_id(result)
    return result


def _take_unique_identities(
    rows: Iterable[dict[str, Any]], count: int, excluded: set[str]
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        identity = _identity(row)
        if identity in excluded:
            continue
        selected.append(deepcopy(row))
        excluded.add(identity)
        if len(selected) == count:
            return selected
    raise ValueError(f"could select only {len(selected)} of {count} identity-disjoint rows")


def _stage(name: str, rows: list[dict[str, Any]], *, role: str) -> dict[str, Any]:
    strands: dict[str, int] = {}
    for row in rows:
        key = str(int(row["strands"]))
        strands[key] = strands.get(key, 0) + 1
    return {
        "schema": "mastery-v3-curriculum-stage-v1",
        "name": name,
        "role": role,
        "rows": rows,
        "row_count": len(rows),
        "knot_identity_count": len({_identity(row) for row in rows}),
        "representation_count": len({str(row["representation_id"]) for row in rows}),
        "strand_counts": strands,
        "sha256": canonical_sha256(rows),
    }


def build_curriculum(
    q_root: Path,
    dkt_panel_path: Path,
    evidence_snapshot_path: Path,
    *,
    seed: int = 2026081700,
) -> dict[str, Any]:
    """Build identical, identity-disjoint training and screening stages for both v3 arms."""

    rows, q_source = _load_q_rows(q_root)
    dkt_identities, dkt_representations, dkt_source = _load_dkt_exclusions(dkt_panel_path)
    evidence_sha = file_sha256(evidence_snapshot_path)
    source_identity_overlap = sorted({_identity(row) for row in rows} & dkt_identities)
    source_representation_overlap = sorted(
        {str(row["representation_id"]) for row in rows} & dkt_representations
    )
    dkt_source["q4000_identity_overlap_excluded"] = source_identity_overlap
    dkt_source["q4000_representation_overlap_excluded"] = source_representation_overlap
    eligible = [
        row
        for row in rows
        if _identity(row) not in dkt_identities
        and str(row["representation_id"]) not in dkt_representations
    ]
    excluded = set(dkt_identities)

    clean = [row for row in eligible if row["source_q_group"] in CLEAN_Q500_GROUPS]
    simple_pool = sorted(
        (row for row in clean if int(row["strands"]) <= 4),
        key=lambda row: (len(row["word"]), _identity(row), row["representation_id"]),
    )
    adaptation = _take_unique_identities(simple_pool, 10, excluded)

    later = [row for row in eligible if row["source_q_group"] not in CLEAN_Q500_GROUPS]
    synthesized_eight = [
        _markov_stabilize_to_eight(row, seed)
        for row in _stable_order((row for row in later if int(row["strands"]) == 7), seed + 8)
    ]
    heavy = []
    for strands in REQUIRED_CAPACITY_STRANDS:
        pool = (
            synthesized_eight
            if strands == 8
            else [row for row in later if int(row["strands"]) == strands]
        )
        pool = sorted(
            pool,
            key=lambda row: (-len(row["word"]), _identity(row), row["representation_id"]),
        )
        heavy.extend(_take_unique_identities(pool, 4, excluded))

    screen_pool = [row for row in eligible if _identity(row) not in excluded]
    screening = []
    for strands in REQUIRED_CAPACITY_STRANDS:
        pool = (
            synthesized_eight
            if strands == 8
            else [row for row in screen_pool if int(row["strands"]) == strands]
        )
        row = _take_unique_identities(_stable_order(pool, seed + 100 + strands), 1, excluded)[0]
        screening.append(row)
    screening.extend(
        _take_unique_identities(
            _stable_order(
                (row for row in screen_pool if _identity(row) not in excluded), seed + 200
            ),
            20 - len(screening),
            excluded,
        )
    )
    screening = _stable_order(screening, seed + 300)

    stages = {
        "simple_adaptation": _stage(
            "simple-adaptation-10", adaptation, role="training: ten simple knot identities"
        ),
        "heavy_capacity": _stage(
            "heavy-capacity-28",
            heavy,
            role="training: four representations for each strand count 6 through 12",
        ),
        "screening": _stage(
            "frozen-screening-20", screening, role="evaluation-only: paired, frozen screening"
        ),
    }
    training_identities = {
        _identity(row)
        for key in ("simple_adaptation", "heavy_capacity")
        for row in stages[key]["rows"]
    }
    screening_identities = {_identity(row) for row in stages["screening"]["rows"]}
    if training_identities & screening_identities:
        raise AssertionError("training and screening knot identities overlap")
    if dkt_identities & (training_identities | screening_identities):
        raise AssertionError("DKT72 knot identities leaked into mastery-v3 curriculum")

    return {
        "schema": "mastery-v3-controlled-curriculum-v1",
        "seed": seed,
        "immutable": True,
        "sharing": "none",
        "candidates": ["cyclic-memory-deep-v3", "cyclic-graph-dual-v3"],
        "control": "high-combined-dual-v2-frozen",
        "implementation": {
            "architecture_module": "pgx_mcts_bench.mastery_v3",
            "checkpoint_preparation": "scripts/prepare_mastery_v3.py",
            "inference_module": "pgx_mcts_bench.gpu_inference",
            "candidate_encoders": {
                "cyclic-memory-deep-v3": "cyclic-memory-deep-v3",
                "cyclic-graph-dual-v3": "cyclic-graph-dual-v3",
            },
            "parent_scientist": "cyclic-memory-12",
            "migration_report_required": True,
            "persistent_inference_required_for_gpu_gate": True,
            "launch_authorized": {
                "stages": [
                    "migration",
                    "pretraining",
                    "proof-distillation",
                    "paired-screening-20",
                ],
                "full_240": False,
            },
        },
        "paired_screening_seeds": list(PAIRED_SCREENING_SEEDS),
        "sources": {
            "q4000": q_source,
            "dkt72_exclusion_panel": dkt_source,
            "evidence_snapshot": {
                "path": str(evidence_snapshot_path),
                "sha256": evidence_sha,
            },
        },
        "pretraining": {
            "kind": "equivalence-and-self-supervised",
            "implementation": "pgx_mcts_bench.mastery_v3_pretrain",
            "source": "pinned evidence snapshot plus registered Q representations",
            "deterministic_seed": seed + 1,
            "replayable_transformations_required": True,
            "live_evidence_reading": False,
        },
        "proof_distillation": {
            "kind": "certified-bounds-and-exact-witnesses",
            "implementation": "pgx_mcts_bench.mastery_v3_distill",
            "ratios": [10, 1000],
            "minimum_training_steps": 500,
            "samples_per_side": 4,
            "negative_label": "B < ratio * certified_lower_bound",
            "positive_label": "B >= ratio * replayed_crossing_changes + replayed_moves",
            "ambiguous_interval_masked": True,
            "operational_p_solve_is_separate": True,
            "screening_requires_passed_report": True,
        },
        "stages": stages,
        "promotion": {
            "only_one_candidate_advances_to_full_240": True,
            "full_240_launch_requires_audit_status": "passed",
            "screening_retention_minimum": 0.90,
            "retention_safety_floor_by_strand": 0.80,
            "capacity_exceptions_allowed": 0,
            "gpu_end_to_end_speedup_minimum": 5.0,
            "strict_or_genuine_improvement_required": True,
        },
        "overlap_audit": {
            "training_screening_identity_intersection": sorted(
                training_identities & screening_identities
            ),
            "dkt_training_identity_intersection": sorted(dkt_identities & training_identities),
            "dkt_screening_identity_intersection": sorted(dkt_identities & screening_identities),
        },
    }


def audit_promotion(curriculum: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Audit candidate screening reports and select at most one winner."""

    candidates = list(curriculum["candidates"])
    expected_keys = {
        (str(row["representation_id"]), int(seed))
        for row in curriculum["stages"]["screening"]["rows"]
        for seed in curriculum["paired_screening_seeds"]
    }
    audits: dict[str, Any] = {}
    eligible = []
    for candidate in candidates:
        arm = (report.get("arms") or {}).get(candidate) or {}
        observed_keys = {
            (str(item.get("representation_id")), int(item.get("seed", -1)))
            for item in arm.get("screening_items", [])
        }
        failures = []
        migration_error = arm.get("migration_max_abs_error")
        migration_tolerance = arm.get("migration_tolerance", 1e-6)
        if migration_error is None or float(migration_error) > float(migration_tolerance):
            failures.append("migration_tolerance")
        if observed_keys != expected_keys:
            failures.append("screening_key_set")
        retention = arm.get("screening_retention")
        if retention is None or float(retention) < 0.90:
            failures.append("screening_retention")
        stratified = {
            int(k): float(v)
            for k, v in (arm.get("retention_by_strand") or {}).items()
            if v is not None
        }
        if set(stratified) != set(REQUIRED_CAPACITY_STRANDS) or any(
            stratified.get(strands, -1.0) < 0.80 for strands in REQUIRED_CAPACITY_STRANDS
        ):
            failures.append("stratified_retention")
        if int(arm.get("capacity_exceptions", -1)) != 0:
            failures.append("capacity_exception")
        speedup = arm.get("gpu_end_to_end_speedup")
        if speedup is None or float(speedup) < 5.0:
            failures.append("gpu_speedup")
        strict = int(arm.get("strict_successes", 0))
        improvements = int(arm.get("genuine_improvements", 0))
        if strict + improvements < 1:
            failures.append("no_strict_evidence")
        evidence_rate = arm.get("strict_evidence_per_accelerator_second")
        if evidence_rate is None or float(evidence_rate) < 0:
            failures.append("evidence_rate")
        audits[candidate] = {
            "status": "passed" if not failures else "failed",
            "failures": failures,
            "screening_keys_expected": len(expected_keys),
            "screening_keys_observed": len(observed_keys),
        }
        if not failures:
            eligible.append(candidate)

    winner = None
    if eligible:

        def rank(candidate: str) -> tuple[float, float, float, float, str]:
            arm = report["arms"][candidate]
            return (
                float(arm.get("genuine_improvements", 0)),
                float(arm.get("strict_successes", 0)),
                -float(arm.get("capped_l1000", float("inf"))),
                float(arm.get("strict_evidence_per_accelerator_second", 0)),
                candidate,
            )

        winner = max(eligible, key=rank)
    return {
        "schema": "mastery-v3-promotion-audit-v1",
        "status": "passed" if winner is not None else "failed",
        "winner": winner,
        "advance_to_full_240": [winner] if winner is not None else [],
        "candidate_audits": audits,
        "common_screening_keys": len(expected_keys),
    }
