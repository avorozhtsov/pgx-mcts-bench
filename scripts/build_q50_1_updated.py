#!/usr/bin/env python3
"""Build balanced simple-strand Q50-1-updated after completed Q104."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

BRIDGE_MIX = {7: 2, 8: 2, 9: 2, 10: 2, 11: 2}
Q104_STAGE = "q44-2-updated-scheduled-no-sharing"


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


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _report_paths(q104_root: Path) -> list[Path]:
    paths = sorted(q104_root.glob(f"branches/*/{Q104_STAGE}/report.json"))
    if len(paths) != 10:
        raise ValueError(f"Q50 build requires 10 durable Q104 reports, found {len(paths)}")
    return paths


def _q104_rates(q104_root: Path, q44_bank: Path) -> tuple[dict[int, dict[str, float]], dict]:
    bank = json.loads(q44_bank.read_text())
    strands = {str(row["id"]): int(row["strands"]) for row in bank["rows"]}
    counts = {strand: {"L10": [0, 0], "L1000": [0, 0]} for strand in range(5, 9)}
    signatures = {}
    for path in _report_paths(q104_root):
        report = json.loads(path.read_text())
        signatures[str(path.relative_to(q104_root))] = file_sha256(path)
        if int(report.get("completed_rungs", 0)) != 44:
            raise ValueError(f"incomplete Q104 report: {path}")
        for event in report["events"]:
            selected = str(event["selected"])
            strand = strands[selected]
            if strand not in counts:
                continue
            scientist_rows = event["scientists"]
            if len(scientist_rows) != 1:
                raise ValueError(f"Q104 branch is not a single-scientist result: {path}")
            scientist = next(iter(scientist_rows.values()))
            for label, ratio in (("L10", "10.0"), ("L1000", "1000.0")):
                attempts = scientist["evaluation"][ratio]["attempts"]
                counts[strand][label][0] += sum(bool(row["solved"]) for row in attempts)
                counts[strand][label][1] += len(attempts)
    rates = {
        strand: {
            label: solved / attempts if attempts else 0.0
            for label, (solved, attempts) in rows.items()
        }
        for strand, rows in counts.items()
    }
    detail = {"counts": counts, "rates": rates, "report_sha256": signatures}
    return rates, detail


def _markov_bridge(
    source: Mapping[str, Any], target: int, *, seed: int, ordinal: int
) -> dict[str, Any]:
    source_strands = int(source["strands"])
    if target <= source_strands:
        raise ValueError("simple bridge target must exceed source strands")
    letters = []
    for generator in range(source_strands, target):
        token = f"q50:{seed}:{ordinal}:{source['name']}:{target}:{generator}"
        sign = 1 if int(hashlib.sha256(token.encode()).hexdigest(), 16) & 1 else -1
        letters.append(sign * generator)
    row = deepcopy(dict(source))
    word = [int(value) for value in source["word"]] + letters
    upper = int(source["certified_unknotting_upper_bound"])
    score = float(10 * target + 5 * upper + len(word))
    row.update(
        {
            "id": f"{source['name']}::q50-simple-bridge-b{source_strands}-to-b{target}-v1",
            "word": word,
            "strands": target,
            "crossings": len(word),
            "presentation_crossings": len(word),
            "acs": score,
            "acs10": score,
            "acs5": float(5 * target + 5 * upper + len(word)),
            "cheap_score": score,
            "capacity_bridge": True,
            "capacity_bridge_source_id": str(source["id"]),
            "capacity_bridge_source_strands": source_strands,
            "dataset_origin": "q50-1-updated-balanced-simple-strand-bridge",
            "difficulty_quartile": 0,
            "markov_stabilization_letters": letters,
            "native_presentation": False,
            "representation_variant": "deterministic-simple-markov-stabilization-v1",
            "selection_queue": "registered-schedule",
            "source_representation_id": representation_id(source),
            "verification": {
                "certificate_steps": len(letters),
                "method": "replayable-markov-stabilization-certificate-v1",
                "preserves_knot_identity": True,
                "source_components": 1,
                "target_components": 1,
            },
        }
    )
    row["representation_id"] = representation_id(row)
    return row


def _audit_bridge(source: Mapping[str, Any], bridge: Mapping[str, Any]) -> None:
    from rf_knots.invariants import alexander_polynomial, jones_polynomial
    from rf_knots.reference import num_components

    source_state = (tuple(source["word"]), int(source["strands"]))
    bridge_state = (tuple(bridge["word"]), int(bridge["strands"]))
    if num_components(*source_state) != 1 or num_components(*bridge_state) != 1:
        raise ValueError(f"non-knot bridge: {bridge['id']}")
    if alexander_polynomial(*source_state) != alexander_polynomial(*bridge_state):
        raise ValueError(f"Alexander mismatch: {bridge['id']}")
    if jones_polynomial(*source_state) != jones_polynomial(*bridge_state):
        raise ValueError(f"Jones mismatch: {bridge['id']}")


def _simple_sources(
    q_root: Path, q44_bank: Path, excluded_names: set[str], dkt_names: set[str]
) -> list[dict[str, Any]]:
    candidates = []
    for path in (q_root / "q20.json", q_root / "q40-1.json", q44_bank):
        for row in json.loads(path.read_text())["rows"]:
            if (
                int(row["strands"]) <= 4
                and str(row["name"]) not in excluded_names
                and str(row["name"]) not in dkt_names
                and int(row["certified_unknotting_upper_bound"]) <= 2
            ):
                candidates.append(deepcopy(row))
    candidates.sort(
        key=lambda row: (
            int(row["presentation_crossings"]),
            int(row["strands"]),
            str(row["name"]),
            str(row["id"]),
        )
    )
    unique = []
    seen = set()
    for row in candidates:
        if str(row["name"]) in seen:
            continue
        unique.append(row)
        seen.add(str(row["name"]))
        if len(unique) == 10:
            return unique
    raise ValueError(f"could select only {len(unique)} simple bridge sources")


def build(
    q_root: Path,
    q44_bank: Path,
    q104_root: Path,
    registration: Path,
    policy: Path,
    output_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    policy_payload = json.loads(policy.read_text())
    q200_path = q_root / "q200-1.json"
    q200 = json.loads(q200_path.read_text())
    expected_q200_hash = policy_payload["source_pool"]["byte_sha256"]
    if file_sha256(q200_path) != expected_q200_hash:
        raise ValueError("frozen Q200-1 source hash mismatch")
    low = [deepcopy(row) for row in q200["rows"] if int(row["strands"]) < 6]
    low.sort(
        key=lambda row: (
            float(row.get("acs10", row["acs"])),
            int(row["strands"]),
            int(row["presentation_crossings"]),
            str(row["id"]),
        )
    )
    base_rows = low[:40]
    _rates, outcome_detail = _q104_rates(q104_root, q44_bank)
    mix = dict(BRIDGE_MIX)
    targets = [strand for strand in sorted(mix) for _ in range(mix[strand])]

    registration_payload = json.loads(registration.read_text())
    exclusion = registration_payload["sources"]["dkt72_exclusion_panel"]
    dkt_names = set(exclusion["q4000_identity_overlap_excluded"])
    dkt_reps = set(exclusion["q4000_representation_overlap_excluded"])
    sources = _simple_sources(
        q_root,
        q44_bank,
        {str(row["name"]) for row in base_rows},
        dkt_names,
    )
    bridges = []
    for ordinal, (source, target) in enumerate(zip(sources, targets, strict=True), start=1):
        bridge = _markov_bridge(source, target, seed=seed, ordinal=ordinal)
        if bridge["representation_id"] in dkt_reps:
            raise ValueError(f"Q50 bridge representation overlaps DKT72: {bridge['id']}")
        _audit_bridge(source, bridge)
        bridges.append(bridge)

    scheduled = []
    for index, bridge in enumerate(bridges):
        scheduled.append(bridge)
        scheduled.extend(base_rows[4 * index : 4 * index + 4])
    if len(scheduled) != 50 or len({str(row["id"]) for row in scheduled}) != 50:
        raise ValueError("Q50 schedule must contain 50 unique representation IDs")
    q20 = json.loads((q_root / "q20.json").read_text())["rows"]
    q40 = json.loads((q_root / "q40-1.json").read_text())["rows"]
    q44 = json.loads(q44_bank.read_text())["rows"]
    prior_rows = [*q20, *q40, *q44]
    bank = {
        "schema": "q4000-training-group-v2",
        "name": "q50-1-updated",
        "size": 50,
        "cumulative_representations": 154,
        "selection": "registered row order; scheduled-no-sharing; no ACS reordering",
        "schedule": "ten cycles of one balanced simple bridge then four Q200-1 base rows",
        "bridge_mix": {str(key): value for key, value in mix.items()},
        "skip_policy": {
            "allowed_reasons": ["capacity", "budget_exhausted", "objective_plateau"],
            "fraction": 0.04,
            "maximum_skips": 2,
            "retained_in_denominators": True,
        },
        "rows": scheduled,
    }
    prior = {
        "schema": "q4000-cumulative-prior-v2",
        "name": "q104-prior-for-q50-1-updated",
        "size": 104,
        "rows": prior_rows,
    }
    audit = {
        "schema": "q50-1-updated-audit-v1",
        "status": "passed",
        "seed": seed,
        "policy_sha256": file_sha256(policy),
        "source_q200_1_sha256": file_sha256(q200_path),
        "q104_outcomes": outcome_detail,
        "bridge_mix": {str(key): value for key, value in mix.items()},
        "base_rows": len(base_rows),
        "base_strand_counts": dict(sorted(Counter(row["strands"] for row in base_rows).items())),
        "quarantined_original_high_rows": sum(int(row["strands"]) >= 6 for row in q200["rows"]),
        "bridge_rows": len(bridges),
        "bridge_source_identities": [str(row["name"]) for row in sources],
        "dkt72_identity_intersection": sorted({str(row["name"]) for row in scheduled} & dkt_names),
        "dkt72_representation_intersection": sorted(
            {representation_id(row) for row in scheduled} & dkt_reps
        ),
        "bank_rows_sha256": canonical_sha256(scheduled),
        "bank_row_order": [str(row["id"]) for row in scheduled],
        "checks": {
            "all_ten_q104_reports_complete": True,
            "all_bridge_invariants_match": True,
            "bridge_fraction_exactly_0_20": True,
            "simple_bridge_strands_exactly_7_through_11": (
                Counter(int(row["strands"]) for row in bridges) == Counter(BRIDGE_MIX)
            ),
            "no_original_complex_q200_rows": all(
                int(row["strands"]) < 6 for row in base_rows
            ),
            "maximum_scheduled_strands_is_11": max(
                int(row["strands"]) for row in scheduled
            ) == 11,
            "scheduled_no_sharing_required": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "q50-1-updated.json", bank)
    atomic_json(output_dir / "prior-q104-for-q50-1-updated.json", prior)
    audit["outputs"] = {
        "bank_sha256": file_sha256(output_dir / "q50-1-updated.json"),
        "prior_sha256": file_sha256(output_dir / "prior-q104-for-q50-1-updated.json"),
    }
    atomic_json(output_dir / "q50-1-updated-audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--q44-bank", type=Path, required=True)
    parser.add_argument("--q104-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081903)
    args = parser.parse_args()
    audit = build(
        args.q_root,
        args.q44_bank,
        args.q104_root,
        args.registration,
        args.policy,
        args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
