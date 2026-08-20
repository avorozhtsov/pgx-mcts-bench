#!/usr/bin/env python3
"""Build the registered Q60 -> Q104 bridge curriculum.

The frozen q40-2 bank is input only.  Its ten 6+ strand rows are removed, its
thirty lower-strand rows are preserved byte-for-byte as JSON objects, and
fourteen deliberately easy Markov-stabilized representations are interleaved
in the registered row order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

TARGET_STRANDS = (5, 6, 5, 7, 5, 7, 5, 6, 5, 8, 6, 7, 6, 7)
EXPECTED_SOURCE_IDENTITIES = (
    "3_1",
    "4_1",
    "5_1",
    "6_2",
    "8_10",
    "8_16",
    "8_17",
    "8_18",
    "8_2",
    "8_20",
    "8_21",
    "8_7",
    "8_9",
    "10_104",
)


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


def _source_rows(q_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for bank_name in ("q20.json", "q40-1.json"):
        bank = json.loads((q_root / bank_name).read_text())
        for index, source in enumerate(bank["rows"]):
            identity = str(source["name"])
            if identity in result:
                continue
            row = deepcopy(source)
            row["bridge_source_bank"] = bank_name
            row["bridge_source_row_index"] = index
            result[identity] = row
    missing = set(EXPECTED_SOURCE_IDENTITIES) - set(result)
    if missing:
        raise ValueError(f"missing registered simple bridge sources: {sorted(missing)}")
    return result


def markov_stabilize(
    source: Mapping[str, Any], target_strands: int, *, seed: int, ordinal: int
) -> dict[str, Any]:
    source_strands = int(source["strands"])
    if target_strands <= source_strands:
        raise ValueError("bridge target must have more strands than its source")
    word = [int(value) for value in source["word"]]
    letters = []
    for generator in range(source_strands, target_strands):
        token = f"{seed}:{ordinal}:{source['name']}:{target_strands}:{generator}"
        sign = 1 if int(hashlib.sha256(token.encode()).hexdigest(), 16) & 1 else -1
        letters.append(sign * generator)
    word.extend(letters)
    upper = int(source["certified_unknotting_upper_bound"])
    score = float(10 * target_strands + 5 * upper + len(word))
    identity = str(source["name"])
    row = deepcopy(dict(source))
    row.update(
        {
            "id": f"{identity}::q44-simple-bridge-b{source_strands}-to-b{target_strands}-v1",
            "name": identity,
            "word": word,
            "strands": target_strands,
            "crossings": len(word),
            "presentation_crossings": len(word),
            "acs": score,
            "acs10": score,
            "acs5": float(5 * target_strands + 5 * upper + len(word)),
            "cheap_score": score,
            "capacity_bridge": True,
            "capacity_bridge_source_id": str(source["id"]),
            "capacity_bridge_source_strands": source_strands,
            "dataset_origin": "q44-2-updated-simple-strand-bridge",
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

    source_word = tuple(int(value) for value in source["word"])
    bridge_word = tuple(int(value) for value in bridge["word"])
    source_strands = int(source["strands"])
    bridge_strands = int(bridge["strands"])
    if num_components(source_word, source_strands) != 1:
        raise ValueError(f"bridge source is not a knot: {source['id']}")
    if num_components(bridge_word, bridge_strands) != 1:
        raise ValueError(f"bridge target is not a knot: {bridge['id']}")
    if alexander_polynomial(source_word, source_strands) != alexander_polynomial(
        bridge_word, bridge_strands
    ):
        raise ValueError(f"Alexander mismatch after Markov stabilization: {bridge['id']}")
    if jones_polynomial(source_word, source_strands) != jones_polynomial(
        bridge_word, bridge_strands
    ):
        raise ValueError(f"Jones mismatch after Markov stabilization: {bridge['id']}")


def build(q_root: Path, registration: Path, output_dir: Path, *, seed: int) -> dict[str, Any]:
    q20_path = q_root / "q20.json"
    q40_1_path = q_root / "q40-1.json"
    frozen_path = q_root / "q40-2.json"
    q20 = json.loads(q20_path.read_text())
    q40_1 = json.loads(q40_1_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    frozen_rows = list(frozen["rows"])
    dropped = [deepcopy(row) for row in frozen_rows if int(row["strands"]) >= 6]
    preserved = [deepcopy(row) for row in frozen_rows if int(row["strands"]) < 6]
    if len(dropped) != 10 or len(preserved) != 30:
        raise ValueError(
            f"expected frozen q40-2 split 10 high/30 preserved, got {len(dropped)}/{len(preserved)}"
        )

    registration_payload = json.loads(registration.read_text())
    exclusion = registration_payload["sources"]["dkt72_exclusion_panel"]
    dkt_identities = {str(value) for value in exclusion["q4000_identity_overlap_excluded"]}
    dkt_representations = {
        str(value) for value in exclusion["q4000_representation_overlap_excluded"]
    }
    sources = _source_rows(q_root)
    bridges = []
    for ordinal, (identity, target) in enumerate(
        zip(EXPECTED_SOURCE_IDENTITIES, TARGET_STRANDS, strict=True), start=1
    ):
        if identity in dkt_identities:
            raise ValueError(f"simple bridge identity overlaps DKT72: {identity}")
        source = sources[identity]
        bridge = markov_stabilize(source, target, seed=seed, ordinal=ordinal)
        if bridge["source_representation_id"] in dkt_representations:
            raise ValueError(f"simple bridge source representation overlaps DKT72: {identity}")
        if bridge["representation_id"] in dkt_representations:
            raise ValueError(f"simple bridge representation overlaps DKT72: {identity}")
        _audit_bridge(source, bridge)
        bridges.append(bridge)

    # The first seven preserved tasks each follow two bridge tasks.  Remaining
    # preserved tasks follow contiguously.  This is the user's registered
    # bridge/base schedule, not an ACS-sorted bank.
    scheduled = []
    for index in range(7):
        scheduled.extend(bridges[2 * index : 2 * index + 2])
        scheduled.append(preserved[index])
    scheduled.extend(preserved[7:])
    if len(scheduled) != 44 or len({str(row["id"]) for row in scheduled}) != 44:
        raise ValueError("Q44 schedule must contain exactly 44 unique representation IDs")

    strand_counts = {
        str(strands): sum(int(row["strands"]) == strands for row in scheduled)
        for strands in sorted({int(row["strands"]) for row in scheduled})
    }
    bank = {
        "schema": "q4000-training-group-v2",
        "name": "q44-2-updated",
        "size": 44,
        "cumulative_representations": 104,
        "selection": "registered row order; scheduled-no-sharing; no ACS reordering",
        "schedule": (
            "two simple strand bridges then one preserved base row for seven cycles; "
            "remaining 23 preserved base rows"
        ),
        "skip_policy": {
            "allowed_reasons": ["capacity", "budget_exhausted", "objective_plateau"],
            "fraction": 0.05,
            "maximum_skips": 2,
            "retained_in_denominators": True,
        },
        "strand_counts_exact": strand_counts,
        "strand_bridge_counts": {"5": 5, "6": 4, "7": 4, "8": 1},
        "strand_quotas": {
            "simple_strands_5_required": 5,
            "simple_strands_6_required": 4,
            "simple_strands_7_required": 4,
            "simple_strands_8_required": 1,
            "strands_ge_9_required": 0,
        },
        "rows": scheduled,
    }
    prior = {
        "schema": "q4000-cumulative-prior-v2",
        "name": "q60-prior-for-q44-2-updated",
        "size": 60,
        "rows": [*q20["rows"], *q40_1["rows"]],
    }
    audit = {
        "schema": "q44-2-updated-audit-v1",
        "status": "passed",
        "seed": seed,
        "name": "q44-2-updated",
        "cumulative_endpoint": "Q104",
        "source_files": {
            "q20.json": file_sha256(q20_path),
            "q40-1.json": file_sha256(q40_1_path),
            "q40-2.json": file_sha256(frozen_path),
            "mastery_v3_registration": file_sha256(registration),
        },
        "frozen_q40_2_unchanged": True,
        "dropped_complex_rows": [str(row["id"]) for row in dropped],
        "dropped_complex_strand_counts": {
            str(strands): sum(int(row["strands"]) == strands for row in dropped)
            for strands in sorted({int(row["strands"]) for row in dropped})
        },
        "preserved_rows": len(preserved),
        "preserved_rows_sha256": canonical_sha256(preserved),
        "bridge_rows": len(bridges),
        "bridge_rows_sha256": canonical_sha256(bridges),
        "bridge_source_identities": list(EXPECTED_SOURCE_IDENTITIES),
        "dkt72_identity_intersection": sorted(
            {str(row["name"]) for row in bridges} & dkt_identities
        ),
        "dkt72_representation_intersection": sorted(
            {str(row["representation_id"]) for row in bridges} & dkt_representations
        ),
        "bank_rows_sha256": canonical_sha256(scheduled),
        "bank_row_order": [str(row["id"]) for row in scheduled],
        "checks": {
            "all_bridge_markov_invariants_match": True,
            "all_bridge_component_counts_are_one": True,
            "all_original_strands_ge_6_removed": True,
            "all_thirty_lower_strand_rows_preserved": True,
            "scheduled_no_sharing_required": True,
        },
    }
    atomic_json(output_dir / "q44-2-updated.json", bank)
    atomic_json(output_dir / "prior-q60-for-q44-2-updated.json", prior)
    audit["outputs"] = {
        "bank": file_sha256(output_dir / "q44-2-updated.json"),
        "prior": file_sha256(output_dir / "prior-q60-for-q44-2-updated.json"),
    }
    atomic_json(output_dir / "q44-2-updated-audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081902)
    args = parser.parse_args()
    audit = build(args.q_root, args.registration, args.output_dir, seed=args.seed)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
