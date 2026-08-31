#!/usr/bin/env python3
"""Build the isolated Q254 bank while Q204 writers remain active."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_q50_1_updated as q50_1  # noqa: E402
import build_q50_2_updated as q50_2  # noqa: E402

BRIDGE_TARGET_ORDER = (11, 12, 9, 10, 11, 12, 9, 10, 11, 12)


def _ordered(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [deepcopy(dict(row)) for row in rows]
    selected.sort(
        key=lambda row: (
            float(row.get("acs10", row["acs"])),
            int(row["strands"]),
            int(row["presentation_crossings"]),
            str(row["id"]),
        )
    )
    return selected


def _simple_sources(
    q_root: Path,
    q44_bank: Path,
    excluded_names: set[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in (q_root / "q20.json", q_root / "q40-1.json", q44_bank):
        for row in json.loads(path.read_text())["rows"]:
            if (
                int(row["strands"]) <= 4
                and str(row["name"]) not in excluded_names
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
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        name = str(row["name"])
        if name in seen:
            continue
        seen.add(name)
        selected.append(row)
        if len(selected) == 10:
            return selected
    raise ValueError(f"could select only {len(selected)} new simple bridge sources")


def _markov_bridge(
    source: Mapping[str, Any], target: int, *, seed: int, ordinal: int
) -> dict[str, Any]:
    source_strands = int(source["strands"])
    if target <= source_strands:
        raise ValueError("simple bridge target must exceed source strands")
    letters: list[int] = []
    for generator in range(source_strands, target):
        token = f"q50-3:{seed}:{ordinal}:{source['name']}:{target}:{generator}"
        sign = 1 if int(hashlib.sha256(token.encode()).hexdigest(), 16) & 1 else -1
        letters.append(sign * generator)
    bridge = deepcopy(dict(source))
    word = [int(value) for value in source["word"]] + letters
    upper = int(source["certified_unknotting_upper_bound"])
    score = float(10 * target + 5 * upper + len(word))
    bridge.update(
        {
            "id": (
                f"{source['name']}::q50-3-simple-bridge-"
                f"b{source_strands}-to-b{target}-v1"
            ),
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
            "dataset_origin": "q50-3-updated-mixed-low-and-simple-strand-bridge",
            "difficulty_quartile": 0,
            "markov_stabilization_letters": letters,
            "native_presentation": False,
            "representation_variant": "deterministic-simple-markov-stabilization-v1",
            "selection_queue": "registered-schedule",
            "source_representation_id": q50_1.representation_id(source),
            "verification": {
                "certificate_steps": len(letters),
                "method": "replayable-markov-stabilization-certificate-v1",
                "preserves_knot_identity": True,
                "source_components": 1,
                "target_components": 1,
            },
        }
    )
    bridge["representation_id"] = q50_1.representation_id(bridge)
    return bridge


def build(
    q_root: Path,
    q44_bank: Path,
    registration: Path,
    policy: Path,
    predecessor_protocol: Path,
    output_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(policy.read_text())
    q200_path = q_root / "q200-1.json"
    if q50_1.file_sha256(q200_path) != payload["source_pool"]["byte_sha256"]:
        raise ValueError("frozen Q200-1 source hash mismatch")

    predecessor = payload["predecessor"]
    bank_path = predecessor_protocol / "q50-2-updated.json"
    prior_path = predecessor_protocol / "prior-q154-for-q50-2-updated.json"
    if q50_1.file_sha256(bank_path) != predecessor["bank_byte_sha256"]:
        raise ValueError("frozen Q50-2 predecessor bank hash mismatch")
    if q50_1.file_sha256(prior_path) != predecessor["prior_bank_byte_sha256"]:
        raise ValueError("frozen Q154 predecessor prior hash mismatch")

    q200_rows = json.loads(q200_path.read_text())["rows"]
    predecessor_rows = json.loads(bank_path.read_text())["rows"]
    prior_rows = json.loads(prior_path.read_text())["rows"]
    cumulative_rows = [*prior_rows, *predecessor_rows]
    used_ids = {str(row["id"]) for row in cumulative_rows}
    used_names = {str(row["name"]) for row in cumulative_rows}
    used_representations = {q50_1.representation_id(row) for row in cumulative_rows}

    registration_payload = json.loads(registration.read_text())
    exclusion = registration_payload["sources"]["dkt72_exclusion_panel"]
    dkt_names = set(exclusion["q4000_identity_overlap_excluded"])
    dkt_representations = set(exclusion["q4000_representation_overlap_excluded"])

    available = [
        row
        for row in q200_rows
        if str(row["id"]) not in used_ids
        and str(row["name"]) not in used_names
        and str(row["name"]) not in dkt_names
        and q50_1.representation_id(row) not in used_representations
        and q50_1.representation_id(row) not in dkt_representations
    ]
    easy_rows = _ordered(row for row in available if int(row["strands"]) <= 4)[:32]
    medium_rows = _ordered(
        row for row in available if 5 <= int(row["strands"]) <= 7
    )[:8]
    if len(easy_rows) != 32 or len(medium_rows) != 8:
        raise ValueError("insufficient unused low-strand rows for Q50-3")

    prior_bridge_names = {
        str(row["name"]) for row in cumulative_rows if row.get("capacity_bridge")
    }
    sources = _simple_sources(
        q_root,
        q44_bank,
        prior_bridge_names
        | dkt_names
        | {str(row["name"]) for row in easy_rows + medium_rows},
    )
    bridges: list[dict[str, Any]] = []
    for ordinal, (source, target) in enumerate(
        zip(sources, BRIDGE_TARGET_ORDER, strict=True), start=1
    ):
        bridge = _markov_bridge(source, target, seed=seed, ordinal=ordinal)
        representation = q50_1.representation_id(bridge)
        if representation in used_representations or representation in dkt_representations:
            raise ValueError(f"Q50-3 bridge representation is not new: {bridge['id']}")
        q50_1._audit_bridge(source, bridge)
        bridges.append(bridge)

    scheduled = q50_2.schedule_rows(easy_rows, medium_rows, bridges)
    identities = {str(row["id"]) for row in scheduled}
    representations = {q50_1.representation_id(row) for row in scheduled}
    if len(scheduled) != 50 or len(identities) != 50 or len(representations) != 50:
        raise ValueError("Q50-3 schedule must contain 50 unique identities and representations")
    bridge_positions = [
        index for index, row in enumerate(scheduled, start=1) if row.get("capacity_bridge")
    ]
    expected_positions = list(range(5, 51, 5))
    if bridge_positions != expected_positions:
        raise ValueError("Q50-3 bridges are not evenly interleaved")

    bank = {
        "schema": "q4000-training-group-v2",
        "name": "q50-3-updated",
        "size": 50,
        "cumulative_representations": 254,
        "selection": "registered exact row order; scheduled-no-sharing; no ACS reordering",
        "schedule": "ten cycles of four low-strand rows then one simple high-strand bridge",
        "bridge_target_order": list(BRIDGE_TARGET_ORDER),
        "rows": scheduled,
    }
    prior = {
        "schema": "q4000-cumulative-prior-v2",
        "name": "q204-prior-for-q50-3-updated",
        "size": 204,
        "rows": cumulative_rows,
    }
    audit = {
        "schema": "q50-3-updated-audit-v1-cohort-scoped",
        "status": "passed",
        "seed": seed,
        "policy_sha256": q50_1.file_sha256(policy),
        "source_pool": str(q200_path),
        "source_pool_sha256": q50_1.file_sha256(q200_path),
        "predecessor_bank_sha256": q50_1.file_sha256(bank_path),
        "predecessor_prior_sha256": q50_1.file_sha256(prior_path),
        "low_strand_counts": dict(
            sorted(Counter(int(row["strands"]) for row in easy_rows + medium_rows).items())
        ),
        "bridge_counts": dict(sorted(Counter(int(row["strands"]) for row in bridges).items())),
        "bridge_target_order": list(BRIDGE_TARGET_ORDER),
        "bridge_positions_one_based": bridge_positions,
        "bank_row_order": [str(row["id"]) for row in scheduled],
        "checks": {
            "all_low_rows_have_strands_at_most_7": all(
                int(row["strands"]) <= 7 for row in easy_rows + medium_rows
            ),
            "exactly_8_medium_low_rows": len(medium_rows) == 8,
            "bridge_fraction_exactly_0_20": len(bridges) / len(scheduled) == 0.2,
            "bridges_evenly_interleaved": bridge_positions == expected_positions,
            "bridge_target_order_exact": tuple(int(row["strands"]) for row in bridges)
            == BRIDGE_TARGET_ORDER,
            "two_batch_bridge_balance_exact": Counter(
                q50_2.BRIDGE_TARGET_ORDER + BRIDGE_TARGET_ORDER
            )
            == Counter({9: 5, 10: 5, 11: 5, 12: 5}),
            "all_bridge_invariants_match": True,
            "no_predecessor_identity_reuse": not (identities & used_ids),
            "no_predecessor_representation_reuse": not (
                representations & used_representations
            ),
            "no_dkt72_overlap": not ({str(row["name"]) for row in scheduled} & dkt_names)
            and not (representations & dkt_representations),
            "scheduled_no_sharing_required": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    q50_1.atomic_json(output_dir / "q50-3-updated.json", bank)
    q50_1.atomic_json(output_dir / "prior-q204-for-q50-3-updated.json", prior)
    audit["outputs"] = {
        "bank_sha256": q50_1.file_sha256(output_dir / "q50-3-updated.json"),
        "prior_sha256": q50_1.file_sha256(
            output_dir / "prior-q204-for-q50-3-updated.json"
        ),
    }
    q50_1.atomic_json(output_dir / "q50-3-updated-audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--q44-bank", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--predecessor-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026082301)
    args = parser.parse_args()
    audit = build(
        args.q_root,
        args.q44_bank,
        args.registration,
        args.policy,
        args.predecessor_protocol,
        args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
