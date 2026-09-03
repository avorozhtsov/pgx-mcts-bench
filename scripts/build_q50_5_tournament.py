#!/usr/bin/env python3
"""Build the registered Q305-Q354 same-root tournament bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_q50_1_updated as q50_1  # noqa: E402
import build_q50_2_updated as q50_2  # noqa: E402
import build_q50_3_updated as q50_3  # noqa: E402

BRIDGE_TARGET_ORDER = (12, 11, 12, 10, 12, 11, 12, 11, 12, 12)


def _markov_bridge(
    source: Mapping[str, Any], target: int, *, seed: int, ordinal: int
) -> dict[str, Any]:
    source_strands = int(source["strands"])
    letters = []
    for generator in range(source_strands, target):
        token = f"q50-5:{seed}:{ordinal}:{source['name']}:{target}:{generator}"
        sign = 1 if int(hashlib.sha256(token.encode()).hexdigest(), 16) & 1 else -1
        letters.append(sign * generator)
    bridge = deepcopy(dict(source))
    word = [int(value) for value in source["word"]] + letters
    upper = int(source["certified_unknotting_upper_bound"])
    score = float(10 * target + 5 * upper + len(word))
    bridge.update(
        {
            "id": f"{source['name']}::q50-5-simple-bridge-b{source_strands}-to-b{target}-v1",
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
            "dataset_origin": "q50-5-tournament-mixed-low-and-simple-strand-bridge",
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
    output_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(policy.read_text())
    source_path = Path(payload["source_pool"]["bank"])
    predecessor_bank = Path(payload["predecessor"]["bank"])
    predecessor_prior = Path(payload["predecessor"]["prior_bank"])
    for path, expected in (
        (source_path, payload["source_pool"]["byte_sha256"]),
        (predecessor_bank, payload["predecessor"]["bank_byte_sha256"]),
        (predecessor_prior, payload["predecessor"]["prior_bank_byte_sha256"]),
    ):
        if q50_1.file_sha256(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    if source_path.name not in registration.read_text():
        raise ValueError("Q200 source is not registered")
    cumulative = [
        *json.loads(predecessor_prior.read_text())["rows"],
        *json.loads(predecessor_bank.read_text())["rows"],
    ]
    used_ids = {str(row["id"]) for row in cumulative}
    used_names = {str(row["name"]) for row in cumulative}
    used_representations = {q50_1.representation_id(row) for row in cumulative}
    exclusion = json.loads(registration.read_text())["sources"]["dkt72_exclusion_panel"]
    dkt_names = set(exclusion["q4000_identity_overlap_excluded"])
    dkt_representations = set(exclusion["q4000_representation_overlap_excluded"])
    available = [
        row
        for row in json.loads(source_path.read_text())["rows"]
        if str(row["id"]) not in used_ids
        and str(row["name"]) not in used_names
        and str(row["name"]) not in dkt_names
        and q50_1.representation_id(row) not in used_representations
        and q50_1.representation_id(row) not in dkt_representations
    ]
    easy = q50_3._ordered(row for row in available if int(row["strands"]) <= 4)[:32]
    medium = q50_3._ordered(row for row in available if 5 <= int(row["strands"]) <= 7)[:8]
    if len(easy) != 32 or len(medium) != 8:
        raise ValueError("insufficient unused low-strand rows for Q50-5")
    prior_bridge_names = {str(row["name"]) for row in cumulative if row.get("capacity_bridge")}
    sources = q50_3._simple_sources(
        q_root,
        q44_bank,
        prior_bridge_names | dkt_names | {str(row["name"]) for row in easy + medium},
    )
    bridges = []
    for ordinal, (source, target) in enumerate(zip(sources, BRIDGE_TARGET_ORDER, strict=True), 1):
        bridge = _markov_bridge(source, target, seed=seed, ordinal=ordinal)
        q50_1._audit_bridge(source, bridge)
        bridges.append(bridge)
    scheduled = q50_2.schedule_rows(easy, medium, bridges)
    identities = {str(row["id"]) for row in scheduled}
    representations = {q50_1.representation_id(row) for row in scheduled}
    positions = [index for index, row in enumerate(scheduled, 1) if row.get("capacity_bridge")]
    if len(identities) != 50 or len(representations) != 50:
        raise ValueError("Q50-5 rows are not unique")
    if positions != list(range(5, 51, 5)):
        raise ValueError("Q50-5 bridge schedule differs")
    if identities & used_ids or representations & used_representations:
        raise ValueError("Q50-5 overlaps its cumulative predecessor")
    bank = {
        "schema": "q4000-training-group-v2",
        "name": "q50-5-tournament",
        "size": 50,
        "first_cumulative_index": 305,
        "cumulative_representations": 354,
        "selection": "registered exact row order; same-root trajectory tournament",
        "schedule": "ten cycles of four low-strand rows then one simple high-strand bridge",
        "bridge_target_order": list(BRIDGE_TARGET_ORDER),
        "rows": scheduled,
    }
    prior = {
        "schema": "q4000-cumulative-prior-v2",
        "name": "q304-prior-for-q50-5-tournament",
        "size": 304,
        "rows": cumulative,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / "q50-5-tournament.json"
    prior_path = output_dir / "prior-q304-for-q50-5-tournament.json"
    q50_1.atomic_json(bank_path, bank)
    q50_1.atomic_json(prior_path, prior)
    prior_counts = Counter(
        int(row["strands"])
        for row in cumulative
        if row.get("capacity_bridge") and int(row["strands"]) in (9, 10, 11, 12)
    )
    combined_counts = prior_counts + Counter(BRIDGE_TARGET_ORDER)
    audit = {
        "schema": "q50-5-tournament-audit-v1-cohort-scoped",
        "status": "passed",
        "seed": seed,
        "policy_sha256": q50_1.file_sha256(policy),
        "source_pool_sha256": q50_1.file_sha256(source_path),
        "bank_row_order": [str(row["id"]) for row in scheduled],
        "bridge_positions_one_based": positions,
        "bridge_target_order": list(BRIDGE_TARGET_ORDER),
        "combined_bridge_counts": dict(sorted(combined_counts.items())),
        "checks": {
            "all_low_rows_have_strands_at_most_7": all(
                int(row["strands"]) <= 7 for row in easy + medium
            ),
            "bridge_fraction_exactly_0_20": len(bridges) / len(scheduled) == 0.2,
            "balanced_bridge_strands_9_to_12": max(combined_counts.values())
            - min(combined_counts.values())
            <= 1,
            "no_predecessor_identity_reuse": not (identities & used_ids),
            "no_predecessor_representation_reuse": not (representations & used_representations),
            "no_dkt72_overlap": not ({str(row["name"]) for row in scheduled} & dkt_names)
            and not (representations & dkt_representations),
            "ten_equal_budget_trajectories_required": True,
            "invalid_routes_unknown_not_negative": True,
        },
        "outputs": {
            "bank_sha256": q50_1.file_sha256(bank_path),
            "prior_sha256": q50_1.file_sha256(prior_path),
        },
    }
    failed = [name for name, passed in audit["checks"].items() if not passed]
    if failed:
        raise RuntimeError(f"Q50-5 audit failed: {failed}")
    q50_1.atomic_json(output_dir / "q50-5-tournament-audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--q44-bank", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202609030305)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.q_root,
                args.q44_bank,
                args.registration,
                args.policy,
                args.output_dir,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
