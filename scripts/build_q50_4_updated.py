#!/usr/bin/env python3
"""Build the isolated Q304 bank while Q254 writers remain active."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_q50_1_updated as q50_1  # noqa: E402
import build_q50_2_updated as q50_2  # noqa: E402
import build_q50_3_updated as q50_3  # noqa: E402

BRIDGE_TARGET_ORDER = q50_2.BRIDGE_TARGET_ORDER


def _markov_bridge(source: Mapping[str, Any], target: int, *, seed: int, ordinal: int) -> dict[str, Any]:
    source_strands = int(source["strands"])
    letters: list[int] = []
    for generator in range(source_strands, target):
        token = f"q50-4:{seed}:{ordinal}:{source['name']}:{target}:{generator}"
        sign = 1 if int(hashlib.sha256(token.encode()).hexdigest(), 16) & 1 else -1
        letters.append(sign * generator)
    bridge = deepcopy(dict(source))
    word = [int(value) for value in source["word"]] + letters
    upper = int(source["certified_unknotting_upper_bound"])
    score = float(10 * target + 5 * upper + len(word))
    bridge.update({
        "id": f"{source['name']}::q50-4-simple-bridge-b{source_strands}-to-b{target}-v1",
        "word": word, "strands": target, "crossings": len(word),
        "presentation_crossings": len(word), "acs": score, "acs10": score,
        "acs5": float(5 * target + 5 * upper + len(word)), "cheap_score": score,
        "capacity_bridge": True, "capacity_bridge_source_id": str(source["id"]),
        "capacity_bridge_source_strands": source_strands,
        "dataset_origin": "q50-4-updated-mixed-low-and-simple-strand-bridge",
        "difficulty_quartile": 0, "markov_stabilization_letters": letters,
        "native_presentation": False,
        "representation_variant": "deterministic-simple-markov-stabilization-v1",
        "selection_queue": "registered-schedule",
        "source_representation_id": q50_1.representation_id(source),
        "verification": {"certificate_steps": len(letters),
            "method": "replayable-markov-stabilization-certificate-v1",
            "preserves_knot_identity": True, "source_components": 1, "target_components": 1},
    })
    bridge["representation_id"] = q50_1.representation_id(bridge)
    return bridge


def build(q_root: Path, q44_bank: Path, registration: Path, policy: Path,
          predecessor_protocol: Path, output_dir: Path, *, seed: int) -> dict[str, Any]:
    payload = json.loads(policy.read_text())
    q200_path = Path(payload["source_pool"]["bank"])
    if q50_1.file_sha256(q200_path) != payload["source_pool"]["byte_sha256"]:
        raise ValueError("frozen rotated Q200 source hash mismatch")
    if q200_path.name not in registration.read_text():
        raise ValueError("rotated Q200 source is not registered")
    bank_path = predecessor_protocol / "q50-3-updated.json"
    prior_path = predecessor_protocol / "prior-q204-for-q50-3-updated.json"
    predecessor = payload["predecessor"]
    if q50_1.file_sha256(bank_path) != predecessor["bank_byte_sha256"]:
        raise ValueError("frozen Q50-3 predecessor bank hash mismatch")
    if q50_1.file_sha256(prior_path) != predecessor["prior_bank_byte_sha256"]:
        raise ValueError("frozen Q204 predecessor prior hash mismatch")
    predecessor_rows = json.loads(bank_path.read_text())["rows"]
    prior_rows = json.loads(prior_path.read_text())["rows"]
    cumulative_rows = [*prior_rows, *predecessor_rows]
    used_ids = {str(row["id"]) for row in cumulative_rows}
    used_names = {str(row["name"]) for row in cumulative_rows}
    used_representations = {q50_1.representation_id(row) for row in cumulative_rows}
    exclusion = json.loads(registration.read_text())["sources"]["dkt72_exclusion_panel"]
    dkt_names = set(exclusion["q4000_identity_overlap_excluded"])
    dkt_representations = set(exclusion["q4000_representation_overlap_excluded"])
    available = [row for row in json.loads(q200_path.read_text())["rows"]
        if str(row["id"]) not in used_ids and str(row["name"]) not in used_names
        and str(row["name"]) not in dkt_names
        and q50_1.representation_id(row) not in used_representations
        and q50_1.representation_id(row) not in dkt_representations]
    easy = q50_3._ordered(row for row in available if int(row["strands"]) <= 4)[:32]
    medium = q50_3._ordered(row for row in available if 5 <= int(row["strands"]) <= 7)[:8]
    if len(easy) != 32 or len(medium) != 8:
        raise ValueError("insufficient unused low-strand rows for Q50-4")
    prior_bridge_names = {str(row["name"]) for row in cumulative_rows if row.get("capacity_bridge")}
    sources = q50_3._simple_sources(q_root, q44_bank,
        prior_bridge_names | dkt_names | {str(row["name"]) for row in easy + medium})
    bridges = []
    for ordinal, (source, target) in enumerate(zip(sources, BRIDGE_TARGET_ORDER, strict=True), 1):
        bridge = _markov_bridge(source, target, seed=seed, ordinal=ordinal)
        representation = q50_1.representation_id(bridge)
        if representation in used_representations or representation in dkt_representations:
            raise ValueError(f"Q50-4 bridge representation is not new: {bridge['id']}")
        q50_1._audit_bridge(source, bridge)
        bridges.append(bridge)
    scheduled = q50_2.schedule_rows(easy, medium, bridges)
    identities = {str(row["id"]) for row in scheduled}
    representations = {q50_1.representation_id(row) for row in scheduled}
    positions = [i for i, row in enumerate(scheduled, 1) if row.get("capacity_bridge")]
    if len(identities) != 50 or len(representations) != 50 or positions != list(range(5, 51, 5)):
        raise ValueError("Q50-4 exact unique schedule audit failed")
    bank = {"schema": "q4000-training-group-v2", "name": "q50-4-updated", "size": 50,
        "cumulative_representations": 304,
        "selection": "registered exact row order; scheduled-no-sharing; no ACS reordering",
        "schedule": "ten cycles of four low-strand rows then one simple high-strand bridge",
        "bridge_target_order": list(BRIDGE_TARGET_ORDER), "rows": scheduled}
    prior = {"schema": "q4000-cumulative-prior-v2", "name": "q254-prior-for-q50-4-updated",
        "size": 254, "rows": cumulative_rows}
    audit = {"schema": "q50-4-updated-audit-v1-cohort-scoped", "status": "passed",
        "seed": seed, "policy_sha256": q50_1.file_sha256(policy),
        "source_pool_sha256": q50_1.file_sha256(q200_path),
        "bridge_counts": dict(sorted(Counter(int(row["strands"]) for row in bridges).items())),
        "bridge_target_order": list(BRIDGE_TARGET_ORDER), "bridge_positions_one_based": positions,
        "bank_row_order": [str(row["id"]) for row in scheduled],
        "checks": {"all_low_rows_have_strands_at_most_7": all(int(r["strands"]) <= 7 for r in easy + medium),
            "bridge_fraction_exactly_0_20": len(bridges) / len(scheduled) == 0.2,
            "q50_3_plus_q50_4_bridge_balance_exact": Counter(q50_3.BRIDGE_TARGET_ORDER + BRIDGE_TARGET_ORDER)
                == Counter({9: 5, 10: 5, 11: 5, 12: 5}),
            "no_predecessor_identity_reuse": not (identities & used_ids),
            "no_predecessor_representation_reuse": not (representations & used_representations),
            "no_dkt72_overlap": not ({str(r["name"]) for r in scheduled} & dkt_names)
                and not (representations & dkt_representations),
            "scheduled_no_sharing_required": True}}
    output_dir.mkdir(parents=True, exist_ok=True)
    q50_1.atomic_json(output_dir / "q50-4-updated.json", bank)
    q50_1.atomic_json(output_dir / "prior-q254-for-q50-4-updated.json", prior)
    audit["outputs"] = {"bank_sha256": q50_1.file_sha256(output_dir / "q50-4-updated.json"),
        "prior_sha256": q50_1.file_sha256(output_dir / "prior-q254-for-q50-4-updated.json")}
    q50_1.atomic_json(output_dir / "q50-4-updated-audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True); parser.add_argument("--q44-bank", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--predecessor-protocol", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026082401)
    args = parser.parse_args()
    print(json.dumps(build(args.q_root, args.q44_bank, args.registration, args.policy,
        args.predecessor_protocol, args.output_dir, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__": main()
