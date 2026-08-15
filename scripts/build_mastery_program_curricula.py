#!/usr/bin/env python3
"""Build two long bound-improvement curricula and a verified KnotInfo reservoir."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_q_rows(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for group in manifest["groups"]:
        bank_path = root / group["bank"]
        if sha256(bank_path) != group["bank_sha256"]:
            raise ValueError(f"Q bank hash mismatch: {bank_path}")
        bank = json.loads(bank_path.read_text())
        for row in bank["rows"]:
            lower = int(row["certified_unknotting_lower_bound"])
            upper = int(row["certified_unknotting_upper_bound"])
            if upper <= lower:
                continue
            rows.append(
                {
                    **row,
                    "_source_bank": group["bank"],
                    "_source_bank_sha256": group["bank_sha256"],
                    "_source_manifest": str(manifest_path),
                    "_source_manifest_sha256": sha256(manifest_path),
                }
            )
    # A knot may occur in several capacity bridges.  Retain the simplest exact
    # representation for the general ranking while preserving every strand
    # count as an option for the high-strand curriculum.
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["name"]), int(row["strands"]))
        prior = unique.get(key)
        rank = (int(row["acs10"]), int(row["acs5"]), str(row["id"]))
        if prior is None or rank < (
            int(prior["acs10"]),
            int(prior["acs5"]),
            str(prior["id"]),
        ):
            unique[key] = row
    return list(unique.values())


def choose_high_sequence(rows: list[dict[str, Any]], length: int) -> list[dict[str, Any]]:
    by_knot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_knot[str(row["name"])].append(row)
    extreme = []
    moderate = []
    low = []
    for _name, options in by_knot.items():
        extreme_options = [row for row in options if int(row["strands"]) >= 9]
        high_options = [row for row in options if int(row["strands"]) >= 6]
        if extreme_options:
            selected = min(
                extreme_options,
                key=lambda row: (
                    -int(row["strands"]),
                    int(row["acs10"]),
                    str(row["id"]),
                ),
            )
            extreme.append(selected)
        elif high_options:
            selected = min(
                high_options,
                key=lambda row: (
                    int(row["acs10"]),
                    -int(row["strands"]),
                    str(row["id"]),
                ),
            )
            moderate.append(selected)
        else:
            low.append(
                min(
                    options,
                    key=lambda row: (int(row["acs10"]), str(row["id"])),
                )
            )
    extreme.sort(
        key=lambda row: (int(row["acs10"]), -int(row["strands"]), str(row["name"]))
    )
    moderate.sort(
        key=lambda row: (int(row["acs10"]), -int(row["strands"]), str(row["name"]))
    )
    low.sort(key=lambda row: (int(row["acs10"]), int(row["strands"]), str(row["name"])))
    extreme_queue = deque(extreme)
    moderate_queue = deque(moderate)
    low_queue = deque(low)
    selected = []
    # Each checkpoint block gets the simplest available high-strand cases plus
    # easy anchors. Extreme 9--12-strand cases are spread across all blocks
    # instead of being hidden behind their simpler 6/7-strand equivalents.
    blocks = (length + 19) // 20
    for block in range(blocks):
        extreme_slots = (
            (len(extreme_queue) + (blocks - block) - 1) // (blocks - block)
            if extreme_queue
            else 0
        )
        extreme_slots = min(extreme_slots, 10)
        for _ in range(extreme_slots):
            if extreme_queue and len(selected) < length:
                selected.append(extreme_queue.popleft())
        for _ in range(10 - extreme_slots):
            source = moderate_queue if moderate_queue else extreme_queue
            if source and len(selected) < length:
                selected.append(source.popleft())
        for _ in range(10):
            source = low_queue if low_queue else (moderate_queue or extreme_queue)
            if source and len(selected) < length:
                selected.append(source.popleft())
    if len(selected) < length:
        raise ValueError(f"only {len(selected)} unique high-sequence knots available")
    return selected


def choose_simple_sequence(
    rows: list[dict[str, Any]], excluded_knots: set[str], length: int
) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["name"])
        if name in excluded_knots:
            continue
        prior = best.get(name)
        rank = (int(row["acs10"]), int(row["acs5"]), str(row["id"]))
        if prior is None or rank < (
            int(prior["acs10"]),
            int(prior["acs5"]),
            str(prior["id"]),
        ):
            best[name] = row
    ordered = sorted(
        best.values(),
        key=lambda row: (int(row["acs10"]), int(row["acs5"]), str(row["name"])),
    )
    if len(ordered) < length:
        raise ValueError(f"only {len(ordered)} disjoint simple-sequence knots available")
    return ordered[:length]


def challenge_row(row: dict[str, Any], *, prefix: str, index: int) -> dict[str, Any]:
    lower = int(row["certified_unknotting_lower_bound"])
    upper = int(row["certified_unknotting_upper_bound"])
    return {
        "challenge_id": f"{prefix}-{index + 1:03d}-{row['name']}",
        "canonical_name": str(row["name"]),
        "representation_id": str(row["id"]),
        "stored_representation": {
            "encoding": "braid-word-v1",
            "word": [int(value) for value in row["word"]],
            "strands": int(row["strands"]),
        },
        "bound_interval": [lower, upper],
        "target_u": upper - 1,
        "acs10": int(row["acs10"]),
        "acs5": int(row["acs5"]),
        "provenance": {
            "evidence_source": "pinned-knot-table-bound",
            "source_bank": row["_source_bank"],
            "source_bank_sha256": row["_source_bank_sha256"],
            "source_manifest": row["_source_manifest"],
            "source_manifest_sha256": row["_source_manifest_sha256"],
            "dataset_origin": row.get("dataset_origin"),
            "representation_variant": row.get("representation_variant"),
        },
    }


def sequence_payload(name: str, rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    challenges = [
        challenge_row(row, prefix=prefix, index=index) for index, row in enumerate(rows)
    ]
    counts = Counter(row["stored_representation"]["strands"] for row in challenges)
    return {
        "schema": "multi-knot-mastery-sequence-v1",
        "name": name,
        "size": len(challenges),
        "checkpoint_group_size": 20,
        "strand_counts": dict(sorted(counts.items())),
        "challenges": challenges,
    }


def knotinfo_reservoir(
    rows: list[dict[str, Any]], evidence_path: Path, *, max_l10: int
) -> dict[str, Any]:
    evidence = json.loads(evidence_path.read_text())
    by_knot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_knot[str(row["name"])].append(row)
    selected = []
    for name, knot in evidence.get("knots", {}).items():
        external = knot.get("knotinfo-shortest-evidence", {})
        if not (
            external.get("rankable_by_l10")
            and external.get("replay_verified")
            and int(external.get("l10", max_l10 + 1)) <= max_l10
            and name in by_knot
        ):
            continue
        row = min(
            by_knot[name],
            key=lambda item: (int(item["acs10"]), int(item["strands"]), str(item["id"])),
        )
        item = challenge_row(row, prefix="knotinfo", index=len(selected))
        item["provenance"].update(
            {
                "evidence_source": "knotinfo-shortest-evidence",
                "knotinfo_evidence_l10": int(external["l10"]),
                "knotinfo_evidence_l1000": int(external["l1000"]),
                "knotinfo_evidence_instance_id": external["instance_id"],
                "knotinfo_evidence_replay_verified": True,
                "evidence_index": str(evidence_path),
                "evidence_index_sha256": sha256(evidence_path),
            }
        )
        selected.append(item)
    selected.sort(
        key=lambda row: (
            int(row["provenance"]["knotinfo_evidence_l10"]),
            row["canonical_name"],
        )
    )
    return {
        "schema": "multi-knot-mastery-reservoir-v1",
        "name": "replay-verified-simple-knotinfo-only",
        "max_l10": max_l10,
        "size": len(selected),
        "fail_closed": True,
        "challenges": selected,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--evidence-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=240)
    parser.add_argument("--knotinfo-max-l10", type=int, default=160)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_q_rows(args.q_root)
    high = choose_high_sequence(rows, args.length)
    simple = choose_simple_sequence(
        rows, {str(row["name"]) for row in high}, args.length
    )
    paths = {
        "simple": args.output / "sequence-simple-240.json",
        "high_strand": args.output / "sequence-high-strand-240.json",
        "knotinfo_reservoir": args.output / "knotinfo-reservoir.json",
    }
    write_json(
        paths["simple"],
        sequence_payload("simple-bound-improvement-240", simple, "simple"),
    )
    write_json(
        paths["high_strand"],
        sequence_payload("high-strand-bound-improvement-240", high, "high"),
    )
    reservoir = knotinfo_reservoir(
        rows, args.evidence_index, max_l10=args.knotinfo_max_l10
    )
    write_json(paths["knotinfo_reservoir"], reservoir)
    manifest = {
        "schema": "multi-knot-mastery-curricula-manifest-v1",
        "q_root": str(args.q_root),
        "evidence_index": str(args.evidence_index),
        "evidence_index_sha256": sha256(args.evidence_index),
        "files": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "disjoint_predefined_sequences": not (
            {str(row["name"]) for row in simple}
            & {str(row["name"]) for row in high}
        ),
        "knotinfo_reservoir_size": reservoir["size"],
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
