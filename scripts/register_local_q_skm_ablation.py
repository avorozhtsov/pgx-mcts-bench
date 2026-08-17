#!/usr/bin/env python3
"""Freeze common train/evaluation panels for the local Q-vs-SKM comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def q_row(challenge: dict[str, Any], index: int) -> dict[str, Any]:
    representation = challenge["stored_representation"]
    lower, upper = (int(value) for value in challenge["bound_interval"])
    return {
        "id": str(challenge["representation_id"]),
        "name": str(challenge["canonical_name"]),
        "crossings": len(representation["word"]),
        "word": [int(value) for value in representation["word"]],
        "strands": int(representation["strands"]),
        "cheap_score": float(challenge["acs10"]),
        "difficulty_quartile": index // 5,
        "known_unknotting_number": upper,
        "certified_unknotting_upper_bound": upper,
        "certified_unknotting_lower_bound": lower,
        "acs10": int(challenge["acs10"]),
        "acs5": int(challenge.get("acs5", challenge["acs10"])),
        "provenance": dict(challenge.get("provenance", {})),
    }


def write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--evaluation-size", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty {args.output}")
    source = json.loads(args.source_sequence.read_text())
    required = args.train_size + args.evaluation_size
    challenges = list(source["challenges"][:required])
    if len(challenges) != required:
        raise ValueError("source sequence is shorter than the requested paired panels")
    train = challenges[: args.train_size]
    evaluation = challenges[args.train_size :]
    if {row["representation_id"] for row in train} & {
        row["representation_id"] for row in evaluation
    }:
        raise ValueError("training and evaluation representations overlap")
    if any(int(row["stored_representation"]["strands"]) > 12 for row in challenges):
        raise ValueError("panel exceeds the registered 12-strand capacity")
    args.output.mkdir(parents=True)
    sequence = {
        "schema": "q-skm-short-ablation-sequence-v1",
        "name": "local-q-vs-skm-train-20",
        "size": len(train),
        "source_sequence": str(args.source_sequence.resolve()),
        "source_sequence_sha256": sha256(args.source_sequence),
        "challenges": train,
    }
    train_bank = {
        "schema": "q-skm-common-bank-v1",
        "name": "local-q-vs-skm-train-20",
        "size": len(train),
        "skip_policy": {"maximum_skips": 1, "failures_remain_in_denominator": True},
        "rows": [q_row(row, index) for index, row in enumerate(train)],
    }
    evaluation_bank = {
        "schema": "q-skm-common-bank-v1",
        "name": "local-q-vs-skm-heldout-20",
        "size": len(evaluation),
        "skip_policy": {"maximum_skips": 1, "failures_remain_in_denominator": True},
        "rows": [q_row(row, index) for index, row in enumerate(evaluation)],
    }
    write(args.output / "train-sequence.json", sequence)
    write(args.output / "train-bank.json", train_bank)
    write(args.output / "heldout-bank.json", evaluation_bank)
    files = ("train-sequence.json", "train-bank.json", "heldout-bank.json")
    manifest = {
        "schema": "local-q-skm-ablation-inputs-v1",
        "source_sequence": str(args.source_sequence.resolve()),
        "source_sequence_sha256": sha256(args.source_sequence),
        "train_representation_ids": [row["representation_id"] for row in train],
        "heldout_representation_ids": [row["representation_id"] for row in evaluation],
        "files": {name: sha256(args.output / name) for name in files},
    }
    write(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
