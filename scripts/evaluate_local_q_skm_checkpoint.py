#!/usr/bin/env python3
"""Evaluate a Q or SKM scientist state on one frozen held-out panel."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from pgx_mcts_bench.adaptive_scientists import load_scientist
from pgx_mcts_bench.mastery_v3_curriculum import file_sha256
from pgx_mcts_bench.mastery_v3_screening import _retention, _run_phase
from pgx_mcts_bench.single_knot_mastery import MasteryConfig, ScientistMasteryBackend


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as source:
            return torch.load(source, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)


def plain_checkpoint(path: Path, name: str, directory: Path) -> Path:
    payload = load_payload(path)
    if "scientists" in payload:
        if name not in payload["scientists"]:
            raise ValueError(f"{path} has no scientist named {name}")
        payload = {"network": payload["scientists"][name]["network"]}
    elif "network" not in payload:
        raise ValueError(f"{path} has no network payload")
    target = directory / "checkpoint.pt"
    torch.save(payload, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--scientist", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026081803)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    bank = json.loads(args.bank.read_text())
    rows = []
    source_ids = {}
    for source_row in bank["rows"]:
        source_id = str(source_row.get("representation_id") or source_row["id"])
        suffix = hashlib.sha256(source_id.encode()).hexdigest()[-8:]
        evaluation_id = f"{source_id}::{suffix}"
        rows.append({**source_row, "representation_id": evaluation_id})
        source_ids[evaluation_id] = source_id
    seeds = [args.seed]
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with tempfile.TemporaryDirectory() as raw_directory:
        checkpoint = plain_checkpoint(args.state, args.scientist, Path(raw_directory))
        scientist = load_scientist(
            args.scientist,
            checkpoint,
            seed=args.seed,
            device="cpu",
            simulations=args.simulations,
            require_factorized=True,
            objective_budget_channel=True,
        )
        config = MasteryConfig(
            knot_name="local-q-skm-heldout",
            initial_target_u=0,
            ratio=1000.0,
            move_allowance=128,
            parallel_searches=args.threads,
            torch_threads=args.threads,
            attempts_per_node=1,
            simulations=args.simulations,
            train_steps_per_batch=1,
            seed=args.seed,
        )
        backend = ScientistMasteryBackend(scientist, config)
        journal_root = args.output.parent / f".{args.output.stem}-journals"
        strict = _run_phase(
            backend,
            rows,
            seeds,
            journal_root / "strict.json",
            simulations=args.simulations,
            strict=True,
        )
        relaxed = _run_phase(
            backend,
            rows,
            seeds,
            journal_root / "relaxed.json",
            simulations=args.simulations,
            strict=False,
        )
    retention, retention_by_strand = _retention(relaxed, strict)
    strict_successes = sum(bool(item["solved"]) for item in strict)
    evaluations = sum(int(item["scheduled_network_evaluations"]) for item in strict)
    wall_seconds = sum(float(item["wall_seconds"]) for item in strict)
    capped = sum(float(item["l1000_capped"]) for item in strict) / len(strict)
    strand_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in strict:
        strand_rows[int(item["strands"])].append(item)
    report = {
        "schema": "local-q-skm-heldout-evaluation-v1",
        "scientist": args.scientist,
        "state": str(args.state.resolve()),
        "state_sha256": file_sha256(args.state),
        "bank": str(args.bank.resolve()),
        "bank_sha256": file_sha256(args.bank),
        "items": len(strict),
        "simulations": args.simulations,
        "seed": args.seed,
        "strict_successes": strict_successes,
        "relaxed_successes": sum(bool(item["solved"]) for item in relaxed),
        "mean_capped_l1000": capped,
        "scheduled_network_evaluations": evaluations,
        "strict_successes_per_million_evaluations": (
            strict_successes * 1_000_000 / evaluations if evaluations else 0.0
        ),
        "strict_successes_per_cpu_hour": (
            strict_successes * 3600 / wall_seconds if wall_seconds else 0.0
        ),
        "retention_of_relaxed_successes_at_strict_target": retention,
        "retention_by_strand": retention_by_strand,
        "capacity_exceptions": sum(bool(item["capacity_exception"]) for item in strict),
        "representation_id_map": source_ids,
        "strict_by_strand": {
            str(strands): sum(bool(item["solved"]) for item in items) / len(items)
            for strands, items in sorted(strand_rows.items())
        },
        "strict_items": strict,
        "relaxed_items": relaxed,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
