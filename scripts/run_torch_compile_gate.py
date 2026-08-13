#!/usr/bin/env python3
"""Run the isolated training compilation gate without changing launchers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.compile_gate import run_compile_gate
from pgx_mcts_bench.ladder import candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", default="raster-axial-v3")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--actor-batch", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    by_name = {candidate.name: candidate for candidate in candidates()}
    if args.candidate not in by_name:
        parser.error(f"unknown candidate: {args.candidate}")
    report = run_compile_gate(
        by_name[args.candidate],
        args.output,
        stage_index=args.stage,
        simulations=args.simulations,
        actor_batch=args.actor_batch,
        measured_train_steps=args.train_steps,
        seed=args.seed,
        torch_threads=args.torch_threads,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
