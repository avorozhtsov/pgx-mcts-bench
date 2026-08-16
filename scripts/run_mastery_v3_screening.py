#!/usr/bin/env python3
"""Run the registered paired 20-knot mastery-v3 screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.mastery_v3_screening import run_screening


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--deep-checkpoint", type=Path, required=True)
    parser.add_argument("--graph-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--benchmark-simulations", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            run_screening(
                args.curriculum,
                args.source_checkpoint,
                args.deep_checkpoint,
                args.graph_checkpoint,
                args.output,
                device=args.device,
                simulations=args.simulations,
                benchmark_simulations=args.benchmark_simulations,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
