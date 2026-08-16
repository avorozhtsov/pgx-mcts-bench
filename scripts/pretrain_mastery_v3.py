#!/usr/bin/env python3
"""Run registered mastery-v3 equivalence pretraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.mastery_v3_pretrain import pretrain_mastery_v3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=("cyclic-memory-deep-v3", "cyclic-graph-dual-v3"),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026081701)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = pretrain_mastery_v3(
        args.checkpoint,
        args.curriculum,
        args.output,
        candidate_name=args.candidate,
        steps=args.steps,
        batch_size=args.batch_size,
        views_per_identity=args.views,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
