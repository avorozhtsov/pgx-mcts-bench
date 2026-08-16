#!/usr/bin/env python3
"""Run registered proof-aware L10/L1000 mastery-v3 distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.mastery_v3_distill import distill_mastery_v3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--evidence-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=("cyclic-memory-deep-v3", "cyclic-graph-dual-v3"),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--samples-per-side", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--policy-weight", type=float, default=0.5)
    parser.add_argument("--preservation-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026081701)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = distill_mastery_v3(
        args.checkpoint,
        args.curriculum,
        args.evidence_snapshot,
        args.output,
        candidate_name=args.candidate,
        steps=args.steps,
        samples_per_side=args.samples_per_side,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_weight=args.policy_weight,
        preservation_weight=args.preservation_weight,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
