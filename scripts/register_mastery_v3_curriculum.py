#!/usr/bin/env python3
"""Register or audit the controlled mastery-v3 curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.mastery_v3_curriculum import audit_promotion, build_curriculum


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--q-root", type=Path, required=True)
    build.add_argument("--dkt-panel", type=Path, required=True)
    build.add_argument("--evidence-snapshot", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=2026081700)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--curriculum", type=Path, required=True)
    audit.add_argument("--report", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        value = build_curriculum(
            args.q_root, args.dkt_panel, args.evidence_snapshot, seed=args.seed
        )
    else:
        value = audit_promotion(
            json.loads(args.curriculum.read_text()), json.loads(args.report.read_text())
        )
    _write(args.output, value)
    print(json.dumps({"output": str(args.output), "status": value.get("status", "registered")}))


if __name__ == "__main__":
    main()
