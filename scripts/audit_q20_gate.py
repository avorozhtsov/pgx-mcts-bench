#!/usr/bin/env python3
"""Produce the fail-closed Q20 capacity-gate decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.q_gates import audit_q20_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_q20_gate(args.root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return {"passed": 0, "failed": 1, "pending": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
