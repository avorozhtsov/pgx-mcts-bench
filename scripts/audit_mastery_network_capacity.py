#!/usr/bin/env python3
"""Emit a parameter-count and observed-retention diagnostic for mastery NNs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.mastery_v2 import atomic_json, parameter_capacity_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--retention", action="append", default=[], metavar="NAME=RATE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = []
    for value in args.checkpoint:
        name, path = value.split("=", 1)
        checkpoints.append((name, Path(path)))
    retention = {}
    for value in args.retention:
        name, rate = value.split("=", 1)
        retention[name] = float(rate)
    report = parameter_capacity_report(checkpoints, retention)
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
