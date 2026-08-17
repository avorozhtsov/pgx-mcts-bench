#!/usr/bin/env python3
"""Fail-closed audit for one local single-scientist Q4000 group."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retention-floor", type=float, default=0.8)
    args = parser.parse_args()

    group = json.loads(args.group.read_text())
    report_path = args.output / "report.json"
    state_path = args.output / "state.pt.gz"
    report = json.loads(report_path.read_text())
    rows = {str(row["id"]): row for row in group["rows"]}
    expected = int(group["size"])
    skip_count = sum(int(value) for value in report.get("curriculum_skips", {}).values())
    skip_reasons = [
        str(scientist["curriculum_skip"].get("reason"))
        for event in report.get("events", [])
        for scientist in event.get("scientists", {}).values()
        if scientist.get("curriculum_skip") is not None
    ]
    maximum_skips = int(group["skip_policy"]["maximum_skips"])

    failures: list[str] = []
    if report.get("bank_sha256") != sha256(args.group):
        failures.append("bank hash differs from the frozen Q4000 group")
    if int(report.get("completed_rungs", -1)) != expected:
        failures.append(f"completed_rungs != {expected}")
    if len(report.get("processed", [])) != expected:
        failures.append(f"processed identities != {expected}")
    if len(report.get("events", [])) != expected:
        failures.append(f"durable report events != {expected}")
    native_events = sorted((args.output / "native-events").glob("*.json"))
    if len(native_events) != expected:
        failures.append(f"durable native events {len(native_events)} != {expected}")
    if not state_path.is_file() or state_path.stat().st_size == 0:
        failures.append("coordinated state is missing or empty")
    if skip_count > maximum_skips:
        failures.append(f"skip ledger {skip_count} exceeds {maximum_skips}")
    if "capacity" in skip_reasons:
        failures.append("capacity exception recorded")

    retention_rates: list[float] = []
    for block in report.get("block_reports", []):
        for scientist in block.get("scientists", {}).values():
            retention = scientist.get("retention_after")
            if retention and int(retention.get("attempts", 0)):
                retention_rates.append(
                    int(retention["solved"]) / int(retention["attempts"])
                )
    if retention_rates and min(retention_rates) < args.retention_floor:
        failures.append(
            f"retention {min(retention_rates):.6f} < {args.retention_floor:.6f}"
        )

    high_strand_native_successes: list[str] = []
    for event in report.get("events", []):
        identity = str(event.get("selected"))
        if identity not in rows or int(rows[identity]["strands"]) < 6:
            continue
        for scientist in event.get("scientists", {}).values():
            iterations = scientist.get("iterations", [])
            if any(int(item.get("selfplay_solved", 0)) > 0 for item in iterations):
                high_strand_native_successes.append(identity)
                break
    if not high_strand_native_successes:
        failures.append("no native self-play success on a 6+ strand representation")

    result = {
        "schema": "local-q4000-group-audit-v1",
        "group": group["name"],
        "expected_rows": expected,
        "completed_rungs": report.get("completed_rungs"),
        "native_event_files": len(native_events),
        "skip_count": skip_count,
        "skip_reasons": skip_reasons,
        "maximum_skips": maximum_skips,
        "retention_rates": retention_rates,
        "retention_floor": args.retention_floor,
        "high_strand_native_successes": sorted(set(high_strand_native_successes)),
        "passed": not failures,
        "failures": failures,
    }
    destination = args.output / "local-group-audit.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
