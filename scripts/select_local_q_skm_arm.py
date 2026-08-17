#!/usr/bin/env python3
"""Select exactly one paired arm from held-out reports using the registered order."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="LABEL=REPORT.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("initialization", "process"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    rows = []
    for item in args.report:
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        report = json.loads(path.read_text())
        rows.append(
            {
                "label": label,
                "report": str(path.resolve()),
                "strict_successes": int(report["strict_successes"]),
                "relaxed_successes": int(report["relaxed_successes"]),
                "mean_capped_l1000": float(report["mean_capped_l1000"]),
                "strict_successes_per_million_evaluations": float(
                    report["strict_successes_per_million_evaluations"]
                ),
                "strict_successes_per_cpu_hour": float(
                    report["strict_successes_per_cpu_hour"]
                ),
                "retention": report["retention_of_relaxed_successes_at_strict_target"],
                "capacity_exceptions": int(report["capacity_exceptions"]),
            }
        )
    eligible = [row for row in rows if row["capacity_exceptions"] == 0]
    if not eligible:
        raise RuntimeError("all paired arms had capacity exceptions")
    def rank(row):
        return (
            -row["strict_successes"],
            -row["relaxed_successes"],
            row["mean_capped_l1000"],
            -row["strict_successes_per_million_evaluations"],
            -row["strict_successes_per_cpu_hour"],
        )

    ranked = sorted(eligible, key=lambda row: (*rank(row), row["label"]))
    if len(ranked) > 1 and rank(ranked[0]) == rank(ranked[1]):
        raise RuntimeError("paired arms are indistinguishable at the registered dose")
    selected = ranked[0]
    payload = {
        "schema": "local-q-skm-selection-v1",
        "kind": args.kind,
        "selected": selected["label"],
        "primary_order": [
            "strict_successes desc",
            "relaxed_successes desc when strict successes tie",
            "mean_capped_l1000 asc",
            "strict_successes_per_million_evaluations desc",
            "strict_successes_per_cpu_hour desc",
        ],
        "retention_threshold": 0.8,
        "retention_is_safety_flag_not_selection_override": True,
        "selected_retention_alert": (
            selected["retention"] is not None and float(selected["retention"]) < 0.8
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(selected["label"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
