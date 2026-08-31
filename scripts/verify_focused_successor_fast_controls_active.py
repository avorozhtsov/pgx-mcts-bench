#!/usr/bin/env python3
"""Write repeated-sample evidence that both focused Fast controls are active."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818/focused-successor-v1"
)
GATE = ROOT / "FOCUSED_FAST_CONTROLS_RECOVERY_V3_VERIFIED.json"
OUTPUT = ROOT / "FOCUSED_FAST_CONTROLS_RECOVERY_V3_VERIFIED_ACTIVE.json"
STAGE = "q50-4-updated-scheduled-no-sharing-bounded-recovery-v3"


def processes():
    raw = subprocess.check_output(["ps", "-axo", "pid=,ppid=,time=,command="], text=True)
    found = {}
    for line in raw.splitlines():
        if "multiprocessing.spawn" not in line:
            continue
        parts = line.strip().split(maxsplit=3)
        if len(parts) == 4:
            found[int(parts[0])] = {"ppid": int(parts[1]), "cpu_time": parts[2], "command": parts[3]}
    return found


def seconds(value: str) -> float:
    fields = value.split(":")
    if len(fields) == 2:
        return int(fields[0]) * 60 + float(fields[1])
    return int(fields[0]) * 3600 + int(fields[1]) * 60 + float(fields[2])


def main() -> None:
    gate = json.loads(GATE.read_text())
    first = processes()
    time.sleep(8)
    second = processes()
    branch_pids = {}
    ps_raw = subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True)
    for line_name in gate["focused_lines"]:
        token = f"--line {line_name} "
        parents = []
        for row in ps_raw.splitlines():
            if token in row and "run_focused_successor_branch.py" in row:
                parents.append(int(row.strip().split(maxsplit=2)[0]))
        if len(parents) != 1:
            raise RuntimeError(f"expected one branch writer for {line_name}, got {parents}")
        children = [pid for pid, item in second.items() if item["ppid"] == parents[0]]
        if len(children) != 1 or children[0] not in first:
            raise RuntimeError(f"expected one stable scientist for {line_name}, got {children}")
        pid = children[0]
        delta = seconds(second[pid]["cpu_time"]) - seconds(first[pid]["cpu_time"])
        if delta <= 0:
            raise RuntimeError(f"scientist CPU did not advance: {line_name}")
        manifest = ROOT / "branches" / line_name / STAGE / "manifest.json"
        if not manifest.is_file() or time.time() - manifest.stat().st_mtime > 300:
            raise RuntimeError(f"focused manifest is not fresh: {line_name}")
        branch_pids[line_name] = {
            "branch_pid": parents[0], "scientist_pid": pid, "cpu_delta_seconds": delta,
            "manifest": str(manifest), "manifest_mtime": manifest.stat().st_mtime,
        }
    free = os.statvfs(ROOT)
    free_bytes = free.f_bavail * free.f_frsize
    if free_bytes <= 8 * 1024**3:
        raise RuntimeError("disk headroom is below 8 GiB")
    value = {
        "schema": "focused-successor-v1-fast-controls-verified-active-v3",
        "status": "VERIFIED ACTIVE",
        "verified_at": datetime.now(UTC).isoformat(),
        "gate": str(GATE),
        "branches": branch_pids,
        "duplicate_writers": False,
        "free_bytes": free_bytes,
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
