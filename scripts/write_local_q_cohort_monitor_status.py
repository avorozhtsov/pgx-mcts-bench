#!/usr/bin/env python3
"""Write independent, fresh process evidence for one local Q cohort.

This utility never rewrites launcher-owned status or gate files.  It samples the
existing process tree twice and atomically writes a separate monitor status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "local-q-cohort-monitor-status-v1"
WRITER_MARKERS = (
    "run_local_q204_fast6_branch.py",
    "run_local_q254_fast6_branch.py",
    "braid-sv2-coordinated",
)


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    cpu_seconds: float
    command: str


def _cpu_seconds(value: str) -> float:
    day_parts = value.split("-", 1)
    days = int(day_parts[0]) if len(day_parts) == 2 else 0
    clock = day_parts[-1].split(":")
    if len(clock) == 2:
        hours = 0
        minutes, seconds = clock
    elif len(clock) == 3:
        hours, minutes, seconds = clock
    else:
        raise ValueError(f"unsupported ps CPU time: {value!r}")
    return days * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_ps(text: str) -> dict[int, Process]:
    processes: dict[int, Process] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        pid, ppid, cpu, command = line.split(None, 3)
        process = Process(int(pid), int(ppid), _cpu_seconds(cpu), command)
        processes[process.pid] = process
    return processes


def read_ps() -> dict[int, Process]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,time=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_ps(result.stdout)


def descendants(processes: dict[int, Process], root_pid: int) -> set[int]:
    result: set[int] = set()
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        children = [p.pid for p in processes.values() if p.ppid == parent]
        for child in children:
            if child not in result:
                result.add(child)
                frontier.append(child)
    return result


def is_ancestor(processes: dict[int, Process], ancestor: int, child: int) -> bool:
    seen: set[int] = set()
    current = child
    while current in processes and current not in seen:
        seen.add(current)
        current = processes[current].ppid
        if current == ancestor:
            return True
    return False


def output_root(command: str) -> str | None:
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    try:
        return argv[argv.index("--output") + 1]
    except (ValueError, IndexError):
        return None


def leaf_writers(
    processes: dict[int, Process], launcher_pid: int, artifact_root: Path
) -> list[Process]:
    tree = descendants(processes, launcher_pid)
    candidates = [
        processes[pid]
        for pid in tree
        if str(artifact_root) in processes[pid].command
        and any(marker in processes[pid].command for marker in WRITER_MARKERS)
    ]
    return [
        process
        for process in candidates
        if not any(
            other.pid != process.pid
            and output_root(other.command) == output_root(process.command)
            and is_ancestor(processes, process.pid, other.pid)
            for other in candidates
        )
    ]


def launchd_pid(label: str) -> int:
    uid = os.getuid()
    result = subprocess.run(
        ["launchctl", "print", f"gui/{uid}/{label}"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^\s*pid = (\d+)\s*$", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(f"launchd job {label!r} has no running pid")
    return int(match.group(1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_status(
    *,
    label: str,
    cohort: str,
    phase: str,
    artifact_root: Path,
    launcher_script: Path,
    gate: Path,
    min_scientists: int,
    max_scientists: int,
    launcher_pid: int,
    before: dict[int, Process],
    after: dict[int, Process],
    disk_free_bytes: int,
    sampled_at: datetime,
) -> dict[str, object]:
    launcher = after.get(launcher_pid)
    launcher_ok = launcher is not None and str(launcher_script) in launcher.command
    tree = descendants(after, launcher_pid) if launcher_ok else set()
    scientists = sorted(
        pid
        for pid in tree
        if "multiprocessing.spawn" in after[pid].command
        and "resource_tracker" not in after[pid].command
    )
    cpu_samples = {
        str(pid): {
            "before_seconds": before[pid].cpu_seconds if pid in before else None,
            "after_seconds": after[pid].cpu_seconds,
            "advanced": pid in before and after[pid].cpu_seconds > before[pid].cpu_seconds,
        }
        for pid in scientists
    }
    writers = leaf_writers(after, launcher_pid, artifact_root) if launcher_ok else []
    writer_roots: dict[str, list[int]] = {}
    for writer in writers:
        writer_roots.setdefault(output_root(writer.command) or "<missing-output>", []).append(
            writer.pid
        )
    duplicate_roots = {root: pids for root, pids in writer_roots.items() if len(pids) > 1}
    gate_ok = gate.is_file()
    source_ok = launcher_script.is_file()
    disk_ok = disk_free_bytes >= 8 * 1024**3
    scientist_count_ok = min_scientists <= len(scientists) <= max_scientists
    cpu_ok = bool(scientists) and all(sample["advanced"] for sample in cpu_samples.values())
    verified = all(
        (launcher_ok, gate_ok, source_ok, disk_ok, scientist_count_ok, cpu_ok, not duplicate_roots)
    )
    return {
        "schema": SCHEMA,
        "state": "VERIFIED ACTIVE" if verified else "LAUNCHED",
        "cohort": cohort,
        "phase": phase,
        "updated_at": sampled_at.astimezone(UTC).isoformat(),
        "launchd": {"label": label, "pid": launcher_pid, "launcher_matches": launcher_ok},
        "artifact_root": str(artifact_root),
        "launcher_script": {
            "path": str(launcher_script),
            "sha256": sha256(launcher_script) if source_ok else None,
        },
        "gate": {"path": str(gate), "sha256": sha256(gate) if gate_ok else None},
        "scientists": {
            "minimum": min_scientists,
            "maximum": max_scientists,
            "pids": scientists,
            "cpu": cpu_samples,
        },
        "writers": {
            "leaf_pids": [writer.pid for writer in writers],
            "roots": writer_roots,
            "duplicate_roots": duplicate_roots,
        },
        "disk": {"free_bytes": disk_free_bytes, "minimum_bytes": 8 * 1024**3, "passed": disk_ok},
        "checks": {
            "launcher": launcher_ok,
            "gate": gate_ok,
            "source": source_ok,
            "scientist_count": scientist_count_ok,
            "scientist_cpu_advanced": cpu_ok,
            "duplicate_writers_absent": not duplicate_roots,
            "disk": disk_ok,
        },
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--launcher-script", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-scientists", type=int, required=True)
    parser.add_argument("--max-scientists", type=int, required=True)
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    launcher_pid = launchd_pid(args.label)
    before = read_ps()
    time.sleep(args.sample_seconds)
    after = read_ps()
    free = shutil.disk_usage(args.artifact_root).free
    payload = build_status(
        label=args.label,
        cohort=args.cohort,
        phase=args.phase,
        artifact_root=args.artifact_root,
        launcher_script=args.launcher_script,
        gate=args.gate,
        min_scientists=args.min_scientists,
        max_scientists=args.max_scientists,
        launcher_pid=launcher_pid,
        before=before,
        after=after,
        disk_free_bytes=free,
        sampled_at=datetime.now(UTC),
    )
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["state"] == "VERIFIED ACTIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
