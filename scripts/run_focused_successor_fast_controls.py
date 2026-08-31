#!/usr/bin/env python3
"""Durably launch the two runnable focused-successor Fast controls."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
POPULATION = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818"
)
ROOT = POPULATION / "focused-successor-v1"
GATE = ROOT / "FOCUSED_FAST_CONTROLS_VERIFIED.json"
STATUS = ROOT / "fast-controls-launcher-status.json"
LOCK = ROOT / "fast-controls-launcher.lock"
MARKER = ROOT / "ALL_FOCUSED_FAST_CONTROLS_Q304_COMPLETE"
STAGE = "q50-4-updated-scheduled-no-sharing-bounded"
CHECKPOINTS = {
    "strand-graph-12": Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/nebius-main32-final-20260817/artifacts/q4000-strand12-20260814/migrated/strand-graph-12/checkpoint.pt"),
    "raster-axial-12": Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/nebius-main32-final-20260817/artifacts/q4000-strand12-20260814/migrated/raster-axial-12/checkpoint.pt"),
}

status_lock = threading.Lock()
status: dict[str, Any] = {
    "schema": "focused-successor-v1-fast-controls-launcher",
    "state": "LAUNCHED",
    "pid": os.getpid(),
    "created_at": datetime.now(UTC).isoformat(),
    "maximum_experiment_workers": 2,
    "branches": {},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_status() -> None:
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATUS)


def set_status(line: str, state: str, detail: str | None = None) -> None:
    with status_lock:
        status["branches"][line] = {"state": state, "detail": detail, "updated_at": datetime.now(UTC).isoformat()}
        atomic_status()


def verify_gate() -> dict[str, Any]:
    gate = json.loads(GATE.read_text())
    if gate.get("schema") != "semantic-v2-q254-first-block-seeded-order-v1" or not gate.get("passed"):
        raise RuntimeError("focused Fast control gate did not pass")
    if gate.get("cohort") != "focused-successor-v1-fast-controls" or set(gate.get("focused_lines", {})) != {
        "strand-graph-12-rl-control", "raster-axial-12-control"
    }:
        raise RuntimeError("focused Fast control registration differs")
    for raw, expected in gate["source_sha256"].items():
        if sha256(Path(raw)) != expected:
            raise RuntimeError(f"focused source hash changed: {raw}")
    for line, binding in gate["focused_lines"].items():
        if sha256(Path(binding["initial_state"])) != binding["initial_state_sha256"]:
            raise RuntimeError(f"focused state hash changed: {line}")
    return gate


def output(line: str) -> Path:
    return ROOT / "branches" / line / STAGE


def run_line(line: str, binding: dict[str, Any]) -> None:
    report = output(line) / "report.json"
    if report.is_file() and json.loads(report.read_text()).get("completed_rungs") == 50:
        set_status(line, "COMPLETED", "existing durable terminal report")
        return
    set_status(line, "LAUNCHED", "awaiting scientist verification")
    log = ROOT / "logs" / f"{line}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(REPO / ".venv/bin/python"), str(REPO / "scripts/run_focused_successor_branch.py"),
        "--line", line, "--output", str(output(line)), "--gate", str(GATE),
        "--checkpoint", str(CHECKPOINTS[binding["scientist"]]),
    ]
    with log.open("a") as handle:
        subprocess.run(command, cwd=REPO, check=True, stdout=handle, stderr=subprocess.STDOUT)
    if not report.is_file():
        raise RuntimeError(f"focused branch returned without report: {line}")
    (ROOT / "branches" / line / "Q304_COMPLETE").touch()
    set_status(line, "COMPLETED")


def main() -> None:
    os.environ.update({
        "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another focused Fast launcher holds the lock") from error
    gate = verify_gate()
    status["gate"] = str(GATE)
    status["gate_sha256"] = sha256(GATE)
    atomic_status()
    failures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run_line, line, binding): line for line, binding in gate["focused_lines"].items()}
        for future in as_completed(futures):
            line = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(f"{line}: {error!r}")
                set_status(line, "BLOCKED", repr(error))
    status["state"] = "BLOCKED" if failures else "COMPLETED"
    status["failures"] = failures
    status["finished_at"] = datetime.now(UTC).isoformat()
    atomic_status()
    if failures:
        raise SystemExit("; ".join(failures))
    MARKER.touch()


if __name__ == "__main__":
    main()
