#!/usr/bin/env python3
"""Recover the five failed fast-6 Q134 -> Q154 branch transitions.

The original fast-6 launcher remains attached to the already-active
q-grown-raster-axial-12 writer.  This recovery launcher owns only the other
five output roots, so it cannot duplicate that active branch.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import run_local_q154_updated_continuation as launcher

ACTIVE_ORIGINAL_LABEL = "q-grown-raster-axial-12"
RECOVERY_STATUS = launcher.ROOT / "fast6-transition-recovery-status.json"
RECOVERY_LOCK = launcher.ROOT / "fast6-transition-recovery.lock"
RECOVERY_BRANCHES = tuple(
    branch for branch in launcher.BRANCHES if branch[0] != ACTIVE_ORIGINAL_LABEL
)


def _atomic_status(payload: dict[str, Any]) -> None:
    temporary = RECOVERY_STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, RECOVERY_STATUS)


def _output(label: str) -> Path:
    return (
        launcher.ROOT
        / "branches"
        / label
        / "q50-1-updated-scheduled-no-sharing-bounded"
    )


def _coordinated_processes(label: str) -> list[tuple[int, int]]:
    needle = f"braid-sv2-coordinated --output {_output(label)}"
    rows: list[tuple[int, int]] = []
    process = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in process.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and needle in fields[2]:
            rows.append((int(fields[0]), int(fields[1])))
    return rows


def _verify_writer_partition() -> None:
    active = _coordinated_processes(ACTIVE_ORIGINAL_LABEL)
    if len(active) != 2 or active[1][1] != active[0][0]:
        raise RuntimeError(
            "expected exactly one uv/python coordinated writer chain for "
            f"{ACTIVE_ORIGINAL_LABEL}, found {active}"
        )
    conflicts: dict[str, list[tuple[int, int]]] = {}
    for label, *_rest in RECOVERY_BRANCHES:
        processes = _coordinated_processes(label)
        if processes:
            conflicts[label] = processes
    if conflicts:
        raise RuntimeError(f"recovery output already has an active writer: {conflicts}")


def _terminal_report(label: str) -> bool:
    report_path = _output(label) / "report.json"
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text())
    return report.get("completed_rungs") == 50 and bool(
        report.get("terminal_retention_audit")
    )


def _write_fast6_completion_marker() -> None:
    if not all(_terminal_report(label) for label in launcher.FAST_6_LABELS):
        raise RuntimeError("cannot publish fast-6 completion before six terminal reports")
    launcher.COMPLETION_MARKER.touch()


def main() -> None:
    RECOVERY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = RECOVERY_LOCK.open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("fast-6 transition recovery launcher already exists") from error

    launcher.STATUS = RECOVERY_STATUS
    launcher._status_lock = threading.Lock()
    launcher._status = {
        "schema": "q154-fast-6-transition-recovery-v1",
        "cohort": launcher.COHORT,
        "created_at": datetime.now(UTC).isoformat(),
        "state": "LAUNCHED",
        "stage": "Q135-Q154 recovery dispatch",
        "pid": os.getpid(),
        "excluded_active_original_branch": ACTIVE_ORIGINAL_LABEL,
        "maximum_experiment_workers": len(RECOVERY_BRANCHES),
        "gate": str(launcher.BOUNDED_REHEARSAL_FIX_GATE),
        "branches": {},
    }
    _atomic_status(launcher._status)
    _verify_writer_partition()
    launcher._verify_bounded_rehearsal_gate()
    debt_payload = json.loads(launcher.REHEARSAL_DEBT.read_text())
    debts = debt_payload["lineages"]

    failures: list[str] = []
    jobs = []
    for index, (label, scientist, simulations, timeout) in enumerate(RECOVERY_BRANCHES):
        initial_state = (
            launcher.ROOT
            / "initial-q104-states"
            / label
            / scientist
            / "state.pt.gz"
        )
        jobs.append(
            (
                label,
                partial(
                    launcher._run_branch,
                    label,
                    scientist,
                    simulations,
                    timeout,
                    initial_state,
                    202608190300 + launcher.PRIMARY_8_SEED_INDEX[label],
                    int(debts[label]),
                    phase="q154-final",
                ),
            )
        )
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(job): label for label, job in jobs}
        for future in as_completed(futures):
            label = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(f"{label}: {error!r}")
                launcher._set_status(label, "BLOCKED", "q154-final", repr(error))
    if failures:
        launcher._status["state"] = "BLOCKED"
        launcher._status["failures"] = failures
        launcher._status["finished_at"] = datetime.now(UTC).isoformat()
        _atomic_status(launcher._status)
        raise SystemExit("; ".join(failures))

    while not _terminal_report(ACTIVE_ORIGINAL_LABEL):
        launcher._status["state"] = "QUEUED"
        launcher._status["stage"] = "awaiting original active branch terminal report"
        launcher._status["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_status(launcher._status)
        time.sleep(30)
    _write_fast6_completion_marker()
    launcher._status["state"] = "COMPLETED"
    launcher._status["stage"] = "Q154"
    launcher._status["finished_at"] = datetime.now(UTC).isoformat()
    _atomic_status(launcher._status)


if __name__ == "__main__":
    main()
