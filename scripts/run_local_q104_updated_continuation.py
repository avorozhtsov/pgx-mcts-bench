#!/usr/bin/env python3
"""Queue and run all registered Q60 lineages through Q44-2-updated to Q104."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
RUN = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817")
ARCHIVE = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "nebius-main32-final-20260817/artifacts/q4000-strand12-20260814"
)
POPULATION = RUN / "continuation/q4000-v1-population-20260818"
ALIGNED = POPULATION / "aligned-20260819"
PRESERVED = ALIGNED / "preserved-history-20260820"
EXPLORATORY = RUN / "continuation/q4000-v1-exploratory-q40-1-20260818"
ROOT = POPULATION / "q104-updated-20260819"
STATUS = ROOT / "launcher-status.json"
LOCK = ROOT / "launcher.lock"
Q60_MARKER = PRESERVED / "ALL_PROMISING_LINEAGES_Q60_COMPLETE"
Q60_STATUS = PRESERVED / "launcher-status.json"
REHEARSAL_FIX_GATE = ROOT / "REHEARSAL_TIMEOUT_CHECKPOINT_FIX_VERIFIED"
BANK = REPO / "research/local-q-skm-ablation/q44-2-updated/q44-2-updated.json"
PRIOR = REPO / "research/local-q-skm-ablation/q44-2-updated/prior-q60-for-q44-2-updated.json"
BUILD_AUDIT = REPO / "research/local-q-skm-ablation/q44-2-updated/q44-2-updated-audit.json"
MAX_EXPERIMENT_CORES = 6
INVARIANT_TIMEOUT_SECONDS = 3600
F_NATIVE_LEVELS = (4, 6, 8, 12, 16)
SIMULATION_LEVELS = (40, 64, 80, 128, 256)

Q_CHECKPOINTS = {
    "raster-axial-12": ARCHIVE / "migrated/raster-axial-12/checkpoint.pt",
    "raster-invariant-combined-dual-12": (
        ARCHIVE / "migrated/raster-invariant-combined-dual-12/checkpoint.pt"
    ),
    "strand-graph-12": ARCHIVE / "migrated/strand-graph-12/checkpoint.pt",
    "cyclic-memory-12": ARCHIVE / "migrated/cyclic-memory-12/checkpoint.pt",
}
V3_CHECKPOINTS = {
    "cyclic-memory-deep-v3": (
        RUN.parent / "mastery-v3-l40s-20260816/artifacts/cyclic-memory-deep-v3-migrated.pt"
    ),
    "cyclic-graph-dual-v3": (
        RUN.parent / "mastery-v3-l40s-20260816/artifacts/cyclic-graph-dual-v3-migrated.pt"
    ),
}
SINGLE_SELECTIONS = {
    "raster-axial-12": REPO / "research/local-q-skm-ablation/single-raster-axial-12-selection.json",
    "raster-invariant-combined-dual-12": REPO
    / "research/local-q-skm-ablation/single-raster-invariant-combined-dual-12-selection.json",
    "strand-graph-12": REPO / "research/local-q-skm-ablation/single-strand-graph-12-selection.json",
    "cyclic-memory-12": REPO
    / "research/local-q-skm-ablation/single-cyclic-memory-12-selection.json",
}

BRANCHES = (
    ("cyclic-memory-deep-v3", "cyclic-memory-deep-v3", 40, False),
    ("cyclic-graph-dual-v3", "cyclic-graph-dual-v3", 40, False),
    ("q-grown-raster-axial-12", "raster-axial-12", 40, False),
    (
        "q-grown-raster-invariant-combined-dual-12",
        "raster-invariant-combined-dual-12",
        40,
        True,
    ),
    ("q-grown-strand-graph-12", "strand-graph-12", 40, False),
    ("q-grown-cyclic-memory-12", "cyclic-memory-12", 40, False),
    ("skm-v2-high-cyclic-memory", "cyclic-memory-12", 40, False),
    ("skm-v1-simple-raster-axial", "raster-axial-12", 40, False),
    ("skm-v1-simple-strand-graph", "strand-graph-12", 40, False),
    (
        "skm-v2-high-combined-dual",
        "raster-invariant-combined-dual-12",
        40,
        True,
    ),
)

_status_lock = threading.Lock()
_status: dict[str, Any] = {
    "schema": "q104-updated-population-launcher-v1",
    "created_at": datetime.now(UTC).isoformat(),
    "state": "QUEUED",
    "stage": "awaiting durable Q60 completion",
    "pid": os.getpid(),
    "max_experiment_cores": MAX_EXPERIMENT_CORES,
    "bank": str(BANK),
    "bank_sha256": None,
    "build_audit": str(BUILD_AUDIT),
    "rehearsal_checkpoint_fix_gate": str(REHEARSAL_FIX_GATE),
    "branches": {},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_status() -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(_status, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATUS)


def _set_status(label: str, state: str, stage: str, detail: str | None = None) -> None:
    with _status_lock:
        _status["branches"][label] = {
            "state": state,
            "stage": stage,
            "detail": detail,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _write_status()


def _run(command: list[str], *, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write("EXEC " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=REPO,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def _wait_for_q60() -> None:
    for label, _scientist, _simulations, _timeout in BRANCHES:
        _set_status(label, "QUEUED", "awaiting durable Q60 checkpoint")
    while not Q60_MARKER.is_file():
        if Q60_STATUS.is_file():
            q60 = json.loads(Q60_STATUS.read_text())
            if q60.get("state") == "BLOCKED":
                raise RuntimeError(f"Q60 population is BLOCKED: {q60.get('failures')}")
        with _status_lock:
            _status["last_q60_check_at"] = datetime.now(UTC).isoformat()
            _write_status()
        time.sleep(60)


def _wait_for_rehearsal_checkpoint_fix() -> None:
    for label, _scientist, _simulations, _timeout in BRANCHES:
        _set_status(label, "QUEUED", "awaiting verified rehearsal checkpoint fix")
    while not REHEARSAL_FIX_GATE.is_file():
        with _status_lock:
            _status["state"] = "QUEUED"
            _status["stage"] = "awaiting verified rehearsal checkpoint fix"
            _status["last_rehearsal_fix_check_at"] = datetime.now(UTC).isoformat()
            _write_status()
        time.sleep(15)


def _export_state(coordinated_state: Path, selection: Path, destination: Path, log: Path) -> None:
    if destination.is_file():
        return
    if not coordinated_state.is_file():
        raise FileNotFoundError(coordinated_state)
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/export_sv2_scientist_states.py",
            str(coordinated_state),
            str(selection),
            str(destination.parents[1]),
        ],
        log=log,
    )
    if not destination.is_file():
        raise RuntimeError(f"export did not create {destination}")


def _prepare_initial_states() -> dict[str, Path]:
    initial_root = ROOT / "initial-q60-states"
    log = ROOT / "logs/prepare.log"
    v3_state = EXPLORATORY / "q40-1-static-no-sharing/state.pt.gz"
    v3_selection = REPO / "research/local-q-skm-ablation/q44-2-updated-lineage-selection.json"
    for scientist in V3_CHECKPOINTS:
        destination = initial_root / scientist / scientist / "state.pt.gz"
        if not destination.is_file():
            _run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/export_sv2_scientist_states.py",
                    str(v3_state),
                    str(v3_selection),
                    str(destination.parents[1]),
                    "--f-native",
                    "4",
                    "--simulations",
                    "40",
                ],
                log=log,
            )
        if not destination.is_file():
            raise RuntimeError(f"initial V3 aligned export missing: {destination}")
    result = {
        scientist: initial_root / scientist / scientist / "state.pt.gz"
        for scientist in V3_CHECKPOINTS
    }
    for label, scientist, _simulations, _timeout in BRANCHES[2:]:
        source = PRESERVED / "final-q60-carry-states" / label / scientist / "state.pt.gz"
        if not source.is_file():
            raise FileNotFoundError(source)
        result[label] = source
    return result


def _checkpoint(scientist: str) -> Path:
    if scientist in V3_CHECKPOINTS:
        return V3_CHECKPOINTS[scientist]
    return Q_CHECKPOINTS[scientist]


def _run_branch(
    label: str,
    scientist: str,
    simulations: int,
    timeout: bool,
    initial_state: Path,
    seed: int,
) -> None:
    output = ROOT / "branches" / label / "q44-2-updated-scheduled-no-sharing"
    log = ROOT / "logs" / f"{label}.log"
    if (output / "report.json").is_file():
        _set_status(label, "COMPLETED", "Q104", "existing durable report")
        return
    command = [
        "uv",
        "run",
        "pgx-mcts-bench",
        "braid-sv2-coordinated",
        "--output",
        str(output),
        "--bank",
        str(BANK),
        "--prior-bank",
        str(PRIOR),
        "--scientist",
        f"{scientist}={_checkpoint(scientist)}",
        "--initial-state",
        f"{scientist}={initial_state}",
        "--arm",
        "scheduled-no-sharing",
        "--ratios",
        "10,1000",
        "--simulations",
        str(simulations),
        "--qualification-simulations",
        str(simulations),
        "--qualification-attempts",
        "1",
        "--f-native",
        "4",
        "--selfplay-games",
        "4",
        "--train-steps",
        "24",
        "--batch-size",
        "64",
        "--evaluation-attempts",
        "2",
        "--block-size",
        "10",
        "--retention-target",
        "0.8",
        "--action-horizon",
        "128",
        "--rungs",
        "0",
        "--seed",
        str(seed),
        "--torch-threads",
        "1",
        "--parallel-scientists",
        "--adaptive-compute",
        "--f-native-levels",
        ",".join(map(str, F_NATIVE_LEVELS)),
        "--simulation-levels",
        ",".join(map(str, SIMULATION_LEVELS)),
        "--device",
        "cpu",
    ]
    if timeout:
        command += [
            "--scientist-task-timeout-seconds",
            str(INVARIANT_TIMEOUT_SECONDS),
        ]
    if (output / "manifest.json").is_file():
        command.append("--resume")
    _set_status(label, "LAUNCHED", "Q44-2-updated", "awaiting worker verification")
    _run(command, log=log)
    if not (output / "report.json").is_file():
        raise RuntimeError(f"Q104 branch returned without durable report: {label}")
    (ROOT / "branches" / label / "Q104_COMPLETE").touch()
    _set_status(label, "COMPLETED", "Q104")


def _acquire_lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    handle = LOCK.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another Q104 updated launcher holds the lock") from error
    return handle


def main() -> None:
    os.environ.update(
        {
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    _lock_handle = _acquire_lock()
    # Keep the handle alive for the process lifetime; closing it releases flock.
    audit = json.loads(BUILD_AUDIT.read_text())
    if audit.get("status") != "passed":
        raise RuntimeError("Q44-2-updated build audit did not pass")
    _status["bank_sha256"] = _sha256(BANK)
    _write_status()
    try:
        _wait_for_q60()
        _wait_for_rehearsal_checkpoint_fix()
        initial_states = _prepare_initial_states()
        with _status_lock:
            _status["state"] = "LAUNCHED"
            _status["stage"] = "Q44-2-updated wave dispatch"
            _status["q60_completed_at"] = datetime.now(UTC).isoformat()
            _write_status()
        jobs: list[tuple[str, Callable[[], None]]] = []
        for index, (label, scientist, simulations, timeout) in enumerate(BRANCHES):
            state_key = scientist if label in V3_CHECKPOINTS else label
            jobs.append(
                (
                    label,
                    partial(
                        _run_branch,
                        label,
                        scientist,
                        simulations,
                        timeout,
                        initial_states[state_key],
                        202608190200 + index,
                    ),
                )
            )
        failures = []
        with ThreadPoolExecutor(max_workers=MAX_EXPERIMENT_CORES) as executor:
            futures = {executor.submit(job): label for label, job in jobs}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    future.result()
                except Exception as error:
                    failures.append(f"{label}: {error!r}")
                    _set_status(label, "BLOCKED", "Q44-2-updated", repr(error))
        with _status_lock:
            _status["finished_at"] = datetime.now(UTC).isoformat()
            _status["state"] = "BLOCKED" if failures else "COMPLETED"
            _status["stage"] = "Q104"
            _status["failures"] = failures
            _write_status()
        if failures:
            raise SystemExit("; ".join(failures))
        (ROOT / "ALL_REGISTERED_LINEAGES_Q104_COMPLETE").touch()
    except Exception as error:
        with _status_lock:
            _status["state"] = "BLOCKED"
            _status["stage"] = "launcher"
            _status["detail"] = repr(error)
            _status["updated_at"] = datetime.now(UTC).isoformat()
            _write_status()
        raise


if __name__ == "__main__":
    main()
