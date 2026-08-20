#!/usr/bin/env python3
"""Finish the two interrupted Q60 tails without replaying durable identities."""

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

from pgx_mcts_bench.sv2_curriculum import _load_state

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
RUN = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817")
ARCHIVE = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "nebius-main32-final-20260817/artifacts/q4000-strand12-20260814"
)
POPULATION = RUN / "continuation/q4000-v1-population-20260818"
ALIGNED = POPULATION / "aligned-20260819"
ROOT = ALIGNED / "preserved-history-20260820"
STATUS = ROOT / "launcher-status.json"
LOCK = ROOT / "launcher.lock"
QROOT = RUN / "inputs/q4000-v1"
MARKER = ROOT / "ALL_PROMISING_LINEAGES_Q60_COMPLETE"
MAX_WORKERS = 2
INVARIANT_TIMEOUT_SECONDS = 3600
F_NATIVE_LEVELS = (4, 6, 8, 12, 16)
SIMULATION_LEVELS = (40, 64, 80, 128, 256)

CHECKPOINTS = {
    "raster-axial-12": ARCHIVE / "migrated/raster-axial-12/checkpoint.pt",
    "raster-invariant-combined-dual-12": (
        ARCHIVE / "migrated/raster-invariant-combined-dual-12/checkpoint.pt"
    ),
    "strand-graph-12": ARCHIVE / "migrated/strand-graph-12/checkpoint.pt",
    "cyclic-memory-12": ARCHIVE / "migrated/cyclic-memory-12/checkpoint.pt",
}
SELECTIONS = {
    name: REPO / f"research/local-q-skm-ablation/single-{name}-selection.json"
    for name in CHECKPOINTS
}

TAILS = (
    (
        "q-grown-cyclic-memory-12",
        "cyclic-memory-12",
        ALIGNED / "branches/q-grown-cyclic-memory-12/q40-1-aligned/state.pt.gz",
        False,
        2026082001,
    ),
    (
        "skm-v2-high-combined-dual",
        "raster-invariant-combined-dual-12",
        ALIGNED / "branches/skm-v2-high-combined-dual/q40-1-aligned/state.pt.gz",
        True,
        2026082002,
    ),
)

COMPLETED_SOURCES = {
    "q-grown-raster-axial-12": (
        "raster-axial-12",
        ALIGNED / "branches/q-grown-raster-axial-12/q40-1-aligned-remaining/state.pt.gz",
    ),
    "q-grown-raster-invariant-combined-dual-12": (
        "raster-invariant-combined-dual-12",
        ALIGNED
        / "branches/q-grown-raster-invariant-combined-dual-12/q40-1-aligned-remaining/state.pt.gz",
    ),
    "q-grown-strand-graph-12": (
        "strand-graph-12",
        ALIGNED / "branches/q-grown-strand-graph-12/q40-1-aligned-remaining/state.pt.gz",
    ),
    "skm-v2-high-cyclic-memory": (
        "cyclic-memory-12",
        ALIGNED / "branches/skm-v2-high-cyclic-memory/q40-1-aligned-remaining/state.pt.gz",
    ),
    "skm-v1-simple-raster-axial": (
        "raster-axial-12",
        ALIGNED / "branches/skm-v1-simple-raster-axial/q40-1-aligned-remaining/state.pt.gz",
    ),
    "skm-v1-simple-strand-graph": (
        "strand-graph-12",
        ALIGNED / "branches/skm-v1-simple-strand-graph/q40-1-aligned/state.pt.gz",
    ),
}

_status_lock = threading.Lock()
_status: dict[str, Any] = {
    "schema": "q60-preserved-history-launcher-v1",
    "created_at": datetime.now(UTC).isoformat(),
    "state": "PREPARED",
    "stage": "preserved scheduled Q40-1 tails",
    "pid": os.getpid(),
    "maximum_experiment_workers": MAX_WORKERS,
    "arm": "scheduled-no-sharing",
    "controller_transition": "carry-forward-without-reset",
    "adaptive_protocol": {
        "initial_F_native": 4,
        "F_native_levels": list(F_NATIVE_LEVELS),
        "initial_simulations": 40,
        "simulation_levels": list(SIMULATION_LEVELS),
        "selfplay_games_per_iteration": 4,
        "optimizer_steps_per_iteration": 24,
        "evaluation_attempts_per_objective": 2,
        "qualification_attempts": 1,
        "batch_size": 64,
    },
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


def _acquire_lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    handle = LOCK.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another preserved-history Q60 launcher holds the lock") from error
    return handle


def _export_carry(label: str, scientist: str, source: Path) -> Path:
    output = ROOT / "final-q60-carry-states" / label
    destination = output / scientist / "state.pt.gz"
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/export_sv2_scientist_states.py",
            str(source),
            str(SELECTIONS[scientist]),
            str(output),
        ],
        log=ROOT / "logs" / f"{label}.log",
    )
    if not destination.is_file():
        raise RuntimeError(f"carry-forward export missing: {destination}")
    exported = _load_state(destination)
    coordinated = _load_state(source)
    expected_f = int(coordinated["f_native"][scientist])
    expected_simulations = int(coordinated["simulations"][scientist])
    if exported["f_native"] != expected_f or exported["simulations"] != expected_simulations:
        raise RuntimeError(f"adaptive controller was not carried for {label}")
    return destination


def _build_tail(label: str, source: Path) -> tuple[Path, Path, dict[str, Any]]:
    protocol = ROOT / "protocol" / label
    remaining = protocol / "q40-1-remaining-scheduled.json"
    prior = protocol / "prior-q40-1-preserved.json"
    manifest = protocol / "preserved-boundary-manifest.json"
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/build_q_aligned_boundary.py",
            "--source-bank",
            str(QROOT / "q40-1.json"),
            "--initial-prior-bank",
            str(QROOT / "prior-q40-1.json"),
            "--state",
            str(source),
            "--remaining-bank",
            str(remaining),
            "--prior-bank",
            str(prior),
            "--manifest",
            str(manifest),
            "--order",
            "scheduled",
        ],
        log=ROOT / "logs" / f"{label}.log",
    )
    payload = json.loads(manifest.read_text())
    if payload.get("arm") != "scheduled-no-sharing":
        raise RuntimeError(f"tail builder did not register scheduled-no-sharing for {label}")
    return remaining, prior, payload


def _run_tail(
    label: str,
    scientist: str,
    source: Path,
    timeout: bool,
    seed: int,
) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    remaining, prior, manifest = _build_tail(label, source)
    _set_status(
        label,
        "PREPARED",
        f"Q40-1 scheduled tail {manifest['remaining_size']}",
        f"durable={manifest['completed_in_source']}/40 source_sha256={manifest['state_sha256']}",
    )
    initial = ROOT / "initial-carry-states" / label / scientist / "state.pt.gz"
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/export_sv2_scientist_states.py",
            str(source),
            str(SELECTIONS[scientist]),
            str(initial.parents[1]),
        ],
        log=ROOT / "logs" / f"{label}.log",
    )
    output = ROOT / "branches" / label / "q40-1-preserved-scheduled-tail"
    if int(manifest["remaining_size"]) == 0:
        final = source
    elif (output / "report.json").is_file():
        final = output / "state.pt.gz"
    else:
        command = [
            "uv",
            "run",
            "pgx-mcts-bench",
            "braid-sv2-coordinated",
            "--output",
            str(output),
            "--bank",
            str(remaining),
            "--prior-bank",
            str(prior),
            "--scientist",
            f"{scientist}={CHECKPOINTS[scientist]}",
            "--initial-state",
            f"{scientist}={initial}",
            "--arm",
            "scheduled-no-sharing",
            "--ratios",
            "10,1000",
            "--simulations",
            "40",
            "--qualification-simulations",
            "40",
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
            command += ["--scientist-task-timeout-seconds", str(INVARIANT_TIMEOUT_SECONDS)]
        _set_status(label, "LAUNCHED", f"Q40-1 scheduled tail {manifest['remaining_size']}")
        _run(command, log=ROOT / "logs" / f"{label}.log")
        if not (output / "report.json").is_file():
            raise RuntimeError(f"tail returned without durable report: {label}")
        final = output / "state.pt.gz"
    _export_carry(label, scientist, final)
    (ROOT / "branches" / label / "Q60_COMPLETE").touch()
    _set_status(label, "COMPLETED", "Q60 preserved-history continuation")
    return final


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
    lock_handle = _acquire_lock()
    try:
        for label, scientist, source in (
            (label, scientist, source)
            for label, scientist, source, _timeout, _seed in TAILS
        ):
            state = _load_state(source)
            _status["branches"][label] = {
                "state": "PREPARED",
                "stage": "durable Q40-1 boundary",
                "completed_in_stage": len(state["processed"]),
                "source_state": str(source),
                "source_state_sha256": _sha256(source),
                "F_native": int(state["f_native"][scientist]),
                "simulations": int(state["simulations"][scientist]),
            }
        for label, (scientist, source) in COMPLETED_SOURCES.items():
            state = _load_state(source)
            _status["branches"][label] = {
                "state": "COMPLETED",
                "stage": "Q60 durable; carry export pending",
                "source_state": str(source),
                "source_state_sha256": _sha256(source),
                "F_native": int(state["f_native"][scientist]),
                "simulations": int(state["simulations"][scientist]),
            }
        _status["state"] = "LAUNCHED"
        _status["stage"] = "preserved scheduled Q40-1 tails"
        _write_status()

        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(_run_tail, label, scientist, source, timeout, seed): label
                for label, scientist, source, timeout, seed in TAILS
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    future.result()
                except Exception as error:
                    failures.append(f"{label}: {error!r}")
                    _set_status(label, "BLOCKED", "preserved Q40-1 tail", repr(error))
        if failures:
            raise RuntimeError("; ".join(failures))

        for label, (scientist, source) in COMPLETED_SOURCES.items():
            _export_carry(label, scientist, source)
            _set_status(label, "COMPLETED", "Q60 carry-forward export")
        with _status_lock:
            _status["state"] = "COMPLETED"
            _status["stage"] = "Q60 preserved-history population"
            _status["finished_at"] = datetime.now(UTC).isoformat()
            _write_status()
        MARKER.touch()
    except Exception as error:
        with _status_lock:
            _status["state"] = "BLOCKED"
            _status["stage"] = "launcher"
            _status["detail"] = repr(error)
            _status["updated_at"] = datetime.now(UTC).isoformat()
            _write_status()
        raise
    finally:
        del lock_handle


if __name__ == "__main__":
    main()
