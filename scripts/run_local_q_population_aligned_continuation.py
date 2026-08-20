#!/usr/bin/env python3
"""Continue every unfinished Q60 lineage from its own aligned compute boundary."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from pgx_mcts_bench.sv2_curriculum import _load_state

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
RUN = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817")
ARCHIVE = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "nebius-main32-final-20260817/artifacts/q4000-strand12-20260814"
)
MASTERY = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "nebius-main32-final-20260817/artifacts"
)
POPULATION = RUN / "continuation/q4000-v1-population-20260818"
SPLIT = POPULATION / "split-20260819"
ROOT = POPULATION / "aligned-20260819"
STATUS = ROOT / "launcher-status.json"
LOCK = ROOT / "launcher.lock"
QROOT = RUN / "inputs/q4000-v1"
RECOVERY = POPULATION / "q20-recovery"
MAX_EXPERIMENT_CORES = 6
INVARIANT_TIMEOUT_SECONDS = 3600

F_NATIVE_START = 4
F_NATIVE_LEVELS = (4, 6, 8, 12, 16)
SIMULATIONS_START = 40
SIMULATION_LEVELS = (40, 64, 80, 128, 256)
SELFPLAY_GAMES = 4
TRAIN_STEPS = 24
EVALUATION_ATTEMPTS = 2

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

ACTIVE_Q40 = (
    ("q-grown-raster-axial-12", "raster-axial-12", False),
    (
        "q-grown-raster-invariant-combined-dual-12",
        "raster-invariant-combined-dual-12",
        True,
    ),
    ("q-grown-strand-graph-12", "strand-graph-12", False),
    ("skm-v2-high-cyclic-memory", "cyclic-memory-12", False),
    ("skm-v1-simple-raster-axial", "raster-axial-12", False),
)

_status_lock = threading.Lock()
_status: dict[str, Any] = {
    "schema": "q-aligned-population-launcher-v1",
    "created_at": datetime.now(UTC).isoformat(),
    "state": "PREPARED",
    "stage": "aligned boundaries",
    "pid": os.getpid(),
    "maximum_experiment_workers": MAX_EXPERIMENT_CORES,
    "protocol": {
        "F_native_start": F_NATIVE_START,
        "F_native_levels": list(F_NATIVE_LEVELS),
        "simulations_start": SIMULATIONS_START,
        "simulation_levels": list(SIMULATION_LEVELS),
        "selfplay_games_per_iteration": SELFPLAY_GAMES,
        "optimizer_steps_per_iteration": TRAIN_STEPS,
        "evaluation_attempts_per_objective": EVALUATION_ATTEMPTS,
        "adaptive_compute": True,
    },
    "branches": {},
}


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
        raise RuntimeError("another aligned Q60 launcher holds the lock") from error
    return handle


def _export_boundary_state(label: str, scientist: str, state: Path) -> Path:
    output = ROOT / "boundary-states" / label
    destination = output / scientist / "state.pt.gz"
    log = ROOT / "logs" / f"{label}.log"
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/export_sv2_scientist_states.py",
            str(state),
            str(SELECTIONS[scientist]),
            str(output),
            "--f-native",
            str(F_NATIVE_START),
            "--simulations",
            str(SIMULATIONS_START),
        ],
        log=log,
    )
    if not destination.is_file():
        raise RuntimeError(f"aligned boundary export missing: {destination}")
    return destination


def _export_final_q60_state(label: str, scientist: str, state: Path) -> Path:
    output = ROOT / "final-q60-states" / label
    destination = output / scientist / "state.pt.gz"
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/export_sv2_scientist_states.py",
            str(state),
            str(SELECTIONS[scientist]),
            str(output),
            "--f-native",
            str(F_NATIVE_START),
            "--simulations",
            str(SIMULATIONS_START),
        ],
        log=ROOT / "logs" / f"{label}.log",
    )
    if not destination.is_file():
        raise RuntimeError(f"final Q60 export missing: {destination}")
    return destination


def _build_boundary(
    label: str,
    state: Path,
    source_bank: Path,
    initial_prior: Path | None,
) -> tuple[Path, Path, int]:
    protocol = ROOT / "protocol" / label
    remaining = protocol / "remaining.json"
    prior = protocol / "prior.json"
    manifest = protocol / "boundary-manifest.json"
    command = [
        "uv",
        "run",
        "python",
        "scripts/build_q_aligned_boundary.py",
        "--source-bank",
        str(source_bank),
        "--state",
        str(state),
        "--remaining-bank",
        str(remaining),
        "--prior-bank",
        str(prior),
        "--manifest",
        str(manifest),
    ]
    if initial_prior is not None:
        command += ["--initial-prior-bank", str(initial_prior)]
    _run(command, log=ROOT / "logs" / f"{label}.log")
    payload = json.loads(manifest.read_text())
    return remaining, prior, int(payload["remaining_size"])


def _run_stage(
    *,
    label: str,
    stage: str,
    output: Path,
    bank: Path,
    prior_bank: Path | None,
    scientist: str,
    initial_state: Path,
    seed: int,
    timeout: bool,
) -> None:
    log = ROOT / "logs" / f"{label}.log"
    command = [
        "uv",
        "run",
        "pgx-mcts-bench",
        "braid-sv2-coordinated",
        "--output",
        str(output),
        "--bank",
        str(bank),
    ]
    if prior_bank is not None:
        command += ["--prior-bank", str(prior_bank)]
    command += [
        "--scientist",
        f"{scientist}={CHECKPOINTS[scientist]}",
        "--initial-state",
        f"{scientist}={initial_state}",
        "--arm",
        "scheduled-no-sharing",
        "--ratios",
        "10,1000",
        "--simulations",
        str(SIMULATIONS_START),
        "--qualification-simulations",
        str(SIMULATIONS_START),
        "--qualification-attempts",
        "1",
        "--f-native",
        str(F_NATIVE_START),
        "--selfplay-games",
        str(SELFPLAY_GAMES),
        "--train-steps",
        str(TRAIN_STEPS),
        "--batch-size",
        "64",
        "--evaluation-attempts",
        str(EVALUATION_ATTEMPTS),
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
    _set_status(label, "LAUNCHED", stage, "awaiting worker verification")
    _run(command, log=log)
    if not (output / "report.json").is_file():
        raise RuntimeError(f"aligned stage returned without durable report: {label} {stage}")


def _continue_active_q40(
    label: str,
    scientist: str,
    timeout: bool,
    seed: int,
) -> None:
    old = SPLIT / "branches" / label / "q40-1-static-no-sharing"
    state = old / "state.pt.gz"
    initial = _export_boundary_state(label, scientist, state)
    remaining, prior, remaining_size = _build_boundary(
        label,
        state,
        QROOT / "q40-1.json",
        QROOT / "prior-q40-1.json",
    )
    if remaining_size:
        output = ROOT / "branches" / label / "q40-1-aligned-remaining"
        _run_stage(
            label=label,
            stage=f"aligned Q40-1 remaining {remaining_size}",
            output=output,
            bank=remaining,
            prior_bank=prior,
            scientist=scientist,
            initial_state=initial,
            seed=seed,
            timeout=timeout,
        )
        final_state = output / "state.pt.gz"
    else:
        final_state = state
    _export_final_q60_state(label, scientist, final_state)
    (ROOT / "branches" / label / "Q60_COMPLETE").touch()


def _continue_cyclic_memory() -> None:
    label = "q-grown-cyclic-memory-12"
    scientist = "cyclic-memory-12"
    old = SPLIT / "branches" / label / "q20-recovery-tail-static-no-sharing"
    state = old / "state.pt.gz"
    initial = _export_boundary_state(label, scientist, state)
    remaining, prior, remaining_size = _build_boundary(
        label,
        state,
        RECOVERY / "q20-tail.json",
        RECOVERY / "prior-q20-tail.json",
    )
    if remaining_size:
        q20_output = ROOT / "branches" / label / "q20-aligned-remaining"
        _run_stage(
            label=label,
            stage=f"aligned Q20 remaining {remaining_size}",
            output=q20_output,
            bank=remaining,
            prior_bank=prior,
            scientist=scientist,
            initial_state=initial,
            seed=2026081961,
            timeout=False,
        )
        q20_state = q20_output / "state.pt.gz"
    else:
        q20_state = state
    q40_initial = _export_boundary_state(label + "-q40", scientist, q20_state)
    q40_output = ROOT / "branches" / label / "q40-1-aligned"
    _run_stage(
        label=label,
        stage="aligned Q40-1",
        output=q40_output,
        bank=QROOT / "q40-1.json",
        prior_bank=QROOT / "prior-q40-1.json",
        scientist=scientist,
        initial_state=q40_initial,
        seed=2026081962,
        timeout=False,
    )
    _export_final_q60_state(label, scientist, q40_output / "state.pt.gz")
    (ROOT / "branches" / label / "Q60_COMPLETE").touch()


def _bridge_mastery(label: str, scientist: str, f_old: int) -> Path:
    version, source_name = {
        "skm-v1-simple-strand-graph": ("multi-knot-mastery-v1-20260815", "simple-strand-graph"),
        "skm-v2-high-combined-dual": ("multi-knot-mastery-v2-20260815", "high-combined-dual"),
    }[label]
    source = MASTERY / version / "scientists" / source_name / "scientist-state.pt.gz"
    destination = ROOT / "boundary-states" / label / f"{label}.pt.gz"
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/bridge_mastery_state_to_sv2.py",
            str(source),
            str(destination),
            "--scientist",
            scientist,
            "--f-old",
            str(f_old),
            "--f-native",
            str(F_NATIVE_START),
            "--simulations",
            str(SIMULATIONS_START),
            "--lineage",
            label,
        ],
        log=ROOT / "logs" / f"{label}.log",
    )
    return destination


def _run_queued_branch(
    label: str,
    scientist: str,
    f_old: int,
    timeout: bool,
    seed: int,
) -> None:
    initial = _bridge_mastery(label, scientist, f_old)
    q20_output = ROOT / "branches" / label / "q20-aligned"
    q40_output = ROOT / "branches" / label / "q40-1-aligned"
    _run_stage(
        label=label,
        stage="aligned Q20",
        output=q20_output,
        bank=QROOT / "q20.json",
        prior_bank=None,
        scientist=scientist,
        initial_state=initial,
        seed=seed,
        timeout=timeout,
    )
    q40_initial = _export_boundary_state(label + "-q40", scientist, q20_output / "state.pt.gz")
    _run_stage(
        label=label,
        stage="aligned Q40-1",
        output=q40_output,
        bank=QROOT / "q40-1.json",
        prior_bank=QROOT / "prior-q40-1.json",
        scientist=scientist,
        initial_state=q40_initial,
        seed=seed + 1,
        timeout=timeout,
    )
    _export_final_q60_state(label, scientist, q40_output / "state.pt.gz")
    (ROOT / "branches" / label / "Q60_COMPLETE").touch()


def _record_source_boundaries() -> None:
    rows = {}
    for label, scientist, _timeout in ACTIVE_Q40:
        state = SPLIT / "branches" / label / "q40-1-static-no-sharing/state.pt.gz"
        payload = _load_state(state)
        rows[label] = {
            "scientist": scientist,
            "stage": "Q40-1",
            "completed_in_stage": len(payload["processed"]),
            "state": str(state),
            "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
        }
    cyclic_state = (
        SPLIT
        / "branches/q-grown-cyclic-memory-12/q20-recovery-tail-static-no-sharing/state.pt.gz"
    )
    cyclic = _load_state(cyclic_state)
    rows["q-grown-cyclic-memory-12"] = {
        "scientist": "cyclic-memory-12",
        "stage": "Q20 recovery tail",
        "completed_in_stage": len(cyclic["processed"]),
        "state": str(cyclic_state),
        "state_sha256": hashlib.sha256(cyclic_state.read_bytes()).hexdigest(),
    }
    _status["source_boundaries"] = rows
    _write_status()


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
    _write_status()
    try:
        _record_source_boundaries()
        jobs: list[tuple[str, Callable[[], None]]] = [
            *[
                (
                    label,
                    partial(
                        _continue_active_q40,
                        label,
                        scientist,
                        timeout,
                        2026081963 + index,
                    ),
                )
                for index, (label, scientist, timeout) in enumerate(ACTIVE_Q40)
            ],
            ("q-grown-cyclic-memory-12", _continue_cyclic_memory),
            (
                "skm-v1-simple-strand-graph",
                partial(
                    _run_queued_branch,
                    "skm-v1-simple-strand-graph",
                    "strand-graph-12",
                    8,
                    False,
                    2026081971,
                ),
            ),
            (
                "skm-v2-high-combined-dual",
                partial(
                    _run_queued_branch,
                    "skm-v2-high-combined-dual",
                    "raster-invariant-combined-dual-12",
                    2,
                    True,
                    2026081973,
                ),
            ),
        ]
        for label, _job in jobs:
            _set_status(label, "QUEUED", "awaiting one of six aligned experiment slots")
        with _status_lock:
            _status["state"] = "LAUNCHED"
            _status["stage"] = "aligned Q60 dispatch"
            _write_status()

        failures = []
        with ThreadPoolExecutor(max_workers=MAX_EXPERIMENT_CORES) as executor:
            futures = {executor.submit(job): label for label, job in jobs}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    future.result()
                    _set_status(label, "COMPLETED", "Q60")
                except Exception as error:
                    failures.append(f"{label}: {error!r}")
                    _set_status(label, "BLOCKED", "aligned Q60", repr(error))
        with _status_lock:
            _status["state"] = "BLOCKED" if failures else "COMPLETED"
            _status["stage"] = "aligned Q60"
            _status["failures"] = failures
            _status["finished_at"] = datetime.now(UTC).isoformat()
            _write_status()
        if failures:
            raise SystemExit("; ".join(failures))
        (ROOT / "ALL_PROMISING_LINEAGES_Q60_COMPLETE").touch()
    finally:
        del _lock_handle


if __name__ == "__main__":
    main()
