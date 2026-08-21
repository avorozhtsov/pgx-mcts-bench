#!/usr/bin/env python3
"""Queue Q104 outcome calibration and the registered Q50-1 continuation to Q154."""

from __future__ import annotations

import fcntl
import gzip
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

import torch

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
RUN = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817")
ARCHIVE = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "nebius-main32-final-20260817/artifacts/q4000-strand12-20260814"
)
POPULATION = RUN / "continuation/q4000-v1-population-20260818"
Q104_ROOT = POPULATION / "q104-updated-20260819"
Q104_MARKER = Q104_ROOT / "ALL_REGISTERED_LINEAGES_Q104_COMPLETE"
Q104_STATUS = Q104_ROOT / "launcher-status.json"
ROOT = POPULATION / "q154-updated-20260819"
STATUS = ROOT / "launcher-status.json"
LOCK = ROOT / "launcher.lock"
QROOT = RUN / "inputs/q4000-v1"
POLICY = REPO / "research/local-q-skm-ablation/q50-1-updated-policy.json"
REGISTRATION = REPO / "research/mastery-v3-curriculum/curriculum.json"
Q44_BANK = REPO / "research/local-q-skm-ablation/q44-2-updated/q44-2-updated.json"
BANK = ROOT / "protocol/q50-1-updated.json"
PRIOR = ROOT / "protocol/prior-q104-for-q50-1-updated.json"
BUILD_AUDIT = ROOT / "protocol/q50-1-updated-audit.json"
MAX_EXPERIMENT_CORES = 6
INVARIANT_TIMEOUT_SECONDS = 3600
BOUNDED_REHEARSAL_FIX_GATE = ROOT / "STRICT_NO_SHARING_BOUNDED_REHEARSAL_FIX_VERIFIED.json"
BOUNDED_REHEARSAL_SOURCES = (
    REPO / "src/pgx_mcts_bench/sv2_curriculum.py",
    REPO / "src/pgx_mcts_bench/data.py",
    REPO / "src/pgx_mcts_bench/training.py",
    REPO / "src/pgx_mcts_bench/cli.py",
    REPO / "tests/test_data.py",
    REPO / "tests/test_q154_launcher.py",
    REPO / "tests/test_sv2_curriculum.py",
    REPO / "scripts/run_local_q154_updated_continuation.py",
    REPO / "research/local-q-skm-ablation/EXECUTION-CONTRACT.md",
)
REHEARSAL_DEBT = ROOT / "protocol/q104-rehearsal-debt.json"
REHEARSAL_PANEL_SIZE = 20
Q104_STAGE = "q44-2-updated-scheduled-no-sharing"
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
SELECTIONS = {
    "raster-axial-12": REPO / "research/local-q-skm-ablation/single-raster-axial-12-selection.json",
    "raster-invariant-combined-dual-12": REPO
    / "research/local-q-skm-ablation/single-raster-invariant-combined-dual-12-selection.json",
    "strand-graph-12": REPO / "research/local-q-skm-ablation/single-strand-graph-12-selection.json",
    "cyclic-memory-12": REPO
    / "research/local-q-skm-ablation/single-cyclic-memory-12-selection.json",
    "cyclic-memory-deep-v3": REPO
    / "research/local-q-skm-ablation/cyclic-memory-deep-v3-selection.json",
    "cyclic-graph-dual-v3": REPO
    / "research/local-q-skm-ablation/cyclic-graph-dual-v3-selection.json",
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
    "schema": "q154-updated-population-launcher-v1",
    "created_at": datetime.now(UTC).isoformat(),
    "state": "QUEUED",
    "stage": "awaiting durable Q104 completion",
    "pid": os.getpid(),
    "policy": str(POLICY),
    "policy_sha256": None,
    "maximum_experiment_workers": MAX_EXPERIMENT_CORES,
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


def _checkpoint(scientist: str) -> Path:
    if scientist in V3_CHECKPOINTS:
        return V3_CHECKPOINTS[scientist]
    return Q_CHECKPOINTS[scientist]


def _acquire_lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    handle = LOCK.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another Q154 launcher holds the lock") from error
    return handle


def _wait_for_q104() -> None:
    for label, _scientist, _simulations, _timeout in BRANCHES:
        _set_status(label, "QUEUED", "awaiting durable Q104 checkpoint")
    while not Q104_MARKER.is_file():
        if Q104_STATUS.is_file():
            upstream = json.loads(Q104_STATUS.read_text())
            if upstream.get("state") == "BLOCKED":
                raise RuntimeError(f"Q104 population is BLOCKED: {upstream.get('failures')}")
        with _status_lock:
            _status["last_q104_check_at"] = datetime.now(UTC).isoformat()
            _write_status()
        time.sleep(60)


def _verify_bounded_rehearsal_gate() -> None:
    if not BOUNDED_REHEARSAL_FIX_GATE.is_file():
        raise RuntimeError(f"missing bounded rehearsal gate: {BOUNDED_REHEARSAL_FIX_GATE}")
    gate = json.loads(BOUNDED_REHEARSAL_FIX_GATE.read_text())
    if gate.get("schema") != "bounded-rehearsal-protocol-fix-gate-v1" or not gate.get("passed"):
        raise RuntimeError("bounded rehearsal protocol gate did not pass")
    expected = {str(path): _sha256(path) for path in BOUNDED_REHEARSAL_SOURCES}
    if gate.get("source_sha256") != expected:
        raise RuntimeError("bounded rehearsal protocol source hashes changed")
    with _status_lock:
        _status["bounded_rehearsal_fix_gate"] = str(BOUNDED_REHEARSAL_FIX_GATE)
        _status["bounded_rehearsal_fix_verified_at"] = datetime.now(UTC).isoformat()
        _write_status()


def _load_state(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=False)


def _build_rehearsal_debt() -> dict[str, int]:
    debts: dict[str, int] = {}
    for label, scientist, _simulations, _timeout in BRANCHES:
        state_path = Q104_ROOT / "branches" / label / Q104_STAGE / "state.pt.gz"
        state = _load_state(state_path)
        debt = 0
        for event in state.get("events", []):
            rehearsal = event.get("scientists", {}).get(scientist, {}).get("rehearsal") or {}
            if rehearsal.get("hard_timeout") is not None:
                debt += max(
                    0,
                    int(rehearsal.get("F_old", 0)) - len(rehearsal.get("iterations", [])),
                )
        debts[label] = debt
    REHEARSAL_DEBT.parent.mkdir(parents=True, exist_ok=True)
    debt_payload = {
        "schema": "semantic-v2-q104-rehearsal-debt-v1",
        "formula": "sum(max(0,F_old-completed_rehearsal_iterations)) over censored blocks",
        "sharing": "none",
        "lineages": debts,
    }
    serialized = json.dumps(debt_payload, indent=2, sort_keys=True) + "\n"
    if REHEARSAL_DEBT.exists() and REHEARSAL_DEBT.read_text() != serialized:
        raise RuntimeError(f"frozen Q104 debt artifact changed: {REHEARSAL_DEBT}")
    if not REHEARSAL_DEBT.exists():
        temporary = REHEARSAL_DEBT.with_suffix(".tmp")
        temporary.write_text(serialized)
        os.replace(temporary, REHEARSAL_DEBT)
    return debts


def _build_bank() -> None:
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/build_q50_1_updated.py",
            "--q-root",
            str(QROOT),
            "--q44-bank",
            str(Q44_BANK),
            "--q104-root",
            str(Q104_ROOT),
            "--registration",
            str(REGISTRATION),
            "--policy",
            str(POLICY),
            "--output-dir",
            str(ROOT / "protocol"),
        ],
        log=ROOT / "logs/build.log",
    )
    audit = json.loads(BUILD_AUDIT.read_text())
    if audit.get("status") != "passed":
        raise RuntimeError("Q50-1-updated build audit did not pass")
    with _status_lock:
        _status["bank"] = str(BANK)
        _status["bank_sha256"] = _sha256(BANK)
        _status["bridge_mix"] = audit["bridge_mix"]
        _status["stage"] = "Q50-1-updated built and audited"
        _write_status()


def _export_initial_states() -> dict[str, Path]:
    result = {}
    log = ROOT / "logs/prepare.log"
    for label, scientist, _simulations, _timeout in BRANCHES:
        source = Q104_ROOT / "branches" / label / Q104_STAGE / "state.pt.gz"
        output_root = ROOT / "initial-q104-states" / label
        destination = output_root / scientist / "state.pt.gz"
        if not destination.is_file():
            _run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/export_sv2_scientist_states.py",
                    str(source),
                    str(SELECTIONS[scientist]),
                    str(output_root),
                ],
                log=log,
            )
        if not destination.is_file():
            raise RuntimeError(f"Q104 state export missing: {destination}")
        exported = _load_state(destination)
        source_state = _load_state(source)
        source_hash = _sha256(source)
        if exported.get("exact_representation_objectives_enriched_from") != source_hash:
            replay = exported["scientist"]["replay"]
            replay._ensure_replay_state()
            for event in source_state.get("events", []):
                representation = event.get("selected")
                native_best = event.get("scientists", {}).get(scientist, {}).get("native_best", {})
                if not representation:
                    continue
                for ratio_key, witness in native_best.items():
                    if isinstance(witness, dict) and witness.get("objective") is not None:
                        replay.record_native_objective(
                            str(representation), float(ratio_key), float(witness["objective"])
                        )
            exported["exact_representation_objectives_enriched_from"] = source_hash
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with gzip.open(temporary, "wb", compresslevel=1) as handle:
                torch.save(exported, handle)
            os.replace(temporary, destination)
        result[label] = destination
    return result


def _run_branch(
    label: str,
    scientist: str,
    simulations: int,
    timeout: bool,
    initial_state: Path,
    seed: int,
    repair_debt: int,
) -> None:
    output = ROOT / "branches" / label / "q50-1-updated-scheduled-no-sharing-bounded"
    log = ROOT / "logs" / f"{label}.log"
    if (output / "report.json").is_file():
        _set_status(label, "COMPLETED", "Q154", "existing durable report")
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
        "--rehearsal-panel-size",
        str(REHEARSAL_PANEL_SIZE),
        "--strict-own-budget-rehearsal",
        "--rehearsal-repair-debt",
        f"{scientist}={repair_debt}",
        "--terminal-full-retention-audit",
    ]
    if timeout:
        command += [
            "--scientist-task-timeout-seconds",
            str(INVARIANT_TIMEOUT_SECONDS),
            "--resumable-rehearsal-segments",
        ]
    if (output / "manifest.json").is_file():
        command.append("--resume")
    _set_status(label, "LAUNCHED", "Q50-1-updated", "awaiting worker verification")
    _run(command, log=log)
    if not (output / "report.json").is_file():
        raise RuntimeError(f"Q154 branch returned without durable report: {label}")
    (ROOT / "branches" / label / "Q154_COMPLETE").touch()
    _set_status(label, "COMPLETED", "Q154")


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
    _status["policy_sha256"] = _sha256(POLICY)
    _write_status()
    try:
        _wait_for_q104()
        _verify_bounded_rehearsal_gate()
        _build_bank()
        rehearsal_debts = _build_rehearsal_debt()
        initial_states = _export_initial_states()
        with _status_lock:
            _status["state"] = "LAUNCHED"
            _status["stage"] = "Q50-1-updated wave dispatch"
            _status["q104_completed_at"] = datetime.now(UTC).isoformat()
            _write_status()
        jobs: list[tuple[str, Callable[[], None]]] = []
        for index, (label, scientist, simulations, timeout) in enumerate(BRANCHES):
            jobs.append(
                (
                    label,
                    partial(
                        _run_branch,
                        label,
                        scientist,
                        simulations,
                        timeout,
                        initial_states[label],
                        202608190300 + index,
                        rehearsal_debts[label],
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
                    _set_status(label, "BLOCKED", "Q50-1-updated", repr(error))
        with _status_lock:
            _status["finished_at"] = datetime.now(UTC).isoformat()
            _status["state"] = "BLOCKED" if failures else "COMPLETED"
            _status["stage"] = "Q154"
            _status["failures"] = failures
            _write_status()
        if failures:
            raise SystemExit("; ".join(failures))
        (ROOT / "ALL_REGISTERED_LINEAGES_Q154_COMPLETE").touch()
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
