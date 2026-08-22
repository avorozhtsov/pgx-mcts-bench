#!/usr/bin/env python3
"""Run the four slow Q lineages in one independent one-core continuation."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
POPULATION = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818"
)
Q104_ROOT = POPULATION / "q104-updated-20260819"
PRIMARY_Q104_MARKER = Q104_ROOT / "PRIMARY_8_LINEAGES_Q104_COMPLETE.json"
V3_Q104_MARKER = Q104_ROOT / "V3_BACKFILL_Q104_COMPLETE.json"
SLOW_Q104_MARKER = Q104_ROOT / "SLOW_4_LINEAGES_Q104_READY.json"
PRIMARY_Q154_ROOT = POPULATION / "q154-updated-20260819"
ROOT = POPULATION / "q154-slow4-20260822"
STATUS = ROOT / "slow4-orchestrator-status.json"
LOCK = ROOT / "slow4-orchestrator.lock"
GATE = PRIMARY_Q154_ROOT / "FAST6_SLOW4_COHORT_SPLIT_V5_VERIFIED.json"
Q104_STAGE = "q44-2-updated-scheduled-no-sharing"
Q154_STAGE = "q50-1-updated-scheduled-no-sharing-bounded"
MAX_EXPERIMENT_WORKERS = 1
SLOW_TIMEOUT_SECONDS = 7200
SLOW_TRAINING_SECONDS_PER_ITERATION_AT_REFERENCE = 7200
SLOW_BRANCHES = (
    (
        "q-grown-raster-invariant-combined-dual-12",
        "raster-invariant-combined-dual-12",
        40,
        True,
    ),
    (
        "skm-v2-high-combined-dual",
        "raster-invariant-combined-dual-12",
        40,
        True,
    ),
    ("cyclic-memory-deep-v3", "cyclic-memory-deep-v3", 40, True),
    ("cyclic-graph-dual-v3", "cyclic-graph-dual-v3", 40, True),
)
COMBINED_LABELS = tuple(branch[0] for branch in SLOW_BRANCHES[:2])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _status(state: str, stage: str, detail: str | None = None) -> None:
    _atomic_json(
        STATUS,
        {
            "schema": "q-slow4-orchestrator-v1",
            "cohort": "slow-4",
            "state": state,
            "stage": stage,
            "detail": detail,
            "pid": os.getpid(),
            "maximum_experiment_workers": MAX_EXPERIMENT_WORKERS,
            "q104_launcher": "scripts/run_local_q104_v3_backfill.py",
            "q154_launcher": "scripts/run_local_q_slow4_continuation.py",
            "slow_rehearsal_segment_timeout_seconds": SLOW_TIMEOUT_SECONDS,
            "slow_rehearsal_training_seconds_per_iteration_at_reference": (
                SLOW_TRAINING_SECONDS_PER_ITERATION_AT_REFERENCE
            ),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _acquire_lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    handle = LOCK.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another slow-4 orchestrator holds the lock") from error
    return handle


def _run_v3_q104() -> None:
    if V3_Q104_MARKER.is_file():
        return
    _status("LAUNCHED", "V3 Q104 serial backfill")
    log = ROOT / "logs/v3-q104-backfill.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        subprocess.run(
            ["uv", "run", "python", "scripts/run_local_q104_v3_backfill.py"],
            cwd=REPO,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if not V3_Q104_MARKER.is_file():
        raise RuntimeError("V3 Q104 launcher returned without its completion marker")


def _write_slow_q104_marker() -> None:
    primary = json.loads(PRIMARY_Q104_MARKER.read_text())
    v3 = json.loads(V3_Q104_MARKER.read_text())
    expected_primary = list(COMBINED_LABELS)
    expected_v3 = [branch[0] for branch in SLOW_BRANCHES[2:]]
    if any(label not in primary.get("lineages", []) for label in expected_primary):
        raise RuntimeError("primary Q104 marker is missing a combined-dual lineage")
    if v3.get("lineages") != expected_v3:
        raise RuntimeError("V3 Q104 marker order differs from the slow cohort")
    artifacts: dict[str, Any] = {}
    for label, _scientist, _simulations, _timeout in SLOW_BRANCHES:
        branch = Q104_ROOT / "branches" / label / Q104_STAGE
        report = branch / "report.json"
        state = branch / "state.pt.gz"
        if not report.is_file() or not state.is_file():
            raise RuntimeError(f"slow Q104 input is incomplete: {label}")
        artifacts[label] = {
            "report_sha256": _sha256(report),
            "state_sha256": _sha256(state),
        }
    payload = {
        "schema": "q104-slow-4-ready-v1",
        "cohort": "slow-4",
        "lineages": [branch[0] for branch in SLOW_BRANCHES],
        "artifacts": artifacts,
        "primary_marker_sha256": _sha256(PRIMARY_Q104_MARKER),
        "v3_marker_sha256": _sha256(V3_Q104_MARKER),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    if SLOW_Q104_MARKER.is_file() and json.loads(SLOW_Q104_MARKER.read_text()) != payload:
        # Timestamps are intentionally excluded from an idempotence decision.
        previous = json.loads(SLOW_Q104_MARKER.read_text())
        payload_without_time = {key: value for key, value in payload.items() if key != "completed_at"}
        previous_without_time = {
            key: value for key, value in previous.items() if key != "completed_at"
        }
        if previous_without_time != payload_without_time:
            raise RuntimeError("frozen slow-4 Q104 marker changed")
        return
    if not SLOW_Q104_MARKER.is_file():
        _atomic_json(SLOW_Q104_MARKER, payload)


def _import_combined_repair_carries() -> None:
    imports: dict[str, Any] = {}
    for label in COMBINED_LABELS:
        source = (
            PRIMARY_Q154_ROOT
            / "branches"
            / label
            / Q154_STAGE
            / "q104-rehearsal-repair-v1"
        )
        destination = ROOT / "branches" / label / Q154_STAGE / source.name
        source_hashes = _tree_hashes(source)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
        target_hashes = _tree_hashes(destination)
        if target_hashes != source_hashes:
            raise RuntimeError(f"imported rehearsal repair carry differs: {label}")
        imports[label] = {
            "source": str(source),
            "destination": str(destination),
            "files": source_hashes,
        }
    marker = ROOT / "IMPORTED_PRIMARY_COMBINED_DUAL_REPAIR_CARRIES.json"
    payload = {
        "schema": "q154-slow4-repair-carry-import-v1",
        "sharing": "none",
        "imports": imports,
    }
    if marker.is_file() and json.loads(marker.read_text()) != payload:
        raise RuntimeError("frozen repair-carry import marker changed")
    if not marker.is_file():
        _atomic_json(marker, payload)


def _q154_launcher_module():
    source = REPO / "scripts/run_local_q154_updated_continuation.py"
    spec = importlib.util.spec_from_file_location("q154_slow4_launcher", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Q154 launcher: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_q154() -> None:
    launcher = _q154_launcher_module()
    launcher.BRANCHES = SLOW_BRANCHES
    launcher.Q104_MARKER = SLOW_Q104_MARKER
    launcher.Q104_MARKER_SCHEMA = "q104-slow-4-ready-v1"
    launcher.Q104_STATUS = STATUS
    launcher.ROOT = ROOT
    launcher.STATUS = ROOT / "launcher-status.json"
    launcher.LOCK = ROOT / "launcher.lock"
    launcher.BANK = ROOT / "protocol/q50-1-updated.json"
    launcher.PRIOR = ROOT / "protocol/prior-q104-for-q50-1-updated.json"
    launcher.BUILD_AUDIT = ROOT / "protocol/q50-1-updated-audit.json"
    launcher.REHEARSAL_DEBT = ROOT / "protocol/q104-rehearsal-debt.json"
    launcher.COMPLETION_MARKER = ROOT / "ALL_SLOW_4_LINEAGES_Q154_COMPLETE"
    launcher.Q134_BARRIER_MARKER = ROOT / "ALL_SLOW_4_LINEAGES_Q134_COMPLETE.json"
    launcher.BOUNDED_REHEARSAL_FIX_GATE = GATE
    launcher.MAX_EXPERIMENT_CORES = MAX_EXPERIMENT_WORKERS
    launcher.INVARIANT_TIMEOUT_SECONDS = SLOW_TIMEOUT_SECONDS
    launcher.REHEARSAL_TRAINING_SECONDS_PER_ITERATION_AT_REFERENCE = (
        SLOW_TRAINING_SECONDS_PER_ITERATION_AT_REFERENCE
    )
    launcher.COHORT = "slow-4"
    launcher.Q134_MARKER_SCHEMA = "q154-slow-4-q134-task-order-transition-boundary-v1"
    launcher._status = {
        "schema": "q154-slow-4-population-launcher-v1",
        "cohort": "slow-4",
        "created_at": datetime.now(UTC).isoformat(),
        "state": "QUEUED",
        "stage": "awaiting durable slow-4 Q104 completion",
        "pid": os.getpid(),
        "policy": str(launcher.POLICY),
        "policy_sha256": None,
        "maximum_experiment_workers": MAX_EXPERIMENT_WORKERS,
        "branches": {},
    }
    launcher.main()


def main() -> None:
    _lock_handle = _acquire_lock()
    try:
        if not GATE.is_file():
            raise RuntimeError(f"missing verified fast-6/slow-4 split gate: {GATE}")
        _run_v3_q104()
        _write_slow_q104_marker()
        _import_combined_repair_carries()
        _status("LAUNCHED", "slow-4 Q154 serial continuation")
        _run_q154()
        _status("COMPLETED", "slow-4 Q154")
    except Exception as error:
        _status("BLOCKED", "slow-4 continuation", repr(error))
        raise
    finally:
        del _lock_handle


if __name__ == "__main__":
    main()
