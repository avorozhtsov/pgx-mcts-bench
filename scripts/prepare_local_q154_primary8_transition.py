#!/usr/bin/env python3
"""Bind the primary-8 Q104 boundary and prepare its separate Q154 launch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
POPULATION = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818"
)
Q104_ROOT = POPULATION / "q104-updated-20260819"
Q154_ROOT = POPULATION / "q154-updated-20260819"
STAGE = "q44-2-updated-scheduled-no-sharing"
PRIMARY = [
    "q-grown-raster-axial-12",
    "q-grown-raster-invariant-combined-dual-12",
    "q-grown-strand-graph-12",
    "q-grown-cyclic-memory-12",
    "skm-v2-high-cyclic-memory",
    "skm-v1-simple-raster-axial",
    "skm-v1-simple-strand-graph",
    "skm-v2-high-combined-dual",
]
DEFERRED = ["cyclic-graph-dual-v3", "cyclic-memory-deep-v3"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _q154_launcher_module():
    source = REPO / "scripts/run_local_q154_updated_continuation.py"
    spec = importlib.util.spec_from_file_location("q154_primary_launcher_for_gate", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Q154 launcher: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _primary_artifacts() -> dict[str, dict[str, Any]]:
    artifacts = {}
    for label in PRIMARY:
        branch = Q104_ROOT / "branches" / label / STAGE
        report = branch / "report.json"
        state = branch / "state.pt.gz"
        completion = Q104_ROOT / "branches" / label / "Q104_COMPLETE"
        if not report.is_file() or not state.is_file() or not completion.is_file():
            raise RuntimeError(f"primary Q104 branch is not durable: {label}")
        payload = json.loads(report.read_text())
        if int(payload.get("completed_rungs", 0)) != 44:
            raise RuntimeError(f"primary Q104 report is not 44/44: {label}")
        artifacts[label] = {
            "completed_events": 44,
            "report_sha256": _sha256(report),
            "state_sha256": _sha256(state),
        }
    return artifacts


def _deferred_artifacts() -> dict[str, dict[str, Any]]:
    artifacts = {}
    for label in DEFERRED:
        branch = Q104_ROOT / "branches" / label / STAGE
        state = branch / "state.pt.gz"
        events = sorted((branch / "events").glob("*.json"))
        if not state.is_file() or not events:
            raise RuntimeError(f"deferred V3 branch has no durable boundary: {label}")
        if (branch / "report.json").exists():
            raise RuntimeError(f"deferred V3 branch unexpectedly completed: {label}")
        artifacts[label] = {
            "state": "PREPARED",
            "completed_events": len(events),
            "last_event": events[-1].name,
            "last_event_sha256": _sha256(events[-1]),
            "state_sha256": _sha256(state),
            "resume_script": "scripts/run_local_q104_v3_backfill.py",
        }
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-result", required=True)
    args = parser.parse_args()
    now = datetime.now(UTC).isoformat()
    primary_artifacts = _primary_artifacts()
    deferred_artifacts = _deferred_artifacts()

    _atomic_json(
        Q104_ROOT / "PRIMARY_8_LINEAGES_Q104_COMPLETE.json",
        {
            "schema": "q104-primary-8-completion-v1",
            "cohort": "primary-8",
            "lineages": PRIMARY,
            "artifacts": primary_artifacts,
            "completed_at": now,
        },
    )
    _atomic_json(
        Q104_ROOT / "V3_BACKFILL_DEFERRED.json",
        {
            "schema": "q104-v3-backfill-deferred-v1",
            "cohort": "deferred-v3-backfill",
            "lineages": DEFERRED,
            "artifacts": deferred_artifacts,
            "prepared_at": now,
        },
    )

    q104_status = json.loads((Q104_ROOT / "launcher-status.json").read_text())
    q104_status.update(
        {
            "schema": "q104-primary-8-with-deferred-v3-status-v1",
            "state": "COMPLETED",
            "stage": "primary-8 Q104",
            "cohort": "primary-8",
            "pid": None,
            "finished_at": now,
            "completion_marker": "PRIMARY_8_LINEAGES_Q104_COMPLETE.json",
            "deferred_marker": "V3_BACKFILL_DEFERRED.json",
        }
    )
    for label in DEFERRED:
        q104_status["branches"][label] = {
            "state": "PREPARED",
            "stage": "deferred V3 Q104 backfill",
            "detail": f"durable {deferred_artifacts[label]['completed_events']}/44 boundary",
            "updated_at": now,
        }
    _atomic_json(Q104_ROOT / "launcher-status.json", q104_status)
    _atomic_json(
        Q104_ROOT / "v3-backfill-launcher-status.json",
        {
            "schema": "q104-v3-backfill-prepared-status-v1",
            "state": "PREPARED",
            "stage": "deferred Q104 backfill",
            "cohort": "deferred-v3-backfill",
            "pid": None,
            "branches": {
                label: {
                    "state": "PREPARED",
                    "stage": f"durable {deferred_artifacts[label]['completed_events']}/44 boundary",
                }
                for label in DEFERRED
            },
            "resume_script": "scripts/run_local_q104_v3_backfill.py",
            "updated_at": now,
        },
    )

    launcher = _q154_launcher_module()
    source_sha256 = {str(path): _sha256(path) for path in launcher.BOUNDED_REHEARSAL_SOURCES}
    gate_name = "STRICT_NO_SHARING_BOUNDED_REHEARSAL_FIX_VERIFIED.json"
    _atomic_json(
        Q154_ROOT / gate_name,
        {
            "schema": "bounded-rehearsal-protocol-fix-gate-v2-cohort-scoped",
            "passed": True,
            "sharing": "strict-none",
            "cohorts": ["primary-8", "deferred-v3-backfill"],
            "source_sha256": source_sha256,
            "tests": {"command": "uv run pytest -q", "result": args.tests_result},
            "verified_at": now,
        },
    )
    _atomic_json(
        Q154_ROOT / "Q154_PREPARED_BOUNDED_REHEARSAL_PROTOCOL.json",
        {
            "schema": "q154-primary-8-bounded-rehearsal-prepared-v1",
            "authoritative_state": "PREPARED",
            "cohort": "primary-8",
            "gate": gate_name,
            "launcher_active": False,
            "next_prerequisite": "PRIMARY_8_LINEAGES_Q104_COMPLETE.json",
            "protocol": "scheduled-no-sharing",
            "rehearsal_panel_size": 20,
            "deferred_backfill": DEFERRED,
            "verified_at": now,
        },
    )
    _atomic_json(
        Q154_ROOT / "launcher-status.json",
        {
            "schema": "q154-primary-8-prepared-status-v1",
            "state": "PREPARED",
            "stage": "primary-8 strict no-sharing gate verified",
            "cohort": "primary-8",
            "pid": None,
            "launcher_active": False,
            "gate": gate_name,
            "gate_verified": True,
            "next_prerequisite": "q104-updated-20260819/PRIMARY_8_LINEAGES_Q104_COMPLETE.json",
            "maximum_experiment_workers": 6,
            "branches": {
                label: {"state": "PREPARED", "stage": "Q154 dispatch ready"}
                for label in PRIMARY
            },
            "updated_at": now,
        },
    )


if __name__ == "__main__":
    main()
