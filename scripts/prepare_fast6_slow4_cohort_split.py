#!/usr/bin/env python3
"""Write the verified fast-6 / slow-4 Q154 cohort-split protocol gate."""

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
GATE = Q154_ROOT / "FAST6_SLOW4_COHORT_SPLIT_V5_VERIFIED.json"
PREDECESSOR_GATE = Q154_ROOT / "FAST6_SLOW4_COHORT_SPLIT_V4_VERIFIED.json"
STAGE = "q50-1-updated-scheduled-no-sharing-bounded"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _launcher_module():
    source = REPO / "scripts/run_local_q154_updated_continuation.py"
    spec = importlib.util.spec_from_file_location("q154_fast6_slow4_gate", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Q154 launcher: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fast_boundaries(labels: tuple[str, ...]) -> dict[str, Any]:
    if PREDECESSOR_GATE.is_file():
        predecessor = json.loads(PREDECESSOR_GATE.read_text())
        artifacts = predecessor.get("fast_q134_boundaries", {})
        if set(artifacts) != set(labels):
            raise RuntimeError("predecessor gate fast cohort differs")
        for label in labels:
            output = Q154_ROOT / "branches" / label / STAGE
            report = output / "barrier-report-030.json"
            event = output / "events/029.json"
            if _sha256(report) != artifacts[label]["barrier_report_sha256"]:
                raise RuntimeError(f"frozen fast barrier report changed: {label}")
            if _sha256(event) != artifacts[label]["last_event_sha256"]:
                raise RuntimeError(f"frozen fast Q134 event changed: {label}")
        return artifacts
    artifacts: dict[str, Any] = {}
    for label in labels:
        output = Q154_ROOT / "branches" / label / STAGE
        report = output / "barrier-report-030.json"
        state = output / "state.pt.gz"
        events = sorted((output / "events").glob("*.json"))
        if len(events) != 30 or events[-1].name != "029.json":
            raise RuntimeError(f"fast lineage is not exactly at Q134: {label}")
        payload = json.loads(report.read_text())
        if payload.get("completed_rungs") != 30 or not payload.get(
            "paused_at_rehearsal_barrier"
        ):
            raise RuntimeError(f"invalid fast lineage Q134 barrier: {label}")
        artifacts[label] = {
            "barrier_report_sha256": _sha256(report),
            "state_sha256": _sha256(state),
            "event_count": len(events),
            "last_event_sha256": _sha256(events[-1]),
        }
    return artifacts


def _slow_boundaries(labels: tuple[str, ...]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    deferred = Q104_ROOT / "V3_BACKFILL_DEFERRED.json"
    deferred_payload = json.loads(deferred.read_text())
    for label in labels:
        if label in {"cyclic-memory-deep-v3", "cyclic-graph-dual-v3"}:
            row = deferred_payload["artifacts"][label]
            branch = Q104_ROOT / "branches" / label / "q44-2-updated-scheduled-no-sharing"
            state = branch / "state.pt.gz"
            event = branch / "events" / row["last_event"]
            if _sha256(state) != row["state_sha256"] or _sha256(event) != row[
                "last_event_sha256"
            ]:
                raise RuntimeError(f"deferred V3 boundary hash changed: {label}")
            artifacts[label] = {
                "stage": "Q104 backfill",
                "completed_events": row["completed_events"],
                "state_sha256": row["state_sha256"],
                "last_event_sha256": row["last_event_sha256"],
            }
            continue
        repair = Q154_ROOT / "branches" / label / STAGE / "q104-rehearsal-repair-v1"
        checkpoint = repair / "phase-checkpoint.pt.gz"
        state = repair / "state.pt.gz"
        event = repair / "events/000.json"
        artifacts[label] = {
            "stage": "Q104 rehearsal debt repair",
            "repair_state_sha256": _sha256(state),
            "repair_checkpoint_sha256": _sha256(checkpoint),
            "completed_repair_event_sha256": _sha256(event),
        }
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-result", required=True)
    args = parser.parse_args()
    launcher = _launcher_module()
    manifests = sorted((Q154_ROOT / "branches").glob(f"*/{STAGE}/manifest.json"))
    if not manifests:
        raise RuntimeError("no frozen Q154 manifest exists")
    manifest = json.loads(manifests[0].read_text())
    if manifest.get("rehearsal_panel_size") != 20:
        raise RuntimeError("Q154 manifest is not the bounded-panel protocol")
    fast = tuple(launcher.FAST_6_LABELS)
    slow = tuple(launcher.SLOW_4_LABELS)
    payload = {
        "schema": "semantic-v2-fast6-slow4-cohort-split-v5",
        "passed": True,
        "fast_cohort": list(fast),
        "slow_cohort": list(slow),
        "boundary_completed_rungs": 30,
        "boundary_label": "Q134",
        "from_policy": "priority-exposure-v1",
        "to_policy": "seeded-outcome-interleaved-exposure-v1",
        "panel_membership_policy": "exact-bank-order-expanding-round-robin-v1",
        "retention_order_policy": "same-seeded-permutation-before-and-after",
        "training_order_policy": "outcome-signature-interleave-with-exposure-first",
        "checkpoint_policy": (
            "schema-v4-atomic-with-game-and-optimizer-sub-iteration-cursors"
        ),
        "legacy_q104_resume_policy": (
            "normalize-only-protocol-neutral-default-spellings-with-resume-provenance"
        ),
        "launcher_status_ownership": (
            "recovery-status-isolated-before-gate-verification"
        ),
        "sharing": "strict-none",
        "bank_sha256": manifest["bank_sha256"],
        "bank_byte_sha256": _sha256(launcher.BANK),
        "resources": {
            "fast_parallel_workers": 6,
            "slow_serial_workers": 1,
            "maximum_total_experiment_workers": 7,
            "threads_per_worker": 1,
            "slow_rehearsal_segment_timeout_seconds": 7200,
            "slow_rehearsal_training_seconds_per_iteration_at_reference": 7200,
            "native_timeout_seconds_slow": 7200,
        },
        "fast_seed_indices": launcher.PRIMARY_8_SEED_INDEX,
        "fast_q134_boundaries": _fast_boundaries(fast),
        "predecessor_gate_sha256": (
            _sha256(PREDECESSOR_GATE) if PREDECESSOR_GATE.is_file() else None
        ),
        "slow_durable_boundaries": _slow_boundaries(slow),
        "source_sha256": {
            str(path): _sha256(path) for path in launcher.BOUNDED_REHEARSAL_SOURCES
        },
        "tests": {"command": "uv run pytest -q", "result": args.tests_result},
        "verified_at": datetime.now(UTC).isoformat(),
    }
    temporary = GATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, GATE)


if __name__ == "__main__":
    main()
