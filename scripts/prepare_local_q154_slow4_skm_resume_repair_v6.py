#!/usr/bin/env python3
"""Build the exact gate for the Slow-Q154 SKM resume repair."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime

import run_local_q154_slow4_skm_debt_recovery_v4 as base

REPO = base.REPO
ROOT = base.ROOT
OUTPUT = base.OUTPUT
GATE = ROOT / "SLOW4_Q154_SKM_RESUME_REPAIR_V6_VERIFIED.json"
TRANSITION = ROOT / "SLOW4_Q154_SKM_TIMEOUT_EXTENSION_V6.json"
LAUNCHER = REPO / "scripts/run_local_q154_slow4_skm_resume_repair_v6.py"
PREPARER = REPO / "scripts/prepare_local_q154_slow4_skm_resume_repair_v6.py"
TEST = REPO / "tests/test_q154_slow4_skm_resume_repair_v6.py"


def main() -> None:
    branch = subprocess.check_output(
        ["git", "-C", str(REPO), "branch", "--show-current"], text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    origin = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "origin/main"], text=True
    ).strip()
    if branch != "main" or head != origin:
        raise RuntimeError("main is not aligned with origin/main")

    events = sorted((OUTPUT / "events").glob("*.json"))
    native_events = sorted((OUTPUT / "native-events").glob("*.json"))
    if len(events) != 29 or len(native_events) != 30:
        raise RuntimeError("unexpected Q134 boundary event counts")
    repair_report_path = OUTPUT / "q104-rehearsal-repair-v1/report.json"
    repair_report = json.loads(repair_report_path.read_text())
    if repair_report.get("completed_iterations") != 38:
        raise RuntimeError("completed repair report is absent")

    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    transition = {
        "schema": "semantic-v2-timeout-extension-v1",
        "passed": True,
        "output": str(OUTPUT.resolve()),
        "frozen_protocol_sha256": manifest.get("protocol_sha256"),
        "old_timeout_seconds": 7200,
        "new_timeout_seconds": 21600,
        "allowed_protocol_fields": [
            "scientist_task_timeout_seconds",
            "rehearsal_segment_timeout_seconds",
        ],
        "reason": "atomic cell and completed-debt resume repair",
        "commit": head,
        "verified_at": datetime.now(UTC).isoformat(),
    }
    transition_temporary = TRANSITION.with_suffix(TRANSITION.suffix + ".tmp")
    transition_temporary.write_text(json.dumps(transition, indent=2, sort_keys=True) + "\n")
    os.replace(transition_temporary, TRANSITION)

    source_paths = [
        LAUNCHER,
        PREPARER,
        TEST,
        REPO / "scripts/run_local_q154_slow4_skm_atomic_cell_recovery_v5.py",
        REPO / "scripts/run_local_q154_slow4_skm_debt_recovery_v4.py",
        REPO / "src/pgx_mcts_bench/cli.py",
        REPO / "src/pgx_mcts_bench/sv2_curriculum.py",
        REPO / "research/local-q-skm-ablation/EXECUTION-CONTRACT.md",
    ]
    input_paths = [
        ROOT / "SLOW4_Q154_SKM_ATOMIC_CELL_RECOVERY_V5_VERIFIED.json",
        TRANSITION,
        manifest_path,
        ROOT / "protocol/q50-1-updated.json",
        ROOT / "protocol/prior-q104-for-q50-1-updated.json",
        OUTPUT / "state.pt.gz",
        OUTPUT / "phase-checkpoints/af6136425633ce70.pt.gz",
        repair_report_path,
        REPO
        / "artifacts/nebius-main32-final-20260817/artifacts/"
        "q4000-strand12-20260814/migrated/"
        "raster-invariant-combined-dual-12/checkpoint.pt",
    ]
    for path in [*source_paths, *input_paths]:
        if not path.is_file():
            raise RuntimeError(f"required resume-repair input is absent: {path}")

    payload = {
        "schema": "slow4-q154-skm-resume-repair-gate-v6",
        "passed": True,
        "sharing": "strict-none",
        "checkout": str(REPO),
        "branch": branch,
        "commit": head,
        "maximum_experiment_workers": 1,
        "recovery_lineage": "skm-v2-high-combined-dual",
        "durable_events": len(events),
        "durable_native_events": len(native_events),
        "completed_rehearsal_debt": 38,
        "scientist_task_timeout_seconds": 21600,
        "scientific_change": "none; skip already-completed repair debt on resume",
        "source_hashes": {str(path): base._sha256(path) for path in source_paths},
        "input_hashes": {str(path): base._sha256(path) for path in input_paths},
        "verified_at": datetime.now(UTC).isoformat(),
    }
    temporary = GATE.with_suffix(GATE.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, GATE)
    print(base._sha256(GATE), GATE)


if __name__ == "__main__":
    main()
