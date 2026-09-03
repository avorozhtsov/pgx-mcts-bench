#!/usr/bin/env python3
"""Build the exact gate for the Slow-Q154 SKM atomic-cell recovery."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime

import run_local_q154_slow4_skm_debt_recovery_v4 as base

REPO = base.REPO
ROOT = base.ROOT
OUTPUT = base.OUTPUT
GATE = ROOT / "SLOW4_Q154_SKM_ATOMIC_CELL_RECOVERY_V5_VERIFIED.json"
LAUNCHER = REPO / "scripts/run_local_q154_slow4_skm_atomic_cell_recovery_v5.py"
PREPARER = REPO / "scripts/prepare_local_q154_slow4_skm_atomic_cell_recovery_v5.py"
TEST = REPO / "tests/test_q154_slow4_skm_atomic_cell_recovery_v5.py"


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

    source_paths = [
        LAUNCHER,
        PREPARER,
        TEST,
        REPO / "scripts/run_local_q154_slow4_skm_debt_recovery_v4.py",
        REPO / "src/pgx_mcts_bench/cli.py",
        REPO / "src/pgx_mcts_bench/sv2_curriculum.py",
        REPO / "research/local-q-skm-ablation/EXECUTION-CONTRACT.md",
    ]
    input_paths = [
        ROOT / "SLOW4_Q154_SKM_DEBT_RECOVERY_V4_VERIFIED.json",
        ROOT / "protocol/q50-1-updated.json",
        ROOT / "protocol/prior-q104-for-q50-1-updated.json",
        OUTPUT / "state.pt.gz",
        OUTPUT / "phase-checkpoints/af6136425633ce70.pt.gz",
        OUTPUT / "q104-rehearsal-repair-v1/report.json",
        OUTPUT / "q104-rehearsal-repair-v1/state.pt.gz",
        REPO
        / "artifacts/nebius-main32-final-20260817/artifacts/"
        "q4000-strand12-20260814/migrated/"
        "raster-invariant-combined-dual-12/checkpoint.pt",
    ]
    for path in [*source_paths, *input_paths]:
        if not path.is_file():
            raise RuntimeError(f"required atomic-cell recovery input is absent: {path}")

    payload = {
        "schema": "slow4-q154-skm-atomic-cell-recovery-gate-v5",
        "passed": True,
        "sharing": "strict-none",
        "checkout": str(REPO),
        "branch": branch,
        "commit": head,
        "maximum_experiment_workers": 1,
        "recovery_lineage": "skm-v2-high-combined-dual",
        "durable_events": len(events),
        "durable_native_events": len(native_events),
        "scientist_task_timeout_seconds": 21600,
        "scientific_change": "none; wall-time repair for one atomic retention cell only",
        "preserved": [
            "bank order",
            "seed",
            "lineage-local state",
            "search and training doses",
            "rewards",
            "strict no sharing",
        ],
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
