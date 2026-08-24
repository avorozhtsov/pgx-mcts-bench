#!/usr/bin/env python3
"""Launch six independent Q254 repair-v2 continuations."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
sys.path.insert(0, str(REPO / "scripts"))

import run_local_q254_fast6 as base  # noqa: E402

_base_branch_command = base.branch_command

base.GATE = base.ROOT / "FAST6_Q254_FIRST_BLOCK_SEEDED_REPAIR_V2_VERIFIED.json"
base.STATUS = base.ROOT / "launcher-status-repair-v2.json"
base.LOCK = base.ROOT / "launcher-repair-v2.lock"
base.STAGE = "q50-3-updated-scheduled-no-sharing-bounded-repair-v2"
base._status = {
    "schema": "q254-fast-6-population-launcher-repair-v2",
    "cohort": "fast-6",
    "state": "LAUNCHED",
    "stage": "Q254 repair-v2 dispatch",
    "pid": os.getpid(),
    "created_at": datetime.now(UTC).isoformat(),
    "maximum_experiment_workers": 6,
    "branches": {},
}


def branch_command(label: str, scientist: str, state: Path) -> list[str]:
    command = _base_branch_command(label, scientist, state)
    command[1] = str(REPO / "scripts/run_local_q254_fast6_branch_repair_v2.py")
    return command


base.branch_command = branch_command


if __name__ == "__main__":
    base.main()
