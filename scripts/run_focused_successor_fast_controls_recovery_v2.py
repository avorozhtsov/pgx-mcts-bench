#!/usr/bin/env python3
"""Run focused Fast controls from corrected one-scientist carries."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
sys.path.insert(0, str(REPO / "scripts"))

import run_focused_successor_fast_controls as base  # noqa: E402

base.GATE = base.ROOT / "FOCUSED_FAST_CONTROLS_RECOVERY_V2_VERIFIED.json"
base.STATUS = base.ROOT / "fast-controls-launcher-status-recovery-v2.json"
base.LOCK = base.ROOT / "fast-controls-launcher-recovery-v2.lock"
base.STAGE = "q50-4-updated-scheduled-no-sharing-bounded-recovery-v2"
base.status = {
    "schema": "focused-successor-v1-fast-controls-launcher-recovery-v2",
    "state": "LAUNCHED",
    "pid": os.getpid(),
    "created_at": datetime.now(UTC).isoformat(),
    "maximum_experiment_workers": 2,
    "branches": {},
}


if __name__ == "__main__":
    base.main()
