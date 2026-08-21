#!/usr/bin/env python3
"""Run the deferred two-lineage V3 Q154 backfill in a separate root."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
SOURCE = REPO / "scripts/run_local_q154_updated_continuation.py"


def _launcher_module():
    spec = importlib.util.spec_from_file_location("q154_primary_launcher", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Q154 launcher: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    launcher = _launcher_module()
    launcher.BRANCHES = (
        ("cyclic-memory-deep-v3", "cyclic-memory-deep-v3", 40, False),
        ("cyclic-graph-dual-v3", "cyclic-graph-dual-v3", 40, False),
    )
    launcher.Q104_MARKER = launcher.Q104_ROOT / "V3_BACKFILL_Q104_COMPLETE.json"
    launcher.Q104_MARKER_SCHEMA = "q104-v3-backfill-completion-v1"
    launcher.Q104_STATUS = launcher.Q104_ROOT / "v3-backfill-launcher-status.json"
    launcher.ROOT = launcher.POPULATION / "q154-v3-backfill-20260821"
    launcher.STATUS = launcher.ROOT / "launcher-status.json"
    launcher.LOCK = launcher.ROOT / "launcher.lock"
    launcher.BANK = launcher.ROOT / "protocol/q50-1-updated.json"
    launcher.PRIOR = launcher.ROOT / "protocol/prior-q104-for-q50-1-updated.json"
    launcher.BUILD_AUDIT = launcher.ROOT / "protocol/q50-1-updated-audit.json"
    launcher.REHEARSAL_DEBT = launcher.ROOT / "protocol/q104-rehearsal-debt.json"
    launcher.COMPLETION_MARKER = launcher.ROOT / "V3_BACKFILL_Q154_COMPLETE"
    launcher._status.update(
        {
            "schema": "q154-v3-backfill-launcher-v1",
            "cohort": "deferred-v3-backfill",
            "status_file": str(launcher.STATUS),
            "completion_marker": str(launcher.COMPLETION_MARKER),
        }
    )
    launcher.main()


if __name__ == "__main__":
    main()
