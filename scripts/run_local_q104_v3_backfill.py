#!/usr/bin/env python3
"""Resume only the two deferred V3 Q104 branches as a separate cohort."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
SOURCE = REPO / "scripts/run_local_q104_updated_continuation.py"


def _launcher_module():
    spec = importlib.util.spec_from_file_location("q104_population_launcher", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Q104 launcher: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    launcher = _launcher_module()
    launcher.BRANCHES = launcher.BRANCHES[:2]
    launcher.MAX_EXPERIMENT_CORES = 1
    launcher.STATUS = launcher.ROOT / "v3-backfill-launcher-status.json"
    launcher.LOCK = launcher.ROOT / "v3-backfill-launcher.lock"
    launcher.COMPLETION_MARKER = launcher.ROOT / "V3_BACKFILL_Q104_COMPLETE.json"
    launcher.COMPLETION_SCHEMA = "q104-v3-backfill-completion-v1"
    launcher._status.update(
        {
            "schema": "q104-v3-backfill-launcher-v1",
            "cohort": "deferred-v3-backfill",
            "status_file": str(launcher.STATUS),
            "completion_marker": str(launcher.COMPLETION_MARKER),
            "max_experiment_cores": 1,
        }
    )
    launcher.main()


if __name__ == "__main__":
    main()
