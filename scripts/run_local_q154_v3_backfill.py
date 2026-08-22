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
    raise RuntimeError(
        "superseded by scripts/run_local_q_slow4_continuation.py; "
        "the V3 lineages now share one serial cohort with the two combined-dual carries"
    )


if __name__ == "__main__":
    main()
