#!/usr/bin/env python3
"""Run one Q254 repair-v2 branch using the newly hash-bound isolated runtime."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
sys.path.insert(0, str(REPO / "scripts"))

import run_local_q254_fast6_branch as base  # noqa: E402


if __name__ == "__main__":
    base.main()
