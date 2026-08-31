#!/usr/bin/env python3
"""Build a repair-v2 Q254 gate with the seeded-to-seeded policy transition."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
sys.path.insert(0, str(REPO / "scripts"))

import prepare_local_q254_fast6 as base  # noqa: E402

_base_transform_runtime = base.transform_runtime

base.RUNTIME = (
    base.PROTOCOL / "runtime/pgx_mcts_bench/q254_sv2_curriculum_runtime_repair_v2.py"
)
base.GATE = base.ROOT / "FAST6_Q254_FIRST_BLOCK_SEEDED_REPAIR_V2_VERIFIED.json"
base.SOURCE_FILES = base.SOURCE_FILES + (
    REPO / "scripts/prepare_local_q254_fast6_repair_v2.py",
    REPO / "scripts/run_local_q254_fast6_repair_v2.py",
    REPO / "scripts/run_local_q254_fast6_branch_repair_v2.py",
    REPO / "tests/test_q254_fast6_repair_v2.py",
)


def transform_runtime(source: str) -> str:
    transformed = _base_transform_runtime(source)
    old = '"from_policy": "priority-exposure-v1",'
    new = '"from_policy": "seeded-outcome-interleaved-exposure-v1",'
    if transformed.count(old) != 1:
        raise RuntimeError("Q254 repair-v2 from-policy patch anchor count differs")
    return (
        "# Repair-v2: Q204 already uses seeded outcome-interleaved exposure.\n"
        + transformed.replace(old, new)
    )


base.transform_runtime = transform_runtime


if __name__ == "__main__":
    base.main()
