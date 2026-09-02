#!/usr/bin/env python3
"""Build the exact gate for the isolated Slow-Q154 SKM recovery."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench")
OLD_REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
ROOT = REPO / (
    "artifacts/local-q-skm-ablation-20260817/continuation/"
    "q4000-v1-population-20260818/q154-slow4-20260822"
)
GATE = ROOT / "SLOW4_Q154_SKM_DEBT_RECOVERY_V4_VERIFIED.json"
LAUNCHER = REPO / "scripts/run_local_q154_slow4_skm_debt_recovery_v4.py"
PREPARER = REPO / "scripts/prepare_local_q154_slow4_skm_debt_recovery_v4.py"
TEST = REPO / "tests/test_q154_slow4_skm_debt_recovery_v4.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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

    parity_files = ("src/pgx_mcts_bench/cli.py",)
    for relative in parity_files:
        if sha256(REPO / relative) != sha256(OLD_REPO / relative):
            raise RuntimeError(f"runtime source parity differs: {relative}")
    curriculum = REPO / "src/pgx_mcts_bench/sv2_curriculum.py"
    frozen_curriculum = OLD_REPO / "src/pgx_mcts_bench/sv2_curriculum.py"

    qgrown = ROOT / (
        "branches/q-grown-raster-invariant-combined-dual-12/"
        "q50-1-updated-scheduled-no-sharing-bounded/q104-rehearsal-repair-v1"
    )
    skm = ROOT / (
        "branches/skm-v2-high-combined-dual/"
        "q50-1-updated-scheduled-no-sharing-bounded/q104-rehearsal-repair-v1"
    )
    source_paths = [
        LAUNCHER,
        PREPARER,
        TEST,
        REPO / "research/local-q-skm-ablation/EXECUTION-CONTRACT.md",
        curriculum,
        *(REPO / relative for relative in parity_files),
    ]
    input_paths = [
        ROOT / "protocol/q50-1-updated.json",
        ROOT / "protocol/prior-q104-for-q50-1-updated.json",
        ROOT / "protocol/q50-1-updated-audit.json",
        ROOT / "protocol/q104-rehearsal-debt.json",
        ROOT / "SLOW4_Q154_CONTRACT_TRANSITION_RECOVERY_V3_VERIFIED.json",
        frozen_curriculum,
        qgrown / "state.pt.gz",
        qgrown / "phase-checkpoint.pt.gz",
        skm / "state.pt.gz",
        skm / "phase-checkpoint.pt.gz",
        ROOT
        / "initial-q104-states/skm-v2-high-combined-dual/"
        "raster-invariant-combined-dual-12/state.pt.gz",
        REPO
        / "artifacts/nebius-main32-final-20260817/artifacts/"
        "q4000-strand12-20260814/migrated/"
        "raster-invariant-combined-dual-12/checkpoint.pt",
    ]
    for path in [*source_paths, *input_paths]:
        if not path.is_file():
            raise RuntimeError(f"required recovery input is absent: {path}")

    payload = {
        "schema": "slow4-q154-skm-debt-recovery-gate-v4",
        "passed": True,
        "sharing": "strict-none",
        "checkout": str(REPO),
        "branch": branch,
        "commit": head,
        "maximum_experiment_workers": 1,
        "blocked_peer": "q-grown-raster-invariant-combined-dual-12",
        "blocked_peer_reason": "rehearsal debt repair exhausted its cumulative cap",
        "recovery_lineage": "skm-v2-high-combined-dual",
        "scientific_change": "none; repair bookkeeping accepts a final partial debt chunk",
        "runtime_compatibility_change": (
            "negative repair rounds keep their exact positive chunk size and do not feed "
            "that size into the adaptive F_old level controller"
        ),
        "source_hashes": {str(path): sha256(path) for path in source_paths},
        "input_hashes": {str(path): sha256(path) for path in input_paths},
        "verified_at": datetime.now(UTC).isoformat(),
    }
    temporary = GATE.with_suffix(GATE.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, GATE)
    print(sha256(GATE), GATE)


if __name__ == "__main__":
    main()
