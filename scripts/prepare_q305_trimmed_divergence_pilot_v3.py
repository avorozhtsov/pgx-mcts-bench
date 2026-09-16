#!/usr/bin/env python3
"""Prepare the fourth Fast descendant: trimmed divergence-local Q305 pilot."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime

import prepare_q305_same_root_tournament_pilot as base

PREVIOUS = base.ROOT / "Q305_SAME_ROOT_TOURNAMENT_PILOT_RECOVERY_V2_VERIFIED.json"
PROMOTION = base.ROOT / "q305-tournament-v1/Q305_Q314_EXACT_COMMON_PROMOTION_RECOVERY_V2.json"
GATE = base.ROOT / "Q305_TRIMMED_DIVERGENCE_PILOT_V3_VERIFIED.json"
OUTPUT = (
    base.ROOT
    / "q305-tournament-v1/branches"
    / "strand-graph-12-proof-distilled-trimmed-divergence/q305-q314-pilot-v3"
)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("trimmed-divergence Q305 output already exists")
    previous = json.loads(PREVIOUS.read_text())
    promotion = json.loads(PROMOTION.read_text())
    if promotion.get("passed") is not False or promotion.get("status") != "BLOCKED":
        raise RuntimeError("v3 requires the failed stable-loss promotion gate")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.REPO, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=base.REPO, text=True
    ).strip()
    if head != remote:
        raise RuntimeError("Q305 v3 requires pushed main")
    sources = [
        base.REPO / "scripts/run_q305_trimmed_divergence_pilot_v3.py",
        base.REPO / "scripts/run_q305_same_root_tournament_pilot.py",
        base.REPO / "scripts/prepare_q305_trimmed_divergence_pilot_v3.py",
        base.REPO / "src/pgx_mcts_bench/sv2_curriculum.py",
        base.REPO / "src/pgx_mcts_bench/trajectory_tournament.py",
        base.REPO / "src/pgx_mcts_bench/training.py",
        base.REPO / "src/pgx_mcts_bench/data.py",
        base.REPO / "tests/test_trajectory_tournament.py",
    ]
    payload = {
        **previous,
        "schema": "q305-same-root-trajectory-tournament-pilot-gate-v1",
        "status": "PREPARED",
        "commit": head,
        "verified_at": datetime.now(UTC).isoformat(),
        "scientist": "strand-graph-12-proof-distilled-trimmed-divergence-v3",
        "lineage_variant": "fourth-fast-descendant-trimmed-divergence-v3",
        "output": str(OUTPUT),
        "predecessor_gate": str(PREVIOUS),
        "predecessor_gate_sha256": base.sha256(PREVIOUS),
        "failed_promotion_gate": str(PROMOTION),
        "failed_promotion_gate_sha256": base.sha256(PROMOTION),
        "recovery": "trimmed-extremes-divergence-local-credit-v3",
        "trajectory_tournament_strategy": "trimmed-divergence-v3",
        "relative_trajectory_weight": 0.5,
        "relative_trajectory_sample_fraction": 0.25,
        "seed": 202609160305,
        "sources": {str(path): base.sha256(path) for path in sources},
    }
    temporary = GATE.with_name(f".{GATE.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(GATE)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
