#!/usr/bin/env python3
"""Prepare the stable contrastive-loss Q305 pilot recovery gate."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime

import prepare_q305_same_root_tournament_pilot as base

PREVIOUS = base.GATE
PROMOTION = base.ROOT / "q305-tournament-v1/Q305_Q314_EXACT_COMMON_PROMOTION_V1.json"
GATE = base.ROOT / "Q305_SAME_ROOT_TOURNAMENT_PILOT_RECOVERY_V2_VERIFIED.json"
OUTPUT = (
    base.ROOT
    / "q305-tournament-v1/branches"
    / "strand-graph-12-proof-distilled-tournament/q305-q314-stable-loss-recovery-v2"
)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("stable-loss Q305 recovery output already exists")
    previous = json.loads(PREVIOUS.read_text())
    promotion = json.loads(PROMOTION.read_text())
    if promotion.get("passed") is not False:
        raise RuntimeError("stable-loss recovery requires the failed v1 promotion gate")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.REPO, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=base.REPO, text=True
    ).strip()
    if head != remote:
        raise RuntimeError("Q305 recovery requires pushed main")
    sources = [
        base.REPO / "scripts/run_q305_same_root_tournament_pilot_recovery_v2.py",
        base.REPO / "scripts/run_q305_same_root_tournament_pilot.py",
        base.REPO / "src/pgx_mcts_bench/sv2_curriculum.py",
        base.REPO / "src/pgx_mcts_bench/trajectory_tournament.py",
        base.REPO / "src/pgx_mcts_bench/training.py",
        base.REPO / "src/pgx_mcts_bench/data.py",
    ]
    payload = {
        **previous,
        "schema": "q305-same-root-trajectory-tournament-pilot-gate-v1",
        "status": "PREPARED",
        "commit": head,
        "verified_at": datetime.now(UTC).isoformat(),
        "output": str(OUTPUT),
        "predecessor_gate": str(PREVIOUS),
        "predecessor_gate_sha256": base.sha256(PREVIOUS),
        "failed_promotion_gate": str(PROMOTION),
        "failed_promotion_gate_sha256": base.sha256(PROMOTION),
        "recovery": "nonnegative-contrastive-policy-loss-v2",
        "sources": {str(path): base.sha256(path) for path in sources},
    }
    temporary = GATE.with_name(f".{GATE.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(GATE)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
