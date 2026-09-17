#!/usr/bin/env python3
"""Bind roster-name recovery for the fourth Fast Q305 descendant."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime

import prepare_q305_same_root_tournament_pilot as base
import prepare_q305_trimmed_divergence_pilot_v3 as v3

FAILED_GATE = v3.GATE
FAILED_STATUS = base.ROOT / "q305-tournament-v1/q305-pilot-launcher-status-v3.json"
FAILED_MANIFEST = v3.OUTPUT / "manifest.json"
GATE = base.ROOT / "Q305_TRIMMED_DIVERGENCE_RECOVERY_V4_VERIFIED.json"
OUTPUT = (
    base.ROOT
    / "q305-tournament-v1/branches"
    / "strand-graph-12-proof-distilled-trimmed-divergence/q305-q314-recovery-v4"
)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("trimmed-divergence recovery output already exists")
    failed_gate = json.loads(FAILED_GATE.read_text())
    failed_status = json.loads(FAILED_STATUS.read_text())
    failed_manifest = json.loads(FAILED_MANIFEST.read_text())
    if failed_status.get("state") != "BLOCKED" or set(failed_manifest.get("checkpoints", {})) != {
        "strand-graph-12-proof-distilled-trimmed-divergence-v3"
    }:
        raise RuntimeError("recovery requires the roster-name-only v3 failure")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.REPO, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=base.REPO, text=True
    ).strip()
    if head != remote:
        raise RuntimeError("Q305 v4 requires pushed main")
    sources = [
        base.REPO / "scripts/run_q305_trimmed_divergence_recovery_v4.py",
        base.REPO / "scripts/run_q305_same_root_tournament_pilot.py",
        base.REPO / "scripts/prepare_q305_trimmed_divergence_recovery_v4.py",
        base.REPO / "src/pgx_mcts_bench/sv2_curriculum.py",
        base.REPO / "src/pgx_mcts_bench/trajectory_tournament.py",
        base.REPO / "src/pgx_mcts_bench/training.py",
        base.REPO / "src/pgx_mcts_bench/data.py",
        base.REPO / "tests/test_trajectory_tournament.py",
    ]
    payload = {
        **failed_gate,
        "status": "PREPARED",
        "commit": head,
        "verified_at": datetime.now(UTC).isoformat(),
        "scientist": "strand-graph-12",
        "output": str(OUTPUT),
        "recovery": "roster-name-binding-only-v4",
        "failed_gate": str(FAILED_GATE),
        "failed_gate_sha256": base.sha256(FAILED_GATE),
        "failed_status": str(FAILED_STATUS),
        "failed_status_sha256": base.sha256(FAILED_STATUS),
        "failed_manifest": str(FAILED_MANIFEST),
        "failed_manifest_sha256": base.sha256(FAILED_MANIFEST),
        "sources": {str(path): base.sha256(path) for path in sources},
    }
    temporary = GATE.with_name(f".{GATE.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(GATE)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
