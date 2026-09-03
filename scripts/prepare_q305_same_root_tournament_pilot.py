#!/usr/bin/env python3
"""Prepare the isolated Q305--Q314 same-root trajectory tournament pilot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pgx_mcts_bench.sv2_curriculum import _load_state, _save_state

REPO = Path(__file__).resolve().parents[1]
ROOT = (
    REPO
    / "artifacts/local-q-skm-ablation-20260817/continuation"
    / "q4000-v1-population-20260818/focused-successor-v1"
)
PROTOCOL = ROOT / "q305-tournament-v1/protocol"
PARENT_ROOT = (
    ROOT
    / "branches/strand-graph-12-proof-distilled"
    / "q50-4-updated-scheduled-no-sharing-bounded-proof-recovery-v2"
)
CHECKPOINT = ROOT / "proof-distillation/strand-graph-q254-model/checkpoint.pt"
STATE = PARENT_ROOT / "state.pt.gz"
CARRY = ROOT / "q305-tournament-v1/initial-state/strand-graph-12/state.pt.gz"
PARENT_MANIFEST = PARENT_ROOT / "manifest.json"
BANK = PROTOCOL / "q50-5-tournament.json"
PRIOR = PROTOCOL / "prior-q304-for-q50-5-tournament.json"
AUDIT = PROTOCOL / "q50-5-tournament-audit.json"
GATE = ROOT / "Q305_SAME_ROOT_TOURNAMENT_PILOT_V1_VERIFIED.json"
OUTPUT = (
    ROOT
    / "q305-tournament-v1/branches"
    / "strand-graph-12-proof-distilled-tournament/q305-q314-pilot-v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("Q305 tournament output already exists")
    audit = json.loads(AUDIT.read_text())
    if audit.get("status") != "passed":
        raise RuntimeError("Q305 bank audit did not pass")
    parent = json.loads(PARENT_MANIFEST.read_text())
    if int(parent.get("completed_rungs", -1)) != 50:
        raise RuntimeError("proof-distilled Q304 parent is not terminal")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=REPO, text=True
    ).strip()
    if head != remote:
        raise RuntimeError("Q305 gate requires pushed main")
    source_state_sha256 = sha256(STATE)
    if CARRY.exists():
        carry = _load_state(CARRY)
        if carry.get("source_q304_state_sha256") != source_state_sha256:
            raise RuntimeError("existing Q305 carry comes from a different parent")
    else:
        source = _load_state(STATE)
        scientist = "strand-graph-12"
        _save_state(
            CARRY,
            {
                "schema": "q304-to-q305-same-root-tournament-carry-v1",
                "source_q304_state_sha256": source_state_sha256,
                "scientist": source["scientists"][scientist],
                "f_old": int(source["f_old"][scientist]),
                "rehearsal_exposure": source["rehearsal_exposure"][scientist],
                "rehearsal_panel_cursor": int(source["rehearsal_panel_cursor"][scientist]),
                "f_native": 1,
                "simulations": 40,
                "donation_dose": int(source.get("donation_dose", 1)),
                "donation_healthy_streak": int(source.get("donation_healthy_streak", 0)),
                "controller_reset": {
                    "f_native": 1,
                    "simulations": 40,
                    "reason": "registered Q305 tournament compute protocol",
                },
            },
        )
    sources = [
        REPO / "scripts/run_q305_same_root_tournament_pilot.py",
        REPO / "src/pgx_mcts_bench/sv2_curriculum.py",
        REPO / "src/pgx_mcts_bench/trajectory_tournament.py",
        REPO / "src/pgx_mcts_bench/training.py",
        REPO / "src/pgx_mcts_bench/data.py",
    ]
    payload = {
        "schema": "q305-same-root-trajectory-tournament-pilot-gate-v1",
        "status": "PREPARED",
        "checkout": str(REPO),
        "branch": "main",
        "commit": head,
        "verified_at": datetime.now(UTC).isoformat(),
        "scientist": "strand-graph-12",
        "lineage_variant": "proof-distilled-same-root-tournament-v1",
        "parent_checkpoint": str(CHECKPOINT),
        "parent_checkpoint_sha256": sha256(CHECKPOINT),
        "parent_state": str(CARRY),
        "parent_state_sha256": sha256(CARRY),
        "source_parent_state": str(STATE),
        "source_parent_state_sha256": source_state_sha256,
        "parent_manifest": str(PARENT_MANIFEST),
        "parent_manifest_sha256": sha256(PARENT_MANIFEST),
        "bank": str(BANK),
        "bank_sha256": sha256(BANK),
        "prior_bank": str(PRIOR),
        "prior_bank_sha256": sha256(PRIOR),
        "bank_audit": str(AUDIT),
        "bank_audit_sha256": sha256(AUDIT),
        "output": str(OUTPUT),
        "pilot_rows": [305, 314],
        "trajectory_tournament": {
            "size_per_objective": 10,
            "relative_weight": 1.0,
            "invalid_replay_policy": "ignored-not-negative",
            "relative_mass": "zero-sum-equal-episode",
        },
        "compute": {
            "ratios": [10.0, 1000.0],
            "simulations": 40,
            "F_native": 1,
            "train_steps": 24,
            "evaluation_attempts": 2,
            "rehearsal_panel_size": 10,
        },
        "sources": {str(path): sha256(path) for path in sources},
        "no_duplicate_writer_checked": True,
    }
    temporary = GATE.with_name(f".{GATE.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(GATE)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
