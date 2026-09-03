#!/usr/bin/env python3
"""Prepare a hash-bound recovery after the original adapter seed overflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = (
    REPO
    / "artifacts/local-q-skm-ablation-20260817/continuation"
    / "q4000-v1-population-20260818/focused-successor-v1"
)
PREVIOUS = ROOT / "FOCUSED_PROOF_EMBEDDING_CHILD_V1_VERIFIED.json"
GATE = ROOT / "FOCUSED_PROOF_EMBEDDING_CHILD_RECOVERY_V2_VERIFIED.json"
RUNNER = REPO / "scripts/run_focused_successor_embedding_child_recovery_v2.py"
DATA = ROOT / "branches/strand-graph-12-proof-embedding/bridge-v1/on-policy-data.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    previous = json.loads(PREVIOUS.read_text())
    if previous.get("schema") != "focused-proof-embedding-child-gate-v1":
        raise RuntimeError("unexpected predecessor embedding gate")
    if not DATA.is_file():
        raise RuntimeError("recovery requires the completed on-policy dataset")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=REPO, text=True
    ).strip()
    if head != remote:
        raise RuntimeError("recovery gate requires pushed main")
    sources = [
        RUNNER,
        REPO / "scripts/train_on_policy_embedding_value_adapter.py",
        REPO / "scripts/evaluate_embedding_value_adapter.py",
        REPO / "src/pgx_mcts_bench/embedding_value_adapter.py",
        REPO / "src/pgx_mcts_bench/on_policy_embedding_value.py",
        REPO / "src/pgx_mcts_bench/search.py",
    ]
    payload = {
        "schema": "focused-proof-embedding-child-recovery-gate-v2",
        "status": "PREPARED",
        "checkout": str(REPO),
        "branch": "main",
        "commit": head,
        "verified_at": datetime.now(UTC).isoformat(),
        "predecessor_gate": str(PREVIOUS),
        "predecessor_gate_sha256": sha256(PREVIOUS),
        "parent": previous["parent"],
        "embedding": previous["embedding"],
        "on_policy_data": str(DATA),
        "on_policy_data_sha256": sha256(DATA),
        "recovery": "reuse-complete-data-and-replace-out-of-range-seeds",
        "adapter_seed": 2609030306,
        "evaluation_seed": 2609030307,
        "sources": {str(path): sha256(path) for path in sources},
    }
    temporary = GATE.with_name(f".{GATE.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(GATE)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
