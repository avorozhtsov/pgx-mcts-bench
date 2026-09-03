#!/usr/bin/env python3
"""Prepare a stage-disjoint mixed-outcome validation recovery."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = (
    REPO
    / "artifacts/local-q-skm-ablation-20260817/continuation"
    / "q4000-v1-population-20260818/focused-successor-v1"
)
PREVIOUS = ROOT / "FOCUSED_PROOF_EMBEDDING_CHILD_RECOVERY_V2_VERIFIED.json"
GATE = ROOT / "FOCUSED_PROOF_EMBEDDING_CHILD_RECOVERY_V3_VERIFIED.json"
BRANCH = ROOT / "branches/strand-graph-12-proof-embedding/bridge-v1"
SOURCE_DATA = BRANCH / "on-policy-data.pt"
DATA = BRANCH / "on-policy-data-stage-resplit-v3.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    previous = json.loads(PREVIOUS.read_text())
    saved = torch.load(SOURCE_DATA, map_location="cpu", weights_only=False)
    validation_stage = 4
    saved["episodes"] = [
        {
            **row,
            "split": "validation" if int(row["stage_index"]) == validation_stage else "train",
        }
        for row in saved["episodes"]
    ]
    stages = saved["data"]["stage_indexes"].tolist()
    saved["data"]["splits"] = [
        "validation" if int(stage) == validation_stage else "train" for stage in stages
    ]
    saved["protocol"] = {
        **saved["protocol"],
        "train_stages": [1, 6, 7, 11, 34, 36, 38],
        "validation_stages": [validation_stage],
        "recovery": "whole-stage-resplit-v3",
        "stage_disjoint_validation": True,
    }
    validation = [row for row in saved["episodes"] if row["split"] == "validation"]
    outcomes = {bool(row["solved"]) for row in validation}
    if outcomes != {False, True}:
        raise RuntimeError("recovered validation split is not mixed-outcome")
    temporary_data = DATA.with_name(f".{DATA.name}.part")
    torch.save(saved, temporary_data)
    temporary_data.replace(DATA)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=REPO, text=True
    ).strip()
    if head != remote:
        raise RuntimeError("embedding recovery v3 requires pushed main")
    sources = [
        REPO / "scripts/run_focused_successor_embedding_child_recovery_v2.py",
        REPO / "scripts/train_on_policy_embedding_value_adapter.py",
        REPO / "scripts/evaluate_embedding_value_adapter.py",
        REPO / "src/pgx_mcts_bench/embedding_value_adapter.py",
        REPO / "src/pgx_mcts_bench/on_policy_embedding_value.py",
        REPO / "src/pgx_mcts_bench/search.py",
    ]
    payload = {
        **previous,
        "schema": "focused-proof-embedding-child-recovery-gate-v3",
        "commit": head,
        "verified_at": datetime.now(UTC).isoformat(),
        "predecessor_gate": str(PREVIOUS),
        "predecessor_gate_sha256": sha256(PREVIOUS),
        "on_policy_data": str(DATA),
        "on_policy_data_sha256": sha256(DATA),
        "source_on_policy_data": str(SOURCE_DATA),
        "source_on_policy_data_sha256": sha256(SOURCE_DATA),
        "recovery": "stage-disjoint-mixed-outcome-validation-resplit-v3",
        "validation_episode_counts": {
            "successful": sum(bool(row["solved"]) for row in validation),
            "failed": sum(not bool(row["solved"]) for row in validation),
        },
        "sources": {str(path): sha256(path) for path in sources},
    }
    temporary = GATE.with_name(f".{GATE.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(GATE)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
