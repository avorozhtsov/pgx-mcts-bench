#!/usr/bin/env python3
"""Bind exact inputs and sources for focused strand-graph proof distillation."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import run_focused_successor_proof_distilled as run


def main() -> None:
    controls = json.loads((run.ROOT / "FOCUSED_FAST_CONTROLS_RECOVERY_V3_VERIFIED.json").read_text())
    inputs = json.loads((run.ROOT / "FAST_PARENT_INPUTS_PREPARED.json").read_text())
    if controls.get("passed") is not True or inputs.get("proof_supervision", {}).get("status") != "PREPARED":
        raise RuntimeError("focused controls or proof supervision are not prepared")
    value = {
        "schema": "focused-proof-distillation-launch-gate-v1",
        "status": "PREPARED",
        "launch_permitted": True,
        "verified_at": datetime.now(UTC).isoformat(),
        "parent": str(run.PARENT), "parent_sha256": run.sha256(run.PARENT),
        "dataset": str(run.DATASET), "dataset_sha256": run.sha256(run.DATASET),
        "seed": run.SEED,
        "accepted_actions": "set-valued-currently-best-replayed",
        "absent_actions": "unknown-not-negative",
        "worse_completed_routes": "comparisons-only",
        "controls_gate": str(run.ROOT / "FOCUSED_FAST_CONTROLS_RECOVERY_V3_VERIFIED.json"),
        "controls_gate_sha256": run.sha256(run.ROOT / "FOCUSED_FAST_CONTROLS_RECOVERY_V3_VERIFIED.json"),
        "source_sha256": {
            str(run.REPO / "scripts/run_focused_successor_proof_distilled.py"): run.sha256(run.REPO / "scripts/run_focused_successor_proof_distilled.py"),
            str(run.REPO / "scripts/prepare_focused_proof_distillation_gate.py"): run.sha256(run.REPO / "scripts/prepare_focused_proof_distillation_gate.py"),
            str(run.TRAINER): run.sha256(run.TRAINER),
            str(run.PGX_BASE / "src/pgx_mcts_bench/proof_guidance.py"): run.sha256(run.PGX_BASE / "src/pgx_mcts_bench/proof_guidance.py"),
            str(run.REPO / "src/pgx_mcts_bench/networks.py"): run.sha256(run.REPO / "src/pgx_mcts_bench/networks.py"),
            str(run.REPO / "tests/test_focused_successor_proof_distilled.py"): run.sha256(run.REPO / "tests/test_focused_successor_proof_distilled.py"),
        },
    }
    run.atomic_json(run.PREPARED, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
