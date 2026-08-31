#!/usr/bin/env python3
"""Bind the two immediately runnable focused-successor Fast controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
POPULATION = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818"
)
ROOT = POPULATION / "focused-successor-v1"
INPUTS = ROOT / "FAST_PARENT_INPUTS_PREPARED.json"
GATE = ROOT / "FOCUSED_FAST_CONTROLS_VERIFIED.json"
POLICY = REPO / "research/local-q-skm-ablation/focused-successor-v1-policy.json"
Q254_GATE = POPULATION / "q254-fast6-20260823/FAST6_Q254_FIRST_BLOCK_SEEDED_REPAIR_V2_VERIFIED.json"

sys.path.insert(0, str(REPO / "src"))
from pgx_mcts_bench.collaborative_scientists import _json_hash  # noqa: E402

SOURCES = (
    POLICY,
    REPO / "scripts/prepare_focused_successor_fast_inputs.py",
    REPO / "scripts/prepare_focused_successor_fast_controls.py",
    REPO / "scripts/run_focused_successor_branch.py",
    REPO / "scripts/run_focused_successor_fast_controls.py",
    REPO / "tests/test_focused_successor_fast_controls.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-count", type=int, required=True)
    args = parser.parse_args()
    prepared = json.loads(INPUTS.read_text())
    policy = json.loads(POLICY.read_text())
    inherited = json.loads(Q254_GATE.read_text())
    if prepared.get("schema") != "focused-successor-v1-fast-parent-inputs" or prepared.get("status") != "PREPARED":
        raise RuntimeError("focused Fast parent input gate is not PREPARED")
    if policy.get("schema") != "focused-successor-v1-policy" or policy["legacy_q304"]["launch_authorized"]:
        raise RuntimeError("focused policy does not fail closed")
    lines = {
        "strand-graph-12-rl-control": {"scientist": "strand-graph-12"},
        "raster-axial-12-control": {"scientist": "raster-axial-12"},
    }
    for line, binding in lines.items():
        fork = prepared["forks"][line]
        state = Path(fork["initial_state"])
        if fork["status"] != "PREPARED" or sha256(state) != fork["initial_state_sha256"]:
            raise RuntimeError(f"focused parent binding changed: {line}")
        binding.update(
            seed=int(fork["seed"]),
            initial_state=str(state),
            initial_state_sha256=fork["initial_state_sha256"],
        )
    bank = Path(prepared["bank"]["path"])
    prior = Path(prepared["bank"]["prior_path"])
    runtime = Path(inherited["runtime"])
    gate = {
        # The isolated Q254 runtime deliberately accepts this schema for a
        # boundary-zero seeded-to-seeded continuation.
        "schema": "semantic-v2-q254-first-block-seeded-order-v1",
        "passed": True,
        "verified_at": datetime.now(UTC).isoformat(),
        "test_count": args.test_count,
        "cohort": "focused-successor-v1-fast-controls",
        "focused_policy": str(POLICY),
        "focused_policy_sha256": sha256(POLICY),
        "fast_parent_inputs": str(INPUTS),
        "fast_parent_inputs_sha256": sha256(INPUTS),
        "focused_lines": lines,
        "boundary_completed_rungs": 0,
        "from_policy": "seeded-outcome-interleaved-exposure-v1",
        "to_policy": "seeded-outcome-interleaved-exposure-v1",
        "first_rehearsal_block_seeded": True,
        "sharing": "strict-none",
        "bank": str(bank),
        "bank_byte_sha256": sha256(bank),
        "bank_sha256": _json_hash(json.loads(bank.read_text())),
        "prior_bank": str(prior),
        "prior_bank_byte_sha256": sha256(prior),
        "runtime": str(runtime),
        "runtime_sha256": sha256(runtime),
        "inherited_runtime_gate": str(Q254_GATE),
        "inherited_runtime_gate_sha256": sha256(Q254_GATE),
        "source_sha256": {str(path): sha256(path) for path in SOURCES},
        "protocol": {
            "arm": "scheduled-no-sharing",
            "rehearsal_panel_size": 20,
            "selfplay_games": 4,
            "optimizer_steps": 24,
            "ratios": [10, 1000],
            "terminal_full_retention_audit": True,
            "maximum_experiment_workers": 2,
        },
        "proof_distilled": "QUEUED",
        "proof_embedding": "QUEUED",
        "diverse_slow_selection": "QUEUED",
    }
    atomic_json(GATE, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
