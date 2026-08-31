#!/usr/bin/env python3
"""Prepare isolated Fast-parent inputs for focused successor v1.

This is deliberately non-launching.  It creates a fresh focused bank and
hash-bound copies of the two terminal parent states.  Missing embedding and
Slow-selection prerequisites remain queued rather than being substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
RUN = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817")
POPULATION = RUN / "continuation/q4000-v1-population-20260818"
Q254 = POPULATION / "q254-fast6-20260823"
ROOT = POPULATION / "focused-successor-v1"
PROTOCOL = ROOT / "protocol"
GATE = ROOT / "FAST_PARENT_INPUTS_PREPARED.json"
POLICY = REPO / "research/local-q-skm-ablation/focused-successor-v1-policy.json"
Q50_POLICY = REPO / "research/local-q-skm-ablation/q50-4-updated-policy.json"
Q254_STAGE = "q50-3-updated-scheduled-no-sharing-bounded-repair-v2"

sys.path.insert(0, str(REPO / "scripts"))
import build_q50_4_updated as bank_builder  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def parent(label: str) -> dict[str, Any]:
    branch = Q254 / "branches" / label
    stage = branch / Q254_STAGE
    files = {name: stage / name for name in ("state.pt.gz", "report.json", "terminal-retention-audit.json")}
    if not (branch / "Q254_COMPLETE").is_file() or not all(path.is_file() for path in files.values()):
        raise RuntimeError(f"terminal parent is incomplete: {label}")
    return {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026083101)
    args = parser.parse_args()
    policy = json.loads(POLICY.read_text())
    if policy["schema"] != "focused-successor-v1-policy" or policy["legacy_q304"]["launch_authorized"]:
        raise RuntimeError("focused policy does not fail closed against legacy Q304")
    marker = Q254 / "ALL_FAST_6_LINEAGES_Q254_COMPLETE"
    if not marker.is_file():
        raise RuntimeError("Fast Q254 terminal marker is missing")

    audit = bank_builder.build(
        RUN / "inputs/q4000-v1",
        REPO / "research/local-q-skm-ablation/q44-2-updated/q44-2-updated.json",
        REPO / "research/mastery-v3-curriculum/curriculum.json",
        Q50_POLICY,
        Q254 / "protocol",
        PROTOCOL,
        seed=args.seed,
    )
    if audit.get("status") != "passed":
        raise RuntimeError("fresh focused bank audit failed")

    parents = {
        "q-grown-strand-graph-12": parent("q-grown-strand-graph-12"),
        "q-grown-raster-axial-12": parent("q-grown-raster-axial-12"),
    }
    strands = parents["q-grown-strand-graph-12"]["state.pt.gz"]
    raster = parents["q-grown-raster-axial-12"]["state.pt.gz"]
    lines = {
        "strand-graph-12-rl-control": (strands, 202608310101, "PREPARED"),
        "strand-graph-12-proof-distilled": (strands, 202608310102, "QUEUED"),
        "strand-graph-12-proof-embedding": (strands, 202608310103, "QUEUED"),
        "raster-axial-12-control": (raster, 202608310104, "PREPARED"),
    }
    fork_bindings = {}
    for line, (binding, seed, status) in lines.items():
        destination = ROOT / "parents" / line / "initial-state.pt.gz"
        source = Path(binding["path"])
        atomic_copy(source, destination)
        copied_hash = sha256(destination)
        if copied_hash != binding["sha256"]:
            raise RuntimeError(f"parent copy hash differs: {line}")
        fork_bindings[line] = {
            "status": status,
            "seed": seed,
            "initial_state": str(destination),
            "initial_state_sha256": copied_hash,
            "isolation": "copy-once; no state sharing after initialization",
        }

    proof_dataset = Path(policy["proof_supervision"]["dataset"])
    proof_ready = proof_dataset.is_file() and sha256(proof_dataset) == policy["proof_supervision"]["dataset_sha256"]
    gate = {
        "schema": "focused-successor-v1-fast-parent-inputs",
        "status": "PREPARED",
        "prepared_at": datetime.now(UTC).isoformat(),
        "launch_permitted": False,
        "detail": "parent inputs and focused bank are prepared; exact distillation/runtime launch gate remains required",
        "policy": {"path": str(POLICY), "sha256": sha256(POLICY)},
        "fast_terminal_marker": {"path": str(marker), "sha256": sha256(marker)},
        "parents": parents,
        "forks": fork_bindings,
        "proof_supervision": {
            "status": "PREPARED" if proof_ready else "BLOCKED",
            "dataset": str(proof_dataset),
            "sha256": sha256(proof_dataset) if proof_dataset.is_file() else None,
            "unknown_actions": "unknown-not-negative",
        },
        "embedding_child": {"status": "QUEUED", "reason": "terminal embedding selection manifest absent"},
        "diverse_slow_selection": {"status": "QUEUED", "reason": "Slow Q154 terminal exact-common audit absent"},
        "bank": {
            "path": str(PROTOCOL / "q50-4-updated.json"),
            "sha256": sha256(PROTOCOL / "q50-4-updated.json"),
            "prior_path": str(PROTOCOL / "prior-q254-for-q50-4-updated.json"),
            "prior_sha256": sha256(PROTOCOL / "prior-q254-for-q50-4-updated.json"),
            "seed": args.seed,
            "audit": audit,
        },
    }
    atomic_json(GATE, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
