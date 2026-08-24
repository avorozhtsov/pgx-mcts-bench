#!/usr/bin/env python3
"""Bind the lightweight DKT-disjoint shadow gate to terminal Q204 raster state."""

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
RUN = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817")
POPULATION = RUN / "continuation/q4000-v1-population-20260818"
Q204_ROOT = POPULATION / "q204-fast6-20260822"
Q204_STAGE = "q50-2-updated-scheduled-no-sharing-bounded"
BRANCH = Q204_ROOT / "branches/q-grown-raster-axial-12"
TERMINAL = BRANCH / Q204_STAGE
ROOT = POPULATION / "dkt-shadow4-raster-axial-q204-20260823"
PROTOCOL = ROOT / "protocol"
GATE = ROOT / "DKT_SHADOW4_RASTER_AXIAL_PREPARED.json"
POLICY = REPO / "research/local-q-skm-ablation/dkt-shadow4-raster-axial-policy.json"
Q204_GATE = Q204_ROOT / "FAST6_Q204_FIRST_BLOCK_SEEDED_VERIFIED.json"
CHECKPOINT = (
    Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts")
    / "nebius-main32-final-20260817/artifacts/q4000-strand12-20260814"
    / "migrated/raster-axial-12/checkpoint.pt"
)

sys.path.insert(0, str(REPO / "scripts"))
import build_dkt_shadow4_raster_axial as builder  # noqa: E402

SOURCE_FILES = (
    REPO / "scripts/build_dkt_shadow4_raster_axial.py",
    REPO / "scripts/prepare_dkt_shadow4_raster_axial.py",
    REPO / "scripts/run_dkt_shadow4_raster_axial.py",
    REPO / "tests/test_dkt_shadow4_raster_axial.py",
    POLICY,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def terminal_bindings() -> dict[str, Any]:
    marker = BRANCH / "Q204_COMPLETE"
    report = TERMINAL / "report.json"
    audit = TERMINAL / "terminal-retention-audit.json"
    state = TERMINAL / "state.pt.gz"
    if not marker.is_file():
        raise RuntimeError("q-grown-raster-axial-12 Q204 terminal marker is missing")
    payload = json.loads(report.read_text())
    if payload.get("completed_rungs") != 50 or not payload.get("terminal_retention_audit"):
        raise RuntimeError("q-grown-raster-axial-12 Q204 report is not terminal")
    return {
        "marker": str(marker),
        "marker_sha256": sha256(marker),
        "report": str(report),
        "report_sha256": sha256(report),
        "terminal_audit": str(audit),
        "terminal_audit_sha256": sha256(audit),
        "state": str(state),
        "state_sha256": sha256(state),
    }


def prepare(test_count: int) -> dict[str, Any]:
    audit = builder.build(POLICY, PROTOCOL)
    if audit.get("status") != "passed":
        raise RuntimeError("DKT shadow bank audit did not pass")
    q204_gate = json.loads(Q204_GATE.read_text())
    if not q204_gate.get("passed"):
        raise RuntimeError("Q204 source gate did not pass")
    runtime = Path(q204_gate["runtime"])
    if sha256(runtime) != q204_gate["runtime_sha256"]:
        raise RuntimeError("Q204 isolated runtime hash changed")
    bank = PROTOCOL / "dkt-disjoint-shadow4.json"
    selection_audit = PROTOCOL / "dkt-disjoint-shadow4-audit.json"
    bindings = terminal_bindings()
    gate = {
        "schema": "dkt-disjoint-shadow4-raster-axial-prepared-v1",
        "status": "PREPARED",
        "prepared": True,
        "prepared_at": datetime.now(UTC).isoformat(),
        "test_count": int(test_count),
        "lineage": "q-grown-raster-axial-12",
        "scientist": "raster-axial-12",
        "representative_scope": "fast-6 group readiness proxy only",
        "policy": str(POLICY),
        "policy_sha256": sha256(POLICY),
        "bank": str(bank),
        "bank_byte_sha256": sha256(bank),
        "selection_audit": str(selection_audit),
        "selection_audit_sha256": sha256(selection_audit),
        "q204_gate": str(Q204_GATE),
        "q204_gate_sha256": sha256(Q204_GATE),
        "runtime": str(runtime),
        "runtime_sha256": sha256(runtime),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "terminal": bindings,
        "protocol": {
            "learning": False,
            "sharing": False,
            "objective_ratio": 10.0,
            "examples": 4,
            "attempts_per_example": 4,
            "total_attempts": 16,
            "simulations": 256,
            "action_horizon": 128,
            "root_noise": True,
            "temperature": 0.0,
            "base_seed": 2026082304,
            "pass_rule": "at least one of four examples has replay-verified crossing_changes <= registered upper bound",
        },
        "source_sha256": {str(path): sha256(path) for path in SOURCE_FILES},
        "launch_authorized": False,
    }
    atomic_json(GATE, gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-count", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.test_count), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
