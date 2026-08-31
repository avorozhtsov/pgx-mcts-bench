#!/usr/bin/env python3
"""Build and bind the isolated fast-6 Q254 protocol after Q204 completes."""

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
ROOT = POPULATION / "q254-fast6-20260823"
PROTOCOL = ROOT / "protocol"
RUNTIME = PROTOCOL / "runtime/pgx_mcts_bench/q254_sv2_curriculum_runtime.py"
GATE = ROOT / "FAST6_Q254_FIRST_BLOCK_SEEDED_VERIFIED.json"
FAST_MARKER = Q204_ROOT / "ALL_FAST_6_LINEAGES_Q204_COMPLETE"
SV2_SOURCE = REPO / "src/pgx_mcts_bench/sv2_curriculum.py"
Q204_STAGE = "q50-2-updated-scheduled-no-sharing-bounded"

sys.path.insert(0, str(REPO / "scripts"))
import build_q50_3_updated as builder  # noqa: E402
import run_local_q154_updated_continuation as q154  # noqa: E402
from pgx_mcts_bench.collaborative_scientists import _json_hash  # noqa: E402

SOURCE_FILES = (
    REPO / "scripts/prepare_local_q254_fast6.py",
    REPO / "scripts/run_local_q254_fast6.py",
    REPO / "scripts/run_local_q254_fast6_branch.py",
    REPO / "scripts/build_q50_3_updated.py",
    REPO / "tests/test_q254_fast6_launcher.py",
    REPO / "tests/test_q50_3_builder.py",
    REPO / "research/local-q-skm-ablation/q50-3-updated-policy.json",
    REPO / "src/pgx_mcts_bench/sv2_curriculum.py",
    REPO / "src/pgx_mcts_bench/data.py",
    REPO / "src/pgx_mcts_bench/training.py",
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


def q204_output(label: str) -> Path:
    return Q204_ROOT / "branches" / label / Q204_STAGE


def transform_runtime(source: str) -> str:
    replacements = {
        '"boundary_completed_rungs": 30,': '"boundary_completed_rungs": 0,',
        '"semantic-v2-fast6-slow4-cohort-split-v5",': (
            '"semantic-v2-fast6-slow4-cohort-split-v5",\n'
            '            "semantic-v2-q254-first-block-seeded-order-v1",'
        ),
    }
    transformed = source
    for old, new in replacements.items():
        if transformed.count(old) != 1:
            raise RuntimeError(f"Q254 runtime patch anchor count differs: {old}")
        transformed = transformed.replace(old, new)
    return (
        "# Generated isolation runtime: Q254 seeded rehearsal begins at boundary 0.\n"
        "# Active Q204-bound source and runtime remain byte-for-byte unchanged.\n"
        + transformed
    )


def write_runtime() -> None:
    RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    package = RUNTIME.parent / "__init__.py"
    package.write_text("from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n")
    RUNTIME.write_text(transform_runtime(SV2_SOURCE.read_text()))


def build_bank() -> dict[str, Any]:
    return builder.build(
        RUN / "inputs/q4000-v1",
        REPO / "research/local-q-skm-ablation/q44-2-updated/q44-2-updated.json",
        REPO / "research/mastery-v3-curriculum/curriculum.json",
        REPO / "research/local-q-skm-ablation/q50-3-updated-policy.json",
        Q204_ROOT / "protocol",
        PROTOCOL,
        seed=2026082301,
    )


def terminal_bindings() -> dict[str, Any]:
    if not FAST_MARKER.is_file():
        raise RuntimeError("fast-6 Q204 terminal marker is missing")
    bindings: dict[str, Any] = {}
    for label in q154.FAST_6_LABELS:
        output = q204_output(label)
        report = output / "report.json"
        audit = output / "terminal-retention-audit.json"
        state = output / "state.pt.gz"
        branch_marker = Q204_ROOT / "branches" / label / "Q204_COMPLETE"
        payload = json.loads(report.read_text())
        if (
            payload.get("completed_rungs") != 50
            or not payload.get("terminal_retention_audit")
            or not branch_marker.is_file()
        ):
            raise RuntimeError(f"Q204 branch is not terminal: {label}")
        bindings[label] = {
            "report_sha256": sha256(report),
            "terminal_audit_sha256": sha256(audit),
            "state_sha256": sha256(state),
        }
    return bindings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-count", type=int, required=True)
    args = parser.parse_args()
    audit = build_bank()
    if audit.get("status") != "passed":
        raise RuntimeError("Q50-3 bank audit did not pass")
    write_runtime()
    bank = PROTOCOL / "q50-3-updated.json"
    prior = PROTOCOL / "prior-q204-for-q50-3-updated.json"
    gate = {
        "schema": "semantic-v2-q254-first-block-seeded-order-v1",
        "passed": True,
        "verified_at": datetime.now(UTC).isoformat(),
        "test_count": args.test_count,
        "cohort": "fast-6",
        "fast_cohort": list(q154.FAST_6_LABELS),
        "predecessor_marker": str(FAST_MARKER),
        "predecessor_marker_sha256": sha256(FAST_MARKER),
        "branch_boundaries": terminal_bindings(),
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
        "runtime": str(RUNTIME),
        "runtime_sha256": sha256(RUNTIME),
        "source_sha256": {str(path): sha256(path) for path in SOURCE_FILES},
        "protocol": {
            "arm": "scheduled-no-sharing",
            "rehearsal_panel_size": 20,
            "selfplay_games": 4,
            "optimizer_steps": 24,
            "ratios": [10, 1000],
            "terminal_full_retention_audit": True,
        },
    }
    atomic_json(GATE, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
