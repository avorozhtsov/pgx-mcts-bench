#!/usr/bin/env python3
"""Bind the isolated fast-6 Q304 protocol after Q254 completes."""

from __future__ import annotations

import argparse, hashlib, json, os, sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
RUN = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817")
POPULATION = RUN / "continuation/q4000-v1-population-20260818"
Q254_ROOT = POPULATION / "q254-fast6-20260823"
ROOT = POPULATION / "q304-fast6-20260824"
PROTOCOL = ROOT / "protocol"
RUNTIME = PROTOCOL / "runtime/pgx_mcts_bench/q304_sv2_curriculum_runtime.py"
GATE = ROOT / "FAST6_Q304_FIRST_BLOCK_SEEDED_VERIFIED.json"
MARKER = Q254_ROOT / "ALL_FAST_6_LINEAGES_Q254_COMPLETE"
SV2_SOURCE = REPO / "src/pgx_mcts_bench/sv2_curriculum.py"
Q254_STAGE = "q50-3-updated-scheduled-no-sharing-bounded-repair-v2"

sys.path.insert(0, str(REPO / "scripts"))
import build_q50_4_updated as builder  # noqa: E402
import run_local_q154_updated_continuation as q154  # noqa: E402
from focused_successor_policy import assert_legacy_q304_launch_authorized  # noqa: E402
from pgx_mcts_bench.collaborative_scientists import _json_hash  # noqa: E402

SOURCE_FILES = (
    REPO / "scripts/prepare_local_q304_fast6.py", REPO / "scripts/run_local_q304_fast6.py",
    REPO / "scripts/run_local_q304_fast6_branch.py", REPO / "scripts/build_q50_4_updated.py",
    REPO / "tests/test_q304_fast6_launcher.py", REPO / "tests/test_q50_4_builder.py",
    REPO / "research/local-q-skm-ablation/q50-4-updated-policy.json",
    REPO / "src/pgx_mcts_bench/sv2_curriculum.py", REPO / "src/pgx_mcts_bench/data.py",
    REPO / "src/pgx_mcts_bench/training.py",
    REPO / "scripts/focused_successor_policy.py",
    REPO / "research/local-q-skm-ablation/focused-successor-v1-policy.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); os.replace(temporary, path)


def transform_runtime(source: str) -> str:
    replacements = {
        '"boundary_completed_rungs": 30,': '"boundary_completed_rungs": 0,',
        '"from_policy": "priority-exposure-v1",': '"from_policy": "seeded-outcome-interleaved-exposure-v1",',
        '"semantic-v2-fast6-slow4-cohort-split-v5",':
            '"semantic-v2-fast6-slow4-cohort-split-v5",\n            "semantic-v2-q304-first-block-seeded-order-v1",',
    }
    transformed = source
    for old, new in replacements.items():
        if transformed.count(old) != 1: raise RuntimeError(f"Q304 runtime patch anchor count differs: {old}")
        transformed = transformed.replace(old, new)
    return "# Generated isolation runtime: Q304 seeded-to-seeded boundary 0.\n" + transformed


def write_runtime() -> None:
    RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    (RUNTIME.parent / "__init__.py").write_text("from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n")
    RUNTIME.write_text(transform_runtime(SV2_SOURCE.read_text()))


def build_bank() -> dict[str, Any]:
    return builder.build(RUN / "inputs/q4000-v1",
        REPO / "research/local-q-skm-ablation/q44-2-updated/q44-2-updated.json",
        REPO / "research/mastery-v3-curriculum/curriculum.json",
        REPO / "research/local-q-skm-ablation/q50-4-updated-policy.json",
        Q254_ROOT / "protocol", PROTOCOL, seed=2026082401)


def terminal_bindings() -> dict[str, Any]:
    if not MARKER.is_file(): raise RuntimeError("fast-6 Q254 terminal marker is missing")
    bindings = {}
    for label in q154.FAST_6_LABELS:
        output = Q254_ROOT / "branches" / label / Q254_STAGE
        report, audit, state = output / "report.json", output / "terminal-retention-audit.json", output / "state.pt.gz"
        payload = json.loads(report.read_text())
        if payload.get("completed_rungs") != 50 or not payload.get("terminal_retention_audit") or not (Q254_ROOT / "branches" / label / "Q254_COMPLETE").is_file():
            raise RuntimeError(f"Q254 branch is not terminal: {label}")
        bindings[label] = {"report_sha256": sha256(report), "terminal_audit_sha256": sha256(audit), "state_sha256": sha256(state)}
    return bindings


def main() -> None:
    assert_legacy_q304_launch_authorized()
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--test-count", type=int, required=True); args = parser.parse_args()
    audit = build_bank()
    if audit.get("status") != "passed": raise RuntimeError("Q50-4 bank audit did not pass")
    bindings = terminal_bindings(); write_runtime()
    bank, prior = PROTOCOL / "q50-4-updated.json", PROTOCOL / "prior-q254-for-q50-4-updated.json"
    gate = {"schema": "semantic-v2-q304-first-block-seeded-order-v1", "passed": True,
        "verified_at": datetime.now(UTC).isoformat(), "test_count": args.test_count, "cohort": "fast-6",
        "fast_cohort": list(q154.FAST_6_LABELS), "predecessor_marker": str(MARKER),
        "predecessor_marker_sha256": sha256(MARKER), "branch_boundaries": bindings,
        "boundary_completed_rungs": 0, "from_policy": "seeded-outcome-interleaved-exposure-v1",
        "to_policy": "seeded-outcome-interleaved-exposure-v1", "first_rehearsal_block_seeded": True,
        "sharing": "strict-none", "bank": str(bank), "bank_byte_sha256": sha256(bank),
        "bank_sha256": _json_hash(json.loads(bank.read_text())), "prior_bank": str(prior),
        "prior_bank_byte_sha256": sha256(prior), "runtime": str(RUNTIME), "runtime_sha256": sha256(RUNTIME),
        "source_sha256": {str(path): sha256(path) for path in SOURCE_FILES},
        "protocol": {"arm": "scheduled-no-sharing", "rehearsal_panel_size": 20, "selfplay_games": 4,
            "optimizer_steps": 24, "ratios": [10, 1000], "terminal_full_retention_audit": True}}
    atomic_json(GATE, gate); print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__": main()
