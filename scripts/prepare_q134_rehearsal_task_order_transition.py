#!/usr/bin/env python3
"""Write the verified primary-8 Q134 rehearsal task-order transition gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
ROOT = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818/"
    "q154-updated-20260819"
)
GATE = ROOT / "Q134_REHEARSAL_TASK_ORDER_TRANSITION_VERIFIED.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _launcher_module():
    source = REPO / "scripts/run_local_q154_updated_continuation.py"
    spec = importlib.util.spec_from_file_location("q154_q134_transition_gate", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Q154 launcher: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-result", required=True)
    args = parser.parse_args()
    manifests = sorted((ROOT / "branches").glob("*/*/manifest.json"))
    if not manifests:
        raise RuntimeError("no frozen Q154 manifest exists")
    manifest = json.loads(manifests[0].read_text())
    if manifest.get("rehearsal_panel_size") != 20:
        raise RuntimeError("Q154 manifest is not the bounded panel protocol")
    launcher = _launcher_module()
    payload = {
        "schema": "semantic-v2-rehearsal-task-order-transition-v1",
        "cohort": "primary-8",
        "passed": True,
        "boundary_completed_rungs": 30,
        "boundary_label": "Q134",
        "from_policy": "priority-exposure-v1",
        "to_policy": "seeded-outcome-interleaved-exposure-v1",
        "panel_membership_policy": "exact-bank-order-expanding-round-robin-v1",
        "retention_order_policy": "same-seeded-permutation-before-and-after",
        "training_order_policy": "outcome-signature-interleave-with-exposure-first",
        "checkpoint_policy": "persist-retention-and-selected-training-orders",
        "bank_sha256": manifest["bank_sha256"],
        "bank_byte_sha256": _sha256(launcher.BANK),
        "sharing": "strict-none",
        "source_sha256": {
            str(path): _sha256(path) for path in launcher.BOUNDED_REHEARSAL_SOURCES
        },
        "tests": {"command": "uv run pytest -q", "result": args.tests_result},
        "verified_at": datetime.now(UTC).isoformat(),
    }
    temporary = GATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, GATE)


if __name__ == "__main__":
    main()
