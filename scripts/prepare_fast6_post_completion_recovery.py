#!/usr/bin/env python3
"""Bind the durable fast-6 boundary for post-completion process recovery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
ROOT = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818/"
    "q154-updated-20260819"
)
GATE = ROOT / "FAST6_POST_COMPLETION_RECOVERY_V6_VERIFIED.json"
PREDECESSOR_GATE = ROOT / "FAST6_SLOW4_COHORT_SPLIT_V5_VERIFIED.json"
RESUME_TRANSACTION_GATE = ROOT / "FAST6_SLOW4_COHORT_SPLIT_V4_VERIFIED.json"
STAGE = "q50-1-updated-scheduled-no-sharing-bounded"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _terminal(output: Path) -> bool:
    report = output / "report.json"
    if not report.is_file():
        return False
    payload = json.loads(report.read_text())
    return payload.get("completed_rungs") == 50 and bool(
        payload.get("terminal_retention_audit")
    )


def _branch_boundary(label: str) -> dict[str, Any]:
    output = ROOT / "branches" / label / STAGE
    manifest = output / "manifest.json"
    events = sorted((output / "events").glob("*.json"))
    row: dict[str, Any] = {
        "terminal": _terminal(output),
        "manifest_sha256": _sha256(manifest),
        "event_count": len(events),
    }
    if row["terminal"]:
        row.update(
            {
                "report_sha256": _sha256(output / "report.json"),
                "terminal_audit_sha256": _sha256(
                    output / "terminal-retention-audit.json"
                ),
            }
        )
        return row
    if not events:
        raise RuntimeError(f"missing durable recovery event boundary: {label}")
    row.update(
        {
            "last_event": events[-1].name,
            "last_event_sha256": _sha256(events[-1]),
            "state_sha256": _sha256(output / "state.pt.gz"),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-result", required=True)
    args = parser.parse_args()
    launcher = _module(
        "q154_launcher_for_post_completion_gate",
        "scripts/run_local_q154_updated_continuation.py",
    )
    recovery = _module(
        "q154_recovery_for_post_completion_gate",
        "scripts/run_local_q154_fast6_transition_recovery.py",
    )
    post = _module(
        "q154_post_completion_recovery_for_gate",
        "scripts/run_local_q154_fast6_post_completion_recovery.py",
    )
    conflicts = {
        label: recovery._coordinated_processes(label)
        for label in launcher.FAST_6_LABELS
        if recovery._coordinated_processes(label)
    }
    if conflicts:
        raise RuntimeError(f"cannot bind recovery while a writer is active: {conflicts}")
    boundaries = {
        label: _branch_boundary(label) for label in launcher.FAST_6_LABELS
    }
    original = recovery.ACTIVE_ORIGINAL_LABEL
    if not boundaries[original]["terminal"]:
        raise RuntimeError("original raster branch has not completed")
    unfinished = [label for label, row in boundaries.items() if not row["terminal"]]
    if len(unfinished) != 4:
        raise RuntimeError(f"expected four unfinished fast branches, found {unfinished}")
    sources = post.POST_COMPLETION_SOURCES
    payload = {
        "schema": "semantic-v2-fast6-post-completion-recovery-v1",
        "passed": True,
        "fast_cohort": list(launcher.FAST_6_LABELS),
        "completed_original_branch": original,
        "already_terminal_branches": [
            label for label, row in boundaries.items() if row["terminal"]
        ],
        "unfinished_branches": unfinished,
        "sharing": "strict-none",
        "recovery_rule": (
            "resume only writer-free roots from their latest durable state; "
            "never restart a terminal branch"
        ),
        "bank_byte_sha256": _sha256(launcher.BANK),
        "predecessor_gate_sha256": _sha256(PREDECESSOR_GATE),
        "resume_transaction_gate": str(RESUME_TRANSACTION_GATE),
        "resume_transaction_gate_sha256": _sha256(RESUME_TRANSACTION_GATE),
        "branch_boundaries": boundaries,
        "source_sha256": {str(path): _sha256(path) for path in sources},
        "tests": {"command": "uv run pytest -q", "result": args.tests_result},
        "verified_at": datetime.now(UTC).isoformat(),
    }
    temporary = GATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, GATE)


if __name__ == "__main__":
    main()
