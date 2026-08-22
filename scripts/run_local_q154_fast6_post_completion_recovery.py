#!/usr/bin/env python3
"""Resume fast-6 after the excluded original branch has already completed."""

from __future__ import annotations

import fcntl
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import run_local_q154_fast6_transition_recovery as recovery
import run_local_q154_updated_continuation as launcher

POST_COMPLETION_GATE = (
    launcher.ROOT / "FAST6_POST_COMPLETION_RECOVERY_V6_VERIFIED.json"
)
RESUME_TRANSACTION_GATE = (
    launcher.ROOT / "FAST6_SLOW4_COHORT_SPLIT_V4_VERIFIED.json"
)
POST_COMPLETION_SOURCES = (
    launcher.REPO / "scripts/run_local_q154_fast6_post_completion_recovery.py",
    launcher.REPO / "scripts/prepare_fast6_post_completion_recovery.py",
    launcher.REPO / "tests/test_q154_post_completion_recovery.py",
)


def _sha256(path: Path) -> str:
    return launcher._sha256(path)


def _verify_v5_gate_read_only() -> None:
    gate = json.loads(launcher.BOUNDED_REHEARSAL_FIX_GATE.read_text())
    if (
        gate.get("schema") != "semantic-v2-fast6-slow4-cohort-split-v5"
        or gate.get("fast_cohort") != list(launcher.FAST_6_LABELS)
        or not gate.get("passed")
    ):
        raise RuntimeError("authoritative V5 predecessor gate did not pass")
    expected = {
        str(path): _sha256(path) for path in launcher.BOUNDED_REHEARSAL_SOURCES
    }
    if gate.get("source_sha256") != expected:
        raise RuntimeError("V5 predecessor source hashes changed")


def _verify_branch_binding(label: str, binding: dict[str, Any]) -> None:
    output = recovery._output(label)
    manifest = output / "manifest.json"
    if _sha256(manifest) != binding["manifest_sha256"]:
        raise RuntimeError(f"recovery manifest changed: {label}")
    if binding["terminal"]:
        if not recovery._terminal_report(label):
            raise RuntimeError(f"bound terminal report is no longer terminal: {label}")
        if _sha256(output / "report.json") != binding["report_sha256"]:
            raise RuntimeError(f"terminal report changed: {label}")
        if _sha256(output / "terminal-retention-audit.json") != binding[
            "terminal_audit_sha256"
        ]:
            raise RuntimeError(f"terminal audit changed: {label}")
        return
    events = sorted((output / "events").glob("*.json"))
    if len(events) < int(binding["event_count"]):
        raise RuntimeError(f"recovery event ledger regressed: {label}")
    boundary_event = output / "events" / binding["last_event"]
    if _sha256(boundary_event) != binding["last_event_sha256"]:
        raise RuntimeError(f"recovery boundary event changed: {label}")
    if len(events) == int(binding["event_count"]) and _sha256(
        output / "state.pt.gz"
    ) != binding["state_sha256"]:
        raise RuntimeError(f"recovery boundary state changed without a new event: {label}")


def _verify_post_completion_gate() -> dict[str, Any]:
    gate = json.loads(POST_COMPLETION_GATE.read_text())
    if (
        gate.get("schema") != "semantic-v2-fast6-post-completion-recovery-v1"
        or not gate.get("passed")
        or gate.get("fast_cohort") != list(launcher.FAST_6_LABELS)
        or gate.get("completed_original_branch") != recovery.ACTIVE_ORIGINAL_LABEL
        or gate.get("sharing") != "strict-none"
    ):
        raise RuntimeError("post-completion recovery gate did not pass")
    if gate.get("predecessor_gate_sha256") != _sha256(
        launcher.BOUNDED_REHEARSAL_FIX_GATE
    ):
        raise RuntimeError("post-completion predecessor gate changed")
    if gate.get("resume_transaction_gate_sha256") != _sha256(
        RESUME_TRANSACTION_GATE
    ):
        raise RuntimeError("frozen V4 resume transaction gate changed")
    expected_sources = {str(path): _sha256(path) for path in POST_COMPLETION_SOURCES}
    if gate.get("source_sha256") != expected_sources:
        raise RuntimeError("post-completion recovery source hashes changed")
    for label in launcher.FAST_6_LABELS:
        _verify_branch_binding(label, gate["branch_boundaries"][label])
    return gate


def _verify_no_writers() -> None:
    conflicts = {
        label: recovery._coordinated_processes(label)
        for label in launcher.FAST_6_LABELS
        if recovery._coordinated_processes(label)
    }
    if conflicts:
        raise RuntimeError(f"post-completion recovery found an active writer: {conflicts}")


def _atomic_status(payload: dict[str, Any]) -> None:
    recovery._atomic_status(payload)


def main() -> None:
    recovery.RECOVERY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = recovery.RECOVERY_LOCK.open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("fast-6 recovery launcher already exists") from error

    _verify_no_writers()
    _verify_v5_gate_read_only()
    gate = _verify_post_completion_gate()
    if not recovery._terminal_report(recovery.ACTIVE_ORIGINAL_LABEL):
        raise RuntimeError("excluded original raster branch is not durably complete")

    launcher.STATUS = recovery.RECOVERY_STATUS
    launcher._status_lock = threading.Lock()
    launcher._status = {
        "schema": "q154-fast-6-post-completion-recovery-v1",
        "cohort": launcher.COHORT,
        "created_at": datetime.now(UTC).isoformat(),
        "state": "LAUNCHED",
        "stage": "Q135-Q154 post-completion recovery dispatch",
        "pid": os.getpid(),
        "completed_original_branch": recovery.ACTIVE_ORIGINAL_LABEL,
        "maximum_experiment_workers": len(recovery.RECOVERY_BRANCHES),
        "gate": str(POST_COMPLETION_GATE),
        "gate_sha256": _sha256(POST_COMPLETION_GATE),
        "predecessor_gate": str(launcher.BOUNDED_REHEARSAL_FIX_GATE),
        "branches": {},
    }
    _atomic_status(launcher._status)

    debts = json.loads(launcher.REHEARSAL_DEBT.read_text())["lineages"]
    # These four ledgers were opened under V4 before V5 became authoritative
    # for future dispatch.  Resuming them with a new gate path would change the
    # frozen protocol hash, so preserve the transaction-local V4 gate exactly.
    launcher.BOUNDED_REHEARSAL_FIX_GATE = RESUME_TRANSACTION_GATE
    failures: list[str] = []
    jobs = []
    for label, scientist, simulations, timeout in recovery.RECOVERY_BRANCHES:
        initial_state = (
            launcher.ROOT
            / "initial-q104-states"
            / label
            / scientist
            / "state.pt.gz"
        )
        jobs.append(
            (
                label,
                partial(
                    launcher._run_branch,
                    label,
                    scientist,
                    simulations,
                    timeout,
                    initial_state,
                    202608190300 + launcher.PRIMARY_8_SEED_INDEX[label],
                    int(debts[label]),
                    phase="q154-final",
                ),
            )
        )
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(job): label for label, job in jobs}
        for future in as_completed(futures):
            label = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(f"{label}: {error!r}")
                launcher._set_status(label, "BLOCKED", "q154-final", repr(error))
    if failures:
        launcher._status["state"] = "BLOCKED"
        launcher._status["failures"] = failures
        launcher._status["finished_at"] = datetime.now(UTC).isoformat()
        _atomic_status(launcher._status)
        raise SystemExit("; ".join(failures))

    recovery._write_fast6_completion_marker()
    launcher._status["state"] = "COMPLETED"
    launcher._status["stage"] = "Q154"
    launcher._status["finished_at"] = datetime.now(UTC).isoformat()
    launcher._status["recovery_boundary_gate_verified_at"] = gate["verified_at"]
    _atomic_status(launcher._status)


if __name__ == "__main__":
    main()
