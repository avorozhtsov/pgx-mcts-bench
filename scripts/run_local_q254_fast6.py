#!/usr/bin/env python3
"""Launch six independent fast Q204-to-Q254 continuations."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
RUN = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817")
POPULATION = RUN / "continuation/q4000-v1-population-20260818"
Q204_ROOT = POPULATION / "q204-fast6-20260822"
ROOT = POPULATION / "q254-fast6-20260823"
GATE = ROOT / "FAST6_Q254_FIRST_BLOCK_SEEDED_VERIFIED.json"
STATUS = ROOT / "launcher-status.json"
LOCK = ROOT / "launcher.lock"
MARKER = ROOT / "ALL_FAST_6_LINEAGES_Q254_COMPLETE"
BANK = ROOT / "protocol/q50-3-updated.json"
PRIOR = ROOT / "protocol/prior-q204-for-q50-3-updated.json"
Q204_STAGE = "q50-2-updated-scheduled-no-sharing-bounded"
STAGE = "q50-3-updated-scheduled-no-sharing-bounded"

sys.path.insert(0, str(REPO / "scripts"))
import run_local_q154_updated_continuation as q154  # noqa: E402

_status_lock = threading.Lock()
_status: dict[str, Any] = {
    "schema": "q254-fast-6-population-launcher-v1",
    "cohort": "fast-6",
    "state": "LAUNCHED",
    "stage": "Q254 dispatch",
    "pid": os.getpid(),
    "created_at": datetime.now(UTC).isoformat(),
    "maximum_experiment_workers": 6,
    "branches": {},
}


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


def set_status(label: str, state: str, detail: str | None = None) -> None:
    with _status_lock:
        _status["branches"][label] = {
            "state": state,
            "detail": detail,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(STATUS, _status)


def q204_output(label: str) -> Path:
    return Q204_ROOT / "branches" / label / Q204_STAGE


def output(label: str) -> Path:
    return ROOT / "branches" / label / STAGE


def verify_gate() -> dict[str, Any]:
    gate = json.loads(GATE.read_text())
    if (
        gate.get("schema") != "semantic-v2-q254-first-block-seeded-order-v1"
        or not gate.get("passed")
        or gate.get("fast_cohort") != list(q154.FAST_6_LABELS)
        or gate.get("boundary_completed_rungs") != 0
        or not gate.get("first_rehearsal_block_seeded")
        or gate.get("sharing") != "strict-none"
    ):
        raise RuntimeError("authoritative Q254 fast-6 gate did not pass")
    for raw_path, expected in gate["source_sha256"].items():
        if sha256(Path(raw_path)) != expected:
            raise RuntimeError(f"Q254 source hash changed: {raw_path}")
    if sha256(BANK) != gate["bank_byte_sha256"] or sha256(PRIOR) != gate["prior_bank_byte_sha256"]:
        raise RuntimeError("Q254 protocol bank hash changed")
    for label, binding in gate["branch_boundaries"].items():
        source = q204_output(label)
        for filename, key in (
            ("report.json", "report_sha256"),
            ("terminal-retention-audit.json", "terminal_audit_sha256"),
            ("state.pt.gz", "state_sha256"),
        ):
            if sha256(source / filename) != binding[key]:
                raise RuntimeError(f"Q204 terminal binding changed: {label}/{filename}")
    return gate


def load_state(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=False)


def save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        torch.save(payload, handle)
    os.replace(temporary, path)


def carry_state(label: str, scientist: str, expected_source_hash: str) -> Path:
    source_path = q204_output(label) / "state.pt.gz"
    destination = ROOT / "initial-q204-states" / label / scientist / "state.pt.gz"
    if destination.is_file():
        payload = load_state(destination)
        if payload.get("source_q204_state_sha256") != expected_source_hash:
            raise RuntimeError(f"Q254 carry source differs: {label}")
        return destination
    source = load_state(source_path)
    save_state(
        destination,
        {
            "schema": "semantic-v2-q204-to-q254-scientist-carry-v1",
            "source_q204_state_sha256": expected_source_hash,
            "scientist": source["scientists"][scientist],
            "f_old": int(source["f_old"][scientist]),
            "rehearsal_exposure": source["rehearsal_exposure"][scientist],
            "rehearsal_panel_cursor": int(source["rehearsal_panel_cursor"][scientist]),
            "f_native": int(source["f_native"][scientist]),
            "simulations": int(source["simulations"][scientist]),
            "donation_dose": int(source.get("donation_dose", 1)),
            "donation_healthy_streak": int(source.get("donation_healthy_streak", 0)),
            "controller_reset": None,
        },
    )
    return destination


def branch_command(label: str, scientist: str, state: Path) -> list[str]:
    return [
        str(REPO / ".venv/bin/python"),
        str(REPO / "scripts/run_local_q254_fast6_branch.py"),
        "--output", str(output(label)),
        "--bank", str(BANK),
        "--prior-bank", str(PRIOR),
        "--scientist", scientist,
        "--checkpoint", str(q154._checkpoint(scientist)),
        "--initial-state", str(state),
        "--seed", str(202608230500 + q154.PRIMARY_8_SEED_INDEX[label]),
        "--gate", str(GATE),
    ]


def run_branch(label: str, scientist: str, state: Path) -> None:
    report = output(label) / "report.json"
    if report.is_file():
        payload = json.loads(report.read_text())
        if payload.get("completed_rungs") == 50 and payload.get("terminal_retention_audit"):
            set_status(label, "COMPLETED", "existing durable Q254 terminal report")
            return
    set_status(label, "LAUNCHED", "awaiting branch and scientist verification")
    log = ROOT / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        subprocess.run(
            branch_command(label, scientist, state),
            cwd=REPO,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if not report.is_file():
        raise RuntimeError(f"Q254 branch returned without terminal report: {label}")
    (ROOT / "branches" / label / "Q254_COMPLETE").touch()
    set_status(label, "COMPLETED")


def main() -> None:
    os.environ.update(
        {
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another Q254 fast-6 launcher holds the lock") from error
    gate = verify_gate()
    _status["gate"] = str(GATE)
    _status["gate_sha256"] = sha256(GATE)
    atomic_json(STATUS, _status)
    scientist_by_label = {label: scientist for label, scientist, *_ in q154.BRANCHES}
    carries = {
        label: carry_state(
            label,
            scientist_by_label[label],
            gate["branch_boundaries"][label]["state_sha256"],
        )
        for label in q154.FAST_6_LABELS
    }
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(run_branch, label, scientist_by_label[label], carries[label]): label
            for label in q154.FAST_6_LABELS
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(f"{label}: {error!r}")
                set_status(label, "BLOCKED", repr(error))
    _status["finished_at"] = datetime.now(UTC).isoformat()
    _status["state"] = "BLOCKED" if failures else "COMPLETED"
    _status["failures"] = failures
    atomic_json(STATUS, _status)
    if failures:
        raise SystemExit("; ".join(failures))
    MARKER.touch()


if __name__ == "__main__":
    main()
