#!/usr/bin/env python3
"""Launch six independent fast Q254-to-Q304 continuations."""

from __future__ import annotations

import gzip, json, os, sys
from datetime import UTC, datetime
from pathlib import Path

import torch

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
sys.path.insert(0, str(REPO / "scripts"))
import run_local_q254_fast6 as base  # noqa: E402
from focused_successor_policy import assert_legacy_q304_launch_authorized  # noqa: E402

POPULATION = base.POPULATION; Q254_ROOT = POPULATION / "q254-fast6-20260823"; ROOT = POPULATION / "q304-fast6-20260824"
GATE = ROOT / "FAST6_Q304_FIRST_BLOCK_SEEDED_VERIFIED.json"; STATUS = ROOT / "launcher-status.json"
LOCK = ROOT / "launcher.lock"; MARKER = ROOT / "ALL_FAST_6_LINEAGES_Q304_COMPLETE"
BANK = ROOT / "protocol/q50-4-updated.json"; PRIOR = ROOT / "protocol/prior-q254-for-q50-4-updated.json"
Q254_STAGE = "q50-3-updated-scheduled-no-sharing-bounded-repair-v2"; STAGE = "q50-4-updated-scheduled-no-sharing-bounded"

base.Q204_ROOT = Q254_ROOT; base.ROOT = ROOT; base.GATE = GATE; base.STATUS = STATUS; base.LOCK = LOCK
base.MARKER = MARKER; base.BANK = BANK; base.PRIOR = PRIOR; base.Q204_STAGE = Q254_STAGE; base.STAGE = STAGE
base._status = {"schema": "q304-fast-6-population-launcher-v1", "cohort": "fast-6", "state": "LAUNCHED",
    "stage": "Q304 dispatch", "pid": os.getpid(), "created_at": datetime.now(UTC).isoformat(),
    "maximum_experiment_workers": 6, "branches": {}}


def verify_gate():
    assert_legacy_q304_launch_authorized()
    gate = json.loads(GATE.read_text())
    if gate.get("schema") != "semantic-v2-q304-first-block-seeded-order-v1" or not gate.get("passed") or gate.get("fast_cohort") != list(base.q154.FAST_6_LABELS) or gate.get("boundary_completed_rungs") != 0 or not gate.get("first_rehearsal_block_seeded") or gate.get("sharing") != "strict-none":
        raise RuntimeError("authoritative Q304 fast-6 gate did not pass")
    for raw, expected in gate["source_sha256"].items():
        if base.sha256(Path(raw)) != expected: raise RuntimeError(f"Q304 source hash changed: {raw}")
    if base.sha256(BANK) != gate["bank_byte_sha256"] or base.sha256(PRIOR) != gate["prior_bank_byte_sha256"]:
        raise RuntimeError("Q304 protocol bank hash changed")
    for label, binding in gate["branch_boundaries"].items():
        source = base.q204_output(label)
        for filename, key in (("report.json", "report_sha256"), ("terminal-retention-audit.json", "terminal_audit_sha256"), ("state.pt.gz", "state_sha256")):
            if base.sha256(source / filename) != binding[key]: raise RuntimeError(f"Q254 terminal binding changed: {label}/{filename}")
    return gate


def carry_state(label: str, scientist: str, expected: str) -> Path:
    source_path = base.q204_output(label) / "state.pt.gz"; destination = ROOT / "initial-q254-states" / label / scientist / "state.pt.gz"
    if destination.is_file():
        with gzip.open(destination, "rb") as handle: payload = torch.load(handle, map_location="cpu", weights_only=False)
        if payload.get("source_q254_state_sha256") != expected: raise RuntimeError(f"Q304 carry source differs: {label}")
        return destination
    source = base.load_state(source_path)
    base.save_state(destination, {"schema": "semantic-v2-q254-to-q304-scientist-carry-v1", "source_q254_state_sha256": expected,
        "scientist": source["scientists"][scientist], "f_old": int(source["f_old"][scientist]),
        "rehearsal_exposure": source["rehearsal_exposure"][scientist], "rehearsal_panel_cursor": int(source["rehearsal_panel_cursor"][scientist]),
        "f_native": int(source["f_native"][scientist]), "simulations": int(source["simulations"][scientist]),
        "donation_dose": int(source.get("donation_dose", 1)), "donation_healthy_streak": int(source.get("donation_healthy_streak", 0)), "controller_reset": None})
    return destination


def branch_command(label: str, scientist: str, state: Path) -> list[str]:
    return [str(REPO / ".venv/bin/python"), str(REPO / "scripts/run_local_q304_fast6_branch.py"),
        "--output", str(base.output(label)), "--bank", str(BANK), "--prior-bank", str(PRIOR), "--scientist", scientist,
        "--checkpoint", str(base.q154._checkpoint(scientist)), "--initial-state", str(state),
        "--seed", str(202608240500 + base.q154.PRIMARY_8_SEED_INDEX[label]), "--gate", str(GATE)]


base.verify_gate = verify_gate; base.carry_state = carry_state; base.branch_command = branch_command


if __name__ == "__main__": base.main()
