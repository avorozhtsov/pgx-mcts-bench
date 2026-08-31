#!/usr/bin/env python3
"""Resume the proof-distilled child with its dedicated branch runner."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import run_focused_successor_proof_distilled as base

LAUNCH_GATE = base.ROOT / "PROOF_DISTILLED_RECOVERY_V2_LAUNCH_VERIFIED.json"
GATE = base.ROOT / "PROOF_DISTILLED_FAST_CONTINUATION_RECOVERY_V2_VERIFIED.json"
STATUS = base.ROOT / "proof-distilled-launcher-status-recovery-v2.json"
LOCK = base.ROOT / "proof-distilled-launcher-recovery-v2.lock"
OUTPUT = base.ROOT / "branches/strand-graph-12-proof-distilled/q50-4-updated-scheduled-no-sharing-bounded-proof-recovery-v2"


def status(state: str, detail: str) -> None:
    base.atomic_json(STATUS, {"schema": "focused-proof-distilled-launcher-recovery-v2", "state": state,
                              "detail": detail, "pid": os.getpid(), "updated_at": datetime.now(UTC).isoformat()})


def main() -> None:
    lock = LOCK.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another proof recovery launcher holds the lock") from error
    launch = json.loads(LAUNCH_GATE.read_text())
    if launch.get("schema") != "focused-proof-distilled-recovery-v2-launch-gate" or not launch.get("launch_permitted"):
        raise RuntimeError("proof recovery v2 launch gate did not pass")
    for raw, expected in launch["source_sha256"].items():
        if base.sha256(Path(raw)) != expected:
            raise RuntimeError(f"proof recovery source hash changed: {raw}")
    if base.sha256(base.CARRY) != launch["carry_sha256"] or base.sha256(base.ADAPTER / "adapter.pt") != launch["adapter_sha256"]:
        raise RuntimeError("proof recovery artifact hash changed")
    gate = json.loads((base.ROOT / "PROOF_DISTILLED_FAST_CONTINUATION_VERIFIED.json").read_text())
    gate.update(
        verified_at=datetime.now(UTC).isoformat(), recovery="dedicated-proof-branch-runner-v2",
        predecessor_gate=str(base.ROOT / "PROOF_DISTILLED_FAST_CONTINUATION_VERIFIED.json"),
        predecessor_gate_sha256=base.sha256(base.ROOT / "PROOF_DISTILLED_FAST_CONTINUATION_VERIFIED.json"),
        source_sha256=launch["source_sha256"],
    )
    base.atomic_json(GATE, gate)
    status("LAUNCHED", "proof-distilled RL continuation awaiting scientist verification")
    subprocess.run([
        str(base.REPO / ".venv/bin/python"), str(base.REPO / "scripts/run_focused_successor_proof_branch.py"),
        "--output", str(OUTPUT), "--gate", str(GATE), "--checkpoint", str(base.MODEL / "checkpoint.pt"),
    ], cwd=base.REPO, check=True)
    (base.ROOT / "branches/strand-graph-12-proof-distilled/Q304_COMPLETE").touch()
    status("COMPLETED", "proof-distilled focused Q304 continuation completed")


if __name__ == "__main__":
    main()
