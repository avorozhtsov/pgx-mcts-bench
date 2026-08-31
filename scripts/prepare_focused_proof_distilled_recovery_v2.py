#!/usr/bin/env python3
"""Bind the dedicated proof-branch runner after the v1 cohort-check failure."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import run_focused_successor_proof_distilled as base

V1_GATE = base.ROOT / "PROOF_DISTILLED_FAST_CONTINUATION_VERIFIED.json"
LAUNCH_GATE = base.ROOT / "PROOF_DISTILLED_RECOVERY_V2_LAUNCH_VERIFIED.json"


def main() -> None:
    prior = json.loads(V1_GATE.read_text())
    adapter_manifest = base.ADAPTER / "manifest.json"
    if prior.get("passed") is not True or not adapter_manifest.is_file():
        raise RuntimeError("proof distillation did not complete before recovery")
    value = {
        "schema": "focused-proof-distilled-recovery-v2-launch-gate",
        "status": "PREPARED", "launch_permitted": True,
        "verified_at": datetime.now(UTC).isoformat(),
        "predecessor_gate": str(V1_GATE), "predecessor_gate_sha256": base.sha256(V1_GATE),
        "adapter_manifest": str(adapter_manifest), "adapter_manifest_sha256": base.sha256(adapter_manifest),
        "adapter": str(base.ADAPTER / "adapter.pt"), "adapter_sha256": base.sha256(base.ADAPTER / "adapter.pt"),
        "carry": str(base.CARRY), "carry_sha256": base.sha256(base.CARRY),
        "source_sha256": {
            str(base.REPO / "scripts/run_focused_successor_proof_branch.py"): base.sha256(base.REPO / "scripts/run_focused_successor_proof_branch.py"),
            str(base.REPO / "scripts/run_focused_successor_proof_distilled_recovery_v2.py"): base.sha256(base.REPO / "scripts/run_focused_successor_proof_distilled_recovery_v2.py"),
            str(base.REPO / "scripts/prepare_focused_proof_distilled_recovery_v2.py"): base.sha256(base.REPO / "scripts/prepare_focused_proof_distilled_recovery_v2.py"),
            str(base.REPO / "tests/test_focused_successor_proof_distilled.py"): base.sha256(base.REPO / "tests/test_focused_successor_proof_distilled.py"),
        },
    }
    base.atomic_json(LAUNCH_GATE, value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
