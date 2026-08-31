#!/usr/bin/env python3
"""Resume Slow Q154 after the authorized execution-contract transition."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_local_q154_slow4_seed_recovery as recovery

LAUNCHD_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
ROOT = recovery.ROOT
GATE = ROOT / "SLOW4_Q154_CONTRACT_TRANSITION_RECOVERY_V3_VERIFIED.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_gate() -> dict:
    payload = json.loads(GATE.read_text())
    if payload.get("schema") != "slow4-q154-contract-transition-recovery-gate-v3":
        raise RuntimeError("unexpected Slow Q154 contract recovery gate schema")
    if payload.get("launchd_path") != LAUNCHD_PATH or not payload.get("passed"):
        raise RuntimeError("Slow Q154 contract recovery gate did not pass")
    for raw_path, expected in payload.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Slow Q154 contract recovery source hash mismatch: {path}")
    split_gate = Path(payload["replacement_split_gate"])
    if _sha256(split_gate) != payload["replacement_split_gate_sha256"]:
        raise RuntimeError("replacement split gate hash mismatch")
    return payload


def main() -> None:
    payload = _verify_gate()
    recovery._verify_gate()
    os.environ["PATH"] = LAUNCHD_PATH
    slow = recovery._load_module("q154_slow4_contract_recovery", recovery.SLOW_SOURCE)
    if (ROOT / "ALL_SLOW_4_LINEAGES_Q154_COMPLETE").is_file():
        return
    if not slow.SLOW_Q104_MARKER.is_file():
        raise RuntimeError("slow-4 Q104 completion marker is absent")

    replacement_split_gate = Path(payload["replacement_split_gate"])
    lock_handle = slow._acquire_lock()
    try:
        launcher = slow._q154_launcher_module()
        launcher.PRIMARY_8_SEED_INDEX = recovery._effective_seed_indices(launcher)
        launcher.BOUNDED_REHEARSAL_FIX_GATE = replacement_split_gate
        slow.GATE = replacement_split_gate
        slow._q154_launcher_module = lambda: launcher
        slow._status("LAUNCHED", "slow-4 Q154 contract-transition recovery v3")
        slow._run_q154()
        slow._status("COMPLETED", "slow-4 Q154 contract-transition recovery v3")
    except Exception as error:
        slow._status("BLOCKED", "slow-4 Q154 contract-transition recovery v3", repr(error))
        raise
    finally:
        del lock_handle


if __name__ == "__main__":
    main()
