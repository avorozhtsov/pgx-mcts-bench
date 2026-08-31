#!/usr/bin/env python3
"""Launch the hash-gated slow-4 Q154 recovery with a deterministic tool PATH."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_local_q154_slow4_seed_recovery as recovery

LAUNCHD_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
GATE = recovery.ROOT / "SLOW4_Q154_SEED_INDEX_RECOVERY_V2_VERIFIED.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_gate() -> None:
    payload = json.loads(GATE.read_text())
    if payload.get("schema") != "slow4-q154-seed-index-recovery-gate-v2":
        raise RuntimeError("unexpected slow-4 seed recovery v2 gate schema")
    if payload.get("launchd_path") != LAUNCHD_PATH:
        raise RuntimeError("slow-4 recovery launchd PATH differs from verified gate")
    for raw_path, expected in payload.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"slow-4 recovery v2 source hash mismatch: {path}")


def main() -> None:
    _verify_gate()
    os.environ["PATH"] = LAUNCHD_PATH
    recovery.main()


if __name__ == "__main__":
    main()
