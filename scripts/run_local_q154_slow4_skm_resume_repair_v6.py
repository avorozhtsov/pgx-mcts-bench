#!/usr/bin/env python3
"""Resume SKM Q134 without replaying already-completed rehearsal debt."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_local_q154_slow4_skm_atomic_cell_recovery_v5 as prior

GATE = prior.base.ROOT / "SLOW4_Q154_SKM_RESUME_REPAIR_V6_VERIFIED.json"
TRANSITION = prior.base.ROOT / "SLOW4_Q154_SKM_TIMEOUT_EXTENSION_V6.json"
STATUS = prior.base.ROOT / "slow4-skm-resume-repair-v6-status.json"
LOG = prior.base.ROOT / "logs/skm-v2-high-combined-dual-resume-repair-v6.log"


def _verify_gate() -> dict[str, Any]:
    payload = json.loads(GATE.read_text())
    if payload.get("schema") != "slow4-q154-skm-resume-repair-gate-v6":
        raise RuntimeError("unexpected Slow Q154 SKM resume-repair gate schema")
    if not payload.get("passed") or payload.get("sharing") != "strict-none":
        raise RuntimeError("Slow Q154 SKM resume-repair gate did not pass")
    if payload.get("checkout") != str(prior.base.REPO):
        raise RuntimeError("Slow Q154 SKM resume-repair checkout differs")
    for section in ("source_hashes", "input_hashes"):
        for raw_path, expected in payload.get(section, {}).items():
            path = Path(raw_path)
            if not path.is_file() or prior.base._sha256(path) != expected:
                raise RuntimeError(f"Slow Q154 SKM resume-repair hash mismatch: {path}")
    return payload


def _status(state: str, detail: str | None = None) -> None:
    prior.base._atomic_json(
        STATUS,
        {
            "schema": "slow4-q154-skm-resume-repair-status-v6",
            "state": state,
            "stage": "skm-v2-high-combined-dual Q134 final retention cell",
            "detail": detail,
            "pid": os.getpid(),
            "maximum_experiment_workers": 1,
            "checkout": str(prior.base.REPO),
            "scientist_task_timeout_seconds": int(prior.TIMEOUT_SECONDS),
            "completed_rehearsal_debt_replay": False,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def main() -> None:
    prior.GATE = GATE
    prior.TRANSITION = TRANSITION
    prior.STATUS = STATUS
    prior.LOG = LOG
    prior._verify_gate = _verify_gate
    prior._status = _status
    prior.main()


if __name__ == "__main__":
    main()
