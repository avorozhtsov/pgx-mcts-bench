#!/usr/bin/env python3
"""Resume the SKM Q134 barrier with enough time for one atomic retention cell."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_local_q154_slow4_skm_debt_recovery_v4 as base

GATE = base.ROOT / "SLOW4_Q154_SKM_ATOMIC_CELL_RECOVERY_V5_VERIFIED.json"
STATUS = base.ROOT / "slow4-skm-atomic-cell-recovery-v5-status.json"
LOG = base.ROOT / "logs/skm-v2-high-combined-dual-atomic-cell-recovery-v5.log"
TIMEOUT_SECONDS = "21600"
ORIGINAL_COMMAND = base._command


def _verify_gate() -> dict[str, Any]:
    payload = json.loads(GATE.read_text())
    if payload.get("schema") != "slow4-q154-skm-atomic-cell-recovery-gate-v5":
        raise RuntimeError("unexpected Slow Q154 SKM atomic-cell gate schema")
    if not payload.get("passed") or payload.get("sharing") != "strict-none":
        raise RuntimeError("Slow Q154 SKM atomic-cell gate did not pass")
    if payload.get("checkout") != str(base.REPO):
        raise RuntimeError("Slow Q154 SKM atomic-cell checkout differs")
    for section in ("source_hashes", "input_hashes"):
        for raw_path, expected in payload.get(section, {}).items():
            path = Path(raw_path)
            if not path.is_file() or base._sha256(path) != expected:
                raise RuntimeError(f"Slow Q154 SKM atomic-cell hash mismatch: {path}")
    return payload


def _command() -> list[str]:
    command = ORIGINAL_COMMAND()
    index = command.index("--scientist-task-timeout-seconds")
    command[index + 1] = TIMEOUT_SECONDS
    return command


def _status(state: str, detail: str | None = None) -> None:
    base._atomic_json(
        STATUS,
        {
            "schema": "slow4-q154-skm-atomic-cell-recovery-status-v5",
            "state": state,
            "stage": "skm-v2-high-combined-dual Q134 atomic retention cell",
            "detail": detail,
            "pid": os.getpid(),
            "maximum_experiment_workers": 1,
            "checkout": str(base.REPO),
            "scientist_task_timeout_seconds": int(TIMEOUT_SECONDS),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def main() -> None:
    base.GATE = GATE
    base.STATUS = STATUS
    base.LOG = LOG
    base._verify_gate = _verify_gate
    base._command = _command
    base._status = _status
    base.main()


if __name__ == "__main__":
    main()
