#!/usr/bin/env python3
"""Stop a mastery service only after a verified durable group checkpoint.

The guard is deliberately external to the mastery protocol.  It never edits a
program state or checkpoint; its only experiment-side write is a provenance
record describing why a service was (or was not) stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _challenge_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    challenges = state.get("challenges", {})
    rows = challenges.values() if isinstance(challenges, dict) else challenges
    return [row for row in rows if isinstance(row, dict)]


def scientific_success_reasons(state: dict[str, Any]) -> list[str]:
    """Return durable reasons that should cancel a futility stop."""
    reasons: list[str] = []
    for row in _challenge_rows(state):
        initial = row.get("initial_upper_bound")
        current = row.get("current_upper_bound")
        if initial is not None and current is not None and current < initial:
            reasons.append(f"upper-bound-improvement:{row.get('challenge_id', 'unknown')}")
        if row.get("status") in {"strict-success", "verified-improvement"}:
            reasons.append(f"strict-success:{row.get('challenge_id', 'unknown')}")
    for event in state.get("recent_events", []):
        strict = int(event.get("strict_challenge_solutions") or 0)
        improvements = int(event.get("genuine_upper_bound_improvements") or 0)
        if strict:
            reasons.append(f"recent-strict-successes:{strict}")
        if improvements:
            reasons.append(f"recent-upper-bound-improvements:{improvements}")
    return sorted(set(reasons))


def verify_checkpoint(checkpoint: Path, target: int) -> dict[str, Any]:
    manifest_path = checkpoint / "manifest.json"
    program_path = checkpoint / "program-state.json"
    scientist_path = checkpoint / "scientist-state.pt.gz"
    manifest = _load_json(manifest_path)
    state = _load_json(program_path)
    if int(manifest.get("introduced", -1)) != target:
        raise ValueError(f"checkpoint manifest introduced != {target}")
    if int(state.get("introduced_count", -1)) != target:
        raise ValueError(f"checkpoint state introduced_count != {target}")
    program_sha = _sha256(program_path)
    scientist_sha = _sha256(scientist_path)
    if manifest.get("program_state_sha256") != program_sha:
        raise ValueError("checkpoint program-state hash mismatch")
    if manifest.get("scientist_state_sha256") != scientist_sha:
        raise ValueError("checkpoint scientist-state hash mismatch")
    return {
        "path": str(checkpoint),
        "manifest_sha256": _sha256(manifest_path),
        "program_state_sha256": program_sha,
        "scientist_state_sha256": scientist_sha,
        "introduced": target,
        "step": state.get("step_index"),
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _unit_state(unit: str) -> str:
    result = subprocess.run(
        ["systemctl", "is-active", unit], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def _base_record(
    *, state_path: Path, checkpoint: dict[str, Any], unit: str, target: int
) -> dict[str, Any]:
    state = _load_json(state_path)
    statuses: dict[str, int] = {}
    for row in _challenge_rows(state):
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "schema": "mastery-operational-stop-gate-v1",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "target_introduced": target,
        "unit": unit,
        "unit_state_before": _unit_state(unit),
        "live_state": {
            "path": str(state_path),
            "sha256": _sha256(state_path),
            "introduced": state.get("introduced_count"),
            "step": state.get("step_index"),
            "native_train_steps": state.get("native_train_steps"),
            "challenge_statuses": statuses,
        },
        "selected_checkpoint": checkpoint,
        "scientific_success_reasons": scientific_success_reasons(state),
    }


def run_guard(args: argparse.Namespace) -> int:
    state_path = args.output / "program-state.json"
    checkpoint_path = args.output / "checkpoints" / f"group-{args.target:03d}"
    while True:
        try:
            state = _load_json(state_path)
            reasons = scientific_success_reasons(state)
            if reasons:
                record = {
                    "schema": "mastery-operational-stop-gate-v1",
                    "recorded_utc": datetime.now(UTC).isoformat(),
                    "result": "cancelled-scientific-success",
                    "target_introduced": args.target,
                    "unit": args.unit,
                    "scientific_success_reasons": reasons,
                }
                _atomic_write(args.record, record)
                return 2
            if int(state.get("introduced_count", 0)) >= args.target and checkpoint_path.is_dir():
                checkpoint = verify_checkpoint(checkpoint_path, args.target)
                checkpoint_state = _load_json(checkpoint_path / "program-state.json")
                reasons = scientific_success_reasons(checkpoint_state)
                if reasons:
                    record = {
                        "schema": "mastery-operational-stop-gate-v1",
                        "recorded_utc": datetime.now(UTC).isoformat(),
                        "result": "cancelled-checkpoint-scientific-success",
                        "target_introduced": args.target,
                        "unit": args.unit,
                        "scientific_success_reasons": reasons,
                        "selected_checkpoint": checkpoint,
                    }
                    _atomic_write(args.record, record)
                    return 2
                record = _base_record(
                    state_path=state_path,
                    checkpoint=checkpoint,
                    unit=args.unit,
                    target=args.target,
                )
                if record["unit_state_before"] in {"inactive", "failed"}:
                    record["result"] = "already-inactive-at-verified-checkpoint"
                    record["unit_state_after"] = record["unit_state_before"]
                    _atomic_write(args.record, record)
                    return 0
                record["result"] = "stop-requested"
                _atomic_write(args.record, record)
                subprocess.run(["systemctl", "stop", args.unit], check=True)
                deadline = time.monotonic() + args.stop_timeout
                while _unit_state(args.unit) not in {"inactive", "failed"}:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out stopping {args.unit}")
                    time.sleep(1)
                record["stopped_utc"] = datetime.now(UTC).isoformat()
                record["unit_state_after"] = _unit_state(args.unit)
                record["result"] = "stopped-at-verified-checkpoint"
                _atomic_write(args.record, record)
                return 0
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--stop-timeout", type=float, default=30.0)
    return run_guard(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
