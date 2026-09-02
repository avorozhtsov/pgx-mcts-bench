#!/usr/bin/env python3
"""Resume only the independent SKM Slow-Q154 lineage after a peer capped out."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench")
POPULATION = REPO / (
    "artifacts/local-q-skm-ablation-20260817/continuation/"
    "q4000-v1-population-20260818"
)
ROOT = POPULATION / "q154-slow4-20260822"
OUTPUT = ROOT / (
    "branches/skm-v2-high-combined-dual/"
    "q50-1-updated-scheduled-no-sharing-bounded"
)
GATE = ROOT / "SLOW4_Q154_SKM_DEBT_RECOVERY_V4_VERIFIED.json"
STATUS = ROOT / "slow4-skm-debt-recovery-v4-status.json"
LOCK = ROOT / "slow4-orchestrator.lock"
LOG = ROOT / "logs/skm-v2-high-combined-dual-recovery-v4.log"
LAUNCHD_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _status(state: str, detail: str | None = None) -> None:
    _atomic_json(
        STATUS,
        {
            "schema": "slow4-q154-skm-debt-recovery-status-v4",
            "state": state,
            "stage": "skm-v2-high-combined-dual Q134 barrier",
            "detail": detail,
            "pid": os.getpid(),
            "maximum_experiment_workers": 1,
            "checkout": str(REPO),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _verify_gate() -> dict[str, Any]:
    payload = json.loads(GATE.read_text())
    if payload.get("schema") != "slow4-q154-skm-debt-recovery-gate-v4":
        raise RuntimeError("unexpected Slow Q154 SKM recovery gate schema")
    if not payload.get("passed") or payload.get("sharing") != "strict-none":
        raise RuntimeError("Slow Q154 SKM recovery gate did not pass")
    if payload.get("checkout") != str(REPO):
        raise RuntimeError("Slow Q154 SKM recovery checkout differs")
    for section in ("source_hashes", "input_hashes"):
        for raw_path, expected in payload.get(section, {}).items():
            path = Path(raw_path)
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"Slow Q154 SKM recovery hash mismatch: {path}")
    return payload


def _command() -> list[str]:
    return [
        str(REPO / ".venv/bin/pgx-mcts-bench"),
        "braid-sv2-coordinated",
        "--output",
        str(OUTPUT),
        "--bank",
        str(ROOT / "protocol/q50-1-updated.json"),
        "--prior-bank",
        str(ROOT / "protocol/prior-q104-for-q50-1-updated.json"),
        "--scientist",
        "raster-invariant-combined-dual-12="
        + str(
            REPO
            / "artifacts/nebius-main32-final-20260817/artifacts/"
            "q4000-strand12-20260814/migrated/"
            "raster-invariant-combined-dual-12/checkpoint.pt"
        ),
        "--initial-state",
        "raster-invariant-combined-dual-12="
        + str(
            ROOT
            / "initial-q104-states/skm-v2-high-combined-dual/"
            "raster-invariant-combined-dual-12/state.pt.gz"
        ),
        "--arm",
        "scheduled-no-sharing",
        "--ratios",
        "10,1000",
        "--simulations",
        "40",
        "--qualification-simulations",
        "40",
        "--qualification-attempts",
        "1",
        "--f-native",
        "4",
        "--selfplay-games",
        "4",
        "--train-steps",
        "24",
        "--batch-size",
        "64",
        "--evaluation-attempts",
        "2",
        "--block-size",
        "10",
        "--retention-target",
        "0.8",
        "--action-horizon",
        "128",
        "--rungs",
        "0",
        "--seed",
        "202608190307",
        "--torch-threads",
        "1",
        "--parallel-scientists",
        "--adaptive-compute",
        "--f-native-levels",
        "4,6,8,12,16",
        "--simulation-levels",
        "40,64,80,128,256",
        "--device",
        "cpu",
        "--rehearsal-panel-size",
        "20",
        "--strict-own-budget-rehearsal",
        "--rehearsal-repair-debt",
        "raster-invariant-combined-dual-12=38",
        "--terminal-full-retention-audit",
        "--pause-after-rungs",
        "30",
        "--scientist-task-timeout-seconds",
        "7200",
        "--resumable-rehearsal-segments",
        "--rehearsal-training-seconds-per-iteration-at-reference",
        "7200",
        "--resume",
    ]


def main() -> None:
    try:
        _verify_gate()
        os.environ.update(
            {
                "PATH": LAUNCHD_PATH,
                "VECLIB_MAXIMUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        lock_handle = LOCK.open("a+")
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            report = OUTPUT / "barrier-report-030.json"
            if report.is_file():
                payload = json.loads(report.read_text())
                if payload.get("completed_rungs") != 30:
                    raise RuntimeError("existing SKM Q134 barrier report is invalid")
                _status("COMPLETED", "existing durable Q134 barrier report")
                return
            _status("LAUNCHED", "resuming isolated SKM debt and Q134 work")
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with LOG.open("a") as handle:
                result = subprocess.run(
                    _command(),
                    cwd=REPO,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode != 0:
                _status("BLOCKED", f"worker exited {result.returncode}")
                return
            if not report.is_file():
                _status("BLOCKED", "worker returned without Q134 barrier report")
                return
            payload = json.loads(report.read_text())
            if payload.get("completed_rungs") != 30:
                _status("BLOCKED", "Q134 barrier report has wrong completed-rung count")
                return
            _status("COMPLETED", "durable Q134 barrier report verified")
        finally:
            lock_handle.close()
    except Exception as error:
        _status("BLOCKED", repr(error))


if __name__ == "__main__":
    main()
