#!/usr/bin/env python3
"""Run the gated Q305--Q314 same-root trajectory tournament pilot."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pgx_mcts_bench.sv2_curriculum import run_coordinated_arm

REPO = Path(__file__).resolve().parents[1]
ROOT = (
    REPO
    / "artifacts/local-q-skm-ablation-20260817/continuation"
    / "q4000-v1-population-20260818/focused-successor-v1"
)
GATE = ROOT / "Q305_SAME_ROOT_TOURNAMENT_PILOT_V1_VERIFIED.json"
STATUS = ROOT / "q305-tournament-v1/q305-pilot-launcher-status.json"
LOCK = ROOT / "q305-tournament-v1/q305-pilot-launcher.lock"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(state: str, detail: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "q305-same-root-tournament-pilot-launcher-v1",
        "state": state,
        "detail": detail,
        "pid": os.getpid(),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    temporary = STATUS.with_name(f".{STATUS.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS)


def verify() -> dict:
    gate = json.loads(GATE.read_text())
    if gate.get("schema") != "q305-same-root-trajectory-tournament-pilot-gate-v1":
        raise RuntimeError("unexpected Q305 pilot gate")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if gate.get("status") != "PREPARED" or gate.get("commit") != head:
        raise RuntimeError("Q305 pilot gate is not prepared for current main")
    for raw, expected in gate["sources"].items():
        if sha256(Path(raw)) != expected:
            raise RuntimeError(f"Q305 source changed: {raw}")
    for key in (
        "parent_checkpoint",
        "parent_state",
        "source_parent_state",
        "parent_manifest",
        "bank",
        "prior_bank",
        "bank_audit",
    ):
        if sha256(Path(gate[key])) != gate[f"{key}_sha256"]:
            raise RuntimeError(f"Q305 input changed: {key}")
    return gate


def main() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("duplicate Q305 tournament launcher") from error
        try:
            gate = verify()
            output = Path(gate["output"])
            publish("LAUNCHED", "Q305-Q314 same-root tournament scientist starting")
            report = run_coordinated_arm(
                {gate["scientist"]: Path(gate["parent_checkpoint"])},
                Path(gate["bank"]),
                output,
                arm="scheduled-no-sharing",
                prior_bank=Path(gate["prior_bank"]),
                initial_states={gate["scientist"]: Path(gate["parent_state"])},
                ratios=(10.0, 1000.0),
                simulations=40,
                qualification_simulations=40,
                qualification_attempts=1,
                f_native=1,
                selfplay_games=10,
                trajectory_tournament_size=10,
                relative_trajectory_weight=1.0,
                train_steps=24,
                batch_size=64,
                evaluation_attempts=2,
                evaluation_root_noise=True,
                block_size=10,
                retention_target=0.80,
                action_horizon=128,
                rungs=10,
                seed=202609030305,
                torch_threads=2,
                parallel_scientists=True,
                rehearsal_panel_size=10,
                strict_own_budget_rehearsal=True,
                terminal_full_retention_audit=False,
                device="cpu",
            )
            if int(report["completed_rungs"]) != 10:
                raise RuntimeError("Q305 pilot ended before ten durable rungs")
            (output / "Q314_PILOT_COMPLETE").touch()
            publish("COMPLETED", "Q305-Q314 tournament pilot completed")
        except Exception as error:
            publish("BLOCKED", str(error))
            raise


if __name__ == "__main__":
    main()
