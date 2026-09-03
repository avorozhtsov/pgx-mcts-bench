#!/usr/bin/env python3
"""Resume the proof+embedding child from its completed on-policy dataset."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = (
    REPO
    / "artifacts/local-q-skm-ablation-20260817/continuation"
    / "q4000-v1-population-20260818/focused-successor-v1"
)
OUTPUT = ROOT / "branches/strand-graph-12-proof-embedding/bridge-v1"
GATE = ROOT / "FOCUSED_PROOF_EMBEDDING_CHILD_RECOVERY_V2_VERIFIED.json"
STATUS = ROOT / "proof-embedding-child-launcher-status-recovery-v2.json"
LOCK = ROOT / "proof-embedding-child-launcher-recovery-v2.lock"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def publish(state: str, detail: str) -> None:
    atomic_json(
        STATUS,
        {
            "schema": "focused-proof-embedding-child-recovery-v2",
            "state": state,
            "detail": detail,
            "pid": os.getpid(),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def run(command: list[str], log: Path) -> None:
    with log.open("a") as handle:
        completed = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"child exited {completed.returncode}: {command[1]}")


def main() -> None:
    with LOCK.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("duplicate embedding recovery launcher") from error
        try:
            gate = json.loads(GATE.read_text())
            if (
                gate.get("schema") != "focused-proof-embedding-child-recovery-gate-v2"
                or gate.get("status") != "PREPARED"
            ):
                raise RuntimeError("embedding recovery gate did not pass")
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip()
            if (
                gate.get("checkout") != str(REPO)
                or gate.get("branch") != "main"
                or gate.get("commit") != head
            ):
                raise RuntimeError("embedding recovery checkout differs")
            for raw, expected in gate["sources"].items():
                if sha256(Path(raw)) != expected:
                    raise RuntimeError(f"embedding recovery source changed: {raw}")
            data = Path(gate["on_policy_data"])
            if sha256(data) != gate["on_policy_data_sha256"]:
                raise RuntimeError("on-policy dataset changed")
            adapter = OUTPUT / "adapter"
            evaluation = OUTPUT / "held-out-mcts.json"
            publish("LAUNCHED", "training frozen-parent embedding bridge from preserved data")
            if not (adapter / "adapter.pt").exists():
                run(
                    [
                        sys.executable,
                        "scripts/train_on_policy_embedding_value_adapter.py",
                        "--parent-manifest",
                        gate["parent"]["manifest"],
                        "--parent-checkpoint",
                        gate["parent"]["checkpoint"],
                        "--embedding-checkpoint",
                        gate["embedding"]["checkpoint"],
                        "--data",
                        str(data),
                        "--output",
                        str(adapter),
                        "--device",
                        "cpu",
                        "--steps",
                        "1500",
                        "--batch-size",
                        "128",
                        "--context-mode",
                        "full",
                        "--seed",
                        str(gate["adapter_seed"]),
                    ],
                    OUTPUT / "logs/train-recovery-v2.log",
                )
            publish("LAUNCHED", "running equal-budget held-out MCTS gate")
            if not evaluation.exists():
                run(
                    [
                        sys.executable,
                        "scripts/evaluate_embedding_value_adapter.py",
                        "--parent-manifest",
                        gate["parent"]["manifest"],
                        "--parent-checkpoint",
                        gate["parent"]["checkpoint"],
                        "--embedding-checkpoint",
                        gate["embedding"]["checkpoint"],
                        "--adapter-checkpoint",
                        str(adapter / "adapter.pt"),
                        "--output",
                        str(evaluation),
                        "--device",
                        "cpu",
                        "--stage",
                        "8",
                        "--stage",
                        "37",
                        "--games",
                        "4",
                        "--simulations",
                        "64",
                        "--seed",
                        str(gate["evaluation_seed"]),
                    ],
                    OUTPUT / "logs/evaluate-recovery-v2.log",
                )
            publish("COMPLETED", "embedding bridge trained and held-out MCTS report written")
        except Exception as error:
            publish("BLOCKED", str(error))
            raise


if __name__ == "__main__":
    main()
