#!/usr/bin/env python3
"""Run the gated proof-distilled plus whole-word embedding child."""

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
    REPO / "artifacts/local-q-skm-ablation-20260817/continuation/"
    "q4000-v1-population-20260818/focused-successor-v1"
)
OUTPUT = ROOT / "branches/strand-graph-12-proof-embedding/bridge-v1"
GATE = ROOT / "FOCUSED_PROOF_EMBEDDING_CHILD_V1_VERIFIED.json"
STATUS = ROOT / "proof-embedding-child-launcher-status.json"
LOCK = ROOT / "proof-embedding-child-launcher.lock"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def publish(state: str, detail: str) -> None:
    atomic_json(
        STATUS,
        {
            "schema": "focused-proof-embedding-child-launcher-v1",
            "state": state,
            "detail": detail,
            "pid": os.getpid(),
            "checkout": str(REPO),
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"child command exited {completed.returncode}: {command[1]}")


def verify_gate() -> dict:
    gate = json.loads(GATE.read_text())
    if gate.get("schema") != "focused-proof-embedding-child-gate-v1":
        raise RuntimeError("unexpected embedding child gate")
    if gate.get("status") != "PREPARED" or gate.get("checkout") != str(REPO):
        raise RuntimeError("embedding child gate is not prepared for main checkout")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if gate.get("branch") != "main" or gate.get("commit") != head:
        raise RuntimeError("embedding child gate commit differs")
    for section, key in (
        ("parent", "manifest"),
        ("parent", "checkpoint"),
        ("embedding", "checkpoint"),
    ):
        path = Path(gate[section][key])
        if sha256(path) != gate[section][f"{key}_sha256"]:
            raise RuntimeError(f"{section} {key} hash differs")
    if (
        sha256(Path(gate["embedding"]["selection_manifest"]))
        != gate["embedding"]["selection_manifest_sha256"]
    ):
        raise RuntimeError("embedding selection manifest hash differs")
    for path_text, expected in gate["sources"].items():
        if sha256(Path(path_text)) != expected:
            raise RuntimeError(f"hash-bound source differs: {path_text}")
    return gate


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("duplicate embedding-child launcher") from error
        try:
            gate = verify_gate()
            parent_manifest = gate["parent"]["manifest"]
            parent_checkpoint = gate["parent"]["checkpoint"]
            embedding_checkpoint = gate["embedding"]["checkpoint"]
            data = OUTPUT / "on-policy-data.pt"
            adapter_dir = OUTPUT / "adapter"
            evaluation = OUTPUT / "held-out-mcts.json"
            python = sys.executable
            publish("LAUNCHED", "collecting fresh proof-parent on-policy trajectories")
            if not data.exists():
                run(
                    [
                        python,
                        "scripts/collect_on_policy_embedding_value.py",
                        "--parent-manifest",
                        parent_manifest,
                        "--parent-checkpoint",
                        parent_checkpoint,
                        "--output",
                        str(data),
                        "--train-stage",
                        "1",
                        "--train-stage",
                        "4",
                        "--train-stage",
                        "7",
                        "--train-stage",
                        "11",
                        "--train-stage",
                        "34",
                        "--train-stage",
                        "38",
                        "--validation-stage",
                        "6",
                        "--validation-stage",
                        "36",
                        "--train-games",
                        "10",
                        "--validation-games",
                        "5",
                        "--simulations",
                        "32",
                        "--ratio",
                        "1000",
                        "--seed",
                        "202609030305",
                        "--device",
                        "cpu",
                    ],
                    OUTPUT / "logs/collect.log",
                )
            publish("LAUNCHED", "training zero-initialized frozen embedding bridge")
            if not (adapter_dir / "adapter.pt").exists():
                run(
                    [
                        python,
                        "scripts/train_on_policy_embedding_value_adapter.py",
                        "--parent-manifest",
                        parent_manifest,
                        "--parent-checkpoint",
                        parent_checkpoint,
                        "--embedding-checkpoint",
                        embedding_checkpoint,
                        "--data",
                        str(data),
                        "--output",
                        str(adapter_dir),
                        "--device",
                        "cpu",
                        "--steps",
                        "1500",
                        "--batch-size",
                        "128",
                        "--context-mode",
                        "full",
                        "--seed",
                        "202609030306",
                    ],
                    OUTPUT / "logs/train.log",
                )
            publish("LAUNCHED", "running equal-budget held-out MCTS gate")
            if not evaluation.exists():
                run(
                    [
                        python,
                        "scripts/evaluate_embedding_value_adapter.py",
                        "--parent-manifest",
                        parent_manifest,
                        "--parent-checkpoint",
                        parent_checkpoint,
                        "--embedding-checkpoint",
                        embedding_checkpoint,
                        "--adapter-checkpoint",
                        str(adapter_dir / "adapter.pt"),
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
                        "202609030307",
                    ],
                    OUTPUT / "logs/evaluate.log",
                )
            publish("COMPLETED", "embedding bridge trained and held-out MCTS report written")
        except Exception as error:
            publish("BLOCKED", str(error))
            raise


if __name__ == "__main__":
    main()
