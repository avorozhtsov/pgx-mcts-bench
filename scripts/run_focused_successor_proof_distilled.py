#!/usr/bin/env python3
"""Distill replay-validated NodeDB targets, then continue the proof child."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
PGX_BASE = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench")
RF = Path("/Users/artemvorozhtsov/projects/rf-knots")
UNKNOTDB = Path("/Users/artemvorozhtsov/projects/unknotdb")
ROOT = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818/focused-successor-v1"
)
PARENT = ROOT / "parents/strand-graph-12-proof-distilled/initial-state.pt.gz"
DATASET = UNKNOTDB / "outputs/unknotdb-dgkt-cc-frontier-supervision-v0.sqlite"
MODEL = ROOT / "proof-distillation/strand-graph-q254-model"
ADAPTER = ROOT / "proof-distillation/strand-graph-q254-adapter"
CARRY = ROOT / "parents/strand-graph-12-proof-distilled/proof-distilled-carry-v1.pt.gz"
PREPARED = ROOT / "PROOF_DISTILLATION_LAUNCH_VERIFIED.json"
GATE = ROOT / "PROOF_DISTILLED_FAST_CONTINUATION_VERIFIED.json"
STATUS = ROOT / "proof-distilled-launcher-status.json"
LOCK = ROOT / "proof-distilled-launcher.lock"
OUTPUT = ROOT / "branches/strand-graph-12-proof-distilled/q50-4-updated-scheduled-no-sharing-bounded-proof-v1"
SEED = 202608310102
TRAINER = UNKNOTDB / "tools/train_q254_cc_frontier_adapter.py"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from pgx_mcts_bench.adaptive_scientists import load_scientist  # noqa: E402
import prepare_focused_successor_fast_controls as controls  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    with gzip.open(path, "rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=False)


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        torch.save(value, handle)
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def set_status(state: str, detail: str) -> None:
    atomic_json(STATUS, {"schema": "focused-proof-distilled-launcher-v1", "state": state, "detail": detail,
                         "pid": os.getpid(), "updated_at": datetime.now(UTC).isoformat()})


def prepare_model() -> dict:
    state = load(PARENT)
    scientist_state = state["scientists"]["strand-graph-12"]
    MODEL.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL / "checkpoint.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({"network": scientist_state["network"]}, temporary)
    os.replace(temporary, checkpoint)
    manifest = {
        "schema": "focused-strand-graph-q254-model-v1",
        "scientist": "strand-graph-12",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "objective_ratio": 1000,
        "model_id": f"q-grown-strand-graph-12:Q254:{sha256(PARENT)}",
        "source_parent": str(PARENT),
        "source_parent_sha256": sha256(PARENT),
    }
    atomic_json(MODEL / "manifest.json", manifest)
    return manifest


def integrate(manifest: dict) -> str:
    adapter_manifest = json.loads((ADAPTER / "manifest.json").read_text())
    if adapter_manifest["status"] != "trained_not_published":
        raise RuntimeError("proof adapter is not a completed isolated training result")
    if adapter_manifest["dataset"]["sha256"] != sha256(DATASET):
        raise RuntimeError("proof adapter dataset differs")
    if adapter_manifest["proof_policy_contract"]["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise RuntimeError("proof adapter parent differs")
    parent = load(PARENT)
    old_network = parent["scientists"]["strand-graph-12"]["network"]
    scientist = load_scientist("strand-graph-12", MODEL / "checkpoint.pt", seed=SEED, device="cpu",
                               require_factorized=True, objective_budget_channel=True)
    adapter = scientist.network.attach_option_policy_adapter()
    adapter.load_state_dict(torch.load(ADAPTER / "adapter.pt", map_location="cpu", weights_only=False)["adapter"])
    new_network = scientist.network.state_dict()
    for key, value in old_network.items():
        if key not in new_network or not torch.equal(value, new_network[key]):
            raise RuntimeError(f"proof integration changed frozen parent parameter: {key}")
    updated = dict(parent["scientists"]["strand-graph-12"])
    updated["network"] = new_network
    carry = {
        "schema": "focused-successor-v1-proof-distilled-carry-v1",
        "source_q254_state_sha256": sha256(PARENT),
        "proof_adapter_sha256": sha256(ADAPTER / "adapter.pt"),
        "scientist": updated,
        "f_old": int(parent["f_old"]["strand-graph-12"]),
        "rehearsal_exposure": parent["rehearsal_exposure"]["strand-graph-12"],
        "rehearsal_panel_cursor": int(parent["rehearsal_panel_cursor"]["strand-graph-12"]),
        "f_native": int(parent["f_native"]["strand-graph-12"]),
        "simulations": int(parent["simulations"]["strand-graph-12"]),
        "donation_dose": int(parent.get("donation_dose", 1)),
        "donation_healthy_streak": int(parent.get("donation_healthy_streak", 0)),
        "controller_reset": None,
    }
    save(CARRY, carry)
    return sha256(CARRY)


def continuation_gate(carry_sha: str) -> dict:
    base_gate = json.loads((ROOT / "FOCUSED_FAST_CONTROLS_RECOVERY_V3_VERIFIED.json").read_text())
    base_gate.update(
        verified_at=datetime.now(UTC).isoformat(),
        cohort="focused-successor-v1-proof-distilled",
        focused_lines={"strand-graph-12-proof-distilled": {
            "scientist": "strand-graph-12", "seed": SEED,
            "initial_state": str(CARRY), "initial_state_sha256": carry_sha,
        }},
        proof_distillation_manifest=str(ADAPTER / "manifest.json"),
        proof_distillation_manifest_sha256=sha256(ADAPTER / "manifest.json"),
        proof_adapter_sha256=sha256(ADAPTER / "adapter.pt"),
        source_sha256={
            str(REPO / "scripts/run_focused_successor_branch.py"): sha256(REPO / "scripts/run_focused_successor_branch.py"),
            str(Path(__file__)): sha256(Path(__file__)),
            str(TRAINER): sha256(TRAINER),
            str(PGX_BASE / "src/pgx_mcts_bench/proof_guidance.py"): sha256(PGX_BASE / "src/pgx_mcts_bench/proof_guidance.py"),
            str(REPO / "src/pgx_mcts_bench/networks.py"): sha256(REPO / "src/pgx_mcts_bench/networks.py"),
        },
    )
    atomic_json(GATE, base_gate)
    return base_gate


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another proof-distilled launcher holds the lock") from error
    launch_gate = json.loads(PREPARED.read_text())
    if launch_gate.get("schema") != "focused-proof-distillation-launch-gate-v1" or launch_gate.get("status") != "PREPARED" or not launch_gate.get("launch_permitted"):
        raise RuntimeError("proof distillation launch gate did not pass")
    if sha256(PARENT) != launch_gate["parent_sha256"] or sha256(DATASET) != launch_gate["dataset_sha256"]:
        raise RuntimeError("proof distillation input hash changed")
    for raw, expected in launch_gate["source_sha256"].items():
        if sha256(Path(raw)) != expected:
            raise RuntimeError(f"proof distillation source hash changed: {raw}")
    manifest = prepare_model()
    if not (ADAPTER / "manifest.json").is_file():
        set_status("LAUNCHED", "isolated set-valued proof distillation")
        subprocess.run([
            str(REPO / ".venv/bin/python"), str(TRAINER), "--sidecar", str(DATASET),
            "--model-dir", str(MODEL), "--output-dir", str(ADAPTER), "--pgx-root", str(PGX_BASE),
            "--rf-root", str(RF), "--epochs", "4", "--batch-size", "128", "--seed", str(SEED),
        ], cwd=REPO, check=True)
    carry_sha = integrate(manifest)
    continuation_gate(carry_sha)
    set_status("LAUNCHED", "proof-distilled RL continuation awaiting scientist verification")
    subprocess.run([
        str(REPO / ".venv/bin/python"), str(REPO / "scripts/run_focused_successor_branch.py"),
        "--line", "strand-graph-12-proof-distilled", "--output", str(OUTPUT), "--gate", str(GATE),
        "--checkpoint", str(MODEL / "checkpoint.pt"),
    ], cwd=REPO, check=True)
    (ROOT / "branches/strand-graph-12-proof-distilled/Q304_COMPLETE").touch()
    set_status("COMPLETED", "proof distillation and focused Q304 continuation completed")


if __name__ == "__main__":
    main()
