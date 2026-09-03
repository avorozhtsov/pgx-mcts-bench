#!/usr/bin/env python3
"""Prepare the hash-bound Q304 proof+embedding child without launching it."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder

REPO = Path(__file__).resolve().parents[1]
ROOT = (
    REPO / "artifacts/local-q-skm-ablation-20260817/continuation/"
    "q4000-v1-population-20260818/focused-successor-v1"
)
PROOF_BRANCH = (
    ROOT / "branches/strand-graph-12-proof-distilled/"
    "q50-4-updated-scheduled-no-sharing-bounded-proof-recovery-v2"
)
EMBEDDING_TOURNAMENT = Path(
    "/Users/artemvorozhtsov/projects/rf-knots/artifacts/"
    "embedding-tournament-rung1-5x1m-v0/tournament.json"
)
BASE_PARENT_MANIFEST = (
    REPO / "artifacts/nebius-main32-final-20260817/artifacts/"
    "q4000-strand12-20260814/migrated/strand-graph-12/manifest.json"
)
OUTPUT = ROOT / "branches/strand-graph-12-proof-embedding/bridge-v1"
INPUTS = OUTPUT / "inputs"
GATE = ROOT / "FOCUSED_PROOF_EMBEDDING_CHILD_V1_VERIFIED.json"
POLICY = REPO / "research/local-q-skm-ablation/focused-successor-v1-policy.json"


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


def atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    torch.save(payload, temporary)
    temporary.replace(path)


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def selected_embedding(manifest_path: Path = EMBEDDING_TOURNAMENT) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "rf-knots-embedding-checkpoint-tournament-v0":
        raise RuntimeError("unexpected embedding selection schema")
    if manifest.get("status") != "completed":
        raise RuntimeError("embedding tournament is not terminal")
    selected = manifest.get("selected") or {}
    checkpoint = Path(str(selected.get("checkpoint", "")))
    if selected.get("status") != "completed" or not checkpoint.is_file():
        raise RuntimeError("selected embedding checkpoint is not complete")
    if sha256(checkpoint) != selected.get("checkpoint_sha256"):
        raise RuntimeError("selected embedding checkpoint hash differs")
    sources = manifest.get("source_sha256") or {}
    required_sources = {"pairs", "geodesic", "hard_cases", "gordian", "validation_protocol"}
    if set(sources) != required_sources or not all(
        len(str(value)) == 64 for value in sources.values()
    ):
        raise RuntimeError("embedding selection does not bind every required source hash")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("schema") != "rf-knots-embedding-calibration-v0":
        raise RuntimeError("unsupported embedding checkpoint schema")
    config = saved.get("config") or {}
    if int(config.get("embedding_dim", -1)) != 64:
        raise RuntimeError("whole-word embedding dimension must be 64")
    model = ScalableBraidEmbeddingEncoder(
        width=int(config["width"]),
        depth=int(config["depth"]),
        embedding_dim=int(config["embedding_dim"]),
    )
    model.load_state_dict(saved["model"])
    model.eval()
    return {"manifest": manifest, "selected": selected, "checkpoint": checkpoint}


def extract_parent() -> tuple[Path, Path]:
    state_path = PROOF_BRANCH / "state.pt.gz"
    marker = PROOF_BRANCH.parent / "Q304_COMPLETE"
    if not marker.exists() or not state_path.exists():
        raise RuntimeError("proof-distilled Q304 parent is not terminal")
    with gzip.open(state_path, "rb") as handle:
        state = torch.load(handle, map_location="cpu", weights_only=False)
    scientist = (state.get("scientists") or {}).get("strand-graph-12") or {}
    network = scientist.get("network")
    if not isinstance(network, dict) or not network:
        raise RuntimeError("proof-distilled Q304 state has no strand-graph network")
    checkpoint = INPUTS / "proof-distilled-q304-network.pt"
    atomic_torch(
        checkpoint,
        {
            "network": network,
            "schema": "focused-proof-distilled-q304-network-v1",
            "source_state_sha256": sha256(state_path),
        },
    )
    base = json.loads(BASE_PARENT_MANIFEST.read_text())
    manifest = INPUTS / "proof-distilled-q304-parent.json"
    atomic_json(
        manifest,
        {
            "schema": "focused-proof-distilled-q304-parent-v1",
            "candidate": base["candidate"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "source_state": str(state_path.resolve()),
            "source_state_sha256": sha256(state_path),
        },
    )
    return manifest, checkpoint


def main() -> None:
    if REPO != Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench"):
        raise RuntimeError("embedding child must be prepared from the main checkout")
    branch = git_value("branch", "--show-current")
    if branch != "main":
        raise RuntimeError(f"embedding child requires main, found {branch}")
    if git_value("rev-parse", "HEAD") != git_value("rev-parse", "origin/main"):
        raise RuntimeError("main must match origin/main before gate creation")
    selected = selected_embedding()
    parent_manifest, parent_checkpoint = extract_parent()
    source_paths = [
        REPO / "src/pgx_mcts_bench/embedding_value_adapter.py",
        REPO / "src/pgx_mcts_bench/on_policy_embedding_value.py",
        REPO / "src/pgx_mcts_bench/search.py",
        REPO / "scripts/collect_on_policy_embedding_value.py",
        REPO / "scripts/train_on_policy_embedding_value_adapter.py",
        REPO / "scripts/evaluate_embedding_value_adapter.py",
        REPO / "scripts/run_focused_successor_embedding_child.py",
        Path("/Users/artemvorozhtsov/projects/rf-knots/src/rf_knots/embedding_encoder.py"),
    ]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing hash-bound sources: {missing}")
    tournament = selected["manifest"]
    payload = {
        "schema": "focused-proof-embedding-child-gate-v1",
        "status": "PREPARED",
        "checkout": str(REPO),
        "branch": branch,
        "commit": git_value("rev-parse", "HEAD"),
        "policy": {"path": str(POLICY), "sha256": sha256(POLICY)},
        "parent": {
            "manifest": str(parent_manifest),
            "manifest_sha256": sha256(parent_manifest),
            "checkpoint": str(parent_checkpoint),
            "checkpoint_sha256": sha256(parent_checkpoint),
            "terminal_marker": str(PROOF_BRANCH.parent / "Q304_COMPLETE"),
        },
        "embedding": {
            "selection_manifest": str(EMBEDDING_TOURNAMENT),
            "selection_manifest_sha256": sha256(EMBEDDING_TOURNAMENT),
            "checkpoint": str(selected["checkpoint"]),
            "checkpoint_sha256": selected["selected"]["checkpoint_sha256"],
            "source_sha256": tournament["source_sha256"],
            "embedding_dim": 64,
        },
        "training": {
            "parent_frozen": True,
            "embedding_frozen": True,
            "policy_unchanged": True,
            "bridge_initialization": "exact-zero-residual",
            "train_games_per_stage": 10,
            "validation_games_per_stage": 5,
            "simulations": 32,
            "adapter_steps": 1500,
            "seed": 202609030305,
        },
        "sources": {str(path): sha256(path) for path in source_paths},
        "output": str(OUTPUT),
    }
    atomic_json(GATE, payload)
    print(json.dumps({"gate": str(GATE), "sha256": sha256(GATE)}, indent=2))


if __name__ == "__main__":
    main()
