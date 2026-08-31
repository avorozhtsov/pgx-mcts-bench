#!/usr/bin/env python3
"""Write a non-launching, hash-bound readiness snapshot for focused successor v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from focused_successor_policy import POLICY, load_policy

POPULATION = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818"
)
OUTPUT = POPULATION / "focused-successor-v1/FOCUSED_SUCCESSOR_V1_READINESS.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_binding(path: Path) -> dict[str, Any]:
    binding: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        binding["sha256"] = sha256(path)
    return binding


def terminal_parent_binding(marker: Path, label: str) -> dict[str, Any]:
    branch = marker.parent / "branches" / label
    completion = branch / "Q254_COMPLETE"
    stage = branch / "q50-3-updated-scheduled-no-sharing-bounded-repair-v2"
    files = {
        name: file_binding(stage / name)
        for name in ("report.json", "terminal-retention-audit.json", "state.pt.gz")
    }
    complete = completion.is_file() and all(value["exists"] for value in files.values())
    return {
        "status": "COMPLETED" if complete else "QUEUED",
        "completion_marker": file_binding(completion),
        "files": files,
    }


def proof_binding(policy: dict[str, Any]) -> dict[str, Any]:
    contract = policy["proof_supervision"]
    dataset = Path(contract["dataset"])
    manifest_path = Path(contract["reference_manifest"])
    result: dict[str, Any] = {
        "status": "QUEUED",
        "dataset": file_binding(dataset),
        "reference_manifest": file_binding(manifest_path),
    }
    if not dataset.is_file() or not manifest_path.is_file():
        return result
    manifest = json.loads(manifest_path.read_text())
    declared = manifest.get("dataset", {})
    ready = (
        sha256(dataset) == contract["dataset_sha256"]
        and declared.get("sha256") == contract["dataset_sha256"]
        and declared.get("unknown_actions_receive_graph_gradient") is False
    )
    result.update(
        {
            "status": "PREPARED" if ready else "BLOCKED",
            "manifest_schema": manifest.get("schema"),
            "manifest_status": manifest.get("status"),
            "replay_contract_verified": ready,
        }
    )
    return result


def embedding_binding(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "QUEUED",
            "detail": "terminal embedding selection manifest has not been bound",
        }
    result: dict[str, Any] = {"status": "BLOCKED", "manifest": file_binding(path)}
    if not path.is_file():
        return result
    manifest = json.loads(path.read_text())
    selected = manifest.get("selected", {})
    checkpoint = Path(selected.get("checkpoint", ""))
    expected = selected.get("checkpoint_sha256")
    valid = (
        manifest.get("status") == "completed"
        and selected.get("status") == "completed"
        and checkpoint.is_file()
        and isinstance(expected, str)
        and sha256(checkpoint) == expected
    )
    result.update(
        {
            "status": "PREPARED" if valid else "BLOCKED",
            "manifest_schema": manifest.get("schema"),
            "checkpoint": file_binding(checkpoint),
        }
    )
    return result


def audit(embedding_selection_manifest: Path | None = None) -> dict[str, Any]:
    policy = load_policy()
    prerequisites = policy["transition_prerequisites"]
    fast_marker = Path(prerequisites["fast_parent_marker"])
    slow_marker = Path(prerequisites["slow_selection_marker"])
    parents = {
        label: terminal_parent_binding(fast_marker, label)
        for label in ("q-grown-strand-graph-12", "q-grown-raster-axial-12")
    }
    proof = proof_binding(policy)
    embedding = embedding_binding(embedding_selection_manifest)
    fast_complete = fast_marker.is_file()
    slow_complete = slow_marker.is_file()
    ready = (
        fast_complete
        and slow_complete
        and all(parent["status"] == "COMPLETED" for parent in parents.values())
        and proof["status"] == "PREPARED"
        and embedding["status"] == "PREPARED"
    )
    return {
        "schema": "focused-successor-v1-readiness",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PREPARED" if ready else "QUEUED",
        "launch_permitted": ready,
        "policy": file_binding(POLICY),
        "legacy_q304_launch_authorized": policy["legacy_q304"]["launch_authorized"],
        "fast_terminal_marker": file_binding(fast_marker),
        "slow_terminal_marker": file_binding(slow_marker),
        "parents": parents,
        "proof_supervision": proof,
        "embedding": embedding,
        "next_action": (
            "write exact transition gate and launch focused cohort"
            if ready
            else "wait for and bind every missing terminal prerequisite"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-selection-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit(args.embedding_selection_manifest)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
