#!/usr/bin/env python3
"""Prepare and audit a mastery-v3 checkpoint fork without launching it."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import load_scientist
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, mastery_v3_arms
from pgx_mcts_bench.mastery_v3 import (
    CyclicMemoryDeepV3,
    migration_max_abs_difference,
)
from pgx_mcts_bench.networks import make_braid_network


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observation_tensor(observation: np.ndarray, device: str) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(observation))
        .permute(2, 0, 1)[None]
        .float()
        .to(device)
    )


def load_source_payload(path: Path, device: str) -> dict[str, object]:
    """Load either a ladder checkpoint or a durable compressed SKM state."""

    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            payload = torch.load(stream, map_location=device, weights_only=False)
    else:
        payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "network" not in payload:
        raise ValueError(f"{path} does not contain a scientist network payload")
    return payload


@torch.inference_mode()
def migrate(
    source_checkpoint: Path,
    output: Path,
    *,
    candidate_name: str,
    seed: int,
    device: str,
    tolerance: float,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source_payload = load_source_payload(source_checkpoint, device)
    with tempfile.NamedTemporaryFile(suffix=".pt") as source_file:
        torch.save(source_payload, source_file.name)
        source = load_scientist(
            "cyclic-memory-12",
            Path(source_file.name),
            seed=seed,
            device=device,
            require_factorized=True,
            objective_budget_channel=True,
        )
    candidate = next(
        arm for arm in mastery_v3_arms() if arm.name == candidate_name
    )
    config = _config(
        candidate,
        ("R(3,12)#0", 0),
        seed,
        device,
        selfplay_games=1,
    )
    child = make_braid_network(config.game, config.model).to(device).eval()
    if not isinstance(child, CyclicMemoryDeepV3):
        raise TypeError(f"unexpected child network {type(child).__name__}")
    child.load_parent_state_dict(source.network.state_dict())

    child_game = make_game(config.game)
    maxima = {
        "policy": 0.0,
        "value": 0.0,
        "p_solve_logits": 0.0,
        "conditional_crossings": 0.0,
        "conditional_moves": 0.0,
    }
    probes = (
        ([1], 2),
        ([1, 1, 1], 2),
        ([1, 2, 1, 2], 3),
        (list(range(1, 12)), 12),
    )
    probe_rows = []
    for word, strands in probes:
        for ratio in (10.0, 1000.0):
            parent_transition = source.game.from_word(word, strands, math.log(ratio))
            child_transition = child_game.from_word(word, strands, math.log(ratio))
            differences = migration_max_abs_difference(
                source.network,
                child,
                observation_tensor(parent_transition.observation, device),
                observation_tensor(child_transition.observation, device),
            )
            for name, value in differences.items():
                maxima[name] = max(maxima[name], value)
            probe_rows.append(
                {
                    "word": word,
                    "strands": strands,
                    "ratio": ratio,
                    "max_abs_difference": differences,
                }
            )
    maximum = max(maxima.values())
    if maximum > tolerance:
        raise AssertionError(
            f"function-preserving migration failed: {maximum} > {tolerance}"
        )

    repo_root = Path(__file__).resolve().parents[1]
    source_files = (
        repo_root / "src/pgx_mcts_bench/mastery_v3.py",
        repo_root / "src/pgx_mcts_bench/gpu_inference.py",
        repo_root / "research/mastery-v3-curriculum/protocol-spec.json",
        repo_root / "research/mastery-v3-curriculum/curriculum.json",
    )
    initialization = {
        "schema": "mastery-v3-function-preserving-migration-v1",
        "candidate": candidate_name,
        "source_scientist": "cyclic-memory-12",
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "seed": seed,
        "tolerance": tolerance,
        "max_abs_difference": maxima,
        "passed": True,
        "probe_rows": probe_rows,
        "parameter_report": child.parameter_report(),
        "source_hashes": {
            str(path.relative_to(repo_root)): sha256(path) for path in source_files
        },
        "launched": False,
    }
    payload = {
        "network": child.state_dict(),
        "candidate_spec": asdict(candidate),
        "mastery_v3_migration": initialization,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        **initialization,
        "checkpoint": str(output.resolve()),
        "checkpoint_sha256": sha256(output),
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=("cyclic-memory-deep-v3", "cyclic-graph-dual-v3"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=2026081700)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    report = migrate(
        args.source_checkpoint,
        args.output,
        candidate_name=args.candidate,
        seed=args.seed,
        device=args.device,
        tolerance=args.tolerance,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
