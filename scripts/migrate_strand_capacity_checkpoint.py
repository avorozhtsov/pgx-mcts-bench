#!/usr/bin/env python3
"""Create a pinned, semantics-aware strand-capacity checkpoint child."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from pgx_mcts_bench.adaptive_scientists import load_scientist


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--child", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    parent = load_scientist(
        args.parent,
        args.source_checkpoint,
        seed=args.seed,
        device="cpu",
        require_factorized=True,
        objective_budget_channel=True,
    )
    child = load_scientist(
        args.child,
        args.source_checkpoint,
        seed=args.seed,
        device="cpu",
        require_factorized=True,
        objective_budget_channel=True,
    )
    if child.game.config.max_strands <= parent.game.config.max_strands:
        raise ValueError("child capacity must exceed parent capacity")

    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / "checkpoint.pt"
    torch.save(
        {
            "network": child.network.state_dict(),
            "capacity_migration": {
                "schema": "semantic-strand-capacity-migration-v1",
                "parent": args.parent,
                "child": args.child,
                "seed": args.seed,
            },
        },
        checkpoint,
    )
    manifest = {
        "schema": "semantic-strand-capacity-migration-v1",
        "parent": args.parent,
        "child": args.child,
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": sha256(args.source_checkpoint),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_observation_channels": parent.game.config.observation_channels,
        "target_observation_channels": child.game.config.observation_channels,
        "source_actions": parent.game.num_actions,
        "target_actions": child.game.num_actions,
        "candidate": asdict(child.candidate),
        "migration": (
            "all shape-stable tensors copied exactly; old positional action rows "
            "and crossing-change row copied semantically; new generator rows freshly initialized"
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
