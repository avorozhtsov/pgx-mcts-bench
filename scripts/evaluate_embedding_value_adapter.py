#!/usr/bin/env python3
"""Paired equal-budget MCTS diagnostic for the frozen-embedding value adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder

from pgx_mcts_bench.embedding_value_adapter import FrozenEmbeddingValueAdapter
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import STAGES, Candidate, _config, evaluate_stage
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_embedding(checkpoint: Path, device: str) -> ScalableBraidEmbeddingEncoder:
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    config = saved["config"]
    model = ScalableBraidEmbeddingEncoder(
        width=int(config["width"]),
        depth=int(config["depth"]),
        embedding_dim=int(config["embedding_dim"]),
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def paired_cost_outcome(row: dict) -> str:
    """Compare the actual U-witness objective, not only whether both sides solved."""
    baseline_solved = bool(row["baseline_solved"])
    adapter_solved = bool(row["adapter_solved"])
    if adapter_solved and not baseline_solved:
        return "adapter_better"
    if baseline_solved and not adapter_solved:
        return "baseline_better"
    if not baseline_solved:
        return "both_failed"
    baseline_cost = (int(row["baseline_crossings"]), int(row["baseline_moves"]))
    adapter_cost = (int(row["adapter_crossings"]), int(row["adapter_moves"]))
    if adapter_cost < baseline_cost:
        return "adapter_better"
    if baseline_cost < adapter_cost:
        return "baseline_better"
    return "equal_cost"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", type=int, action="append", required=True)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.games < 1 or args.simulations < 1:
        raise ValueError("invalid evaluation budget")

    manifest = json.loads(args.parent_manifest.read_text())
    candidate = Candidate(**manifest["candidate"])
    base_config = _config(candidate, STAGES[0], args.seed, args.device, selfplay_games=1)
    base_config = replace(
        base_config,
        search=replace(base_config.search, simulations=args.simulations),
    )
    parent = make_braid_network(base_config.game, base_config.model).to(args.device)
    parent_saved = torch.load(args.parent_checkpoint, map_location=args.device, weights_only=False)
    load_policy_value_state_dict(parent, parent_saved.get("network", parent_saved))
    parent.eval()

    adapter_saved = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    adapter = FrozenEmbeddingValueAdapter(
        parent,
        load_embedding(args.embedding_checkpoint, args.device),
        observation_channels=base_config.game.observation_channels,
        embedding_dim=int(adapter_saved["embedding_dim"]),
        width=int(adapter_saved["width"]),
        maximum_value_residual=float(adapter_saved["maximum_value_residual"]),
        use_embedding=bool(adapter_saved.get("use_embedding", True)),
    ).to(args.device)
    adapter.load_state_dict(adapter_saved["adapter"], strict=False)
    adapter.eval()

    results = []
    for stage_index in args.stage:
        if not 0 <= stage_index < len(STAGES):
            raise ValueError(f"stage {stage_index} is outside 0..{len(STAGES) - 1}")
        config = _config(
            candidate,
            STAGES[stage_index],
            args.seed,
            args.device,
            selfplay_games=1,
        )
        config = replace(config, search=replace(config.search, simulations=args.simulations))
        game = make_game(config.game)
        stage_seed = args.seed + 1_000_003 * stage_index
        baseline = evaluate_stage(
            game,
            parent,
            config,
            args.games,
            stage_seed,
            ratios=(1000.0,),
            include_attempts=True,
        )[1000.0]
        adapted = evaluate_stage(
            game,
            adapter,
            config,
            args.games,
            stage_seed,
            ratios=(1000.0,),
            include_attempts=True,
        )[1000.0]
        paired = []
        for base_attempt, adapter_attempt in zip(
            baseline["attempts"], adapted["attempts"], strict=True
        ):
            if (base_attempt["seed"], base_attempt["word"], base_attempt["strands"]) != (
                adapter_attempt["seed"],
                adapter_attempt["word"],
                adapter_attempt["strands"],
            ):
                raise RuntimeError("paired evaluation generated different instances")
            paired.append(
                {
                    "seed": base_attempt["seed"],
                    "baseline_solved": base_attempt["solved"],
                    "adapter_solved": adapter_attempt["solved"],
                    "baseline_crossings": base_attempt["crossings"],
                    "adapter_crossings": adapter_attempt["crossings"],
                    "baseline_moves": base_attempt["moves"],
                    "adapter_moves": adapter_attempt["moves"],
                }
            )
        results.append(
            {
                "stage_index": stage_index,
                "stage": STAGES[stage_index],
                "baseline": baseline,
                "adapter": adapted,
                "paired": paired,
            }
        )

    baseline_wins = adapter_wins = ties = 0
    cost_outcomes = {
        "adapter_better": 0,
        "baseline_better": 0,
        "equal_cost": 0,
        "both_failed": 0,
    }
    for stage in results:
        for row in stage["paired"]:
            pair = (row["baseline_solved"], row["adapter_solved"])
            if pair == (False, True):
                adapter_wins += 1
            elif pair == (True, False):
                baseline_wins += 1
            else:
                ties += 1
            cost_outcomes[paired_cost_outcome(row)] += 1
    report = {
        "schema": "pgx-frozen-embedding-value-adapter-mcts-ablation-v0",
        "status": "small_development_diagnostic",
        "inputs": {
            "parent_checkpoint_sha256": sha256(args.parent_checkpoint),
            "embedding_checkpoint_sha256": sha256(args.embedding_checkpoint),
            "adapter_checkpoint_sha256": sha256(args.adapter_checkpoint),
        },
        "protocol": {
            "same_generated_instances_and_seeds": True,
            "same_policy_logits": True,
            "same_simulations": args.simulations,
            "games_per_stage": args.games,
            "ratio": 1000.0,
            "root_noise": False,
            "temperature": 0.0,
            "development_only": True,
        },
        "paired_solve_outcomes": {
            "adapter_wins": adapter_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
        },
        "paired_witness_cost_outcomes": cost_outcomes,
        "unique_initial_instances": len(
            {
                (tuple(attempt["word"]), attempt["strands"])
                for stage in results
                for attempt in stage["baseline"]["attempts"]
            }
        ),
        "embedding_cache": adapter.embedding_cache_stats,
        "stages": results,
        "interpretation_contract": (
            "This is a small equal-budget development diagnostic, not a promotion gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.part")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
