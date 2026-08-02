#!/usr/bin/env python3
"""Paired shaped-scalar versus factorized-critic training on shared replay.

The source policy generates one replay buffer.  Two checkpoint clones receive
identical minibatches and optimizer steps.  The shaped arm learns remaining
return, while the factorized arm learns terminal solve probability and
solve-conditional crossing/move costs and composes those quantities for MCTS.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pgx_mcts_bench.data import Position, ReplayBuffer
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import STAGES, _config, candidates
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games, train_alphazero_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", default="s-window-128")
    parser.add_argument("--stage", type=int, default=21)
    parser.add_argument("--selfplay-games", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def observations(positions: list[Position], device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.stack([position.observation for position in positions]))
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )


def shaped_mae(network, positions: list[Position], device: torch.device) -> float:
    network.eval()
    errors: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(positions), 256):
            batch = positions[start : start + 256]
            _, values = network(observations(batch, device))
            targets = torch.tensor(
                [position.outcome for position in batch], device=device
            )
            errors.append((values - targets).abs().cpu())
    return float(torch.cat(errors).mean())


def factorized_metrics(
    network, positions: list[Position], device: torch.device
) -> dict[str, float]:
    network.eval()
    solve_probability: list[torch.Tensor] = []
    solve_target: list[torch.Tensor] = []
    crossing_prediction: list[torch.Tensor] = []
    move_prediction: list[torch.Tensor] = []
    crossing_target: list[torch.Tensor] = []
    move_target: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(positions), 256):
            batch = positions[start : start + 256]
            obs = observations(batch, device)
            _, _, auxiliary = network.forward_with_auxiliary(obs)
            if auxiliary is None:
                raise RuntimeError("Network has no factorized critic")
            solve_logits, crossings, moves = auxiliary
            eligible = torch.tensor(
                [position.role == 1 and position.solved >= 0 for position in batch],
                dtype=torch.bool,
                device=device,
            )
            solved = torch.tensor(
                [max(position.solved, 0.0) for position in batch], device=device
            )
            solve_probability.append(solve_logits.sigmoid().mean(dim=1)[eligible].cpu())
            solve_target.append(solved[eligible].cpu())
            solved_eligible = eligible & (solved > 0.5)
            crossing_prediction.append(crossings.mean(dim=1)[solved_eligible].cpu())
            move_prediction.append(moves.mean(dim=1)[solved_eligible].cpu())
            crossing_target.append(
                torch.tensor(
                    [position.final_crossing_changes for position in batch], device=device
                )[solved_eligible].cpu()
            )
            move_target.append(
                torch.tensor(
                    [position.final_moves for position in batch], device=device
                )[solved_eligible].cpu()
            )
    probabilities = torch.cat(solve_probability)
    solved = torch.cat(solve_target)
    predicted_crossings = torch.cat(crossing_prediction)
    predicted_moves = torch.cat(move_prediction)
    target_crossings = torch.cat(crossing_target)
    target_moves = torch.cat(move_target)
    return {
        "solve_brier": float(F.mse_loss(probabilities, solved)),
        "crossings_mae_solved": float((predicted_crossings - target_crossings).abs().mean()),
        "moves_mae_solved": float((predicted_moves - target_moves).abs().mean()),
        "eligible_positions": int(solved.numel()),
        "solved_positions": int(target_crossings.numel()),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.stage < len(STAGES):
        raise ValueError(f"stage must be in 0..{len(STAGES) - 1}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    candidate = next(item for item in candidates() if item.name == args.candidate)
    config = _config(
        candidate,
        STAGES[args.stage],
        args.seed,
        args.device,
        frontier=args.stage,
        mix_decay=0.5,
        selfplay_games=args.selfplay_games,
    )
    shaped_config = replace(
        config, search=replace(config.search, potential_cost_shaping=True)
    )
    factorized_config = replace(
        config, model=replace(config.model, use_auxiliary_value=True)
    )
    game = make_game(config.game)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)

    source = make_braid_network(config.game, config.model).to(device)
    load_policy_value_state_dict(source, payload.get("network", payload))
    shaped = copy.deepcopy(source)
    factorized = copy.deepcopy(source)
    factorized.use_auxiliary_value = True
    shaped_optimizer = torch.optim.AdamW(shaped.parameters(), lr=1e-3, weight_decay=1e-4)
    factorized_optimizer = torch.optim.AdamW(
        factorized.parameters(), lr=1e-3, weight_decay=1e-4
    )
    if "optimizer" in payload:
        shaped_optimizer.load_state_dict(payload["optimizer"])
        factorized_optimizer.load_state_dict(payload["optimizer"])

    episode_seeds = [
        args.seed + 1_000_000 + index for index in range(args.selfplay_games)
    ]
    search = NeuralMCTS(game, source, config.search, args.device)
    records = play_selfplay_games(
        game,
        search,
        [np.random.default_rng(seed + 7) for seed in episode_seeds],
        episode_seeds,
        temperature_moves=12,
    )
    shaped_records = copy.deepcopy(records)
    factorized_records = copy.deepcopy(records)
    for record in shaped_records:
        for position in record:
            position.outcome -= position.value_potential

    shaped_replay = ReplayBuffer(200_000, np.random.default_rng(args.seed + 99))
    factorized_replay = ReplayBuffer(200_000, np.random.default_rng(args.seed + 99))
    for record in shaped_records:
        shaped_replay.add(record)
    for record in factorized_records:
        factorized_replay.add(record)
    shaped_positions = [position for record in shaped_records for position in record]
    factorized_positions = [
        position for record in factorized_records for position in record
    ]

    metrics: dict[str, object] = {
        "schema": "factorized-critic-ablation-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "candidate": args.candidate,
        "stage": args.stage,
        "stage_identity": STAGES[args.stage],
        "selfplay_games": args.selfplay_games,
        "positions": len(shaped_positions),
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "simulations": config.search.simulations,
        "seed": args.seed,
        "pre": {
            "shaped_mae": shaped_mae(shaped, shaped_positions, device),
            "factorized": factorized_metrics(factorized, factorized_positions, device),
        },
    }
    shaped_losses: list[dict[str, float]] = []
    factorized_losses: list[dict[str, float]] = []
    for step in range(args.train_steps):
        shaped_losses.append(
            train_alphazero_step(
                shaped, shaped_optimizer, shaped_replay, args.batch_size, device
            )
        )
        factorized_losses.append(
            train_alphazero_step(
                factorized,
                factorized_optimizer,
                factorized_replay,
                args.batch_size,
                device,
            )
        )
        if (step + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "shaped_value_loss": shaped_losses[-1]["value"],
                        "factorized_value_loss": factorized_losses[-1]["value"],
                        "factorized_auxiliary_loss": factorized_losses[-1]["auxiliary"],
                    }
                ),
                flush=True,
            )

    metrics["post"] = {
        "shaped_mae": shaped_mae(shaped, shaped_positions, device),
        "factorized": factorized_metrics(factorized, factorized_positions, device),
    }
    metrics["final_loss"] = {
        "shaped": shaped_losses[-1],
        "factorized": factorized_losses[-1],
    }
    torch.save(
        {"network": shaped.state_dict(), "config": shaped_config.to_dict()},
        args.output / "shaped-scalar.pt",
    )
    torch.save(
        {"network": factorized.state_dict(), "config": factorized_config.to_dict()},
        args.output / "factorized.pt",
    )
    (args.output / "training-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
