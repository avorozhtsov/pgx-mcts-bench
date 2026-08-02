#!/usr/bin/env python3
"""Paired critic-target ablation on one established braid checkpoint.

One baseline policy generates the replay data. Two identical checkpoint clones
then receive the same sampled minibatches and optimizer steps; only the scalar
value targets differ. The control clone retains terminal payoff targets while
the treatment clone learns the potential-shaped remaining-return target.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

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
    parser.add_argument("--selfplay-games", type=int, default=24)
    parser.add_argument("--train-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def value_mae(network, positions: list[Position], device: torch.device) -> float:
    network.eval()
    errors: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(positions), 256):
            batch = positions[start : start + 256]
            observations = torch.from_numpy(
                np.stack([position.observation for position in batch])
            ).permute(0, 3, 1, 2).contiguous().to(device=device, dtype=torch.float32)
            _, values = network(observations)
            targets = torch.tensor(
                [position.outcome for position in batch],
                dtype=torch.float32,
                device=device,
            )
            errors.append((values - targets).abs().cpu())
    return float(torch.cat(errors).mean())


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
        config,
        search=replace(config.search, potential_cost_shaping=True),
    )
    game = make_game(config.game)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)

    control = make_braid_network(config.game, config.model).to(device)
    load_policy_value_state_dict(control, payload.get("network", payload))
    treatment = copy.deepcopy(control)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=1e-3, weight_decay=1e-4)
    treatment_optimizer = torch.optim.AdamW(
        treatment.parameters(), lr=1e-3, weight_decay=1e-4
    )
    if "optimizer" in payload:
        control_optimizer.load_state_dict(payload["optimizer"])
        treatment_optimizer.load_state_dict(payload["optimizer"])

    selfplay_seeds = [args.seed + 1_000_000 + index for index in range(args.selfplay_games)]
    search = NeuralMCTS(game, control, config.search, args.device)
    records = play_selfplay_games(
        game,
        search,
        [np.random.default_rng(seed + 7) for seed in selfplay_seeds],
        selfplay_seeds,
        temperature_moves=12,
    )
    control_records = copy.deepcopy(records)
    treatment_records = copy.deepcopy(records)
    for record in treatment_records:
        for position in record:
            position.outcome -= position.value_potential

    control_replay = ReplayBuffer(100_000, np.random.default_rng(args.seed + 99))
    treatment_replay = ReplayBuffer(100_000, np.random.default_rng(args.seed + 99))
    for record in control_records:
        control_replay.add(record)
    for record in treatment_records:
        treatment_replay.add(record)
    control_positions = [position for record in control_records for position in record]
    treatment_positions = [position for record in treatment_records for position in record]

    metrics: dict[str, object] = {
        "schema": "potential-cost-shaping-ablation-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "candidate": args.candidate,
        "stage": args.stage,
        "stage_identity": STAGES[args.stage],
        "selfplay_games": args.selfplay_games,
        "positions": len(control_positions),
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "simulations": config.search.simulations,
        "seed": args.seed,
        "pre_mae": {
            "terminal": value_mae(control, control_positions, device),
            "remaining": value_mae(treatment, treatment_positions, device),
        },
    }
    control_losses: list[dict[str, float]] = []
    treatment_losses: list[dict[str, float]] = []
    for step in range(args.train_steps):
        control_losses.append(
            train_alphazero_step(
                control, control_optimizer, control_replay, args.batch_size, device
            )
        )
        treatment_losses.append(
            train_alphazero_step(
                treatment,
                treatment_optimizer,
                treatment_replay,
                args.batch_size,
                device,
            )
        )
        if (step + 1) % 16 == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "control_value_loss": control_losses[-1]["value"],
                        "treatment_value_loss": treatment_losses[-1]["value"],
                    }
                ),
                flush=True,
            )

    metrics["post_mae"] = {
        "terminal": value_mae(control, control_positions, device),
        "remaining": value_mae(treatment, treatment_positions, device),
    }
    metrics["final_loss"] = {
        "control": control_losses[-1],
        "treatment": treatment_losses[-1],
    }
    torch.save(
        {"network": control.state_dict(), "config": config.to_dict()},
        args.output / "control-terminal.pt",
    )
    torch.save(
        {"network": treatment.state_dict(), "config": shaped_config.to_dict()},
        args.output / "treatment-shaped.pt",
    )
    (args.output / "training-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
