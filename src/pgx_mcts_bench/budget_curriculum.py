"""Small, rollback-guarded curriculum for a collaboration-roster budget critic."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import (
    COLLABORATION_K3,
    FixedWordGame,
    load_scientist,
    smallest_crossing_pool,
)
from pgx_mcts_bench.collaborative_scientists import _atomic_json, _observation_tensor, _sha256
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, candidates, evaluate_stage
from pgx_mcts_bench.rapid_adaptation import promoted_checkpoint_metadata
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games, train_alphazero_step


def _budget_curve(scientist, items, ratio: float, caps: tuple[float, ...]) -> list[dict[str, Any]]:
    observations = [
        FixedWordGame(scientist.game, item, ratio, objective_cap=cap).reset(0).observation
        for item in items
        for cap in caps
    ]
    tensor = _observation_tensor(observations, torch.device(scientist.config.train.device))
    with torch.inference_mode():
        probability = (
            scientist.network.eval()
            .forward_with_auxiliary(tensor)[2][0]
            .sigmoid()
            .mean(dim=1)
            .reshape(len(items), len(caps))
        )
    return [
        {
            "item": item.name,
            "p_solve": values.tolist(),
            "spread": float(values.max() - values.min()),
            "monotone": bool(torch.all(values[1:] >= values[:-1] - 1e-7)),
        }
        for item, values in zip(items, probability.cpu(), strict=True)
    ]


def _rung_evaluation(
    scientist,
    metadata: dict[str, Any],
    games: int,
    seed: int,
    simulations: int | None = None,
) -> dict[str, Any]:
    candidate = next(candidate for candidate in candidates() if candidate.name == scientist.name)
    if simulations is not None:
        candidate = replace(candidate, simulations=simulations)
    config = _config(
        candidate,
        (metadata["source"], metadata["scramble"]),
        seed,
        scientist.config.train.device,
        selfplay_games=1,
    )
    config = replace(
        config,
        game=replace(config.game, objective_budget_channel=True),
        model=scientist.config.model,
    )
    measured = evaluate_stage(
        make_game(config.game),
        scientist.network,
        config,
        games,
        seed,
        ratios=(10.0,),
    )[10.0]
    return {
        "games": games,
        "solve_rate": float(measured["solved"]),
        "crossings": float(measured["crossings"]),
        "moves": float(measured["moves"]),
    }


def _promoted_rung_rehearsal(
    scientist,
    metadata: dict[str, Any],
    games: int,
    simulations: int,
    seed: int,
) -> list:
    if games <= 0:
        return []
    candidate = next(candidate for candidate in candidates() if candidate.name == scientist.name)
    candidate = replace(candidate, simulations=simulations)
    config = _config(
        candidate,
        (metadata["source"], metadata["scramble"]),
        seed,
        scientist.config.train.device,
        selfplay_games=1,
    )
    config = replace(
        config,
        game=replace(config.game, objective_budget_channel=True),
        model=scientist.config.model,
    )
    game = make_game(config.game)
    search = NeuralMCTS(game, scientist.network, config.search, config.train.device)
    seeds = [seed + index for index in range(games)]
    return play_selfplay_games(
        game,
        search,
        [np.random.default_rng(game_seed + 7) for game_seed in seeds],
        seeds,
        config.train.temperature_moves,
    )


def train_budget_curriculum(
    checkpoint: Path,
    output: Path,
    *,
    scientist_name: str = "s-window-128",
    items: int = 5,
    ratio: float = 10.0,
    cap_fractions: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 1.0),
    games_per_cap: int = 2,
    train_steps_per_item: int = 32,
    simulations: int = 32,
    rung_eval_games: int = 4,
    rehearsal_games: int = 8,
    seed: int = 20261040,
    device: str = "cpu",
) -> dict[str, Any]:
    if scientist_name not in COLLABORATION_K3:
        raise ValueError(
            f"budget-critic curriculum is admitted only for {COLLABORATION_K3}, "
            f"not {scientist_name!r}"
        )
    metadata = promoted_checkpoint_metadata(checkpoint, scientist_name)
    scientist = load_scientist(
        scientist_name,
        checkpoint,
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    preservation_teacher = copy.deepcopy(scientist.network).eval()
    for parameter in preservation_teacher.parameters():
        parameter.requires_grad_(False)
    # Bypass nn.Module registration: the teacher is a runtime training guard,
    # not a child whose weights should be serialized into every checkpoint.
    object.__setattr__(
        scientist.network,
        "_policy_value_preservation_teacher",
        preservation_teacher,
    )
    simple = smallest_crossing_pool(items)
    global_cap = (ratio + 1.0) * scientist.config.game.simplify_budget
    caps = tuple(max(1.0, round(global_cap * fraction)) for fraction in cap_fractions)
    before_state = copy.deepcopy(scientist.network.state_dict())
    before_curve = _budget_curve(scientist, simple, ratio, caps)
    before_rung = _rung_evaluation(scientist, metadata, rung_eval_games, seed + 800_000_000)
    rehearsal = _promoted_rung_rehearsal(
        scientist,
        metadata,
        rehearsal_games,
        simulations,
        seed + 700_000_000,
    )
    for record in rehearsal:
        scientist.replay.add(record)

    rows = []
    last_metrics = None
    for item_index, item in enumerate(simple):
        cap_rows = []
        for cap_index, cap in enumerate(caps):
            fixed = FixedWordGame(scientist.game, item, ratio, objective_cap=cap)
            search = NeuralMCTS(fixed, scientist.network, scientist.config.search, device)
            seeds = [
                seed + item_index * 1_000_000 + cap_index * 10_000 + game_index
                for game_index in range(games_per_cap)
            ]
            records = play_selfplay_games(
                fixed,
                search,
                [np.random.default_rng(game_seed + 7) for game_seed in seeds],
                seeds,
                scientist.config.train.temperature_moves,
            )
            for record in records:
                scientist.replay.add(record)
            cap_rows.append(
                {
                    "cap": cap,
                    "games": len(records),
                    "solved": sum(bool(record and record[0].solved > 0.5) for record in records),
                    "censored": sum(
                        bool(record and record[0].objective_censored) for record in records
                    ),
                }
            )
        for _ in range(train_steps_per_item):
            last_metrics = train_alphazero_step(
                scientist.network,
                scientist.optimizer,
                scientist.replay,
                32,
                torch.device(device),
                collaboration_replay=True,
                shared_fraction=0.0,
            )
        rows.append({"item": item.name, "crossings": item.crossings, "caps": cap_rows})

    trained_state = copy.deepcopy(scientist.network.state_dict())
    after_curve = _budget_curve(scientist, simple, ratio, caps)
    after_rung = _rung_evaluation(scientist, metadata, rung_eval_games, seed + 800_000_000)
    outcomes_by_item = {
        row["item"]: [cap["solved"] > 0 for cap in row["caps"]]
        for row in rows
    }
    informative = {
        item
        for item, outcomes in outcomes_by_item.items()
        if any(outcomes) and not all(outcomes)
    }
    curves_by_item = {row["item"]: row for row in after_curve}
    informative_sensitive = sum(
        curves_by_item[item]["spread"] > 1e-4 for item in informative
    )
    sensitive = sum(row["spread"] > 1e-4 for row in after_curve)
    monotone = sum(row["monotone"] for row in after_curve)
    accepted = (
        after_rung["solve_rate"] >= before_rung["solve_rate"]
        and bool(informative)
        and informative_sensitive == len(informative)
        and monotone == len(simple)
    )
    if not accepted:
        scientist.network.load_state_dict(before_state)

    output.mkdir(parents=True, exist_ok=True)
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    torch.save(
        {
            **source_payload,
            "network": trained_state,
            "budget_curriculum": {"accepted": accepted, "source": str(checkpoint.resolve())},
        },
        output / "trained-candidate.pt",
    )
    torch.save(
        {
            **source_payload,
            "network": scientist.network.state_dict(),
            "budget_curriculum": {"accepted": accepted, "source": str(checkpoint.resolve())},
        },
        output / "rollback-guarded.pt",
    )
    report = {
        "schema": "collaboration-budget-curriculum-v2",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "scientist": scientist_name,
        "architecture": {
            "solve_encoder_gradients": True,
            "cost_first": True,
            "manual_objective": "L=A*cc+B*moves",
            "budget_skips": ["shared_body", "scalar_value", "cc_moves", "p_solve"],
        },
        "protocol": {
            "items": items,
            "item_names": [item.name for item in simple],
            "ratio": ratio,
            "caps": list(caps),
            "games_per_cap": games_per_cap,
            "train_steps_per_item": train_steps_per_item,
            "simulations": simulations,
            "rung_eval_games": rung_eval_games,
            "rehearsal_games": rehearsal_games,
            "seed": seed,
            "device": device,
        },
        "training": rows,
        "last_metrics": last_metrics,
        "before": {"curve": before_curve, "promoted_rung": before_rung},
        "after": {"curve": after_curve, "promoted_rung": after_rung},
        "decision": {
            "accepted": accepted,
            "sensitive_items": sensitive,
            "informative_items": sorted(informative),
            "informative_sensitive": informative_sensitive,
            "monotone_items": monotone,
            "rollback_applied": not accepted,
            "next": (
                "expand the easy curriculum" if accepted else "restart from the earliest rung"
            ),
        },
        "model_config": asdict(scientist.config.model),
    }
    _atomic_json(output / "report.json", report)
    return report
