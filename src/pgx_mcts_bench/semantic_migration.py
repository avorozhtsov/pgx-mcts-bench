"""Exact migration gate for adding the remaining semantic-objective channel."""

from __future__ import annotations

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
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _sha256,
    verified_record_cost,
)
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games


def _tensor(observation: np.ndarray, device: str) -> torch.Tensor:
    return (
        torch.from_numpy(np.array(observation[None], copy=True))
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )


def _outputs(scientist, observation: np.ndarray) -> dict[str, torch.Tensor]:
    scientist.network.eval()
    with torch.inference_mode():
        policy, legacy, auxiliary = scientist.network.forward_with_auxiliary(
            _tensor(observation, scientist.config.train.device)
        )
    solve, crossings, moves = auxiliary
    return {
        "policy": policy.cpu(),
        "legacy_value": legacy.cpu(),
        "p_solve_logits": solve.cpu(),
        "crossings": crossings.cpu(),
        "semantic_moves": moves.cpu(),
    }


def _attempt(scientist, item, ratio: float, simulations: int, seed: int) -> dict[str, Any]:
    game = FixedWordGame(scientist.game, item, ratio)
    search = NeuralMCTS(
        game,
        scientist.network,
        scientist.config.search,
        scientist.config.train.device,
    )
    record = play_selfplay_games(
        game,
        search,
        [np.random.default_rng(seed + 7)],
        [seed],
        scientist.config.train.temperature_moves,
    )[0]
    verified = verified_record_cost(scientist.game, item, ratio, record)
    return {
        "actions": [int(position.action) for position in record],
        "verified_semantic_cost": (
            [int(verified[0]), int(verified[1])] if verified is not None else None
        ),
        "objective_censored": bool(record and record[0].objective_censored),
        "simulations": simulations,
    }


def run_semantic_budget_migration_gate(
    checkpoints: dict[str, Path],
    output: Path,
    *,
    items: int = 3,
    ratios: tuple[float, ...] = (0.1, 10.0, 1000.0),
    behavior_items: int = 2,
    simulations: int = 8,
    seed: int = 20261130,
    device: str = "cpu",
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Prove a zero-weight channel migration preserves real checkpoint behavior."""
    unexpected = sorted(set(checkpoints) - set(COLLABORATION_K3))
    if unexpected:
        raise ValueError(f"scientists are outside the collaboration roster: {unexpected}")
    if not checkpoints:
        raise ValueError("at least one scientist checkpoint is required")
    if min(items, behavior_items, simulations) < 1:
        raise ValueError("items, behavior_items, and simulations must be positive")
    pool = smallest_crossing_pool(max(items, behavior_items))
    reports: dict[str, Any] = {}
    for scientist_index, (name, checkpoint) in enumerate(checkpoints.items()):
        scientist_seed = seed + scientist_index * 1_000_000
        old = load_scientist(
            name,
            checkpoint,
            seed=scientist_seed,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=False,
        )
        migrated = load_scientist(
            name,
            checkpoint,
            seed=scientist_seed,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=True,
        )
        comparisons = []
        maxima: dict[str, float] = {}
        for item in pool[:items]:
            for ratio in ratios:
                old_transition = FixedWordGame(old.game, item, ratio).reset(0)
                new_transition = FixedWordGame(migrated.game, item, ratio).reset(0)
                old_outputs = _outputs(old, old_transition.observation)
                new_outputs = _outputs(migrated, new_transition.observation)
                differences = {
                    key: float((old_outputs[key] - new_outputs[key]).abs().max().item())
                    for key in old_outputs
                }
                for key, difference in differences.items():
                    maxima[key] = max(maxima.get(key, 0.0), difference)
                comparisons.append(
                    {"item": item.name, "ratio": ratio, "max_abs": differences}
                )
        behaviors = []
        for item_index, item in enumerate(pool[:behavior_items]):
            attempt_seed = scientist_seed + 800_000_000 + item_index * 10_000
            old_attempt = _attempt(old, item, 10.0, simulations, attempt_seed)
            new_attempt = _attempt(migrated, item, 10.0, simulations, attempt_seed)
            behaviors.append(
                {
                    "item": item.name,
                    "ratio": 10.0,
                    "old": old_attempt,
                    "migrated": new_attempt,
                    "actions_identical": old_attempt["actions"] == new_attempt["actions"],
                    "semantic_cost_identical": (
                        old_attempt["verified_semantic_cost"]
                        == new_attempt["verified_semantic_cost"]
                    ),
                }
            )
        passed = all(value <= tolerance for value in maxima.values()) and all(
            row["actions_identical"] and row["semantic_cost_identical"]
            for row in behaviors
        )
        reports[name] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "max_abs": maxima,
            "comparisons": comparisons,
            "paired_behavior": behaviors,
            "passed": passed,
        }
    report = {
        "schema": "semantic-objective-budget-migration-gate-v1",
        "contract": {
            "new_input": "remaining semantic L budget",
            "initialization": "all new paths are zero initialized",
            "required": "identical logits, values, auxiliary outputs, actions, and costs",
        },
        "items": [item.name for item in pool[:items]],
        "ratios": list(ratios),
        "behavior_items": [item.name for item in pool[:behavior_items]],
        "simulations": simulations,
        "seed": seed,
        "tolerance": tolerance,
        "scientists": reports,
        "decision": {
            "passed": all(row["passed"] for row in reports.values()),
            "next": "fine-tune budget critics" if all(
                row["passed"] for row in reports.values()
            ) else "repair checkpoint migration",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "report.json", report)
    return report
