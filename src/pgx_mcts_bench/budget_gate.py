"""Held-out admission gate for the s-window remaining-objective input."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pgx_mcts_bench.adaptive_scientists import FixedWordGame, load_scientist, smallest_crossing_pool
from pgx_mcts_bench.budget_curriculum import _budget_curve, _rung_evaluation
from pgx_mcts_bench.collaborative_scientists import _atomic_json, _sha256
from pgx_mcts_bench.rapid_adaptation import promoted_checkpoint_metadata
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def calibration(scores: list[float], labels: list[int]) -> dict[str, Any]:
    brier = sum((score - label) ** 2 for score, label in zip(scores, labels, strict=True)) / len(
        labels
    )
    bins = []
    ece = 0.0
    for index in range(5):
        low, high = index / 5, (index + 1) / 5
        members = [
            (score, label)
            for score, label in zip(scores, labels, strict=True)
            if low <= score < high or (index == 4 and score == 1.0)
        ]
        if not members:
            continue
        mean_score = sum(score for score, _ in members) / len(members)
        observed = sum(label for _, label in members) / len(members)
        ece += len(members) / len(labels) * abs(mean_score - observed)
        bins.append(
            {
                "low": low,
                "high": high,
                "count": len(members),
                "mean_p_solve": mean_score,
                "observed_solve_rate": observed,
            }
        )
    return {
        "attempts": len(labels),
        "positives": sum(labels),
        "mean_p_solve": sum(scores) / len(scores),
        "observed_solve_rate": sum(labels) / len(labels),
        "brier": brier,
        "auc": roc_auc(scores, labels),
        "ece_5": ece,
        "bins": bins,
    }


def gate_decision(
    *,
    items: int,
    monotone: int,
    informative_items: int,
    informative_sensitive: int,
    never_solved_items: int,
    never_solved_low_probability: int,
    trained_calibration: dict[str, Any],
    baseline_calibration: dict[str, Any],
    trained_solves: int,
    baseline_solves: int,
    trained_rung_rate: float,
    baseline_rung_rate: float,
) -> dict[str, Any]:
    checks = {
        "monotone_all": monotone == items,
        "informative_items_exist": informative_items > 0,
        "all_informative_items_sensitive": informative_sensitive == informative_items,
        "never_solved_items_low_probability": (
            never_solved_low_probability == never_solved_items
        ),
        "both_solve_labels_present": (
            0 < trained_calibration["positives"] < trained_calibration["attempts"]
        ),
        "brier_at_most_0_25": trained_calibration["brier"] <= 0.25,
        "brier_improves_baseline": trained_calibration["brier"] < baseline_calibration["brier"],
        "auc_at_least_0_70": (
            trained_calibration["auc"] is not None and trained_calibration["auc"] >= 0.70
        ),
        "heldout_coverage_noninferior": trained_solves >= baseline_solves,
        "promoted_rung_noninferior": trained_rung_rate >= baseline_rung_rate,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "next": (
            "admit a bounded search-savings ablation"
            if passed
            else "keep predicted caps disabled and expand the easy curriculum"
        ),
    }


def _evaluate_model(
    scientist,
    items,
    *,
    ratio: float,
    caps: tuple[float, ...],
    games_per_cap: int,
    seed: int,
) -> dict[str, Any]:
    curves = _budget_curve(scientist, items, ratio, caps)
    probability = {row["item"]: row["p_solve"] for row in curves}
    attempts = []
    for item_index, item in enumerate(items):
        for cap_index, cap in enumerate(caps):
            fixed = FixedWordGame(scientist.game, item, ratio, objective_cap=cap)
            search = NeuralMCTS(
                fixed,
                scientist.network,
                scientist.config.search,
                scientist.config.train.device,
            )
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
            for game_index, record in enumerate(records):
                attempts.append(
                    {
                        "id": f"{item.name}@{cap:g}#{game_index}",
                        "item": item.name,
                        "cap": cap,
                        "game": game_index,
                        "p_solve": probability[item.name][cap_index],
                        "solved": bool(record and record[0].solved > 0.5),
                        "censored": bool(record and record[0].objective_censored),
                        "crossings": float(record[0].final_crossing_changes) if record else None,
                        "moves": float(record[0].final_moves) if record else None,
                    }
                )
    scores = [float(row["p_solve"]) for row in attempts]
    labels = [int(row["solved"]) for row in attempts]
    return {
        "curves": curves,
        "attempts": attempts,
        "calibration": calibration(scores, labels),
        "solved": sum(labels),
    }


def run_budget_gate(
    baseline_checkpoint: Path,
    trained_checkpoint: Path,
    output: Path,
    *,
    scientist_name: str = "s-window-128",
    training_items: int = 5,
    heldout_start: int = 5,
    heldout_items: int = 10,
    ratio: float = 10.0,
    cap_fractions: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 1.0),
    games_per_cap: int = 4,
    simulations: int = 32,
    rung_eval_games: int = 12,
    rung_simulations: int = 128,
    seed: int = 20261140,
    device: str = "cpu",
) -> dict[str, Any]:
    if heldout_start < training_items:
        raise ValueError("heldout_start must exclude every training item")
    all_items = smallest_crossing_pool(heldout_start + heldout_items)
    training_names = [item.name for item in all_items[:training_items]]
    heldout = all_items[heldout_start : heldout_start + heldout_items]
    baseline = load_scientist(
        scientist_name,
        baseline_checkpoint,
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    trained = load_scientist(
        scientist_name,
        trained_checkpoint,
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    global_cap = (ratio + 1.0) * baseline.config.game.simplify_budget
    caps = tuple(max(1.0, round(global_cap * fraction)) for fraction in cap_fractions)
    baseline_result = _evaluate_model(
        baseline,
        heldout,
        ratio=ratio,
        caps=caps,
        games_per_cap=games_per_cap,
        seed=seed,
    )
    trained_result = _evaluate_model(
        trained,
        heldout,
        ratio=ratio,
        caps=caps,
        games_per_cap=games_per_cap,
        seed=seed,
    )
    metadata = promoted_checkpoint_metadata(baseline_checkpoint, scientist_name)
    baseline_rung = _rung_evaluation(
        baseline,
        metadata,
        rung_eval_games,
        seed + 900_000_000,
        simulations=rung_simulations,
    )
    trained_rung = _rung_evaluation(
        trained,
        metadata,
        rung_eval_games,
        seed + 900_000_000,
        simulations=rung_simulations,
    )
    baseline_solved = {row["id"] for row in baseline_result["attempts"] if row["solved"]}
    trained_solved = {row["id"] for row in trained_result["attempts"] if row["solved"]}
    sensitive = sum(row["spread"] > 1e-4 for row in trained_result["curves"])
    monotone = sum(row["monotone"] for row in trained_result["curves"])
    attempts_by_item = {
        item.name: [row for row in trained_result["attempts"] if row["item"] == item.name]
        for item in heldout
    }
    curves_by_item = {row["item"]: row for row in trained_result["curves"]}
    informative = {
        item
        for item, attempts in attempts_by_item.items()
        if any(row["solved"] for row in attempts) and any(not row["solved"] for row in attempts)
    }
    never_solved = {
        item
        for item, attempts in attempts_by_item.items()
        if not any(row["solved"] for row in attempts)
    }
    informative_sensitive = sum(curves_by_item[item]["spread"] > 1e-4 for item in informative)
    never_solved_low_probability = sum(
        max(curves_by_item[item]["p_solve"]) <= 0.1 for item in never_solved
    )
    decision = gate_decision(
        items=len(heldout),
        monotone=monotone,
        informative_items=len(informative),
        informative_sensitive=informative_sensitive,
        never_solved_items=len(never_solved),
        never_solved_low_probability=never_solved_low_probability,
        trained_calibration=trained_result["calibration"],
        baseline_calibration=baseline_result["calibration"],
        trained_solves=trained_result["solved"],
        baseline_solves=baseline_result["solved"],
        trained_rung_rate=trained_rung["solve_rate"],
        baseline_rung_rate=baseline_rung["solve_rate"],
    )
    report = {
        "schema": "collaboration-budget-heldout-gate-v2",
        "scientist": scientist_name,
        "baseline_checkpoint": str(baseline_checkpoint.resolve()),
        "baseline_sha256": _sha256(baseline_checkpoint),
        "trained_checkpoint": str(trained_checkpoint.resolve()),
        "trained_sha256": _sha256(trained_checkpoint),
        "protocol": {
            "training_items_excluded": training_names,
            "selection_items_excluded": [
                item.name for item in all_items[training_items:heldout_start]
            ],
            "heldout_start": heldout_start,
            "heldout_items": [item.name for item in heldout],
            "ratio": ratio,
            "caps": list(caps),
            "games_per_cap": games_per_cap,
            "simulations": simulations,
            "rung_eval_games": rung_eval_games,
            "rung_simulations": rung_simulations,
            "seed": seed,
            "device": device,
        },
        "baseline": {**baseline_result, "promoted_rung": baseline_rung},
        "trained": {**trained_result, "promoted_rung": trained_rung},
        "paired": {
            "shared_solved": sorted(baseline_solved & trained_solved),
            "trained_only": sorted(trained_solved - baseline_solved),
            "baseline_only": sorted(baseline_solved - trained_solved),
        },
        "summary": {
            "sensitive_items": sensitive,
            "monotone_items": monotone,
            "informative_items": sorted(informative),
            "informative_sensitive": informative_sensitive,
            "never_solved_items": sorted(never_solved),
            "never_solved_low_probability": never_solved_low_probability,
            "baseline_solved": baseline_result["solved"],
            "trained_solved": trained_result["solved"],
        },
        "decision": decision,
    }
    _atomic_json(output, report)
    return report
