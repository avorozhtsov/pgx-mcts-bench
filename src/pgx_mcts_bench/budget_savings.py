"""Paired full-budget versus cap-and-restart search-savings ablation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pgx_mcts_bench.adaptive_scientists import load_scientist, smallest_crossing_pool
from pgx_mcts_bench.collaborative_scientists import (
    BankItem,
    _atomic_json,
    _play,
    _sha256,
    play_with_objective_restarts,
    prediction_details,
    verified_record_cost,
)


def savings_decision(
    *,
    full_solved: set[str],
    budgeted_solved: set[str],
    full_objective: float,
    budgeted_objective: float,
    full_evaluations: int,
    budgeted_evaluations: int,
    minimum_savings: float = 0.20,
) -> dict[str, Any]:
    savings = (
        1.0 - budgeted_evaluations / full_evaluations if full_evaluations else 0.0
    )
    checks = {
        "identical_final_solved_set": budgeted_solved == full_solved,
        "objective_noninferior": budgeted_objective <= full_objective,
        "scheduled_evaluation_savings": savings >= minimum_savings,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "scheduled_network_evaluation_savings": savings,
        "minimum_savings": minimum_savings,
        "next": (
            "admit solve-gated caps as an optional arm when the 200 pilot is unblocked"
            if all(checks.values())
            else "keep predicted caps disabled"
        ),
    }


def run_budget_savings(
    checkpoint: Path,
    output: Path,
    *,
    start_index: int = 25,
    items: int = 20,
    games_per_item: int = 4,
    ratio: float = 10.0,
    multiplier: float = 2.0,
    strategy: str = "restart",
    solve_threshold: float = 0.04,
    simulations: int = 32,
    minimum_savings: float = 0.20,
    seed: int = 20261240,
    device: str = "cpu",
) -> dict[str, Any]:
    if strategy not in {"restart", "solve-gated-probe"}:
        raise ValueError("strategy must be restart or solve-gated-probe")
    pool = smallest_crossing_pool(start_index + items)
    selected = pool[start_index : start_index + items]
    full = load_scientist(
        "s-window-128",
        checkpoint,
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    budgeted = load_scientist(
        "s-window-128",
        checkpoint,
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    global_cap = (ratio + 1.0) * full.config.game.simplify_budget
    rows = []
    for item_offset, knot in enumerate(selected):
        bank_item = BankItem(knot.name, knot, float(knot.crossings), 0)
        prediction = prediction_details(budgeted, bank_item, (ratio,))[0]
        predicted_objective = (
            ratio * prediction["predicted_crossing_changes"] + prediction["predicted_moves"]
        )
        for game_index in range(games_per_item):
            attempt_seed = seed + (start_index + item_offset) * 100_000 + game_index
            started = time.perf_counter()
            full_record = _play(
                full,
                knot,
                ratio,
                simulations=simulations,
                seed=attempt_seed,
                objective_cap=global_cap,
            )
            full_seconds = time.perf_counter() - started
            retained = []
            started = time.perf_counter()
            use_probe = (
                strategy == "restart" or prediction["p_solve"] < solve_threshold
            )
            if use_probe:
                budgeted_record, budget = play_with_objective_restarts(
                    budgeted,
                    knot,
                    ratio,
                    predicted_objective=predicted_objective,
                    simulations=simulations,
                    seed=attempt_seed,
                    multiplier=multiplier,
                    max_restarts=None if strategy == "restart" else 0,
                    retained_records=retained,
                )
            else:
                budgeted_record = _play(
                    budgeted,
                    knot,
                    ratio,
                    simulations=simulations,
                    seed=attempt_seed,
                    objective_cap=global_cap,
                )
                retained.append(budgeted_record)
                budget = {
                    "initial_cap": global_cap,
                    "final_cap": global_cap,
                    "global_cap": global_cap,
                    "restart_count": 0,
                    "attempts": [
                        {
                            "cap": global_cap,
                            "solved": bool(
                                budgeted_record and budgeted_record[0].solved > 0.5
                            ),
                            "objective_budget_exhausted": False,
                            "moves": (
                                float(budgeted_record[0].final_moves)
                                if budgeted_record
                                else 0.0
                            ),
                        }
                    ],
                }
            budgeted_seconds = time.perf_counter() - started
            full_verified = verified_record_cost(full.game, knot, ratio, full_record)
            budgeted_verified = verified_record_cost(
                budgeted.game, knot, ratio, budgeted_record
            )
            rows.append(
                {
                    "id": f"{knot.name}#{game_index}",
                    "item": knot.name,
                    "game": game_index,
                    "prediction": prediction,
                    "predicted_objective": predicted_objective,
                    "search_mode": "probe" if use_probe else "global",
                    "full": {
                        "solved": full_verified is not None,
                        "cost": list(full_verified[:2]) if full_verified is not None else None,
                        "scheduled_network_evaluations": len(full_record) * (simulations + 1),
                        "wall_seconds": full_seconds,
                    },
                    "budgeted": {
                        "solved": budgeted_verified is not None,
                        "cost": (
                            list(budgeted_verified[:2])
                            if budgeted_verified is not None
                            else None
                        ),
                        "scheduled_network_evaluations": sum(len(record) for record in retained)
                        * (simulations + 1),
                        "wall_seconds": budgeted_seconds,
                        "budget": budget,
                    },
                }
            )
    full_solved = {row["id"] for row in rows if row["full"]["solved"]}
    budgeted_solved = {row["id"] for row in rows if row["budgeted"]["solved"]}
    full_objective = sum(
        ratio * row["full"]["cost"][0] + row["full"]["cost"][1]
        for row in rows
        if row["full"]["solved"]
    )
    budgeted_objective = sum(
        ratio * row["budgeted"]["cost"][0] + row["budgeted"]["cost"][1]
        for row in rows
        if row["budgeted"]["solved"]
    )
    full_evaluations = sum(row["full"]["scheduled_network_evaluations"] for row in rows)
    budgeted_evaluations = sum(
        row["budgeted"]["scheduled_network_evaluations"] for row in rows
    )
    decision = savings_decision(
        full_solved=full_solved,
        budgeted_solved=budgeted_solved,
        full_objective=full_objective,
        budgeted_objective=budgeted_objective,
        full_evaluations=full_evaluations,
        budgeted_evaluations=budgeted_evaluations,
        minimum_savings=minimum_savings,
    )
    report = {
        "schema": "s-window-budget-search-savings-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "protocol": {
            "start_index": start_index,
            "items": [knot.name for knot in selected],
            "games_per_item": games_per_item,
            "ratio": ratio,
            "multiplier": multiplier,
            "strategy": strategy,
            "solve_threshold": solve_threshold,
            "initial_cap": "min(global_cap, ceil(multiplier * predicted_L))",
            "restart": "geometric doubling to global cap",
            "global_cap": global_cap,
            "simulations": simulations,
            "minimum_savings": minimum_savings,
            "seed": seed,
            "device": device,
        },
        "rows": rows,
        "summary": {
            "attempt_pairs": len(rows),
            "full_solved": len(full_solved),
            "budgeted_solved": len(budgeted_solved),
            "shared_solved": sorted(full_solved & budgeted_solved),
            "full_only": sorted(full_solved - budgeted_solved),
            "budgeted_only": sorted(budgeted_solved - full_solved),
            "full_objective_sum": full_objective,
            "budgeted_objective_sum": budgeted_objective,
            "full_scheduled_network_evaluations": full_evaluations,
            "budgeted_scheduled_network_evaluations": budgeted_evaluations,
            "full_wall_seconds": sum(row["full"]["wall_seconds"] for row in rows),
            "budgeted_wall_seconds": sum(
                row["budgeted"]["wall_seconds"] for row in rows
            ),
            "total_restarts": sum(
                row["budgeted"]["budget"]["restart_count"] for row in rows
            ),
            "probe_attempts": sum(row["search_mode"] == "probe" for row in rows),
            "global_attempts": sum(row["search_mode"] == "global" for row in rows),
        },
        "decision": decision,
    }
    _atomic_json(output, report)
    return report
