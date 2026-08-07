"""Paired continual-rehearsal smoke for a serial braid scientist."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import load_scientist, score_pool
from pgx_mcts_bench.collaboration_eval import _evaluation_record
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _json_hash,
    _sha256,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.native_learning_gate import (
    _add_records,
    _discover,
    admission_decision,
)
from pgx_mcts_bench.rapid_adaptation import checkpoint_regression_gate
from pgx_mcts_bench.training import train_alphazero_step

DOSES = (64, 128, 256)
ACQUISITION_TIERS = ((64, 4, 64), (128, 2, 96), (256, 2, 128))
LADDER_REHEARSAL_IDS = (
    "3_1",
    "5_1",
    "6_3",
    "8_19",
    "8_21",
    "9_1",
    "10_132",
)


def _seed(seed: int, identity: str, attempt: int, namespace: str) -> int:
    digest = hashlib.sha256(
        f"{seed}:{namespace}:{identity}:{attempt}".encode()
    ).digest()
    return seed + int.from_bytes(digest[:4], "big") % 500_000_000


def _evaluation_scientist(scientist: Any, action_horizon: int) -> Any:
    configured = copy.copy(scientist)
    game_config = replace(scientist.config.game, simplify_budget=action_horizon)
    configured.config = replace(scientist.config, game=game_config)
    configured.game = make_game(game_config)
    return configured


def panel_evaluation(
    scientist: Any,
    items: list[Any],
    *,
    ratio: float,
    simulations: int,
    attempts: int,
    action_horizon: int,
    seed: int,
    namespace: str,
    objective_cap: float | None = None,
    root_noise: bool = False,
) -> dict[str, Any]:
    """Evaluate a representation panel at one fixed, paired search dose."""
    configured = _evaluation_scientist(scientist, action_horizon)
    rows: dict[str, dict[str, Any]] = {}
    scheduled = 0
    for item in items:
        attempt_rows = []
        for attempt in range(attempts):
            attempt_seed = _seed(seed, item.id, attempt, namespace)
            verified, compute = _evaluation_record(
                configured,
                item.knot,
                ratio,
                simulations,
                attempt_seed,
                objective_cap=objective_cap,
                add_root_noise=root_noise,
            )
            scheduled += int(compute["scheduled_network_evaluations"])
            row = {"seed": attempt_seed, "solved": verified is not None}
            if verified is not None:
                row.update(
                    crossing_changes=int(verified[0]),
                    moves=int(verified[1]),
                    objective=ratio * int(verified[0]) + int(verified[1]),
                )
            attempt_rows.append(row)
        solved_attempts = [row for row in attempt_rows if row["solved"]]
        best = min(
            solved_attempts,
            key=lambda row: (
                float(row["objective"]),
                int(row["crossing_changes"]),
                int(row["moves"]),
            ),
            default=None,
        )
        rows[item.id] = {
            "solved": best is not None,
            "solved_attempts": len(solved_attempts),
            "attempts": attempt_rows,
            **(
                {
                    "crossing_changes": best["crossing_changes"],
                    "moves": best["moves"],
                    "objective": best["objective"],
                }
                if best is not None
                else {}
            ),
        }
    solved = sorted(identity for identity, row in rows.items() if row["solved"])
    failure = (
        float(objective_cap)
        if objective_cap is not None
        else ratio * 20 + action_horizon
    )
    return {
        "rows": rows,
        "solved": solved,
        "representations": len(items),
        "representation_solve_rate": len(solved) / len(items) if items else 0.0,
        "attempt_solve_rate": (
            sum(int(attempt["solved"]) for row in rows.values() for attempt in row["attempts"])
            / (len(items) * attempts)
            if items
            else 0.0
        ),
        "capped_objective": sum(
            min(float(row.get("objective", failure)), failure)
            for row in rows.values()
        ),
        "failure_cap": failure,
        "simulations": simulations,
        "attempts_per_representation": attempts,
        "action_horizon": action_horizon,
        "objective_cap": objective_cap,
        "attempt_protocol": (
            "paired-seed-dirichlet-root-noise-temperature-zero"
            if root_noise
            else "deterministic-temperature-zero"
        ),
        "scheduled_network_evaluations": scheduled,
        "allocated_network_evaluations": (
            len(items) * attempts * action_horizon * (simulations + 1)
        ),
    }


def empirical_objective_cap(evaluation: dict[str, Any]) -> float:
    """Freeze the maximum verified objective observed on a calibration panel."""
    objectives = [
        float(attempt["objective"])
        for row in evaluation["rows"].values()
        for attempt in row["attempts"]
        if attempt["solved"]
    ]
    if not objectives:
        raise ValueError("cannot derive an empirical objective cap without a solve")
    return max(objectives)


def portfolio_progress_decision(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Admit block progress only when coverage and capped loss are nonworse."""
    before_solved = set(before["solved"])
    after_solved = set(after["solved"])
    solved_delta = len(after_solved) - len(before_solved)
    objective_delta = float(
        after["capped_objective"] - before["capped_objective"]
    )
    passed = (
        solved_delta >= 0
        and objective_delta <= 0.0
        and (solved_delta > 0 or objective_delta < 0.0)
    )
    return {
        "passed": passed,
        "solved_delta": solved_delta,
        "capped_objective_delta": objective_delta,
        "lost": sorted(before_solved - after_solved),
        "gained": sorted(after_solved - before_solved),
        "needs_recovery": solved_delta < 0 or objective_delta > 0.0,
    }


def calibrate_evaluation_dose(
    checkpoint: Path,
    scientist_name: str,
    items: list[Any],
    *,
    ratio: float,
    attempts: int,
    minimum_coverage: float,
    action_horizon: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Select the lowest registered dose meeting representation coverage."""
    rows = []
    for simulations in DOSES:
        scientist = load_scientist(
            scientist_name,
            checkpoint,
            seed=seed,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=True,
        )
        evaluation = panel_evaluation(
            scientist,
            items,
            ratio=ratio,
            simulations=simulations,
            attempts=attempts,
            action_horizon=action_horizon,
            seed=seed,
            namespace=f"calibration-{simulations}",
        )
        rows.append(evaluation)
        if evaluation["representation_solve_rate"] >= minimum_coverage:
            return {
                "passed": True,
                "selected_simulations": simulations,
                "minimum_coverage": minimum_coverage,
                "rows": rows,
            }
    return {
        "passed": False,
        "selected_simulations": None,
        "minimum_coverage": minimum_coverage,
        "rows": rows,
    }


def frontier_splits(source_payload: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build the registered 10/20/10 outcome-blind engineering frontier."""
    if len(source_payload) < 40:
        raise ValueError("continual smoke source needs at least 40 representations")
    ordered = sorted(
        source_payload,
        key=lambda row: (float(row["cheap_score"]), str(row["id"])),
    )[:40]
    splits = {
        "calibration": ordered[:10],
        "new": ordered[10:30],
        "heldout": ordered[30:40],
    }
    identities = [str(row["id"]) for rows in splits.values() for row in rows]
    if len(identities) != len(set(identities)):
        raise AssertionError("frontier splits are not identity-disjoint")
    if len({int(row["certified_unknotting_lower_bound"]) for row in splits["new"]}) < 3:
        raise ValueError("new-task split does not span at least three certified u lower bounds")
    return splits


def ladder_rehearsal_payload(excluded: set[str]) -> list[dict[str, Any]]:
    """Build the old-task bank from canonical identities seen by the rung ladder."""
    from rf_knots.knot_table import load_table

    table = load_table()["knots"]
    rows = []
    for identity in LADDER_REHEARSAL_IDS:
        if identity in excluded or identity not in table:
            continue
        row = table[identity]
        rows.append(
            {
                "id": identity,
                "name": identity,
                "crossings": int(row["crossings"]),
                "strands": int(row["strands"]),
                "word": [int(letter) for letter in row["braid"]],
                "cheap_score": float(10 + len(row["braid"])),
                "difficulty_quartile": 0,
            }
        )
    return rows


def select_rehearsal_identity(
    identities: list[str],
    *,
    initial_rates: dict[str, float],
    latest_rates: dict[str, float],
    solve_probabilities: dict[str, float],
    exposures: dict[str, int],
) -> str:
    """Prioritize decline, then uncertainty, then low prior rehearsal exposure."""
    if not identities:
        raise ValueError("cannot select from an empty rehearsal bank")
    return min(
        identities,
        key=lambda identity: (
            -(initial_rates.get(identity, 0.0) - latest_rates.get(identity, 0.0)),
            latest_rates.get(identity, 0.0),
            solve_probabilities.get(identity, 0.0),
            exposures.get(identity, 0),
            identity,
        ),
    )


def cycle_certificate(losses: list[dict[str, float]]) -> dict[str, Any]:
    """Summarize the realized target and replay strata for one round."""
    fields = (
        "policy_value_targets",
        "solve_targets",
        "crossing_targets",
        "move_targets",
        "replay_current_success_positions",
        "replay_rehearsal_success_positions",
        "replay_ordinary_failure_positions",
        "replay_budget_censored_failure_positions",
    )
    totals = {field: sum(float(row.get(field, 0.0)) for row in losses) for field in fields}
    positive = (
        totals["replay_current_success_positions"]
        + totals["replay_rehearsal_success_positions"]
    )
    negative = (
        totals["replay_ordinary_failure_positions"]
        + totals["replay_budget_censored_failure_positions"]
    )
    total = positive + negative
    return {
        **totals,
        "optimizer_updates": len(losses),
        "positive_positions": positive,
        "negative_positions": negative,
        "positive_fraction": positive / total if total else None,
        "budget_censored_fraction_of_failures": (
            totals["replay_budget_censored_failure_positions"] / negative
            if negative
            else None
        ),
    }


def _search_compute(search_rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "scheduled": sum(int(row["scheduled_network_evaluations"]) for row in search_rows),
        "allocated": sum(int(row["allocated_network_evaluations"]) for row in search_rows),
    }


def _discover_add(
    scientist: Any,
    item: Any,
    *,
    ratio: float,
    seed: int,
    device: str,
    objective_cap: float | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    records, search_rows = _discover(
        scientist,
        item,
        ratio=ratio,
        seed=seed,
        tiers=ACQUISITION_TIERS,
        device=device,
        objective_cap=objective_cap,
    )
    _add_records(scientist, item, records)
    return search_rows, any(int(row.get("solved", 0)) for row in search_rows)


def _train(
    scientist: Any,
    *,
    current: str,
    rehearsal: set[str],
    steps: int,
    batch_size: int,
    device: str,
) -> list[dict[str, float]]:
    losses = []
    for _ in range(steps):
        losses.append(
            train_alphazero_step(
                scientist.network,
                scientist.optimizer,
                scientist.replay,
                batch_size,
                torch.device(device),
                policy_value_success_only=True,
                replay_current_representation=current,
                replay_positions_per_episode=4,
                continual_replay=True,
                replay_rehearsal_representations=rehearsal,
            )
        )
    return losses


def _archive_ids(scientist: Any, ratio: float) -> list[str]:
    return sorted(
        identity
        for identity, archived_ratio in scientist.replay.best_native_solution_records
        if float(archived_ratio) == float(ratio)
    )


def lifetime_solution_bank_metric(
    scientist: Any,
    items: list[Any],
    *,
    ratio: float,
    objective_cap: float,
) -> dict[str, Any]:
    """Score the permanent verified-solution bank on one fixed portfolio."""
    rows: dict[str, dict[str, Any]] = {}
    solved: list[str] = []
    capped_objective = 0.0
    for item in items:
        objective = scientist.replay.best_native_objective(item.id, ratio)
        if objective is None:
            rows[item.id] = {"solved": False, "objective": objective_cap}
            capped_objective += objective_cap
            continue
        capped = min(float(objective), objective_cap)
        rows[item.id] = {
            "solved": True,
            "objective": float(objective),
            "capped_objective": capped,
        }
        solved.append(item.id)
        capped_objective += capped
    return {
        "rows": rows,
        "solved": sorted(solved),
        "representations": len(items),
        "representation_solve_rate": len(solved) / len(items) if items else 0.0,
        "capped_objective": capped_objective,
        "failure_cap": objective_cap,
    }


def _save_state(path: Path, scientist: Any, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "network": scientist.network.state_dict(),
            "optimizer": scientist.optimizer.state_dict(),
            "replay": scientist.replay,
            "state": state,
        },
        path,
    )


def _save_block_start(path: Path, scientist: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "network": scientist.network.state_dict(),
            "optimizer": scientist.optimizer.state_dict(),
        },
        path,
    )


def _run_arm(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(payload["torch_threads"]))
    started = time.perf_counter()
    output = Path(payload["output"])
    output.mkdir(parents=True, exist_ok=True)
    scientist = load_scientist(
        payload["scientist"],
        Path(payload["checkpoint"]),
        seed=int(payload["seed"]),
        device=payload["device"],
        simulations=int(payload["evaluation_simulations"]),
        require_factorized=True,
        objective_budget_channel=True,
    )
    new_items = _bank_from_payload(payload["new"])
    heldout_items = _bank_from_payload(payload["heldout"])
    hard_items = _bank_from_payload(payload["hard_stress"])
    rehearsal_items = _bank_from_payload(payload["rehearsal"])
    by_id = {item.id: item for item in rehearsal_items + new_items}
    ratio = float(payload["ratio"])
    simulations = int(payload["evaluation_simulations"])
    attempts = int(payload["evaluation_attempts"])
    horizon = int(payload["evaluation_action_horizon"])
    objective_cap = (
        float(payload["objective_cap"])
        if payload.get("objective_cap") is not None
        else None
    )
    bank_failure_cap = (
        objective_cap if objective_cap is not None else ratio * 20 + horizon
    )
    evaluation_seed = int(payload["seed"]) + 700_000_000
    initial_path = output / "initial.pt"

    if initial_path.exists() and payload["resume"]:
        saved = torch.load(initial_path, map_location=payload["device"], weights_only=False)
        scientist.network.load_state_dict(saved["network"])
        scientist.optimizer.load_state_dict(saved["optimizer"])
        scientist.replay = saved["replay"]
        state = saved["state"]
        initial = json.loads((output / "initial-evaluation.json").read_text())
    else:
        search_compute = {"scheduled": 0, "allocated": 0}
        initialization = []
        for index, item in enumerate(rehearsal_items):
            rows, solved = _discover_add(
                scientist,
                item,
                ratio=ratio,
                seed=int(payload["seed"]) + 10_000_000 + index * 1_000_000,
                device=payload["device"],
                objective_cap=objective_cap,
            )
            measured = _search_compute(rows)
            for field in search_compute:
                search_compute[field] += measured[field]
            initialization.append({"item": item.id, "solved": solved, "search": rows})
        archived = set(_archive_ids(scientist, ratio))
        active_rehearsal = [item for item in rehearsal_items if item.id in archived]
        if len(active_rehearsal) < int(payload["minimum_rehearsal_solutions"]):
            return {
                "arm": payload["arm"],
                "seed": payload["seed"],
                "completed": False,
                "reason": "insufficient certified rehearsal solutions",
                "initialization": initialization,
                "active_rehearsal": sorted(archived),
            }
        initial = {
            "new": panel_evaluation(
                scientist,
                new_items,
                ratio=ratio,
                simulations=simulations,
                attempts=attempts,
                action_horizon=horizon,
                seed=evaluation_seed,
                namespace="new",
                objective_cap=objective_cap,
            ),
            "heldout": panel_evaluation(
                scientist,
                heldout_items,
                ratio=ratio,
                simulations=simulations,
                attempts=attempts,
                action_horizon=horizon,
                seed=evaluation_seed,
                namespace="heldout",
                objective_cap=objective_cap,
            ),
            "hard_stress": panel_evaluation(
                scientist,
                hard_items,
                ratio=ratio,
                simulations=simulations,
                attempts=attempts,
                action_horizon=horizon,
                seed=evaluation_seed,
                namespace="hard-stress",
                objective_cap=objective_cap,
            ),
            "rehearsal": panel_evaluation(
                scientist,
                active_rehearsal,
                ratio=ratio,
                simulations=simulations,
                attempts=attempts,
                action_horizon=horizon,
                seed=evaluation_seed,
                namespace="rehearsal",
                objective_cap=objective_cap,
            ),
            "initialization": initialization,
        }
        for split in ("new", "heldout", "hard_stress", "rehearsal"):
            search_compute["scheduled"] += int(
                initial[split]["scheduled_network_evaluations"]
            )
            search_compute["allocated"] += int(
                initial[split]["allocated_network_evaluations"]
            )
        rates = {
            identity: row["solved_attempts"] / attempts
            for identity, row in initial["rehearsal"]["rows"].items()
        }
        state = {
            "next_round": 0,
            "rounds": [],
            "initial_rehearsal_rates": rates,
            "latest_rehearsal_rates": dict(rates),
            "rehearsal_exposures": {identity: 0 for identity in rates},
            "search_compute": search_compute,
            "optimizer_updates": 0,
            "base_rehearsal_ids": [item.id for item in active_rehearsal],
            "block_baseline": None,
        }
        if payload.get("progress_guard", False):
            block_items = active_rehearsal + new_items[: int(payload["block_size"])]
            state["block_baseline"] = panel_evaluation(
                scientist,
                block_items,
                ratio=ratio,
                simulations=simulations,
                attempts=attempts,
                action_horizon=horizon,
                seed=evaluation_seed,
                namespace="portfolio-block-0",
                objective_cap=objective_cap,
            )
            state["search_compute"]["scheduled"] += int(
                state["block_baseline"]["scheduled_network_evaluations"]
            )
            state["search_compute"]["allocated"] += int(
                state["block_baseline"]["allocated_network_evaluations"]
            )
            _save_block_start(output / "block-starts" / "000.pt", scientist)
        _atomic_json(output / "initial-evaluation.json", initial)
        _save_state(initial_path, scientist, state)

    rounds_dir = output / "rounds"
    for round_index in range(int(state["next_round"]), len(new_items)):
        round_state_path = rounds_dir / f"{round_index:03d}" / "state.pt"
        round_report_path = rounds_dir / f"{round_index:03d}" / "report.json"
        if round_state_path.exists() and payload["resume"]:
            saved = torch.load(
                round_state_path, map_location=payload["device"], weights_only=False
            )
            scientist.network.load_state_dict(saved["network"])
            scientist.optimizer.load_state_dict(saved["optimizer"])
            scientist.replay = saved["replay"]
            state = saved["state"]
            continue

        block_size = int(payload["block_size"])
        block_index = round_index // block_size
        if (
            payload.get("progress_guard", False)
            and round_index > 0
            and round_index % block_size == 0
        ):
            block_stop = min(len(new_items), round_index + block_size)
            block_items = [
                by_id[identity] for identity in state["base_rehearsal_ids"]
            ] + new_items[:block_stop]
            state["block_baseline"] = panel_evaluation(
                scientist,
                block_items,
                ratio=ratio,
                simulations=simulations,
                attempts=attempts,
                action_horizon=horizon,
                seed=evaluation_seed,
                namespace=f"portfolio-block-{block_index}",
                objective_cap=objective_cap,
            )
            state["search_compute"]["scheduled"] += int(
                state["block_baseline"]["scheduled_network_evaluations"]
            )
            state["search_compute"]["allocated"] += int(
                state["block_baseline"]["allocated_network_evaluations"]
            )
            _save_block_start(
                output / "block-starts" / f"{block_index:03d}.pt", scientist
            )

        target = new_items[round_index]
        archived_ids = _archive_ids(scientist, ratio)
        rehearsal_candidates = [
            identity for identity in archived_ids if identity != target.id and identity in by_id
        ]
        candidate_items = [by_id[identity].knot for identity in rehearsal_candidates]
        probabilities = (
            score_pool(scientist, candidate_items)[0]
            if candidate_items
            else np.asarray([], dtype=np.float32)
        )
        selected_old = select_rehearsal_identity(
            rehearsal_candidates,
            initial_rates=state["initial_rehearsal_rates"],
            latest_rates=state["latest_rehearsal_rates"],
            solve_probabilities={
                identity: float(probability)
                for identity, probability in zip(
                    rehearsal_candidates, probabilities, strict=True
                )
            },
            exposures=state["rehearsal_exposures"],
        )
        protected = [by_id[identity] for identity in rehearsal_candidates] + [target]
        before = panel_evaluation(
            scientist,
            protected,
            ratio=ratio,
            simulations=simulations,
            attempts=1,
            action_horizon=horizon,
            seed=evaluation_seed,
            namespace=f"round-{round_index}-admission",
            objective_cap=objective_cap,
        )
        network_before = copy.deepcopy(scientist.network.state_dict())
        optimizer_before = copy.deepcopy(scientist.optimizer.state_dict())
        losses: list[dict[str, float]] = []
        searches = []
        for iteration in range(int(payload["f_new"])):
            rows, solved = _discover_add(
                scientist,
                target,
                ratio=ratio,
                seed=(
                    int(payload["seed"])
                    + 100_000_000
                    + round_index * 10_000_000
                    + iteration * 1_000_000
                ),
                device=payload["device"],
                objective_cap=objective_cap,
            )
            measured = _search_compute(rows)
            for field in state["search_compute"]:
                state["search_compute"][field] += measured[field]
            searches.append(
                {"kind": "new", "iteration": iteration, "solved": solved, "tiers": rows}
            )
            losses.extend(
                _train(
                    scientist,
                    current=target.id,
                    rehearsal=set(rehearsal_candidates),
                    steps=int(payload["train_steps_per_iteration"]),
                    batch_size=int(payload["batch_size"]),
                    device=payload["device"],
                )
            )

        old_item = by_id[selected_old]
        rows, old_solved = _discover_add(
            scientist,
            old_item,
            ratio=ratio,
            seed=int(payload["seed"]) + 500_000_000 + round_index * 1_000_000,
            device=payload["device"],
            objective_cap=objective_cap,
        )
        measured = _search_compute(rows)
        for field in state["search_compute"]:
            state["search_compute"][field] += measured[field]
        if not old_solved:
            archived_record = scientist.replay.best_native_solution_record(selected_old, ratio)
            if archived_record is not None:
                scientist.replay.add(
                    archived_record,
                    representation_id=selected_old,
                    objective_ratio=ratio,
                )
        searches.append(
            {
                "kind": "rehearsal",
                "representation": selected_old,
                "solved": old_solved,
                "tiers": rows,
            }
        )
        losses.extend(
            _train(
                scientist,
                current=target.id,
                rehearsal=set(rehearsal_candidates),
                steps=int(payload["train_steps_per_iteration"]),
                batch_size=int(payload["batch_size"]),
                device=payload["device"],
            )
        )
        state["optimizer_updates"] += len(losses)
        state["rehearsal_exposures"][selected_old] = (
            int(state["rehearsal_exposures"].get(selected_old, 0)) + 1
        )
        after = panel_evaluation(
            scientist,
            protected,
            ratio=ratio,
            simulations=simulations,
            attempts=1,
            action_horizon=horizon,
            seed=evaluation_seed,
            namespace=f"round-{round_index}-admission",
            objective_cap=objective_cap,
        )
        for evaluation in (before, after):
            state["search_compute"]["scheduled"] += int(
                evaluation["scheduled_network_evaluations"]
            )
            state["search_compute"]["allocated"] += int(
                evaluation["allocated_network_evaluations"]
            )
        decision = admission_decision(before, after, target=target.id)
        accepted = True
        if payload["arm"] == "transactional" and not decision["passed"]:
            scientist.network.load_state_dict(network_before)
            scientist.optimizer.load_state_dict(optimizer_before)
            accepted = False

        block_evaluation = None
        block_progress = None
        recovery = None
        restored_block_start = False
        if (round_index + 1) % block_size == 0:
            # Both arms are measured on exactly the same old-plus-seen task set.
            # A different archive membership must never change the denominator.
            block_items = [
                by_id[identity] for identity in state["base_rehearsal_ids"]
            ] + new_items[: round_index + 1]
            block_evaluation = panel_evaluation(
                scientist,
                block_items,
                ratio=ratio,
                simulations=simulations,
                attempts=attempts,
                action_horizon=horizon,
                seed=evaluation_seed,
                namespace=(
                    f"portfolio-block-{block_index}"
                    if payload.get("progress_guard", False)
                    else f"block-{block_index}"
                ),
                objective_cap=objective_cap,
            )
            state["search_compute"]["scheduled"] += int(
                block_evaluation["scheduled_network_evaluations"]
            )
            state["search_compute"]["allocated"] += int(
                block_evaluation["allocated_network_evaluations"]
            )
            for identity, row in block_evaluation["rows"].items():
                state["latest_rehearsal_rates"][identity] = (
                    row["solved_attempts"] / attempts
                )
            if payload.get("progress_guard", False):
                block_progress = portfolio_progress_decision(
                    state["block_baseline"], block_evaluation
                )
                if block_progress["needs_recovery"]:
                    before_rows = state["block_baseline"]["rows"]
                    after_rows = block_evaluation["rows"]
                    recovery_candidates = [
                        item
                        for item in block_items
                        if item.id in block_progress["lost"]
                        or float(after_rows[item.id].get("objective", objective_cap))
                        > float(before_rows[item.id].get("objective", objective_cap))
                    ]
                    recovery_ids = sorted(
                        recovery_candidates,
                        key=lambda item: (
                            item.id in block_progress["lost"],
                            float(after_rows[item.id].get("objective", objective_cap))
                            - float(before_rows[item.id].get("objective", objective_cap)),
                            item.id,
                        ),
                        reverse=True,
                    )[: int(payload.get("maximum_recovery_tasks", 5))]
                    recovery_searches = []
                    recovery_losses: list[dict[str, float]] = []
                    rehearsal_ids = set(_archive_ids(scientist, ratio))
                    for recovery_index, recovery_item in enumerate(recovery_ids):
                        rows, solved = _discover_add(
                            scientist,
                            recovery_item,
                            ratio=ratio,
                            seed=(
                                int(payload["seed"])
                                + 900_000_000
                                + block_index * 10_000_000
                                + recovery_index * 1_000_000
                            ),
                            device=payload["device"],
                            objective_cap=objective_cap,
                        )
                        measured = _search_compute(rows)
                        for field in state["search_compute"]:
                            state["search_compute"][field] += measured[field]
                        if not solved:
                            archived_record = scientist.replay.best_native_solution_record(
                                recovery_item.id, ratio
                            )
                            if archived_record is not None:
                                scientist.replay.add(
                                    archived_record,
                                    representation_id=recovery_item.id,
                                    objective_ratio=ratio,
                                )
                        recovery_losses.extend(
                            _train(
                                scientist,
                                current=recovery_item.id,
                                rehearsal=rehearsal_ids - {recovery_item.id},
                                steps=int(payload["train_steps_per_iteration"]),
                                batch_size=int(payload["batch_size"]),
                                device=payload["device"],
                            )
                        )
                        recovery_searches.append(
                            {"item": recovery_item.id, "solved": solved, "tiers": rows}
                        )
                    state["optimizer_updates"] += len(recovery_losses)
                    recovered_evaluation = panel_evaluation(
                        scientist,
                        block_items,
                        ratio=ratio,
                        simulations=simulations,
                        attempts=attempts,
                        action_horizon=horizon,
                        seed=evaluation_seed,
                        namespace=f"portfolio-block-{block_index}",
                        objective_cap=objective_cap,
                    )
                    for field in state["search_compute"]:
                        state["search_compute"][field] += int(
                            recovered_evaluation[
                                f"{field}_network_evaluations"
                            ]
                        )
                    recovered_progress = portfolio_progress_decision(
                        state["block_baseline"], recovered_evaluation
                    )
                    recovery = {
                        "tasks": [item.id for item in recovery_ids],
                        "searches": recovery_searches,
                        "cycle_certificate": cycle_certificate(recovery_losses),
                        "evaluation": recovered_evaluation,
                        "progress": recovered_progress,
                    }
                    block_evaluation = recovered_evaluation
                    block_progress = recovered_progress
                    for identity, row in block_evaluation["rows"].items():
                        state["latest_rehearsal_rates"][identity] = (
                            row["solved_attempts"] / attempts
                        )
                if block_progress["needs_recovery"]:
                    saved_start = torch.load(
                        output / "block-starts" / f"{block_index:03d}.pt",
                        map_location=payload["device"],
                        weights_only=False,
                    )
                    scientist.network.load_state_dict(saved_start["network"])
                    scientist.optimizer.load_state_dict(saved_start["optimizer"])
                    block_evaluation = state["block_baseline"]
                    restored_block_start = True

        round_row = {
            "round": round_index,
            "target": target.id,
            "selected_rehearsal": selected_old,
            "searches": searches,
            "admission": decision,
            "accepted": accepted,
            "temporary_lost": decision["lost"],
            "temporary_gained": decision["gained"],
            "cycle_certificate": cycle_certificate(losses),
            "last_loss": losses[-1] if losses else None,
            "block_evaluation": block_evaluation,
            "lifetime_solution_bank": (
                lifetime_solution_bank_metric(
                    scientist,
                    [by_id[identity] for identity in state["base_rehearsal_ids"]]
                    + new_items[: round_index + 1],
                    ratio=ratio,
                    objective_cap=bank_failure_cap,
                )
                if block_evaluation is not None
                else None
            ),
            "block_progress": block_progress,
            "recovery": recovery,
            "restored_block_start": restored_block_start,
        }
        state["rounds"].append(round_row)
        state["next_round"] = round_index + 1
        _atomic_json(round_report_path, round_row)
        _save_state(round_state_path, scientist, state)

    active_rehearsal = [by_id[identity] for identity in state["base_rehearsal_ids"]]
    complete_portfolio = active_rehearsal + new_items
    final = {
        "new": panel_evaluation(
            scientist,
            new_items,
            ratio=ratio,
            simulations=simulations,
            attempts=attempts,
            action_horizon=horizon,
            seed=evaluation_seed,
            namespace="new",
            objective_cap=objective_cap,
        ),
        "heldout": panel_evaluation(
            scientist,
            heldout_items,
            ratio=ratio,
            simulations=simulations,
            attempts=attempts,
            action_horizon=horizon,
            seed=evaluation_seed,
            namespace="heldout",
            objective_cap=objective_cap,
        ),
        "hard_stress": panel_evaluation(
            scientist,
            hard_items,
            ratio=ratio,
            simulations=simulations,
            attempts=attempts,
            action_horizon=horizon,
            seed=evaluation_seed,
            namespace="hard-stress",
            objective_cap=objective_cap,
        ),
        "rehearsal": panel_evaluation(
            scientist,
            active_rehearsal,
            ratio=ratio,
            simulations=simulations,
            attempts=attempts,
            action_horizon=horizon,
            seed=evaluation_seed,
            namespace="rehearsal",
            objective_cap=objective_cap,
        ),
        "portfolio": panel_evaluation(
            scientist,
            complete_portfolio,
            ratio=ratio,
            simulations=simulations,
            attempts=attempts,
            action_horizon=horizon,
            seed=evaluation_seed,
            namespace="complete-portfolio",
            objective_cap=objective_cap,
        ),
        "lifetime_solution_bank": lifetime_solution_bank_metric(
            scientist,
            complete_portfolio,
            ratio=ratio,
            objective_cap=bank_failure_cap,
        ),
    }
    for split_name in ("new", "heldout", "hard_stress", "rehearsal", "portfolio"):
        split = final[split_name]
        state["search_compute"]["scheduled"] += int(
            split["scheduled_network_evaluations"]
        )
        state["search_compute"]["allocated"] += int(
            split["allocated_network_evaluations"]
        )
    result = {
        "arm": payload["arm"],
        "seed": payload["seed"],
        "completed": True,
        "initial": initial,
        "final": final,
        "rounds": state["rounds"],
        "optimizer_updates": state["optimizer_updates"],
        "compute": state["search_compute"],
        "gained_new": sorted(set(final["new"]["solved"]) - set(initial["new"]["solved"])),
        "lost_new": sorted(set(initial["new"]["solved"]) - set(final["new"]["solved"])),
        "lost_rehearsal": sorted(
            set(initial["rehearsal"]["solved"]) - set(final["rehearsal"]["solved"])
        ),
        "temporary_forgetting_events": sum(bool(row["temporary_lost"]) for row in state["rounds"]),
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(output / "report.json", result)
    return result


def analyze_paired_smoke(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the registered engineering-smoke decision to paired arms."""
    by_arm = {row["arm"]: row for row in rows}
    if set(by_arm) != {"continual", "transactional"} or not all(
        row.get("completed") for row in rows
    ):
        return {
            "passed": False,
            "reason": "both paired arms did not complete",
            "next_step": "repair the smoke; keep the 50-representation gate closed",
        }
    continual = by_arm["continual"]
    transactional = by_arm["transactional"]
    allocated_delta = (
        continual["compute"]["allocated"] - transactional["compute"]["allocated"]
    ) / max(transactional["compute"]["allocated"], 1)
    scheduled_delta = (
        continual["compute"]["scheduled"] - transactional["compute"]["scheduled"]
    ) / max(transactional["compute"]["scheduled"], 1)
    capped_delta = (
        continual["final"]["new"]["capped_objective"]
        - transactional["final"]["new"]["capped_objective"]
    )
    heldout_delta = (
        continual["final"]["heldout"]["capped_objective"]
        - transactional["final"]["heldout"]["capped_objective"]
    )
    certificates_present = all(
        row["cycle_certificate"]["optimizer_updates"] > 0
        for arm in rows
        for row in arm.get("rounds", [])
    )
    passed = (
        continual["final"]["new"]["representation_solve_rate"] >= 0.70
        and transactional["final"]["new"]["representation_solve_rate"] >= 0.70
        and continual["final"]["rehearsal"]["representation_solve_rate"] >= 0.70
        and capped_delta <= 0.0
        and heldout_delta <= 0.0
        and abs(allocated_delta) <= 0.02
        and certificates_present
    )
    return {
        "passed": passed,
        "new_solved_intersection": sorted(
            set(continual["final"]["new"]["solved"])
            & set(transactional["final"]["new"]["solved"])
        ),
        "continual_only": sorted(
            set(continual["final"]["new"]["solved"])
            - set(transactional["final"]["new"]["solved"])
        ),
        "transactional_only": sorted(
            set(transactional["final"]["new"]["solved"])
            - set(continual["final"]["new"]["solved"])
        ),
        "continual_new_solve_rate": continual["final"]["new"]["representation_solve_rate"],
        "transactional_new_solve_rate": transactional["final"]["new"]["representation_solve_rate"],
        "continual_rehearsal_solve_rate": continual["final"]["rehearsal"][
            "representation_solve_rate"
        ],
        "new_capped_objective_delta_continual_minus_transactional": capped_delta,
        "heldout_capped_objective_delta_continual_minus_transactional": heldout_delta,
        "allocated_compute_fractional_delta": allocated_delta,
        "scheduled_compute_fractional_delta": scheduled_delta,
        "certificates_present": certificates_present,
        "next_step": (
            "run the 50-representation three-seed continual-learning gate"
            if passed
            else "keep the 50-representation and downstream collaboration gates closed"
        ),
    }


def run_continual_learning_smoke(
    checkpoint: Path,
    source_bank: Path,
    hard_stress_bank: Path,
    output: Path,
    *,
    scientist: str = "s-window-128",
    ratio: float = 1000.0,
    evaluation_attempts: int = 4,
    evaluation_action_horizon: int = 64,
    minimum_calibration_coverage: float = 0.70,
    f_new: int = 5,
    f_old: int = 1,
    train_steps_per_iteration: int = 24,
    batch_size: int = 32,
    block_size: int = 10,
    seed: int = 20261700,
    device: str = "cpu",
    resume: bool = False,
    torch_threads_per_arm: int = 4,
) -> dict[str, Any]:
    """Run one paired 20-round engineering smoke before any larger gate."""
    if f_old != 1:
        raise ValueError("the registered smoke requires F_old=1")
    source_payload = json.loads(source_bank.read_text())
    hard_payload = json.loads(hard_stress_bank.read_text())
    splits = frontier_splits(source_payload)
    excluded = {str(row["id"]) for rows in splits.values() for row in rows}
    rehearsal = ladder_rehearsal_payload(excluded)
    calibration_items = _bank_from_payload(splits["calibration"])
    calibration = calibrate_evaluation_dose(
        checkpoint,
        scientist,
        calibration_items,
        ratio=ratio,
        attempts=evaluation_attempts,
        minimum_coverage=minimum_calibration_coverage,
        action_horizon=evaluation_action_horizon,
        seed=seed + 600_000_000,
        device=device,
    )
    gate = checkpoint_regression_gate(
        checkpoint,
        scientist,
        ratio=ratio,
        games=6,
        minimum_solve_rate=0.8,
        seed=seed + 650_000_000,
        simulations=int(calibration["selected_simulations"] or DOSES[-1]),
        device=device,
    )
    protocol = {
        "schema": "paired-continual-rehearsal-smoke-v1",
        "scientist": scientist,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_regression_gate": gate,
        "source_bank": str(source_bank.resolve()),
        "source_bank_sha256": _json_hash(source_payload),
        "hard_stress_bank": str(hard_stress_bank.resolve()),
        "hard_stress_bank_sha256": _json_hash(hard_payload),
        "splits": splits,
        "rehearsal": rehearsal,
        "calibration": calibration,
        "ratio": ratio,
        "evaluation_attempts": evaluation_attempts,
        "evaluation_action_horizon": evaluation_action_horizon,
        "f_new": f_new,
        "f_old": f_old,
        "acquisition_tiers": [list(tier) for tier in ACQUISITION_TIERS],
        "train_steps_per_iteration": train_steps_per_iteration,
        "batch_size": batch_size,
        "block_size": block_size,
        "seed": seed,
        "device": device,
        "torch_threads_per_arm": torch_threads_per_arm,
        "objective_budgeting": "global environment cap only; no predicted or structural first cap",
        "arms": ["continual", "transactional"],
        "transactional_control": "per-round exact-retention and capped-objective rollback",
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{manifest} exists; pass resume=True")
        if json.loads(manifest.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("continual-learning smoke resume protocol differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, protocol)
    if not calibration["passed"]:
        report = {
            **protocol,
            "completed": False,
            "decision": {
                "passed": False,
                "reason": "no registered evaluation dose reached 70% calibration coverage",
                "next_step": "redesign the outcome-blind frontier; do not start training",
            },
        }
        _atomic_json(output / "report.json", report)
        return report

    payloads = []
    for arm in protocol["arms"]:
        payloads.append(
            {
                **protocol,
                "arm": arm,
                "output": str(output / "arms" / arm),
                "new": splits["new"],
                "heldout": splits["heldout"],
                "hard_stress": hard_payload,
                "rehearsal": rehearsal,
                "evaluation_simulations": calibration["selected_simulations"],
                "minimum_rehearsal_solutions": 3,
                "resume": resume,
                "torch_threads": torch_threads_per_arm,
            }
        )
    rows = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_run_arm, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["arm"])
    decision = analyze_paired_smoke(rows)
    report = {
        **protocol,
        "completed": all(row.get("completed") for row in rows),
        "arms_results": rows,
        "decision": decision,
    }
    _atomic_json(output / "report.json", report)
    return report


def analyze_portfolio_progress_smoke(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare block-progress rehearsal against per-round rollback under L10."""
    by_arm = {row["arm"]: row for row in rows}
    if set(by_arm) != {"block-progress", "transactional"} or not all(
        row.get("completed") for row in rows
    ):
        return {
            "passed": False,
            "reason": "both L10 arms did not complete",
            "next_step": "repair the local smoke; keep longer gates closed",
        }
    treatment = by_arm["block-progress"]
    control = by_arm["transactional"]
    treatment_portfolio = treatment["final"]["portfolio"]
    control_portfolio = control["final"]["portfolio"]
    treatment_solved = set(treatment_portfolio["solved"])
    control_solved = set(control_portfolio["solved"])
    capped_delta = float(
        treatment_portfolio["capped_objective"]
        - control_portfolio["capped_objective"]
    )
    allocated_delta = (
        treatment["compute"]["allocated"] - control["compute"]["allocated"]
    ) / max(control["compute"]["allocated"], 1)
    treatment_blocks = [
        row for row in treatment["rounds"] if row.get("block_progress") is not None
    ]
    nonregressed_blocks = all(
        not row["block_progress"]["needs_recovery"] or row["restored_block_start"]
        for row in treatment_blocks
    )
    progress_observed = any(
        row["block_progress"]["solved_delta"] > 0
        or row["block_progress"]["capped_objective_delta"] < 0.0
        for row in treatment_blocks
    )
    # The transactional arm is a paired diagnostic, not the retention target.
    # A valid continual learner may temporarily trade which tasks it solves.
    passed = (
        treatment["final"]["new"]["representation_solve_rate"] >= 0.70
        and nonregressed_blocks
        and progress_observed
    )
    return {
        "passed": passed,
        "portfolio_representations": treatment_portfolio["representations"],
        "treatment_solved": len(treatment_solved),
        "control_solved": len(control_solved),
        "solved_intersection": sorted(treatment_solved & control_solved),
        "treatment_only": sorted(treatment_solved - control_solved),
        "control_only": sorted(control_solved - treatment_solved),
        "treatment_capped_L10": treatment_portfolio["capped_objective"],
        "control_capped_L10": control_portfolio["capped_objective"],
        "capped_L10_delta_treatment_minus_control": capped_delta,
        "allocated_compute_fractional_delta": allocated_delta,
        "nonregressed_or_restored_blocks": nonregressed_blocks,
        "progress_observed": progress_observed,
        "treatment_lifetime_bank_solved": len(
            treatment["final"]["lifetime_solution_bank"]["solved"]
        ),
        "treatment_lifetime_bank_capped_L10": treatment["final"][
            "lifetime_solution_bank"
        ]["capped_objective"],
        "control_lifetime_bank_solved": len(
            control["final"]["lifetime_solution_bank"]["solved"]
        ),
        "control_lifetime_bank_capped_L10": control["final"][
            "lifetime_solution_bank"
        ]["capped_objective"],
        "treatment_new_solve_rate": treatment["final"]["new"][
            "representation_solve_rate"
        ],
        "control_new_solve_rate": control["final"]["new"][
            "representation_solve_rate"
        ],
        "heldout_capped_L10_delta_treatment_minus_control": (
            treatment["final"]["heldout"]["capped_objective"]
            - control["final"]["heldout"]["capped_objective"]
        ),
        "next_step": (
            "run the 50-representation three-seed L10 gate"
            if passed
            else "keep the 50-representation and downstream collaboration gates closed"
        ),
        "secondary_control_comparison": {
            "treatment_solved_minus_control": len(treatment_solved)
            - len(control_solved),
            "capped_L10_delta_treatment_minus_control": capped_delta,
            "allocated_compute_fractional_delta": allocated_delta,
        },
    }


def run_portfolio_progress_smoke(
    checkpoint: Path,
    source_bank: Path,
    hard_stress_bank: Path,
    output: Path,
    *,
    scientist: str = "s-window-128",
    evaluation_attempts: int = 4,
    evaluation_action_horizon: int = 64,
    minimum_calibration_coverage: float = 0.70,
    f_new: int = 5,
    train_steps_per_iteration: int = 24,
    batch_size: int = 32,
    block_size: int = 10,
    maximum_recovery_tasks: int = 5,
    seed: int = 20261720,
    device: str = "cpu",
    resume: bool = False,
    torch_threads_per_arm: int = 4,
) -> dict[str, Any]:
    """Run the corrected L10 complete-portfolio block-progress smoke."""
    ratio = 10.0
    source_payload = json.loads(source_bank.read_text())
    hard_payload = json.loads(hard_stress_bank.read_text())
    splits = frontier_splits(source_payload)
    excluded = {str(row["id"]) for rows in splits.values() for row in rows}
    rehearsal = ladder_rehearsal_payload(excluded)
    calibration = calibrate_evaluation_dose(
        checkpoint,
        scientist,
        _bank_from_payload(splits["calibration"]),
        ratio=ratio,
        attempts=evaluation_attempts,
        minimum_coverage=minimum_calibration_coverage,
        action_horizon=evaluation_action_horizon,
        seed=seed + 600_000_000,
        device=device,
    )
    selected = calibration["rows"][-1]
    objective_cap = empirical_objective_cap(selected)
    gate = checkpoint_regression_gate(
        checkpoint,
        scientist,
        ratio=ratio,
        games=6,
        minimum_solve_rate=0.8,
        seed=seed + 650_000_000,
        simulations=int(calibration["selected_simulations"] or DOSES[-1]),
        device=device,
    )
    protocol = {
        "schema": "paired-complete-portfolio-L10-progress-smoke-v2",
        "scientist": scientist,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_regression_gate": gate,
        "source_bank": str(source_bank.resolve()),
        "source_bank_sha256": _json_hash(source_payload),
        "hard_stress_bank": str(hard_stress_bank.resolve()),
        "hard_stress_bank_sha256": _json_hash(hard_payload),
        "splits": splits,
        "rehearsal": rehearsal,
        "calibration": calibration,
        "ratio": ratio,
        "empirical_objective_cap": objective_cap,
        "empirical_cap_source": (
            "maximum verified L10 among successful attempts at the selected "
            "frozen calibration dose"
        ),
        "evaluation_attempts": evaluation_attempts,
        "evaluation_action_horizon": evaluation_action_horizon,
        "f_new": f_new,
        "f_old": 1,
        "acquisition_tiers": [list(tier) for tier in ACQUISITION_TIERS],
        "train_steps_per_iteration": train_steps_per_iteration,
        "batch_size": batch_size,
        "block_size": block_size,
        "maximum_recovery_tasks": maximum_recovery_tasks,
        "seed": seed,
        "device": device,
        "torch_threads_per_arm": torch_threads_per_arm,
        "arms": ["block-progress", "transactional"],
        "block_progress_rule": (
            "complete old-plus-seen portfolio solved count nondecreasing and "
            "empirically capped sum L10 nonincreasing; recover regressions, then "
            "restore the block-start network if recovery fails"
        ),
        "lifetime_bank_metric": "best verified semantic L10 per representation",
        "current_network_metric": "four-attempt fixed-dose complete portfolio",
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{manifest} exists; pass resume=True")
        if json.loads(manifest.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("L10 portfolio-progress smoke resume protocol differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, protocol)
    if not calibration["passed"]:
        report = {
            **protocol,
            "completed": False,
            "decision": {
                "passed": False,
                "reason": "no registered L10 dose reached 70% calibration coverage",
            },
        }
        _atomic_json(output / "report.json", report)
        return report

    payloads = []
    for arm in protocol["arms"]:
        payloads.append(
            {
                **protocol,
                "arm": arm,
                "output": str(output / "arms" / arm),
                "new": splits["new"],
                "heldout": splits["heldout"],
                "hard_stress": hard_payload,
                "rehearsal": rehearsal,
                "evaluation_simulations": calibration["selected_simulations"],
                "objective_cap": objective_cap,
                "minimum_rehearsal_solutions": 3,
                "progress_guard": arm == "block-progress",
                "resume": resume,
                "torch_threads": torch_threads_per_arm,
            }
        )
    rows = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_run_arm, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["arm"])
    decision = analyze_portfolio_progress_smoke(rows)
    report = {
        **protocol,
        "completed": all(row.get("completed") for row in rows),
        "arms_results": rows,
        "decision": decision,
    }
    _atomic_json(output / "report.json", report)
    return report
