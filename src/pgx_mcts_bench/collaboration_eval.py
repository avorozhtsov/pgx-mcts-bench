"""Resumable held-out evaluation and checkpoint export for collaboration runs."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import FixedWordGame, load_scientist
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _json_hash,
    _play,
    _restore_scientist,
    _round_dirs,
    _sha256,
    load_round_state,
    play_with_objective_restarts,
    prediction_details,
    verified_record_cost,
)
from pgx_mcts_bench.data import GameRecord, Position
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.search import NeuralMCTS


def export_collaboration_scientist(run: Path, name: str, output: Path) -> dict[str, Any]:
    rounds = _round_dirs(run)
    if not rounds:
        raise FileNotFoundError(f"no committed rounds under {run}")
    state = load_round_state(rounds[-1], map_location="cpu")
    if name not in state["scientists"]:
        raise ValueError(f"scientist {name!r} not present in run")
    manifest = json.loads((run / "manifest.json").read_text())
    initial = next(item for item in manifest["checkpoints"] if item["name"] == name)
    source_payload = torch.load(initial["path"], map_location="cpu", weights_only=False)
    scientist = state["scientists"][name]
    payload = {
        "network": scientist["network"],
        "optimizer": scientist["optimizer"],
        "candidate_spec": source_payload.get("candidate_spec"),
        "solve_calibration": source_payload.get("solve_calibration", {}),
        "collaboration": {
            "run": str(run.resolve()),
            "protocol_sha256": manifest["protocol_sha256"],
            "completed_round": len(rounds) - 1,
            "initial_checkpoint_sha256": initial["sha256"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary)
    os.replace(temporary, output)
    report = {
        "scientist": name,
        "checkpoint": str(output.resolve()),
        "checkpoint_sha256": _sha256(output),
        **payload["collaboration"],
    }
    _atomic_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def _evaluation_record(
    scientist,
    knot,
    ratio: float,
    simulations: int,
    seed: int,
    *,
    objective_cap: float | None = None,
):
    fixed = FixedWordGame(
        scientist.game,
        knot,
        ratio,
        objective_cap=objective_cap,
        cap_type="empirical-global" if objective_cap is not None else "global",
    )
    search_config = replace(scientist.config.search, simulations=simulations)
    search = NeuralMCTS(
        fixed, scientist.network, search_config, scientist.config.train.device
    )
    transition = fixed.reset(seed)
    record: GameRecord = []
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    while not transition.terminated:
        result = search.run(
            transition.state,
            transition.observation,
            transition.legal_actions,
            rng,
            temperature=0.0,
            add_root_noise=False,
        )
        record.append(
            Position(
                observation=transition.observation,
                legal_actions=transition.legal_actions,
                policy=result.policy.astype(np.float32),
                action=result.action,
                player=transition.player,
                role=1,
                episode_seed=seed,
            )
        )
        transition = fixed.step(transition.state, result.action)
    return (
        verified_record_cost(scientist.game, knot, ratio, record),
        {
            "moves_searched": len(record),
            # AlphaZero performs one root evaluation and at most one leaf
            # evaluation per simulation. Keep this explicitly labelled as a
            # scheduled upper bound rather than pretending it is a hardware
            # counter when terminal leaves avoid inference.
            "scheduled_network_evaluations": len(record) * (simulations + 1),
            "wall_seconds": time.perf_counter() - started,
        },
    )


def evaluate_collaboration(
    run: Path,
    output: Path,
    *,
    state: Literal["initial", "final"] = "final",
    split: Literal["new70", "base"] = "new70",
    simulations: int = 128,
    attempts_per_representation: int = 4,
    limit: int = 0,
    seed: int = 0,
    device: str = "cpu",
    resume: bool = False,
    bank: Path | None = None,
) -> dict[str, Any]:
    if attempts_per_representation < 1:
        raise ValueError("attempts_per_representation must be positive")
    run_manifest = json.loads((run / "manifest.json").read_text())
    source = bank or run / ("new-70.json" if split == "new70" else "base.json")
    items = _bank_from_payload(json.loads(source.read_text()))
    if limit:
        items = items[:limit]
    ratios = tuple(float(value) for value in run_manifest["ratios"])
    protocol = {
        "schema": "collaboration-evaluation-v2-paired-attempts",
        "run": str(run.resolve()),
        "run_protocol_sha256": run_manifest["protocol_sha256"],
        "state": state,
        "split": split,
        "external_bank": str(bank.resolve()) if bank is not None else None,
        "split_sha256": _json_hash(json.loads(source.read_text())),
        "simulations": simulations,
        "attempts_per_representation": attempts_per_representation,
        "limit": limit,
        "seed": seed,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; pass resume=True")
        existing = json.loads(manifest_path.read_text())
        if existing["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("evaluation resume protocol differs")
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume without {manifest_path}")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest_path, protocol)

    scientists = [
        load_scientist(
            item["name"],
            Path(item["path"]),
            seed=seed + index * 10_000,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=bool(
                run_manifest.get(
                    "remaining_budget_channel",
                    run_manifest.get("objective_budget", False),
                )
            ),
        )
        for index, item in enumerate(run_manifest["checkpoints"])
    ]
    if state == "final":
        rounds = _round_dirs(run)
        if not rounds:
            raise FileNotFoundError(f"no committed rounds under {run}")
        saved = load_round_state(rounds[-1], map_location=device)
        for scientist in scientists:
            _restore_scientist(scientist, saved["scientists"][scientist.name])
    action_horizon = int(
        run_manifest.get("solution_definition", {}).get(
            "native_action_horizon", scientists[0].config.game.simplify_budget
        )
    )
    for scientist in scientists:
        scientist.config = replace(
            scientist.config,
            game=replace(scientist.config.game, simplify_budget=action_horizon),
        )
        scientist.game = make_game(scientist.config.game)

    item_dir = output / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    for item_index, item in enumerate(items):
        path = item_dir / f"{item_index:04d}.json"
        if path.exists():
            continue
        attempts = []
        for scientist_index, scientist in enumerate(scientists):
            for ratio_index, ratio in enumerate(ratios):
                for attempt_index in range(attempts_per_representation):
                    attempt_seed = (
                        seed
                        + item_index * 1_000_000
                        + scientist_index * 10_000
                        + ratio_index * 1_000
                        + attempt_index
                    )
                    verified, compute = _evaluation_record(
                        scientist, item.knot, ratio, simulations, attempt_seed
                    )
                    row = {
                        "scientist": scientist.name,
                        "ratio": ratio,
                        "attempt": attempt_index,
                        "solved": verified is not None,
                        "compute": compute,
                    }
                    if verified is not None:
                        row.update(
                            crossing_changes=verified[0],
                            moves=verified[1],
                            objective=ratio * verified[0] + verified[1],
                        )
                    attempts.append(row)
        _atomic_json(
            path,
            {
                "item": item.id,
                "difficulty_quartile": item.difficulty_quartile,
                "attempts": attempts,
            },
        )

    rows = [json.loads(path.read_text()) for path in sorted(item_dir.glob("*.json"))]
    summary = {}
    move_budget = int(scientists[0].config.game.simplify_budget)
    for ratio in ratios:
        coverage = 0
        capped_sum = 0.0
        crossing_capped_sum = 0.0
        solved_objectives = []
        solved_items = []
        best_by_item = {}
        for row in rows:
            solved = [
                attempt
                for attempt in row["attempts"]
                if attempt["ratio"] == ratio and attempt["solved"]
            ]
            if solved:
                coverage += 1
                best = min(
                    solved,
                    key=lambda attempt: (
                        attempt["objective"],
                        attempt["crossing_changes"],
                        attempt["moves"],
                    ),
                )
                capped_sum += best["objective"]
                crossing_capped_sum += best["crossing_changes"]
                solved_objectives.append(best["objective"])
                solved_items.append(row["item"])
                best_by_item[row["item"]] = {
                    key: best[key]
                    for key in ("scientist", "crossing_changes", "moves", "objective")
                }
            else:
                capped_sum += ratio * 20 + move_budget
                crossing_capped_sum += 20
        summary[str(ratio)] = {
            "portfolio_solved": coverage,
            "representations": len(rows),
            "solve_rate": coverage / len(rows) if rows else 0.0,
            "capped_objective_sum": capped_sum,
            "crossing_capped_sum": crossing_capped_sum,
            "conditional_mean_objective": (
                sum(solved_objectives) / len(solved_objectives)
                if solved_objectives
                else None
            ),
            "solved_items": sorted(solved_items),
            "best_by_item": best_by_item,
            "compute": {
                "scheduled_network_evaluations": sum(
                    attempt.get("compute", {}).get("scheduled_network_evaluations", 0)
                    for row in rows
                    for attempt in row["attempts"]
                    if attempt["ratio"] == ratio
                ),
                "wall_seconds": sum(
                    attempt.get("compute", {}).get("wall_seconds", 0.0)
                    for row in rows
                    for attempt in row["attempts"]
                    if attempt["ratio"] == ratio
                ),
            },
        }
    report = {**protocol, "completed_items": len(rows), "summary": summary}
    _atomic_json(output / "report.json", report)
    return report


def compare_collaboration_evaluations(
    treatment: Path,
    control: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Pair two completed evaluations by representation identity and ratio."""
    treatment_report = json.loads((treatment / "report.json").read_text())
    control_report = json.loads((control / "report.json").read_text())
    if treatment_report["split_sha256"] != control_report["split_sha256"]:
        raise ValueError("evaluation splits differ")
    if treatment_report["completed_items"] != control_report["completed_items"]:
        raise ValueError("evaluation item counts differ")

    comparisons = {}
    shared_ratios = sorted(
        set(treatment_report["summary"]) & set(control_report["summary"]),
        key=float,
    )
    for ratio in shared_ratios:
        treatment_summary = treatment_report["summary"][ratio]
        control_summary = control_report["summary"][ratio]
        treatment_solved = set(treatment_summary.get("solved_items", ()))
        control_solved = set(control_summary.get("solved_items", ()))
        common = treatment_solved & control_solved
        treatment_best = treatment_summary.get("best_by_item", {})
        control_best = control_summary.get("best_by_item", {})
        common_deltas = {
            item: treatment_best[item]["objective"] - control_best[item]["objective"]
            for item in sorted(common)
        }
        comparisons[ratio] = {
            "intersection": sorted(common),
            "treatment_only": sorted(treatment_solved - control_solved),
            "control_only": sorted(control_solved - treatment_solved),
            "union": sorted(treatment_solved | control_solved),
            "jaccard": (
                len(common) / len(treatment_solved | control_solved)
                if treatment_solved | control_solved
                else 1.0
            ),
            "common_objective_deltas": common_deltas,
            "common_objective_delta_sum": sum(common_deltas.values()),
            "capped_objective_delta": (
                treatment_summary["capped_objective_sum"]
                - control_summary["capped_objective_sum"]
            ),
            "solve_delta": (
                treatment_summary["portfolio_solved"]
                - control_summary["portfolio_solved"]
            ),
            "scheduled_network_evaluation_delta": (
                treatment_summary.get("compute", {}).get(
                    "scheduled_network_evaluations", 0
                )
                - control_summary.get("compute", {}).get(
                    "scheduled_network_evaluations", 0
                )
            ),
            "wall_seconds_delta": (
                treatment_summary.get("compute", {}).get("wall_seconds", 0.0)
                - control_summary.get("compute", {}).get("wall_seconds", 0.0)
            ),
        }
    report = {
        "schema": "collaboration-paired-comparison-v1",
        "treatment": str(treatment.resolve()),
        "control": str(control.resolve()),
        "split_sha256": treatment_report["split_sha256"],
        "comparisons": comparisons,
    }
    if output is not None:
        _atomic_json(output, report)
    return report


def benchmark_objective_budget(
    checkpoints: dict[str, Path],
    bank_path: Path,
    output: Path,
    *,
    ratio: float = 10.0,
    simulations: int = 16,
    limit: int = 10,
    item_ids: tuple[str, ...] = (),
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Paired fixed-clock versus cap-and-restart regression on frozen tasks."""
    all_items = _bank_from_payload(json.loads(bank_path.read_text()))
    if item_ids:
        by_id = {item.id: item for item in all_items}
        missing = [item_id for item_id in item_ids if item_id not in by_id]
        if missing:
            raise ValueError(f"unknown bank items: {missing}")
        items = [by_id[item_id] for item_id in item_ids]
    else:
        items = all_items[:limit]
    bank_index = {item.id: index for index, item in enumerate(all_items)}
    rows = []
    for scientist_index, (name, checkpoint) in enumerate(checkpoints.items()):
        baseline = load_scientist(
            name,
            checkpoint,
            seed=seed + scientist_index * 10_000,
            device=device,
            simulations=simulations,
            require_factorized=True,
        )
        budgeted = load_scientist(
            name,
            checkpoint,
            seed=seed + scientist_index * 10_000,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=True,
        )
        for item in items:
            attempt_seed = (
                seed + scientist_index * 100_000 + bank_index[item.id] * 100
            )
            detail = prediction_details(budgeted, item, (ratio,))[0]
            predicted = (
                ratio * detail["predicted_crossing_changes"]
                + detail["predicted_moves"]
            )
            started = time.perf_counter()
            baseline_record = _play(
                baseline,
                item.knot,
                ratio,
                simulations=simulations,
                seed=attempt_seed,
            )
            baseline_seconds = time.perf_counter() - started
            started = time.perf_counter()
            budget_record, budget = play_with_objective_restarts(
                budgeted,
                item.knot,
                ratio,
                predicted_objective=predicted,
                simulations=simulations,
                seed=attempt_seed,
                max_restarts=0,
            )
            budget_seconds = time.perf_counter() - started
            baseline_verified = verified_record_cost(
                baseline.game, item.knot, ratio, baseline_record
            )
            budget_verified = verified_record_cost(
                budgeted.game, item.knot, ratio, budget_record
            )
            audit_recovery = None
            if baseline_verified is not None and budget_verified is None:
                audit_record, audit_budget = play_with_objective_restarts(
                    budgeted,
                    item.knot,
                    ratio,
                    predicted_objective=predicted,
                    simulations=simulations,
                    seed=attempt_seed,
                )
                audit_verified = verified_record_cost(
                    budgeted.game, item.knot, ratio, audit_record
                )
                audit_recovery = {
                    "solved": audit_verified is not None,
                    "cost": list(audit_verified[:2]) if audit_verified else None,
                    "budget": audit_budget,
                }
            rows.append(
                {
                    "scientist": name,
                    "item": item.id,
                    "predicted_objective": predicted,
                    "baseline": {
                        "solved": baseline_verified is not None,
                        "cost": list(baseline_verified[:2]) if baseline_verified else None,
                        "moves_searched": len(baseline_record),
                        "scheduled_network_evaluations": len(baseline_record)
                        * (simulations + 1),
                        "wall_seconds": baseline_seconds,
                    },
                    "budgeted": {
                        "solved": budget_verified is not None,
                        "cost": list(budget_verified[:2]) if budget_verified else None,
                        "moves_searched": sum(
                            int(attempt["moves"]) for attempt in budget["attempts"]
                        ),
                        "scheduled_network_evaluations": sum(
                            int(attempt["moves"]) for attempt in budget["attempts"]
                        )
                        * (simulations + 1),
                        "wall_seconds": budget_seconds,
                        "budget": budget,
                    },
                    "audit_recovery": audit_recovery,
                }
            )
    baseline_evaluations = sum(
        row["baseline"]["scheduled_network_evaluations"] for row in rows
    )
    budget_evaluations = sum(
        row["budgeted"]["scheduled_network_evaluations"] for row in rows
    )
    hard_cap_lost = [
        f'{row["scientist"]}:{row["item"]}'
        for row in rows
        if row["baseline"]["solved"] and not row["budgeted"]["solved"]
    ]
    unrecovered = [
        f'{row["scientist"]}:{row["item"]}'
        for row in rows
        if row["baseline"]["solved"]
        and not row["budgeted"]["solved"]
        and not (row["audit_recovery"] or {}).get("solved", False)
    ]
    item_names = sorted({row["item"] for row in rows})
    baseline_portfolio = {
        item
        for item in item_names
        if any(row["item"] == item and row["baseline"]["solved"] for row in rows)
    }
    budgeted_portfolio = {
        item
        for item in item_names
        if any(row["item"] == item and row["budgeted"]["solved"] for row in rows)
    }
    audit_recovered_portfolio = {
        item
        for item in item_names
        if any(
            row["item"] == item
            and (
                row["budgeted"]["solved"]
                or (row["audit_recovery"] or {}).get("solved", False)
            )
            for row in rows
        )
    }
    report = {
        "schema": "objective-budget-regression-v1",
        "bank": str(bank_path.resolve()),
        "ratio": ratio,
        "simulations": simulations,
        "limit": limit,
        "item_ids": list(item_ids),
        "seed": seed,
        "rows": rows,
        "summary": {
            "attempt_pairs": len(rows),
            "baseline_solved": sum(row["baseline"]["solved"] for row in rows),
            "budgeted_solved": sum(row["budgeted"]["solved"] for row in rows),
            "hard_cap_lost_baseline_solves": hard_cap_lost,
            "audit_unrecovered_baseline_solves": unrecovered,
            "baseline_portfolio_solved_items": sorted(baseline_portfolio),
            "budgeted_portfolio_solved_items": sorted(budgeted_portfolio),
            "hard_cap_lost_portfolio_items": sorted(
                baseline_portfolio - budgeted_portfolio
            ),
            "audit_unrecovered_portfolio_items": sorted(
                baseline_portfolio - audit_recovered_portfolio
            ),
            "baseline_scheduled_network_evaluations": baseline_evaluations,
            "budgeted_scheduled_network_evaluations": budget_evaluations,
            "scheduled_network_evaluation_savings": (
                1.0 - budget_evaluations / baseline_evaluations
                if baseline_evaluations
                else 0.0
            ),
            "baseline_wall_seconds": sum(
                row["baseline"]["wall_seconds"] for row in rows
            ),
            "budgeted_wall_seconds": sum(
                row["budgeted"]["wall_seconds"] for row in rows
            ),
        },
    }
    report["summary"]["accepted"] = (
        not (baseline_portfolio - audit_recovered_portfolio)
        and report["summary"]["scheduled_network_evaluation_savings"] >= 0.20
    )
    _atomic_json(output, report)
    return report
