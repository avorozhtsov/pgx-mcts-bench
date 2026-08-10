"""Paired audit of the common structural objective-budget protocol."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import KnotItem, load_scientist
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _json_hash,
    _play,
    _sha256,
    play_with_common_objective_restarts,
    verified_record_cost,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.invariant_pretrain import _variant
from pgx_mcts_bench.ladder import STAGES, _config, candidates
from pgx_mcts_bench.semantic_verifier import SemanticBraidVerifier


def _candidate(name: str):
    return next(candidate for candidate in candidates() if candidate.name == name)


def ladder_representation_panel(
    count: int,
    *,
    seed: int,
    scientist: str = "s-window-128",
    stage_count: int = 10,
    excluded: set[tuple[tuple[int, ...], int]] | None = None,
) -> list[KnotItem]:
    """Generate a deterministic representation panel from mastered ladder rungs."""
    if count < 1:
        raise ValueError("representation count must be positive")
    if not 1 <= stage_count <= len(STAGES):
        raise ValueError("stage_count is outside the ladder")
    candidate = _candidate(scientist)
    # Unscrambled source rungs have only one representation.  The audit needs
    # 100 distinct representations, not 100 renamed copies of the same word,
    # so draw only from the mastered scrambled rungs and reject exact repeats.
    stage_indices = [
        index for index, (_, scramble) in enumerate(STAGES[:stage_count]) if scramble > 0
    ]
    if not stage_indices:
        raise ValueError("the selected ladder prefix has no scrambled rungs")
    configs = {
        index: _config(candidate, STAGES[index], seed, "cpu", selfplay_games=1)
        for index in stage_indices
    }
    games = {index: make_game(configs[index].game) for index in stage_indices}
    verifiers = {
        index: SemanticBraidVerifier.from_config(configs[index].game)
        for index in stage_indices
    }
    seen = set(excluded or set())
    panel = []
    for index in range(count):
        stage_index = stage_indices[index % len(stage_indices)]
        sample_index = index // len(stage_indices)
        for retry in range(10_000):
            episode_seed = (
                seed
                + stage_index * 10_000_019
                + sample_index * 100_003
                + retry * 1_000_003
            )
            transition = games[stage_index].reset(episode_seed)
            raw = games[stage_index].unwrap(transition.state)
            generated = KnotItem(
                name="generated",
                crossings=int(np.count_nonzero(np.asarray(raw._word))),
                word=tuple(int(value) for value in np.asarray(raw._word) if int(value)),
                strands=int(np.asarray(raw._n)),
            )
            view = _variant(
                verifiers[stage_index],
                generated,
                depth=1 + (sample_index + retry) % 16,
                seed=episode_seed + 700_000_001,
            )
            word = tuple(int(value) for value in view["word"])
            strands = int(view["strands"])
            representation = (word, strands)
            if representation not in seen:
                seen.add(representation)
                break
        else:
            raise RuntimeError(
                f"could not generate {count} distinct ladder representations"
            )
        source, scramble = STAGES[stage_index]
        panel.append(
            KnotItem(
                name=f"r{stage_index:02d}-{source}+{scramble}-v{sample_index:03d}",
                crossings=len(word),
                word=word,
                strands=strands,
            )
        )
    if len({item.name for item in panel}) != len(panel):
        raise AssertionError("ladder representation identifiers are not unique")
    return panel


def _record_summary(scientist: Any, knot: KnotItem, ratio: float, record: Any) -> dict[str, Any]:
    verified = verified_record_cost(scientist.game, knot, ratio, record)
    return {
        "solved": verified is not None,
        "cost": list(verified[:2]) if verified is not None else None,
        "actions": [int(position.action) for position in record],
        "termination_reason": record[0].termination_reason if record else "empty-record",
        "objective_censored": bool(record and record[0].objective_censored),
        "semantic_moves": float(record[0].final_moves) if record else 0.0,
        "native_plies": float(record[0].final_native_plies) if record else 0.0,
        "best_residual_word_length": (
            int(record[0].best_residual_word_length) if record else len(knot.word)
        ),
    }


def _scheduled_evaluations(records: list[Any], simulations: int) -> int:
    return sum(len(record) for record in records) * (simulations + 1)


def _calibration_worker(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    scientist = load_scientist(
        payload["scientist"],
        Path(payload["checkpoint"]),
        seed=payload["seed"],
        device=payload["device"],
        simulations=payload["simulations"],
        require_factorized=True,
        objective_budget_channel=True,
    )
    knots = [KnotItem(**item) for item in payload["panel"]]
    horizon = int(scientist.config.game.simplify_budget)
    global_cap = (payload["ratio"] + 1.0) * horizon
    rows = []
    for item_index, knot in enumerate(knots):
        for attempt in range(payload["attempts_per_representation"]):
            attempt_seed = payload["seed"] + item_index * 100_003 + attempt * 1_009
            record = _play(
                scientist,
                knot,
                payload["ratio"],
                simulations=payload["simulations"],
                seed=attempt_seed,
                objective_cap=global_cap,
                cap_type="common-global-calibration",
            )
            rows.append(
                {
                    "representation": knot.name,
                    "attempt": attempt,
                    "seed": attempt_seed,
                    **_record_summary(scientist, knot, payload["ratio"], record),
                }
            )
    return {
        "scientist": payload["scientist"],
        "simulations": payload["simulations"],
        "native_action_horizon": horizon,
        "rows": rows,
    }


def _audit_worker(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    direct = load_scientist(
        payload["scientist"],
        Path(payload["checkpoint"]),
        seed=payload["seed"],
        device=payload["device"],
        simulations=payload["simulations"],
        require_factorized=True,
        objective_budget_channel=True,
    )
    structural = load_scientist(
        payload["scientist"],
        Path(payload["checkpoint"]),
        seed=payload["seed"],
        device=payload["device"],
        simulations=payload["simulations"],
        require_factorized=True,
        objective_budget_channel=True,
    )
    knots = [KnotItem(**item) for item in payload["panel"]]
    horizon = int(direct.config.game.simplify_budget)
    if horizon != int(structural.config.game.simplify_budget):
        raise AssertionError("paired scientists disagree on action horizon")
    global_cap = (payload["ratio"] + 1.0) * horizon
    rows = []
    for item_index, knot in enumerate(knots):
        for attempt in range(payload["attempts_per_representation"]):
            attempt_seed = payload["seed"] + item_index * 100_003 + attempt * 1_009
            direct_record = _play(
                direct,
                knot,
                payload["ratio"],
                simulations=payload["simulations"],
                seed=attempt_seed,
                objective_cap=global_cap,
                cap_type="common-global-audit",
            )
            retained = []
            structural_record, budget = play_with_common_objective_restarts(
                structural,
                knot,
                payload["ratio"],
                simulations=payload["simulations"],
                seed=attempt_seed,
                retained_records=retained,
            )
            direct_summary = _record_summary(direct, knot, payload["ratio"], direct_record)
            structural_summary = _record_summary(
                structural, knot, payload["ratio"], structural_record
            )
            rows.append(
                {
                    "representation": knot.name,
                    "attempt": attempt,
                    "seed": attempt_seed,
                    "direct": {
                        **direct_summary,
                        "scheduled_network_evaluations": _scheduled_evaluations(
                            [direct_record], payload["simulations"]
                        ),
                    },
                    "structural": {
                        **structural_summary,
                        "scheduled_network_evaluations": _scheduled_evaluations(
                            retained, payload["simulations"]
                        ),
                        "budget": budget,
                    },
                    "exact_final_actions": (
                        direct_summary["actions"] == structural_summary["actions"]
                    ),
                    "exact_final_outcome": (
                        direct_summary["solved"] == structural_summary["solved"]
                        and direct_summary["cost"] == structural_summary["cost"]
                        and direct_summary["termination_reason"]
                        == structural_summary["termination_reason"]
                    ),
                }
            )
    return {
        "scientist": payload["scientist"],
        "simulations": payload["simulations"],
        "native_action_horizon": horizon,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }


def _portfolio(
    scientist_rows: list[dict[str, Any]],
    representations: list[str],
    *,
    arm: str,
    ratio: float,
    failure_cap: float,
) -> dict[str, Any]:
    best: dict[str, tuple[float, str, int] | None] = {
        representation: None for representation in representations
    }
    for scientist in scientist_rows:
        for row in scientist["rows"]:
            result = row[arm]
            if not result["solved"]:
                continue
            objective = ratio * result["cost"][0] + result["cost"][1]
            candidate = (objective, scientist["scientist"], int(row["attempt"]))
            current = best[row["representation"]]
            if current is None or candidate < current:
                best[row["representation"]] = candidate
    solved = sorted(representation for representation, value in best.items() if value)
    return {
        "solved": solved,
        "solve_rate": len(solved) / len(representations),
        "capped_objective": sum(
            value[0] if value is not None else failure_cap for value in best.values()
        ),
        "best": {
            representation: (
                None
                if value is None
                else {"objective": value[0], "scientist": value[1], "attempt": value[2]}
            )
            for representation, value in best.items()
        },
    }


def structural_audit_decision(
    *,
    direct: dict[str, Any],
    structural: dict[str, Any],
    exact_pair_outcomes: bool,
    exact_pair_actions: bool,
    minimum_solve_rate: float,
    direct_evaluations: int,
    structural_evaluations: int,
) -> dict[str, Any]:
    checks = {
        "direct_portfolio_solve_rate_at_least_minimum": (
            direct["solve_rate"] >= minimum_solve_rate
        ),
        "identical_portfolio_solved_set": direct["solved"] == structural["solved"],
        "identical_portfolio_capped_objective": (
            direct["capped_objective"] == structural["capped_objective"]
        ),
        "exact_paired_final_outcomes": exact_pair_outcomes,
    }
    compute_ratio = (
        structural_evaluations / direct_evaluations if direct_evaluations else float("inf")
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostics": {"exact_paired_final_actions": exact_pair_actions},
        "minimum_solve_rate": minimum_solve_rate,
        "structural_over_direct_evaluation_ratio": compute_ratio,
        "economical": compute_ratio <= 1.0,
        "main_experiment_status": (
            "equivalence admitted; keep opt-in pending a separate learning-distribution gate"
            if all(checks.values())
            else "keep structural objective budgeting disabled in the main comparison"
        ),
    }


def _calibration_portfolio(
    rows: list[dict[str, Any]], representations: list[str]
) -> dict[str, Any]:
    scientist_rates = {}
    solved = set()
    for scientist in rows:
        scientist_solved = {
            row["representation"] for row in scientist["rows"] if row["solved"]
        }
        scientist_rates[scientist["scientist"]] = len(scientist_solved) / len(representations)
        solved.update(scientist_solved)
    return {
        "portfolio_solved": sorted(solved),
        "portfolio_solve_rate": len(solved) / len(representations),
        "scientist_solve_rates": scientist_rates,
    }


def run_structural_budget_audit(
    checkpoints: dict[str, Path],
    output: Path,
    *,
    representations: int = 100,
    calibration_representations: int = 20,
    attempts_per_representation: int = 2,
    ratio: float = 1000.0,
    simulation_doses: tuple[int, ...] = (32, 64, 128),
    minimum_solve_rate: float = 0.70,
    seed: int = 20261600,
    workers: int = 3,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    if not checkpoints:
        raise ValueError("at least one scientist checkpoint is required")
    if attempts_per_representation < 1:
        raise ValueError("attempts per representation must be positive")
    if not simulation_doses or any(dose < 1 for dose in simulation_doses):
        raise ValueError("simulation doses must be positive")
    calibration_panel = ladder_representation_panel(
        calibration_representations, seed=seed + 10_000_000
    )
    audit_panel = ladder_representation_panel(
        representations,
        seed=seed + 20_000_000,
        excluded={(item.word, item.strands) for item in calibration_panel},
    )
    protocol = {
        "schema": "common-structural-budget-equivalence-v1",
        "checkpoints": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in checkpoints.items()
        },
        "representations": representations,
        "calibration_representations": calibration_representations,
        "attempts_per_representation": attempts_per_representation,
        "ratio": ratio,
        "simulation_doses": list(simulation_doses),
        "minimum_solve_rate": minimum_solve_rate,
        "panel_source": "first-ten-ladder-rungs-with-disjoint-generator-seeds",
        "seed": seed,
        "device": device,
        "scientist_prediction_used": False,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{manifest} exists; pass resume=True")
        if json.loads(manifest.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("structural-budget audit resume protocol differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, protocol)
        _atomic_json(
            output / "calibration-panel.json",
            [asdict(item) for item in calibration_panel],
        )
        _atomic_json(output / "audit-panel.json", [asdict(item) for item in audit_panel])

    selected_simulations = None
    calibration_rows = []
    for dose in simulation_doses:
        dose_rows = []
        pending = []
        for scientist_index, (scientist, checkpoint) in enumerate(checkpoints.items()):
            path = output / "calibration" / f"sim{dose}-{scientist}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                dose_rows.append(json.loads(path.read_text()))
                continue
            payload = {
                "scientist": scientist,
                "checkpoint": str(checkpoint),
                "panel": [asdict(item) for item in calibration_panel],
                "ratio": ratio,
                "simulations": dose,
                "attempts_per_representation": attempts_per_representation,
                "seed": seed + scientist_index * 100_000_003,
                "device": device,
            }
            pending.append((path, payload))
        if pending:
            with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
                futures = {
                    executor.submit(_calibration_worker, payload): path
                    for path, payload in pending
                }
                for future in as_completed(futures):
                    row = future.result()
                    _atomic_json(futures[future], row)
                    dose_rows.append(row)
        dose_rows.sort(key=lambda row: row["scientist"])
        calibration = _calibration_portfolio(
            dose_rows, [item.name for item in calibration_panel]
        )
        calibration_rows.append({"simulations": dose, **calibration})
        if calibration["portfolio_solve_rate"] >= minimum_solve_rate:
            selected_simulations = dose
            break
    if selected_simulations is None:
        report = {
            **protocol,
            "calibration": calibration_rows,
            "decision": {
                "passed": False,
                "reason": "no registered simulation dose reached the minimum portfolio solve rate",
                "main_experiment_status": "keep structural objective budgeting disabled",
            },
        }
        _atomic_json(output / "report.json", report)
        return report

    scientist_rows = []
    pending = []
    for scientist_index, (scientist, checkpoint) in enumerate(checkpoints.items()):
        path = output / "audit" / f"{scientist}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            scientist_rows.append(json.loads(path.read_text()))
            continue
        payload = {
            "scientist": scientist,
            "checkpoint": str(checkpoint),
            "panel": [asdict(item) for item in audit_panel],
            "ratio": ratio,
            "simulations": selected_simulations,
            "attempts_per_representation": attempts_per_representation,
            "seed": seed + 500_000_000 + scientist_index * 100_000_003,
            "device": device,
        }
        pending.append((path, payload))
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(_audit_worker, payload): path for path, payload in pending}
            for future in as_completed(futures):
                row = future.result()
                _atomic_json(futures[future], row)
                scientist_rows.append(row)
    scientist_rows.sort(key=lambda row: row["scientist"])
    horizons = {int(row["native_action_horizon"]) for row in scientist_rows}
    if len(horizons) != 1:
        raise ValueError(f"scientists disagree on native action horizon: {sorted(horizons)}")
    failure_cap = (ratio + 1.0) * horizons.pop()
    representation_names = [item.name for item in audit_panel]
    direct = _portfolio(
        scientist_rows,
        representation_names,
        arm="direct",
        ratio=ratio,
        failure_cap=failure_cap,
    )
    structural = _portfolio(
        scientist_rows,
        representation_names,
        arm="structural",
        ratio=ratio,
        failure_cap=failure_cap,
    )
    all_rows = [row for scientist in scientist_rows for row in scientist["rows"]]
    direct_evaluations = sum(
        int(row["direct"]["scheduled_network_evaluations"]) for row in all_rows
    )
    structural_evaluations = sum(
        int(row["structural"]["scheduled_network_evaluations"]) for row in all_rows
    )
    decision = structural_audit_decision(
        direct=direct,
        structural=structural,
        exact_pair_outcomes=all(bool(row["exact_final_outcome"]) for row in all_rows),
        exact_pair_actions=all(bool(row["exact_final_actions"]) for row in all_rows),
        minimum_solve_rate=minimum_solve_rate,
        direct_evaluations=direct_evaluations,
        structural_evaluations=structural_evaluations,
    )
    report = {
        **protocol,
        "selected_simulations": selected_simulations,
        "calibration": calibration_rows,
        "direct": direct,
        "structural": structural,
        "pair_summary": {
            "pairs": len(all_rows),
            "exact_final_outcomes": sum(bool(row["exact_final_outcome"]) for row in all_rows),
            "exact_final_actions": sum(bool(row["exact_final_actions"]) for row in all_rows),
            "restarts": sum(
                int(row["structural"]["budget"]["restart_count"]) for row in all_rows
            ),
            "direct_scheduled_network_evaluations": direct_evaluations,
            "structural_scheduled_network_evaluations": structural_evaluations,
        },
        "scientist_artifacts": [
            {
                "scientist": row["scientist"],
                "wall_seconds": row["wall_seconds"],
                "artifact": str((output / "audit" / f"{row['scientist']}.json").resolve()),
            }
            for row in scientist_rows
        ],
        "decision": decision,
    }
    _atomic_json(output / "report.json", report)
    return report
