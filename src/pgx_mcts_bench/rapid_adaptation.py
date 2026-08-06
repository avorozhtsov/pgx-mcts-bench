"""Disposable per-representation ladder-like adaptation experiments."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import FixedWordGame, KnotItem, load_scientist
from pgx_mcts_bench.collaboration_eval import _evaluation_record
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _json_hash,
    _replay_representation_embedding,
    _sha256,
    verified_record_cost,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, candidates, evaluate_stage
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import (
    attach_policy_value_preservation_teacher,
    play_selfplay_games,
    train_alphazero_step,
)

F_LEVELS = (5, 8, 10, 12, 14, 16)
BLOCK_SIZES = (20, 40, 40, 40, 40, 20)


def promoted_checkpoint_metadata(checkpoint: Path, scientist: str) -> dict[str, Any]:
    """Reject terminal snapshots that never cleared the rung they attempted."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_scientist = payload.get("candidate")
    if checkpoint_scientist is not None and checkpoint_scientist != scientist:
        raise ValueError(f"checkpoint belongs to {checkpoint_scientist!r}, not {scientist!r}")
    stage_result = payload.get("stage_result")
    if not isinstance(stage_result, dict):
        raise ValueError("rapid adaptation requires a checkpoint with embedded stage_result")
    if not bool(stage_result.get("promoted")):
        raise ValueError(
            "rapid adaptation requires a promoted checkpoint; "
            f"{checkpoint} records stage={stage_result.get('stage')}, "
            f"solve_rate={stage_result.get('solve_rate')}, "
            f"reason={stage_result.get('reason')!r}"
        )
    return {
        "candidate": checkpoint_scientist,
        "stage": int(stage_result["stage"]),
        "source": str(stage_result["source"]),
        "scramble": int(stage_result["scramble"]),
        "embedded_solve_rate": float(stage_result["solve_rate"]),
        "reason": str(stage_result["reason"]),
    }


def checkpoint_regression_gate(
    checkpoint: Path,
    scientist: str,
    *,
    ratio: float,
    games: int,
    minimum_solve_rate: float,
    seed: int,
    simulations: int,
    device: str,
) -> dict[str, Any]:
    """Reproduce held-out performance on the checkpoint's promoted rung."""
    metadata = promoted_checkpoint_metadata(checkpoint, scientist)
    by_name = {candidate.name: candidate for candidate in candidates()}
    candidate = by_name[scientist]
    if simulations:
        candidate = replace(candidate, simulations=simulations)
    stage = (metadata["source"], metadata["scramble"])
    loaded = load_scientist(
        scientist,
        checkpoint,
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    config = _config(candidate, stage, seed, device, selfplay_games=1)
    config = replace(
        config,
        game=replace(config.game, objective_budget_channel=True),
        model=loaded.config.model,
    )
    game = make_game(config.game)
    network = loaded.network.eval()
    measured = evaluate_stage(
        game,
        network,
        config,
        games,
        seed,
        ratios=(ratio,),
    )[ratio]
    solve_rate = float(measured["solved"])
    result = {
        **metadata,
        "games": games,
        "ratio": ratio,
        "simulations": simulations,
        "move_budget": int(config.game.simplify_budget),
        "seed": seed,
        "minimum_solve_rate": minimum_solve_rate,
        "measured_solve_rate": solve_rate,
        "measured_crossings": float(measured["crossings"]),
        "measured_moves": float(measured["moves"]),
        "passed": solve_rate >= minimum_solve_rate,
    }
    if not result["passed"]:
        raise ValueError(
            "checkpoint regression gate failed: "
            f"measured {solve_rate:.3f} < {minimum_solve_rate:.3f} "
            f"on {metadata['source']}+{metadata['scramble']}"
        )
    return result


def adaptation_blocks(total: int) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` blocks, truncating the final declared block."""
    blocks = []
    start = 0
    for size in BLOCK_SIZES:
        if start >= total:
            break
        stop = min(total, start + size)
        blocks.append((start, stop))
        start = stop
    if start < total:
        blocks.append((start, total))
    return blocks


def next_f(current: int, solve_rate: float, threshold: float = 0.8) -> int:
    """Advance one registered F level after a sub-threshold completed block."""
    if solve_rate >= threshold:
        return current
    index = F_LEVELS.index(current)
    return F_LEVELS[min(index + 1, len(F_LEVELS) - 1)]


def select_old_items(old_bank: list[Any], target: KnotItem, count: int, seed: int) -> list[Any]:
    """Choose distinct deterministic rehearsal items, excluding the target."""
    eligible = [item for item in old_bank if item.id != target.name and item.id != "10_124"]
    eligible.sort(
        key=lambda item: hashlib.sha256(f"{seed}:{target.name}:{item.id}".encode()).digest()
    )
    if len(eligible) < count:
        raise ValueError("old bank has too few distinct rehearsal representations")
    return eligible[:count]


def _iteration_schedule(current_iterations: int, old_items: list[Any]) -> list[tuple[str, Any]]:
    """Interleave rehearsal evenly; for 5+1 this is C,C,C,O,C,C."""
    schedule: list[tuple[str, Any]] = [("current", None)] * current_iterations
    for old_index, item in enumerate(old_items):
        position = round((old_index + 1) * len(schedule) / (len(old_items) + 1))
        schedule.insert(position, ("old", item))
    return schedule


def replay_has_native_success(scientist: Any) -> bool:
    """Whether replay contains a genuine solved self-play episode."""
    return any(
        game
        and not bool(getattr(game[0], "shared_witness", False))
        and not bool(getattr(game[0], "objective_censored", False))
        and float(getattr(game[0], "solved", -1.0)) > 0.5
        for game in scientist.replay.games
    )


def _diagnostic_evaluation(
    scientist: Any,
    target: KnotItem,
    ratio: float,
    simulations: int,
    seed: int,
    games: int,
) -> tuple[dict[str, Any], int]:
    rows = []
    compute = 0
    for game_index in range(games):
        verified, measured = _evaluation_record(
            scientist,
            target,
            ratio,
            simulations,
            seed + game_index,
        )
        compute += int(measured["scheduled_network_evaluations"])
        rows.append(
            {
                "seed": seed + game_index,
                "solved": verified is not None,
                "crossing_changes": verified[0] if verified is not None else None,
                "moves": verified[1] if verified is not None else None,
                "objective": (
                    ratio * verified[0] + verified[1] if verified is not None else None
                ),
            }
        )
    return {
        "games": games,
        "solved": sum(bool(row["solved"]) for row in rows),
        "solve_rate": sum(bool(row["solved"]) for row in rows) / games,
        "solved_seeds": [row["seed"] for row in rows if row["solved"]],
        "rows": rows,
    }, compute


def _run_one_task(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    target = KnotItem(**payload["target"])
    old_bank = _bank_from_payload(payload["old_bank"])
    old_items = select_old_items(old_bank, target, payload["f_old"], payload["seed"])
    scientist = load_scientist(
        payload["scientist"],
        Path(payload["checkpoint"]),
        seed=payload["seed"],
        device=payload["device"],
        simulations=payload["simulations"],
        require_factorized=True,
        objective_budget_channel=True,
    )
    if payload.get("preservation_teacher", False):
        attach_policy_value_preservation_teacher(scientist.network)
    diagnostic_games = int(payload.get("diagnostic_evaluation_games", 0))
    diagnostic_seed = int(payload.get("diagnostic_evaluation_seed", payload["seed"] + 850_000_000))
    before_diagnostic = None
    diagnostic_compute = 0
    if diagnostic_games:
        before_diagnostic, diagnostic_compute = _diagnostic_evaluation(
            scientist,
            target,
            payload["ratio"],
            payload["simulations"],
            diagnostic_seed,
            diagnostic_games,
        )
    best: tuple[int, int, list[int]] | None = None
    adaptation_compute = 0
    training_seconds = 0.0
    iteration_rows = []
    first_solve: dict[str, Any] | None = None
    schedule = _iteration_schedule(payload["f"], old_items)
    current_ordinal = 0
    old_ordinal = 0
    for iteration, (kind, old_item) in enumerate(schedule):
        knot = target if kind == "current" else old_item.knot
        fixed = FixedWordGame(scientist.game, knot, payload["ratio"])
        search = NeuralMCTS(
            fixed,
            scientist.network,
            replace(scientist.config.search, simulations=payload["simulations"]),
            payload["device"],
        )
        if kind == "current":
            ordinal = current_ordinal
            current_ordinal += 1
            seed_base = payload["seed"] + ordinal * 100_000
        else:
            ordinal = old_ordinal
            old_ordinal += 1
            seed_base = payload["seed"] + 50_000_000 + ordinal * 100_000
        seeds = [seed_base + game_index for game_index in range(payload["selfplay_games"])]
        phase = time.perf_counter()
        records = play_selfplay_games(
            fixed,
            search,
            [np.random.default_rng(seed + 7) for seed in seeds],
            seeds,
            12,
        )
        selfplay_seconds = time.perf_counter() - phase
        if payload.get("replay_sampler") == "v3":
            scientist.replay.set_representation_embedding(
                knot.name, _replay_representation_embedding(knot)
            )
        for record in records:
            scientist.replay.add(record, representation_id=knot.name)
            adaptation_compute += len(record) * (payload["simulations"] + 1)
            if kind == "current":
                verified = verified_record_cost(scientist.game, target, payload["ratio"], record)
                if verified is not None and (
                    best is None
                    or payload["ratio"] * verified[0] + verified[1]
                    < payload["ratio"] * best[0] + best[1]
                ):
                    best = verified
                if verified is not None and first_solve is None:
                    first_solve = {
                        "phase": "selfplay",
                        "iteration": iteration,
                        "current_ordinal": ordinal,
                        "before_any_training": iteration == 0,
                    }
        phase = time.perf_counter()
        losses = []
        requested_steps = int(payload["train_steps"])
        can_train = not payload.get("require_replay_success", False) or replay_has_native_success(
            scientist
        )
        if can_train:
            for _ in range(requested_steps):
                losses.append(
                    train_alphazero_step(
                        scientist.network,
                        scientist.optimizer,
                        scientist.replay,
                        payload["batch_size"],
                        torch.device(payload["device"]),
                        collaboration_replay=payload.get("success_balanced_replay", False),
                        shared_fraction=0.0,
                        policy_value_success_only=payload.get(
                            "policy_value_success_only", False
                        ),
                        replay_current_representation=(
                            target.name if payload.get("replay_sampler") == "v3" else ""
                        ),
                        replay_current_fraction=(
                            0.25 if payload.get("replay_sampler") == "v3" else 0.0
                        ),
                        replay_similar_fraction=(
                            0.25 if payload.get("replay_sampler") == "v3" else 0.0
                        ),
                        replay_positions_per_episode=(
                            4 if payload.get("replay_sampler") == "v3" else 1
                        ),
                    )
                )
        train_seconds = time.perf_counter() - phase
        training_seconds += train_seconds
        iteration_rows.append(
            {
                "iteration": iteration,
                "kind": kind,
                "representation": knot.name,
                "selfplay_games": len(records),
                "selfplay_solved": sum(
                    bool(record and record[0].solved > 0.5) for record in records
                ),
                "selfplay_seconds": selfplay_seconds,
                "training_seconds": train_seconds,
                "train_steps_requested": requested_steps,
                "train_steps_applied": len(losses),
                "train_steps_without_policy_value_targets": sum(
                    loss["policy_value_targets"] == 0.0 for loss in losses
                ),
                "train_skipped_no_success": bool(requested_steps and not can_train),
                "replay_had_success": replay_has_native_success(scientist),
                "last_loss": losses[-1] if losses else None,
            }
        )

    after_diagnostic = None
    if diagnostic_games:
        after_diagnostic, after_compute = _diagnostic_evaluation(
            scientist,
            target,
            payload["ratio"],
            payload["simulations"],
            diagnostic_seed,
            diagnostic_games,
        )
        diagnostic_compute += after_compute

    final_verified, final_compute = _evaluation_record(
        scientist,
        target,
        payload["ratio"],
        payload["simulations"],
        payload["seed"] + 900_000_000,
    )
    if final_verified is not None:
        candidate = (final_verified[0], final_verified[1], [])
        if best is None or (
            payload["ratio"] * candidate[0] + candidate[1] < payload["ratio"] * best[0] + best[1]
        ):
            best = candidate
        if first_solve is None:
            first_solve = {
                "phase": "final_evaluation",
                "iteration": None,
                "before_any_training": False,
            }
    retention = []
    retention_compute = 0
    for old_index, old_item in enumerate(old_items):
        verified, compute = _evaluation_record(
            scientist,
            old_item.knot,
            payload["ratio"],
            payload["simulations"],
            payload["seed"] + 950_000_000 + old_index,
        )
        retention_compute += compute["scheduled_network_evaluations"]
        row = {"item": old_item.id, "solved": verified is not None}
        if verified is not None:
            row.update(
                crossing_changes=verified[0],
                moves=verified[1],
                objective=payload["ratio"] * verified[0] + verified[1],
            )
        retention.append(row)
    result = {
        "index": payload["index"],
        "item": target.name,
        "f": payload["f"],
        "f_old": payload["f_old"],
        "replay_sampler": payload.get("replay_sampler", "ordinary"),
        "old_items": [item.id for item in old_items],
        "solved": best is not None,
        "best_crossing_changes": best[0] if best is not None else None,
        "best_moves": best[1] if best is not None else None,
        "best_objective": (payload["ratio"] * best[0] + best[1] if best is not None else None),
        "first_solve": first_solve,
        "diagnostic_evaluation": {
            "before": before_diagnostic,
            "after": after_diagnostic,
        },
        "old_retention": retention,
        "iterations": iteration_rows,
        "compute": {
            "adaptation_scheduled_network_evaluations": adaptation_compute,
            "final_scheduled_network_evaluations": final_compute["scheduled_network_evaluations"],
            "retention_scheduled_network_evaluations": retention_compute,
            "diagnostic_scheduled_network_evaluations": diagnostic_compute,
            "training_seconds": training_seconds,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    return result


def run_rapid_adaptation(
    checkpoint: Path,
    target_bank: Path,
    old_bank: Path,
    output: Path,
    *,
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    initial_f: int = 5,
    f_old: int = 1,
    threshold: float = 0.8,
    simulations: int = 128,
    selfplay_games: int = 8,
    train_steps: int = 96,
    batch_size: int = 32,
    limit: int = 200,
    workers: int = 6,
    seed: int = 20260840,
    device: str = "cpu",
    resume: bool = False,
    gate_games: int = 12,
    gate_min_solve_rate: float = 0.8,
) -> dict[str, Any]:
    if initial_f not in F_LEVELS:
        raise ValueError(f"initial_f must be one of {F_LEVELS}")
    checkpoint_gate = checkpoint_regression_gate(
        checkpoint,
        scientist,
        ratio=ratio,
        games=gate_games,
        minimum_solve_rate=gate_min_solve_rate,
        seed=seed + 700_000_000,
        simulations=simulations,
        device=device,
    )
    target_payload = json.loads(target_bank.read_text())[:limit]
    old_payload = json.loads(old_bank.read_text())
    targets = _bank_from_payload(target_payload)
    protocol = {
        "schema": "rapid-adaptation-v1",
        "scientist": scientist,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_gate": checkpoint_gate,
        "target_bank": str(target_bank.resolve()),
        "target_bank_sha256": _json_hash(target_payload),
        "old_bank": str(old_bank.resolve()),
        "old_bank_sha256": _json_hash(old_payload),
        "ratio": ratio,
        "initial_f": initial_f,
        "f_old": f_old,
        "threshold": threshold,
        "f_levels": list(F_LEVELS),
        "blocks": adaptation_blocks(len(targets)),
        "simulations": simulations,
        "selfplay_games": selfplay_games,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "limit": limit,
        "seed": seed,
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{manifest} exists; pass resume=True")
        if json.loads(manifest.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("resume protocol differs from frozen manifest")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, protocol)
    item_dir = output / "items"
    item_dir.mkdir(parents=True, exist_ok=True)

    current_f = initial_f
    block_reports = []
    for block_index, (start, stop) in enumerate(adaptation_blocks(len(targets))):
        if block_index:
            previous = block_reports[-1]
            current_f = next_f(current_f, previous["solve_rate"], threshold)
        pending = []
        for index in range(start, stop):
            path = item_dir / f"{index:04d}.json"
            if not path.exists():
                item = targets[index].knot
                pending.append(
                    {
                        "index": index,
                        "target": {
                            "name": item.name,
                            "crossings": item.crossings,
                            "word": item.word,
                            "strands": item.strands,
                        },
                        "old_bank": old_payload,
                        "scientist": scientist,
                        "checkpoint": str(checkpoint),
                        "ratio": ratio,
                        "f": current_f,
                        "f_old": f_old,
                        "simulations": simulations,
                        "selfplay_games": selfplay_games,
                        "train_steps": train_steps,
                        "batch_size": batch_size,
                        "seed": seed + index * 10_000_000,
                        "device": device,
                    }
                )
        if pending:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_run_one_task, payload) for payload in pending]
                for future in as_completed(futures):
                    row = future.result()
                    _atomic_json(item_dir / f"{row['index']:04d}.json", row)
        rows = [
            json.loads((item_dir / f"{index:04d}.json").read_text()) for index in range(start, stop)
        ]
        if any(row["f"] != current_f for row in rows):
            raise ValueError("completed block contains a different F schedule")
        solved = sum(bool(row["solved"]) for row in rows)
        block_reports.append(
            {
                "block": block_index,
                "start": start,
                "stop": stop,
                "f": current_f,
                "f_old": f_old,
                "solved": solved,
                "representations": len(rows),
                "solve_rate": solved / len(rows),
            }
        )
        _atomic_json(output / "blocks.json", block_reports)
    report = {
        **protocol,
        "completed": len(targets),
        "solved": sum(block["solved"] for block in block_reports),
        "solve_rate": sum(block["solved"] for block in block_reports) / len(targets),
        "blocks": block_reports,
    }
    _atomic_json(output / "report.json", report)
    return report
