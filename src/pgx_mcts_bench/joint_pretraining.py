"""Joint simple-knot pretraining for the budget-aware window scientist.

This is deliberately a curriculum runner, not another collaboration arm.  It
starts from the ladder distribution that made ``s-window-128`` work, trains the
native policy/value and factorized ``cc``/``moves``/``p(solve)`` heads together,
and then presents the same easy representations under several objective caps.
The old rung checkpoint is a warm start; a random initialization is retained as
the matched control.  A remaining-internal-budget variant is a separate arm so
the extra legality rule is never silently folded into the primary result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import FixedWordGame, KnotItem
from pgx_mcts_bench.data import ReplayBuffer
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import (
    STAGES,
    Candidate,
    _config,
    evaluate_stage,
    run_ladder,
    serial_arms,
)
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games, train_alphazero_step


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def joint_pretraining_candidates(
    *,
    simulations: int = 128,
    train_steps: int = 96,
) -> dict[str, Candidate]:
    """Return the primary, random-control and internal-budget architectures."""
    historical = next(candidate for candidate in serial_arms() if candidate.name == "s-window-128")
    common = {
        **asdict(historical),
        "simulations": simulations,
        "train_steps": train_steps,
        "objective_budget_channel": True,
        # cc/moves heads learn from the shared representation but their noisy
        # regression loss does not rewrite it.  Solve BCE is the admitted
        # auxiliary signal that reaches the encoder/body.
        "auxiliary_backprop_to_encoder": False,
        "auxiliary_solve_backprop_to_encoder": True,
        # Joint cc/moves gradients make the body less stationary than in the
        # earlier critic-only calibration. Weight 1.0 is the first tested dose
        # that made every easy-prefix curve monotone.
        "auxiliary_budget_monotonic_weight": 1.0,
        "auxiliary_budget_conditioning": True,
        # Keep the historical scalar in MCTS until the factorized critic passes
        # calibration.  All auxiliary heads are nevertheless trained jointly.
        "use_auxiliary_value": False,
    }
    return {
        "warm": Candidate(
            **{
                **common,
                "name": "s-window-128-joint-warm-v2",
                "rationale": (
                    "rung-18 warm start; joint policy/value/cost/solve curriculum "
                    "with remaining semantic L"
                ),
            }
        ),
        "scratch": Candidate(
            **{
                **common,
                "name": "s-window-128-joint-scratch-v2",
                "rationale": ("random-init control for the joint remaining-semantic-L curriculum"),
            }
        ),
        "h5": Candidate(
            **{
                **common,
                "name": "s-window-128-joint-h5-v2",
                "rationale": (
                    "warm-start ablation with remaining semantic L and a five-step "
                    "remaining internal-action budget"
                ),
                "serial_internal_horizon": 5,
                "serial_internal_budget_remaining": True,
            }
        ),
    }


def _network_from_checkpoint(
    candidate: Candidate,
    checkpoint: Path,
    *,
    seed: int,
    device: str,
):
    config = _config(candidate, STAGES[0], seed, device, selfplay_games=1)
    network = make_braid_network(config.game, config.model).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("network", payload) if isinstance(payload, dict) else payload
    migrated = load_policy_value_state_dict(network, state)
    return config, network, migrated


@torch.inference_mode()
def migration_equivalence(
    source_checkpoint: Path,
    target: Candidate,
    *,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Prove that appended budget planes do not change initial predictions."""
    source = next(candidate for candidate in serial_arms() if candidate.name == "s-window-128")
    source_config, source_network, _ = _network_from_checkpoint(
        source, source_checkpoint, seed=seed, device=device
    )
    target_config, target_network, migrated = _network_from_checkpoint(
        target, source_checkpoint, seed=seed, device=device
    )
    source_game = make_game(source_config.game)
    target_game = make_game(target_config.game)
    source_network.eval()
    target_network.eval()
    maxima = {"policy": 0.0, "value": 0.0, "solve": 0.0, "cc": 0.0, "moves": 0.0}
    rows = []
    for index, stage_index in enumerate((0, 3, 5)):
        source_name, scramble = STAGES[stage_index]
        generator_source = next(
            value for value in source_game.generator.sources if value.name == source_name
        )
        instance_seed = seed + 81_000_000 + index * 100_003
        instance = source_game.generator.generate(
            generator_source, scramble, np.random.default_rng(instance_seed)
        )
        for ratio in (10.0, 1000.0):
            source_transition = source_game.from_word(
                list(instance.word), instance.strands, float(np.log(ratio))
            )
            target_transition = target_game.from_word(
                list(instance.word), instance.strands, float(np.log(ratio))
            )
            source_observation = (
                torch.from_numpy(source_transition.observation)
                .permute(2, 0, 1)[None]
                .float()
                .to(device)
            )
            target_observation = (
                torch.from_numpy(target_transition.observation)
                .permute(2, 0, 1)[None]
                .float()
                .to(device)
            )
            source_outputs = source_network.forward_with_auxiliary(source_observation)
            target_outputs = target_network.forward_with_auxiliary(target_observation)
            named = (
                ("policy", source_outputs[0], target_outputs[0]),
                ("value", source_outputs[1], target_outputs[1]),
                ("solve", source_outputs[2][0], target_outputs[2][0]),
                ("cc", source_outputs[2][1], target_outputs[2][1]),
                ("moves", source_outputs[2][2], target_outputs[2][2]),
            )
            differences = {}
            for name, before, after in named:
                difference = float((before - after).abs().max().cpu())
                differences[name] = difference
                maxima[name] = max(maxima[name], difference)
            rows.append(
                {
                    "stage": stage_index,
                    "source": source_name,
                    "scramble": scramble,
                    "ratio": ratio,
                    "max_absolute_difference": differences,
                }
            )
    return {
        "migrated": bool(migrated),
        "source_observation_channels": source_config.game.observation_channels,
        "target_observation_channels": target_config.game.observation_channels,
        "max_absolute_difference": maxima,
        "bit_exact": all(value == 0.0 for value in maxima.values()),
        "rows": rows,
    }


def _early_items(
    candidate: Candidate,
    items: int,
    seed: int,
    device: str,
    *,
    split: str,
) -> list[KnotItem]:
    config = _config(candidate, STAGES[0], seed, device, selfplay_games=1)
    game = make_game(config.game)
    selected = []
    for index in range(items):
        stage_index = index % min(10, len(STAGES))
        source_name, scramble = STAGES[stage_index]
        source = next(value for value in game.generator.sources if value.name == source_name)
        item_seed = seed + 91_000_000 + index * 100_003
        instance = game.generator.generate(source, scramble, np.random.default_rng(item_seed))
        selected.append(
            KnotItem(
                f"{split}:rung-{stage_index}:{source_name}+{scramble}:sample-{index // 10}",
                len(instance.word),
                tuple(int(letter) for letter in instance.word),
                int(instance.strands),
            )
        )
    return selected


@torch.inference_mode()
def _budget_curve(network, game, items, ratio: float, caps, device: str) -> list[dict]:
    network.eval()
    rows = []
    for item in items:
        probabilities = []
        for cap in caps:
            observation = FixedWordGame(game, item, ratio, objective_cap=cap).reset(0).observation
            tensor = torch.from_numpy(observation).permute(2, 0, 1)[None].float().to(device)
            solve_logits = network.forward_with_auxiliary(tensor)[2][0]
            probabilities.append(float(solve_logits.sigmoid().mean().cpu()))
        rows.append(
            {
                "item": item.name,
                "p_solve": probabilities,
                "spread": max(probabilities) - min(probabilities),
                "monotone": all(
                    high + 1e-7 >= low
                    for low, high in zip(probabilities, probabilities[1:], strict=False)
                ),
            }
        )
    return rows


def _retention_panel(
    candidate: Candidate,
    network,
    checkpoint_payload: dict[str, Any],
    *,
    games: int,
    seed: int,
    device: str,
    ratios: tuple[float, ...] = (10.0, 1000.0),
) -> dict[str, Any]:
    """Measure the mastered prefix under paired seeds and a capped objective."""
    rows = []
    for stage_row in checkpoint_payload.get("stages", []):
        if not bool(stage_row.get("promoted", False)):
            continue
        stage_index = int(stage_row["stage"])
        stage = (str(stage_row["source"]), int(stage_row["scramble"]))
        config = _config(candidate, stage, seed, device, selfplay_games=1)
        game = make_game(config.game)
        measured = evaluate_stage(
            game,
            network,
            config,
            games,
            seed + stage_index * 100_003,
            ratios=ratios,
        )
        for ratio, result in measured.items():
            solve_rate = float(result["solved"])
            cap = (float(ratio) + 1.0) * config.game.simplify_budget
            crossings = float(result["crossings"])
            moves = float(result["moves"])
            solved_cost = (
                float(ratio) * crossings + moves
                if solve_rate > 0.0 and math.isfinite(crossings) and math.isfinite(moves)
                else 0.0
            )
            rows.append(
                {
                    "stage": stage_index,
                    "source": stage[0],
                    "scramble": stage[1],
                    "ratio": float(ratio),
                    "solve_rate": solve_rate,
                    "crossings": crossings,
                    "moves": moves,
                    "capped_loss_per_attempt": (
                        solve_rate * solved_cost + (1.0 - solve_rate) * cap
                    ),
                }
            )
    return {
        "games_per_stage_ratio": games,
        "rows": rows,
        "solved_attempts": sum(row["solve_rate"] * games for row in rows),
        "attempts": len(rows) * games,
        "capped_loss": sum(row["capped_loss_per_attempt"] * games for row in rows),
        "minimum_solve_rate": min((row["solve_rate"] for row in rows), default=float("nan")),
    }


def _portfolio_retention_decision(before: dict[str, Any], after: dict[str, Any]) -> dict[str, bool]:
    """Keep aggregate progress primary and exact cell retention diagnostic."""
    return {
        "portfolio_noninferior": (
            after["solved_attempts"] >= before["solved_attempts"]
            and after["capped_loss"] <= before["capped_loss"] + 1e-6
        ),
        "exact_cell_retention_secondary": (not after["rows"] or after["minimum_solve_rate"] >= 0.8),
    }


def run_joint_budget_calibration(
    candidate: Candidate,
    checkpoint: Path,
    output: Path,
    *,
    items: int = 10,
    cap_fractions: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 1.0),
    games_per_cap: int = 2,
    train_steps_per_item: int = 32,
    simulations: int = 64,
    retention_games: int = 4,
    ratios: tuple[float, ...] = (10.0, 1000.0),
    seed: int = 20261800,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fine-tune every native and auxiliary head on varied objective caps."""
    candidate = replace(candidate, simulations=simulations)
    config, network, _ = _network_from_checkpoint(candidate, checkpoint, seed=seed, device=device)
    before_state = copy.deepcopy(network.state_dict())
    game = make_game(config.game)
    optimizer = torch.optim.AdamW(network.parameters(), lr=2.5e-4, weight_decay=1e-4)
    replay = ReplayBuffer(20_000, np.random.default_rng(seed))
    pool = _early_items(candidate, items, seed, device, split="train")
    validation_pool = _early_items(
        candidate,
        items,
        seed + 37_000_003,
        device,
        split="heldout",
    )
    caps_by_ratio = {
        ratio: tuple(
            max(
                1.0,
                round((ratio + 1.0) * config.game.simplify_budget * fraction),
            )
            for fraction in cap_fractions
        )
        for ratio in ratios
    }

    def curves(items_to_measure) -> list[dict[str, Any]]:
        return [
            {**row, "ratio": ratio}
            for ratio in ratios
            for row in _budget_curve(
                network,
                game,
                items_to_measure,
                ratio,
                caps_by_ratio[ratio],
                device,
            )
        ]

    before = curves(pool)
    before_heldout = curves(validation_pool)
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    before_retention = _retention_panel(
        candidate,
        network,
        source_payload,
        games=retention_games,
        seed=seed + 61_000_000,
        device=device,
    )
    rows = []
    last_metrics: dict[str, float] | None = None
    for item_index, item in enumerate(pool):
        ratio_rows = []
        for ratio_index, ratio in enumerate(ratios):
            cap_rows = []
            for cap_index, cap in enumerate(caps_by_ratio[ratio]):
                fixed = FixedWordGame(game, item, ratio, objective_cap=cap)
                search_config = replace(config.search, simulations=simulations)
                search = NeuralMCTS(fixed, network, search_config, device)
                seeds = [
                    seed
                    + item_index * 10_000_000
                    + ratio_index * 1_000_000
                    + cap_index * 10_000
                    + game_index
                    for game_index in range(games_per_cap)
                ]
                records = play_selfplay_games(
                    fixed,
                    search,
                    [np.random.default_rng(value + 7) for value in seeds],
                    seeds,
                    config.train.temperature_moves,
                )
                for record in records:
                    replay.add(record)
                cap_rows.append(
                    {
                        "cap": cap,
                        "attempts": len(records),
                        "solved": sum(
                            bool(record and record[0].solved > 0.5) for record in records
                        ),
                        "budget_censored": sum(
                            bool(record and record[0].objective_censored) for record in records
                        ),
                    }
                )
            ratio_rows.append({"ratio": ratio, "caps": cap_rows})
        for _ in range(train_steps_per_item):
            last_metrics = train_alphazero_step(
                network,
                optimizer,
                replay,
                32,
                torch.device(device),
                collaboration_replay=True,
                shared_fraction=0.0,
                replay_positions_per_episode=4,
            )
        rows.append({"item": item.name, "ratios": ratio_rows})
    after = curves(pool)
    after_heldout = curves(validation_pool)
    after_retention = _retention_panel(
        candidate,
        network,
        source_payload,
        games=retention_games,
        seed=seed + 61_000_000,
        device=device,
    )
    informative = {
        f"{row['item']}@{ratio_row['ratio']:g}"
        for row in rows
        for ratio_row in row["ratios"]
        if 0
        < sum(cap["solved"] for cap in ratio_row["caps"])
        < sum(cap["attempts"] for cap in ratio_row["caps"])
    }
    sensitive = {f"{row['item']}@{row['ratio']:g}" for row in after if float(row["spread"]) > 1e-4}
    retention_decision = _portfolio_retention_decision(before_retention, after_retention)
    accepted = (
        bool(informative)
        and informative <= sensitive
        and all(bool(row["monotone"]) for row in after)
        and all(bool(row["monotone"]) for row in after_heldout)
        and retention_decision["portfolio_noninferior"]
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **source_payload,
            "network": network.state_dict(),
            "joint_budget_calibration": {
                "accepted": accepted,
                "source": str(checkpoint.resolve()),
            },
        },
        output / "trained-candidate.pt",
    )
    torch.save(
        {
            **source_payload,
            "network": network.state_dict() if accepted else before_state,
            "joint_budget_calibration": {
                "accepted": accepted,
                "source": str(checkpoint.resolve()),
                "rollback_applied": not accepted,
            },
        },
        output / "joint-pretrained.pt",
    )
    report = {
        "schema": "joint-simple-knot-budget-pretraining-v1",
        "candidate": asdict(candidate),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "protocol": {
            "items": items,
            "item_names": [item.name for item in pool],
            "heldout_item_names": [item.name for item in validation_pool],
            "ratios": list(ratios),
            "caps_by_ratio": {str(ratio): list(caps) for ratio, caps in caps_by_ratio.items()},
            "games_per_cap": games_per_cap,
            "train_steps_per_item": train_steps_per_item,
            "simulations": simulations,
            "retention_games_per_stage_ratio": retention_games,
            "seed": seed,
            "device": device,
            "policy_value_rule": (
                "ordinary native outcomes train policy/value; budget-censored failures "
                "train p(solve) only"
            ),
            "replay_rule": "50:50 success/failure when both strata exist",
        },
        "training": rows,
        "last_metrics": last_metrics,
        "before_curve": before,
        "after_curve": after,
        "before_heldout_curve": before_heldout,
        "after_heldout_curve": after_heldout,
        "before_retention": before_retention,
        "after_retention": after_retention,
        "decision": {
            "accepted": accepted,
            "informative_items": sorted(informative),
            "sensitive_items": sorted(sensitive),
            "monotone_items": sum(bool(row["monotone"]) for row in after),
            "monotone_heldout_items": sum(bool(row["monotone"]) for row in after_heldout),
            **retention_decision,
            "rollback_applied": not accepted,
        },
    }
    _atomic_json(output / "report.json", report)
    return report


def _ladder_admitted(result, stop_after: int, minimum_solve_rate: float) -> bool:
    by_stage = {stage.stage: stage for stage in result.stages}
    return all(
        index in by_stage
        and by_stage[index].promoted
        and by_stage[index].solve_rate >= minimum_solve_rate
        for index in range(stop_after + 1)
    )


def run_joint_pretraining(
    source_checkpoint: Path,
    output: Path,
    *,
    smoke_stop_after: int = 9,
    continue_stop_after: int = 9,
    include_warm: bool = True,
    include_scratch: bool = True,
    include_h5: bool = True,
    allow_warm_rewind_ablation: bool = False,
    max_iterations_per_stage: int = 100,
    selfplay_games: int = 8,
    eval_games: int = 12,
    simulations: int = 128,
    train_steps: int = 96,
    rehearsal_games_per_cleared_stage: int = 1,
    minimum_solve_rate: float = 0.8,
    budget_items: int = 10,
    budget_games_per_cap: int = 2,
    budget_train_steps_per_item: int = 32,
    budget_simulations: int = 64,
    seed: int = 20261780,
    device: str = "cpu",
    log=print,
) -> dict[str, Any]:
    """Run migration, matched early-rung gates, and admitted warm continuation."""
    if smoke_stop_after < 0 or continue_stop_after < smoke_stop_after:
        raise ValueError("continuation must include the non-negative smoke prefix")
    if rehearsal_games_per_cleared_stage < 0:
        raise ValueError("rehearsal games per cleared stage must be non-negative")
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_stage_result = (
        source_payload.get("stage_result", {}) if isinstance(source_payload, dict) else {}
    )
    source_stage = int(source_stage_result.get("stage", -1))
    if (
        source_stage > smoke_stop_after
        and (include_warm or include_h5)
        and not allow_warm_rewind_ablation
    ):
        raise ValueError(
            f"refusing to rewind a rung-{source_stage} warm checkpoint to rung "
            f"{smoke_stop_after}; use the guarded budget curriculum for warm "
            "fine-tuning, or run the joint architecture forward from random "
            "initialization. Pass allow_warm_rewind_ablation only to reproduce "
            "the catastrophic-forgetting ablation."
        )
    output.mkdir(parents=True, exist_ok=True)
    candidates = joint_pretraining_candidates(simulations=simulations, train_steps=train_steps)
    equivalence = {
        name: migration_equivalence(source_checkpoint, candidates[name], seed=seed, device=device)
        for name in ("warm", "h5")
        if (name == "warm" and include_warm) or (name == "h5" and include_h5)
    }
    if not all(row["bit_exact"] for row in equivalence.values()):
        raise RuntimeError("function-preserving budget migration failed")

    selected = []
    if include_warm:
        selected.append("warm")
    if include_scratch:
        selected.append("scratch")
    if include_h5:
        selected.append("h5")
    if not selected:
        raise ValueError("at least one joint-pretraining arm must be selected")
    primary_name = "warm" if include_warm else "scratch" if include_scratch else "h5"
    ladder_results = {}
    checkpoint_root = output / "ladder-checkpoints"
    for arm_index, name in enumerate(selected):
        candidate = candidates[name]
        log(f"[{name}] starting joint-pretraining smoke through rung {smoke_stop_after}")
        result = run_ladder(
            candidate,
            seed=seed + arm_index * 10_000,
            device=device,
            checkpoint_dir=checkpoint_root,
            max_iterations_per_stage=max_iterations_per_stage,
            selfplay_games=selfplay_games,
            eval_games=eval_games,
            promote_at=minimum_solve_rate,
            stop_after=smoke_stop_after,
            retro_games=4,
            balanced_replay=False,
            rehearsal_games_per_cleared_stage=(
                rehearsal_games_per_cleared_stage if name == "scratch" else 0
            ),
            initial_checkpoint=(None if name == "scratch" else source_checkpoint),
            log=log,
        )
        ladder_results[name] = result
    primary_smoke_admitted = _ladder_admitted(
        ladder_results[primary_name], smoke_stop_after, minimum_solve_rate
    )
    if primary_smoke_admitted and continue_stop_after > smoke_stop_after:
        log(f"[{primary_name}] smoke admitted; continuing through rung {continue_stop_after}")
        ladder_results[primary_name] = run_ladder(
            candidates[primary_name],
            seed=seed + selected.index(primary_name) * 10_000,
            device=device,
            checkpoint_dir=checkpoint_root,
            max_iterations_per_stage=max_iterations_per_stage,
            selfplay_games=selfplay_games,
            eval_games=eval_games,
            promote_at=minimum_solve_rate,
            stop_after=continue_stop_after,
            retro_games=4,
            balanced_replay=False,
            rehearsal_games_per_cleared_stage=(
                rehearsal_games_per_cleared_stage if primary_name == "scratch" else 0
            ),
            initial_checkpoint=(None if primary_name == "scratch" else source_checkpoint),
            log=log,
        )
    primary_curriculum_admitted = _ladder_admitted(
        ladder_results[primary_name], continue_stop_after, minimum_solve_rate
    )

    budget_report = None
    primary_checkpoint = checkpoint_root / f"{candidates[primary_name].name}.pt"
    if primary_curriculum_admitted and primary_checkpoint.is_file():
        budget_report = run_joint_budget_calibration(
            candidates[primary_name],
            primary_checkpoint,
            output / "budget-calibration",
            items=budget_items,
            games_per_cap=budget_games_per_cap,
            train_steps_per_item=budget_train_steps_per_item,
            simulations=budget_simulations,
            seed=seed + 50_000,
            device=device,
        )

    report = {
        "schema": "joint-simple-knot-pretraining-gate-v1",
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "protocol": {
            "smoke_stop_after": smoke_stop_after,
            "continue_stop_after": continue_stop_after,
            "max_iterations_per_stage": max_iterations_per_stage,
            "selfplay_games": selfplay_games,
            "eval_games_per_ratio": eval_games,
            "simulations": simulations,
            "train_steps_per_iteration": train_steps,
            "minimum_solve_rate": minimum_solve_rate,
            "source_checkpoint_stage": source_stage,
            "warm_rewind_ablation": (source_stage > smoke_stop_after and primary_name != "scratch"),
            "primary_arm": primary_name,
            "ladder_replay": "historical natural episode distribution",
            "budget_replay": "50:50 success/failure when both strata exist",
            "scratch_explicit_rehearsal_games_per_cleared_stage": (
                rehearsal_games_per_cleared_stage
            ),
            "ratios": [1000.0, 10.0, 0.1],
            "seed": seed,
            "device": device,
        },
        "architectures": {name: asdict(candidates[name]) for name in selected},
        "migration_equivalence": equivalence,
        "ladder": {
            name: {
                "highest_stage": result.highest_stage,
                "seconds": result.seconds,
                "stages": [asdict(stage) for stage in result.stages],
                "admitted_on_smoke_prefix": _ladder_admitted(
                    result, smoke_stop_after, minimum_solve_rate
                ),
            }
            for name, result in ladder_results.items()
        },
        "budget_calibration": budget_report,
        "decision": {
            "primary_arm": primary_name,
            "primary_smoke_admitted": primary_smoke_admitted,
            "primary_curriculum_admitted": primary_curriculum_admitted,
            "warm_smoke_admitted": (
                _ladder_admitted(ladder_results["warm"], smoke_stop_after, minimum_solve_rate)
                if "warm" in ladder_results
                else None
            ),
            "internal_budget_calibration_passed": (
                bool(budget_report and budget_report["decision"]["accepted"])
            ),
            "candidate_checkpoint": (
                str((output / "budget-calibration" / "joint-pretrained.pt").resolve())
                if budget_report is not None and budget_report["decision"]["accepted"]
                else None
            ),
            "joint_pretrained_checkpoint": None,
            "source_disjoint_gate_required": True,
            "capacity_expansion_allowed": False,
            "next": (
                "run the untouched source-disjoint solver and critic gate"
                if primary_curriculum_admitted
                and budget_report is not None
                and budget_report["decision"]["accepted"]
                else "recover aggregate solved attempts and capped objective before continuation"
            ),
        },
    }
    _atomic_json(output / "report.json", report)
    return report
