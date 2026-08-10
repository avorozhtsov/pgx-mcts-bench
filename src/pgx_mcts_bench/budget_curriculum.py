"""Small, rollback-guarded curriculum for a collaboration-roster budget critic."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import (
    BUDGET_PROTOTYPES,
    FixedWordGame,
    KnotItem,
    calibrated_solve_probability,
    load_scientist,
    smallest_crossing_pool,
)
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _observation_tensor,
    _sha256,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import STAGES, _config, candidates, evaluate_stage
from pgx_mcts_bench.rapid_adaptation import promoted_checkpoint_metadata
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import (
    attach_policy_value_preservation_teacher,
    play_selfplay_games,
    train_alphazero_step,
)


def _budget_curve(scientist, items, ratio: float, caps: tuple[float, ...]) -> list[dict[str, Any]]:
    observations = [
        FixedWordGame(scientist.game, item, ratio, objective_cap=cap).reset(0).observation
        for item in items
        for cap in caps
    ]
    tensor = _observation_tensor(observations, torch.device(scientist.config.train.device))
    with torch.inference_mode():
        solve_logits = scientist.network.eval().forward_with_auxiliary(tensor)[2][0]
        probability = calibrated_solve_probability(scientist, solve_logits).reshape(
            len(items), len(caps)
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
    ratio: float,
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
        ratios=(ratio,),
    )[ratio]
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


def _early_rung_pool(scientist, items: int, seed: int) -> tuple[list[KnotItem], list[dict]]:
    """Freeze deterministic instances from the ladder's proven easy prefix.

    The rung-18 checkpoints learned by climbing this distribution.  Starting a
    budget critic on unrelated table knots confounds budget learning with a
    distribution jump and can yield no positive episodes at all.  Replaying the
    prefix tests the new channel without changing the old curriculum premise.
    """
    if scientist.game.generator is None:
        raise ValueError("early-rung curriculum requires the graded generator")
    selected: list[KnotItem] = []
    provenance: list[dict] = []
    for index in range(items):
        stage_index = index % len(STAGES)
        repeat = index // len(STAGES)
        source_name, scramble = STAGES[stage_index]
        source = next(
            source
            for source in scientist.game.generator.sources
            if source.name == source_name
        )
        instance_seed = seed + 90_000_000 + index * 100_003
        instance = scientist.game.generator.generate(
            source,
            scramble,
            np.random.default_rng(instance_seed),
        )
        name = f"rung-{stage_index}:{source_name}+{scramble}:sample-{repeat}"
        word = tuple(int(letter) for letter in instance.word)
        selected.append(KnotItem(name, len(word), word, int(instance.strands)))
        provenance.append(
            {
                "item": name,
                "stage": stage_index,
                "source": source_name,
                "scramble": scramble,
                "instance_seed": instance_seed,
                "word": list(word),
                "strands": int(instance.strands),
            }
        )
    return selected, provenance


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
    bank: Path | None = None,
    curriculum_source: str = "early-rungs",
    monotonic_weight: float | None = None,
    retention_stage: tuple[str, int] | None = None,
    minimum_mean_budget_spread: float = 0.01,
) -> dict[str, Any]:
    if scientist_name not in BUDGET_PROTOTYPES:
        raise ValueError(
            f"budget-critic curriculum is admitted only for {BUDGET_PROTOTYPES}, "
            f"not {scientist_name!r}"
        )
    if retention_stage is None:
        metadata = promoted_checkpoint_metadata(checkpoint, scientist_name)
        retention_anchor = "embedded-promoted-rung"
    else:
        source, scramble = retention_stage
        metadata = {"source": source, "scramble": int(scramble)}
        retention_anchor = "explicit-frozen-stage"
    scientist = load_scientist(
        scientist_name,
        checkpoint,
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    if monotonic_weight is not None:
        if monotonic_weight < 0.0:
            raise ValueError("monotonic_weight must be non-negative")
        scientist.network.auxiliary_budget_monotonic_weight = monotonic_weight
    attach_policy_value_preservation_teacher(scientist.network)
    if bank is not None and curriculum_source != "bank":
        raise ValueError("--bank requires curriculum_source='bank'")
    if curriculum_source == "early-rungs":
        simple, item_provenance = _early_rung_pool(scientist, items, seed)
    elif curriculum_source == "table":
        simple = smallest_crossing_pool(items)
        item_provenance = [
            {
                "item": item.name,
                "source": "knot-table",
                "word": list(item.word),
                "strands": item.strands,
            }
            for item in simple
        ]
    elif curriculum_source == "bank" and bank is not None:
        import json

        simple = [item.knot for item in _bank_from_payload(json.loads(bank.read_text()))]
        if len(simple) != items:
            raise ValueError(f"{bank} contains {len(simple)} identities, expected {items}")
        item_provenance = [
            {
                "item": item.name,
                "source": "certified-bank",
                "word": list(item.word),
                "strands": item.strands,
            }
            for item in simple
        ]
    else:
        raise ValueError(
            "curriculum_source must be one of 'early-rungs', 'table', or 'bank'"
        )
    global_cap = (ratio + 1.0) * scientist.config.game.simplify_budget
    caps = tuple(max(1.0, round(global_cap * fraction)) for fraction in cap_fractions)
    before_state = copy.deepcopy(scientist.network.state_dict())
    before_curve = _budget_curve(scientist, simple, ratio, caps)
    before_rung = _rung_evaluation(
        scientist,
        metadata,
        rung_eval_games,
        seed + 800_000_000,
        ratio,
    )
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
    after_rung = _rung_evaluation(
        scientist,
        metadata,
        rung_eval_games,
        seed + 800_000_000,
        ratio,
    )
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
    mean_spread = float(np.mean([row["spread"] for row in after_curve]))
    accepted = (
        after_rung["solve_rate"] >= before_rung["solve_rate"]
        and bool(informative)
        and informative_sensitive == len(informative)
        and monotone == len(simple)
        and mean_spread >= minimum_mean_budget_spread
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
        "schema": "semantic-cost-collaboration-budget-curriculum-v3",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "scientist": scientist_name,
        "architecture": {
            "solve_encoder_gradients": True,
            "cost_first": True,
            "manual_objective": "L=A*cc+B*portable_semantic_moves",
            "native_plies_excluded_from_objective": True,
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
            "bank": str(bank.resolve()) if bank is not None else None,
            "curriculum_source": curriculum_source,
            "item_provenance": item_provenance,
            "monotonic_weight": float(
                scientist.network.auxiliary_budget_monotonic_weight
            ),
            "minimum_mean_budget_spread": minimum_mean_budget_spread,
            "retention_anchor": {
                "kind": retention_anchor,
                "source": metadata["source"],
                "scramble": metadata["scramble"],
            },
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
            "mean_budget_spread": mean_spread,
            "rollback_applied": not accepted,
            "next": (
                "expand the easy curriculum" if accepted else "restart from the earliest rung"
            ),
        },
        "model_config": asdict(scientist.config.model),
    }
    _atomic_json(output / "report.json", report)
    return report
