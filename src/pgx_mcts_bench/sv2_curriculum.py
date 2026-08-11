"""Semantic-v2 fixed-representation curriculum and first static paper arm.

The old readiness gate asked frozen foundation checkpoints to transfer to table
knots.  SV2 instead gives every representation a declared task-local learning
dose.  One rung is one fixed braid representation, not a source distribution.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import multiprocessing
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import FixedWordGame, KnotItem, load_scientist
from pgx_mcts_bench.collaboration_eval import _evaluation_record
from pgx_mcts_bench.collaborative_scientists import (
    _bank_from_payload,
    _json_hash,
    _replay_representation_embedding,
    _restore_scientist,
    _scientist_state,
    _sha256,
    expected_capped_scores,
    translate_semantic_record,
    verified_record_cost,
)
from pgx_mcts_bench.data import GameRecord, ReplayBuffer
from pgx_mcts_bench.foundation_pretraining import source_provenance
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, foundation_arms
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games, train_alphazero_step

SV2_PREFIX_PHASES: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        0,
        ("unknot", "T(2,3)", "P(3,4)#0", "T(2,5)", "P(4,5)#0", "P(4,7)#0"),
    ),
    (
        2,
        ("unknot", "T(2,3)", "P(3,4)#0", "T(2,5)", "P(4,5)#0", "P(4,7)#0"),
    ),
    (1, ("P(3,4)#1", "P(3,4)#2", "P(4,5)#1")),
    (2, ("P(4,5)#2", "P(3,6)#0", "P(3,6)#1")),
    (3, ("P(3,6)#2", "P(4,7)#1", "P(4,7)#2")),
    (4, ("P(3,8)#0", "P(3,8)#1", "P(3,8)#2")),
)

F_OLD_LEVELS = (1, 2, 4, 8)
DONATION_DOSES = (1, 2, 3)
COORDINATED_ARMS = (
    "adaptive-no-sharing",
    "static-sharing",
    "adaptive-sharing",
)
CoordinatedArm = Literal[
    "adaptive-no-sharing",
    "static-sharing",
    "adaptive-sharing",
]


@dataclass(frozen=True)
class DonationDoseDecision:
    dose: int
    healthy_streak: int
    reason: str


def adapt_donation_dose(
    current: int,
    *,
    healthy_streak: int,
    eligible_donations: int,
    donation_only_regression: bool,
    portfolio_noninferior: bool,
    minimum_evidence: int = 10,
) -> DonationDoseDecision:
    """Adapt exact donor-witness replay exposures after a paired block.

    An upward step needs two healthy blocks and enough distinct eligible
    donations.  A regression lowers the dose immediately.  This controller is
    deliberately independent of canonical-route loss.
    """
    if current not in DONATION_DOSES:
        raise ValueError(f"donation dose must be one of {DONATION_DOSES}")
    if healthy_streak < 0 or eligible_donations < 0 or minimum_evidence < 1:
        raise ValueError("donation evidence counts must be non-negative")
    index = DONATION_DOSES.index(current)
    if donation_only_regression or not portfolio_noninferior:
        return DonationDoseDecision(
            DONATION_DOSES[max(0, index - 1)], 0, "lower after paired regression"
        )
    if eligible_donations < minimum_evidence:
        return DonationDoseDecision(current, 0, "hold: insufficient eligible donations")
    streak = healthy_streak + 1
    if streak < 2 or index == len(DONATION_DOSES) - 1:
        return DonationDoseDecision(current, streak, "hold: accumulating healthy evidence")
    return DonationDoseDecision(DONATION_DOSES[index + 1], 0, "raise after two healthy blocks")


def auditable_complexity(
    *, strands: int, unknotting_number: int, word_length: int
) -> float:
    """Outcome-blind ACS; scramble is already represented inside the word."""
    if strands < 1 or unknotting_number < 0 or word_length < 0:
        raise ValueError("ACS inputs must be non-negative and strands positive")
    return float(10 * strands + 5 * unknotting_number + word_length)


def _source_game():
    candidate = foundation_arms()[0]
    config = _config(candidate, ("unknot", 0), 0, "cpu", selfplay_games=1)
    return make_game(config.game)


def build_prefix24(*, seed: int = 20262000) -> list[dict[str, Any]]:
    """Build the fixed 6+6+3+3+3+3 reception-class curriculum."""
    game = _source_game()
    sources = {source.name: source for source in game.generator.sources}
    generated: list[dict[str, Any]] = []
    for phase, (scramble, names) in enumerate(SV2_PREFIX_PHASES):
        for source_name in names:
            source = sources[source_name]
            identity_seed = int.from_bytes(
                hashlib.sha256(
                    f"sv2-prefix24:{seed}:{phase}:{source_name}:{scramble}".encode()
                ).digest()[:8],
                "big",
            )
            instance = game.generator.generate(
                source, scramble, np.random.default_rng(identity_seed)
            )
            word = tuple(int(letter) for letter in instance.word)
            strands = int(instance.strands)
            known_u = int(source.unknotting_number)
            if known_u < 0:
                raise ValueError(f"SV2 prefix source lacks exact u: {source_name}")
            score = auditable_complexity(
                strands=strands,
                unknotting_number=known_u,
                word_length=len(word),
            )
            generated.append(
                {
                    "source": source_name,
                    "scramble_moves": scramble,
                    "phase": phase,
                    "generation_seed": identity_seed,
                    "known_unknotting_number": known_u,
                    "source_word_length": len(source.word),
                    "presentation_crossings": len(word),
                    "strands": strands,
                    "word": list(word),
                    "acs": score,
                }
            )

    output = []
    for index, row in enumerate(
        sorted(
            generated,
            key=lambda value: (
                value["acs"],
                value["strands"],
                value["known_unknotting_number"],
                value["presentation_crossings"],
                value["source"],
                value["phase"],
            ),
        )
    ):
        safe_source = (
            str(row["source"])
            .replace("(", "_")
            .replace(")", "")
            .replace(",", "_")
            .replace("#", "_")
        )
        identity = f"sv2-r24-{index:02d}-{safe_source}-s{row['scramble_moves']}"
        output.append(
            {
                "id": identity,
                "name": identity,
                "crossings": row["presentation_crossings"],
                "cheap_score": float(row["acs"]),
                "difficulty_quartile": int(row["phase"]),
                **row,
            }
        )
    if len(output) != 24 or len({row["id"] for row in output}) != 24:
        raise AssertionError("SV2 prefix must contain 24 distinct representations")
    return output


def write_prefix24(path: Path, *, seed: int = 20262000) -> dict[str, Any]:
    rows = build_prefix24(seed=seed)
    payload = {
        "schema": "semantic-v2-prefix24-v1",
        "seed": seed,
        "ordering": "global ACS=10*strands+5*u+presentation_crossings",
        "presentation_crossings": "braid word length, not minimal knot crossing number",
        "rows": rows,
    }
    _atomic_json(path, payload)
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _bank_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("SV2 bank must contain a non-empty row list")
    return payload if isinstance(payload, dict) else {"rows": rows}, rows


def _fixed_items(rows: list[dict[str, Any]]) -> list[KnotItem]:
    bank_rows = [
        {
            **row,
            "cheap_score": float(row.get("cheap_score", row.get("acs", 0.0))),
            "difficulty_quartile": int(row.get("difficulty_quartile", row.get("phase", 0))),
        }
        for row in rows
    ]
    return [item.knot for item in _bank_from_payload(bank_rows)]


def next_rehearsal_dose(
    current: int,
    *,
    retention_solve_rate: float,
    capped_cost_worsened: bool,
    target: float = 0.80,
) -> int:
    """Raise only F_old; the R24 native and search doses stay fixed."""
    if current not in F_OLD_LEVELS:
        raise ValueError(f"F_old must be one of {F_OLD_LEVELS}")
    if retention_solve_rate >= target and not capped_cost_worsened:
        return current
    index = F_OLD_LEVELS.index(current)
    return F_OLD_LEVELS[min(index + 1, len(F_OLD_LEVELS) - 1)]


def _iteration(
    scientist: Any,
    knot: KnotItem,
    *,
    ratios: tuple[float, ...],
    simulations: int,
    selfplay_games: int,
    train_steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    scientist.replay.set_representation_embedding(
        knot.name, _replay_representation_embedding(knot)
    )
    records = []
    scheduled = 0
    per_ratio = selfplay_games // len(ratios)
    remainder = selfplay_games % len(ratios)
    for ratio_index, ratio in enumerate(ratios):
        games = per_ratio + int(ratio_index < remainder)
        if not games:
            continue
        fixed = FixedWordGame(scientist.game, knot, ratio)
        search = NeuralMCTS(
            fixed,
            scientist.network,
            replace(scientist.config.search, simulations=simulations),
            scientist.config.train.device,
        )
        seeds = [seed + ratio_index * 10_000 + index for index in range(games)]
        batch = play_selfplay_games(
            fixed,
            search,
            [np.random.default_rng(value + 7) for value in seeds],
            seeds,
            12,
        )
        records.extend(batch)
        for record in batch:
            scientist.replay.add(
                record, representation_id=knot.name, objective_ratio=ratio
            )
            scheduled += len(record) * (simulations + 1)

    losses = []
    if any(scientist.replay.games):
        for _ in range(train_steps):
            losses.append(
                train_alphazero_step(
                    scientist.network,
                    scientist.optimizer,
                    scientist.replay,
                    batch_size,
                    torch.device(scientist.config.train.device),
                    collaboration_replay=True,
                    shared_fraction=0.0,
                    policy_value_success_only=True,
                    replay_current_representation=knot.name,
                    replay_current_fraction=0.25,
                    replay_similar_fraction=0.25,
                    replay_positions_per_episode=4,
                )
            )
    return {
        "selfplay_games": len(records),
        "selfplay_solved": sum(
            bool(record and float(record[0].solved) > 0.5) for record in records
        ),
        "train_steps": len(losses),
        "scheduled_network_evaluations": scheduled,
        "last_loss": losses[-1] if losses else None,
    }


def _evaluate(
    scientist: Any,
    knot: KnotItem,
    *,
    ratios: tuple[float, ...],
    attempts: int,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    cells = {}
    for ratio_index, ratio in enumerate(ratios):
        rows = []
        witnesses: list[tuple[float, int, int, list[int]]] = []
        for attempt in range(attempts):
            verified, measured = _evaluation_record(
                scientist,
                knot,
                ratio,
                simulations,
                seed + ratio_index * 10_000 + attempt,
            )
            rows.append(
                {
                    "attempt": attempt,
                    "solved": verified is not None,
                    "crossing_changes": verified[0] if verified is not None else None,
                    "semantic_moves": verified[1] if verified is not None else None,
                    "objective": (
                        ratio * verified[0] + verified[1]
                        if verified is not None
                        else None
                    ),
                    "scheduled_network_evaluations": measured[
                        "scheduled_network_evaluations"
                    ],
                }
            )
            if verified is not None:
                crossing_changes, semantic_moves, semantic_actions = verified
                witnesses.append(
                    (
                        ratio * crossing_changes + semantic_moves,
                        crossing_changes,
                        semantic_moves,
                        semantic_actions,
                    )
                )
        solved = [row for row in rows if row["solved"]]
        best_witness = min(witnesses, default=None)
        cells[str(ratio)] = {
            "solve_rate": len(solved) / attempts,
            "best_objective": min(
                (float(row["objective"]) for row in solved), default=None
            ),
            "best_witness": (
                {
                    "objective": float(best_witness[0]),
                    "crossing_changes": int(best_witness[1]),
                    "semantic_moves": int(best_witness[2]),
                    "semantic_actions": [int(action) for action in best_witness[3]],
                }
                if best_witness is not None
                else None
            ),
            "attempts": rows,
        }
    return cells


def _retention_summary(
    scientist: Any,
    knots: list[KnotItem],
    *,
    ratios: tuple[float, ...],
    simulations: int,
    seed: int,
    identity_indices: dict[str, int] | None = None,
) -> dict[str, Any]:
    cells = {}
    solved = 0
    capped = 0.0
    for item_index, knot in enumerate(knots):
        evaluation = _evaluate(
            scientist,
            knot,
            ratios=ratios,
            attempts=1,
            simulations=simulations,
            seed=seed
            + (
                identity_indices[knot.name]
                if identity_indices is not None
                else item_index
            )
            * 100_000,
        )
        cells[knot.name] = evaluation
        for ratio in ratios:
            row = evaluation[str(ratio)]
            solved += int(row["best_objective"] is not None)
            failure = ratio * 20.0 + int(scientist.config.game.simplify_budget)
            capped += min(
                failure,
                float(row["best_objective"])
                if row["best_objective"] is not None
                else failure,
            )
    attempts = len(knots) * len(ratios)
    return {
        "attempts": attempts,
        "solved": solved,
        "solve_rate": solved / attempts if attempts else 0.0,
        "capped_cost": capped,
        "mean_capped_cost": capped / attempts if attempts else 0.0,
        "cells": cells,
    }


def _rehearsal_priority(
    knots: list[KnotItem], retention: dict[str, Any], ratios: tuple[float, ...]
) -> list[KnotItem]:
    """Put failed and expensive retained tasks before exposure balancing."""
    def key(knot: KnotItem) -> tuple[int, float, str]:
        cells = retention["cells"][knot.name]
        failures = sum(cells[str(ratio)]["best_objective"] is None for ratio in ratios)
        cost = 0.0
        for ratio in ratios:
            objective = cells[str(ratio)]["best_objective"]
            cost += float(objective) if objective is not None else ratio * 20.0 + 128.0
        return (-failures, -cost, knot.name)

    return sorted(knots, key=key)


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        torch.save(payload, handle)
    os.replace(temporary, path)


def _load_state(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=False)


def _load_roster(
    checkpoints: dict[str, Path],
    *,
    seed: int,
    device: str,
    simulations: int,
    action_horizon: int,
) -> list[Any]:
    scientists = []
    for scientist_index, (name, checkpoint) in enumerate(checkpoints.items()):
        scientist = load_scientist(
            name,
            checkpoint,
            seed=seed + scientist_index * 100_000_000,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=True,
        )
        scientist.config = replace(
            scientist.config,
            game=replace(scientist.config.game, simplify_budget=action_horizon),
        )
        scientist.game = make_game(scientist.config.game)
        scientists.append(scientist)
    return scientists


def _native_witnesses(
    scientist: Any,
    knot: KnotItem,
    ratios: tuple[float, ...],
    evaluation: dict[str, Any],
) -> dict[str, dict[str, Any] | None]:
    """Return the best verified native witness found in training or evaluation."""
    result: dict[str, dict[str, Any] | None] = {}
    for ratio in ratios:
        candidates: list[dict[str, Any]] = []
        evaluated = evaluation[str(ratio)].get("best_witness")
        if evaluated is not None:
            candidates.append({**evaluated, "source": "evaluation"})
            scientist.replay.record_native_objective(
                knot.name, ratio, float(evaluated["objective"])
            )
        archived = scientist.replay.best_native_solution_record(knot.name, ratio)
        if archived is not None:
            verified = verified_record_cost(scientist.game, knot, ratio, archived)
            if verified is not None:
                crossing_changes, semantic_moves, semantic_actions = verified
                candidates.append(
                    {
                        "objective": float(ratio * crossing_changes + semantic_moves),
                        "crossing_changes": int(crossing_changes),
                        "semantic_moves": int(semantic_moves),
                        "semantic_actions": [int(action) for action in semantic_actions],
                        "source": "selfplay",
                    }
                )
        result[str(ratio)] = (
            min(
                candidates,
                key=lambda row: (
                    float(row["objective"]),
                    int(row["crossing_changes"]),
                    int(row["semantic_moves"]),
                    str(row["source"]),
                ),
            )
            if candidates
            else None
        )
    return result


def _portfolio_summary(
    summaries: dict[str, dict[str, Any]],
    knots: list[KnotItem],
    ratios: tuple[float, ...],
    *,
    action_horizon: int,
) -> dict[str, Any]:
    """Take the best paired retention result over the scientist roster."""
    cells: dict[str, Any] = {}
    solved = 0
    capped_cost = 0.0
    objectives = {
        str(ratio): {"attempts": len(knots), "solved": 0, "capped_cost": 0.0}
        for ratio in ratios
    }
    for knot in knots:
        for ratio in ratios:
            candidates = []
            for scientist, summary in summaries.items():
                objective = summary["cells"][knot.name][str(ratio)]["best_objective"]
                if objective is not None:
                    candidates.append((float(objective), scientist))
            failure = ratio * 20.0 + action_horizon
            if candidates:
                objective, scientist = min(candidates)
                solved += 1
                capped_cost += min(failure, objective)
                objectives[str(ratio)]["solved"] += 1
                objectives[str(ratio)]["capped_cost"] += min(failure, objective)
                cells[f"{knot.name}|{ratio:g}"] = {
                    "representation": knot.name,
                    "ratio": ratio,
                    "solved": True,
                    "objective": objective,
                    "scientist": scientist,
                }
            else:
                capped_cost += failure
                objectives[str(ratio)]["capped_cost"] += failure
                cells[f"{knot.name}|{ratio:g}"] = {
                    "representation": knot.name,
                    "ratio": ratio,
                    "solved": False,
                    "objective": failure,
                    "scientist": None,
                }
    attempts = len(knots) * len(ratios)
    return {
        "attempts": attempts,
        "solved": solved,
        "solve_rate": solved / attempts if attempts else 0.0,
        "capped_cost": capped_cost,
        "mean_capped_cost": capped_cost / attempts if attempts else 0.0,
        "objectives": objectives,
        "cells": cells,
    }


def _active_donation_records(scientist: Any) -> list[GameRecord]:
    records = scientist.replay.active_distillation_records()
    return sorted(
        records,
        key=lambda record: (
            str(record[0].representation_id),
            float(record[0].objective_ratio),
            float(record[0].final_crossing_changes) * float(record[0].objective_ratio)
            + float(record[0].final_moves),
        ),
    )


def _donation_is_still_eligible(scientist: Any, record: GameRecord) -> bool:
    if not record or not bool(record[0].shared_witness):
        return False
    representation = str(record[0].representation_id)
    ratio = float(record[0].objective_ratio)
    objective = ratio * float(record[0].final_crossing_changes) + float(
        record[0].final_moves
    )
    native = scientist.replay.best_native_objective(representation, ratio)
    return native is None or objective < native


def _distill_active_donations(
    scientist: Any,
    *,
    dose: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Apply exactly ``dose`` optimizer updates to every active best donation."""
    records = _active_donation_records(scientist)
    exposures = []
    for record_index, record in enumerate(records):
        representation = str(record[0].representation_id)
        ratio = float(record[0].objective_ratio)
        objective = ratio * float(record[0].final_crossing_changes) + float(
            record[0].final_moves
        )
        losses = []
        for exposure in range(dose):
            if not _donation_is_still_eligible(scientist, record):
                break
            isolated = ReplayBuffer(
                max(1024, 2 * len(record)),
                np.random.default_rng(seed + record_index * 10_000 + exposure),
                representation_capacity=1,
            )
            isolated.add(
                deepcopy(record),
                representation_id=representation,
                objective_ratio=ratio,
            )
            losses.append(
                train_alphazero_step(
                    scientist.network,
                    scientist.optimizer,
                    isolated,
                    batch_size,
                    torch.device(scientist.config.train.device),
                    collaboration_replay=True,
                    shared_fraction=1.0,
                    policy_value_success_only=True,
                    replay_current_representation=representation,
                    replay_positions_per_episode=4,
                )
            )
        exposures.append(
            {
                "representation": representation,
                "ratio": ratio,
                "objective": objective,
                "requested_exposures": dose,
                "completed_exposures": len(losses),
                "last_loss": losses[-1] if losses else None,
            }
        )
    return {
        "eligible_witnesses": len(records),
        "completed_optimizer_exposures": sum(
            row["completed_exposures"] for row in exposures
        ),
        "witnesses": exposures,
    }


_SV2_PHASE_SCIENTISTS: dict[str, Any] = {}


def _state_blob(state: dict[str, Any]) -> bytes:
    """Serialize tensor-rich state without multiprocessing FD-per-storage sharing."""
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def _state_from_blob(blob: bytes) -> dict[str, Any]:
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)


def _phase_scientist(initial: dict[str, Any]) -> Any:
    name = str(initial["name"])
    scientist = _SV2_PHASE_SCIENTISTS.get(name)
    if scientist is None:
        scientist = _load_roster(
            {name: Path(initial["checkpoint"])},
            seed=int(initial["seed"]),
            device=str(initial["device"]),
            simulations=int(initial["simulations"]),
            action_horizon=int(initial["action_horizon"]),
        )[0]
        _SV2_PHASE_SCIENTISTS[name] = scientist
    return scientist


def _sv2_phase_operation(scientist: Any, operation: str, payload: dict[str, Any]) -> Any:
    if operation == "assess":
        remaining = payload["remaining"]
        scores = expected_capped_scores(scientist, remaining, (10.0,))
        proposal_index = min(
            range(len(remaining)),
            key=lambda index: (float(scores[index]), remaining[index].id),
        )
        proposal = remaining[proposal_index]
        evidence = _evaluate(
            scientist,
            proposal.knot,
            ratios=(10.0,),
            attempts=int(payload["qualification_attempts"]),
            simulations=int(payload["qualification_simulations"]),
            seed=int(payload["seed"])
            + 600_000_000
            + int(payload["static_index"][proposal.id]) * 100_000,
        )["10.0"]
        evidence_cost = (
            float(evidence["best_objective"])
            if evidence["best_objective"] is not None
            else 10.0 * 20.0 + int(payload["action_horizon"])
        )
        return {
            "scientist": scientist.name,
            "representation": proposal.id,
            "predicted_normalized_capped_loss": float(scores[proposal_index]),
            "evidence": evidence,
            "evidence_capped_L10": evidence_cost,
        }

    if operation == "native":
        selected = payload["selected"]
        ratios = tuple(payload["ratios"])
        scientist.replay.set_representation_embedding(
            selected.id, _replay_representation_embedding(selected.knot)
        )
        iterations = []
        for iteration in range(int(payload["f_native"])):
            iterations.append(
                _iteration(
                    scientist,
                    selected.knot,
                    ratios=ratios,
                    simulations=int(payload["simulations"]),
                    selfplay_games=int(payload["selfplay_games"]),
                    train_steps=int(payload["train_steps"]),
                    batch_size=int(payload["batch_size"]),
                    seed=int(payload["seed"])
                    + int(payload["static_index"]) * 10_000_000
                    + iteration * 100_000,
                )
            )
        evaluation = _evaluate(
            scientist,
            selected.knot,
            ratios=ratios,
            attempts=int(payload["evaluation_attempts"]),
            simulations=int(payload["simulations"]),
            seed=int(payload["seed"])
            + 500_000_000
            + int(payload["static_index"]) * 100_000,
        )
        witnesses = _native_witnesses(scientist, selected.knot, ratios, evaluation)
        qualification_witness = payload.get("qualification_witness")
        if qualification_witness is not None:
            scientist.replay.record_native_objective(
                selected.id, 10.0, float(qualification_witness["objective"])
            )
            candidate = {**qualification_witness, "source": "qualification"}
            incumbent = witnesses["10.0"]
            if incumbent is None or float(candidate["objective"]) < float(
                incumbent["objective"]
            ):
                witnesses["10.0"] = candidate
        return {
            "scientist_event": {
                "iterations": iterations,
                "evaluation": evaluation,
                "native_best": witnesses,
                "rehearsal": None,
            },
            "native_witnesses": witnesses,
        }

    if operation == "translate":
        selected = payload["selected"]
        rows = []
        for donation in payload["donations"]:
            ratio = float(donation["ratio"])
            donor = donation["donor"]
            translated = translate_semantic_record(
                scientist,
                selected.knot,
                ratio,
                list(donor["semantic_actions"]),
                seed=int(payload["seed"])
                + 900_000_000
                + int(payload["static_index"]) * 10_000
                + int(donation["ratio_index"]) * 100
                + int(payload["receiver_index"]),
                internal_action_cap=5,
            )
            translated_cost = (
                verified_record_cost(scientist.game, selected.knot, ratio, translated)
                if translated is not None
                else None
            )
            shared_objective = (
                ratio * translated_cost[0] + translated_cost[1]
                if translated_cost is not None
                else None
            )
            native_objective = scientist.replay.best_native_objective(selected.id, ratio)
            admitted = shared_objective is not None and (
                native_objective is None or shared_objective < native_objective
            )
            if admitted:
                scientist.replay.add(
                    translated,
                    representation_id=selected.id,
                    objective_ratio=ratio,
                )
            rows.append(
                {
                    "ratio": ratio,
                    "author": donation["author"],
                    "receiver": scientist.name,
                    "translated": translated is not None,
                    "admitted": admitted,
                    "receiver_native_objective": native_objective,
                    "donated_objective": shared_objective,
                }
            )
        return rows

    if operation == "rehearse":
        processed_knots = payload["processed_knots"]
        ratios = tuple(payload["ratios"])
        identity_indices = payload["identity_indices"]
        seed = int(payload["seed"])
        round_index = int(payload["round_index"])
        before = _retention_summary(
            scientist,
            processed_knots,
            ratios=ratios,
            simulations=int(payload["simulations"]),
            seed=seed + 700_000_000 + round_index * 100_000,
            identity_indices=identity_indices,
        )
        dose_before = int(payload["f_old"])
        exposure = dict(payload["rehearsal_exposure"])
        priority = _rehearsal_priority(processed_knots, before, ratios)
        priority_rank = {item.name: index for index, item in enumerate(priority)}
        selected_old = sorted(
            priority,
            key=lambda item: (
                exposure.get(item.name, 0),
                priority_rank[item.name],
            ),
        )[:dose_before]
        rehearsal_rows = []
        for old in selected_old:
            previous_exposures = exposure.get(old.name, 0)
            rehearsal_rows.append(
                {
                    "representation": old.name,
                    **_iteration(
                        scientist,
                        old,
                        ratios=ratios,
                        simulations=int(payload["simulations"]),
                        selfplay_games=int(payload["selfplay_games"]),
                        train_steps=int(payload["train_steps"]),
                        batch_size=int(payload["batch_size"]),
                        seed=seed
                        + 800_000_000
                        + int(identity_indices[old.name]) * 1_000_000
                        + previous_exposures * 10_000,
                    ),
                }
            )
            exposure[old.name] = previous_exposures + 1
        after = _retention_summary(
            scientist,
            processed_knots,
            ratios=ratios,
            simulations=int(payload["simulations"]),
            seed=seed + 700_000_000 + round_index * 100_000,
            identity_indices=identity_indices,
        )
        worsened = after["capped_cost"] > before["capped_cost"] + 1e-9
        next_f_old = next_rehearsal_dose(
            dose_before,
            retention_solve_rate=float(after["solve_rate"]),
            capped_cost_worsened=worsened,
            target=float(payload["retention_target"]),
        )
        return {
            "next_F_old": next_f_old,
            "rehearsal_exposure": exposure,
            "retention_after": after,
            "event": {
                "F_old": dose_before,
                "next_F_old": next_f_old,
                "selected": [item.name for item in selected_old],
                "iterations": rehearsal_rows,
                "before": before,
                "after": after,
                "capped_cost_worsened": worsened,
            },
        }

    if operation == "distill-retention":
        exposure = _distill_active_donations(
            scientist,
            dose=int(payload["dose"]),
            batch_size=int(payload["batch_size"]),
            seed=int(payload["distillation_seed"]),
        )
        retention = _retention_summary(
            scientist,
            payload["processed_knots"],
            ratios=tuple(payload["ratios"]),
            simulations=int(payload["simulations"]),
            seed=int(payload["retention_seed"]),
            identity_indices=payload["identity_indices"],
        )
        return {"exposure": exposure, "retention": retention}

    raise ValueError(f"unknown SV2 scientist phase: {operation}")


def _sv2_phase_worker(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(job["initial"]["torch_threads"]))
    scientist = _phase_scientist(job["initial"])
    if job.get("restore_state_blob") is not None:
        _restore_scientist(scientist, _state_from_blob(job["restore_state_blob"]))
    result = _sv2_phase_operation(scientist, str(job["operation"]), job["payload"])
    return {"result": result, "state_blob": _state_blob(_scientist_state(scientist))}


class _ScientistPhaseCoordinator:
    def __init__(
        self,
        checkpoints: dict[str, Path],
        *,
        seed: int,
        device: str,
        simulations: int,
        action_horizon: int,
        torch_threads: int,
        parallel: bool,
        restored_states: dict[str, dict[str, Any]] | None,
    ) -> None:
        self.names = list(checkpoints)
        self.initial = {
            name: {
                "name": name,
                "checkpoint": str(checkpoint),
                "seed": seed + index * 100_000_000,
                "device": device,
                "simulations": simulations,
                "action_horizon": action_horizon,
                "torch_threads": torch_threads,
            }
            for index, (name, checkpoint) in enumerate(checkpoints.items())
        }
        self.states: dict[str, dict[str, Any] | None] = {
            name: (restored_states or {}).get(name) for name in self.names
        }
        self._restore_next = {name: self.states[name] is not None for name in self.names}
        self.parallel = parallel
        self.executors = (
            {
                # JAX starts helper threads during module initialization.  A
                # forked child can inherit their locks and deadlock on Linux;
                # spawn gives every persistent scientist a clean interpreter.
                name: ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=multiprocessing.get_context("spawn"),
                )
                for name in self.names
            }
            if parallel
            else {}
        )
        self.local = (
            {
                scientist.name: scientist
                for scientist in _load_roster(
                    checkpoints,
                    seed=seed,
                    device=device,
                    simulations=simulations,
                    action_horizon=action_horizon,
                )
            }
            if not parallel
            else {}
        )
        if not parallel and restored_states is not None:
            for name, state in restored_states.items():
                _restore_scientist(self.local[name], state)
                self.states[name] = _scientist_state(self.local[name])
                self._restore_next[name] = False

    def run(self, operation: str, payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if self.parallel:
            futures = {}
            for name in self.names:
                futures[name] = self.executors[name].submit(
                    _sv2_phase_worker,
                    {
                        "initial": self.initial[name],
                        "restore_state_blob": (
                            _state_blob(self.states[name])
                            if self._restore_next[name]
                            else None
                        ),
                        "operation": operation,
                        "payload": payloads[name],
                    },
                )
            rows = {name: future.result() for name, future in futures.items()}
            for name, row in rows.items():
                self.states[name] = _state_from_blob(row["state_blob"])
                self._restore_next[name] = False
            return {name: row["result"] for name, row in rows.items()}

        result = {}
        for name in self.names:
            scientist = self.local[name]
            result[name] = _sv2_phase_operation(
                scientist, operation, payloads[name]
            )
            self.states[name] = _scientist_state(scientist)
        return result

    def restore_trainable(self, snapshots: dict[str, dict[str, Any]]) -> None:
        for name in self.names:
            assert self.states[name] is not None
            self.states[name]["network"] = snapshots[name]["network"]
            self.states[name]["optimizer"] = snapshots[name]["optimizer"]
            self.states[name]["prediction_source"] = snapshots[name]["prediction_source"]
            self._restore_next[name] = True
            if not self.parallel:
                _restore_scientist(self.local[name], self.states[name])
                self._restore_next[name] = False

    def trainable_snapshot(self) -> dict[str, dict[str, Any]]:
        snapshots = {}
        for name in self.names:
            state = self.states[name]
            assert state is not None
            snapshots[name] = {
                "network": {
                    key: value.detach().cpu().clone()
                    for key, value in state["network"].items()
                },
                "optimizer": deepcopy(state["optimizer"]),
                "prediction_source": state["prediction_source"],
            }
        return snapshots

    def serializable_states(self) -> dict[str, dict[str, Any]]:
        if any(self.states[name] is None for name in self.names):
            raise RuntimeError("scientist state requested before its first phase")
        return {name: self.states[name] for name in self.names}

    def close(self) -> None:
        for executor in self.executors.values():
            executor.shutdown()


def _coordinated_name(
    arm: CoordinatedArm,
    *,
    scientists: int,
    representations: int,
    simulations: int,
    f_native: int,
    evaluation_attempts: int,
) -> str:
    schedule = "ADAPTIVE-" if arm.startswith("adaptive") else ""
    sharing = "SHARING" if arm.endswith("sharing") and "no-sharing" not in arm else "NO-SHARING"
    return (
        f"SV2-{scientists}S-R{representations}-SIM{simulations}-F{f_native}-"
        f"AR-EV{evaluation_attempts}-{schedule}{sharing}"
    )


def run_coordinated_arm(
    checkpoints: dict[str, Path],
    bank: Path,
    output: Path,
    *,
    arm: CoordinatedArm,
    ratios: tuple[float, ...] = (10.0, 1000.0),
    simulations: int = 64,
    qualification_simulations: int = 64,
    qualification_attempts: int = 1,
    f_native: int = 10,
    selfplay_games: int = 8,
    train_steps: int = 96,
    batch_size: int = 64,
    evaluation_attempts: int = 4,
    block_size: int = 10,
    retention_target: float = 0.80,
    action_horizon: int = 128,
    rungs: int = 0,
    seed: int = 20262020,
    torch_threads: int = 2,
    parallel_scientists: bool = True,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    """Run one synchronized R24 arm with evidence-backed order and/or sharing."""
    if arm not in COORDINATED_ARMS:
        raise ValueError(f"unsupported coordinated SV2 arm: {arm}")
    if not checkpoints:
        raise ValueError("at least one scientist checkpoint is required")
    if not ratios or any(ratio <= 0 for ratio in ratios):
        raise ValueError("objective ratios must be positive")
    if rungs < 0:
        raise ValueError("rungs must be non-negative")
    if min(
        simulations,
        qualification_simulations,
        qualification_attempts,
        f_native,
        selfplay_games,
        train_steps,
        batch_size,
        evaluation_attempts,
        block_size,
        action_horizon,
        torch_threads,
    ) < 1:
        raise ValueError("all SV2 compute and horizon parameters must be positive")
    if arm.startswith("adaptive") and 10.0 not in ratios:
        raise ValueError("adaptive SV2 ordering requires the L10 objective")

    torch.set_num_threads(torch_threads)
    bank_payload, rows = _bank_rows(bank)
    items = _bank_from_payload(rows)
    target_rungs = min(rungs, len(items)) if rungs else len(items)
    by_id = {item.id: item for item in items}
    static_index = {item.id: index for index, item in enumerate(items)}
    name = _coordinated_name(
        arm,
        scientists=len(checkpoints),
        representations=len(items),
        simulations=simulations,
        f_native=f_native,
        evaluation_attempts=evaluation_attempts,
    )
    protocol = {
        "schema": "semantic-v2-coordinated-arm-v2",
        "arm": arm,
        "name": name,
        "source_provenance": source_provenance(),
        "checkpoints": {
            scientist: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for scientist, path in checkpoints.items()
        },
        "bank": str(bank.resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "representations": len(items),
        "requested_rungs": target_rungs,
        "ratios": list(ratios),
        "scheduling_objective": 10.0 if arm.startswith("adaptive") else None,
        "adaptive_selection": (
            "minimum actual L10 qualification evidence among scientist proposals"
            if arm.startswith("adaptive")
            else "global ACS"
        ),
        "simulations": simulations,
        "qualification_simulations": qualification_simulations,
        "qualification_attempts": qualification_attempts,
        "F_native": f_native,
        "selfplay_games_per_iteration": selfplay_games,
        "optimizer_steps_per_iteration": train_steps,
        "batch_size": batch_size,
        "evaluation_attempts_per_objective": evaluation_attempts,
        "adaptive_rehearsal_only": True,
        "F_old_levels": list(F_OLD_LEVELS),
        "retention_target": retention_target,
        "block_size": block_size,
        "sharing": "no-sharing" not in arm,
        "donation_dose_levels": list(DONATION_DOSES),
        "donation_internal_action_cap_per_edit": 5,
        "donation_rule": "verified and strictly better than receiver native incumbent",
        "donation_training": (
            "block-boundary exact optimizer exposures with donation-only rollback"
            if "no-sharing" not in arm
            else None
        ),
        "action_horizon": action_horizon,
        "representation_keyed_seeds": True,
        "seed": seed,
        "torch_threads": torch_threads,
        "parallel_scientists": parallel_scientists,
        "scientist_execution": (
            "one persistent process per scientist"
            if parallel_scientists
            else "sequential equivalence reference"
        ),
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest_path = output / "manifest.json"
    state_path = output / "state.pt.gz"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; pass resume=True")
        previous = json.loads(manifest_path.read_text())
        if previous.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("SV2 resume protocol differs from frozen manifest")
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume without {manifest_path}")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest_path, protocol)
        _atomic_json(output / "bank.json", bank_payload)

    scientist_seeds = {
        name: seed + index * 100_000_000
        for index, name in enumerate(checkpoints)
    }
    if state_path.exists():
        state = _load_state(state_path)
        restored_states = state["scientists"]
        processed = [str(value) for value in state["processed"]]
        events = list(state["events"])
        f_old = {str(key): int(value) for key, value in state["f_old"].items()}
        rehearsal_exposure = {
            str(name): {str(key): int(value) for key, value in rows.items()}
            for name, rows in state["rehearsal_exposure"].items()
        }
        donation_dose = int(state.get("donation_dose", 1))
        donation_healthy_streak = int(state.get("donation_healthy_streak", 0))
    else:
        restored_states = None
        processed = []
        events = []
        f_old = {name: 1 for name in checkpoints}
        rehearsal_exposure = {name: {} for name in checkpoints}
        donation_dose = 1
        donation_healthy_streak = 0

    coordinator = _ScientistPhaseCoordinator(
        checkpoints,
        seed=seed,
        device=device,
        simulations=simulations,
        action_horizon=action_horizon,
        torch_threads=torch_threads,
        parallel=parallel_scientists,
        restored_states=restored_states,
    )

    started = time.perf_counter()
    block_reports: list[dict[str, Any]] = []
    for path in sorted((output / "blocks").glob("*.json")):
        block_reports.append(json.loads(path.read_text()))
    while len(processed) < target_rungs:
        round_index = len(processed)
        remaining = [item for item in items if item.id not in set(processed)]
        qualification: list[dict[str, Any]] = []
        if arm.startswith("adaptive"):
            assessed = coordinator.run(
                "assess",
                {
                    name: {
                        "remaining": remaining,
                        "qualification_attempts": qualification_attempts,
                        "qualification_simulations": qualification_simulations,
                        "seed": scientist_seeds[name],
                        "static_index": static_index,
                        "action_horizon": action_horizon,
                    }
                    for name in coordinator.names
                },
            )
            qualification = [assessed[name] for name in coordinator.names]
            proposals = [
                (
                    float(row["evidence_capped_L10"]),
                    float(row["predicted_normalized_capped_loss"]),
                    str(row["representation"]),
                )
                for row in qualification
            ]
            selected_id = min(proposals)[2]
        else:
            selected_id = min(remaining, key=lambda item: (item.cheap_score, item.id)).id
        selected = by_id[selected_id]

        qualification_witnesses = {
            name: next(
                (
                    row["evidence"].get("best_witness")
                    for row in qualification
                    if row["scientist"] == name and row["representation"] == selected.id
                ),
                None,
            )
            for name in coordinator.names
        }
        native_rows = coordinator.run(
            "native",
            {
                name: {
                    "selected": selected,
                    "ratios": ratios,
                    "f_native": f_native,
                    "simulations": simulations,
                    "selfplay_games": selfplay_games,
                    "train_steps": train_steps,
                    "batch_size": batch_size,
                    "evaluation_attempts": evaluation_attempts,
                    "seed": scientist_seeds[name],
                    "static_index": static_index[selected.id],
                    "qualification_witness": qualification_witnesses[name],
                }
                for name in coordinator.names
            },
        )
        scientist_events = {
            name: native_rows[name]["scientist_event"] for name in coordinator.names
        }
        native_witnesses = {
            name: native_rows[name]["native_witnesses"] for name in coordinator.names
        }

        translations: list[dict[str, Any]] = []
        if "no-sharing" not in arm:
            donations = []
            for ratio_index, ratio in enumerate(ratios):
                candidates = []
                for scientist_name in coordinator.names:
                    witness = native_witnesses[scientist_name][str(ratio)]
                    if witness is not None:
                        candidates.append(
                            (
                                float(witness["objective"]),
                                scientist_name,
                                witness,
                            )
                        )
                if not candidates:
                    continue
                _, author, donor = min(candidates, key=lambda row: (row[0], row[1]))
                donations.append(
                    {
                        "ratio": ratio,
                        "ratio_index": ratio_index,
                        "author": author,
                        "donor": donor,
                    }
                )
            translated_rows = coordinator.run(
                "translate",
                {
                    name: {
                        "selected": selected,
                        "donations": [
                            donation
                            for donation in donations
                            if donation["author"] != name
                        ],
                        "seed": scientist_seeds[name],
                        "static_index": static_index[selected.id],
                        "receiver_index": receiver_index,
                    }
                    for receiver_index, name in enumerate(coordinator.names)
                },
            )
            translations = [
                row for name in coordinator.names for row in translated_rows[name]
            ]

        processed.append(selected.id)
        block_boundary = (
            len(processed) % block_size == 0 or len(processed) == target_rungs
        )
        donation_guard = None
        if block_boundary:
            processed_knots = [by_id[item_id].knot for item_id in processed]
            rehearsal_rows = coordinator.run(
                "rehearse",
                {
                    name: {
                        "processed_knots": processed_knots,
                        "ratios": ratios,
                        "identity_indices": static_index,
                        "seed": scientist_seeds[name],
                        "round_index": round_index,
                        "simulations": simulations,
                        "f_old": f_old[name],
                        "rehearsal_exposure": rehearsal_exposure[name],
                        "selfplay_games": selfplay_games,
                        "train_steps": train_steps,
                        "batch_size": batch_size,
                        "retention_target": retention_target,
                    }
                    for name in coordinator.names
                },
            )
            retention_after = {
                name: rehearsal_rows[name]["retention_after"]
                for name in coordinator.names
            }
            for name in coordinator.names:
                f_old[name] = int(rehearsal_rows[name]["next_F_old"])
                rehearsal_exposure[name] = rehearsal_rows[name][
                    "rehearsal_exposure"
                ]
                scientist_events[name]["rehearsal"] = rehearsal_rows[name]["event"]

            if "no-sharing" not in arm:
                portfolio_before = _portfolio_summary(
                    retention_after,
                    processed_knots,
                    ratios,
                    action_horizon=action_horizon,
                )
                snapshot = coordinator.trainable_snapshot()
                dose_before = donation_dose
                distilled = coordinator.run(
                    "distill-retention",
                    {
                        name: {
                            "dose": donation_dose,
                            "batch_size": batch_size,
                            "distillation_seed": seed
                            + 950_000_000
                            + round_index * 1_000_000
                            + scientist_index * 10_000,
                            "processed_knots": processed_knots,
                            "ratios": ratios,
                            "simulations": simulations,
                            "retention_seed": scientist_seeds[name]
                            + 700_000_000
                            + round_index * 100_000,
                            "identity_indices": static_index,
                        }
                        for scientist_index, name in enumerate(coordinator.names)
                    },
                )
                exposure_rows = {
                    name: distilled[name]["exposure"] for name in coordinator.names
                }
                eligible = sum(
                    row["eligible_witnesses"] for row in exposure_rows.values()
                )
                retention_after_donation = {
                    name: distilled[name]["retention"] for name in coordinator.names
                }
                portfolio_after = _portfolio_summary(
                    retention_after_donation,
                    processed_knots,
                    ratios,
                    action_horizon=action_horizon,
                )
                accepted = all(
                    portfolio_after["objectives"][str(ratio)]["solved"]
                    >= portfolio_before["objectives"][str(ratio)]["solved"]
                    and portfolio_after["objectives"][str(ratio)]["capped_cost"]
                    <= portfolio_before["objectives"][str(ratio)]["capped_cost"]
                    + 1e-9
                    for ratio in ratios
                )
                if not accepted:
                    coordinator.restore_trainable(snapshot)
                decision = adapt_donation_dose(
                    donation_dose,
                    healthy_streak=donation_healthy_streak,
                    eligible_donations=eligible,
                    donation_only_regression=not accepted,
                    portfolio_noninferior=accepted,
                )
                donation_dose = decision.dose
                donation_healthy_streak = decision.healthy_streak
                donation_guard = {
                    "dose": dose_before,
                    "next_dose": donation_dose,
                    "healthy_streak": donation_healthy_streak,
                    "decision": decision.reason,
                    "eligible_distinct_witnesses": eligible,
                    "exposures": exposure_rows,
                    "before": portfolio_before,
                    "after": portfolio_after,
                    "accepted": accepted,
                    "rolled_back": not accepted,
                }

        event = {
            "round": round_index,
            "arm": arm,
            "selected": selected.id,
            "static_index": static_index[selected.id],
            "qualification": qualification,
            "scientists": scientist_events,
            "translations": translations,
            "donation_guard": donation_guard,
        }
        events.append(event)
        _atomic_json(output / "events" / f"{round_index:03d}.json", event)
        if block_boundary:
            block_report = coordinated_block_report(
                events,
                block_size=block_size,
                f_old=f_old,
                donation_dose=donation_dose,
            )
            block_reports = [
                row
                for row in block_reports
                if row["completed_rungs"] != block_report["completed_rungs"]
            ]
            block_reports.append(block_report)
            block_reports.sort(key=lambda row: row["completed_rungs"])
            _atomic_json(
                output / "blocks" / f"{len(processed):03d}.json", block_report
            )
        _save_state(
            state_path,
            {
                "processed": processed,
                "events": events,
                "f_old": f_old,
                "rehearsal_exposure": rehearsal_exposure,
                "donation_dose": donation_dose,
                "donation_healthy_streak": donation_healthy_streak,
                "scientists": coordinator.serializable_states(),
            },
        )

    report = {
        **protocol,
        "completed_rungs": len(processed),
        "processed": processed,
        "final_F_old": f_old,
        "final_donation_dose": donation_dose,
        "block_reports": block_reports,
        "events": events,
        "wall_seconds_this_invocation": time.perf_counter() - started,
    }
    _atomic_json(output / "report.json", report)
    coordinator.close()
    return report


def coordinated_block_report(
    events: list[dict[str, Any]],
    *,
    block_size: int,
    f_old: dict[str, int],
    donation_dose: int,
) -> dict[str, Any]:
    """Build a compact, reproducible report for a completed arm block."""
    if not events or block_size < 1:
        raise ValueError("a block report needs events and a positive block size")
    block_length = len(events) % block_size or min(block_size, len(events))
    block = events[-block_length:]
    names = sorted(block[-1]["scientists"])
    scientists: dict[str, Any] = {}
    for name in names:
        native_iterations = [
            iteration
            for event in block
            for iteration in event["scientists"][name]["iterations"]
        ]
        evaluation = [
            attempt
            for event in block
            for cell in event["scientists"][name]["evaluation"].values()
            for attempt in cell["attempts"]
        ]
        rehearsal = block[-1]["scientists"][name].get("rehearsal")
        scientists[name] = {
            "native_selfplay_solved": sum(
                int(row["selfplay_solved"]) for row in native_iterations
            ),
            "native_selfplay_games": sum(
                int(row["selfplay_games"]) for row in native_iterations
            ),
            "native_network_evaluations": sum(
                int(row["scheduled_network_evaluations"])
                for row in native_iterations
            ),
            "evaluation_solved": sum(bool(row["solved"]) for row in evaluation),
            "evaluation_attempts": len(evaluation),
            "evaluation_network_evaluations": sum(
                int(row["scheduled_network_evaluations"]) for row in evaluation
            ),
            "rehearsal_F_old_used": (
                int(rehearsal["F_old"]) if rehearsal is not None else None
            ),
            "next_F_old": int(f_old[name]),
            "retention_after": (
                {
                    "solved": int(rehearsal["after"]["solved"]),
                    "attempts": int(rehearsal["after"]["attempts"]),
                    "capped_cost": float(rehearsal["after"]["capped_cost"]),
                }
                if rehearsal is not None
                else None
            ),
        }
    translations = [row for event in block for row in event["translations"]]
    guard = block[-1]["donation_guard"]
    return {
        "schema": "semantic-v2-coordinated-block-v1",
        "arm": block[-1]["arm"],
        "rounds": [int(block[0]["round"]), int(block[-1]["round"])],
        "completed_rungs": len(events),
        "selected": [str(event["selected"]) for event in block],
        "scientists": scientists,
        "sharing": {
            "translated": len(translations),
            "admitted": sum(bool(row.get("admitted")) for row in translations),
            "guard": guard,
            "next_donation_dose": int(donation_dose),
        },
    }


def _run_scientist(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(payload.get("torch_threads", 1)))
    output = Path(payload["output"])
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.pt.gz"
    scientist = load_scientist(
        payload["scientist"],
        Path(payload["checkpoint"]),
        seed=int(payload["seed"]),
        device=str(payload["device"]),
        simulations=int(payload["simulations"]),
        require_factorized=True,
        objective_budget_channel=True,
    )
    if payload.get("action_horizon") is not None:
        scientist.config = replace(
            scientist.config,
            game=replace(
                scientist.config.game,
                simplify_budget=int(payload["action_horizon"]),
            ),
        )
        scientist.game = make_game(scientist.config.game)
    knots = _fixed_items(payload["bank_rows"])
    if state_path.exists():
        state = _load_state(state_path)
        _restore_scientist(scientist, state["scientist"])
        completed = int(state["completed"])
        events = list(state["events"])
        f_old = int(state["f_old"])
        retention_reference = state.get("retention_reference")
        rehearsal_exposure = dict(state.get("rehearsal_exposure", {}))
    else:
        completed = 0
        events = []
        f_old = 1
        retention_reference = None
        rehearsal_exposure: dict[str, int] = {}

    started = time.perf_counter()
    for index in range(completed, len(knots)):
        knot = knots[index]
        iterations = []
        for iteration in range(int(payload["f_native"])):
            iterations.append(
                _iteration(
                    scientist,
                    knot,
                    ratios=tuple(payload["ratios"]),
                    simulations=int(payload["simulations"]),
                    selfplay_games=int(payload["selfplay_games"]),
                    train_steps=int(payload["train_steps"]),
                    batch_size=int(payload["batch_size"]),
                    seed=int(payload["seed"]) + index * 10_000_000 + iteration * 100_000,
                )
            )
        evaluation = _evaluate(
            scientist,
            knot,
            ratios=tuple(payload["ratios"]),
            attempts=int(payload["evaluation_attempts"]),
            simulations=int(payload["simulations"]),
            seed=int(payload["seed"]) + 500_000_000 + index * 100_000,
        )
        event: dict[str, Any] = {
            "rung": index,
            "representation": knot.name,
            "F_native": int(payload["f_native"]),
            "simulations": int(payload["simulations"]),
            "iterations": iterations,
            "evaluation": evaluation,
            "rehearsal": None,
        }

        block_boundary = (index + 1) % int(payload["block_size"]) == 0 or index + 1 == len(knots)
        if block_boundary:
            before = _retention_summary(
                scientist,
                knots[: index + 1],
                ratios=tuple(payload["ratios"]),
                simulations=int(payload["simulations"]),
                seed=int(payload["seed"]) + 700_000_000 + index * 100_000,
            )
            dose_before = f_old
            priority = _rehearsal_priority(
                knots[: index + 1], before, tuple(payload["ratios"])
            )
            selected = sorted(
                priority,
                key=lambda item: (
                    rehearsal_exposure.get(item.name, 0),
                    priority.index(item),
                ),
            )[:f_old]
            rehearsal_rows = []
            for rehearsal_index, old in enumerate(selected):
                rehearsal_rows.append(
                    {
                        "representation": old.name,
                        **_iteration(
                            scientist,
                            old,
                            ratios=tuple(payload["ratios"]),
                            simulations=int(payload["simulations"]),
                            selfplay_games=int(payload["selfplay_games"]),
                            train_steps=int(payload["train_steps"]),
                            batch_size=int(payload["batch_size"]),
                            seed=int(payload["seed"])
                            + 800_000_000
                            + index * 100_000
                            + rehearsal_index * 10_000,
                        ),
                    }
                )
                rehearsal_exposure[old.name] = rehearsal_exposure.get(old.name, 0) + 1
            after = _retention_summary(
                scientist,
                knots[: index + 1],
                ratios=tuple(payload["ratios"]),
                simulations=int(payload["simulations"]),
                seed=int(payload["seed"]) + 700_000_000 + index * 100_000,
            )
            # The set grows between blocks, so totals from different block
            # boundaries are not comparable.  The paired before/after probe is.
            worsened = after["capped_cost"] > before["capped_cost"] + 1e-9
            f_old = next_rehearsal_dose(
                f_old,
                retention_solve_rate=float(after["solve_rate"]),
                capped_cost_worsened=worsened,
                target=float(payload["retention_target"]),
            )
            retention_reference = after
            event["rehearsal"] = {
                "F_old": dose_before,
                "next_F_old": f_old,
                "selected": [item.name for item in selected],
                "iterations": rehearsal_rows,
                "before": before,
                "after": after,
                "capped_cost_worsened": worsened,
            }

        events.append(event)
        completed = index + 1
        _atomic_json(output / "events" / f"{index:03d}.json", event)
        _save_state(
            state_path,
            {
                "completed": completed,
                "events": events,
                "f_old": f_old,
                "retention_reference": retention_reference,
                "rehearsal_exposure": rehearsal_exposure,
                "scientist": _scientist_state(scientist),
            },
        )

    report = {
        "scientist": payload["scientist"],
        "completed_rungs": completed,
        "F_native": int(payload["f_native"]),
        "final_F_old": f_old,
        "simulations": int(payload["simulations"]),
        "evaluation_attempts": int(payload["evaluation_attempts"]),
        "events": events,
        "wall_seconds_this_invocation": time.perf_counter() - started,
    }
    _atomic_json(output / "report.json", report)
    return report


def run_static_no_sharing(
    checkpoints: dict[str, Path],
    bank: Path,
    output: Path,
    *,
    ratios: tuple[float, ...] = (10.0, 1000.0),
    simulations: int = 64,
    f_native: int = 10,
    selfplay_games: int = 8,
    train_steps: int = 96,
    batch_size: int = 64,
    evaluation_attempts: int = 4,
    block_size: int = 10,
    retention_target: float = 0.80,
    action_horizon: int = 128,
    seed: int = 20262020,
    workers: int = 3,
    torch_threads_per_worker: int = 2,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    """Run three independent scientists sequentially over the fixed R24 bank."""
    if not checkpoints:
        raise ValueError("at least one scientist checkpoint is required")
    if f_native < 1 or evaluation_attempts < 1 or simulations < 1:
        raise ValueError("native, evaluation, and simulation doses must be positive")
    if workers < 1 or torch_threads_per_worker < 1:
        raise ValueError("worker counts and thread counts must be positive")
    bank_payload, rows = _bank_rows(bank)
    protocol = {
        "schema": "semantic-v2-static-no-sharing-v1",
        "arm": "static-no-sharing",
        "name": "SV2-3S-R24-SIM64-F10-AR-EV4-NO-SHARING",
        "source_provenance": source_provenance(),
        "checkpoints": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in checkpoints.items()
        },
        "bank": str(bank.resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "representations": len(rows),
        "ratios": list(ratios),
        "simulations": simulations,
        "F_native": f_native,
        "selfplay_games_per_iteration": selfplay_games,
        "optimizer_steps_per_iteration": train_steps,
        "batch_size": batch_size,
        "evaluation_attempts_per_objective": evaluation_attempts,
        "adaptive_compute": False,
        "adaptive_rehearsal_only": True,
        "F_old_levels": list(F_OLD_LEVELS),
        "retention_target": retention_target,
        "block_size": block_size,
        "action_horizon": action_horizon,
        "seed": seed,
        "workers": workers,
        "torch_threads_per_worker": torch_threads_per_worker,
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{manifest} exists; pass resume=True")
        if json.loads(manifest.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("SV2 resume protocol differs from frozen manifest")
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume without {manifest}")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, protocol)
        _atomic_json(output / "bank.json", bank_payload)

    jobs = [
        {
            "scientist": name,
            "checkpoint": str(checkpoint),
            "output": str(output / "scientists" / name),
            "bank_rows": rows,
            "ratios": ratios,
            "simulations": simulations,
            "f_native": f_native,
            "selfplay_games": selfplay_games,
            "train_steps": train_steps,
            "batch_size": batch_size,
            "evaluation_attempts": evaluation_attempts,
            "block_size": block_size,
            "retention_target": retention_target,
            "action_horizon": action_horizon,
            "seed": seed + index * 100_000_000,
            "device": device,
            "torch_threads": torch_threads_per_worker,
        }
        for index, (name, checkpoint) in enumerate(checkpoints.items())
    ]
    if workers == 1:
        reports = [_run_scientist(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            reports = list(executor.map(_run_scientist, jobs))
    report = {**protocol, "scientists": reports}
    _atomic_json(output / "report.json", report)
    return report
