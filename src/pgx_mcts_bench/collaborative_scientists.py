"""Resumable heterogeneous scientist collaboration over a frozen task bank.

This runner extracts the underlying semantic braid edits from a
solved serial trajectory and routes each edit through the receiver's own head and
memory action space.  A translated record is admitted only when exact replay
still reaches the unknot within the receiver's budget.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import (
    FixedWordGame,
    KnotItem,
    Scientist,
    _observation_tensor,
    calibrated_solve_probability,
    load_scientist,
)
from pgx_mcts_bench.data import GameRecord, Position
from pgx_mcts_bench.distill import _best_destination, train_bounded_option_step
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.networks import load_policy_value_state_dict
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.serial_braid import SerialBraidGame
from pgx_mcts_bench.training import (
    attach_policy_value_preservation_teacher,
    play_selfplay_games,
    train_alphazero_step,
)

Arm = Literal[
    "adaptive-sharing",
    "adaptive-sharing-direct",
    "adaptive-sharing-aux-only",
    "adaptive-no-sharing",
    "static-sharing",
    "static-no-sharing",
    "solo-compute-matched",
]

DIRECT_SHARING_ARMS = frozenset(
    {"adaptive-sharing", "adaptive-sharing-direct", "static-sharing"}
)
SHARING_ARMS = frozenset({*DIRECT_SHARING_ARMS, "adaptive-sharing-aux-only"})

ASSESSOR_SCAN_STEPS = 5
ASSESSOR_SCORING_RULE = "mean-head-sweep-v1"

_WORKER_SCIENTISTS: dict[tuple[str, str, str, bool, int], Scientist] = {}


@dataclass(frozen=True)
class BankItem:
    id: str
    knot: KnotItem
    cheap_score: float
    difficulty_quartile: int
    known_unknotting_number: int | None = None
    certified_unknotting_lower_bound: int | None = None


@dataclass(frozen=True)
class AttemptRun:
    record: GameRecord
    budget: dict[str, Any] | None = None
    # Includes lower-cap failed attempts followed by the returned final attempt.
    # Only the final record decides task success; every attempt is useful for the
    # conditional solve-probability target at its own encoded budget.
    replay_records: tuple[GameRecord, ...] = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _compatible_table() -> list[KnotItem]:
    from rf_knots.knot_table import load_table

    items = []
    for name, row in load_table()["knots"].items():
        word = tuple(int(letter) for letter in row["braid"])
        strands = int(row["strands"])
        crossings = int(row["crossings"])
        if crossings <= 0 or strands > 5 or len(word) > 48:
            continue
        items.append(KnotItem(name, crossings, word, strands))
    return items


def _cheap_score(knot: KnotItem) -> float:
    # Universally available, deliberately non-neural proxy.  Coefficients are
    # fixed engineering choices for the pilot, not a fitted scientific baseline.
    return float(len(knot.word) + 8 * (knot.strands - 2) + knot.crossings)


def _replay_representation_embedding(knot: KnotItem) -> np.ndarray:
    """Cheap deterministic similarity key; learned embeddings can replace it later."""
    word = np.asarray(knot.word, dtype=np.float32)
    signs = np.sign(word)
    sign_changes = float(np.count_nonzero(signs[1:] != signs[:-1])) if len(word) > 1 else 0.0
    generator_histogram = np.zeros(4, dtype=np.float32)
    for generator in np.abs(word).astype(int):
        if 1 <= generator <= len(generator_histogram):
            generator_histogram[generator - 1] += 1.0
    if len(word):
        generator_histogram /= len(word)
    return np.asarray(
        [
            len(word) / 48.0,
            knot.strands / 5.0,
            knot.crossings / 20.0,
            float(word.sum()) / 48.0,
            float(np.abs(word).sum()) / 192.0,
            float(np.count_nonzero(word > 0)) / max(len(word), 1),
            sign_changes / max(len(word) - 1, 1),
            *generator_histogram.tolist(),
        ],
        dtype=np.float32,
    )


@lru_cache(maxsize=1)
def _unknotting_sources() -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    from rf_knots.knot_table import DATA, unknotting_numbers

    lower_payload = json.loads((DATA / "lower_bounds.json").read_text())
    return unknotting_numbers()["values"], lower_payload["values"]


def _unknotting_metadata(name: str) -> tuple[int | None, int | None]:
    exact_values, lower_values = _unknotting_sources()
    exact = exact_values.get(name)
    row = lower_values.get(name, {})
    candidates = [
        abs(int(row.get("ozsvath_szabo_tau", 0))),
        abs(int(row.get("rasmussen_s", 0))) // 2,
    ]
    if row.get("nakanishi_lower") is not None:
        candidates.append(int(row["nakanishi_lower"]))
    lower = max(candidates) if candidates else None
    return (int(exact) if exact is not None else None, lower)


def _bank_item(knot: KnotItem, quartile: int) -> BankItem:
    exact, lower = _unknotting_metadata(knot.name)
    return BankItem(
        knot.name,
        knot,
        _cheap_score(knot),
        quartile,
        known_unknotting_number=exact,
        certified_unknotting_lower_bound=lower,
    )


def stratified_banks(
    size: int = 200, anchor_size: int = 70, seed: int = 0
) -> tuple[list[BankItem], list[BankItem]]:
    """Create identity-disjoint, difficulty-spanning pilot and anchor banks."""
    compatible = sorted(_compatible_table(), key=lambda k: (_cheap_score(k), k.name))
    if len(compatible) < size + anchor_size:
        raise ValueError(
            f"only {len(compatible)} compatible identities; requested {size + anchor_size}"
        )
    quartiles = [list(part) for part in np.array_split(np.asarray(compatible, dtype=object), 4)]
    chosen: list[BankItem] = []
    anchors: list[BankItem] = []
    for quartile, items in enumerate(quartiles):
        ranked = sorted(
            items,
            key=lambda knot: hashlib.sha256(f"{seed}:{quartile}:{knot.name}".encode()).digest(),
        )
        take = size // 4 + int(quartile < size % 4)
        anchor_take = anchor_size // 4 + int(quartile < anchor_size % 4)
        for knot in ranked[:take]:
            chosen.append(_bank_item(knot, quartile))
        for knot in ranked[take : take + anchor_take]:
            anchors.append(_bank_item(knot, quartile))
    chosen.sort(key=lambda item: (item.cheap_score, item.id))
    anchors.sort(key=lambda item: (item.difficulty_quartile, item.id))
    return chosen, anchors


def _bank_payload(items: list[BankItem]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        row = {
            "id": item.id,
            "name": item.knot.name,
            "crossings": item.knot.crossings,
            "word": list(item.knot.word),
            "strands": item.knot.strands,
            "cheap_score": item.cheap_score,
            "difficulty_quartile": item.difficulty_quartile,
        }
        if item.known_unknotting_number is not None:
            row["known_unknotting_number"] = item.known_unknotting_number
        if item.certified_unknotting_lower_bound is not None:
            row["certified_unknotting_lower_bound"] = item.certified_unknotting_lower_bound
        rows.append(row)
    return rows


def _bank_from_payload(rows: list[dict[str, Any]]) -> list[BankItem]:
    return [
        BankItem(
            str(row["id"]),
            KnotItem(
                str(row["name"]),
                int(row["crossings"]),
                tuple(int(x) for x in row["word"]),
                int(row["strands"]),
            ),
            float(row["cheap_score"]),
            int(row["difficulty_quartile"]),
            (
                int(row["known_unknotting_number"])
                if row.get("known_unknotting_number") is not None
                else None
            ),
            (
                int(row["certified_unknotting_lower_bound"])
                if row.get("certified_unknotting_lower_bound") is not None
                else None
            ),
        )
        for row in rows
    ]


@torch.inference_mode()
def expected_capped_scores(
    scientist: Scientist,
    items: list[BankItem],
    ratios: tuple[float, ...],
    *,
    failure_crossings: float = 20.0,
) -> np.ndarray:
    """Post-scan mean normalized expected loss used by adaptive scheduling."""
    if not isinstance(scientist.game, SerialBraidGame):
        raise TypeError("adaptive assessor requires a bounded serial game")
    scanned = [
        scientist.game.assessment_scan(
            list(item.knot.word),
            item.knot.strands,
            math.log(ratio),
            steps=ASSESSOR_SCAN_STEPS,
        )
        for item in items
        for ratio in ratios
    ]
    observations = [view for views, _coverage in scanned for view in views]
    tensor = _observation_tensor(observations, torch.device(scientist.config.train.device))
    scientist.network.eval()
    _, legacy, auxiliary = scientist.network.forward_with_auxiliary(tensor)
    budget = float(scientist.config.game.simplify_budget)
    if auxiliary is None or scientist.prediction_source == "legacy_proxy":
        probability = ((legacy + 1.0) / 2.0).clamp(0.0, 1.0)
        crossings = failure_crossings * (1.0 - probability)
        moves = budget * (1.0 - probability)
    else:
        solve_logits, predicted_crossings, predicted_moves = auxiliary
        probability = calibrated_solve_probability(scientist, solve_logits)
        crossings = predicted_crossings.mean(dim=1)
        moves = predicted_moves.mean(dim=1)
    shape = (len(items), len(ratios), ASSESSOR_SCAN_STEPS)
    probability = probability.reshape(shape).mean(dim=2)
    crossings = crossings.reshape(shape).mean(dim=2)
    moves = moves.reshape(shape).mean(dim=2)
    ratio_tensor = torch.tensor(ratios, device=tensor.device)[None, :]
    failure = ratio_tensor * failure_crossings + budget
    expected = probability * (ratio_tensor * crossings + moves) + (1.0 - probability) * failure
    normalized = (expected / failure).mean(dim=1)
    return normalized.cpu().numpy()


@torch.inference_mode()
def prediction_details(
    scientist: Scientist,
    item: BankItem,
    ratios: tuple[float, ...],
    *,
    failure_crossings: float = 20.0,
) -> list[dict[str, Any]]:
    if not isinstance(scientist.game, SerialBraidGame):
        raise TypeError("adaptive assessor requires a bounded serial game")
    scanned = [
        scientist.game.assessment_scan(
            list(item.knot.word),
            item.knot.strands,
            math.log(ratio),
            steps=ASSESSOR_SCAN_STEPS,
        )
        for ratio in ratios
    ]
    observations = [view for views, _coverage in scanned for view in views]
    tensor = _observation_tensor(observations, torch.device(scientist.config.train.device))
    scientist.network.eval()
    _, legacy, auxiliary = scientist.network.forward_with_auxiliary(tensor)
    budget = float(scientist.config.game.simplify_budget)
    if auxiliary is None or scientist.prediction_source == "legacy_proxy":
        probability = ((legacy + 1.0) / 2.0).clamp(0.0, 1.0)
        crossings = failure_crossings * (1.0 - probability)
        moves = budget * (1.0 - probability)
    else:
        solve_logits, predicted_crossings, predicted_moves = auxiliary
        probability = calibrated_solve_probability(scientist, solve_logits)
        crossings = predicted_crossings.mean(dim=1)
        moves = predicted_moves.mean(dim=1)
    shape = (len(ratios), ASSESSOR_SCAN_STEPS)
    probability = probability.reshape(shape).mean(dim=1)
    crossings = crossings.reshape(shape).mean(dim=1)
    moves = moves.reshape(shape).mean(dim=1)
    rows = []
    for index, ratio in enumerate(ratios):
        failure = ratio * failure_crossings + budget
        expected = float(
            probability[index] * (ratio * crossings[index] + moves[index])
            + (1.0 - probability[index]) * failure
        )
        rows.append(
            {
                "ratio": ratio,
                "p_solve": float(probability[index]),
                "predicted_crossing_changes": float(crossings[index]),
                "predicted_moves": float(moves[index]),
                "expected_capped_loss": expected,
                "normalized_expected_capped_loss": expected / failure,
                "assessment_scan_steps": ASSESSOR_SCAN_STEPS,
                "assessment_scan_coverage": float(scanned[index][1]),
                "assessment_scoring_rule": ASSESSOR_SCORING_RULE,
            }
        )
    return rows


def _play(
    scientist: Scientist,
    knot: KnotItem,
    ratio: float,
    *,
    simulations: int,
    seed: int,
    objective_cap: float | None = None,
    cap_type: str = "global",
) -> GameRecord:
    fixed = FixedWordGame(
        scientist.game,
        knot,
        ratio,
        objective_cap=objective_cap,
        cap_type=cap_type,
    )
    search_config = replace(scientist.config.search, simulations=simulations)
    search = NeuralMCTS(fixed, scientist.network, search_config, scientist.config.train.device)
    return play_selfplay_games(
        fixed,
        search,
        [np.random.default_rng(seed)],
        [seed],
        scientist.config.train.temperature_moves,
    )[0]


def play_with_objective_restarts(
    scientist: Scientist,
    knot: KnotItem,
    ratio: float,
    *,
    predicted_objective: float,
    simulations: int,
    seed: int,
    multiplier: float = 2.0,
    max_restarts: int | None = None,
    retained_records: list[GameRecord] | None = None,
) -> tuple[GameRecord, dict[str, Any]]:
    """Run the historical, rejected learned-cap budget-savings ablation.

    The collaboration runner does not call this helper.  It remains available
    only so old budget-savings artifacts can be reproduced and compared with
    the prediction-independent structural-cap protocol.
    """
    if not scientist.config.game.objective_budget_channel:
        raise ValueError("objective restart search requires its observation channel")
    global_cap = (ratio + 1.0) * scientist.config.game.simplify_budget
    cap = min(global_cap, max(1.0, math.ceil(multiplier * predicted_objective)))
    attempts: list[dict[str, Any]] = []
    attempt_index = 0
    while True:
        record = _play(
            scientist,
            knot,
            ratio,
            simulations=simulations,
            # Keep escalation paired with the original attempt. Changing the
            # seed would confound "more budget" with a different noisy search
            # and could fail to recover a baseline solution even at global cap.
            seed=seed,
            objective_cap=cap,
            cap_type="predicted" if attempt_index == 0 else "restart",
        )
        if retained_records is not None:
            retained_records.append(record)
        solved = bool(record and record[0].solved > 0.5)
        final_moves = float(record[0].final_moves) if record else 0.0
        final_native_plies = float(record[0].final_native_plies) if record else 0.0
        # If the ordinary move clock fired, raising an objective cap cannot add
        # search depth. Otherwise an unsolved attempt below the global cap is a
        # censored result and must be retried geometrically.
        objective_exhausted = bool(
            record and record[0].objective_censored and not solved and cap < global_cap
        )
        attempts.append(
            {
                "cap": cap,
                "solved": solved,
                "objective_budget_exhausted": objective_exhausted,
                "semantic_moves": final_moves,
                "native_plies": final_native_plies,
            }
        )
        if (
            solved
            or cap >= global_cap
            or (max_restarts is not None and attempt_index >= max_restarts)
        ):
            return record, {
                "initial_cap": attempts[0]["cap"],
                "final_cap": cap,
                "global_cap": global_cap,
                "restart_count": attempt_index,
                "attempts": attempts,
            }
        next_cap = min(global_cap, max(cap + 1.0, math.ceil(cap * 2.0)))
        if next_cap <= cap:
            return record, {
                "initial_cap": attempts[0]["cap"],
                "final_cap": cap,
                "global_cap": global_cap,
                "restart_count": attempt_index,
                "attempts": attempts,
            }
        cap = next_cap
        attempt_index += 1


def common_structural_objective_cap(
    knot: KnotItem,
    ratio: float,
    action_horizon: int,
) -> float:
    """Return a scientist-independent first budget for one representation.

    A local serial critic cannot see the unseen part of the braid at its initial
    head position, so its predicted cost is not admissible as an attempt cap.
    The common first tier instead uses the complete observed braid-word length.
    It permits one crossing change for every two observed intersections plus the
    full semantic-move allowance.  This is an economical probe, not a claim that
    the environment can solve the representation within that tier; every
    budget-censored failure is retried at the declared global cap.
    """
    if ratio <= 0.0 or not math.isfinite(ratio):
        raise ValueError("objective ratio must be finite and positive")
    if action_horizon < 1:
        raise ValueError("action horizon must be positive")
    observed_intersections = len(knot.word)
    crossing_allowance = math.ceil(observed_intersections / 2)
    global_cap = (ratio + 1.0) * action_horizon
    structural_cap = ratio * crossing_allowance + action_horizon
    return float(min(global_cap, max(1.0, structural_cap)))


def play_with_common_objective_restarts(
    scientist: Scientist,
    knot: KnotItem,
    ratio: float,
    *,
    simulations: int,
    seed: int,
    retained_records: list[GameRecord] | None = None,
) -> tuple[GameRecord, dict[str, Any]]:
    """Run a common structural probe, then recover every censored failure.

    The first cap depends only on the complete representation and the declared
    action horizon, never on one scientist's local value prediction.  A capped
    failure restarts once, with the same seed, at the common global cap.  An
    ordinary action-horizon failure is not rerun because it was not censored by
    the objective budget.
    """
    if not scientist.config.game.objective_budget_channel:
        raise ValueError("objective restart search requires its observation channel")
    action_horizon = int(scientist.config.game.simplify_budget)
    global_cap = (ratio + 1.0) * action_horizon
    initial_cap = common_structural_objective_cap(knot, ratio, action_horizon)
    caps = [initial_cap] if initial_cap >= global_cap else [initial_cap, global_cap]
    attempts: list[dict[str, Any]] = []
    final_record: GameRecord | None = None
    for attempt_index, cap in enumerate(caps):
        record = _play(
            scientist,
            knot,
            ratio,
            simulations=simulations,
            seed=seed,
            objective_cap=cap,
            cap_type="structural" if attempt_index == 0 else "common-global-restart",
        )
        final_record = record
        if retained_records is not None:
            retained_records.append(record)
        solved = bool(record and record[0].solved > 0.5)
        objective_exhausted = bool(
            record and record[0].objective_censored and not solved and cap < global_cap
        )
        attempts.append(
            {
                "cap": cap,
                "cap_type": "structural" if attempt_index == 0 else "common-global-restart",
                "solved": solved,
                "objective_budget_exhausted": objective_exhausted,
                "semantic_moves": float(record[0].final_moves) if record else 0.0,
                "native_plies": float(record[0].final_native_plies) if record else 0.0,
            }
        )
        if solved or not objective_exhausted:
            break
    assert final_record is not None
    return final_record, {
        "initial_cap": initial_cap,
        "final_cap": attempts[-1]["cap"],
        "global_cap": global_cap,
        "restart_count": len(attempts) - 1,
        "cap_source": "full-representation-structural-v1",
        "scientist_prediction_used": False,
        "restart_policy": "every-censored-failure-to-common-global-cap",
        "attempts": attempts,
    }


def _run_attempt(
    scientist: Scientist,
    knot: KnotItem,
    ratio: float,
    *,
    simulations: int,
    seed: int,
    objective_budget: bool,
) -> AttemptRun:
    if not objective_budget:
        return AttemptRun(_play(scientist, knot, ratio, simulations=simulations, seed=seed))
    replay_records: list[GameRecord] = []
    record, budget = play_with_common_objective_restarts(
        scientist,
        knot,
        ratio,
        simulations=simulations,
        seed=seed,
        retained_records=replay_records,
    )
    return AttemptRun(record, budget, tuple(replay_records))


def _worker_play_bundle(
    name: str,
    checkpoint: str,
    network_state: dict[str, torch.Tensor],
    knot: KnotItem,
    ratios: tuple[float, ...],
    simulations: int,
    seeds: tuple[int, ...],
    device: str,
    objective_budget: bool,
    remaining_budget_channel: bool,
    action_horizon: int,
) -> list[AttemptRun]:
    """Process-pool entrypoint; one network transfer serves every ratio."""
    torch.set_num_threads(1)
    key = (name, checkpoint, device, remaining_budget_channel, action_horizon)
    scientist = _WORKER_SCIENTISTS.get(key)
    if scientist is None:
        scientist = load_scientist(
            name,
            Path(checkpoint),
            seed=seeds[0],
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=remaining_budget_channel,
        )
        scientist.config = replace(
            scientist.config,
            game=replace(scientist.config.game, simplify_budget=action_horizon),
        )
        scientist.game = make_game(scientist.config.game)
        _WORKER_SCIENTISTS[key] = scientist
    load_policy_value_state_dict(scientist.network, network_state)
    return [
        _run_attempt(
            scientist,
            knot,
            ratio,
            simulations=simulations,
            seed=attempt_seed,
            objective_budget=objective_budget,
        )
        for ratio, attempt_seed in zip(ratios, seeds, strict=True)
    ]


def _play_bundles(
    executor: ProcessPoolExecutor | None,
    jobs: list[
        tuple[
            Scientist,
            KnotItem,
            tuple[float, ...],
            int,
            tuple[int, ...],
            bool,
            bool,
            int,
        ]
    ],
) -> list[list[AttemptRun]]:
    if executor is None:
        return [
            [
                _run_attempt(
                    scientist,
                    knot,
                    ratio,
                    simulations=simulations,
                    seed=seed,
                    objective_budget=objective_budget,
                )
                for ratio, seed in zip(ratios, seeds, strict=True)
            ]
            for (
                scientist,
                knot,
                ratios,
                simulations,
                seeds,
                objective_budget,
                _remaining_budget_channel,
                _action_horizon,
            ) in jobs
        ]
    futures = [
        executor.submit(
            _worker_play_bundle,
            scientist.name,
            str(scientist.checkpoint),
            scientist.network.state_dict(),
            knot,
            ratios,
            simulations,
            seeds,
            scientist.config.train.device,
            objective_budget,
            remaining_budget_channel,
            action_horizon,
        )
        for (
            scientist,
            knot,
            ratios,
            simulations,
            seeds,
            objective_budget,
            remaining_budget_channel,
            action_horizon,
        ) in jobs
    ]
    return [future.result() for future in futures]


def _adaptive_dose(current: int, healthy: bool, maximum: int) -> int:
    """Hold a healthy rehearsal dose; double an unhealthy one up to its cap."""
    if current < 1 or maximum < current:
        raise ValueError("adaptive rehearsal dose must satisfy 1 <= current <= maximum")
    return current if healthy else min(maximum, max(current + 1, 2 * current))


def _rotating_retention_items(
    items: list[BankItem], count: int, *, seed: int, block: int
) -> list[BankItem]:
    """Select a deterministic rotating panel without changing scientific seeds."""
    if count <= 0 or not items:
        return []
    ranked = sorted(
        items,
        key=lambda item: hashlib.sha256(f"retention:{seed}:{block}:{item.id}".encode()).digest(),
    )
    return ranked[: min(count, len(ranked))]


def _rehearsal_ids(
    processed: list[str],
    priority: list[str],
    exposures: dict[str, int],
    dose: int,
) -> list[str]:
    """Allocate 50% degraded, 25% recent, and 25% exposure-balanced games.

    The list may repeat identities when the history is smaller than the requested
    dose.  This keeps ``F_old`` a bounded *total* per training block instead of
    growing once for every representation ever seen.
    """
    if dose <= 0 or not processed:
        return []
    available = list(dict.fromkeys(processed))
    available_set = set(available)
    priority = [item for item in dict.fromkeys(priority) if item in available_set]
    recent = list(reversed(available[-max(1, min(len(available), dose)) :]))
    uniform = sorted(available, key=lambda item: (exposures.get(item, 0), item))
    quotas = (math.ceil(dose / 2), math.ceil(dose / 4))
    pools = (priority or uniform, recent, uniform)
    counts = (quotas[0], quotas[1], dose - quotas[0] - quotas[1])
    selected: list[str] = []
    for pool, count in zip(pools, counts, strict=True):
        for index in range(max(count, 0)):
            selected.append(pool[index % len(pool)])
    for item in selected:
        exposures[item] = exposures.get(item, 0) + 1
    return selected


def _training_snapshot(
    scientists: list[Scientist],
    option_optimizers: dict[str, torch.optim.Optimizer],
) -> dict[str, dict[str, Any]]:
    """Capture only trainable state; replay and the permanent bank survive rollback."""
    snapshots: dict[str, dict[str, Any]] = {}
    for scientist in scientists:
        row = {
            "network": {
                key: value.detach().cpu().clone()
                for key, value in scientist.network.state_dict().items()
            },
            "optimizer": deepcopy(scientist.optimizer.state_dict()),
            "prediction_source": scientist.prediction_source,
        }
        if scientist.name in option_optimizers:
            row["option_optimizer"] = deepcopy(option_optimizers[scientist.name].state_dict())
        snapshots[scientist.name] = row
    return snapshots


def _restore_training_snapshot(
    scientists: list[Scientist],
    option_optimizers: dict[str, torch.optim.Optimizer],
    snapshots: dict[str, dict[str, Any]],
) -> None:
    for scientist in scientists:
        row = snapshots[scientist.name]
        load_policy_value_state_dict(scientist.network, row["network"])
        scientist.optimizer.load_state_dict(row["optimizer"])
        scientist.prediction_source = str(row["prediction_source"])
        if scientist.name in option_optimizers:
            option_optimizers[scientist.name].load_state_dict(row["option_optimizer"])


def _raw_state(game: Any, state: Any) -> Any:
    return game.unwrap(state) if hasattr(game, "unwrap") else state


def _is_solved(game: Any, state: Any) -> bool:
    raw = _raw_state(game, state)
    return bool((np.asarray(raw._word) == 0).all()) and int(np.asarray(raw._n)) == 1


def _record_semantic_actions(
    game: Any, knot: KnotItem, ratio: float, record: GameRecord
) -> tuple[list[int], list[Any]] | None:
    """Replay a native record and extract global braid-action indices."""
    transition = game.from_word(list(knot.word), knot.strands, math.log(ratio))
    semantic: list[int] = []
    states = [_raw_state(game, transition.state)]
    for position in record:
        if transition.terminated or not transition.legal_actions[position.action]:
            return None
        if isinstance(game, SerialBraidGame):
            state = transition.state
            length = int(np.asarray(state.pgx._word).astype(bool).sum())
            if position.action < game._shift_base:  # noqa: SLF001
                underlying = game.underlying_action(position.action, state.head, length)
                if underlying is None:
                    return None
                semantic.append(int(underlying))
        else:
            semantic.append(int(position.action))
        transition = game.step(transition.state, position.action)
        states.append(_raw_state(game, transition.state))
    if not transition.terminated or not _is_solved(game, transition.state):
        return None
    return semantic, states


def verified_record_cost(
    game: Any, knot: KnotItem, ratio: float, record: GameRecord
) -> tuple[int, int, list[int]] | None:
    """Verify a native solution and return its portable semantic objective cost.

    The semantic verifier deliberately removes controller-only states so the proof
    can be replayed by another architecture. Head, tape, and memory operations
    consume native search clocks but do not enter ``L_A:B``. The returned action
    list is reconstructed from the verified compact witness so its length is
    exactly the semantic move count.
    """
    replayed = _record_semantic_actions(game, knot, ratio, record)
    if replayed is None:
        return None
    semantic, raw_states = replayed
    from rf_knots.evidence import BraidState

    states = [
        BraidState(
            tuple(int(value) for value in np.asarray(state._word) if int(value)),
            int(np.asarray(state._n)),
            game.config._spec.cyclic_band_generators,
        )
        for state in raw_states
    ]
    from pgx_mcts_bench.semantic_verifier import SemanticBraidVerifier

    witness = SemanticBraidVerifier.from_config(game.config).verify_states(states)
    witness.verify()
    semantic = [int(step.action.to_flat(game.config._spec)) for step in witness.steps]
    return witness.crossing_changes, witness.moves, semantic


def translate_semantic_record(
    receiver: Scientist,
    knot: KnotItem,
    ratio: float,
    semantic_actions: list[int],
    *,
    seed: int,
    internal_action_cap: int | None = None,
) -> GameRecord | None:
    """Route semantic edits through one receiver's native controller."""
    game = receiver.game
    transition = game.from_word(list(knot.word), knot.strands, math.log(ratio))
    record: GameRecord = []

    def take(action: int) -> bool:
        nonlocal transition
        if transition.terminated or not bool(transition.legal_actions[action]):
            return False
        policy = np.zeros_like(transition.legal_actions, dtype=np.float32)
        policy[action] = 1.0
        position = Position(
            observation=transition.observation,
            legal_actions=transition.legal_actions,
            policy=policy,
            action=action,
            player=transition.player,
            role=1,
            episode_seed=seed,
            value_potential=game.value_potential(transition.state, transition.player),
            shared_witness=True,
            representation_id=knot.name,
            objective_ratio=float(ratio),
            action_horizon=int(game.config.simplify_budget),
            residual_word_length=int(
                np.count_nonzero(np.asarray(_raw_state(game, transition.state)._word))
            ),
            episode_position_index=len(record),
        )
        nxt = game.step(transition.state, action)
        position.next_terminated = nxt.terminated
        record.append(position)
        transition = nxt
        return True

    for semantic_action in semantic_actions:
        if isinstance(game, SerialBraidGame):
            state = transition.state
            length = int(np.asarray(state.pgx._word).astype(bool).sum())
            destination = _best_destination(game, semantic_action, state.head, length)
            if destination is None:
                return None
            route, _, native_action = destination
            if internal_action_cap is not None and len(route) > internal_action_cap:
                return None
            option_state = state
            option_position = len(record)
            for action in [*route, native_action]:
                if not take(action):
                    return None
            record[option_position].option_state = option_state
            record[option_position].target_external_action = semantic_action
        elif not take(semantic_action):
            return None

    if not transition.terminated or not _is_solved(game, transition.state):
        return None
    raw = _raw_state(game, transition.state)
    best_residual = min(
        [position.residual_word_length for position in record]
        + [int(np.count_nonzero(np.asarray(raw._word)))]
    )
    rewards = game.final_rewards(transition.state)
    verified = verified_record_cost(game, knot, ratio, record)
    if verified is None:
        return None
    crossing_changes, semantic_moves, _ = verified
    native_plies = game.native_ply_count(transition.state)
    internal_plies = game.internal_ply_count(transition.state)
    for position in record:
        position.outcome = float(rewards[position.player])
        position.solved = 1.0
        position.final_crossing_changes = float(crossing_changes)
        position.final_moves = float(semantic_moves)
        position.final_native_plies = float(native_plies)
        position.final_internal_plies = float(internal_plies)
        position.termination_reason = "solved"
        position.best_residual_word_length = best_residual
    return record


def _attempt_payload(
    scientist: Scientist,
    knot: KnotItem,
    ratio: float,
    record: GameRecord,
) -> tuple[dict[str, Any], tuple[int, int, list[int]] | None]:
    verified = verified_record_cost(scientist.game, knot, ratio, record)
    if verified is None:
        return {"scientist": scientist.name, "ratio": ratio, "solved": False}, None
    cc, moves, semantic = verified
    return {
        "scientist": scientist.name,
        "ratio": ratio,
        "solved": True,
        "crossing_changes": cc,
        "moves": moves,
        "native_plies": (
            int(record[0].final_native_plies)
            if record and np.isfinite(record[0].final_native_plies)
            else len(record)
        ),
        "internal_plies": (
            int(record[0].final_internal_plies)
            if record and np.isfinite(record[0].final_internal_plies)
            else max(len(record) - moves, 0)
        ),
        "objective": ratio * cc + moves,
        "semantic_actions": len(semantic),
    }, verified


def _failure_cost(ratio: float, move_budget: int, failure_crossings: int = 20) -> float:
    return ratio * failure_crossings + move_budget


def _portfolio_probe(
    scientists: list[Scientist],
    items: list[BankItem],
    ratios: tuple[float, ...],
    *,
    simulations: int,
    seed: int,
    objective_budget: bool,
    remaining_budget_channel: bool,
    action_horizon: int,
    minimum_attempts: int,
    executor: ProcessPoolExecutor | None,
) -> dict[str, Any]:
    """Measure the paired population portfolio on a small frozen panel."""
    cells_per_attempt = len(scientists) * len(items) * len(ratios)
    attempts_per_cell = max(1, math.ceil(minimum_attempts / max(1, cells_per_attempt)))
    probe_ratios = tuple(
        ratio for _attempt in range(attempts_per_cell) for ratio in ratios
    )
    jobs = []
    coordinates: list[tuple[int, int]] = []
    for scientist_index, scientist in enumerate(scientists):
        for item_index, item in enumerate(items):
            seeds = tuple(
                seed
                + scientist_index * 10_000_000
                + item_index * 10_000
                + attempt_index * 100
                + ratio_index
                for attempt_index in range(attempts_per_cell)
                for ratio_index in range(len(ratios))
            )
            jobs.append(
                (
                    scientist,
                    item.knot,
                    probe_ratios,
                    simulations,
                    seeds,
                    objective_budget,
                    remaining_budget_channel,
                    action_horizon,
                )
            )
            coordinates.append((scientist_index, item_index))
    results = _play_bundles(executor, jobs)
    cells: dict[str, dict[str, Any]] = {}
    failure = {ratio: _failure_cost(ratio, action_horizon) for ratio in ratios}
    for (scientist_index, item_index), runs in zip(coordinates, results, strict=True):
        scientist = scientists[scientist_index]
        item = items[item_index]
        for flat_index, (ratio, run) in enumerate(zip(probe_ratios, runs, strict=True)):
            payload, verified = _attempt_payload(scientist, item.knot, ratio, run.record)
            payload["attempt"] = flat_index // len(ratios)
            key = f"{item.id}|{ratio:g}"
            objective = (
                ratio * verified[0] + verified[1] if verified is not None else failure[ratio]
            )
            incumbent = cells.get(key)
            if incumbent is None or objective < incumbent["objective"]:
                cells[key] = {
                    "item": item.id,
                    "ratio": ratio,
                    "solved": verified is not None,
                    "objective": objective,
                    "scientist": scientist.name,
                    "attempt": payload,
                }
    return {
        "representations": [item.id for item in items],
        "attempts": len(jobs) * len(probe_ratios),
        "attempts_per_cell": attempts_per_cell,
        "solved": sum(bool(row["solved"]) for row in cells.values()),
        "capped_objective": float(sum(row["objective"] for row in cells.values())),
        "cells": cells,
    }


def _portfolio_noninferior(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Require both coverage retention and non-increasing complete capped loss."""
    return bool(
        after["solved"] >= before["solved"]
        and after["capped_objective"] <= before["capped_objective"] + 1e-9
    )


def _degraded_representations(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    degraded = []
    for key, old in before["cells"].items():
        new = after["cells"][key]
        if (old["solved"] and not new["solved"]) or (new["objective"] > old["objective"] + 1e-9):
            degraded.append(str(old["item"]))
    return list(dict.fromkeys(degraded))


def _strict_shared_improvement(
    receiver: Scientist,
    knot: KnotItem,
    ratio: float,
    own_record: GameRecord,
    translated: GameRecord | None,
    *,
    best_native_objective: float | None = None,
) -> tuple[bool, float | None, float | None]:
    """Admit only a donation that beats the receiver's archived native best."""
    own = verified_record_cost(receiver.game, knot, ratio, own_record)
    shared = (
        verified_record_cost(receiver.game, knot, ratio, translated)
        if translated is not None
        else None
    )
    own_objective = ratio * own[0] + own[1] if own is not None else None
    if best_native_objective is not None:
        own_objective = (
            min(own_objective, best_native_objective)
            if own_objective is not None
            else best_native_objective
        )
    shared_objective = ratio * shared[0] + shared[1] if shared is not None else None
    admitted = shared_objective is not None and (
        own_objective is None or shared_objective < own_objective
    )
    return admitted, own_objective, shared_objective


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _commit_round(output: Path, index: int, event: dict[str, Any], state: dict[str, Any]) -> None:
    rounds = output / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    target = rounds / f"{index:06d}"
    if target.exists():
        raise FileExistsError(f"round already committed: {target}")
    with tempfile.TemporaryDirectory(dir=rounds, prefix=f".{index:06d}-") as raw:
        temporary = Path(raw)
        (temporary / "event.json").write_text(
            json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with gzip.open(temporary / "state.pt.gz", "wb", compresslevel=1) as handle:
            torch.save(state, handle)
        os.replace(temporary, target)


def _round_dirs(output: Path) -> list[Path]:
    rounds = output / "rounds"
    return (
        sorted(path for path in rounds.glob("[0-9][0-9][0-9][0-9][0-9][0-9]") if path.is_dir())
        if rounds.exists()
        else []
    )


def load_round_state(round_dir: Path, *, map_location: str | torch.device) -> dict[str, Any]:
    """Load a compressed transaction, retaining compatibility with pilot v1."""
    compressed = round_dir / "state.pt.gz"
    if compressed.exists():
        with gzip.open(compressed, "rb") as handle:
            return torch.load(handle, map_location=map_location, weights_only=False)
    return torch.load(round_dir / "state.pt", map_location=map_location, weights_only=False)


def _refresh_schedule(output: Path) -> list[dict[str, Any]]:
    events = [json.loads((path / "event.json").read_text()) for path in _round_dirs(output)]
    schedule = output / "schedule.jsonl"
    with tempfile.NamedTemporaryFile("w", dir=output, delete=False) as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, schedule)
    return events


def _scientist_state(
    scientist: Scientist,
    option_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    state = {
        "network": scientist.network.state_dict(),
        "optimizer": scientist.optimizer.state_dict(),
        "replay": scientist.replay,
        "prediction_source": scientist.prediction_source,
    }
    if option_optimizer is not None:
        state["option_optimizer"] = option_optimizer.state_dict()
    return state


def _restore_scientist(
    scientist: Scientist,
    state: dict[str, Any],
    option_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    load_policy_value_state_dict(scientist.network, state["network"])
    scientist.optimizer.load_state_dict(state["optimizer"])
    if option_optimizer is not None:
        option_optimizer.load_state_dict(state["option_optimizer"])
    scientist.replay = state["replay"]
    scientist.replay._ensure_replay_state()
    scientist.replay.representation_capacity = 100
    scientist.prediction_source = state["prediction_source"]


def _native_preservation_positions(scientist: Scientist, count: int) -> list[Position]:
    """Select deterministic native states on which sharing should stay inactive."""
    positions = [
        position
        for record in scientist.replay.games
        if record and not bool(getattr(record[0], "shared_witness", False))
        for position in record
    ]
    if count <= 0 or not positions:
        return []
    if len(positions) <= count:
        return positions
    indices = np.linspace(0, len(positions) - 1, count, dtype=int)
    return [positions[int(index)] for index in indices]


def _manifest(
    checkpoints: dict[str, Path],
    *,
    arm: Arm,
    bank: list[BankItem],
    anchors: list[BankItem],
    ratios: tuple[float, ...],
    frontier: int,
    qualification_simulations: int,
    qualification_attempts: int,
    simulations: int,
    full_attempts_per_scientist: int,
    train_every: int,
    train_steps: int,
    adaptive_rehearsal: bool,
    rehearsal_games_per_block: int,
    max_rehearsal_games_per_block: int,
    retention_attempts: int,
    retention_simulations: int,
    direct_shared_fraction: float,
    attempt_workers: int,
    objective_budget: bool,
    remaining_budget_channel: bool,
    native_action_horizon: int,
    assessor_gate: Path | None,
    input_bank: Path | None,
    input_anchor_bank: Path | None,
    bank_seed: int,
    seed: int,
) -> dict[str, Any]:
    from pgx_mcts_bench.foundation_pretraining import source_provenance

    protocol = {
        "schema": "collaborative-scientists-v9-direct-sharing",
        "source_provenance": source_provenance(),
        "arm": arm,
        "schedule": (
            "adaptive-neural-priority"
            if arm.startswith("adaptive") or arm == "solo-compute-matched"
            else "static-cheap-score"
        ),
        "assessor_gate": (
            {"path": str(assessor_gate.resolve()), "sha256": _sha256(assessor_gate)}
            if assessor_gate is not None
            else None
        ),
        "adaptive_assessor": {
            "scoring_rule": ASSESSOR_SCORING_RULE,
            "read_only_head_sweep_steps": ASSESSOR_SCAN_STEPS,
            "aggregation": "mean predictions across deterministic cyclic head views",
            "semantic_actions": 0,
            "solver_episode_budget_charged": False,
        },
        "ratios": list(ratios),
        "frontier": frontier,
        "qualification_simulations": qualification_simulations,
        "qualification_attempts_per_scientist": qualification_attempts,
        "simulations": simulations,
        "full_attempts_per_scientist": full_attempts_per_scientist,
        "train_every": train_every,
        "train_steps": train_steps,
        "adaptive_rehearsal": {
            "enabled": adaptive_rehearsal,
            "initial_games_per_block": rehearsal_games_per_block,
            "maximum_games_per_block": max_rehearsal_games_per_block,
            "dose_rule": "hold-if-healthy-else-double",
            "allocation": {
                "degraded": 0.5,
                "recent": 0.25,
                "exposure_balanced": 0.25,
            },
            "old_solution_source": (
                "permanent best-native bank plus one fresh MCTS attempt per scheduled game"
            ),
            "retention_attempts": retention_attempts,
            "retention_simulations": retention_simulations,
            "acceptance": "solved-nondecreasing-and-capped-objective-nonincreasing",
            "rollback": "network-and-optimizers-only; replay-and-solution-bank-survive",
        },
        "attempt_workers": attempt_workers,
        "objective_budget": objective_budget,
        "remaining_budget_channel": remaining_budget_channel,
        "input_banks": {
            "base": (
                {"path": str(input_bank.resolve()), "sha256": _sha256(input_bank)}
                if input_bank is not None
                else None
            ),
            "anchor": (
                {
                    "path": str(input_anchor_bank.resolve()),
                    "sha256": _sha256(input_anchor_bank),
                }
                if input_anchor_bank is not None
                else None
            ),
        },
        "objective_budget_encoding": "remaining-semantic-L-absolute-global-v3",
        "objective_budget_attempt_protocol": {
            "first_cap": (
                "min(global_cap, A_over_B * ceil(observed_word_intersections / 2) "
                "+ native_action_horizon)"
            ),
            "scientist_prediction_used": False,
            "restart": (
                "every objective-censored failure restarts with the same seed at "
                "the common global cap"
            ),
            "ordinary_horizon_failure_restarted": False,
        },
        "objective_censored_replay": "solve-negative-v2",
        "collaboration_replay_sampling": "exposure-and-representation-balanced-v3",
        "collaboration_replay_history_representations": 100,
        "collaboration_replay_representation_mix": {
            "current": 0.25,
            "structurally_similar": 0.25,
            "global": 0.5,
        },
        "collaboration_replay_similarity": "cheap-braid-structure-cosine-v1",
        "collaboration_replay_positions_per_episode": 4,
        "collaboration_replay_outcome_mix": "native-success-failure-50-50-when-available",
        "policy_value_update": "success-only-native-plus-strictly-better-direct-sharing-v5",
        "direct_distillation": {
            "enabled": arm in DIRECT_SHARING_ARMS,
            "shared_success_fraction": direct_shared_fraction,
            "target": "strictly-better-receiver-native-semantic-witness",
            "optimizer": "native-policy-value",
            "admission": "transactional-complete-portfolio-guard",
        },
        "sharing_policy_adapter": {
            "enabled": False,
            "reason": (
                "paper arms train eligible receiver-native trajectories through the ordinary "
                "policy; there is no permanent sharing-only policy path"
            ),
        },
        "policy_value_preservation": "frozen-starting-scientist-when-weight-positive-v1",
        "shared_witness_internal_action_cap_per_edit": 5,
        "shared_policy_eligibility": (
            "strictly lower semantic objective than best archived native solution, "
            "rechecked before every distillation event and ratio-specific"
        ),
        "stale_shared_witness_use": "critic-upper-bound-only",
        "shared_policy_target": (
            "deterministic shortest neutral receiver route then certified edit, "
            "within 5 internal actions"
        ),
        "solution_definition": {
            "terminal": "exact replay reaches empty braid word and one strand",
            "moves": "verified portable semantic witness steps only",
            "native_plies": "all receiver actions, reported outside L_A:B",
            "internal_plies": (
                "head, tape, register, paint, and memory actions; controller-budgeted "
                "but excluded from L_A:B"
            ),
            "learned_objective_cap": "never used as an attempt cap",
            "crossing_change_cap": None,
            "native_action_horizon": native_action_horizon,
            "completeness_claim": False,
            "timeout_interpretation": "censored beyond the declared action horizon",
        },
        "solve_encoder_gradients": "solve-only-v1",
        "budget_monotonic_training": "paired-margin-v1",
        "bank_seed": bank_seed,
        "seed": seed,
        "checkpoints": [
            {"name": name, "path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in checkpoints.items()
        ],
        "bank_sha256": _json_hash(_bank_payload(bank)),
        "anchor_sha256": _json_hash(_bank_payload(anchors)),
    }
    for label, items in (("bank", bank), ("anchor", anchors)):
        if any(item.certified_unknotting_lower_bound is not None for item in items):
            protocol[f"{label}_unknotting"] = {
                "published_exact": dict(
                    sorted(
                        Counter(
                            item.known_unknotting_number
                            for item in items
                            if item.known_unknotting_number is not None
                        ).items()
                    )
                ),
                "certified_lower": dict(
                    sorted(
                        Counter(
                            item.certified_unknotting_lower_bound
                            for item in items
                            if item.certified_unknotting_lower_bound is not None
                        ).items()
                    )
                ),
            }
    protocol["protocol_sha256"] = _json_hash(protocol)
    return protocol


def run_collaborative_scientists(
    checkpoints: dict[str, Path],
    output: Path,
    *,
    arm: Arm,
    rounds: int = 200,
    pool_size: int = 200,
    anchor_size: int = 70,
    frontier: int = 100,
    ratios: tuple[float, ...] = (10.0, 1000.0),
    qualification_simulations: int = 16,
    qualification_attempts: int = 1,
    simulations: int = 128,
    full_attempts_per_scientist: int = 1,
    train_every: int = 10,
    train_steps: int = 32,
    batch_size: int = 32,
    adaptive_rehearsal: bool = False,
    rehearsal_games_per_block: int = 8,
    max_rehearsal_games_per_block: int = 32,
    retention_attempts: int = 24,
    retention_simulations: int = 64,
    direct_shared_fraction: float = 0.05,
    attempt_workers: int = 1,
    objective_budget: bool = False,
    remaining_budget_channel: bool = False,
    action_horizon: int | None = None,
    assessor_gate: Path | None = None,
    input_bank: Path | None = None,
    input_anchor_bank: Path | None = None,
    bank_seed: int = 0,
    seed: int = 0,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    if arm not in {
        "adaptive-sharing",
        "adaptive-sharing-direct",
        "adaptive-sharing-aux-only",
        "adaptive-no-sharing",
        "static-sharing",
        "static-no-sharing",
        "solo-compute-matched",
    }:
        raise ValueError(f"unknown arm: {arm}")
    if not checkpoints:
        raise ValueError("at least one scientist is required")
    if arm == "solo-compute-matched" and len(checkpoints) != 1:
        raise ValueError("solo-compute-matched requires exactly one scientist")
    if attempt_workers < 1:
        raise ValueError("attempt_workers must be positive")
    if qualification_attempts < 1 or full_attempts_per_scientist < 1:
        raise ValueError("qualification and full attempts must be positive")
    if train_steps < 0:
        raise ValueError("train_steps must be non-negative")
    if rehearsal_games_per_block < 1:
        raise ValueError("rehearsal games per block must be positive")
    if max_rehearsal_games_per_block < rehearsal_games_per_block:
        raise ValueError("maximum rehearsal games must be at least the initial dose")
    if retention_attempts < 1 or retention_simulations < 1:
        raise ValueError("retention attempts and simulations must be positive")
    if not 0.0 <= direct_shared_fraction <= 0.5:
        raise ValueError("direct shared fraction must be in 0..0.5")
    if (input_bank is None) != (input_anchor_bank is None):
        raise ValueError("input_bank and input_anchor_bank must be supplied together")
    adaptive_schedule = arm.startswith("adaptive") or arm == "solo-compute-matched"
    if adaptive_schedule:
        from pgx_mcts_bench.assessor_gate import validate_assessor_gate

        validate_assessor_gate(assessor_gate, checkpoints)
    # A hard objective cap necessarily needs the remaining-L observation.  The
    # converse is intentionally false: new experiments expose the feature while
    # treating the action horizon, rather than a predicted L, as the only cap.
    remaining_budget_channel = remaining_budget_channel or objective_budget
    scientists = [
        load_scientist(
            name,
            path,
            seed=seed + index * 10_000,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=remaining_budget_channel,
        )
        for index, (name, path) in enumerate(checkpoints.items())
    ]
    if action_horizon is not None:
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        for scientist in scientists:
            scientist.config = replace(
                scientist.config,
                game=replace(scientist.config.game, simplify_budget=action_horizon),
            )
            scientist.game = make_game(scientist.config.game)
    action_horizons = {int(scientist.config.game.simplify_budget) for scientist in scientists}
    if len(action_horizons) != 1:
        raise ValueError(
            "collaboration scientists must use one common native action horizon; "
            f"got {sorted(action_horizons)}"
        )
    native_action_horizon = action_horizons.pop()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    bank_path = output / "base.json"
    anchor_path = output / "new-70.json"

    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; pass resume=True")
        manifest = json.loads(manifest_path.read_text())
        bank = _bank_from_payload(json.loads(bank_path.read_text()))
        anchors = _bank_from_payload(json.loads(anchor_path.read_text()))
        expected = _manifest(
            checkpoints,
            arm=arm,
            bank=bank,
            anchors=anchors,
            ratios=ratios,
            frontier=frontier,
            qualification_simulations=qualification_simulations,
            qualification_attempts=qualification_attempts,
            simulations=simulations,
            full_attempts_per_scientist=full_attempts_per_scientist,
            train_every=train_every,
            train_steps=train_steps,
            adaptive_rehearsal=adaptive_rehearsal,
            rehearsal_games_per_block=rehearsal_games_per_block,
            max_rehearsal_games_per_block=max_rehearsal_games_per_block,
            retention_attempts=retention_attempts,
            retention_simulations=retention_simulations,
            direct_shared_fraction=direct_shared_fraction,
            attempt_workers=attempt_workers,
            objective_budget=objective_budget,
            remaining_budget_channel=remaining_budget_channel,
            native_action_horizon=native_action_horizon,
            assessor_gate=assessor_gate,
            input_bank=input_bank,
            input_anchor_bank=input_anchor_bank,
            bank_seed=bank_seed,
            seed=seed,
        )
        if manifest["protocol_sha256"] != expected["protocol_sha256"]:
            raise ValueError("resume protocol differs from the frozen manifest")
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume without {manifest_path}")
        if input_bank is not None and input_anchor_bank is not None:
            bank = _bank_from_payload(json.loads(input_bank.read_text()))
            anchors = _bank_from_payload(json.loads(input_anchor_bank.read_text()))
            bank_ids = [item.id for item in bank]
            anchor_ids = [item.id for item in anchors]
            if len(bank_ids) != len(set(bank_ids)) or len(anchor_ids) != len(set(anchor_ids)):
                raise ValueError("input banks contain duplicate representation identities")
            if set(bank_ids) & set(anchor_ids):
                raise ValueError("input base and anchor banks must be identity-disjoint")
        else:
            bank, anchors = stratified_banks(pool_size, anchor_size, bank_seed)
        manifest = _manifest(
            checkpoints,
            arm=arm,
            bank=bank,
            anchors=anchors,
            ratios=ratios,
            frontier=frontier,
            qualification_simulations=qualification_simulations,
            qualification_attempts=qualification_attempts,
            simulations=simulations,
            full_attempts_per_scientist=full_attempts_per_scientist,
            train_every=train_every,
            train_steps=train_steps,
            adaptive_rehearsal=adaptive_rehearsal,
            rehearsal_games_per_block=rehearsal_games_per_block,
            max_rehearsal_games_per_block=max_rehearsal_games_per_block,
            retention_attempts=retention_attempts,
            retention_simulations=retention_simulations,
            direct_shared_fraction=direct_shared_fraction,
            attempt_workers=attempt_workers,
            objective_budget=objective_budget,
            remaining_budget_channel=remaining_budget_channel,
            native_action_horizon=native_action_horizon,
            assessor_gate=assessor_gate,
            input_bank=input_bank,
            input_anchor_bank=input_anchor_bank,
            bank_seed=bank_seed,
            seed=seed,
        )
        _atomic_json(bank_path, _bank_payload(bank))
        _atomic_json(anchor_path, _bank_payload(anchors))
        _atomic_json(manifest_path, manifest)

    for scientist in scientists:
        scientist.network.shared_auxiliary_only = arm == "adaptive-sharing-aux-only"
        attach_policy_value_preservation_teacher(scientist.network)
    option_optimizers: dict[str, torch.optim.Optimizer] = {}
    completed = _round_dirs(output)
    if completed:
        saved = load_round_state(completed[-1], map_location=device)
        for scientist in scientists:
            _restore_scientist(
                scientist,
                saved["scientists"][scientist.name],
                option_optimizers.get(scientist.name),
            )
        active_ids = list(saved["active_ids"])
        cursor = int(saved["cursor"])
        processed = list(saved["processed"])
        rehearsal_dose = int(saved.get("rehearsal_dose", rehearsal_games_per_block))
        rehearsal_exposures = {
            str(key): int(value) for key, value in saved.get("rehearsal_exposures", {}).items()
        }
        rehearsal_priority_ids = [str(value) for value in saved.get("rehearsal_priority_ids", [])]
    else:
        active_ids = [item.id for item in bank[: min(frontier, len(bank))]]
        cursor = len(active_ids)
        processed = []
        rehearsal_dose = rehearsal_games_per_block
        rehearsal_exposures: dict[str, int] = {}
        rehearsal_priority_ids: list[str] = []

    by_id = {item.id: item for item in bank}
    target_rounds = min(rounds, len(bank))
    executor = ProcessPoolExecutor(max_workers=attempt_workers) if attempt_workers > 1 else None
    for round_index in range(len(processed), target_rounds):
        active = [by_id[item_id] for item_id in active_ids]
        score_rows = [expected_capped_scores(scientist, active, ratios) for scientist in scientists]
        adaptive_schedule = arm.startswith("adaptive") or arm == "solo-compute-matched"
        if adaptive_schedule:
            proposals = [int(np.argmin(scores)) for scores in score_rows]
        else:
            static_index = min(
                range(len(active)), key=lambda i: (active[i].cheap_score, active[i].id)
            )
            proposals = [static_index] * len(scientists)

        qualification_jobs = []
        qualification_ratios = tuple(
            ratio for _attempt in range(qualification_attempts) for ratio in ratios
        )
        for scientist_index, (scientist, proposal_index) in enumerate(
            zip(scientists, proposals, strict=True)
        ):
            seeds = tuple(
                seed
                + round_index * 1_000_000
                + scientist_index * 10_000
                + attempt_index * 100
                + ratio_index
                for attempt_index in range(qualification_attempts)
                for ratio_index in range(len(ratios))
            )
            qualification_jobs.append(
                (
                    scientist,
                    active[proposal_index].knot,
                    qualification_ratios,
                    qualification_simulations,
                    seeds,
                    objective_budget,
                    remaining_budget_channel,
                    native_action_horizon,
                )
            )
        qualification_records = _play_bundles(executor, qualification_jobs)
        qualification: list[dict[str, Any]] = []
        proposal_totals: list[tuple[float, float, int]] = []
        for scientist_index, (scientist, proposal_index) in enumerate(
            zip(scientists, proposals, strict=True)
        ):
            item = active[proposal_index]
            total = 0.0
            rows = []
            for flat_index, (ratio, run) in enumerate(
                zip(
                    qualification_ratios,
                    qualification_records[scientist_index],
                    strict=True,
                )
            ):
                record = run.record
                payload, verified = _attempt_payload(scientist, item.knot, ratio, record)
                payload["attempt"] = flat_index // len(ratios)
                if run.budget is not None:
                    payload["budget"] = run.budget
                rows.append(payload)
                total += (
                    ratio * verified[0] + verified[1]
                    if verified is not None
                    else _failure_cost(ratio, scientist.config.game.simplify_budget)
                ) / _failure_cost(ratio, scientist.config.game.simplify_budget)
            qualification.append({"scientist": scientist.name, "task": item.id, "attempts": rows})
            proposal_totals.append(
                (
                    total / qualification_attempts,
                    float(score_rows[scientist_index][proposal_index]),
                    scientist_index,
                )
            )

        if adaptive_schedule:
            _, _, selecting_scientist = min(proposal_totals)
            selected_index = proposals[selecting_scientist]
        else:
            selecting_scientist = -1
            selected_index = proposals[0]
        selected = active[selected_index]
        replay_embedding = _replay_representation_embedding(selected.knot)
        for scientist in scientists:
            scientist.replay.set_representation_embedding(selected.id, replay_embedding)
        selected_predictions = {
            scientist.name: prediction_details(scientist, selected, ratios)
            for scientist in scientists
        }

        full_jobs = []
        full_ratios = tuple(
            ratio for _attempt in range(full_attempts_per_scientist) for ratio in ratios
        )
        for scientist_index, scientist in enumerate(scientists):
            seeds = tuple(
                seed
                + 500_000_000
                + round_index * 1_000_000
                + scientist_index * 10_000
                + attempt_index * 100
                + ratio_index
                for attempt_index in range(full_attempts_per_scientist)
                for ratio_index in range(len(ratios))
            )
            full_jobs.append(
                (
                    scientist,
                    selected.knot,
                    full_ratios,
                    simulations,
                    seeds,
                    objective_budget,
                    remaining_budget_channel,
                    native_action_horizon,
                )
            )
        full_records = _play_bundles(executor, full_jobs)
        full_attempts: list[dict[str, Any]] = []
        winners: dict[float, tuple[int, GameRecord, tuple[int, int, list[int]]]] = {}
        native_best_records: dict[tuple[int, int], tuple[GameRecord, float | None]] = {}
        for scientist_index, scientist in enumerate(scientists):
            for flat_index, (ratio, run) in enumerate(
                zip(full_ratios, full_records[scientist_index], strict=True)
            ):
                ratio_index = flat_index % len(ratios)
                record = run.record
                records_for_replay = run.replay_records or (record,)
                for replay_record in records_for_replay:
                    scientist.replay.add(
                        replay_record,
                        representation_id=selected.id,
                        objective_ratio=ratio,
                    )
                payload, verified = _attempt_payload(scientist, selected.knot, ratio, record)
                payload["attempt"] = flat_index // len(ratios)
                if run.budget is not None:
                    payload["budget"] = run.budget
                full_attempts.append(payload)
                objective = ratio * verified[0] + verified[1] if verified is not None else None
                native_key = (scientist_index, ratio_index)
                native_incumbent = native_best_records.get(native_key)
                if native_incumbent is None or (
                    objective is not None
                    and (native_incumbent[1] is None or objective < native_incumbent[1])
                ):
                    native_best_records[native_key] = (record, objective)
                if verified is not None:
                    incumbent = winners.get(ratio)
                    if incumbent is None or (
                        ratio * verified[0] + verified[1],
                        verified[0],
                        verified[1],
                        scientist_index,
                    ) < (
                        ratio * incumbent[2][0] + incumbent[2][1],
                        incumbent[2][0],
                        incumbent[2][1],
                        incumbent[0],
                    ):
                        winners[ratio] = (scientist_index, record, verified)

        translations: list[dict[str, Any]] = []
        if arm in SHARING_ARMS:
            for ratio_index, ratio in enumerate(ratios):
                winner = winners.get(ratio)
                if winner is None:
                    continue
                author, _, verified = winner
                for receiver_index, receiver in enumerate(scientists):
                    if receiver_index == author:
                        continue
                    translated = translate_semantic_record(
                        receiver,
                        selected.knot,
                        ratio,
                        verified[2],
                        seed=seed
                        + 900_000_000
                        + round_index * 10_000
                        + ratio_index * 100
                        + receiver_index,
                        internal_action_cap=5,
                    )
                    success = translated is not None
                    admitted, own_objective, shared_objective = _strict_shared_improvement(
                        receiver,
                        selected.knot,
                        ratio,
                        native_best_records[(receiver_index, ratio_index)][0],
                        translated,
                        best_native_objective=receiver.replay.best_native_objective(
                            selected.id, ratio
                        ),
                    )
                    if admitted:
                        receiver.replay.add(
                            translated,
                            representation_id=selected.id,
                            objective_ratio=ratio,
                        )
                    translations.append(
                        {
                            "ratio": ratio,
                            "author": scientists[author].name,
                            "receiver": receiver.name,
                            "translated": success,
                            "admitted": admitted,
                            "receiver_previous_objective": own_objective,
                            "receiver_shared_objective": shared_objective,
                            "receiver_semantic_moves": (
                                int(translated[0].final_moves) if translated else None
                            ),
                            "receiver_native_plies": (
                                int(translated[0].final_native_plies) if translated else None
                            ),
                            "receiver_internal_plies": (
                                int(translated[0].final_internal_plies) if translated else None
                            ),
                        }
                    )

        losses: dict[str, dict[str, float] | None] = {}
        rehearsal_event: dict[str, Any] | None = None
        portfolio_guard: dict[str, Any] | None = None
        if (round_index + 1) % train_every == 0:
            training_snapshot = _training_snapshot(scientists, option_optimizers)
            retention_panel: list[BankItem] = []
            portfolio_before: dict[str, Any] | None = None
            if adaptive_rehearsal and processed:
                rehearsal_schedule = _rehearsal_ids(
                    processed,
                    rehearsal_priority_ids,
                    rehearsal_exposures,
                    rehearsal_dose,
                )
                rehearsal_jobs = []
                rehearsal_coordinates: list[tuple[int, str, float]] = []
                archived_rehearsals = 0
                for scientist_index, scientist in enumerate(scientists):
                    for game_index, item_id in enumerate(rehearsal_schedule):
                        ratio = ratios[(round_index + game_index) % len(ratios)]
                        archived = scientist.replay.best_native_solution_record(
                            item_id, ratio
                        )
                        if archived is not None:
                            scientist.replay.add(
                                archived,
                                representation_id=item_id,
                                objective_ratio=ratio,
                            )
                            archived_rehearsals += 1
                        rehearsal_jobs.append(
                            (
                                scientist,
                                by_id[item_id].knot,
                                (ratio,),
                                simulations,
                                (
                                    seed
                                    + 800_000_000
                                    + round_index * 1_000_000
                                    + scientist_index * 10_000
                                    + game_index,
                                ),
                                objective_budget,
                                remaining_budget_channel,
                                native_action_horizon,
                            )
                        )
                        rehearsal_coordinates.append((scientist_index, item_id, ratio))
                rehearsal_runs = _play_bundles(executor, rehearsal_jobs)
                rehearsal_solved = 0
                for (scientist_index, item_id, ratio), runs in zip(
                    rehearsal_coordinates, rehearsal_runs, strict=True
                ):
                    scientist = scientists[scientist_index]
                    run = runs[0]
                    records_for_replay = run.replay_records or (run.record,)
                    for replay_record in records_for_replay:
                        scientist.replay.add(
                            replay_record,
                            representation_id=item_id,
                            objective_ratio=ratio,
                        )
                    rehearsal_solved += int(
                        verified_record_cost(scientist.game, by_id[item_id].knot, ratio, run.record)
                        is not None
                    )
                panel_items = max(
                    1,
                    math.ceil(retention_attempts / (len(scientists) * len(ratios))),
                )
                seen_items = [by_id[item_id] for item_id in [*processed, selected.id]]
                retention_panel = _rotating_retention_items(
                    seen_items,
                    panel_items,
                    seed=seed,
                    block=(round_index + 1) // train_every,
                )
                probe_seed = seed + 850_000_000 + round_index * 1_000_000
                portfolio_before = _portfolio_probe(
                    scientists,
                    retention_panel,
                    ratios,
                    simulations=retention_simulations,
                    seed=probe_seed,
                    objective_budget=objective_budget,
                    remaining_budget_channel=remaining_budget_channel,
                    action_horizon=native_action_horizon,
                    minimum_attempts=retention_attempts,
                    executor=executor,
                )
                rehearsal_event = {
                    "dose": rehearsal_dose,
                    "scheduled_ids": rehearsal_schedule,
                    "attempts": len(rehearsal_jobs),
                    "solved": rehearsal_solved,
                    "archived_best_solutions_replayed": archived_rehearsals,
                    "exposures": dict(rehearsal_exposures),
                }
            for scientist in scientists:
                last = None
                matched_update_steps = math.ceil(train_steps / 4)
                if scientist.replay.has_trainable_collaboration_positions():
                    for _ in range(train_steps):
                        last = train_alphazero_step(
                            scientist.network,
                            scientist.optimizer,
                            scientist.replay,
                            batch_size,
                            torch.device(device),
                            collaboration_replay=True,
                            shared_fraction=(
                                direct_shared_fraction
                                if arm in DIRECT_SHARING_ARMS
                                else 0.1
                                if arm == "adaptive-sharing-aux-only"
                                else 0.0
                            ),
                            policy_value_success_only=True,
                            replay_current_representation=selected.id,
                            replay_current_fraction=0.25,
                            replay_similar_fraction=0.25,
                            replay_positions_per_episode=4,
                        )
                active_option_records = scientist.replay.active_distillation_records()
                option_positions = [
                    position
                    for game_record in active_option_records
                    for position in game_record
                    if position.option_state is not None and position.target_external_action >= 0
                ]
                if (
                    scientist.name in option_optimizers
                    and isinstance(scientist.game, SerialBraidGame)
                    and option_positions
                    and matched_update_steps
                ):
                    preservation_positions = _native_preservation_positions(scientist, batch_size)
                    option_updates: list[dict[str, float | str]] = []
                    option_losses: list[float] = []
                    for _ in range(matched_update_steps):
                        option_diagnostics: dict[str, float | str] = {}
                        option_losses.append(
                            train_bounded_option_step(
                                scientist.network,
                                option_optimizers[scientist.name],
                                scientist.game,
                                scientist.replay,
                                scientist.replay.rng,
                                batch_size=min(batch_size, len(option_positions)),
                                horizon=5,
                                beam_width=8,
                                device=torch.device(device),
                                adapter_only=True,
                                stable_routes=True,
                                positions=option_positions,
                                preservation_positions=preservation_positions,
                                route_gate_weight=0.1,
                                off_route_kl_weight=1.0,
                                off_route_gate_weight=0.1,
                                diagnostics=option_diagnostics,
                            )
                        )
                        option_updates.append(option_diagnostics)
                    if last is None:
                        last = {}
                    last["bounded_option"] = option_losses[-1]
                    last["bounded_option_updates"] = option_updates
                    last["matched_extra_updates"] = matched_update_steps
                    last["matched_extra_update"] = "bounded-option-with-gate"
                elif scientist.replay.games and matched_update_steps:
                    for _ in range(matched_update_steps):
                        last = train_alphazero_step(
                            scientist.network,
                            scientist.optimizer,
                            scientist.replay,
                            batch_size,
                            torch.device(device),
                            collaboration_replay=True,
                            shared_fraction=0.0,
                            policy_value_success_only=True,
                            replay_current_representation=selected.id,
                            replay_current_fraction=0.25,
                            replay_similar_fraction=0.25,
                            replay_positions_per_episode=4,
                        )
                    assert last is not None
                    last["bounded_option"] = 0.0
                    last["matched_extra_updates"] = matched_update_steps
                    last["matched_extra_update"] = "native-control"
                losses[scientist.name] = last
                if last is not None and scientist.prediction_source == "legacy_proxy":
                    scientist.prediction_source = "factorized_collaboration"
            if portfolio_before is not None:
                probe_seed = seed + 850_000_000 + round_index * 1_000_000
                portfolio_after = _portfolio_probe(
                    scientists,
                    retention_panel,
                    ratios,
                    simulations=retention_simulations,
                    seed=probe_seed,
                    objective_budget=objective_budget,
                    remaining_budget_channel=remaining_budget_channel,
                    action_horizon=native_action_horizon,
                    minimum_attempts=retention_attempts,
                    executor=executor,
                )
                accepted = _portfolio_noninferior(portfolio_before, portfolio_after)
                degraded = _degraded_representations(portfolio_before, portfolio_after)
                dose_before = rehearsal_dose
                rehearsal_dose = _adaptive_dose(
                    rehearsal_dose,
                    accepted,
                    max_rehearsal_games_per_block,
                )
                if not accepted:
                    _restore_training_snapshot(scientists, option_optimizers, training_snapshot)
                    rehearsal_priority_ids = list(
                        dict.fromkeys([*degraded, *rehearsal_priority_ids])
                    )
                portfolio_guard = {
                    "accepted": accepted,
                    "before": portfolio_before,
                    "after": portfolio_after,
                    "degraded_representations": degraded,
                    "rolled_back": not accepted,
                    "dose_before": dose_before,
                    "dose_after": rehearsal_dose,
                }

        active_ids.remove(selected.id)
        processed.append(selected.id)
        if cursor < len(bank):
            active_ids.append(bank[cursor].id)
            cursor += 1
        event = {
            "round": round_index,
            "arm": arm,
            "selected": selected.id,
            "selected_scientist": (
                scientists[selecting_scientist].name if selecting_scientist >= 0 else "static"
            ),
            "proposals": [
                {
                    "scientist": scientist.name,
                    "task": active[proposal].id,
                    "predicted_normalized_capped_loss": float(score_rows[index][proposal]),
                }
                for index, (scientist, proposal) in enumerate(
                    zip(scientists, proposals, strict=True)
                )
            ],
            "selected_predictions": selected_predictions,
            "qualification": qualification,
            "full_attempts": full_attempts,
            "winners": {
                str(ratio): {
                    "author": scientists[value[0]].name,
                    "crossing_changes": value[2][0],
                    "moves": value[2][1],
                }
                for ratio, value in winners.items()
            },
            "translations": translations,
            "training": losses,
            "adaptive_rehearsal": rehearsal_event,
            "portfolio_guard": portfolio_guard,
        }
        state = {
            "active_ids": active_ids,
            "cursor": cursor,
            "processed": processed,
            "rehearsal_dose": rehearsal_dose,
            "rehearsal_exposures": rehearsal_exposures,
            "rehearsal_priority_ids": rehearsal_priority_ids,
            "scientists": {
                scientist.name: _scientist_state(scientist, option_optimizers.get(scientist.name))
                for scientist in scientists
            },
        }
        _commit_round(output, round_index, event, state)
        _refresh_schedule(output)

    if executor is not None:
        executor.shutdown()

    events = _refresh_schedule(output)
    report = {
        **manifest,
        "completed_rounds": len(events),
        "requested_rounds": target_rounds,
        "shared_winners": sum(bool(event["winners"]) for event in events),
        "translations": {
            "attempted": sum(len(event["translations"]) for event in events),
            "succeeded": sum(
                row["translated"] for event in events for row in event["translations"]
            ),
            "admitted": sum(
                row.get("admitted", row["translated"])
                for event in events
                for row in event["translations"]
            ),
        },
        "network_evaluations_proxy": {
            "qualification_simulations": qualification_simulations
            * len(scientists)
            * len(ratios)
            * qualification_attempts
            * len(events),
            "full_simulations": simulations
            * len(scientists)
            * len(ratios)
            * full_attempts_per_scientist
            * len(events),
            "rehearsal_simulations": simulations
            * sum(
                int((event.get("adaptive_rehearsal") or {}).get("attempts", 0)) for event in events
            ),
            "retention_simulations": retention_simulations
            * sum(
                2 * int((event.get("portfolio_guard") or {}).get("before", {}).get("attempts", 0))
                for event in events
            ),
        },
        "adaptive_rehearsal_result": {
            "final_dose": rehearsal_dose,
            "accepted_blocks": sum(
                bool((event.get("portfolio_guard") or {}).get("accepted")) for event in events
            ),
            "rolled_back_blocks": sum(
                bool((event.get("portfolio_guard") or {}).get("rolled_back")) for event in events
            ),
        },
    }
    _atomic_json(output / "report.json", report)
    return report
