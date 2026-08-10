"""Registered adaptive contract for the semantic-move-v1 scientist roster."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

from pgx_mcts_bench.ladder import Candidate, vnext_arms


@dataclass(frozen=True)
class AdaptiveLevels:
    native_cycles: tuple[int, ...] = (5, 8, 12, 16)
    old_cycles: tuple[int, ...] = (1, 2, 4, 8)
    simulations: tuple[int, ...] = (64, 128, 256, 512)
    donation_fraction: tuple[float, ...] = (0.0, 0.10, 0.20, 1 / 3)
    acquisition_target: float = 0.80
    evaluation_target: float = 0.70
    retention_target: float = 0.80


@dataclass(frozen=True)
class AdaptiveState:
    native_index: int = 0
    old_index: int = 0
    simulation_index: int = 0
    donation_index: int = 0


@dataclass(frozen=True)
class BlockMetrics:
    acquisition_rate: float
    evaluation_rate: float
    retention_rate: float
    eligible_donations: int
    solved_before: frozenset[str]
    solved_after: frozenset[str]
    capped_cost_before: float
    capped_cost_after: float


@dataclass(frozen=True)
class BlockDecision:
    accept: bool
    state: AdaptiveState
    lost: tuple[str, ...]
    gained: tuple[str, ...]
    reason: str


DEFAULT_LEVELS = AdaptiveLevels()


def semantic_cost(crossing_changes: int, semantic_moves: int, ratio: float) -> float:
    if crossing_changes < 0 or semantic_moves < 0 or ratio <= 0:
        raise ValueError("cost inputs must be non-negative and ratio must be positive")
    return ratio * crossing_changes + semantic_moves


def donation_is_eligible(
    *,
    donor_cost: float,
    receiver_native_cost: float | None,
    donor_verified: bool,
    same_representation: bool,
    same_objective: bool,
) -> bool:
    """Only a verified, ratio-matched strict improvement may train policy."""
    if not donor_verified or not same_representation or not same_objective:
        return False
    return receiver_native_cost is None or donor_cost < receiver_native_cost


def _raise(index: int, levels: tuple[object, ...]) -> int:
    return min(index + 1, len(levels) - 1)


def _lower(index: int) -> int:
    return max(index - 1, 0)


def decide_block(
    state: AdaptiveState,
    metrics: BlockMetrics,
    levels: AdaptiveLevels = DEFAULT_LEVELS,
) -> BlockDecision:
    """Accept by portfolio progress and adapt the next compute block.

    A local canary may move in either direction. The hard safety condition is the
    complete paired portfolio: solved-set size may not shrink and capped objective
    may not increase.
    """
    lost = tuple(sorted(metrics.solved_before - metrics.solved_after))
    gained = tuple(sorted(metrics.solved_after - metrics.solved_before))
    accept = (
        len(metrics.solved_after) >= len(metrics.solved_before)
        and metrics.capped_cost_after <= metrics.capped_cost_before
    )

    next_state = state
    if metrics.acquisition_rate < levels.acquisition_target:
        next_state = replace(
            next_state,
            native_index=_raise(next_state.native_index, levels.native_cycles),
        )
    if metrics.evaluation_rate < levels.evaluation_target:
        next_state = replace(
            next_state,
            simulation_index=_raise(next_state.simulation_index, levels.simulations),
        )
    if metrics.retention_rate < levels.retention_target or not accept:
        next_state = replace(
            next_state,
            old_index=_raise(next_state.old_index, levels.old_cycles),
            donation_index=_lower(next_state.donation_index),
        )
    elif metrics.eligible_donations >= 3:
        next_state = replace(
            next_state,
            donation_index=_raise(next_state.donation_index, levels.donation_fraction),
        )

    reason = "accepted portfolio progress" if accept else "rollback portfolio regression"
    return BlockDecision(accept, next_state, lost, gained, reason)


def resolved_schedule(
    state: AdaptiveState, levels: AdaptiveLevels = DEFAULT_LEVELS
) -> dict[str, int | float]:
    return {
        "F_native": levels.native_cycles[state.native_index],
        "F_old": levels.old_cycles[state.old_index],
        "simulations_per_move": levels.simulations[state.simulation_index],
        "donation_fraction": levels.donation_fraction[state.donation_index],
    }


def registered_manifest(seed: int = 0) -> dict:
    """Machine-readable starting contract; it contains no checkpoint inheritance."""
    roster: list[Candidate] = vnext_arms()
    return {
        "schema": "semantic-moves-v1-scientist-roster",
        "seed": seed,
        "initialization": "from-scratch",
        "objective": {
            "crossing_changes": "charged",
            "semantic_moves": "charged",
            "controller_internal_actions": "not-charged",
            "ratios": [1000.0, 10.0],
            "sampling_weights": [1.0, 1.0],
            "internal_horizon_between_semantic_actions": 5,
        },
        "block_rounds": 10,
        "levels": asdict(AdaptiveLevels()),
        "initial_schedule": resolved_schedule(AdaptiveState()),
        "scientists": [asdict(candidate) for candidate in roster],
    }


def capped_portfolio_cost(
    best_costs: Mapping[str, float | None], failure_cap: float
) -> float:
    if failure_cap <= 0:
        raise ValueError("failure cap must be positive")
    return sum(
        failure_cap if value is None else min(float(value), failure_cap)
        for value in best_costs.values()
    )
