"""Same-root trajectory tournaments for relative policy improvement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pgx_mcts_bench.data import GameRecord


@dataclass(frozen=True)
class TournamentSplit:
    positive_indexes: tuple[int, ...]
    negative_indexes: tuple[int, ...]
    ignored_indexes: tuple[int, ...]
    boundary: str
    confidence: float


def _finite(value: object, fallback: float = math.inf) -> float:
    number = float(value)
    return number if math.isfinite(number) else fallback


def _first(record: GameRecord):
    if not record:
        raise ValueError("trajectory tournament contains an empty trajectory")
    return record[0]


def _validate_same_root(records: list[GameRecord], expected_size: int) -> list[int]:
    if len(records) != expected_size:
        raise ValueError(f"trajectory tournament requires exactly {expected_size} trajectories")
    roots = {
        (
            str(getattr(_first(record), "representation_id", "")),
            float(getattr(_first(record), "objective_ratio", float("nan"))),
        )
        for record in records
    }
    if len(roots) != 1 or not next(iter(roots))[0] or not math.isfinite(next(iter(roots))[1]):
        raise ValueError("trajectory tournament must have one explicit representation and ratio")
    return [
        index
        for index, record in enumerate(records)
        if bool(getattr(_first(record), "trajectory_replay_valid", True))
    ]


def _best_gap(
    ordered: list[int],
    values: list[float],
    *,
    minimum_gap: float,
) -> tuple[tuple[int, ...], tuple[int, ...], float] | None:
    candidates = []
    for cut in range(1, len(ordered)):
        gap = values[cut] - values[cut - 1]
        if gap >= minimum_gap:
            balance = min(cut, len(ordered) - cut)
            candidates.append((gap, balance, -cut, cut))
    if not candidates:
        return None
    gap, _, _, cut = max(candidates)
    return tuple(ordered[:cut]), tuple(ordered[cut:]), float(gap)


def split_trajectory_tournament(
    records: list[GameRecord],
    *,
    expected_size: int = 10,
) -> TournamentSplit | None:
    """Split ten replay-valid same-root trajectories at an interpretable gap.

    Ordering is objective-aligned and fail-closed: solve status, then crossing
    changes, then semantic moves.  For unsolved trajectories only observed
    progress is compared.  If there is no robust numeric gap, a unique best
    trajectory receives a low-confidence best-vs-rest update.  Invalid routes
    are ignored rather than treated as negatives.
    """

    valid = _validate_same_root(records, expected_size)
    ignored = tuple(index for index in range(len(records)) if index not in valid)
    if len(valid) < 2:
        return None
    solved = [index for index in valid if float(getattr(_first(records[index]), "solved", 0)) > 0.5]
    unsolved = [index for index in valid if index not in solved]
    if solved and unsolved:
        return TournamentSplit(
            positive_indexes=tuple(sorted(solved)),
            negative_indexes=tuple(sorted(unsolved)),
            ignored_indexes=ignored,
            boundary="solved-vs-unsolved",
            confidence=1.0,
        )

    if solved:
        ordered = sorted(
            solved,
            key=lambda index: (
                _finite(getattr(_first(records[index]), "final_crossing_changes", math.inf)),
                _finite(getattr(_first(records[index]), "final_moves", math.inf)),
                index,
            ),
        )
        crossings = [
            _finite(getattr(_first(records[index]), "final_crossing_changes", math.inf))
            for index in ordered
        ]
        split = _best_gap(ordered, crossings, minimum_gap=1.0)
        if split is not None:
            positive, negative, _ = split
            return TournamentSplit(positive, negative, ignored, "crossing-change-gap", 1.0)
        moves = [
            _finite(getattr(_first(records[index]), "final_moves", math.inf)) for index in ordered
        ]
        finite_moves = [value for value in moves if math.isfinite(value)]
        threshold = max(4.0, 0.1 * float(np.median(finite_moves))) if finite_moves else math.inf
        split = _best_gap(ordered, moves, minimum_gap=threshold)
        if split is not None:
            positive, negative, _ = split
            return TournamentSplit(positive, negative, ignored, "semantic-move-gap", 0.75)
    else:
        ordered = sorted(
            unsolved,
            key=lambda index: (
                _finite(getattr(_first(records[index]), "best_residual_word_length", math.inf)),
                _finite(getattr(_first(records[index]), "residual_word_length", math.inf)),
                _finite(getattr(_first(records[index]), "final_moves", math.inf)),
                index,
            ),
        )
        residuals = [
            _finite(getattr(_first(records[index]), "best_residual_word_length", math.inf))
            for index in ordered
        ]
        split = _best_gap(ordered, residuals, minimum_gap=2.0)
        if split is not None:
            positive, negative, _ = split
            return TournamentSplit(positive, negative, ignored, "residual-length-gap", 0.5)

    def key(index: int) -> tuple[float, float, float]:
        row = _first(records[index])
        if solved:
            return (
                _finite(getattr(row, "final_crossing_changes", math.inf)),
                _finite(getattr(row, "final_moves", math.inf)),
                float(index),
            )
        return (
            _finite(getattr(row, "best_residual_word_length", math.inf)),
            _finite(getattr(row, "residual_word_length", math.inf)),
            float(index),
        )

    ordered = sorted(valid, key=key)
    if key(ordered[0])[:2] == key(ordered[1])[:2]:
        return None
    return TournamentSplit(
        positive_indexes=(ordered[0],),
        negative_indexes=tuple(ordered[1:]),
        ignored_indexes=ignored,
        boundary="unique-best-vs-rest",
        confidence=0.25,
    )


def apply_tournament_advantages(
    records: list[GameRecord],
    split: TournamentSplit,
) -> None:
    """Attach zero-sum per-root advantages with equal trajectory mass."""

    positive = split.confidence / len(split.positive_indexes)
    negative = -split.confidence / len(split.negative_indexes)
    for index, record in enumerate(records):
        advantage = (
            positive
            if index in split.positive_indexes
            else negative
            if index in split.negative_indexes
            else 0.0
        )
        for position in record:
            position.relative_trajectory_advantage = advantage
