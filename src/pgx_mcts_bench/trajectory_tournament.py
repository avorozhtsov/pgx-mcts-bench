"""Same-root trajectory tournaments for relative policy improvement."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from pgx_mcts_bench.data import GameRecord, Position


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


def _trimmed_extremes(
    ordered: list[int], width: int = 3
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return disjoint best/worst anchors while leaving the ambiguous middle unused."""

    count = min(width, len(ordered) // 2)
    if count < 1:
        return (), ()
    return tuple(ordered[:count]), tuple(ordered[-count:])


def split_trajectory_tournament_trimmed(
    records: list[GameRecord],
    *,
    expected_size: int = 10,
) -> TournamentSplit | None:
    """Select only clearly separated high- and low-quality trajectory anchors.

    Unlike the original largest-gap split, this rule deliberately drops the
    middle of the tournament.  Mixed outcomes use the categorical
    solved/unsolved boundary.  Homogeneous outcomes require a robust gap
    between the best and worst thirds; otherwise the update is skipped.
    """

    valid = _validate_same_root(records, expected_size)
    ignored_invalid = {index for index in range(len(records)) if index not in valid}
    if len(valid) < 2:
        return None
    solved = [index for index in valid if float(getattr(_first(records[index]), "solved", 0)) > 0.5]
    unsolved = [index for index in valid if index not in solved]

    def solved_key(index: int) -> tuple[float, float, int]:
        return (
            _finite(getattr(_first(records[index]), "final_crossing_changes", math.inf)),
            _finite(getattr(_first(records[index]), "final_moves", math.inf)),
            index,
        )

    def unsolved_key(index: int) -> tuple[float, float, float, int]:
        return (
            _finite(getattr(_first(records[index]), "best_residual_word_length", math.inf)),
            _finite(getattr(_first(records[index]), "residual_word_length", math.inf)),
            _finite(getattr(_first(records[index]), "final_moves", math.inf)),
            index,
        )

    if solved and unsolved:
        positive = tuple(sorted(solved, key=solved_key)[: min(3, len(solved))])
        negative = tuple(sorted(unsolved, key=unsolved_key)[-min(3, len(unsolved)) :])
        ignored = tuple(sorted(set(range(len(records))) - set(positive) - set(negative)))
        return TournamentSplit(positive, negative, ignored, "trimmed-solved-vs-unsolved", 1.0)

    ordered = sorted(valid, key=solved_key if solved else unsolved_key)
    positive, negative = _trimmed_extremes(ordered)
    if not positive or not negative:
        return None
    if solved:
        positive_worst = solved_key(positive[-1])
        negative_best = solved_key(negative[0])
        crossing_gap = negative_best[0] - positive_worst[0]
        move_gap = negative_best[1] - positive_worst[1]
        finite_moves = [
            solved_key(index)[1] for index in valid if math.isfinite(solved_key(index)[1])
        ]
        move_threshold = (
            max(4.0, 0.1 * float(np.median(finite_moves))) if finite_moves else math.inf
        )
        if crossing_gap >= 1.0:
            boundary, confidence = "trimmed-crossing-change-margin", 1.0
        elif crossing_gap == 0.0 and move_gap >= move_threshold:
            boundary, confidence = "trimmed-semantic-move-margin", 0.75
        else:
            return None
    else:
        residual_gap = unsolved_key(negative[0])[0] - unsolved_key(positive[-1])[0]
        if residual_gap < 2.0:
            return None
        boundary, confidence = "trimmed-residual-length-margin", 0.5
    ignored = tuple(sorted(ignored_invalid | (set(valid) - set(positive) - set(negative))))
    return TournamentSplit(positive, negative, ignored, boundary, confidence)


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


def _state_fingerprint(position: Position) -> bytes:
    observation = np.ascontiguousarray(position.observation, dtype=np.float32)
    legal = np.ascontiguousarray(position.legal_actions, dtype=np.uint8)
    digest = hashlib.blake2b(digest_size=20)
    digest.update(str(observation.shape).encode())
    digest.update(observation.tobytes())
    digest.update(legal.tobytes())
    return digest.digest()


def apply_divergence_tournament_advantages(
    records: list[GameRecord],
    split: TournamentSplit,
) -> int:
    """Apply zero-sum credit only where positive and negative paths diverge.

    Actions on a common prefix are not globally bad merely because one later
    continuation lost the tournament.  We therefore supervise only shared
    states at which the selected positive and negative groups chose different,
    group-exclusive actions.  Returns the number of annotated positions.
    """

    for record in records:
        for position in record:
            position.relative_trajectory_advantage = 0.0
    groups: dict[bytes, dict[str, list[Position]]] = {}
    for label, indexes in (
        ("positive", split.positive_indexes),
        ("negative", split.negative_indexes),
    ):
        for index in indexes:
            for position in records[index]:
                groups.setdefault(_state_fingerprint(position), {"positive": [], "negative": []})[
                    label
                ].append(position)

    annotated = 0
    for rows in groups.values():
        positive_rows = rows["positive"]
        negative_rows = rows["negative"]
        if not positive_rows or not negative_rows:
            continue
        positive_actions = {int(row.action) for row in positive_rows}
        negative_actions = {int(row.action) for row in negative_rows}
        positive_targets = [row for row in positive_rows if int(row.action) not in negative_actions]
        negative_targets = [row for row in negative_rows if int(row.action) not in positive_actions]
        if not positive_targets or not negative_targets:
            continue
        positive_mass = split.confidence / len(positive_targets)
        negative_mass = -split.confidence / len(negative_targets)
        for row in positive_targets:
            row.relative_trajectory_advantage = positive_mass
        for row in negative_targets:
            row.relative_trajectory_advantage = negative_mass
        annotated += len(positive_targets) + len(negative_targets)
    return annotated
