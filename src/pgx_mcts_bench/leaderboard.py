"""Live ladder standings derived from promoted-state checkpoints.

The long-running ladder writes ``ladder.json`` only when a process exits.  Its
checkpoint is therefore the authoritative source for a live leaderboard.  This
module discovers those checkpoints rather than hardcoding run names, and treats
server results as ordinary snapshots copied beneath an artifact root.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from rf_knots.knot_table import scheduled_unknotting_number

from pgx_mcts_bench.ladder import RATIOS, STAGES

# The artifact tree contains local runs, copied server snapshots, and newly added
# candidate families.  Discovery plus per-candidate selection below is safer
# than maintaining a list of run names that becomes stale whenever a new root is
# copied in.
DEFAULT_ROOTS = (Path("artifacts"),)

# Negative-result arms remain reproducible and their checkpoints remain intact,
# but they no longer compete in the live standings.  Their post-mortems belong
# in the research record rather than in every current leaderboard.
RETIRED_CANDIDATES = frozenset({"s-ff4-p5"})

# Ratio-10 loss is reported as a paired, per-rung difference from this policy.
# Using a serial policy as the zero point makes the comparison directly answer
# whether another architecture improves on the strongest compact baseline.
LOSS_10_REFERENCE = "s-window-128"


@dataclass(frozen=True)
class RungScore:
    stage: int
    crossings: float
    optimal_crossings: int
    solve_rate: float = float("nan")
    moves_10: float = float("nan")
    loss_10: float = float("nan")


@dataclass(frozen=True)
class LeaderboardRow:
    name: str
    checkpoint: Path
    highest_stage: int
    highest_rung: str
    rungs_cleared: int
    total_iterations: int
    iterations_per_rung: float
    crossings: float
    solve_rate: float
    average_solve_rate: float
    solve_rate_rungs: int
    expected_crossings: float
    rung_scores: tuple[RungScore, ...]
    top_gap: float = float("nan")
    average_gap: float = float("nan")
    gap_rungs: int = 0
    average_move_delta: float = float("nan")
    move_delta_rungs: int = 0
    average_loss_10_delta: float = float("nan")
    loss_10_delta_rungs: int = 0


def discover_checkpoints(roots: Iterable[Path]) -> list[Path]:
    """Find live checkpoint files, excluding per-stage before/after snapshots."""
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file() and root.suffix == ".pt":
            found.add(root)
        elif root.is_dir():
            # A manually collected snapshot may simply be a directory of
            # candidate checkpoints rather than a copy of the artifact tree.
            found.update(root.glob("*.pt"))
            # The resume checkpoint lives immediately inside ``checkpoints``;
            # stage snapshots live one directory deeper and are intentionally
            # not candidates for the leaderboard.
            found.update(root.glob("**/checkpoints/*.pt"))
    return sorted(found)


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _optimal_crossings(record: dict) -> int:
    """Use newly proved scheduled values to backfill historical checkpoints."""
    recorded = int(record.get("optimal_crossings", -1))
    if recorded >= 0:
        return recorded
    known = scheduled_unknotting_number(str(record.get("source", "")))
    return -1 if known is None else known


def load_row(path: Path) -> LeaderboardRow | None:
    """Read one candidate's standings from a trusted training checkpoint."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    name = str(saved.get("candidate") or path.stem)

    # Keep every completed attempt for total_it.  Gap-filling resumes can revisit
    # a rung, and those iterations are real work even though the denominator is
    # the number of distinct current rungs cleared.
    records = [
        row
        for row in (saved.get("stages") or [])
        if (row.get("source"), row.get("scramble")) in STAGES
    ]
    cleared: dict[tuple[str, int], dict] = {}
    for row in records:
        if row.get("promoted"):
            cleared[(row["source"], row["scramble"])] = row
    if not cleared:
        return None

    highest_stage = max(STAGES.index(stage) for stage in cleared)
    top = cleared[STAGES[highest_stage]]
    ratio = str(max(RATIOS))
    high = (top.get("by_ratio") or {}).get(ratio) or {}
    crossings = _number(high.get("crossings"))
    solve_rate = _number(high.get("solved"))
    expected = _number(high.get("expected_crossings"))
    if math.isnan(expected) and solve_rate > 0.0 and not math.isnan(crossings):
        expected = crossings / solve_rate

    total_iterations = sum(int(row.get("iterations") or 0) for row in records)
    rungs_cleared = len(cleared)
    # Best finite crossing-dominant measurement per current rung. A gap-filling
    # resume may revisit a rung; "achieved" means the best result actually
    # recorded there, while total_it still counts every attempt.
    best_crossings: dict[int, tuple[float, int]] = {}
    best_solve_rates: dict[int, float] = {}
    best_moves: dict[int, float] = {}
    best_loss_10: dict[int, float] = {}
    for record in records:
        stage = STAGES.index((record["source"], record["scramble"]))
        measured_crossings = _number(
            ((record.get("by_ratio") or {}).get(ratio) or {}).get("crossings")
        )
        if not math.isnan(measured_crossings):
            previous = best_crossings.get(stage)
            if previous is None or measured_crossings < previous[0]:
                best_crossings[stage] = (
                    measured_crossings,
                    _optimal_crossings(record),
                )
        measured_solve_rate = _number(
            ((record.get("by_ratio") or {}).get(ratio) or {}).get("solved")
        )
        if not math.isnan(measured_solve_rate):
            best_solve_rates[stage] = max(
                best_solve_rates.get(stage, -float("inf")), measured_solve_rate
            )
        ratio_10 = (record.get("by_ratio") or {}).get("10.0") or {}
        measured_moves = _number(ratio_10.get("moves"))
        if not math.isnan(measured_moves):
            best_moves[stage] = min(best_moves.get(stage, float("inf")), measured_moves)
        measured_crossings_10 = _number(ratio_10.get("crossings"))
        if not math.isnan(measured_crossings_10) and not math.isnan(measured_moves):
            measured_loss_10 = 10.0 * measured_crossings_10 + measured_moves
            best_loss_10[stage] = min(
                best_loss_10.get(stage, float("inf")), measured_loss_10
            )
    measured_stages = sorted(
        set(best_crossings) | set(best_solve_rates) | set(best_moves) | set(best_loss_10)
    )
    rung_scores = tuple(
        RungScore(
            stage=stage,
            crossings=best_crossings.get(stage, (float("nan"), -1))[0],
            optimal_crossings=best_crossings.get(stage, (float("nan"), -1))[1],
            solve_rate=best_solve_rates.get(stage, float("nan")),
            moves_10=best_moves.get(stage, float("nan")),
            loss_10=best_loss_10.get(stage, float("nan")),
        )
        for stage in measured_stages
    )
    return LeaderboardRow(
        name=name,
        checkpoint=path,
        highest_stage=highest_stage,
        highest_rung=f"{top['source']}+{top['scramble']}",
        rungs_cleared=rungs_cleared,
        total_iterations=total_iterations,
        iterations_per_rung=total_iterations / rungs_cleared,
        crossings=crossings,
        solve_rate=solve_rate,
        average_solve_rate=(
            sum(best_solve_rates.values()) / len(best_solve_rates)
            if best_solve_rates
            else float("nan")
        ),
        solve_rate_rungs=len(best_solve_rates),
        expected_crossings=expected,
        rung_scores=rung_scores,
    )


def leaderboard(roots: Iterable[Path]) -> tuple[list[LeaderboardRow], list[str]]:
    """Load and rank the newest valid checkpoint for each candidate."""
    best: dict[str, tuple[tuple[int, int, int], LeaderboardRow]] = {}
    warnings: list[str] = []
    for path in discover_checkpoints(roots):
        try:
            row = load_row(path)
        except Exception as error:  # a checkpoint can be observed mid-write
            warnings.append(f"Skipped {path}: {error}")
            continue
        if row is None:
            continue
        if row.name in RETIRED_CANDIDATES:
            continue
        modified = path.stat().st_mtime_ns
        # A fresh smoke/device checkpoint must not hide a deeper resumable run.
        # Prefer progress first, then coverage, and use modification time only
        # to choose between equally advanced snapshots.
        rank = (row.highest_stage, row.rungs_cleared, modified)
        previous = best.get(row.name)
        if previous is None or rank > previous[0]:
            best[row.name] = (rank, row)

    rows = [entry[1] for entry in best.values()]

    # Labelled rungs use their theorem. Unknown rungs use the best achieved cc in
    # the discovered population, turning the reference into a live ratchet.
    references: dict[int, float] = {}
    for row in rows:
        for score in row.rung_scores:
            if math.isnan(score.crossings):
                continue
            if score.optimal_crossings >= 0:
                references[score.stage] = float(score.optimal_crossings)
            else:
                previous = references.get(score.stage, float("inf"))
                references[score.stage] = min(previous, score.crossings)
    rows = [_with_average_gap(row, references) for row in rows]
    u1 = next((row for row in rows if row.name == "u1-puct"), None)
    u1_moves = (
        {
            score.stage: score.moves_10
            for score in u1.rung_scores
            if not math.isnan(score.moves_10)
        }
        if u1 is not None
        else {}
    )
    rows = [_with_average_move_delta(row, u1_moves) for row in rows]
    loss_reference = next(
        (row for row in rows if row.name == LOSS_10_REFERENCE), None
    )
    reference_loss_10 = (
        {
            score.stage: score.loss_10
            for score in loss_reference.rung_scores
            if not math.isnan(score.loss_10)
        }
        if loss_reference is not None
        else {}
    )
    rows = [
        _with_average_loss_10_delta(row, reference_loss_10) for row in rows
    ]
    rows.sort(
        key=lambda row: (
            -row.highest_stage,
            -row.rungs_cleared,
            _sort_number(row.expected_crossings),
            row.name,
        )
    )
    return rows, warnings


def _with_average_gap(
    row: LeaderboardRow, references: dict[int, float]
) -> LeaderboardRow:
    gaps = [
        score.crossings - references[score.stage]
        for score in row.rung_scores
        if score.stage in references and not math.isnan(score.crossings)
    ]
    top = next(
        (score for score in row.rung_scores if score.stage == row.highest_stage),
        None,
    )
    top_gap = (
        top.crossings - references[top.stage]
        if top is not None and top.stage in references and not math.isnan(top.crossings)
        else float("nan")
    )
    return replace(
        row,
        top_gap=top_gap,
        average_gap=sum(gaps) / len(gaps) if gaps else float("nan"),
        gap_rungs=len(gaps),
    )


def _with_average_move_delta(
    row: LeaderboardRow, u1_moves: dict[int, float]
) -> LeaderboardRow:
    deltas = [
        score.moves_10 - u1_moves[score.stage]
        for score in row.rung_scores
        if score.stage in u1_moves and not math.isnan(score.moves_10)
    ]
    return replace(
        row,
        average_move_delta=sum(deltas) / len(deltas) if deltas else float("nan"),
        move_delta_rungs=len(deltas),
    )


def _with_average_loss_10_delta(
    row: LeaderboardRow, reference_loss_10: dict[int, float]
) -> LeaderboardRow:
    deltas = [
        score.loss_10 - reference_loss_10[score.stage]
        for score in row.rung_scores
        if score.stage in reference_loss_10 and not math.isnan(score.loss_10)
    ]
    return replace(
        row,
        average_loss_10_delta=sum(deltas) / len(deltas) if deltas else float("nan"),
        loss_10_delta_rungs=len(deltas),
    )


def _sort_number(value: float) -> float:
    return value if not math.isnan(value) else float("inf")


def _decimal(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.2f}"


def _percent(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.0%}"


def render(rows: list[LeaderboardRow]) -> str:
    """Render a fixed-width table that stays compact in terminals and chat."""
    lines = [
        "candidate            r  top           u    sr    cc   gap   n  "
        "avg_sr   avgΔ  avgΔmv  avgΔL10:1(win)   it   it/r",
        "─" * 124,
    ]
    for row in rows:
        top_score = next(
            (score for score in row.rung_scores if score.stage == row.highest_stage),
            None,
        )
        optimum = top_score.optimal_crossings if top_score is not None else -1
        u = "?" if optimum < 0 else str(optimum)
        lines.append(
            f"{row.name:<20} {row.highest_stage:>2}  {row.highest_rung:<12} {u:>2}  "
            f"{_decimal(row.solve_rate):>4}  {_decimal(row.crossings):>4}  "
            f"{_decimal(row.top_gap):>5}  "
            f"{row.gap_rungs:>2}  "
            f"{_decimal(row.average_solve_rate):>6}  "
            f"{_decimal(row.average_gap):>5}  "
            f"{_decimal(row.average_move_delta):>7}  "
            f"{_decimal(row.average_loss_10_delta):>11} "
            f"{row.total_iterations:>4} {row.iterations_per_rung:>6.2f}"
        )
    return "\n".join(lines) + "\n"
