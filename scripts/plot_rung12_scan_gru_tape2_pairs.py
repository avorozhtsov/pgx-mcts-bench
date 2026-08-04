#!/usr/bin/env python3
"""Plot rung-12 oracle pairs containing s-scan-gru-tape2.

The figure is deliberately generated as plain SVG so it remains inspectable,
diffable, and publication-ready without adding a plotting dependency.
"""

from __future__ import annotations

# SVG fragments are intentionally kept as complete, inspectable output lines.
# ruff: noqa: E501
import html
import math
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/nebius-rung18-20260801-current/runs"
LOCAL = ROOT / "artifacts/local-rung18-backfill-20260802/runs"
OUTPUT = ROOT / "research/images/rung12-scan-gru-tape2-pairs.svg"
CUTOFF = 12
FOCUS = "s-scan-gru-tape2"
REFERENCE = "s-window-128"
POOL = (
    REFERENCE,
    "s-head-budget96",
    "s-scan-gru",
    "s-paint4",
    "u1-puct",
    "d-head128-u1",
    "s-tape4",
    "s-triad-wst",
    FOCUS,
)


@dataclass(frozen=True)
class Rung:
    stage: int
    loss: float
    solve_rate: float
    iterations: int


@dataclass(frozen=True)
class Candidate:
    name: str
    rungs: dict[int, Rung]
    iterations_per_rung: float


@dataclass(frozen=True)
class Point:
    name: str
    x: float
    y: float
    selected: tuple[int, int] = (0, 0)
    ties: int = 0
    coverage: int = 0


def checkpoint(name: str) -> Path:
    local_progress = LOCAL / name / "checkpoints" / name / "progress.pt"
    if local_progress.exists():
        return local_progress
    local_main = LOCAL / name / "checkpoints" / f"{name}.pt"
    if local_main.exists():
        return local_main
    return BASE / name / "checkpoints" / f"{name}.pt"


def load_candidate(name: str) -> Candidate:
    saved = torch.load(checkpoint(name), map_location="cpu", weights_only=False)
    records = [
        row
        for row in saved.get("stages", [])
        if 0 <= int(row.get("stage", -1)) <= CUTOFF
    ]
    grouped: dict[int, list[dict]] = {}
    for row in records:
        grouped.setdefault(int(row["stage"]), []).append(row)

    best: dict[int, Rung] = {}
    for stage, attempts in grouped.items():
        usable: list[tuple[tuple[float, float, int], Rung]] = []
        for row in attempts:
            ratio = (row.get("by_ratio") or {}).get("10.0") or {}
            crossings = float(ratio.get("crossings", math.nan))
            moves = float(ratio.get("moves", math.nan))
            solve_rate = float(ratio.get("solved", math.nan))
            if not all(math.isfinite(value) for value in (crossings, moves, solve_rate)):
                continue
            rung = Rung(
                stage=stage,
                loss=10.0 * crossings + moves,
                solve_rate=solve_rate,
                iterations=int(row.get("iterations") or 0),
            )
            usable.append(((rung.loss, -rung.solve_rate, rung.iterations), rung))
        if usable:
            best[stage] = min(usable, key=lambda item: item[0])[1]

    if not best:
        raise RuntimeError(f"no usable rung records for {name}")
    total_iterations = sum(int(row.get("iterations") or 0) for row in records)
    return Candidate(
        name=name,
        rungs=best,
        iterations_per_rung=total_iterations / len(best),
    )


def single_point(candidate: Candidate, reference: Candidate) -> Point:
    stages = sorted(set(candidate.rungs) & set(reference.rungs))
    deltas = [candidate.rungs[s].loss - reference.rungs[s].loss for s in stages]
    solve_rates = [candidate.rungs[s].solve_rate for s in stages]
    return Point(
        name=candidate.name,
        x=sum(deltas) / len(deltas) + 10.0 * (1.0 - sum(solve_rates) / len(solve_rates)),
        y=candidate.iterations_per_rung,
        coverage=len(stages),
    )


def pair_point(focus: Candidate, other: Candidate, reference: Candidate) -> Point:
    stages = sorted((set(focus.rungs) | set(other.rungs)) & set(reference.rungs))
    chosen: list[Rung] = []
    focus_wins = other_wins = ties = 0
    for stage in stages:
        left = focus.rungs.get(stage)
        right = other.rungs.get(stage)
        if left is None:
            assert right is not None
            chosen.append(right)
            other_wins += 1
            continue
        if right is None:
            chosen.append(left)
            focus_wins += 1
            continue
        left_key = (left.loss, -left.solve_rate, focus.iterations_per_rung)
        right_key = (right.loss, -right.solve_rate, other.iterations_per_rung)
        if math.isclose(left.loss, right.loss, abs_tol=1e-12):
            ties += 1
        if left_key <= right_key:
            chosen.append(left)
            focus_wins += 1
        else:
            chosen.append(right)
            other_wins += 1

    deltas = [
        row.loss - reference.rungs[stage].loss
        for row, stage in zip(chosen, stages, strict=True)
    ]
    solve_rates = [row.solve_rate for row in chosen]
    total = focus_wins + other_wins
    weighted_y = (
        focus_wins * focus.iterations_per_rung
        + other_wins * other.iterations_per_rung
    ) / total
    return Point(
        name=f"{FOCUS} + {other.name}",
        x=sum(deltas) / len(deltas) + 10.0 * (1.0 - sum(solve_rates) / len(solve_rates)),
        y=weighted_y,
        selected=(focus_wins, other_wins),
        ties=ties,
        coverage=len(stages),
    )


def frontier(points: list[Point]) -> set[str]:
    result: set[str] = set()
    for point in points:
        dominated = any(
            other.name != point.name
            and other.x <= point.x
            and other.y <= point.y
            and (other.x < point.x or other.y < point.y)
            for other in points
        )
        if not dominated:
            result.add(point.name)
    return result


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def render(singles: list[Point], pairs: list[Point]) -> str:
    width, height = 1620, 900
    left, right, top, bottom = 95, 1010, 140, 785
    rail_x = 1060
    all_points = singles + pairs
    xmin = math.floor(min(point.x for point in all_points) - 1.0)
    xmax = math.ceil(max(point.x for point in all_points) + 1.0)
    ymin = max(0.0, math.floor(min(point.y for point in all_points) - 3.0))
    ymax = math.ceil(max(point.y for point in all_points) + 4.0)

    def px(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * (right - left)

    def py(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    by_name = {point.name: point for point in singles}
    focus = by_name[FOCUS]
    pair_frontier = frontier(pairs)
    ranked = sorted(pairs, key=lambda point: (point.x, point.y))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Rung-12 oracle pairs containing s-scan-gru-tape2</title>',
        '<desc id="desc">Candidate points and all eight two-policy oracles containing s-scan-gru-tape2. Lower is better on both axes. Records above rung twelve are excluded.</desc>',
        '<metadata>Generated from the final Nebius snapshot and current local backfill. Completed rung records 0 through 12 only; records 13 and above excluded. Missing focus records fall back to the other pair member.</metadata>',
        """<style>
text { font-family: "DejaVu Sans", Arial, sans-serif; fill: #172033; }
.title { font-size: 24px; font-weight: 700; }
.subtitle { font-size: 13px; fill: #59657a; }
.axis-label { font-size: 14px; font-weight: 600; }
.tick { font-size: 12px; fill: #59657a; }
.grid { stroke: #dbe1ea; stroke-width: 1; }
.axis { stroke: #172033; stroke-width: 1.2; }
.zero { stroke: #59657a; stroke-width: 1.2; stroke-dasharray: 5 5; }
.member-link { stroke: #7867a7; stroke-width: 1.2; opacity: .3; }
.single { fill: #7d8799; stroke: #fff; stroke-width: 2; }
.focus { fill: #d97706; stroke: #fff; stroke-width: 2.5; }
.pair { fill: #7867a7; stroke: #fff; stroke-width: 2; }
.frontier { fill: #1677b8; stroke: #fff; stroke-width: 2.5; }
.label { font-size: 11px; font-weight: 600; paint-order: stroke; stroke: #fff; stroke-width: 4px; stroke-linejoin: round; }
.rail-heading { font-size: 14px; font-weight: 700; }
.rail-item { font-size: 12px; }
.rail-value { font-size: 11px; fill: #59657a; }
.legend { font-size: 12px; fill: #59657a; }
</style>""",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        '<text x="95" y="38" class="title">Rung-12 pairs with s-scan-gru-tape2</text>',
        '<text x="95" y="64" class="subtitle">Only completed records through rung 12 are used; higher rungs are excluded.</text>',
        '<text x="95" y="85" class="subtitle">x = mean paired ΔL10:1 vs s-window-128 + 10 × (1 − avg_sr); y = weighted it/r of the selected pair member.</text>',
        f'<text x="95" y="106" class="subtitle">s-scan-gru-tape2 coverage: {focus.coverage}/13 rungs. On a missing rung, the oracle uses the other member.</text>',
        '<circle cx="760" cy="80" r="5" class="single"/><text x="772" y="85" class="legend">single</text>',
        '<circle cx="832" cy="80" r="7" class="focus"/><text x="845" y="85" class="legend">focus</text>',
        '<rect x="905" y="75" width="10" height="10" transform="rotate(45 910 80)" class="pair"/><text x="924" y="85" class="legend">pair</text>',
        '<rect x="973" y="75" width="10" height="10" transform="rotate(45 978 80)" class="frontier"/><text x="992" y="85" class="legend">pair Pareto frontier</text>',
    ]

    x_step = 2 if xmax - xmin <= 18 else 4
    for value in range(int(xmin), int(xmax) + 1, x_step):
        x = px(value)
        lines += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>',
            f'<text x="{x:.1f}" y="{bottom + 24}" text-anchor="middle" class="tick">{value}</text>',
        ]
    y_step = 5
    for value in range(int(math.ceil(ymin / y_step) * y_step), int(ymax) + 1, y_step):
        y = py(value)
        lines += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>',
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value}</text>',
        ]
    if xmin <= 0 <= xmax:
        lines.append(f'<line x1="{px(0):.1f}" y1="{top}" x2="{px(0):.1f}" y2="{bottom}" class="zero"/>')
    lines += [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
        f'<text x="{(left + right) / 2:.1f}" y="850" text-anchor="middle" class="axis-label">solve-penalized relative loss objective</text>',
        f'<text x="28" y="{(top + bottom) / 2:.1f}" transform="rotate(-90 28 {(top + bottom) / 2:.1f})" text-anchor="middle" class="axis-label">iterations per cleared rung</text>',
    ]

    for pair in pairs:
        other_name = pair.name.removeprefix(f"{FOCUS} + ")
        other = by_name[other_name]
        lines += [
            f'<line x1="{px(pair.x):.1f}" y1="{py(pair.y):.1f}" x2="{px(focus.x):.1f}" y2="{py(focus.y):.1f}" class="member-link"/>',
            f'<line x1="{px(pair.x):.1f}" y1="{py(pair.y):.1f}" x2="{px(other.x):.1f}" y2="{py(other.y):.1f}" class="member-link"/>',
        ]

    label_offsets = {
        "s-window-128": (8, -10),
        "s-head-budget96": (-8, 17),
        "s-scan-gru": (8, -9),
        "s-paint4": (8, 17),
        "u1-puct": (8, -9),
        "d-head128-u1": (8, -9),
        "s-tape4": (8, -9),
        "s-triad-wst": (8, 17),
        FOCUS: (8, -12),
    }
    for point in singles:
        cls = "focus" if point.name == FOCUS else "single"
        radius = 7 if point.name == FOCUS else 5.5
        dx, dy = label_offsets[point.name]
        anchor = "end" if dx < 0 else "start"
        lines += [
            f'<circle cx="{px(point.x):.1f}" cy="{py(point.y):.1f}" r="{radius}" class="{cls}"><title>{esc(point.name)}: x={point.x:.3f}, it/r={point.y:.2f}, coverage={point.coverage}/13</title></circle>',
            f'<text x="{px(point.x) + dx:.1f}" y="{py(point.y) + dy:.1f}" text-anchor="{anchor}" class="label">{esc(point.name)}</text>',
        ]

    for index, pair in enumerate(ranked, 1):
        x, y = px(pair.x), py(pair.y)
        cls = "frontier" if pair.name in pair_frontier else "pair"
        size = 13 if pair.name in pair_frontier else 11
        half = size / 2
        short = f"P{index}"
        lines += [
            f'<rect x="{x - half:.1f}" y="{y - half:.1f}" width="{size}" height="{size}" transform="rotate(45 {x:.1f} {y:.1f})" class="{cls}"><title>{esc(pair.name)}: x={pair.x:.3f}, it/r={pair.y:.2f}, selected={pair.selected[0]}:{pair.selected[1]}, ties={pair.ties}</title></rect>',
            f'<text x="{x + 10:.1f}" y="{y - 9:.1f}" class="label">{short}</text>',
        ]

    lines.append(f'<text x="{rail_x}" y="145" class="rail-heading">All focus pairs, ranked by x</text>')
    y = 175
    for index, pair in enumerate(ranked, 1):
        other = pair.name.removeprefix(f"{FOCUS} + ")
        frontier_mark = " · Pareto" if pair.name in pair_frontier else ""
        lines += [
            f'<text x="{rail_x}" y="{y}" class="rail-item">P{index}. + {esc(other)}{frontier_mark}</text>',
            f'<text x="{rail_x}" y="{y + 18}" class="rail-value">x={pair.x:.3f} · it/r={pair.y:.2f} · selected={pair.selected[0]}:{pair.selected[1]} · ties={pair.ties}</text>',
        ]
        y += 66
    lines += [
        f'<text x="{rail_x}" y="{height - 54}" class="rail-value">selected counts are focus:other across {ranked[0].coverage} evaluated rungs.</text>',
        f'<text x="{rail_x}" y="{height - 34}" class="rail-value">Lower is better on both axes.</text>',
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    candidates = {name: load_candidate(name) for name in POOL}
    reference = candidates[REFERENCE]
    singles = [single_point(candidates[name], reference) for name in POOL]
    pairs = [
        pair_point(candidates[FOCUS], candidates[name], reference)
        for name in POOL
        if name != FOCUS
    ]
    OUTPUT.write_text(render(singles, pairs), encoding="utf-8")
    print(OUTPUT)
    for point in sorted(pairs, key=lambda row: (row.x, row.y)):
        print(
            f"{point.name}\tx={point.x:.6f}\tit/r={point.y:.6f}"
            f"\tselected={point.selected[0]}:{point.selected[1]}\tties={point.ties}"
        )


if __name__ == "__main__":
    main()
