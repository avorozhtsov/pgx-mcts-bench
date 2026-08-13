#!/usr/bin/env python3
"""Rank scientists by native, pre-donation innovation on a paired task panel.

The collaborative curriculum writes ``scientists.*.evaluation`` immediately
after native learning and before translation/distillation.  This script uses
that snapshot, never a post-sharing retention result.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_CAPS = {"10.0": 328.0, "1000.0": 20128.0}


def _ratio_key(value: str | float) -> str:
    return str(float(value))


def _load_events(root: Path, max_rounds: int | None) -> list[dict[str, Any]]:
    completed = {
        int(path.stem): json.loads(path.read_text())
        for path in sorted((root / "events").glob("*.json"))
    }
    native = {
        int(path.stem): json.loads(path.read_text())
        for path in sorted((root / "native-events").glob("*.json"))
    }
    events = [
        completed[index] if index in completed else native[index]
        for index in sorted(completed.keys() | native.keys())
    ]
    if max_rounds is not None:
        events = [
            event for event in events if int(event.get("round", event.get("rung", 0))) < max_rounds
        ]
    return events


def _evaluation_evals(evaluation: dict[str, Any]) -> int:
    return sum(
        int(attempt.get("scheduled_network_evaluations", 0))
        for cell in evaluation.values()
        for attempt in cell.get("attempts", [])
    )


def _native_evals(scientist_event: dict[str, Any]) -> int:
    return sum(
        int(iteration.get("scheduled_network_evaluations", 0))
        for iteration in scientist_event.get("iterations", [])
    ) + _evaluation_evals(scientist_event.get("evaluation", {}))


def _probe_evals(event: dict[str, Any], scientist: str) -> int:
    selected = event.get("selected")
    return sum(
        int(attempt.get("scheduled_network_evaluations", 0))
        for row in event.get("qualification", [])
        if row.get("scientist") == scientist and row.get("representation") == selected
        for attempt in row.get("evidence", {}).get("attempts", [])
    )


def load_run(
    label: str, root: Path, max_rounds: int | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return ``identity -> representation -> native observation``."""
    observations: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in _load_events(root, max_rounds):
        representation = event.get("selected", event.get("representation"))
        if representation is None:
            raise ValueError(f"event in {root} has no representation identifier")
        scientist_rows = event.get("scientists")
        if scientist_rows is None:
            scientist_rows = {event.get("scientist", root.name): event}
        for scientist, scientist_event in scientist_rows.items():
            identity = f"{label}/{scientist}"
            if representation in observations[identity]:
                raise ValueError(
                    f"duplicate first-contact observation: {identity} {representation}"
                )
            observations[identity][representation] = {
                "evaluation": scientist_event.get("evaluation", {}),
                "native_network_evaluations": _native_evals(scientist_event),
                "selection_probe_network_evaluations": _probe_evals(event, scientist),
                "round": int(event.get("round", event.get("rung", 0))),
            }
    return dict(observations)


def merge_runs(
    runs: list[tuple[str, Path]], max_rounds: int | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for label, root in runs:
        for identity, rows in load_run(label, root, max_rounds).items():
            if identity in merged:
                raise ValueError(f"duplicate scientist identity: {identity}")
            merged[identity] = rows
    return merged


def rank_innovation(
    observations: dict[str, dict[str, dict[str, Any]]],
    *,
    caps: dict[str, float] | None = None,
    panel: str = "intersection",
    incumbent_bank: dict[str, dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    """Compute objective-specific innovation rankings.

    A native win is the lowest solved objective on a representation. Ties split
    win credit and their margin against the next distinct cost. Portfolio
    marginal contribution is stricter: it is positive only when removing a
    scientist worsens the portfolio.
    """
    if not observations:
        raise ValueError("no scientist observations")
    caps = {_ratio_key(key): float(value) for key, value in (caps or DEFAULT_CAPS).items()}
    representation_sets = [set(rows) for rows in observations.values()]
    if panel == "intersection":
        representations = sorted(set.intersection(*representation_sets))
    elif panel == "union":
        representations = sorted(set.union(*representation_sets))
    else:
        raise ValueError(f"unknown panel mode: {panel}")
    if not representations:
        raise ValueError("paired panel is empty")

    results: dict[str, Any] = {}
    for ratio, cap in sorted(caps.items(), key=lambda row: float(row[0])):
        stats = {
            identity: {
                "scientist": identity,
                "representations": 0,
                "solved_representations": 0,
                "attempts": 0,
                "solved_attempts": 0,
                "conditional_cost_sum": 0.0,
                "capped_cost_sum": 0.0,
                "native_win_credit": 0.0,
                "sole_solves": 0,
                "innovation_margin": 0.0,
                "normalized_innovation_margin": 0.0,
                "portfolio_marginal_contribution": 0.0,
                "record_eligible_representations": 0,
                "record_improvements": 0,
                "record_improvement_margin": 0.0,
                "normalized_record_improvement_margin": 0.0,
                "native_network_evaluations": 0,
                "selection_probe_network_evaluations": 0,
            }
            for identity in observations
        }
        task_rows = []
        for representation in representations:
            costs: dict[str, float] = {}
            solved: dict[str, bool] = {}
            for identity, by_representation in observations.items():
                observation = by_representation.get(representation)
                if observation is None:
                    if panel == "intersection":
                        raise AssertionError("intersection contains a missing observation")
                    continue
                cell = observation["evaluation"].get(ratio, {})
                best = cell.get("best_objective")
                is_solved = best is not None
                cost = float(best) if is_solved else cap
                costs[identity] = cost
                solved[identity] = is_solved
                row = stats[identity]
                row["representations"] += 1
                row["solved_representations"] += int(is_solved)
                attempts = cell.get("attempts", [])
                row["attempts"] += len(attempts)
                row["solved_attempts"] += sum(bool(attempt.get("solved")) for attempt in attempts)
                if is_solved:
                    row["conditional_cost_sum"] += cost
                row["capped_cost_sum"] += cost
                row["native_network_evaluations"] += int(observation["native_network_evaluations"])
                row["selection_probe_network_evaluations"] += int(
                    observation["selection_probe_network_evaluations"]
                )
                incumbent = (incumbent_bank or {}).get(representation, {}).get(ratio)
                if incumbent is not None:
                    incumbent = float(incumbent)
                    row["record_eligible_representations"] += 1
                    if is_solved and cost < incumbent:
                        row["record_improvements"] += 1
                        row["record_improvement_margin"] += incumbent - cost
                        row["normalized_record_improvement_margin"] += (incumbent - cost) / cap
            if not any(solved.values()):
                task_rows.append(
                    {"representation": representation, "winner": [], "best_cost": None}
                )
                continue
            best_cost = min(costs[name] for name, ok in solved.items() if ok)
            winners = sorted(
                name for name, ok in solved.items() if ok and math.isclose(costs[name], best_cost)
            )
            distinct_losers = [cost for name, cost in costs.items() if name not in winners]
            margin = max(0.0, min(distinct_losers) - best_cost) if distinct_losers else 0.0
            for winner in winners:
                stats[winner]["native_win_credit"] += 1.0 / len(winners)
                stats[winner]["innovation_margin"] += margin / len(winners)
                stats[winner]["normalized_innovation_margin"] += margin / cap / len(winners)
            solved_names = [name for name, ok in solved.items() if ok]
            if len(solved_names) == 1:
                stats[solved_names[0]]["sole_solves"] += 1
            all_best = min(costs.values())
            for identity in costs:
                without = min(
                    (cost for name, cost in costs.items() if name != identity), default=cap
                )
                stats[identity]["portfolio_marginal_contribution"] += max(0.0, without - all_best)
            task_rows.append(
                {
                    "representation": representation,
                    "winner": winners,
                    "best_cost": best_cost,
                    "margin": margin,
                }
            )

        rows = []
        for row in stats.values():
            solved_count = row["solved_representations"]
            row["solve_rate"] = (
                solved_count / row["representations"] if row["representations"] else 0.0
            )
            row["attempt_solve_rate"] = (
                row["solved_attempts"] / row["attempts"] if row["attempts"] else 0.0
            )
            row["conditional_average_cost"] = (
                row.pop("conditional_cost_sum") / solved_count if solved_count else None
            )
            row["capped_average_cost"] = (
                row.pop("capped_cost_sum") / row["representations"]
                if row["representations"]
                else None
            )
            total_compute = (
                row["native_network_evaluations"] + row["selection_probe_network_evaluations"]
            )
            row["total_first_contact_network_evaluations"] = total_compute
            row["innovation_score"] = (
                row["normalized_innovation_margin"]
                + row["sole_solves"]
                + row["normalized_record_improvement_margin"]
            )
            row["normalized_margin_per_million_evaluations"] = (
                row["normalized_innovation_margin"] * 1_000_000 / total_compute
                if total_compute
                else None
            )
            rows.append(row)
        rows.sort(
            key=lambda row: (
                -row["innovation_score"],
                -row["portfolio_marginal_contribution"],
                -row["native_win_credit"],
                row["capped_average_cost"],
                row["scientist"],
            )
        )
        results[ratio] = {"cap": cap, "ranking": rows, "tasks": task_rows}
    return {
        "schema": "sv2-native-innovation-ranking-v1",
        "measurement": "post-native-learning, pre-translation, pre-donation paired evaluation",
        "panel_mode": panel,
        "representations": representations,
        "objectives": results,
        "incumbent_bank_supplied": incumbent_bank is not None,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Native innovation leaderboard",
        "",
        (
            f"Paired panel: {len(report['representations'])} representations; "
            f"measurement: {report['measurement']}."
        ),
        "",
    ]
    for ratio, objective in report["objectives"].items():
        lines.extend(
            [
                f"## L{float(ratio):g} (unsolved cap {objective['cap']:g})",
                "",
                "| Rank | Scientist | Innovation score | Solved | Attempt SR | Win credit | "
                "Sole solves | Margin | Portfolio delta | Record gains | Capped avg | "
                "Native evals | Probe evals | Norm margin / 1M evals |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(objective["ranking"], 1):
            lines.append(
                (
                    "| {rank} | {scientist} | {score:.4f} | {solved}/{total} | "
                    "{attempt_sr:.1%} | {wins:.2f} | {sole} | {margin:.1f} | {delta:.1f} | "
                    "{records} | {capped:.2f} | {native:,} | {probe:,} | {eff:.3f} |"
                ).format(
                    rank=rank,
                    scientist=row["scientist"],
                    score=row["innovation_score"],
                    solved=row["solved_representations"],
                    total=row["representations"],
                    attempt_sr=row["attempt_solve_rate"],
                    wins=row["native_win_credit"],
                    sole=row["sole_solves"],
                    margin=row["innovation_margin"],
                    delta=row["portfolio_marginal_contribution"],
                    records=(
                        f"{row['record_improvements']}/{row['record_eligible_representations']}"
                        if report["incumbent_bank_supplied"]
                        else "n/a"
                    ),
                    capped=row["capped_average_cost"],
                    native=row["native_network_evaluations"],
                    probe=row["selection_probe_network_evaluations"],
                    eff=row["normalized_margin_per_million_evaluations"] or 0.0,
                )
            )
        lines.append("")
    return "\n".join(lines)


def _run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid run: {value}")
    return label, path


def _cap(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("cap must be A=VALUE")
    ratio, cap = value.split("=", 1)
    return _ratio_key(ratio), float(cap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_run, metavar="LABEL=PATH")
    parser.add_argument("--cap", action="append", type=_cap, metavar="A=VALUE")
    parser.add_argument("--panel", choices=("intersection", "union"), default="intersection")
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument(
        "--incumbent-bank",
        type=Path,
        help="JSON mapping representation -> objective ratio -> incumbent cost",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    caps = dict(args.cap) if args.cap else DEFAULT_CAPS
    incumbent_bank = json.loads(args.incumbent_bank.read_text()) if args.incumbent_bank else None
    report = rank_innovation(
        merge_runs(args.run, args.max_rounds),
        caps=caps,
        panel=args.panel,
        incumbent_bank=incumbent_bank,
    )
    rendered = markdown(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
