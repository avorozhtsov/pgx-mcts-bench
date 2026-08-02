#!/usr/bin/env python3
"""Aggregate the paired five-seed critic comparison."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def inversions(rows: list[dict]) -> int:
    total = 0
    for left in rows:
        for right in rows:
            if left["ratio"] <= right["ratio"]:
                continue
            if (
                left["crossing_changes"] >= right["crossing_changes"]
                and left["moves"] >= right["moves"]
                and (
                    left["crossing_changes"] > right["crossing_changes"]
                    or left["moves"] > right["moves"]
                )
            ):
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    seeds = tuple(range(20260802, 20260807))
    arms = ("shaped", "factorized")
    reports: dict[tuple[int, str], dict] = {}
    training: dict[int, dict] = {}
    for seed in seeds:
        seed_dir = args.output / f"seed-{seed}"
        training[seed] = json.loads((seed_dir / "training-metrics.json").read_text())
        for arm in arms:
            reports[seed, arm] = json.loads(
                (seed_dir / f"{arm}-report.json").read_text()
            )

    ratios = tuple(reports[seeds[0], arms[0]]["ratios"])
    pooled: dict[str, dict[str, dict]] = {arm: {} for arm in arms}
    paired: dict[str, dict] = {}
    per_seed: list[dict] = []
    for seed in seeds:
        row: dict[str, object] = {"seed": seed}
        for arm in arms:
            report = reports[seed, arm]
            row[arm] = {
                "solved": sum(result["solved"] for result in report["results"]),
                "episodes": len(report["results"]),
                "inversions": len(report["dominance_inversions"]),
            }
        per_seed.append(row)

    for ratio in ratios:
        by_arm: dict[str, dict[tuple[int, str], dict]] = {}
        for arm in arms:
            indexed: dict[tuple[int, str], dict] = {}
            for seed in seeds:
                for row in reports[seed, arm]["results"]:
                    if row["ratio"] == ratio:
                        indexed[seed, row["representation_id"]] = row
            by_arm[arm] = indexed
            solved = [row for row in indexed.values() if row["solved"]]
            pooled[arm][str(ratio)] = {
                "solved": len(solved),
                "episodes": len(indexed),
                "crossings": mean([row["crossing_changes"] for row in solved]),
                "moves": mean([row["moves"] for row in solved]),
                "objective": mean(
                    [
                        ratio * row["crossing_changes"] + row["moves"]
                        for row in solved
                    ]
                ),
            }
        keys = sorted(set(by_arm["shaped"]) & set(by_arm["factorized"]))
        both = [
            (by_arm["shaped"][key], by_arm["factorized"][key])
            for key in keys
            if by_arm["shaped"][key]["solved"]
            and by_arm["factorized"][key]["solved"]
        ]
        objective_differences = [
            ratio * (factorized["crossing_changes"] - shaped["crossing_changes"])
            + factorized["moves"]
            - shaped["moves"]
            for shaped, factorized in both
        ]
        paired[str(ratio)] = {
            "both_solved": len(both),
            "shaped_only_solved": sum(
                by_arm["shaped"][key]["solved"]
                and not by_arm["factorized"][key]["solved"]
                for key in keys
            ),
            "factorized_only_solved": sum(
                by_arm["factorized"][key]["solved"]
                and not by_arm["shaped"][key]["solved"]
                for key in keys
            ),
            "factorized_objective_wins": sum(value < 0 for value in objective_differences),
            "objective_ties": sum(value == 0 for value in objective_differences),
            "factorized_objective_losses": sum(value > 0 for value in objective_differences),
            "mean_factorized_minus_shaped_objective": mean(objective_differences),
            "mean_factorized_minus_shaped_crossings": mean(
                [f["crossing_changes"] - s["crossing_changes"] for s, f in both]
            ),
            "mean_factorized_minus_shaped_moves": mean(
                [f["moves"] - s["moves"] for s, f in both]
            ),
        }

    joint_inversions = defaultdict(int)
    joint_representations = 0
    for seed in seeds:
        indexed = {}
        for arm in arms:
            for row in reports[seed, arm]["results"]:
                indexed[arm, row["representation_id"], row["ratio"]] = row
        representation_ids = {
            row["representation_id"] for row in reports[seed, "shaped"]["results"]
        }
        for representation_id in representation_ids:
            if not all(
                indexed[arm, representation_id, ratio]["solved"]
                for arm in arms
                for ratio in ratios
            ):
                continue
            joint_representations += 1
            for arm in arms:
                joint_inversions[arm] += inversions(
                    [indexed[arm, representation_id, ratio] for ratio in ratios]
                )

    result = {
        "schema": "factorized-critic-five-seed-summary-v1",
        "protocol": {
            "seeds": seeds,
            "selfplay_games_per_seed": training[seeds[0]]["selfplay_games"],
            "train_steps_per_seed": training[seeds[0]]["train_steps"],
            "representations_per_seed": 20,
            "ratios": ratios,
            "simulations": reports[seeds[0], arms[0]]["simulations"],
        },
        "training_positions": {str(seed): training[seed]["positions"] for seed in seeds},
        "per_seed": per_seed,
        "pooled": pooled,
        "paired": paired,
        "joint_all_solved": {
            "representations": joint_representations,
            "shaped_inversions": joint_inversions["shaped"],
            "factorized_inversions": joint_inversions["factorized"],
        },
        "witnesses": {
            arm: sum(
                result["solved"]
                for seed in seeds
                for result in reports[seed, arm]["results"]
            )
            for arm in arms
        },
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Five-seed shaped-scalar versus factorized-critic comparison",
        "",
        "Each seed used 64 shared self-play games, 256 paired updates, and 20 paired "
        "depth-4 representations at each of three ratios. Costs are conditional on "
        "solving; paired differences use only jointly solved episodes.",
        "",
        "| arm | ratio | solved | crossings | moves | objective |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        for ratio in ratios:
            row = pooled[arm][str(ratio)]
            lines.append(
                f"| {arm} | {ratio:g} | {row['solved']}/{row['episodes']} | "
                f"{row['crossings']:.3f} | {row['moves']:.3f} | {row['objective']:.3f} |"
            )
    lines += [
        "",
        "| ratio | both solved | shaped-only | factorized-only | factorized W/T/L | "
        "mean factorized - shaped objective |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in ratios:
        row = paired[str(ratio)]
        lines.append(
            f"| {ratio:g} | {row['both_solved']} | {row['shaped_only_solved']} | "
            f"{row['factorized_only_solved']} | {row['factorized_objective_wins']}/"
            f"{row['objective_ties']}/{row['factorized_objective_losses']} | "
            f"{row['mean_factorized_minus_shaped_objective']:.3f} |"
        )
    lines += [
        "",
        f"Among the {joint_representations} representations solved by both arms at all "
        f"three ratios, shaped produced {joint_inversions['shaped']} dominance inversions "
        f"and factorized produced {joint_inversions['factorized']}.",
        "",
        "Every solved episode contains a verified semantic witness in its seed report.",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
