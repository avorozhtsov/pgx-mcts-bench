"""Paired causal gate for disposable rapid adaptation."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _json_hash,
    _sha256,
)
from pgx_mcts_bench.rapid_adaptation import _run_one_task, checkpoint_regression_gate

ARMS = {
    "frozen-8+1": {"f": 8, "f_old": 1, "train_steps": 0},
    "train-8+0": {"f": 8, "f_old": 0, "train_steps": 96},
    "train-8+1": {"f": 8, "f_old": 1, "train_steps": 96},
}


def _summary(rows: list[dict[str, Any]], ratio: float, move_budget: int) -> dict[str, Any]:
    failure = ratio * 20 + move_budget
    solved_rows = [row for row in rows if row["solved"]]
    retention = [item for row in rows for item in row.get("old_retention", [])]
    return {
        "representations": len(rows),
        "solved": len(solved_rows),
        "solve_rate": len(solved_rows) / len(rows),
        "solved_items": sorted(row["item"] for row in solved_rows),
        "capped_objective_sum": sum(
            float(row["best_objective"]) if row["solved"] else failure for row in rows
        ),
        "first_iteration_solves": sum(
            bool((row.get("first_solve") or {}).get("before_any_training")) for row in solved_rows
        ),
        "after_first_iteration_solves": sum(
            bool(row.get("first_solve")) and not bool(row["first_solve"]["before_any_training"])
            for row in solved_rows
        ),
        "retention_solved": sum(bool(item["solved"]) for item in retention),
        "retention_attempts": len(retention),
        "compute": {
            "scheduled_network_evaluations": sum(
                sum(
                    int(row["compute"].get(key, 0))
                    for key in (
                        "adaptation_scheduled_network_evaluations",
                        "final_scheduled_network_evaluations",
                        "retention_scheduled_network_evaluations",
                    )
                )
                for row in rows
            ),
            "training_seconds": sum(float(row["compute"]["training_seconds"]) for row in rows),
            "worker_wall_seconds": sum(float(row["compute"]["wall_seconds"]) for row in rows),
        },
    }


def analyze_paired_gate(
    item_root: Path,
    *,
    seeds: tuple[int, ...],
    ratio: float,
    move_budget: int,
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    post_training_rescue_seeds = 0
    all_noninferior = True
    for seed_index, seed in enumerate(seeds):
        rows_by_arm = {
            arm: [
                json.loads(path.read_text())
                for path in sorted((item_root / f"seed-{seed_index}" / arm).glob("*.json"))
            ]
            for arm in ARMS
        }
        summaries = {arm: _summary(rows, ratio, move_budget) for arm, rows in rows_by_arm.items()}
        solved_sets = {
            arm: {row["item"] for row in rows if row["solved"]} for arm, rows in rows_by_arm.items()
        }
        treatment_rows = {row["item"]: row for row in rows_by_arm["train-8+1"]}
        frozen = solved_sets["frozen-8+1"]
        rescues = sorted(
            item
            for item in solved_sets["train-8+1"] - frozen
            if treatment_rows[item].get("first_solve")
            and not treatment_rows[item]["first_solve"]["before_any_training"]
        )
        if rescues:
            post_training_rescue_seeds += 1
        comparisons = {}
        seed_noninferior = True
        for control in ("frozen-8+1", "train-8+0"):
            treatment = summaries["train-8+1"]
            baseline = summaries[control]
            noninferior = (
                treatment["solved"] >= baseline["solved"]
                and treatment["capped_objective_sum"] <= baseline["capped_objective_sum"]
            )
            seed_noninferior &= noninferior
            comparisons[control] = {
                "shared_solved": sorted(solved_sets["train-8+1"] & solved_sets[control]),
                "treatment_only": sorted(solved_sets["train-8+1"] - solved_sets[control]),
                "control_only": sorted(solved_sets[control] - solved_sets["train-8+1"]),
                "solve_delta": treatment["solved"] - baseline["solved"],
                "capped_objective_delta": (
                    treatment["capped_objective_sum"] - baseline["capped_objective_sum"]
                ),
                "noninferior": noninferior,
            }
        retention_noninferior = (
            summaries["train-8+1"]["retention_solved"]
            >= summaries["frozen-8+1"]["retention_solved"]
        )
        seed_noninferior &= retention_noninferior
        all_noninferior &= seed_noninferior
        by_seed[str(seed)] = {
            "arms": summaries,
            "comparisons": comparisons,
            "post_training_rescues_vs_frozen": rescues,
            "retention_noninferior": retention_noninferior,
            "passes_noninferiority": seed_noninferior,
        }
    passed = all_noninferior and post_training_rescue_seeds >= 2
    return {
        "by_seed": by_seed,
        "decision": {
            "passed": passed,
            "all_seeds_noninferior": all_noninferior,
            "seeds_with_post_training_rescue": post_training_rescue_seeds,
            "required_rescue_seeds": 2,
            "next_step": (
                "expand to the adaptive 200 run" if passed else "do not run adaptive 200"
            ),
        },
    }


def run_paired_rapid_gate(
    checkpoint: Path,
    bank: Path,
    old_bank: Path,
    output: Path,
    *,
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    simulations: int = 128,
    selfplay_games: int = 8,
    batch_size: int = 32,
    seeds: tuple[int, ...] = (20260980, 20260981, 20260982),
    workers: int = 7,
    device: str = "cpu",
    resume: bool = False,
    gate_games: int = 12,
    gate_min_solve_rate: float = 0.8,
) -> dict[str, Any]:
    bank_payload = json.loads(bank.read_text())
    old_payload = json.loads(old_bank.read_text())
    items = _bank_from_payload(bank_payload)
    checkpoint_gate = checkpoint_regression_gate(
        checkpoint,
        scientist,
        ratio=ratio,
        games=gate_games,
        minimum_solve_rate=gate_min_solve_rate,
        seed=seeds[0] + 700_000_000,
        simulations=simulations,
        device=device,
    )
    move_budget = int(checkpoint_gate["move_budget"])
    protocol = {
        "schema": "paired-rapid-gate-v1",
        "scientist": scientist,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_gate": checkpoint_gate,
        "bank": str(bank.resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "old_bank": str(old_bank.resolve()),
        "old_bank_sha256": _json_hash(old_payload),
        "arms": ARMS,
        "ratio": ratio,
        "simulations": simulations,
        "selfplay_games": selfplay_games,
        "batch_size": batch_size,
        "seeds": list(seeds),
        "move_budget": move_budget,
        "workers": workers,
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{manifest} exists; pass resume=True")
        if json.loads(manifest.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("paired-gate resume protocol differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, protocol)

    pending = []
    item_root = output / "items"
    for seed_index, seed in enumerate(seeds):
        for arm, settings in ARMS.items():
            arm_dir = item_root / f"seed-{seed_index}" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            for index, bank_item in enumerate(items):
                path = arm_dir / f"{index:04d}.json"
                if path.exists():
                    continue
                knot = bank_item.knot
                pending.append(
                    (
                        path,
                        {
                            "index": index,
                            "target": {
                                "name": knot.name,
                                "crossings": knot.crossings,
                                "word": knot.word,
                                "strands": knot.strands,
                            },
                            "old_bank": old_payload,
                            "scientist": scientist,
                            "checkpoint": str(checkpoint),
                            "ratio": ratio,
                            "f": settings["f"],
                            "f_old": settings["f_old"],
                            "simulations": simulations,
                            "selfplay_games": selfplay_games,
                            "train_steps": settings["train_steps"],
                            "batch_size": batch_size,
                            "seed": seed + index * 10_000_000,
                            "device": device,
                        },
                    )
                )
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one_task, payload): path for path, payload in pending}
            for future in as_completed(futures):
                _atomic_json(futures[future], future.result())

    analysis = analyze_paired_gate(
        item_root,
        seeds=seeds,
        ratio=ratio,
        move_budget=move_budget,
    )
    report = {**protocol, **analysis}
    _atomic_json(output / "report.json", report)
    return report
