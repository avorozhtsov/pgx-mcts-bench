"""Small paired admission gate for success-only controller learning."""

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

RETENTION_ITEMS = ("12a_146", "11a_26", "11a_33")
FROZEN_NEVER_ITEMS = ("11n_107", "10_71", "10_137")
ARMS: dict[str, dict[str, Any]] = {
    "frozen": {"train_steps": 0},
    "split-success-24": {
        "train_steps": 24,
        "preservation_teacher": True,
        "success_balanced_replay": True,
        "policy_value_success_only": True,
    },
}


def analyze_split_loss_gate(
    rows_by_seed: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    ratio: float,
    move_budget: int,
) -> dict[str, Any]:
    failure = ratio * 20 + move_budget
    by_seed = {}
    all_noninferior = True
    rescue_counts = {item: 0 for item in FROZEN_NEVER_ITEMS}
    for seed, rows_by_arm in rows_by_seed.items():
        frozen_rows = {row["item"]: row for row in rows_by_arm["frozen"]}
        split_rows = {row["item"]: row for row in rows_by_arm["split-success-24"]}
        frozen = {item for item, row in frozen_rows.items() if row["solved"]}
        split = {item for item, row in split_rows.items() if row["solved"]}
        treatment_only = sorted(split - frozen)
        control_only = sorted(frozen - split)
        rescues = sorted(
            item
            for item in treatment_only
            if item in FROZEN_NEVER_ITEMS
            and split_rows[item].get("first_solve")
            and not split_rows[item]["first_solve"]["before_any_training"]
        )
        for item in rescues:
            rescue_counts[item] += 1

        def capped(rows: dict[str, dict[str, Any]]) -> float:
            return sum(
                float(row["best_objective"]) if row["solved"] else failure
                for row in rows.values()
            )

        frozen_capped = capped(frozen_rows)
        split_capped = capped(split_rows)
        noninferior = not control_only and split_capped <= frozen_capped
        all_noninferior &= noninferior
        critic_only_steps = sum(
            int(iteration.get("train_steps_without_policy_value_targets", 0))
            for row in split_rows.values()
            for iteration in row["iterations"]
        )
        frozen_selfplay = {
            item: sum(
                int(iteration["selfplay_solved"])
                for iteration in row["iterations"]
                if iteration["kind"] == "current"
            )
            for item, row in frozen_rows.items()
        }
        split_selfplay = {
            item: sum(
                int(iteration["selfplay_solved"])
                for iteration in row["iterations"]
                if iteration["kind"] == "current"
            )
            for item, row in split_rows.items()
        }
        by_seed[seed] = {
            "frozen_solved": sorted(frozen),
            "split_solved": sorted(split),
            "shared_solved": sorted(frozen & split),
            "treatment_only": treatment_only,
            "control_only": control_only,
            "frozen_never_rescues": rescues,
            "frozen_capped_objective": frozen_capped,
            "split_capped_objective": split_capped,
            "capped_objective_delta": split_capped - frozen_capped,
            "split_critic_only_optimizer_steps": critic_only_steps,
            "frozen_selfplay_solves_by_item": frozen_selfplay,
            "split_selfplay_solves_by_item": split_selfplay,
            "frozen_selfplay_solves": sum(frozen_selfplay.values()),
            "split_selfplay_solves": sum(split_selfplay.values()),
            "noninferior": noninferior,
        }
    replicated_rescues = sorted(item for item, count in rescue_counts.items() if count >= 2)
    passed = all_noninferior and bool(replicated_rescues)
    return {
        "by_seed": by_seed,
        "rescue_seed_counts": rescue_counts,
        "decision": {
            "passed": passed,
            "all_seeds_solved_set_and_objective_noninferior": all_noninferior,
            "replicated_frozen_never_rescues": replicated_rescues,
            "required_rescue_seed_count": 2,
            "next_step": (
                "run the source-disjoint 20-representation split-loss gate"
                if passed
                else "keep the 20- and 200-representation gates closed"
            ),
        },
    }


def run_split_loss_gate(
    checkpoint: Path,
    bank: Path,
    old_bank: Path,
    output: Path,
    *,
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    simulations: int = 64,
    selfplay_games: int = 8,
    batch_size: int = 32,
    seeds: tuple[int, ...] = (20261420, 20261421, 20261422),
    workers: int = 7,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    bank_payload = json.loads(bank.read_text())
    old_payload = json.loads(old_bank.read_text())
    by_id = {item.id: item.knot for item in _bank_from_payload(bank_payload)}
    selected_ids = RETENTION_ITEMS + FROZEN_NEVER_ITEMS
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise ValueError(f"gate bank is missing fixed identities: {missing}")
    gate = checkpoint_regression_gate(
        checkpoint,
        scientist,
        ratio=ratio,
        games=12,
        minimum_solve_rate=0.8,
        seed=seeds[0] + 700_000_000,
        simulations=simulations,
        device=device,
    )
    protocol = {
        "schema": "split-loss-gate-v1",
        "scientist": scientist,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_gate": gate,
        "bank": str(bank.resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "old_bank": str(old_bank.resolve()),
        "old_bank_sha256": _json_hash(old_payload),
        "selection": {
            "retention_items": list(RETENTION_ITEMS),
            "frozen_never_items": list(FROZEN_NEVER_ITEMS),
            "frozen_never_definition": (
                "zero solves in 128 frozen self-play attempts over the prior two-seed F=8 gate"
            ),
        },
        "arms": ARMS,
        "f": 8,
        "f_old": 1,
        "ratio": ratio,
        "simulations": simulations,
        "selfplay_games": selfplay_games,
        "batch_size": batch_size,
        "seeds": list(seeds),
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{manifest} exists; pass resume=True")
        if json.loads(manifest.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("split-loss gate resume protocol differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, protocol)

    pending = []
    for seed_index, seed in enumerate(seeds):
        for arm, settings in ARMS.items():
            for item_index, item_id in enumerate(selected_ids):
                path = output / "items" / f"seed-{seed_index}" / arm / f"{item_index:04d}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    continue
                knot = by_id[item_id]
                pending.append(
                    (
                        path,
                        {
                            "index": item_index,
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
                            "f": 8,
                            "f_old": 1,
                            "simulations": simulations,
                            "selfplay_games": selfplay_games,
                            "batch_size": batch_size,
                            "seed": seed + item_index * 10_000_000,
                            "device": device,
                            **settings,
                        },
                    )
                )
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one_task, payload): path for path, payload in pending}
            for future in as_completed(futures):
                _atomic_json(futures[future], future.result())

    rows_by_seed = {
        str(seed): {
            arm: [
                json.loads(
                    (
                        output
                        / "items"
                        / f"seed-{seed_index}"
                        / arm
                        / f"{item_index:04d}.json"
                    ).read_text()
                )
                for item_index in range(len(selected_ids))
            ]
            for arm in ARMS
        }
        for seed_index, seed in enumerate(seeds)
    }
    report = {
        **protocol,
        **analyze_split_loss_gate(rows_by_seed, ratio=ratio, move_budget=int(gate["move_budget"])),
    }
    _atomic_json(output / "report.json", report)
    return report


def write_split_loss_early_stop_report(output: Path, completed_seeds: int) -> dict[str, Any]:
    """Analyze complete seed blocks after an irreversible preregistered failure."""
    protocol = json.loads((output / "manifest.json").read_text())
    seeds = tuple(int(seed) for seed in protocol["seeds"][:completed_seeds])
    selected_ids = RETENTION_ITEMS + FROZEN_NEVER_ITEMS
    rows_by_seed = {
        str(seed): {
            arm: [
                json.loads(
                    (
                        output
                        / "items"
                        / f"seed-{seed_index}"
                        / arm
                        / f"{item_index:04d}.json"
                    ).read_text()
                )
                for item_index in range(len(selected_ids))
            ]
            for arm in ARMS
        }
        for seed_index, seed in enumerate(seeds)
    }
    analysis = analyze_split_loss_gate(
        rows_by_seed,
        ratio=float(protocol["ratio"]),
        move_budget=int(protocol["checkpoint_gate"]["move_budget"]),
    )
    analysis["decision"].update(
        passed=False,
        early_stopped=True,
        completed_seeds=completed_seeds,
        declared_seeds=len(protocol["seeds"]),
        reason=(
            "no frozen-never rescue in either complete seed, so a rescue in two seeds "
            "is unreachable with only one declared seed remaining"
        ),
        next_step="keep the 20- and 200-representation gates closed",
    )
    report = {**protocol, **analysis}
    _atomic_json(output / "early-stop-report.json", report)
    return report
