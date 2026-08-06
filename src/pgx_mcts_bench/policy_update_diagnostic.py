"""Matched experiment for task-local policy-update failure modes."""

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

ARMS: dict[str, dict[str, Any]] = {
    "frozen": {"train_steps": 0},
    "legacy-96": {"train_steps": 96},
    "guarded-96": {"train_steps": 96, "preservation_teacher": True},
    "guarded-24": {"train_steps": 24, "preservation_teacher": True},
    "success-gated-24": {
        "train_steps": 24,
        "preservation_teacher": True,
        "require_replay_success": True,
        "success_balanced_replay": True,
    },
}


def analyze_policy_update_rows(rows_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Summarize paired pre/post evaluations and actual optimizer exposure."""
    summaries = {}
    for arm, rows in rows_by_arm.items():
        before = sum(row["diagnostic_evaluation"]["before"]["solved"] for row in rows)
        after = sum(row["diagnostic_evaluation"]["after"]["solved"] for row in rows)
        applied = sum(
            iteration["train_steps_applied"]
            for row in rows
            for iteration in row["iterations"]
        )
        skipped = sum(
            bool(iteration["train_skipped_no_success"])
            for row in rows
            for iteration in row["iterations"]
        )
        selfplay_solved = sum(
            int(iteration["selfplay_solved"])
            for row in rows
            for iteration in row["iterations"]
            if iteration["kind"] == "current"
        )
        current_sequences = [
            [
                int(iteration["selfplay_solved"])
                for iteration in row["iterations"]
                if iteration["kind"] == "current"
            ]
            for row in rows
        ]
        selfplay_by_ordinal = [
            sum(sequence[ordinal] for sequence in current_sequences)
            for ordinal in range(max(map(len, current_sequences), default=0))
        ]
        preservation = [
            float(iteration["last_loss"]["preservation_policy"])
            for row in rows
            for iteration in row["iterations"]
            if iteration["last_loss"] is not None
        ]
        summaries[arm] = {
            "seeds": len(rows),
            "paired_evaluation_games": sum(
                row["diagnostic_evaluation"]["before"]["games"] for row in rows
            ),
            "before_solved": before,
            "after_solved": after,
            "solve_delta": after - before,
            "task_selfplay_solves": selfplay_solved,
            "task_selfplay_solves_by_ordinal": selfplay_by_ordinal,
            "optimizer_steps_applied": applied,
            "iterations_skipped_no_success": skipped,
            "last_step_preservation_policy_mean": (
                sum(preservation) / len(preservation) if preservation else None
            ),
        }
    reference = rows_by_arm["frozen"]
    baseline_consistent = all(
        row["diagnostic_evaluation"]["before"]["rows"]
        == reference[index]["diagnostic_evaluation"]["before"]["rows"]
        for rows in rows_by_arm.values()
        for index, row in enumerate(rows)
    )
    guarded = summaries["guarded-24"]
    gated = summaries["success-gated-24"]
    legacy = summaries["legacy-96"]
    decision = {
        "baseline_bitwise_consistent": baseline_consistent,
        "legacy_update_degrades_paired_solve_count": legacy["solve_delta"] < 0,
        "legacy_update_reduces_matched_selfplay_solves": (
            legacy["task_selfplay_solves"] < summaries["frozen"]["task_selfplay_solves"]
        ),
        "guarded_lower_dose_noninferior_to_frozen": (
            guarded["after_solved"] >= summaries["frozen"]["after_solved"]
        ),
        "success_gating_noninferior_to_frozen": (
            gated["after_solved"] >= summaries["frozen"]["after_solved"]
        ),
    }
    return {"arms": summaries, "decision": decision}


def run_policy_update_diagnostic(
    checkpoint: Path,
    bank: Path,
    old_bank: Path,
    output: Path,
    *,
    target: str = "11a_33",
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    simulations: int = 64,
    selfplay_games: int = 8,
    batch_size: int = 32,
    evaluation_games: int = 8,
    seeds: tuple[int, ...] = (20261361, 20261362, 20261363),
    workers: int = 7,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    bank_payload = json.loads(bank.read_text())
    old_payload = json.loads(old_bank.read_text())
    matches = [item.knot for item in _bank_from_payload(bank_payload) if item.id == target]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one target {target!r}, found {len(matches)}")
    knot = matches[0]
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
        "schema": "policy-update-diagnostic-v1",
        "scientist": scientist,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_gate": gate,
        "bank": str(bank.resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "old_bank": str(old_bank.resolve()),
        "old_bank_sha256": _json_hash(old_payload),
        "target": target,
        "arms": ARMS,
        "f": 8,
        "f_old": 1,
        "ratio": ratio,
        "simulations": simulations,
        "selfplay_games": selfplay_games,
        "batch_size": batch_size,
        "evaluation_games": evaluation_games,
        "seeds": list(seeds),
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{manifest} exists; pass resume=True")
        if json.loads(manifest.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("policy-update diagnostic resume protocol differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, protocol)

    pending = []
    for seed_index, seed in enumerate(seeds):
        for arm, settings in ARMS.items():
            path = output / "items" / f"seed-{seed_index}" / f"{arm}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                continue
            pending.append(
                (
                    path,
                    {
                        "index": 0,
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
                        "seed": seed,
                        "device": device,
                        "diagnostic_evaluation_games": evaluation_games,
                        # This is the same seed family used by the historical
                        # one-shot final evaluation, so a reported lost solve
                        # is present in the paired pre-update control.
                        "diagnostic_evaluation_seed": seed + 900_000_000,
                        **settings,
                    },
                )
            )
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one_task, payload): path for path, payload in pending}
            for future in as_completed(futures):
                _atomic_json(futures[future], future.result())

    rows_by_arm = {
        arm: [
            json.loads((output / "items" / f"seed-{seed_index}" / f"{arm}.json").read_text())
            for seed_index in range(len(seeds))
        ]
        for arm in ARMS
    }
    report = {**protocol, **analyze_policy_update_rows(rows_by_arm)}
    _atomic_json(output / "report.json", report)
    return report
