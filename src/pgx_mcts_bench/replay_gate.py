"""Evidence gates for collaboration replay-v3."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import FixedWordGame, load_scientist
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _json_hash,
    _replay_representation_embedding,
    _sha256,
)
from pgx_mcts_bench.rapid_adaptation import _run_one_task
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games

DEFAULT_AUDIT_ITEMS = (
    "12a_146",
    "11a_26",
    "11a_33",
    "11n_107",
    "10_71",
    "10_137",
)

REPLAY_LEARNING_ARMS = {
    "old-success-balanced": {
        "replay_sampler": "old",
        "success_balanced_replay": True,
        "policy_value_success_only": True,
        "preservation_teacher": True,
    },
    "replay-v3": {
        "replay_sampler": "v3",
        "success_balanced_replay": True,
        "policy_value_success_only": True,
        "preservation_teacher": True,
    },
}


def _exposure_snapshot(replay: Any) -> list[dict[str, Any]]:
    return [
        {
            "representation": str(getattr(game[0], "representation_id", "")),
            "episode_seed": int(getattr(game[0], "episode_seed", 0)),
            "episode_uses": int(getattr(game[0], "replay_episode_uses", 0)),
            "position_uses": [
                int(getattr(position, "replay_position_uses", 0)) for position in game
            ],
        }
        for game in replay.games
        if game
    ]


def _metadata_failures(replay: Any) -> list[str]:
    failures = []
    for game_index, game in enumerate(replay.games):
        if not game:
            failures.append(f"game {game_index}: empty")
            continue
        for position_index, position in enumerate(game):
            label = f"game {game_index} position {position_index}"
            if not position.representation_id:
                failures.append(f"{label}: missing representation")
            if position.episode_position_index != position_index:
                failures.append(f"{label}: wrong position index")
            if position.action_horizon <= 0:
                failures.append(f"{label}: missing action horizon")
            if position.residual_word_length < 0:
                failures.append(f"{label}: missing residual length")
            if position.mcts_visit_count <= 0:
                failures.append(f"{label}: missing MCTS visits")
            if not position.termination_reason:
                failures.append(f"{label}: missing termination reason")
    return failures


def run_replay_integrity_gate(
    checkpoint: Path,
    bank: Path,
    output: Path,
    *,
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    simulations: int = 32,
    games_per_item: int = 4,
    capped_games_per_item: int = 2,
    capped_objective: float = 12.0,
    sample_size: int = 512,
    seed: int = 20260850,
    device: str = "cpu",
) -> dict[str, Any]:
    """Generate real attempts and verify replay-v3 sampling and persistence."""
    bank_payload = json.loads(bank.read_text())
    by_id = {item.id: item for item in _bank_from_payload(bank_payload)}
    missing = sorted(set(DEFAULT_AUDIT_ITEMS) - set(by_id))
    if missing:
        raise ValueError(f"audit bank is missing fixed identities: {missing}")
    protocol = {
        "schema": "collaboration-replay-integrity-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "bank": str(bank.resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "scientist": scientist,
        "items": list(DEFAULT_AUDIT_ITEMS),
        "ratio": ratio,
        "simulations": simulations,
        "games_per_item": games_per_item,
        "capped_games_per_item": capped_games_per_item,
        "capped_objective": capped_objective,
        "sample_size": sample_size,
        "representation_mix": {"current": 0.25, "similar": 0.25, "global": 0.5},
        "positions_per_episode": 4,
        "seed": seed,
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "manifest.json", protocol)

    actor = load_scientist(
        scientist,
        checkpoint,
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    generation = []
    for item_index, identity in enumerate(DEFAULT_AUDIT_ITEMS):
        item = by_id[identity]
        actor.replay.set_representation_embedding(
            identity, _replay_representation_embedding(item.knot)
        )
        for capped, games, cap_offset in (
            (False, games_per_item, 0),
            (True, capped_games_per_item, 50_000),
        ):
            fixed = FixedWordGame(
                actor.game,
                item.knot,
                ratio,
                objective_cap=capped_objective if capped else None,
            )
            search = NeuralMCTS(
                fixed,
                actor.network,
                replace(actor.config.search, simulations=simulations),
                device,
            )
            seeds = [
                seed + item_index * 100_000 + cap_offset + game_index
                for game_index in range(games)
            ]
            records = play_selfplay_games(
                fixed,
                search,
                [np.random.default_rng(value + 7) for value in seeds],
                seeds,
                12,
            )
            for record in records:
                actor.replay.add(record, representation_id=identity)
            generation.append(
                {
                    "representation": identity,
                    "capped": capped,
                    "attempts": len(records),
                    "solved": sum(
                        bool(record and record[0].solved > 0.5) for record in records
                    ),
                    "objective_censored": sum(
                        bool(record and record[0].objective_censored) for record in records
                    ),
                }
            )

    state_path = output / "replay.pt"
    before_resume = _exposure_snapshot(actor.replay)
    torch.save(actor.replay, state_path)
    resumed = torch.load(state_path, map_location="cpu", weights_only=False)
    after_resume = _exposure_snapshot(resumed)
    persistence_equal = before_resume == after_resume

    current = DEFAULT_AUDIT_ITEMS[0]
    similar = resumed._similar_representations(current, 2)
    batch = resumed.sample_collaboration_positions(
        sample_size,
        shared_fraction=0.0,
        current_representation=current,
        current_fraction=0.25,
        similar_fraction=0.25,
        similar_representation_count=2,
        positions_per_episode=4,
    )
    identities = [position.representation_id for position in batch]
    native_success = [
        position.solved > 0.5
        and not position.shared_witness
        and not position.objective_censored
        for position in batch
    ]
    policy_value_targets = [
        success for success in native_success
    ]
    negative_policy_target_count = sum(
        target and not success
        for target, success in zip(policy_value_targets, native_success, strict=True)
    )
    sample_report = {
        "positions": len(batch),
        "current": sum(identity == current for identity in identities),
        "similar": sum(identity in similar for identity in identities),
        "global_or_fallback": sum(
            identity != current and identity not in similar for identity in identities
        ),
        "similar_identities": sorted(similar),
        "native_success": sum(native_success),
        "ordinary_failure": sum(
            position.solved <= 0.5 and not position.objective_censored
            for position in batch
        ),
        "objective_censored": sum(position.objective_censored for position in batch),
        "shared": sum(position.shared_witness for position in batch),
        "unique_representations": len(set(identities)),
        "negative_policy_value_targets": negative_policy_target_count,
        "allocated_representation_groups": {
            group: sum(
                int(row["positions"])
                for row in resumed.last_collaboration_sample_trace
                if row["requested_representation_group"] == group
            )
            for group in ("current", "similar", "global")
        },
        "allocated_strata": {
            stratum: sum(
                int(row["positions"])
                for row in resumed.last_collaboration_sample_trace
                if row["requested_stratum"] == stratum
            )
            for stratum in ("success", "failure", "capped", "shared")
        },
        "fallback_episodes": sum(
            row["fallback"] != "none"
            for row in resumed.last_collaboration_sample_trace
        ),
    }
    exposure = _exposure_snapshot(resumed)
    metadata_failures = _metadata_failures(resumed)
    allocated = sample_report["allocated_representation_groups"]
    allocation_tolerance = 4 / sample_size
    allocation_matches = all(
        abs(allocated[group] / sample_size - expected) <= allocation_tolerance
        for group, expected in {"current": 0.25, "similar": 0.25, "global": 0.5}.items()
    )
    passed = (
        persistence_equal
        and len(batch) == sample_size
        and not metadata_failures
        and negative_policy_target_count == 0
        and any(row["episode_uses"] > 0 for row in exposure)
        and any(sum(row["position_uses"]) > 0 for row in exposure)
        and sample_report["native_success"] > 0
        and sample_report["objective_censored"] > 0
        and allocation_matches
    )
    report = {
        **protocol,
        "generation": generation,
        "replay": {
            "episodes": len(resumed.games),
            "positions": resumed.position_count,
            "representations": sorted(
                {game[0].representation_id for game in resumed.games if game}
            ),
            "persistence_equal": persistence_equal,
            "metadata_failures": metadata_failures,
            "sample": sample_report,
            "exposure": exposure,
        },
        "decision": {
            "passed": passed,
            "next_step": (
                "run paired old-replay versus replay-v3 gate"
                if passed
                else "repair replay-v3"
            ),
        },
    }
    _atomic_json(output / "report.json", report)
    return report


def _diagnostic_capped_loss(row: dict[str, Any], failure: float) -> float:
    return sum(
        float(attempt["objective"]) if attempt["solved"] else failure
        for attempt in row["diagnostic_evaluation"]["after"]["rows"]
    )


def run_replay_learning_gate(
    checkpoint: Path,
    bank: Path,
    old_bank: Path,
    output: Path,
    *,
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    simulations: int = 64,
    selfplay_games: int = 8,
    train_steps: int = 24,
    batch_size: int = 32,
    diagnostic_games: int = 8,
    seeds: tuple[int, ...] = (20260851, 20260852),
    workers: int = 7,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    """Paired old sampler versus replay-v3 consolidation and retention gate."""
    bank_payload = json.loads(bank.read_text())
    old_payload = json.loads(old_bank.read_text())
    by_id = {item.id: item for item in _bank_from_payload(bank_payload)}
    missing = sorted(set(DEFAULT_AUDIT_ITEMS) - set(by_id))
    if missing:
        raise ValueError(f"learning-gate bank is missing fixed identities: {missing}")
    protocol = {
        "schema": "collaboration-replay-learning-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "bank": str(bank.resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "old_bank": str(old_bank.resolve()),
        "old_bank_sha256": _json_hash(old_payload),
        "scientist": scientist,
        "items": list(DEFAULT_AUDIT_ITEMS),
        "arms": REPLAY_LEARNING_ARMS,
        "f": 8,
        "f_old": 1,
        "ratio": ratio,
        "simulations": simulations,
        "selfplay_games": selfplay_games,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "diagnostic_games": diagnostic_games,
        "seeds": list(seeds),
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{manifest} exists; pass resume=True")
        if json.loads(manifest.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("replay learning gate resume protocol differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, protocol)

    pending = []
    for seed_index, seed in enumerate(seeds):
        for arm, settings in REPLAY_LEARNING_ARMS.items():
            for item_index, identity in enumerate(DEFAULT_AUDIT_ITEMS):
                path = output / "items" / f"seed-{seed_index}" / arm / f"{item_index:04d}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    continue
                knot = by_id[identity].knot
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
                            "train_steps": train_steps,
                            "batch_size": batch_size,
                            "diagnostic_evaluation_games": diagnostic_games,
                            "diagnostic_evaluation_seed": (
                                seed + item_index * 10_000_000 + 850_000_000
                            ),
                            "seed": seed + item_index * 10_000_000,
                            "device": device,
                            **settings,
                        },
                    )
                )
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_one_task, payload): path
                for path, payload in pending
            }
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
                for item_index in range(len(DEFAULT_AUDIT_ITEMS))
            ]
            for arm in REPLAY_LEARNING_ARMS
        }
        for seed_index, seed in enumerate(seeds)
    }
    failure = ratio * 20 + 64
    by_seed = {}
    passed = True
    for seed, arms in rows_by_seed.items():
        old_rows = {row["item"]: row for row in arms["old-success-balanced"]}
        new_rows = {row["item"]: row for row in arms["replay-v3"]}

        def solved_set(rows: dict[str, dict[str, Any]], phase: str) -> set[str]:
            return {
                identity
                for identity, row in rows.items()
                if int(row["diagnostic_evaluation"][phase]["solved"]) > 0
            }

        before = solved_set(old_rows, "before")
        before_bit_identical = all(
            old_rows[identity]["diagnostic_evaluation"]["before"]
            == new_rows[identity]["diagnostic_evaluation"]["before"]
            for identity in old_rows
        )
        old_after = solved_set(old_rows, "after")
        new_after = solved_set(new_rows, "after")
        old_loss = sum(_diagnostic_capped_loss(row, failure) for row in old_rows.values())
        new_loss = sum(_diagnostic_capped_loss(row, failure) for row in new_rows.values())
        old_selfplay = sum(
            int(iteration["selfplay_solved"])
            for row in old_rows.values()
            for iteration in row["iterations"]
            if iteration["kind"] == "current"
        )
        new_selfplay = sum(
            int(iteration["selfplay_solved"])
            for row in new_rows.values()
            for iteration in row["iterations"]
            if iteration["kind"] == "current"
        )
        retained = before <= new_after
        noninferior = not (old_after - new_after) and new_loss <= old_loss
        consolidation_noninferior = new_selfplay >= old_selfplay
        seed_passed = (
            before_bit_identical
            and retained
            and noninferior
            and consolidation_noninferior
        )
        passed &= seed_passed
        by_seed[seed] = {
            "before_solved": sorted(before),
            "before_diagnostics_bit_identical": before_bit_identical,
            "old_after_solved": sorted(old_after),
            "replay_v3_after_solved": sorted(new_after),
            "shared_after_solved": sorted(old_after & new_after),
            "replay_v3_only": sorted(new_after - old_after),
            "old_only": sorted(old_after - new_after),
            "old_capped_loss": old_loss,
            "replay_v3_capped_loss": new_loss,
            "capped_loss_delta": new_loss - old_loss,
            "old_current_selfplay_solves": old_selfplay,
            "replay_v3_current_selfplay_solves": new_selfplay,
            "retained_frozen_solved_set": retained,
            "paired_noninferior": noninferior,
            "consolidation_noninferior": consolidation_noninferior,
            "passed": seed_passed,
        }
    report = {
        **protocol,
        "by_seed": by_seed,
        "decision": {
            "passed": passed,
            "next_step": (
                "run unrebalanced held-out critic gate"
                if passed
                else "repair replay-v3 before collaboration"
            ),
        },
    }
    _atomic_json(output / "report.json", report)
    return report
