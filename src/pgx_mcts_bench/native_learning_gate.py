"""Transactional gate for native braid-policy discovery and retention."""

from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import FixedWordGame, KnotItem, load_scientist
from pgx_mcts_bench.collaboration_eval import _evaluation_record
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _json_hash,
    _replay_representation_embedding,
    _sha256,
    verified_record_cost,
)
from pgx_mcts_bench.rapid_adaptation import checkpoint_regression_gate
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games, train_alphazero_step

RETENTION_ITEMS = ("12a_146", "11a_26", "11a_33")
TRANSITION_ITEMS = ("10_149", "12a_1168", "12a_981", "12n_830", "9_28", "11a_106")
FROZEN_NEVER_ITEMS = ("11n_107", "10_71", "10_137")
PANEL_ITEMS = RETENTION_ITEMS + TRANSITION_ITEMS + FROZEN_NEVER_ITEMS
SEARCH_TIERS = ((64, 4), (128, 2), (256, 1))


def _rotated(knot: KnotItem, offset: int) -> KnotItem:
    """Return a closure-equivalent cyclic conjugate of a braid word."""
    if not knot.word:
        return knot
    shift = offset % len(knot.word)
    return KnotItem(
        name=f"{knot.name}#cyclic-{shift}",
        crossings=knot.crossings,
        word=knot.word[shift:] + knot.word[:shift],
        strands=knot.strands,
    )


def _panel_evaluation(
    scientist: Any,
    items: list[Any],
    *,
    ratio: float,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    failure = ratio * 20 + int(scientist.game.config.simplify_budget)
    scheduled = 0
    for index, item in enumerate(items):
        verified, compute = _evaluation_record(
            scientist, item.knot, ratio, simulations, seed + index
        )
        scheduled += int(compute["scheduled_network_evaluations"])
        row = {"solved": verified is not None}
        if verified is not None:
            row.update(
                crossing_changes=verified[0],
                moves=verified[1],
                objective=ratio * verified[0] + verified[1],
            )
        rows[item.id] = row
    solved = {identity for identity, row in rows.items() if row["solved"]}
    return {
        "rows": rows,
        "solved": sorted(solved),
        "capped_objective": sum(
            float(row.get("objective", failure)) for row in rows.values()
        ),
        "failure_cap": failure,
        "scheduled_network_evaluations": scheduled,
    }


def admission_decision(
    before: dict[str, Any], after: dict[str, Any], *, target: str
) -> dict[str, Any]:
    """Require exact retention plus an observable target/objective improvement."""
    before_solved = set(before["solved"])
    after_solved = set(after["solved"])
    lost = sorted(before_solved - after_solved)
    gained = sorted(after_solved - before_solved)
    target_rescue = target in after_solved and target not in before_solved
    objective_delta = float(after["capped_objective"] - before["capped_objective"])
    passed = not lost and objective_delta <= 0.0 and (target_rescue or objective_delta < 0.0)
    return {
        "passed": passed,
        "lost": lost,
        "gained": gained,
        "target_rescue": target_rescue,
        "capped_objective_delta": objective_delta,
    }


def _discover(
    scientist: Any,
    item: Any,
    *,
    ratio: float,
    seed: int,
    tiers: tuple[tuple[int, int], ...],
    device: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    records = []
    rows = []
    initial_length = len(item.knot.word)
    promising = True
    for tier_index, (simulations, games) in enumerate(tiers):
        if tier_index and not promising:
            rows.append(
                {"simulations": simulations, "games": 0, "skipped_not_promising": True}
            )
            continue
        tier_pairs = []
        for game_index in range(games):
            variant = _rotated(item.knot, tier_index + game_index)
            fixed = FixedWordGame(scientist.game, variant, ratio)
            search = NeuralMCTS(
                fixed,
                scientist.network,
                replace(
                    scientist.config.search,
                    simulations=simulations,
                    cpuct=(1.25, 1.5, 2.0)[min(tier_index, 2)],
                ),
                device,
            )
            episode_seed = seed + tier_index * 1_000_000 + game_index
            record = play_selfplay_games(
                fixed,
                search,
                [np.random.default_rng(episode_seed + 7)],
                [episode_seed],
                12,
            )[0]
            for position in record:
                position.representation_id = item.id
            tier_pairs.append((variant, record))
            records.append(record)
        solved = sum(
            verified_record_cost(scientist.game, variant, ratio, record) is not None
            for variant, record in tier_pairs
        )
        residuals = [
            int(record[0].best_residual_word_length)
            for _, record in tier_pairs
            if record and int(record[0].best_residual_word_length) >= 0
        ]
        best_residual = min(residuals, default=initial_length)
        promising = bool(solved) or best_residual < initial_length
        rows.append(
            {
                "simulations": simulations,
                "games": games,
                "solved": solved,
                "best_residual_word_length": best_residual,
                "promising": promising,
                "skipped_not_promising": False,
            }
        )
        if solved:
            break
    return records, rows


def _add_records(scientist: Any, item: Any, records: list[Any]) -> None:
    scientist.replay.set_representation_embedding(
        item.id, _replay_representation_embedding(item.knot)
    )
    for record in records:
        scientist.replay.add(record, representation_id=item.id)


def _run_seed(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    seed = int(payload["seed"])
    scientist = load_scientist(
        payload["scientist"],
        Path(payload["checkpoint"]),
        seed=seed,
        device=payload["device"],
        simulations=payload["evaluation_simulations"],
        require_factorized=True,
        objective_budget_channel=True,
    )
    by_id = {item.id: item for item in _bank_from_payload(payload["bank"])}
    panel = [by_id[identity] for identity in PANEL_ITEMS]
    evaluation_seed = seed + 800_000_000
    initial = _panel_evaluation(
        scientist,
        panel,
        ratio=payload["ratio"],
        simulations=payload["evaluation_simulations"],
        seed=evaluation_seed,
    )

    # Native canary solutions provide rehearsal without an external teacher.
    for index, identity in enumerate(RETENTION_ITEMS):
        records, _ = _discover(
            scientist,
            by_id[identity],
            ratio=payload["ratio"],
            seed=seed + 100_000_000 + index * 10_000,
            tiers=((64, 2),),
            device=payload["device"],
        )
        _add_records(scientist, by_id[identity], records)

    current = initial
    transactions = []
    for target_index, identity in enumerate(TRANSITION_ITEMS + FROZEN_NEVER_ITEMS):
        item = by_id[identity]
        records, search_rows = _discover(
            scientist,
            item,
            ratio=payload["ratio"],
            seed=seed + 200_000_000 + target_index * 10_000_000,
            tiers=tuple(tuple(tier) for tier in payload["search_tiers"]),
            device=payload["device"],
        )
        _add_records(scientist, item, records)
        # The search rows count only records that passed exact witness replay.
        successful = sum(int(row.get("solved", 0)) for row in search_rows)
        transaction: dict[str, Any] = {
            "target": identity,
            "search": search_rows,
            "successful_trajectories": successful,
            "attempted_update": bool(successful),
        }
        if not successful:
            transaction.update(accepted=False, reason="no certified positive trajectory")
            transactions.append(transaction)
            continue

        network_before = copy.deepcopy(scientist.network.state_dict())
        optimizer_before = copy.deepcopy(scientist.optimizer.state_dict())
        losses = []
        for _ in range(payload["train_steps"]):
            losses.append(
                train_alphazero_step(
                    scientist.network,
                    scientist.optimizer,
                    scientist.replay,
                    payload["batch_size"],
                    torch.device(payload["device"]),
                    collaboration_replay=True,
                    shared_fraction=0.0,
                    policy_value_success_only=True,
                    replay_current_representation=identity,
                    replay_current_fraction=0.25,
                    replay_similar_fraction=0.25,
                    replay_positions_per_episode=4,
                    replay_max_position_uses=64,
                )
            )
        candidate = _panel_evaluation(
            scientist,
            panel,
            ratio=payload["ratio"],
            simulations=payload["evaluation_simulations"],
            seed=evaluation_seed,
        )
        decision = admission_decision(current, candidate, target=identity)
        if decision["passed"]:
            current = candidate
        else:
            scientist.network.load_state_dict(network_before)
            scientist.optimizer.load_state_dict(optimizer_before)
        transaction.update(
            accepted=decision["passed"],
            admission=decision,
            optimizer_steps=len(losses),
            last_loss=losses[-1] if losses else None,
        )
        transactions.append(transaction)

    final = _panel_evaluation(
        scientist,
        panel,
        ratio=payload["ratio"],
        simulations=payload["evaluation_simulations"],
        seed=evaluation_seed,
    )
    if final["rows"] != current["rows"]:
        raise AssertionError("final fixed-seed evaluation differs from admitted state")
    initial_solved = set(initial["solved"])
    final_solved = set(final["solved"])
    return {
        "seed": seed,
        "initial": initial,
        "final": final,
        "gained": sorted(final_solved - initial_solved),
        "lost": sorted(initial_solved - final_solved),
        "transactions": transactions,
        "accepted_updates": sum(bool(row.get("accepted")) for row in transactions),
        "wall_seconds": time.perf_counter() - started,
    }


def analyze_native_learning(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rescue_counts = {identity: 0 for identity in TRANSITION_ITEMS + FROZEN_NEVER_ITEMS}
    for row in rows:
        for identity in row["gained"]:
            if identity in rescue_counts:
                rescue_counts[identity] += 1
    replicated = sorted(identity for identity, count in rescue_counts.items() if count >= 2)
    retained = all(not row["lost"] for row in rows)
    objective_noninferior = all(
        row["final"]["capped_objective"] <= row["initial"]["capped_objective"]
        for row in rows
    )
    passed = retained and objective_noninferior and len(replicated) >= 2
    return {
        "rescue_seed_counts": rescue_counts,
        "replicated_rescues": replicated,
        "decision": {
            "passed": passed,
            "exact_retention_all_seeds": retained,
            "capped_objective_noninferior_all_seeds": objective_noninferior,
            "required_replicated_rescues": 2,
            "next_step": (
                "run the 20-representation no-sharing progression smoke"
                if passed
                else "keep schedule, sharing, 200-representation, and cloud gates closed"
            ),
        },
    }


def run_native_learning_gate(
    checkpoint: Path,
    bank: Path,
    output: Path,
    *,
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    evaluation_simulations: int = 64,
    train_steps: int = 24,
    batch_size: int = 32,
    seeds: tuple[int, ...] = (20261520, 20261521, 20261522),
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    bank_payload = json.loads(bank.read_text())
    available = {item.id for item in _bank_from_payload(bank_payload)}
    missing = sorted(set(PANEL_ITEMS) - available)
    if missing:
        raise ValueError(f"native gate bank is missing fixed identities: {missing}")
    gate = checkpoint_regression_gate(
        checkpoint,
        scientist,
        ratio=ratio,
        games=12,
        minimum_solve_rate=0.8,
        seed=seeds[0] + 700_000_000,
        simulations=evaluation_simulations,
        device=device,
    )
    protocol = {
        "schema": "transactional-native-learning-gate-v2",
        "scientist": scientist,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_gate": gate,
        "bank": str(bank.resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "panel": {
            "retention": list(RETENTION_ITEMS),
            "transition": list(TRANSITION_ITEMS),
            "frozen_never": list(FROZEN_NEVER_ITEMS),
        },
        "search_tiers": [list(tier) for tier in SEARCH_TIERS],
        "search_variants": "cyclic braid-word conjugates with tier-specific PUCT",
        "ratio": ratio,
        "evaluation_simulations": evaluation_simulations,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "seeds": list(seeds),
        "device": device,
        "policy_targets": "certified-success-only",
        "failure_use": "critic replay only after a certified positive exists",
        "replay_max_position_uses": 64,
        "admission": (
            "no exact solved-set loss; capped objective nonworse; "
            "target rescue or strict objective gain"
        ),
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{manifest} exists; pass resume=True")
        if json.loads(manifest.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("native-learning gate resume protocol differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, protocol)

    rows_by_seed = {}
    pending = []
    for seed in seeds:
        path = output / "seeds" / f"{seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            rows_by_seed[seed] = json.loads(path.read_text())
            continue
        pending.append((seed, path, {**protocol, "seed": seed, "bank": bank_payload}))
    if pending:
        with ProcessPoolExecutor(max_workers=min(len(pending), 3)) as executor:
            futures = {
                executor.submit(_run_seed, payload): (seed, path)
                for seed, path, payload in pending
            }
            for future in as_completed(futures):
                seed, path = futures[future]
                row = future.result()
                _atomic_json(path, row)
                rows_by_seed[seed] = row
    rows = [rows_by_seed[seed] for seed in seeds]
    report = {
        **protocol,
        "completed_seeds": len(rows),
        "seeds_results": rows,
        **analyze_native_learning(rows),
    }
    _atomic_json(output / "report.json", report)
    return report
