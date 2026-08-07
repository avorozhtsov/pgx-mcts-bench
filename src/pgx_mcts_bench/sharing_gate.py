"""Bounded-option transfer and retention gate for collaboration scientists."""

from __future__ import annotations

import copy
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import (
    BUDGET_PRESERVATION_WEIGHTS,
    FixedWordGame,
    load_scientist,
)
from pgx_mcts_bench.collaboration_eval import _evaluation_record
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _json_hash,
    _replay_representation_embedding,
    _sha256,
    translate_semantic_record,
    verified_record_cost,
)
from pgx_mcts_bench.data import Position
from pgx_mcts_bench.distill import (
    stable_option_route_loss,
    train_bounded_option_step,
)
from pgx_mcts_bench.networks import load_policy_value_state_dict
from pgx_mcts_bench.replay_gate import DEFAULT_AUDIT_ITEMS
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.serial_braid import SerialBraidGame
from pgx_mcts_bench.training import (
    attach_policy_value_preservation_teacher,
    play_selfplay_games,
    train_alphazero_step,
)

OPTION_ADAPTER_LEARNING_RATE = 1e-3


def _evaluate(
    scientist: Any,
    items: list[Any],
    *,
    ratio: float,
    simulations: int,
    games: int,
    seed: int,
) -> dict[str, Any]:
    rows = []
    for item_index, item in enumerate(items):
        attempts = []
        for game_index in range(games):
            verified, compute = _evaluation_record(
                scientist,
                item.knot,
                ratio,
                simulations,
                seed + item_index * 10_000 + game_index,
            )
            attempts.append(
                {
                    "solved": verified is not None,
                    "crossing_changes": verified[0] if verified is not None else None,
                    "moves": verified[1] if verified is not None else None,
                    "objective": (
                        ratio * verified[0] + verified[1] if verified is not None else None
                    ),
                    "compute": compute,
                }
            )
        solved_attempts = [row for row in attempts if row["solved"]]
        rows.append(
            {
                "item": item.id,
                "solved": bool(solved_attempts),
                "solved_attempts": len(solved_attempts),
                "best_objective": (
                    min(float(row["objective"]) for row in solved_attempts)
                    if solved_attempts
                    else None
                ),
                "attempts": attempts,
            }
        )
    return {"rows": rows, "solved_set": sorted(row["item"] for row in rows if row["solved"])}


def _evaluate_item_worker(
    name: str,
    checkpoint: str,
    network_state: dict[str, torch.Tensor],
    item: Any,
    ratio: float,
    simulations: int,
    games: int,
    seed: int,
    device: str,
    adapter_enabled: bool,
    objective_budget_channel: bool,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    scientist = load_scientist(
        name,
        Path(checkpoint),
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=objective_budget_channel,
    )
    load_policy_value_state_dict(scientist.network, network_state)
    scientist.network.option_adapter_enabled = adapter_enabled
    return _evaluate(
        scientist,
        [item],
        ratio=ratio,
        simulations=simulations,
        games=games,
        seed=seed,
    )["rows"][0]


def _evaluate_parallel(
    scientist: Any,
    items: list[Any],
    *,
    ratio: float,
    simulations: int,
    games: int,
    seed: int,
    device: str,
    workers: int,
    adapter_enabled: bool = True,
) -> dict[str, Any]:
    if workers == 1:
        previous = scientist.network.option_adapter_enabled
        scientist.network.option_adapter_enabled = adapter_enabled
        try:
            return _evaluate(
                scientist,
                items,
                ratio=ratio,
                simulations=simulations,
                games=games,
                seed=seed,
            )
        finally:
            scientist.network.option_adapter_enabled = previous
    state = {
        name: value.detach().cpu() for name, value in scientist.network.state_dict().items()
    }
    rows: list[dict[str, Any] | None] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _evaluate_item_worker,
                scientist.name,
                str(scientist.checkpoint),
                state,
                item,
                ratio,
                simulations,
                games,
                seed + item_index * 10_000,
                device,
                adapter_enabled,
                bool(scientist.config.game.objective_budget_channel),
            ): item_index
            for item_index, item in enumerate(items)
        }
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    completed = [row for row in rows if row is not None]
    return {
        "rows": completed,
        "solved_set": sorted(row["item"] for row in completed if row["solved"]),
    }


def run_adapter_counterfactual(
    source_checkpoint: Path,
    trained_checkpoint: Path,
    bank: Path,
    output: Path,
    *,
    scientist_name: str,
    item_ids: tuple[str, ...],
    ratio: float = 10.0,
    simulations: int = 128,
    evaluation_games: int = 8,
    evaluation_workers: int = 4,
    evaluation_seed: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Measure final policy behavior with the learned adapter enabled and bypassed."""
    bank_payload = json.loads(bank.read_text())
    by_id = {item.id: item for item in _bank_from_payload(bank_payload)}
    missing = sorted(set(item_ids) - set(by_id))
    if missing:
        raise ValueError(f"counterfactual identities missing from bank: {missing}")
    items = [by_id[identity] for identity in item_ids]
    scientist = load_scientist(
        scientist_name,
        source_checkpoint,
        seed=evaluation_seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=False,
    )
    payload = torch.load(trained_checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("network", payload)
    load_policy_value_state_dict(scientist.network, state)
    if scientist.network.option_policy_adapter is None:
        raise ValueError("trained checkpoint has no option adapter")
    enabled = _evaluate_parallel(
        scientist,
        items,
        ratio=ratio,
        simulations=simulations,
        games=evaluation_games,
        seed=evaluation_seed,
        device=device,
        workers=evaluation_workers,
        adapter_enabled=True,
    )
    disabled = _evaluate_parallel(
        scientist,
        items,
        ratio=ratio,
        simulations=simulations,
        games=evaluation_games,
        seed=evaluation_seed,
        device=device,
        workers=evaluation_workers,
        adapter_enabled=False,
    )
    state_rows = []
    scientist.network.eval()
    for item in items:
        transition = scientist.game.from_word(
            list(item.knot.word), item.knot.strands, math.log(ratio)
        )
        observation = torch.from_numpy(transition.observation[None]).permute(0, 3, 1, 2)
        observation = observation.float().to(device)
        legal = torch.from_numpy(transition.legal_actions).to(device)
        with torch.inference_mode():
            scientist.network.option_adapter_enabled = False
            base_logits, _ = scientist.network(observation)
            scientist.network.option_adapter_enabled = True
            combined_logits, _ = scientist.network(observation)
            applied_residual, gate = scientist.network.option_policy_components(observation)
            floor = torch.finfo(base_logits.dtype).min
            base_logp = torch.log_softmax(base_logits[0].masked_fill(~legal, floor), dim=-1)
            combined_logp = torch.log_softmax(
                combined_logits[0].masked_fill(~legal, floor), dim=-1
            )
            divergence = float(
                (base_logp.exp() * (base_logp - combined_logp)).sum().item()
            )
            base_action = int(base_logits[0].masked_fill(~legal, floor).argmax().item())
            combined_action = int(
                combined_logits[0].masked_fill(~legal, floor).argmax().item()
            )
        state_rows.append(
            {
                "item": item.id,
                "base_action": base_action,
                "combined_action": combined_action,
                "argmax_flipped": base_action != combined_action,
                "policy_kl_base_to_combined": divergence,
                "applied_residual_l2": float(applied_residual.norm().item()),
                "gate": float(gate.item()),
            }
        )
    scientist.network.option_adapter_enabled = True
    failure = ratio * 20 + 64
    enabled_set = set(enabled["solved_set"])
    disabled_set = set(disabled["solved_set"])
    report = {
        "schema": "option-adapter-counterfactual-v1",
        "scientist": scientist_name,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "trained_checkpoint": str(trained_checkpoint.resolve()),
        "trained_checkpoint_sha256": _sha256(trained_checkpoint),
        "bank": str(bank.resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "items": list(item_ids),
        "ratio": ratio,
        "simulations": simulations,
        "evaluation_games": evaluation_games,
        "evaluation_workers": evaluation_workers,
        "evaluation_seed": evaluation_seed,
        "enabled": enabled,
        "disabled": disabled,
        "enabled_only": sorted(enabled_set - disabled_set),
        "disabled_only": sorted(disabled_set - enabled_set),
        "enabled_capped_loss": _capped_loss(enabled, failure),
        "disabled_capped_loss": _capped_loss(disabled, failure),
        "initial_state_policy": state_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "report.json", report)
    return report


def _capped_loss(evaluation: dict[str, Any], failure: float) -> float:
    return sum(
        float(row["best_objective"]) if row["solved"] else failure
        for row in evaluation["rows"]
    )


def _paired_evaluation_summary(
    sharing: dict[str, Any], control: dict[str, Any], failure: float
) -> dict[str, Any]:
    """Summarize paired policies without hiding identity or compute differences."""
    sharing_rows = {row["item"]: row for row in sharing["rows"]}
    control_rows = {row["item"]: row for row in control["rows"]}
    if set(sharing_rows) != set(control_rows):
        raise ValueError("sharing and control evaluations must contain identical items")
    sharing_set = {item for item, row in sharing_rows.items() if row["solved"]}
    control_set = {item for item, row in control_rows.items() if row["solved"]}
    intersection = sorted(sharing_set & control_set)

    def scheduled(evaluation: dict[str, Any]) -> int:
        return int(
            sum(
                attempt["compute"]["scheduled_network_evaluations"]
                for row in evaluation["rows"]
                for attempt in row["attempts"]
            )
        )

    def worker_seconds(evaluation: dict[str, Any]) -> float:
        return float(
            sum(
                attempt["compute"]["wall_seconds"]
                for row in evaluation["rows"]
                for attempt in row["attempts"]
            )
        )

    sharing_intersection = sum(
        float(sharing_rows[item]["best_objective"]) for item in intersection
    )
    control_intersection = sum(
        float(control_rows[item]["best_objective"]) for item in intersection
    )
    return {
        "sharing_solved": sorted(sharing_set),
        "control_solved": sorted(control_set),
        "intersection": intersection,
        "sharing_only": sorted(sharing_set - control_set),
        "control_only": sorted(control_set - sharing_set),
        "sharing_capped_loss": _capped_loss(sharing, failure),
        "control_capped_loss": _capped_loss(control, failure),
        "intersection_objective_sharing": sharing_intersection,
        "intersection_objective_control": control_intersection,
        "sharing_scheduled_network_evaluations": scheduled(sharing),
        "control_scheduled_network_evaluations": scheduled(control),
        "sharing_worker_seconds": worker_seconds(sharing),
        "control_worker_seconds": worker_seconds(control),
    }


def _evaluation_subset(
    evaluation: dict[str, Any], identities: set[str]
) -> dict[str, Any]:
    rows = [row for row in evaluation["rows"] if row["item"] in identities]
    return {
        "rows": rows,
        "solved_set": sorted(row["item"] for row in rows if row["solved"]),
    }


def _eligible_option_positions(record: list[Position]) -> list[Position]:
    return [
        position
        for position in record
        if position.option_state is not None and position.target_external_action >= 0
    ]


def _active_training_records(
    records: list[list[Position]], training_ids: set[str]
) -> list[list[Position]]:
    """Exclude canaries and other non-target donations from policy training."""
    return [
        record
        for record in records
        if record and record[0].representation_id in training_ids
    ]


def _routable_training_schedule(donated_objectives: dict[str, float]) -> list[str]:
    """Return only targets whose semantic witness translated into this receiver."""
    schedule = sorted(donated_objectives)
    if not schedule:
        raise ValueError("no target witness is routable by the receiver")
    return schedule


def _balanced_witness_positions(
    records: list[list[Position]],
    rng: np.random.Generator,
    *,
    positions_per_witness: int,
) -> list[Position]:
    """Sample the same number of option starts from every donated solution."""
    if positions_per_witness < 1:
        raise ValueError("positions_per_witness must be positive")
    batch: list[Position] = []
    for record in records:
        eligible = _eligible_option_positions(record)
        if not eligible:
            continue
        indices = rng.choice(
            len(eligible),
            size=positions_per_witness,
            replace=len(eligible) < positions_per_witness,
        )
        selected = [eligible[int(index)] for index in indices]
        batch.extend(selected)
    return batch


def _macro_route_losses(
    network: Any,
    game: SerialBraidGame,
    records: list[list[Position]],
    *,
    horizon: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure every witness separately so long solutions cannot dominate."""
    losses: dict[str, float] = {}
    training_states = [(module, module.training) for module in network.modules()]
    network.eval()
    try:
        with torch.inference_mode():
            for record in records:
                if not record:
                    continue
                identity = str(record[0].representation_id)
                positions = _eligible_option_positions(record)
                if not identity or not positions:
                    continue
                losses[identity] = float(
                    stable_option_route_loss(
                        network,
                        game,
                        positions,
                        horizon=horizon,
                        device=device,
                    ).item()
                )
    finally:
        for module, training in training_states:
            module.training = training
    return losses


def _primary_sharing_gate_passed(
    sharing_loss: float,
    control_loss: float,
    sharing_blocks: list[dict[str, Any]],
) -> bool:
    """Primary gate: external objective after at least one real sharing block."""
    return sharing_loss <= control_loss and bool(sharing_blocks)


def _archive_native_evaluation(replay: Any, evaluation: dict[str, Any], ratio: float) -> None:
    """Preserve verified incumbents when evaluation records live in worker processes."""
    for row in evaluation["rows"]:
        if row["solved"]:
            replay.record_native_objective(row["item"], ratio, row["best_objective"])


def _collect_native_evaluation(
    scientist: Any,
    items: list[Any],
    *,
    ratio: float,
    simulations: int,
    games: int,
    seed: int,
) -> tuple[dict[str, Any], list[tuple[str, list[Any]]]]:
    rows = []
    retained_records = []
    for item_index, item in enumerate(items):
        fixed = FixedWordGame(scientist.game, item.knot, ratio)
        search = NeuralMCTS(
            fixed,
            scientist.network,
            replace(scientist.config.search, simulations=simulations),
            str(next(scientist.network.parameters()).device),
        )
        seeds = [seed + item_index * 10_000 + game_index for game_index in range(games)]
        records = play_selfplay_games(
            fixed,
            search,
            [np.random.default_rng(value + 7) for value in seeds],
            seeds,
            12,
        )
        attempts = []
        for record in records:
            verified = verified_record_cost(scientist.game, item.knot, ratio, record)
            attempts.append(
                {
                    "solved": verified is not None,
                    "crossing_changes": verified[0] if verified is not None else None,
                    "moves": verified[1] if verified is not None else None,
                    "objective": (
                        ratio * verified[0] + verified[1] if verified is not None else None
                    ),
                }
            )
            retained_records.append((item.id, record))
        solved_attempts = [row for row in attempts if row["solved"]]
        rows.append(
            {
                "item": item.id,
                "solved": bool(solved_attempts),
                "solved_attempts": len(solved_attempts),
                "best_objective": (
                    min(float(row["objective"]) for row in solved_attempts)
                    if solved_attempts
                    else None
                ),
                "attempts": attempts,
            }
        )
    return (
        {"rows": rows, "solved_set": sorted(row["item"] for row in rows if row["solved"])},
        retained_records,
    )


def run_sharing_admission_gate(
    donor_checkpoint: Path,
    donor_replay: Path,
    bank: Path,
    receivers: dict[str, Path],
    output: Path,
    *,
    donor_name: str = "s-window-128",
    ratio: float = 10.0,
    simulations: int = 64,
    evaluation_games: int = 4,
    train_steps: int = 24,
    batch_size: int = 32,
    seed: int = 20260853,
    device: str = "cpu",
) -> dict[str, Any]:
    """Translate real donor witnesses, learn options, and check paired retention."""
    bank_payload = json.loads(bank.read_text())
    by_id = {item.id: item for item in _bank_from_payload(bank_payload)}
    items = [by_id[identity] for identity in DEFAULT_AUDIT_ITEMS]
    donor = load_scientist(
        donor_name,
        donor_checkpoint,
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    replay = torch.load(donor_replay, map_location="cpu", weights_only=False)
    witnesses: dict[str, tuple[int, int, list[int]]] = {}
    for record in replay.games:
        if not record or record[0].solved <= 0.5:
            continue
        identity = record[0].representation_id
        verified = verified_record_cost(donor.game, by_id[identity].knot, ratio, record)
        if verified is None:
            continue
        incumbent = witnesses.get(identity)
        if incumbent is None or ratio * verified[0] + verified[1] < (
            ratio * incumbent[0] + incumbent[1]
        ):
            witnesses[identity] = verified
    protocol = {
        "schema": "bounded-option-sharing-admission-v1",
        "donor": donor_name,
        "donor_checkpoint": str(donor_checkpoint.resolve()),
        "donor_checkpoint_sha256": _sha256(donor_checkpoint),
        "donor_replay": str(donor_replay.resolve()),
        "donor_replay_sha256": _sha256(donor_replay),
        "bank": str(bank.resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "receivers": [
            {"name": name, "checkpoint": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in receivers.items()
        ],
        "items": list(DEFAULT_AUDIT_ITEMS),
        "witness_items": sorted(witnesses),
        "ratio": ratio,
        "simulations": simulations,
        "evaluation_games": evaluation_games,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "internal_action_cap": 5,
        "seed": seed,
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "manifest.json", protocol)

    receiver_reports = {}
    any_new_identity = False
    all_passed = bool(witnesses)
    failure = ratio * 20 + 64
    for receiver_index, (name, checkpoint) in enumerate(receivers.items()):
        scientist = load_scientist(
            name,
            checkpoint,
            seed=seed + receiver_index * 1_000_000,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=False,
        )
        if not isinstance(scientist.game, SerialBraidGame):
            raise ValueError(f"{name} is not a serial scientist")
        adapter = scientist.network.attach_option_policy_adapter()
        option_optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=scientist.config.train.learning_rate,
            weight_decay=scientist.config.train.weight_decay,
        )
        before = _evaluate(
            scientist,
            items,
            ratio=ratio,
            simulations=simulations,
            games=evaluation_games,
            seed=seed + receiver_index * 1_000_000 + 700_000_000,
        )
        _archive_native_evaluation(scientist.replay, before, ratio)
        translated = []
        for item_index, identity in enumerate(sorted(witnesses)):
            record = translate_semantic_record(
                scientist,
                by_id[identity].knot,
                ratio,
                witnesses[identity][2],
                seed=seed + receiver_index * 1_000_000 + item_index,
                internal_action_cap=5,
            )
            if record is None:
                continue
            scientist.replay.set_representation_embedding(
                identity, _replay_representation_embedding(by_id[identity].knot)
            )
            scientist.replay.add(
                record, representation_id=identity, objective_ratio=ratio
            )
            translated.append(identity)
        option_positions = [
            position
            for record in scientist.replay.active_distillation_records()
            for position in record
            if position.option_state is not None and position.target_external_action >= 0
        ]
        scientist.network.policy_value_preservation_weight = max(
            float(getattr(scientist.network, "policy_value_preservation_weight", 0.0)),
            BUDGET_PRESERVATION_WEIGHTS.get(name, 1.0),
        )
        attach_policy_value_preservation_teacher(scientist.network)
        with torch.inference_mode():
            initial_loss = (
                float(
                    stable_option_route_loss(
                        scientist.network,
                        scientist.game,
                        option_positions,
                        horizon=5,
                        device=torch.device(device),
                    ).item()
                )
                if option_positions
                else 0.0
            )
        losses = []
        for _ in range(train_steps if option_positions else 0):
            losses.append(
                train_bounded_option_step(
                    scientist.network,
                    option_optimizer,
                    scientist.game,
                    scientist.replay,
                    scientist.replay.rng,
                    batch_size=min(batch_size, max(len(option_positions), 1)),
                    horizon=5,
                    beam_width=8,
                    device=torch.device(device),
                    adapter_only=True,
                    positions=option_positions,
                )
            )
        with torch.inference_mode():
            final_loss = (
                float(
                    stable_option_route_loss(
                        scientist.network,
                        scientist.game,
                        option_positions,
                        horizon=5,
                        device=torch.device(device),
                    ).item()
                )
                if option_positions
                else 0.0
            )
        after = _evaluate(
            scientist,
            items,
            ratio=ratio,
            simulations=simulations,
            games=evaluation_games,
            seed=seed + receiver_index * 1_000_000 + 700_000_000,
        )
        before_set = set(before["solved_set"])
        after_set = set(after["solved_set"])
        new_identities = sorted(after_set - before_set)
        lost_identities = sorted(before_set - after_set)
        any_new_identity |= bool(new_identities)
        before_loss = _capped_loss(before, failure)
        after_loss = _capped_loss(after, failure)
        receiver_passed = (
            bool(translated)
            and bool(option_positions)
            and final_loss < initial_loss
            and not lost_identities
            and after_loss <= before_loss
        )
        all_passed &= receiver_passed
        torch.save(
            {
                "network": scientist.network.state_dict(),
                "source_checkpoint": str(checkpoint.resolve()),
                "protocol_sha256": protocol["protocol_sha256"],
            },
            output / f"{name}-trained.pt",
        )
        receiver_reports[name] = {
            "translated_witnesses": translated,
            "preservation_weight": float(
                scientist.network.policy_value_preservation_weight
            ),
            "translated_receiver_unsolved": sorted(set(translated) - before_set),
            "option_positions": len(option_positions),
            "initial_option_loss": initial_loss,
            "final_option_loss": final_loss,
            "training_losses": losses,
            "before": before,
            "after": after,
            "new_identities": new_identities,
            "lost_identities": lost_identities,
            "before_capped_loss": before_loss,
            "after_capped_loss": after_loss,
            "passed": receiver_passed,
        }
    passed = all_passed and any_new_identity
    report = {
        **protocol,
        "receivers": receiver_reports,
        "decision": {
            "passed": passed,
            "all_receivers_translation_loss_and_retention_passed": all_passed,
            "at_least_one_new_receiver_identity": any_new_identity,
            "next_step": (
                "finalize scientist roster" if passed else "repair bounded-option sharing"
            ),
        },
    }
    _atomic_json(output / "report.json", report)
    return report


def run_interleaved_sharing_gate(
    donor_checkpoint: Path,
    donor_replay: Path,
    bank: Path,
    receivers: dict[str, Path],
    output: Path,
    *,
    donor_name: str = "s-window-128",
    ratio: float = 10.0,
    simulations: int = 64,
    evaluation_simulations: int | None = None,
    evaluation_games: int = 4,
    update_cycles: int = 8,
    batch_size: int = 16,
    option_learning_rate_scale: float = 1.0,
    sharing_block_size: int = 10,
    sharing_interval_cycles: int = 10,
    adapter_steps_per_block: int = 16,
    option_positions_per_witness: int = 4,
    witness_bank: Path | None = None,
    item_ids: tuple[str, ...] = (),
    target_item_ids: tuple[str, ...] = (),
    generalization_item_ids: tuple[str, ...] = (),
    native_refresh_games: int = 0,
    evaluation_workers: int = 1,
    gated_adapter: bool = False,
    route_gate_weight: float = 0.1,
    off_route_kl_weight: float = 1.0,
    off_route_gate_weight: float = 0.1,
    off_route_batch_size: int = 32,
    seed: int = 20260854,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compare native+option learning with compute-matched native-only replay."""
    final_simulations = evaluation_simulations or simulations
    if final_simulations < 1:
        raise ValueError("evaluation simulations must be positive")
    if min(
        sharing_block_size,
        sharing_interval_cycles,
        adapter_steps_per_block,
        option_positions_per_witness,
    ) < 1:
        raise ValueError("sharing block parameters must be positive")
    if native_refresh_games < 0:
        raise ValueError("native_refresh_games must be non-negative")
    if evaluation_workers < 1:
        raise ValueError("evaluation_workers must be positive")
    if min(route_gate_weight, off_route_kl_weight, off_route_gate_weight) < 0.0:
        raise ValueError("gate loss weights must be non-negative")
    if off_route_batch_size < 1:
        raise ValueError("off_route_batch_size must be positive")
    bank_payload = json.loads(bank.read_text())
    by_id = {item.id: item for item in _bank_from_payload(bank_payload)}
    evaluation_ids = list(item_ids) if item_ids else list(DEFAULT_AUDIT_ITEMS)
    missing_items = sorted(set(evaluation_ids) - set(by_id))
    if missing_items:
        raise ValueError(f"evaluation identities missing from bank: {missing_items}")
    items = [by_id[identity] for identity in evaluation_ids]
    witnesses: dict[str, tuple[int, int, list[int]]] = {}
    witness_provenance = None
    if witness_bank is not None:
        witness_provenance = json.loads(witness_bank.read_text())
        if witness_provenance.get("schema") != (
            "certified-semantic-collaboration-witness-bank-v2"
        ):
            raise ValueError(
                "v11 requires a certified semantic-cost witness bank; "
                "rebuild historical native-ply banks"
            )
        for identity, row in witness_provenance["witnesses"].items():
            actions = [int(action) for action in row["semantic_actions"]]
            if int(row["moves"]) != len(actions):
                raise ValueError(
                    f"semantic witness move mismatch for {identity}: "
                    f"moves={row['moves']} actions={len(actions)}"
                )
            witnesses[identity] = (
                int(row["crossing_changes"]),
                int(row["moves"]),
                actions,
            )
    else:
        donor = load_scientist(
            donor_name,
            donor_checkpoint,
            seed=seed,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=True,
        )
        donor_buffer = torch.load(donor_replay, map_location="cpu", weights_only=False)
        for record in donor_buffer.games:
            if not record or record[0].solved <= 0.5:
                continue
            identity = record[0].representation_id
            verified = verified_record_cost(donor.game, by_id[identity].knot, ratio, record)
            if verified is None:
                continue
            incumbent = witnesses.get(identity)
            if incumbent is None or ratio * verified[0] + verified[1] < (
                ratio * incumbent[0] + incumbent[1]
            ):
                witnesses[identity] = verified
    training_ids = list(target_item_ids) if target_item_ids else sorted(witnesses)
    training_id_set = set(training_ids)
    missing_targets = sorted(set(training_ids) - set(witnesses))
    if missing_targets:
        raise ValueError(f"target identities lack certified witnesses: {missing_targets}")
    unevaluated_targets = sorted(set(training_ids) - set(evaluation_ids))
    if unevaluated_targets:
        raise ValueError(f"target identities are not evaluation items: {unevaluated_targets}")
    generalization_ids = list(generalization_item_ids)
    missing_generalization = sorted(set(generalization_ids) - set(evaluation_ids))
    if missing_generalization:
        raise ValueError(
            "generalization identities are not evaluation items: "
            f"{missing_generalization}"
        )
    leaked_generalization = sorted(set(generalization_ids) & set(training_ids))
    if leaked_generalization:
        raise ValueError(
            "generalization identities cannot be sharing targets: "
            f"{leaked_generalization}"
        )
    if len(set(training_ids)) < sharing_block_size:
        raise ValueError(
            "sharing block requires at least "
            f"{sharing_block_size} distinct target identities"
        )
    protocol = {
        "schema": "semantic-cost-block-balanced-option-adapter-sharing-v11",
        "donor": donor_name,
        "donor_checkpoint": str(donor_checkpoint.resolve()),
        "donor_checkpoint_sha256": _sha256(donor_checkpoint),
        "donor_replay": str(donor_replay.resolve()),
        "donor_replay_sha256": _sha256(donor_replay),
        "bank": str(bank.resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "receivers": [
            {"name": name, "checkpoint": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in receivers.items()
        ],
        "items": evaluation_ids,
        "target_items": training_ids,
        "generalization_items": generalization_ids,
        "witness_items": sorted(witnesses),
        "ratio": ratio,
        "simulations": simulations,
        "training_simulations": simulations,
        "evaluation_simulations": final_simulations,
        "evaluation_games": evaluation_games,
        "update_cycles": update_cycles,
        "batch_size": batch_size,
        "option_learning_rate_scale": option_learning_rate_scale,
        "option_adapter_base_learning_rate": OPTION_ADAPTER_LEARNING_RATE,
        "option_adapter_effective_learning_rate": (
            OPTION_ADAPTER_LEARNING_RATE * option_learning_rate_scale
        ),
        "sharing_block_size": sharing_block_size,
        "sharing_interval_cycles": sharing_interval_cycles,
        "adapter_steps_per_block": adapter_steps_per_block,
        "option_positions_per_witness": option_positions_per_witness,
        "witness_bank": str(witness_bank.resolve()) if witness_bank else None,
        "witness_bank_sha256": _json_hash(witness_provenance) if witness_provenance else None,
        "native_refresh_games": native_refresh_games,
        "native_refresh_seed_pairing": "identical between sharing and control",
        "evaluation_workers": evaluation_workers,
        "gated_adapter": gated_adapter,
        "route_gate_weight": route_gate_weight,
        "off_route_kl_weight": off_route_kl_weight,
        "off_route_gate_weight": off_route_gate_weight,
        "off_route_batch_size": off_route_batch_size,
        "sharing_update": (
            "fixed-dose adapter/gate updates at scheduled block boundaries over "
            "representation-balanced samples from all active superior donations"
        ),
        "control_update": (
            "native replay-v3 updates matched to at least the adapter optimizer "
            "steps and routed state-action targets"
        ),
        "compute_match_unit": "network state examples consumed by training loss",
        "solution_move_metric": "verified portable semantic witness steps",
        "receiver_compute_metrics": "native_plies and internal_plies, excluded from L_A:B",
        "native_optimizer_isolation": (
            "option controller bypassed, all gradients cleared, owned-only clipping"
        ),
        "option_conditioning": (
            "full serial observation with explicit head-cell and internal-budget features"
        ),
        "shared_policy_eligibility": (
            "best donated semantic objective must be strictly lower than the best "
            "archived native semantic objective for the same identity and ratio; rechecked "
            "before every sharing block"
        ),
        "stale_shared_witness_use": "critic-upper-bound-only",
        "primary_gate": (
            "semantic capped loss noninferiority after at least one sharing block; "
            "the overall gate additionally requires a sharing-only solved identity"
        ),
        "secondary_gate": (
            "exact frozen-solve retention, paired control-only identities, and exact "
            "final-set equality, common-success objective, and per-witness route-loss "
            "changes are reported but do not block admission"
        ),
        "internal_action_cap": 5,
        "seed": seed,
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "manifest.json", protocol)
    failure = ratio * 20 + 64
    reports = {}
    all_primary_passed = True
    all_exact_retention = True
    any_sharing_only = False
    for receiver_index, (name, checkpoint) in enumerate(receivers.items()):
        scientist_seed = seed + receiver_index * 1_000_000
        sharing = load_scientist(
            name,
            checkpoint,
            seed=scientist_seed,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=True,
        )
        control = load_scientist(
            name,
            checkpoint,
            seed=scientist_seed,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=True,
        )
        if not isinstance(sharing.game, SerialBraidGame):
            raise ValueError(f"{name} is not a serial scientist")
        adapter = sharing.network.attach_option_policy_adapter()
        gate = sharing.network.attach_option_policy_gate() if gated_adapter else None
        sharing_parameters = list(adapter.parameters())
        if gate is not None:
            sharing_parameters.extend(gate.parameters())
        option_optimizer = torch.optim.AdamW(
            sharing_parameters,
            # The adapter starts from zero and is optimized independently.  It
            # must not inherit a conservative native fine-tuning rate (s-tape4
            # uses 5e-5), which made the same 16-step dose effectively inert.
            lr=OPTION_ADAPTER_LEARNING_RATE,
            weight_decay=sharing.config.train.weight_decay,
        )
        preservation_weight = BUDGET_PRESERVATION_WEIGHTS.get(name, 1.0)
        for scientist in (sharing, control):
            scientist.network.policy_value_preservation_weight = preservation_weight
            attach_policy_value_preservation_teacher(scientist.network)
        evaluation_seed = scientist_seed + 800_000_000
        if native_refresh_games:
            before = _evaluate_parallel(
                sharing,
                items,
                ratio=ratio,
                simulations=final_simulations,
                games=evaluation_games,
                seed=evaluation_seed,
                device=device,
                workers=evaluation_workers,
            )
            native_records = []
        else:
            before, native_records = _collect_native_evaluation(
                sharing,
                items,
                ratio=ratio,
                simulations=final_simulations,
                games=evaluation_games,
                seed=evaluation_seed,
            )
        for scientist in (sharing, control):
            _archive_native_evaluation(scientist.replay, before, ratio)
        for identity, record in native_records:
            embedding = _replay_representation_embedding(by_id[identity].knot)
            for scientist in (sharing, control):
                scientist.replay.set_representation_embedding(identity, embedding)
                scientist.replay.add(
                    copy.deepcopy(record),
                    representation_id=identity,
                    objective_ratio=ratio,
                )
        translated = []
        translated_records = []
        translated_costs: dict[str, dict[str, float]] = {}
        for item_index, identity in enumerate(
            sorted(set(witnesses) & set(training_ids))
        ):
            record = translate_semantic_record(
                sharing,
                by_id[identity].knot,
                ratio,
                witnesses[identity][2],
                seed=scientist_seed + item_index,
                internal_action_cap=5,
            )
            if record is None:
                continue
            translated_verified = verified_record_cost(
                sharing.game, by_id[identity].knot, ratio, record
            )
            donor_crossings, donor_moves, _ = witnesses[identity]
            if translated_verified is None or translated_verified[:2] != (
                donor_crossings,
                donor_moves,
            ):
                raise RuntimeError(
                    f"translation changed semantic cost for {identity}: "
                    f"donor={(donor_crossings, donor_moves)} "
                    f"receiver={None if translated_verified is None else translated_verified[:2]}"
                )
            translated_costs[identity] = {
                "crossing_changes": float(donor_crossings),
                "semantic_moves": float(donor_moves),
                "semantic_objective": ratio * donor_crossings + donor_moves,
                "receiver_native_plies": float(record[0].final_native_plies),
                "receiver_internal_plies": float(record[0].final_internal_plies),
            }
            sharing.replay.add(
                record, representation_id=identity, objective_ratio=ratio
            )
            translated.append(identity)
            translated_records.append((identity, record))
        active_training_records = _active_training_records(
            sharing.replay.active_distillation_records(), training_id_set
        )
        option_positions = [
            position
            for record in active_training_records
            for position in record
            if position.option_state is not None and position.target_external_action >= 0
        ]
        if not translated_records:
            raise ValueError(f"no donor witness translated into {name}")
        training_pairs = [
            (identity, record)
            for identity, record in translated_records
            if identity in training_id_set
        ]
        if not training_pairs:
            raise ValueError(f"no target witness translated into {name}")
        donated_objectives = {
            identity: ratio * float(record[0].final_crossing_changes)
            + float(record[0].final_moves)
            for identity, record in training_pairs
        }
        routable_training_ids = _routable_training_schedule(donated_objectives)
        retention_positions = []
        for item in items:
            if item.id in set(training_ids) | set(generalization_ids):
                continue
            transition = sharing.game.from_word(
                list(item.knot.word), item.knot.strands, math.log(ratio)
            )
            retention_positions.append(
                Position(
                    transition.observation,
                    transition.legal_actions,
                    np.zeros(sharing.game.num_actions, dtype=np.float32),
                    0,
                    transition.player,
                    representation_id=item.id,
                )
            )
        initial_route_losses = _macro_route_losses(
            sharing.network,
            sharing.game,
            active_training_records,
            horizon=5,
            device=torch.device(device),
        )
        initial_option_loss = (
            float(np.mean(list(initial_route_losses.values())))
            if initial_route_losses
            else 0.0
        )
        updates = []
        sharing_blocks: list[dict[str, Any]] = []
        for cycle in range(update_cycles):
            current = routable_training_ids[cycle % len(routable_training_ids)]
            refreshed_sharing_positions = []
            if native_refresh_games:
                current_item = by_id[current]
                for scientist in (sharing, control):
                    _, refreshed = _collect_native_evaluation(
                        scientist,
                        [current_item],
                        ratio=ratio,
                        simulations=simulations,
                        games=native_refresh_games,
                        seed=(
                            scientist_seed
                            + 850_000_000
                            + cycle * 10_000
                        ),
                    )
                    embedding = _replay_representation_embedding(current_item.knot)
                    scientist.replay.set_representation_embedding(current, embedding)
                    for _, record in refreshed:
                        scientist.replay.add(
                            record,
                            representation_id=current,
                            objective_ratio=ratio,
                        )
                        if scientist is sharing:
                            refreshed_sharing_positions.extend(record)
            preservation_positions = list(retention_positions)
            remaining = max(off_route_batch_size - len(preservation_positions), 0)
            if remaining and refreshed_sharing_positions:
                indices = np.linspace(
                    0,
                    len(refreshed_sharing_positions) - 1,
                    min(remaining, len(refreshed_sharing_positions)),
                    dtype=int,
                )
                preservation_positions.extend(
                    refreshed_sharing_positions[int(index)] for index in indices
                )
            common = {
                "batch_size": batch_size,
                "device": torch.device(device),
                "collaboration_replay": True,
                "shared_fraction": 0.0,
                "policy_value_success_only": True,
                "replay_current_representation": current,
                "replay_current_fraction": 0.25,
                "replay_similar_fraction": 0.25,
                "replay_positions_per_episode": 4,
            }
            sharing_native = train_alphazero_step(
                sharing.network, sharing.optimizer, sharing.replay, **common
            )
            control_native = train_alphazero_step(
                control.network, control.optimizer, control.replay, **common
            )
            option_updates = []
            option_started = time.perf_counter()
            active_records = {
                record[0].representation_id: record
                for record in _active_training_records(
                    sharing.replay.active_distillation_records(), training_id_set
                )
            }
            block_due = (cycle + 1) % sharing_interval_cycles == 0
            selected_records = (
                [active_records[identity] for identity in sorted(active_records)]
                if block_due and len(active_records) >= sharing_block_size
                else []
            )
            route_loss_before = (
                _macro_route_losses(
                    sharing.network,
                    sharing.game,
                    selected_records,
                    horizon=5,
                    device=torch.device(device),
                )
                if selected_records
                else {}
            )
            option_loss = 0.0
            if selected_records:
                for option_step in range(adapter_steps_per_block):
                    balanced_positions = _balanced_witness_positions(
                        selected_records,
                        sharing.replay.rng,
                        positions_per_witness=option_positions_per_witness,
                    )
                    option_diagnostics: dict[str, float | str] = {}
                    option_loss = train_bounded_option_step(
                        sharing.network,
                        option_optimizer,
                        sharing.game,
                        sharing.replay,
                        sharing.replay.rng,
                        batch_size=len(balanced_positions),
                        horizon=5,
                        beam_width=8,
                        device=torch.device(device),
                        positions=balanced_positions,
                        learning_rate_scale=option_learning_rate_scale,
                        adapter_only=True,
                        stable_routes=True,
                        preservation_positions=(
                            preservation_positions if gated_adapter else None
                        ),
                        route_gate_weight=route_gate_weight if gated_adapter else 0.0,
                        off_route_kl_weight=(
                            off_route_kl_weight if gated_adapter else 0.0
                        ),
                        off_route_gate_weight=(
                            off_route_gate_weight if gated_adapter else 0.0
                        ),
                        diagnostics=option_diagnostics,
                    )
                    option_diagnostics["step"] = float(option_step)
                    option_updates.append(option_diagnostics)
            option_seconds = time.perf_counter() - option_started
            route_loss_after = (
                _macro_route_losses(
                    sharing.network,
                    sharing.game,
                    selected_records,
                    horizon=5,
                    device=torch.device(device),
                )
                if selected_records
                else {}
            )
            route_loss_reductions = {
                identity: (
                    route_loss_before[identity] - route_loss_after[identity]
                )
                / max(abs(route_loss_before[identity]), 1e-12)
                for identity in sorted(route_loss_before)
                if identity in route_loss_after
            }
            adapter_targets = int(
                sum(
                    float(row["action_targets"]) + float(row["off_route_positions"])
                    for row in option_updates
                )
            )
            control_steps = max(
                len(option_updates), math.ceil(adapter_targets / batch_size)
            )
            control_started = time.perf_counter()
            control_extra = [
                train_alphazero_step(
                    control.network, control.optimizer, control.replay, **common
                )
                for _ in range(control_steps)
            ]
            control_seconds = time.perf_counter() - control_started
            sharing_block = None
            if selected_records:
                sharing_block = {
                    "cycle": cycle,
                    "witnesses": sorted(active_records),
                    "distinct_witnesses": len(selected_records),
                    "adapter_optimizer_steps": len(option_updates),
                    "adapter_state_examples": adapter_targets,
                    "route_loss_before": route_loss_before,
                    "route_loss_after": route_loss_after,
                    "route_loss_relative_reduction": route_loss_reductions,
                    "mean_route_loss_relative_reduction": (
                        float(np.mean(list(route_loss_reductions.values())))
                        if route_loss_reductions
                        else None
                    ),
                    "improved_route_loss_witnesses": sorted(
                        identity
                        for identity, reduction in route_loss_reductions.items()
                        if reduction > 0.0
                    ),
                    "regressed_route_loss_witnesses": sorted(
                        identity
                        for identity, reduction in route_loss_reductions.items()
                        if reduction < 0.0
                    ),
                }
                sharing_blocks.append(sharing_block)
            updates.append(
                {
                    "cycle": cycle,
                    "current": current,
                    "donated_objective": donated_objectives[current],
                    "best_native_objective": sharing.replay.best_native_objective(
                        current, ratio
                    ),
                    "active_witnesses": sorted(active_records),
                    "active_witness_count": len(active_records),
                    "sharing_block_due": block_due,
                    "sharing_block_completed": sharing_block is not None,
                    "zero_policy_updates": not option_updates,
                    "stale_current_witness": current not in active_records,
                    "sharing_native": sharing_native,
                    "option_loss": option_loss,
                    "option_updates": option_updates,
                    "sharing_block": sharing_block,
                    "adapter_optimizer_steps": len(option_updates),
                    "adapter_state_examples": adapter_targets,
                    "adapter_seconds": option_seconds,
                    "control_native": control_native,
                    "control_extra": control_extra,
                    "control_optimizer_steps": control_steps,
                    "control_state_examples": control_steps * batch_size,
                    "control_seconds": control_seconds,
                }
            )
        final_active_records = _active_training_records(
            sharing.replay.active_distillation_records(), training_id_set
        )
        final_route_losses = _macro_route_losses(
            sharing.network,
            sharing.game,
            final_active_records,
            horizon=5,
            device=torch.device(device),
        )
        final_option_loss = (
            float(np.mean(list(final_route_losses.values())))
            if final_route_losses
            else 0.0
        )
        sharing_after = _evaluate_parallel(
            sharing,
            items,
            ratio=ratio,
            simulations=final_simulations,
            games=evaluation_games,
            seed=evaluation_seed,
            device=device,
            workers=evaluation_workers,
        )
        control_after = _evaluate_parallel(
            control,
            items,
            ratio=ratio,
            simulations=final_simulations,
            games=evaluation_games,
            seed=evaluation_seed,
            device=device,
            workers=evaluation_workers,
        )
        before_set = set(before["solved_set"])
        sharing_set = set(sharing_after["solved_set"])
        control_set = set(control_after["solved_set"])
        sharing_only = sorted(sharing_set - control_set)
        control_only = sorted(control_set - sharing_set)
        lost_from_before = sorted(before_set - sharing_set)
        sharing_loss = _capped_loss(sharing_after, failure)
        control_loss = _capped_loss(control_after, failure)
        before_rows = {row["item"]: row for row in before["rows"]}
        sharing_rows = {row["item"]: row for row in sharing_after["rows"]}
        control_rows = {row["item"]: row for row in control_after["rows"]}
        target_before = sorted(
            identity for identity in training_ids if before_rows[identity]["solved"]
        )
        target_sharing = sorted(
            identity for identity in training_ids if sharing_rows[identity]["solved"]
        )
        target_control = sorted(
            identity for identity in training_ids if control_rows[identity]["solved"]
        )
        primary_passed = _primary_sharing_gate_passed(
            sharing_loss, control_loss, sharing_blocks
        )
        exact_retention = not lost_from_before
        all_primary_passed &= primary_passed
        all_exact_retention &= exact_retention
        any_sharing_only |= bool(sharing_only)
        torch.save(
            {
                "network": sharing.network.state_dict(),
                "protocol_sha256": protocol["protocol_sha256"],
            },
            output / f"{name}-sharing.pt",
        )
        torch.save(
            {
                "network": control.network.state_dict(),
                "protocol_sha256": protocol["protocol_sha256"],
            },
            output / f"{name}-control.pt",
        )
        reports[name] = {
            "preservation_weight": preservation_weight,
            "translated_witnesses": translated,
            "translated_costs": translated_costs,
            "translated_receiver_unsolved": sorted(set(translated) - before_set),
            "trained_witnesses": [identity for identity, _ in training_pairs],
            "unroutable_target_witnesses": sorted(
                set(training_ids) - set(routable_training_ids)
            ),
            "option_positions": len(option_positions),
            "final_active_witnesses": sorted(
                record[0].representation_id for record in final_active_records
            ),
            "stale_witnesses": sorted(
                set(training_ids)
                - {record[0].representation_id for record in final_active_records}
            ),
            "initial_option_loss": initial_option_loss,
            "final_option_loss": final_option_loss,
            "initial_route_losses": initial_route_losses,
            "final_route_losses": final_route_losses,
            "before": before,
            "sharing_after": sharing_after,
            "control_after": control_after,
            "sharing_only": sharing_only,
            "control_only": control_only,
            "lost_from_before": lost_from_before,
            "sharing_capped_loss": sharing_loss,
            "control_capped_loss": control_loss,
            "target_metrics": {
                "items": training_ids,
                "frozen_solved": target_before,
                "sharing_solved": target_sharing,
                "control_solved": target_control,
                "sharing_transfer_rate": len(target_sharing) / len(training_ids),
                "control_transfer_rate": len(target_control) / len(training_ids),
                "sharing_only": sorted(set(target_sharing) - set(target_control)),
                "control_only": sorted(set(target_control) - set(target_sharing)),
            },
            "updates": updates,
            "sharing_blocks": sharing_blocks,
            "completed_sharing_blocks": len(sharing_blocks),
            "paired_final": _paired_evaluation_summary(
                sharing_after, control_after, failure
            ),
            "paired_training_targets": _paired_evaluation_summary(
                _evaluation_subset(sharing_after, set(training_ids)),
                _evaluation_subset(control_after, set(training_ids)),
                failure,
            ),
            "paired_non_target_canaries": _paired_evaluation_summary(
                _evaluation_subset(
                    sharing_after,
                    set(evaluation_ids) - set(training_ids) - set(generalization_ids),
                ),
                _evaluation_subset(
                    control_after,
                    set(evaluation_ids) - set(training_ids) - set(generalization_ids),
                ),
                failure,
            ),
            "paired_generalization": _paired_evaluation_summary(
                _evaluation_subset(sharing_after, set(generalization_ids)),
                _evaluation_subset(control_after, set(generalization_ids)),
                failure,
            ),
            "passed_primary": primary_passed,
            "exact_retention_secondary": exact_retention,
            "exact_final_set_match_secondary": sharing_set == control_set,
            "control_only_secondary": control_only,
        }
    passed = all_primary_passed and any_sharing_only
    report = {
        **protocol,
        "receivers": reports,
        "decision": {
            "passed": passed,
            "all_receivers_primary_passed": all_primary_passed,
            "at_least_one_sharing_only_identity": any_sharing_only,
            "all_receivers_exact_retention_secondary": all_exact_retention,
            "next_step": (
                "continue the preregistered v11 multi-seed gate"
                if passed
                else "keep sharing arms closed"
            ),
        },
    }
    _atomic_json(output / "report.json", report)
    return report


def summarize_sharing_multiseed(
    runs: tuple[Path, ...], output: Path
) -> dict[str, Any]:
    """Combine independent v11 semantic-cost seeds without hiding paired variation."""
    if len(runs) < 3:
        raise ValueError("the confirmatory sharing summary requires at least three seeds")
    payloads = []
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text())
        report = json.loads((run / "report.json").read_text())
        if manifest["schema"] != "semantic-cost-block-balanced-option-adapter-sharing-v11":
            raise ValueError(f"not a v11 semantic-cost sharing run: {run}")
        payloads.append((run, manifest, report))
    seeds = [int(manifest["seed"]) for _, manifest, _ in payloads]
    if len(set(seeds)) != len(seeds):
        raise ValueError("sharing summary seeds must be distinct")
    comparison_keys = (
        "bank_sha256",
        "items",
        "target_items",
        "generalization_items",
        "ratio",
        "training_simulations",
        "evaluation_simulations",
        "evaluation_games",
        "update_cycles",
        "batch_size",
        "option_learning_rate_scale",
        "sharing_block_size",
        "sharing_interval_cycles",
        "adapter_steps_per_block",
        "option_positions_per_witness",
        "option_adapter_base_learning_rate",
        "option_adapter_effective_learning_rate",
        "native_refresh_games",
        "gated_adapter",
        "route_gate_weight",
        "off_route_kl_weight",
        "off_route_gate_weight",
        "off_route_batch_size",
        "receivers",
    )
    reference = {key: payloads[0][1][key] for key in comparison_keys}
    for run, manifest, _ in payloads[1:]:
        candidate = {key: manifest[key] for key in comparison_keys}
        if candidate != reference:
            raise ValueError(f"sharing seed protocol mismatch: {run}")
    receiver_names = [row["name"] for row in reference["receivers"]]
    receiver_reports: dict[str, Any] = {}
    for name in receiver_names:
        seed_rows = []
        for run, manifest, report in payloads:
            receiver = report["receivers"][name]
            paired = receiver["paired_final"]
            delta = float(paired["sharing_capped_loss"]) - float(
                paired["control_capped_loss"]
            )
            seed_rows.append(
                {
                    "seed": int(manifest["seed"]),
                    "run": str(run.resolve()),
                    "protocol_sha256": manifest["protocol_sha256"],
                    "completed_sharing_blocks": int(
                        receiver["completed_sharing_blocks"]
                    ),
                    "capped_loss_delta_sharing_minus_control": delta,
                    "paired_final": paired,
                    "paired_training_targets": receiver[
                        "paired_training_targets"
                    ],
                    "paired_non_target_canaries": receiver.get(
                        "paired_non_target_canaries", receiver.get("paired_heldout")
                    ),
                    "paired_generalization": receiver.get(
                        "paired_generalization",
                        receiver.get("paired_non_target_canaries", receiver.get("paired_heldout")),
                    ),
                    "lost_from_before": receiver["lost_from_before"],
                    "route_loss_block_means": [
                        block["mean_route_loss_relative_reduction"]
                        for block in receiver["sharing_blocks"]
                    ],
                }
            )
        deltas = [row["capped_loss_delta_sharing_minus_control"] for row in seed_rows]

        def subset_summary(
            field: str, rows: list[dict[str, Any]]
        ) -> dict[str, Any]:
            subset_deltas = [
                float(row[field]["sharing_capped_loss"])
                - float(row[field]["control_capped_loss"])
                for row in rows
            ]
            return {
                "deltas_sharing_minus_control": subset_deltas,
                "mean_delta_sharing_minus_control": float(np.mean(subset_deltas)),
                "median_delta_sharing_minus_control": float(np.median(subset_deltas)),
                "sharing_wins": sum(delta < 0.0 for delta in subset_deltas),
                "sharing_ties": sum(delta == 0.0 for delta in subset_deltas),
                "control_wins": sum(delta > 0.0 for delta in subset_deltas),
                "mean_sharing_solved": float(
                    np.mean([len(row[field]["sharing_solved"]) for row in rows])
                ),
                "mean_control_solved": float(
                    np.mean([len(row[field]["control_solved"]) for row in rows])
                ),
            }

        all_blocks = all(row["completed_sharing_blocks"] > 0 for row in seed_rows)
        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))
        sharing_only = sorted(
            {
                identity
                for row in seed_rows
                for identity in row["paired_final"]["sharing_only"]
            }
        )
        passed = all_blocks and mean_delta < 0.0 and median_delta < 0.0 and bool(sharing_only)
        receiver_reports[name] = {
            "seeds": seed_rows,
            "mean_capped_loss_delta_sharing_minus_control": mean_delta,
            "median_capped_loss_delta_sharing_minus_control": median_delta,
            "sharing_wins": sum(delta < 0.0 for delta in deltas),
            "sharing_ties": sum(delta == 0.0 for delta in deltas),
            "control_wins": sum(delta > 0.0 for delta in deltas),
            "sharing_only_union": sharing_only,
            "all_seeds_completed_sharing_block": all_blocks,
            "training_targets_summary": subset_summary(
                "paired_training_targets", seed_rows
            ),
            "non_target_canary_summary": subset_summary(
                "paired_non_target_canaries", seed_rows
            ),
            "generalization_summary": subset_summary(
                "paired_generalization", seed_rows
            ),
            "passed": passed,
        }
    protocol = {
        "schema": "semantic-cost-sharing-multiseed-summary-v2",
        "runs": [str(run.resolve()) for run, _, _ in payloads],
        "seeds": seeds,
        "comparison": reference,
        "primary_gate": (
            "every seed completes a sharing block; both mean and median paired capped-loss "
            "delta are negative; at least one sharing-only identity exists"
        ),
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    passed = all(row["passed"] for row in receiver_reports.values())
    report = {
        **protocol,
        "receivers": receiver_reports,
        "decision": {
            "passed": passed,
            "next_step": (
                "open the 30--50-representation v11 pilot"
                if passed
                else "keep long sharing and paid compute closed"
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "manifest.json", protocol)
    _atomic_json(output / "report.json", report)
    return report


def run_sharing_simulation_dose(
    gate_run: Path,
    output: Path,
    *,
    simulations: tuple[int, ...] = (32, 64, 128, 256),
    evaluation_games: int = 4,
    evaluation_workers: int = 4,
    seed: int = 20260970,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate one trained sharing/control pair across fixed MCTS budgets."""
    if not simulations or any(value < 1 for value in simulations):
        raise ValueError("simulation levels must be positive")
    manifest = json.loads((gate_run / "manifest.json").read_text())
    semantic_v11 = manifest.get("schema") == (
        "semantic-cost-block-balanced-option-adapter-sharing-v11"
    )
    if not semantic_v11:
        raise ValueError("semantic simulation-dose evaluation requires a v11 gate run")
    source_report = json.loads((gate_run / "report.json").read_text())
    bank_payload = json.loads(Path(manifest["bank"]).read_text())
    by_id = {item.id: item for item in _bank_from_payload(bank_payload)}
    items = [by_id[identity] for identity in manifest["items"]]
    ratio = float(manifest["ratio"])
    failure = ratio * 20 + 64
    protocol = {
        "schema": "semantic-cost-sharing-mcts-simulation-dose-v2",
        "source_gate": str(gate_run.resolve()),
        "source_gate_protocol_sha256": manifest["protocol_sha256"],
        "source_gate_report_sha256": _json_hash(source_report),
        "bank": manifest["bank"],
        "bank_sha256": manifest["bank_sha256"],
        "items": manifest["items"],
        "target_items": manifest.get("target_items", []),
        "generalization_items": manifest.get("generalization_items", []),
        "ratio": ratio,
        "training_simulations": int(manifest["simulations"]),
        "evaluation_simulations": list(simulations),
        "evaluation_games": evaluation_games,
        "evaluation_workers": evaluation_workers,
        "seed": seed,
        "device": device,
        "pairing": "identical item and attempt seeds for sharing and control",
        "solution_move_metric": "verified portable semantic witness steps",
        "objective_budget_channel": semantic_v11,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "manifest.json", protocol)
    receivers: dict[str, Any] = {}
    for receiver_index, receiver in enumerate(manifest["receivers"]):
        name = receiver["name"]
        source_checkpoint = Path(receiver["checkpoint"])
        trained = {}
        for arm in ("sharing", "control"):
            checkpoint = gate_run / f"{name}-{arm}.pt"
            scientist = load_scientist(
                name,
                source_checkpoint,
                seed=seed + receiver_index * 1_000_000,
                device=device,
                simulations=max(simulations),
                require_factorized=True,
                objective_budget_channel=semantic_v11,
            )
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            load_policy_value_state_dict(scientist.network, payload["network"])
            trained[arm] = scientist
        rows = []
        for level in simulations:
            level_seed = seed + receiver_index * 1_000_000
            evaluations = {}
            elapsed = {}
            for arm in ("sharing", "control"):
                started = time.perf_counter()
                evaluations[arm] = _evaluate_parallel(
                    trained[arm],
                    items,
                    ratio=ratio,
                    simulations=level,
                    games=evaluation_games,
                    seed=level_seed,
                    device=device,
                    workers=evaluation_workers,
                )
                elapsed[arm] = time.perf_counter() - started
            targets = set(manifest.get("target_items", []))
            generalization = set(manifest.get("generalization_items", []))
            canaries = set(manifest["items"]) - targets - generalization
            rows.append(
                {
                    "simulations": level,
                    **_paired_evaluation_summary(
                        evaluations["sharing"], evaluations["control"], failure
                    ),
                    "paired_training_targets": _paired_evaluation_summary(
                        _evaluation_subset(evaluations["sharing"], targets),
                        _evaluation_subset(evaluations["control"], targets),
                        failure,
                    ),
                    "paired_non_target_canaries": _paired_evaluation_summary(
                        _evaluation_subset(evaluations["sharing"], canaries),
                        _evaluation_subset(evaluations["control"], canaries),
                        failure,
                    ),
                    "paired_generalization": _paired_evaluation_summary(
                        _evaluation_subset(evaluations["sharing"], generalization),
                        _evaluation_subset(evaluations["control"], generalization),
                        failure,
                    ),
                    "sharing_elapsed_seconds": elapsed["sharing"],
                    "control_elapsed_seconds": elapsed["control"],
                    "sharing": evaluations["sharing"],
                    "control": evaluations["control"],
                }
            )
        receivers[name] = {"rows": rows}
    report = {**protocol, "receivers": receivers}
    _atomic_json(output / "report.json", report)
    return report
