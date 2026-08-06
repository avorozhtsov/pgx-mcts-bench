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
            verified, _ = _evaluation_record(
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
) -> dict[str, Any]:
    torch.set_num_threads(1)
    scientist = load_scientist(
        name,
        Path(checkpoint),
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=False,
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
            scientist.replay.add(record, representation_id=identity)
            translated.append(identity)
        option_positions = [
            position
            for record in scientist.replay.games
            for position in record
            if position.option_state is not None and position.target_external_action >= 0
        ]
        scientist.network.policy_value_preservation_weight = max(
            float(getattr(scientist.network, "policy_value_preservation_weight", 0.0)),
            BUDGET_PRESERVATION_WEIGHTS.get(name, 1.0),
        )
        attach_policy_value_preservation_teacher(scientist.network)
        with torch.inference_mode():
            initial_loss = float(
                stable_option_route_loss(
                    scientist.network,
                    scientist.game,
                    option_positions,
                    horizon=5,
                    device=torch.device(device),
                ).item()
            )
        losses = []
        for _ in range(train_steps):
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
                )
            )
        with torch.inference_mode():
            final_loss = float(
                stable_option_route_loss(
                    scientist.network,
                    scientist.game,
                    option_positions,
                    horizon=5,
                    device=torch.device(device),
                ).item()
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
    evaluation_games: int = 4,
    update_cycles: int = 8,
    batch_size: int = 16,
    option_learning_rate_scale: float = 1.0,
    option_target_reduction: float = 0.1,
    max_option_steps: int = 16,
    witness_bank: Path | None = None,
    item_ids: tuple[str, ...] = (),
    target_item_ids: tuple[str, ...] = (),
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
    if not 0.0 < option_target_reduction < 1.0:
        raise ValueError("option_target_reduction must be between zero and one")
    if max_option_steps < 1:
        raise ValueError("max_option_steps must be positive")
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
        for identity, row in witness_provenance["witnesses"].items():
            witnesses[identity] = (
                int(row["crossing_changes"]),
                int(row["moves"]),
                [int(action) for action in row["semantic_actions"]],
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
    missing_targets = sorted(set(training_ids) - set(witnesses))
    if missing_targets:
        raise ValueError(f"target identities lack certified witnesses: {missing_targets}")
    unevaluated_targets = sorted(set(training_ids) - set(evaluation_ids))
    if unevaluated_targets:
        raise ValueError(f"target identities are not evaluation items: {unevaluated_targets}")
    protocol = {
        "schema": "threshold-compute-matched-option-adapter-sharing-v5",
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
        "witness_items": sorted(witnesses),
        "ratio": ratio,
        "simulations": simulations,
        "evaluation_games": evaluation_games,
        "update_cycles": update_cycles,
        "batch_size": batch_size,
        "option_learning_rate_scale": option_learning_rate_scale,
        "option_target_reduction": option_target_reduction,
        "max_option_steps": max_option_steps,
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
            "canonical-route adapter updates after one native replay-v3 update, "
            "stopping at target loss reduction or the hard step cap"
        ),
        "control_update": (
            "native replay-v3 updates matched to at least the adapter optimizer "
            "steps and routed state-action targets"
        ),
        "compute_match_unit": "network state examples consumed by training loss",
        "internal_action_cap": 5,
        "seed": seed,
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "manifest.json", protocol)
    failure = ratio * 20 + 64
    reports = {}
    all_noninferior = True
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
            objective_budget_channel=False,
        )
        control = load_scientist(
            name,
            checkpoint,
            seed=scientist_seed,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=False,
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
            lr=sharing.config.train.learning_rate,
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
                simulations=simulations,
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
                simulations=simulations,
                games=evaluation_games,
                seed=evaluation_seed,
            )
        for identity, record in native_records:
            embedding = _replay_representation_embedding(by_id[identity].knot)
            for scientist in (sharing, control):
                scientist.replay.set_representation_embedding(identity, embedding)
                scientist.replay.add(
                    copy.deepcopy(record), representation_id=identity
                )
        translated = []
        translated_records = []
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
            sharing.replay.add(record, representation_id=identity)
            translated.append(identity)
            translated_records.append((identity, record))
        option_positions = [
            position
            for record in sharing.replay.games
            for position in record
            if position.option_state is not None and position.target_external_action >= 0
        ]
        if not translated_records:
            raise ValueError(f"no donor witness translated into {name}")
        training_pairs = [
            (identity, record)
            for identity, record in translated_records
            if identity in set(training_ids)
        ]
        if not training_pairs:
            raise ValueError(f"no target witness translated into {name}")
        training_records = [record for _, record in training_pairs]
        retention_positions = []
        for item in items:
            if item.id in set(training_ids):
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
        with torch.inference_mode():
            initial_option_loss = float(
                stable_option_route_loss(
                    sharing.network,
                    sharing.game,
                    option_positions,
                    horizon=5,
                    device=torch.device(device),
                ).item()
            )
        updates = []
        for cycle in range(update_cycles):
            current = training_ids[cycle % len(training_ids)]
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
                        scientist.replay.add(record, representation_id=current)
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
            target_loss = None
            training_record = training_records[cycle % len(training_records)]
            for option_step in range(max_option_steps):
                option_diagnostics: dict[str, float | str] = {}
                option_loss = train_bounded_option_step(
                    sharing.network,
                    option_optimizer,
                    sharing.game,
                    sharing.replay,
                    sharing.replay.rng,
                    batch_size=1,
                    horizon=5,
                    beam_width=8,
                    device=torch.device(device),
                    positions=training_record,
                    learning_rate_scale=option_learning_rate_scale,
                    adapter_only=True,
                    stable_routes=True,
                    preservation_positions=(
                        preservation_positions if gated_adapter else None
                    ),
                    route_gate_weight=route_gate_weight if gated_adapter else 0.0,
                    off_route_kl_weight=off_route_kl_weight if gated_adapter else 0.0,
                    off_route_gate_weight=(
                        off_route_gate_weight if gated_adapter else 0.0
                    ),
                    diagnostics=option_diagnostics,
                )
                if target_loss is None:
                    target_loss = float(option_diagnostics["loss_before"]) * (
                        1.0 - option_target_reduction
                    )
                option_diagnostics["step"] = float(option_step)
                option_updates.append(option_diagnostics)
                if float(option_diagnostics["loss_after"]) <= target_loss:
                    break
            option_seconds = time.perf_counter() - option_started
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
            updates.append(
                {
                    "cycle": cycle,
                    "current": current,
                    "sharing_native": sharing_native,
                    "option_loss": option_loss,
                    "option_updates": option_updates,
                    "option_target_loss": target_loss,
                    "option_target_reached": (
                        bool(option_updates)
                        and float(option_updates[-1]["loss_after"]) <= float(target_loss)
                    ),
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
        with torch.inference_mode():
            final_option_loss = float(
                stable_option_route_loss(
                    sharing.network,
                    sharing.game,
                    option_positions,
                    horizon=5,
                    device=torch.device(device),
                ).item()
            )
        sharing_after = _evaluate_parallel(
            sharing,
            items,
            ratio=ratio,
            simulations=simulations,
            games=evaluation_games,
            seed=evaluation_seed,
            device=device,
            workers=evaluation_workers,
        )
        control_after = _evaluate_parallel(
            control,
            items,
            ratio=ratio,
            simulations=simulations,
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
        noninferior = (
            not control_only
            and not lost_from_before
            and sharing_loss <= control_loss
            and final_option_loss < initial_option_loss
        )
        all_noninferior &= noninferior
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
            "translated_receiver_unsolved": sorted(set(translated) - before_set),
            "trained_witnesses": [identity for identity, _ in training_pairs],
            "option_positions": len(option_positions),
            "initial_option_loss": initial_option_loss,
            "final_option_loss": final_option_loss,
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
            "passed_noninferiority": noninferior,
        }
    passed = all_noninferior and any_sharing_only
    report = {
        **protocol,
        "receivers": reports,
        "decision": {
            "passed": passed,
            "all_receivers_noninferior": all_noninferior,
            "at_least_one_sharing_only_identity": any_sharing_only,
            "next_step": (
                "finalize scientist roster" if passed else "keep sharing arms closed"
            ),
        },
    }
    _atomic_json(output / "report.json", report)
    return report
