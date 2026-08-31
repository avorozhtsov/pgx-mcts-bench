#!/usr/bin/env python3
"""Train the frozen-embedding residual on actual played MCTS outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder
from torch import Tensor
from torch.nn import functional as F

from pgx_mcts_bench.embedding_value_adapter import (
    FrozenEmbeddingValueAdapter,
    head_position_features,
)
from pgx_mcts_bench.ladder import STAGES, Candidate, _config
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.on_policy_embedding_value import SCHEMA, split_mask


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_embedding(path: Path, device: torch.device) -> ScalableBraidEmbeddingEncoder:
    saved = torch.load(path, map_location=device, weights_only=False)
    config = saved["config"]
    model = ScalableBraidEmbeddingEncoder(
        width=int(config["width"]),
        depth=int(config["depth"]),
        embedding_dim=int(config["embedding_dim"]),
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


@torch.inference_mode()
def materialize_contexts(
    data: dict,
    embedding_model,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    output = []
    words = data["words"]
    for start in range(0, len(words), batch_size):
        batch_words = words[start : start + batch_size]
        lengths = torch.tensor([len(word) for word in batch_words], device=device)
        width = max(int(lengths.max().item()), 1)
        padded = torch.zeros((len(batch_words), width), dtype=torch.long, device=device)
        for row, word in enumerate(batch_words):
            if word:
                padded[row, : len(word)] = torch.tensor(word, dtype=torch.long, device=device)
        strands = data["strands"][start : start + batch_size].to(device)
        embedding = embedding_model(padded, lengths, strands)["knot"]
        position = torch.stack(
            [
                head_position_features(int(head), len(word), device=device)
                for head, word in zip(
                    data["heads"][start : start + batch_size], batch_words, strict=True
                )
            ]
        )
        output.append(torch.cat((embedding, position), dim=1).cpu())
    return torch.cat(output)


@torch.inference_mode()
def parent_values(parent, observations: Tensor, batch_size: int, device: torch.device) -> Tensor:
    values = []
    for start in range(0, len(observations), batch_size):
        _, value = parent(observations[start : start + batch_size].to(device))
        values.append(value.cpu())
    return torch.cat(values)


@torch.inference_mode()
def predictions(
    adapter, data: dict[str, Tensor], indexes: Tensor, batch_size: int, device
) -> Tensor:
    output = []
    for start in range(0, len(indexes), batch_size):
        selected = indexes[start : start + batch_size]
        base = data["base"][selected].to(device)
        residual = adapter.value_residual(
            data["observation"][selected].to(device),
            base,
            data["context"][selected].to(device),
        )
        output.append(torch.clamp(base + residual, -1.0, 1.0).cpu())
    return torch.cat(output)


def metrics(target: Tensor, prediction: Tensor, episode_ids: Tensor, positions: Tensor) -> dict:
    initial = positions == 0
    target_initial = target[initial]
    prediction_initial = prediction[initial]
    if len(target_initial) > 1:
        left, right = torch.triu_indices(len(target_initial), len(target_initial), offset=1)
        delta = target_initial[left] - target_initial[right]
        informative = delta != 0
        ordering = (
            (
                (prediction_initial[left] - prediction_initial[right])[informative]
                * delta[informative]
                > 0
            )
            .float()
            .mean()
            .item()
            if informative.any()
            else float("nan")
        )
    else:
        ordering = float("nan")
    unique_episodes = torch.unique(episode_ids).numel()
    successful = target > -0.5
    failed = ~successful
    class_losses = [
        F.smooth_l1_loss(prediction[mask], target[mask])
        for mask in (successful, failed)
        if mask.any()
    ]
    return {
        "positions": int(len(target)),
        "episodes": int(unique_episodes),
        "mae_all_positions": (prediction - target).abs().mean().item(),
        "smooth_l1_all_positions": F.smooth_l1_loss(prediction, target).item(),
        "smooth_l1_balanced_outcomes": torch.stack(class_losses).mean().item(),
        "mae_successful_positions": (
            (prediction[successful] - target[successful]).abs().mean().item()
            if successful.any()
            else float("nan")
        ),
        "mae_failed_positions": (
            (prediction[failed] - target[failed]).abs().mean().item()
            if failed.any()
            else float("nan")
        ),
        "mae_initial_positions": (prediction_initial - target_initial).abs().mean().item(),
        "pairwise_initial_ordering": ordering,
        "mean_prediction": prediction.mean().item(),
        "mean_target": target.mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--maximum-value-residual", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--preservation-weight", type=float, default=0.1)
    parser.add_argument("--context-mode", choices=("full", "head-only"), default="full")
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2:
        raise ValueError("steps must be positive and batch size at least two")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    saved_data = torch.load(args.data, map_location="cpu", weights_only=False)
    if saved_data.get("schema") != SCHEMA or saved_data.get("status") != "complete":
        raise ValueError("unsupported or incomplete on-policy dataset")
    raw = saved_data["data"]
    manifest = json.loads(args.parent_manifest.read_text())
    candidate = Candidate(**manifest["candidate"])
    config = _config(candidate, STAGES[0], args.seed, args.device, selfplay_games=1)
    parent = make_braid_network(config.game, config.model).to(device)
    saved_parent = torch.load(args.parent_checkpoint, map_location=device, weights_only=False)
    load_policy_value_state_dict(parent, saved_parent.get("network", saved_parent))
    parent.eval()
    embedding_model = load_embedding(args.embedding_checkpoint, device)
    data = {
        **raw,
        "context": materialize_contexts(
            raw, embedding_model, batch_size=args.batch_size, device=device
        ),
        "base": parent_values(parent, raw["observation"], args.batch_size, device),
    }
    if args.context_mode == "head-only":
        data["context"] = torch.cat(
            (torch.zeros_like(data["context"][:, :64]), data["context"][:, 64:]), dim=1
        )
    train_indexes = torch.nonzero(split_mask(raw, "train"), as_tuple=False).flatten()
    validation_indexes = torch.nonzero(split_mask(raw, "validation"), as_tuple=False).flatten()
    if not len(train_indexes) or not len(validation_indexes):
        raise ValueError("dataset must contain train and validation rows")

    adapter = FrozenEmbeddingValueAdapter(
        parent,
        embedding_model,
        observation_channels=config.game.observation_channels,
        width=args.width,
        maximum_value_residual=args.maximum_value_residual,
        use_embedding=args.context_mode == "full",
    ).to(device)
    adapter.train()
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    zero = predictions(adapter, data, validation_indexes, args.batch_size, device)
    if not torch.equal(zero, data["base"][validation_indexes]):
        raise RuntimeError("zero-initialized adapter changed value before training")

    generator = torch.Generator().manual_seed(args.seed)
    successful_train = train_indexes[data["solved"][train_indexes]]
    failed_train = train_indexes[~data["solved"][train_indexes]]
    if not len(successful_train) or not len(failed_train):
        raise ValueError("training data must contain successful and failed trajectories")
    validation_solved = data["solved"][validation_indexes]
    if validation_solved.all() or (~validation_solved).all():
        raise ValueError("validation data must contain successful and failed trajectories")
    best_score = math.inf
    best_state = None
    for step in range(1, args.steps + 1):
        successful_count = args.batch_size // 2
        failed_count = args.batch_size - successful_count
        sampled = torch.cat(
            (
                successful_train[
                    torch.randint(len(successful_train), (successful_count,), generator=generator)
                ],
                failed_train[
                    torch.randint(len(failed_train), (failed_count,), generator=generator)
                ],
            )
        )
        observation = data["observation"][sampled].to(device)
        context = data["context"][sampled].to(device)
        base = data["base"][sampled].to(device)
        target = data["targets"][sampled].to(device)
        residual = adapter.value_residual(observation, base, context)
        fitted = torch.clamp(base + residual, -1.0, 1.0)
        value_loss = F.smooth_l1_loss(fitted, target)
        permutation = torch.randperm(len(sampled), generator=generator).to(device)
        target_delta = target - target[permutation]
        informative = target_delta != 0
        ranking_loss = (
            F.softplus(
                -target_delta[informative].sign() * (fitted - fitted[permutation])[informative]
            ).mean()
            if informative.any()
            else fitted.new_zeros(())
        )
        preservation = residual.square().mean()
        loss = (
            value_loss
            + args.ranking_weight * ranking_loss
            + args.preservation_weight * preservation
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            validation = predictions(adapter, data, validation_indexes, args.batch_size, device)
            validation_target = data["targets"][validation_indexes]
            validation_successful = data["solved"][validation_indexes]
            score = 0.5 * (
                F.smooth_l1_loss(
                    validation[validation_successful],
                    validation_target[validation_successful],
                ).item()
                + F.smooth_l1_loss(
                    validation[~validation_successful],
                    validation_target[~validation_successful],
                ).item()
            )
            score += 0.02 * (validation - data["base"][validation_indexes]).abs().mean().item()
            if score < best_score:
                best_score = score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in adapter.state_dict().items()
                    if name.startswith(("local.", "global_context.", "residual."))
                }
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    adapter.load_state_dict(best_state, strict=False)
    adapter.eval()
    validation = predictions(adapter, data, validation_indexes, args.batch_size, device)
    base_validation = data["base"][validation_indexes]
    target_validation = data["targets"][validation_indexes]
    episodes_validation = data["episode_ids"][validation_indexes]
    positions_validation = data["position_indexes"][validation_indexes]

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "adapter.pt"
    torch.save(
        {
            "schema": "pgx-frozen-embedding-value-adapter-v0",
            "adapter": best_state,
            "candidate": asdict(candidate),
            "width": args.width,
            "maximum_value_residual": args.maximum_value_residual,
            "embedding_dim": 64,
            "position_features": "encoder-aligned-sincos-1-2-4-plus-phase-v0",
            "parent_checkpoint_sha256": sha256(args.parent_checkpoint),
            "embedding_checkpoint_sha256": sha256(args.embedding_checkpoint),
            "on_policy_data_sha256": sha256(args.data),
            "use_embedding": args.context_mode == "full",
        },
        checkpoint,
    )
    report = {
        "schema": "pgx-on-policy-embedding-value-adapter-training-v0",
        "status": "offline_validation_complete_mcts_not_yet_evaluated",
        "inputs": {
            "parent_checkpoint_sha256": sha256(args.parent_checkpoint),
            "embedding_checkpoint_sha256": sha256(args.embedding_checkpoint),
            "on_policy_data_sha256": sha256(args.data),
        },
        "protocol": {
            "target": "actual terminal semantic payoff",
            "policy_unchanged": True,
            "parent_frozen": True,
            "embedding_frozen": True,
            "test_split_used": False,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "ranking_weight": args.ranking_weight,
            "preservation_weight": args.preservation_weight,
            "maximum_value_residual": args.maximum_value_residual,
            "context_mode": args.context_mode,
            "outcome_balanced_training_batches": True,
            "natural_frequency_validation": True,
            "seed": args.seed,
        },
        "parameters": {
            "trainable": sum(parameter.numel() for parameter in trainable),
            "zero_initial_value_exact": True,
        },
        "validation": {
            "base": metrics(
                target_validation,
                base_validation,
                episodes_validation,
                positions_validation,
            ),
            "adapter": metrics(
                target_validation,
                validation,
                episodes_validation,
                positions_validation,
            ),
            "mean_absolute_residual": (validation - base_validation).abs().mean().item(),
            "selection_score": best_score,
        },
        "checkpoint": str(checkpoint.resolve()),
        "interpretation_contract": (
            "Stage-disjoint offline validation against actual played outcomes is not an "
            "MCTS promotion result. A fresh equal-budget paired panel is still required."
        ),
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
