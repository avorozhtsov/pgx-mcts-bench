#!/usr/bin/env python3
"""Fit a frozen-embedding value residual on identity-disjoint proof-graph labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder
from torch import Tensor

from pgx_mcts_bench.embedding_value_adapter import (
    FrozenEmbeddingValueAdapter,
    head_position_features,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import STAGES, Candidate, _config
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.search import _observation_batch


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


def _varint(data: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return value, cursor
        shift += 7


def decode_representation(data: bytes) -> tuple[list[int], int, bool]:
    """Decode the two UnknotDB braid storage codecs used by the pinned graph."""
    if data.startswith(b"UKB0"):
        flags = data[5]
        strands = int.from_bytes(data[6:8], "little")
        length = int.from_bytes(data[8:12], "little")
        word = [
            int.from_bytes(data[index : index + 2], "little", signed=True)
            for index in range(12, 12 + 2 * length, 2)
        ]
        return word, strands, bool(flags & 1)
    if not data or data[0] != 0xB1:
        raise ValueError("unsupported UnknotDB representation codec")
    flags = data[1]
    strands, cursor = _varint(data, 2)
    length, cursor = _varint(data, cursor)
    mode = flags & 6
    if mode == 2:
        codes = [(data[cursor + index // 2] >> (4 * (index % 2))) & 0xF for index in range(length)]
    elif mode == 4:
        codes = list(data[cursor : cursor + length])
    else:
        codes = [
            int.from_bytes(data[index : index + 2], "little")
            for index in range(cursor, cursor + 2 * length, 2)
        ]
    word = [-(code // 2 + 1) if code % 2 else code // 2 + 1 for code in codes]
    return word, strands, bool(flags & 1)


def load_rows(
    identification_path: Path,
    pairs_path: Path,
    embedding_cache: dict,
    *,
    maximum_length: int,
    maximum_strands: int,
) -> list[dict]:
    node_ids = [int(node_id) for node_id in embedding_cache["node_ids"]]
    embedding_by_node = {
        node_id: embedding_cache["knot_embeddings"][index].float()
        for index, node_id in enumerate(node_ids)
    }
    pairs = sqlite3.connect(f"file:{pairs_path}?mode=ro", uri=True)
    identification = sqlite3.connect(f"file:{identification_path}?mode=ro", uri=True)
    try:
        split_by_key = {}
        for key, split, word_length in pairs.execute(
            "SELECT mirror_key,split,word_length FROM representations ORDER BY representation_id"
        ):
            split_by_key.setdefault(bytes(key), (str(split), int(word_length)))
        rows = []
        query = (
            "SELECT v.node_id,m.knot_id,v.u_upper,v.rep_key,v.encoding "
            "FROM graph_vertex_knot_map m JOIN graph_vertices v USING(rep_key)"
        )
        for node_id, knot_id, u_upper, rep_key, encoding in identification.execute(query):
            node_id = int(node_id)
            pair = split_by_key.get(bytes(rep_key))
            if node_id not in embedding_by_node or pair is None or u_upper is None:
                continue
            word, strands, cyclic = decode_representation(bytes(encoding))
            if cyclic or len(word) > maximum_length or strands > maximum_strands:
                continue
            rows.append(
                {
                    "node_id": node_id,
                    "knot_id": str(knot_id),
                    "u_upper": int(u_upper),
                    "split": pair[0],
                    "word": word,
                    "strands": strands,
                    "embedding": embedding_by_node[node_id],
                }
            )
    finally:
        pairs.close()
        identification.close()
    best = {}
    for row in rows:
        previous = best.get(row["knot_id"])
        rank = (len(row["word"]), row["node_id"])
        if previous is None or rank < (len(previous["word"]), previous["node_id"]):
            best[row["knot_id"]] = row
    return sorted(best.values(), key=lambda row: row["knot_id"])


def load_parent(manifest_path: Path, checkpoint_path: Path, device: str):
    manifest = json.loads(manifest_path.read_text())
    candidate = Candidate(**manifest["candidate"])
    config = _config(candidate, STAGES[0], 20260830, device, selfplay_games=1)
    game = make_game(config.game)
    parent = make_braid_network(config.game, config.model).to(device)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = saved.get("network", saved) if isinstance(saved, dict) else saved
    load_policy_value_state_dict(parent, state)
    parent.eval()
    return candidate, config, game, parent


def load_embedding_model(checkpoint_path: Path, device: str) -> ScalableBraidEmbeddingEncoder:
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = saved["config"]
    model = ScalableBraidEmbeddingEncoder(
        width=int(config["width"]),
        depth=int(config["depth"]),
        embedding_dim=int(config["embedding_dim"]),
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def head_views(game, row: dict, count: int, ratio: float) -> list[tuple[object, np.ndarray, int]]:
    initial = game.from_word(row["word"], row["strands"], math.log(ratio))
    length = max(len(row["word"]), 1)
    heads = sorted({index * length // count for index in range(count)})
    out = []
    state = initial.state
    for head in heads:
        view = game._view(  # noqa: SLF001 - exact read-only serial view for offline fitting
            state.pgx,
            head,
            state.registers,
            state.colours,
            state.colour,
            state.tape,
            reward=0.0,
            internal_steps=state.internal_steps,
            semantic_moves=state.semantic_moves,
            invariant_vector=state.invariant_vector,
        )
        out.append((view.state, view.observation, head))
    return out


@torch.inference_mode()
def materialize(
    rows: list[dict],
    game,
    parent,
    *,
    heads: int,
    ratio: float,
    budget: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Tensor]:
    observations = []
    contexts = []
    targets = []
    row_indexes = []
    for row_index, row in enumerate(rows):
        for _, observation, head in head_views(game, row, heads, ratio):
            observations.append(observation)
            position = head_position_features(head, len(row["word"]), device=torch.device("cpu"))
            contexts.append(torch.cat((row["embedding"], position)))
            # A replayed U upper bound gives a feasible crossing cost. At ratio
            # 1000, omitting the route's braid moves changes value by <0.002.
            target = 1.0 - 2.0 * ratio * row["u_upper"] / ((ratio + 1.0) * budget)
            targets.append(max(-1.0, min(1.0, target)))
            row_indexes.append(row_index)
    observation_tensor = _observation_batch(observations, torch.device("cpu"))
    values = []
    for start in range(0, len(observations), batch_size):
        _, batch_values = parent(observation_tensor[start : start + batch_size].to(device))
        values.append(batch_values.cpu())
    return {
        "observation": observation_tensor,
        "context": torch.stack(contexts),
        "target": torch.tensor(targets, dtype=torch.float32),
        "base": torch.cat(values),
        "row_index": torch.tensor(row_indexes),
    }


def metrics(target: Tensor, prediction: Tensor, row_index: Tensor) -> dict[str, float]:
    distinct = int(row_index.max().item()) + 1
    target_by_row = torch.zeros(distinct).scatter_reduce_(
        0, row_index, target, reduce="mean", include_self=False
    )
    prediction_by_row = torch.zeros(distinct).scatter_reduce_(
        0, row_index, prediction, reduce="mean", include_self=False
    )
    left, right = torch.triu_indices(distinct, distinct, offset=1)
    delta_target = target_by_row[left] - target_by_row[right]
    keep = delta_target != 0
    ordering = (
        ((prediction_by_row[left] - prediction_by_row[right])[keep] * delta_target[keep] > 0)
        .float()
        .mean()
        .item()
    )
    centred_target = target_by_row - target_by_row.mean()
    centred_prediction = prediction_by_row - prediction_by_row.mean()
    pearson = (
        (centred_target * centred_prediction).sum()
        / (centred_target.square().sum().sqrt() * centred_prediction.square().sum().sqrt())
    ).item()
    return {
        "mae": (target_by_row - prediction_by_row).abs().mean().item(),
        "pairwise_ordering_accuracy": ordering,
        "pearson": pearson,
    }


def prediction(adapter, data: dict[str, Tensor], device: torch.device, batch_size: int) -> Tensor:
    rows = []
    with torch.inference_mode():
        for start in range(0, len(data["target"]), batch_size):
            index = slice(start, start + batch_size)
            base = data["base"][index].to(device)
            residual = adapter.value_residual(
                data["observation"][index].to(device),
                base,
                data["context"][index].to(device),
            )
            rows.append(torch.clamp(base + residual, -1.0, 1.0).cpu())
    return torch.cat(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--identification", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--maximum-value-residual", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument(
        "--loss-mode",
        choices=("calibrated-return", "ranking-residual"),
        default="calibrated-return",
    )
    parser.add_argument("--ranking-margin", type=float, default=0.02)
    parser.add_argument("--preservation-weight", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.heads < 1:
        raise ValueError("invalid training dose")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    candidate, config, game, parent = load_parent(
        args.parent_manifest, args.parent_checkpoint, args.device
    )
    embedding_model = load_embedding_model(args.embedding_checkpoint, args.device)
    embedding_cache = torch.load(args.embedding_cache, map_location="cpu", weights_only=False)
    rows = load_rows(
        args.identification,
        args.pairs,
        embedding_cache,
        maximum_length=config.game.max_len,
        maximum_strands=config.game.max_strands,
    )
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    materialized = {
        split: materialize(
            split_rows[split],
            game,
            parent,
            heads=args.heads,
            ratio=args.ratio,
            budget=config.game.simplify_budget,
            batch_size=args.batch_size,
            device=device,
        )
        for split in ("train", "validation")
    }
    adapter = FrozenEmbeddingValueAdapter(
        parent,
        embedding_model,
        observation_channels=config.game.observation_channels,
        width=args.width,
        maximum_value_residual=args.maximum_value_residual,
    ).to(device)
    adapter.train()
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)

    zero_validation = prediction(adapter, materialized["validation"], device, args.batch_size)
    if not torch.equal(zero_validation, materialized["validation"]["base"]):
        raise RuntimeError("zero-initialized adapter changed value before training")
    best_loss = math.inf
    best_selection_score = -math.inf
    best_state = None
    generator = torch.Generator().manual_seed(args.seed)
    train = materialized["train"]
    for step in range(1, args.steps + 1):
        indexes = torch.randint(len(train["target"]), (args.batch_size,), generator=generator)
        observation = train["observation"][indexes].to(device)
        context = train["context"][indexes].to(device)
        base = train["base"][indexes].to(device)
        target = train["target"][indexes].to(device)
        residual = adapter.value_residual(observation, base, context)
        unbounded = base + residual
        fitted = torch.clamp(unbounded, -1.0, 1.0)
        value_loss = torch.nn.functional.smooth_l1_loss(fitted, target)
        permutation = torch.randperm(len(indexes), generator=generator).to(device)
        target_delta = target - target[permutation]
        informative = target_delta != 0
        if informative.any():
            if args.loss_mode == "ranking-residual":
                direction = target_delta[informative].sign()
                margin = target_delta[informative].abs().clamp_max(args.ranking_margin)
                ranking_loss = torch.relu(
                    margin - direction * (unbounded - unbounded[permutation])[informative]
                ).mean()
            else:
                ranking_loss = torch.nn.functional.smooth_l1_loss(
                    (fitted - fitted[permutation])[informative],
                    target_delta[informative],
                )
        else:
            ranking_loss = fitted.new_zeros(())
        preservation = residual.square().mean() + residual.mean().square()
        loss = (
            ranking_loss + args.preservation_weight * preservation
            if args.loss_mode == "ranking-residual"
            else value_loss + args.ranking_weight * ranking_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            validation_prediction = prediction(
                adapter, materialized["validation"], device, args.batch_size
            )
            validation_loss = torch.nn.functional.smooth_l1_loss(
                validation_prediction, materialized["validation"]["target"]
            ).item()
            validation_metrics = metrics(
                materialized["validation"]["target"],
                validation_prediction,
                materialized["validation"]["row_index"],
            )
            residual_size = (
                (validation_prediction - materialized["validation"]["base"]).abs().mean().item()
            )
            selection_score = (
                validation_metrics["pairwise_ordering_accuracy"] - 0.05 * residual_size
                if args.loss_mode == "ranking-residual"
                else -validation_loss
            )
            if selection_score > best_selection_score:
                best_loss = validation_loss
                best_selection_score = selection_score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in adapter.state_dict().items()
                    if name.startswith(("local.", "global_context.", "residual."))
                }
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    adapter.load_state_dict(best_state, strict=False)
    validation_prediction = prediction(adapter, materialized["validation"], device, args.batch_size)
    adapter_state = {
        name: value.detach().cpu()
        for name, value in adapter.state_dict().items()
        if name.startswith(("local.", "global_context.", "residual."))
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "adapter.pt"
    torch.save(
        {
            "schema": "pgx-frozen-embedding-value-adapter-v0",
            "adapter": adapter_state,
            "candidate": asdict(candidate),
            "width": args.width,
            "maximum_value_residual": args.maximum_value_residual,
            "embedding_dim": 64,
            "position_features": "encoder-aligned-sincos-1-2-4-plus-phase-v0",
            "parent_checkpoint_sha256": sha256(args.parent_checkpoint),
            "embedding_checkpoint_sha256": sha256(args.embedding_checkpoint),
        },
        checkpoint,
    )
    report = {
        "schema": "pgx-frozen-embedding-value-adapter-training-v0",
        "status": "offline_validation_complete_mcts_not_yet_evaluated",
        "inputs": {
            "parent_manifest": str(args.parent_manifest.resolve()),
            "parent_checkpoint": str(args.parent_checkpoint.resolve()),
            "parent_checkpoint_sha256": sha256(args.parent_checkpoint),
            "embedding_checkpoint": str(args.embedding_checkpoint.resolve()),
            "embedding_checkpoint_sha256": sha256(args.embedding_checkpoint),
            "embedding_cache_sha256": sha256(args.embedding_cache),
            "identification_sha256": sha256(args.identification),
            "pairs_sha256": sha256(args.pairs),
        },
        "protocol": {
            "identity_disjoint_split": True,
            "one_shortest_representation_per_knot": True,
            "test_split_used": False,
            "heads_per_word": args.heads,
            "head_position_encoding": "same 1x/2x/4x cyclic basis as frozen encoder",
            "embedding_cache_key": "exact word plus strands; head excluded",
            "ratio": args.ratio,
            "target": "optimistic crossing-dominant return from replayed U upper bound",
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "ranking_weight": args.ranking_weight,
            "loss_mode": args.loss_mode,
            "ranking_margin": args.ranking_margin,
            "preservation_weight": args.preservation_weight,
            "maximum_value_residual": args.maximum_value_residual,
            "seed": args.seed,
        },
        "rows": {split: len(split_rows[split]) for split in split_rows},
        "views": {split: len(materialized[split]["target"]) for split in materialized},
        "parameters": {
            "trainable": sum(parameter.numel() for parameter in trainable),
            "parent_frozen": True,
            "embedding_frozen": True,
            "policy_unchanged_by_construction": True,
            "zero_initial_value_exact": True,
        },
        "validation": {
            "base": metrics(
                materialized["validation"]["target"],
                materialized["validation"]["base"],
                materialized["validation"]["row_index"],
            ),
            "adapter": metrics(
                materialized["validation"]["target"],
                validation_prediction,
                materialized["validation"]["row_index"],
            ),
            "best_smooth_l1": best_loss,
            "best_selection_score": best_selection_score,
        },
        "checkpoint": str(checkpoint.resolve()),
        "interpretation_contract": (
            "This establishes only held-out offline value calibration against replayed upper "
            "bounds. It does not establish an MCTS solve-rate improvement."
        ),
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
