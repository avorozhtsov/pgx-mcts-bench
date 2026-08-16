"""Registered equivalence pretraining for the two mastery-v3 branches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from pgx_mcts_bench.adaptive_scientists import load_scientist
from pgx_mcts_bench.mastery_v3 import CyclicGraphDualV3, CyclicMemoryDeepV3
from pgx_mcts_bench.mastery_v3_curriculum import file_sha256


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _rotations(word: list[int], count: int, seed: int) -> list[list[int]]:
    if not word:
        return [[] for _ in range(count)]
    offset = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16) % len(word)
    return [
        word[(offset + view) % len(word) :] + word[: (offset + view) % len(word)]
        for view in range(count)
    ]


def _embedding(network: CyclicMemoryDeepV3, observations: Tensor) -> Tensor:
    if isinstance(network, CyclicGraphDualV3):
        return network._graph_features(observations)[0]
    return network.encode_v3(observations)


@torch.inference_mode()
def _retrieval(network: CyclicMemoryDeepV3, views: Tensor) -> dict[str, float]:
    left = F.normalize(_embedding(network, views[:, 0]), dim=1)
    right = F.normalize(_embedding(network, views[:, 1]), dim=1)
    similarities = left @ right.T
    labels = torch.arange(left.shape[0], device=left.device)
    ranks = torch.argsort(similarities, dim=1, descending=True)
    positions = (ranks == labels[:, None]).nonzero()[:, 1] + 1
    return {
        "top1": float((positions == 1).float().mean().item()),
        "mean_reciprocal_rank": float((1.0 / positions.float()).mean().item()),
        "positive_cosine": float(similarities.diag().mean().item()),
    }


def pretrain_mastery_v3(
    checkpoint: Path,
    curriculum_path: Path,
    output: Path,
    *,
    candidate_name: str,
    steps: int = 2_000,
    batch_size: int = 32,
    views_per_identity: int = 4,
    learning_rate: float = 3e-4,
    temperature: float = 0.1,
    seed: int = 2026081701,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train only the new representation branch on replayable braid rotations."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if views_per_identity < 2:
        raise ValueError("at least two views per identity are required")
    curriculum = json.loads(curriculum_path.read_text())
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rows = [
        row
        for stage in ("simple_adaptation", "heavy_capacity")
        for row in curriculum["stages"][stage]["rows"]
    ]
    if not rows:
        raise ValueError("registered training curriculum is empty")
    scientist = load_scientist(
        candidate_name,
        checkpoint,
        seed=seed,
        device=device,
        require_factorized=True,
        objective_budget_channel=True,
    )
    network = scientist.network
    if not isinstance(network, CyclicMemoryDeepV3):
        raise TypeError(type(network).__name__)

    provenance = []
    identity_views = []
    for row_index, row in enumerate(rows):
        rotations = _rotations(
            [int(item) for item in row["word"]],
            views_per_identity,
            seed + row_index * 1_000_003,
        )
        observations = [
            scientist.game.from_word(word, int(row["strands"]), 0.0).observation
            for word in rotations
        ]
        identity_views.append(
            torch.from_numpy(np.stack(observations)).permute(0, 3, 1, 2).float()
        )
        provenance.append(
            {
                "identity": str(row.get("name") or row.get("id")),
                "representation_id": str(row["representation_id"]),
                "strands": int(row["strands"]),
                "source_word": [int(item) for item in row["word"]],
                "views": rotations,
                "transformation": "cyclic-word-rotation (closure conjugacy)",
            }
        )
    views = torch.stack(identity_views).to(device)
    network.eval()
    before = _retrieval(network, views)
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    branch_parameters = [
        parameter
        for name, parameter in network.named_parameters()
        if not name.startswith("parent.")
        and not any(
            token in name
            for token in (
                "policy_residual",
                "value_residual",
                "solve_residual",
                "cost_residual",
                "graph_policy_residual",
                "graph_value_residual",
                "graph_solve_residual",
                "graph_cost_residual",
                "invalid_capacity_head",
            )
        )
    ]
    for parameter in branch_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(branch_parameters, lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 77)
    losses: list[float] = []
    network.train()
    for _ in range(steps):
        size = min(batch_size, len(rows))
        indexes = rng.choice(len(rows), size=size, replace=False)
        left_views = rng.integers(0, views_per_identity, size=size)
        right_views = (
            left_views + rng.integers(1, views_per_identity, size=size)
        ) % views_per_identity
        row_tensor = torch.as_tensor(indexes, device=device)
        left = views[row_tensor, torch.as_tensor(left_views, device=device)]
        right = views[row_tensor, torch.as_tensor(right_views, device=device)]
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            left_embedding = F.normalize(_embedding(network, left), dim=1)
            right_embedding = F.normalize(_embedding(network, right), dim=1)
            logits = left_embedding @ right_embedding.T / temperature
            labels = torch.arange(size, device=device)
            loss = 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch_parameters, 5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    network.eval()
    after = _retrieval(network, views)
    for parameter in network.parameters():
        parameter.requires_grad_(True)

    report = {
        "schema": "mastery-v3-equivalence-pretraining-v1",
        "candidate": candidate_name,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "curriculum": str(curriculum_path.resolve()),
        "curriculum_sha256": file_sha256(curriculum_path),
        "rows": len(rows),
        "views_per_identity": views_per_identity,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "seed": seed,
        "device": device,
        "final_loss": losses[-1] if losses else None,
        "mean_last_100_loss": sum(losses[-100:]) / len(losses[-100:]) if losses else None,
        "retrieval_before": before,
        "retrieval_after": after,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(
            {
                "network": network.state_dict(),
                "candidate_spec": scientist.candidate,
                "mastery_v3_migration": source_payload.get("mastery_v3_migration"),
                "mastery_v3_pretraining": report,
            },
            temporary,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report["checkpoint"] = str(output.resolve())
    report["checkpoint_sha256"] = file_sha256(output)
    _atomic_json(output.with_suffix(output.suffix + ".json"), report)
    _atomic_json(output.with_suffix(output.suffix + ".representations.json"), provenance)
    return report
