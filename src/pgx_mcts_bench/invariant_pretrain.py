"""Self-supervised equivalence pretraining for the cyclic-memory scientist."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rf_knots.actions import CROSSING_CHANGE, PASS
from torch.nn import functional as F

from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _compatible_table,
    _sha256,
    stratified_banks,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, candidates
from pgx_mcts_bench.networks import CyclicMemoryBraidNet, make_braid_network
from pgx_mcts_bench.semantic_verifier import SemanticBraidVerifier


def _variant(verifier: SemanticBraidVerifier, knot, depth: int, seed: int) -> dict[str, Any]:
    state = verifier.state(knot.word, knot.strands)
    rng = np.random.default_rng(seed)
    actions = []
    for _ in range(depth):
        allowed = []
        for action in verifier.legal_actions(state, allow_crossing_change=False):
            kind, _, _, _ = verifier.spec.decode(int(action))
            if kind not in {CROSSING_CHANGE, PASS}:
                allowed.append(int(action))
        if not allowed:
            break
        action = int(rng.choice(allowed))
        actions.append(action)
        state = verifier.apply(state, action, allow_crossing_change=False)
    return {
        "word": list(state.word),
        "strands": state.strands,
        "actions": actions,
        "action_descriptions": [verifier.spec.describe(action) for action in actions],
        "seed": seed,
    }


def _tensor(game, representations: list[dict[str, Any]]) -> torch.Tensor:
    observations = [
        game.from_word(item["word"], item["strands"], 0.0).observation
        for item in representations
    ]
    return torch.from_numpy(np.stack(observations)).permute(0, 3, 1, 2).float()


@torch.inference_mode()
def _retrieval(network: CyclicMemoryBraidNet, views: torch.Tensor) -> dict[str, float]:
    first = F.normalize(network.encode_global(views[:, 0]), dim=1)
    second = F.normalize(network.encode_global(views[:, 1]), dim=1)
    similarities = first @ second.T
    labels = torch.arange(first.shape[0], device=first.device)
    ranks = torch.argsort(similarities, dim=1, descending=True)
    positions = (ranks == labels[:, None]).nonzero()[:, 1] + 1
    positive = similarities.diag().mean()
    negative = similarities[
        ~torch.eye(len(first), dtype=torch.bool, device=similarities.device)
    ].mean()
    return {
        "top1": float((positions == 1).float().mean()),
        "mean_reciprocal_rank": float((1.0 / positions.float()).mean()),
        "positive_cosine": float(positive),
        "negative_cosine": float(negative),
    }


def pretrain_cyclic_invariants(
    checkpoint: Path,
    output: Path,
    *,
    identities: int = 400,
    calibration_identities: int = 50,
    views_per_identity: int = 4,
    steps: int = 1_000,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    temperature: float = 0.1,
    bank_seed: int = 20260802,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    if views_per_identity < 2:
        raise ValueError("at least two views per identity are required")
    candidate = next(
        item for item in candidates() if item.name == "s-cyclic-tape8-192"
    )
    config = _config(candidate, ("R(3,12)#0", 0), seed, device, selfplay_games=1)
    game = make_game(config.game)
    verifier = SemanticBraidVerifier.from_config(config.game)
    network = make_braid_network(config.game, config.model).to(device)
    if not isinstance(network, CyclicMemoryBraidNet):
        raise TypeError(type(network).__name__)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    network.load_state_dict(payload.get("network", payload))

    bank, anchors = stratified_banks(200, 70, bank_seed)
    reserved = {item.id for item in bank + anchors}
    available = [knot for knot in _compatible_table() if knot.name not in reserved]
    available.sort(
        key=lambda knot: hashlib.sha256(f"{seed}:{knot.name}".encode()).digest()
    )
    needed = identities + calibration_identities
    if len(available) < needed:
        raise ValueError(f"only {len(available)} unreserved identities; need {needed}")
    selected = available[:needed]
    provenance = []
    tensors = []
    for identity_index, knot in enumerate(selected):
        representations = []
        for view in range(views_per_identity):
            representation_seed = (
                seed + identity_index * 1_000_003 + view * 100_003
            )
            representations.append(
                _variant(verifier, knot, view + 1, representation_seed)
            )
        provenance.append(
            {
                "knot_id": knot.name,
                "source_word": list(knot.word),
                "source_strands": knot.strands,
                "representations": representations,
            }
        )
        tensors.append(_tensor(game, representations))
    views = torch.stack(tensors).to(device)
    training = views[:identities]
    calibration = views[identities:]

    network.eval()
    before = _retrieval(network, calibration)
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    trainable = list(network.input_project.parameters()) + list(
        network.cyclic_blocks.parameters()
    )
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=1e-4
    )
    rng = np.random.default_rng(seed + 77)
    losses = []
    network.train()
    for _ in range(steps):
        indexes = rng.choice(identities, size=min(batch_size, identities), replace=False)
        left_views = rng.integers(0, views_per_identity, size=len(indexes))
        right_views = (
            left_views + rng.integers(1, views_per_identity, size=len(indexes))
        ) % views_per_identity
        rows = torch.as_tensor(indexes, device=device)
        left = training[rows, torch.as_tensor(left_views, device=device)]
        right = training[rows, torch.as_tensor(right_views, device=device)]
        left_embedding = F.normalize(network.encode_global(left), dim=1)
        right_embedding = F.normalize(network.encode_global(right), dim=1)
        logits = left_embedding @ right_embedding.T / temperature
        labels = torch.arange(len(indexes), device=device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    network.eval()
    after = _retrieval(network, calibration)
    for parameter in network.parameters():
        parameter.requires_grad_(True)

    output.parent.mkdir(parents=True, exist_ok=True)
    trained = {
        "network": network.state_dict(),
        "candidate_spec": payload.get("candidate_spec"),
        "invariant_pretraining": {
            "source_checkpoint": str(checkpoint.resolve()),
            "source_sha256": _sha256(checkpoint),
            "bank_seed": bank_seed,
            "identities": identities,
            "calibration_identities": calibration_identities,
            "views_per_identity": views_per_identity,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "seed": seed,
        },
    }
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(trained, temporary)
    os.replace(temporary, output)
    provenance_path = output.with_suffix(output.suffix + ".representations.json")
    _atomic_json(provenance_path, provenance)
    report = {
        **trained["invariant_pretraining"],
        "checkpoint": str(output.resolve()),
        "checkpoint_sha256": _sha256(output),
        "representations": str(provenance_path.resolve()),
        "final_loss": losses[-1] if losses else None,
        "mean_last_100_loss": (
            sum(losses[-100:]) / len(losses[-100:]) if losses else None
        ),
        "calibration_before": before,
        "calibration_after": after,
    }
    _atomic_json(output.with_suffix(output.suffix + ".json"), report)
    return report
