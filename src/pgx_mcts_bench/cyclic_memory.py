"""Function-preserving initialization for the cyclic-memory scientist."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, candidates
from pgx_mcts_bench.networks import (
    CyclicMemoryBraidNet,
    load_policy_value_state_dict,
    make_braid_network,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cyclic_memory_checkpoint(
    window_checkpoint: Path,
    output: Path,
    *,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    candidate = next(
        item for item in candidates() if item.name == "s-cyclic-tape8-192"
    )
    config = _config(candidate, ("R(3,12)#0", 0), seed, device, selfplay_games=1)
    network = make_braid_network(config.game, config.model).to(device)
    if not isinstance(network, CyclicMemoryBraidNet):
        raise TypeError(f"unexpected network type: {type(network).__name__}")
    parent = torch.load(window_checkpoint, map_location=device, weights_only=False)
    network.load_window_state_dict(parent.get("network", parent))
    payload = {
        "network": network.state_dict(),
        "candidate_spec": asdict(candidate),
        "initialization": {
            "kind": "function-preserving-window-plus-zero-residual",
            "window_checkpoint": str(window_checkpoint.resolve()),
            "window_sha256": _sha256(window_checkpoint),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary)
    os.replace(temporary, output)
    parameters = sum(parameter.numel() for parameter in network.parameters())
    trainable = sum(
        parameter.numel() for parameter in network.parameters() if parameter.requires_grad
    )
    report = {
        "checkpoint": str(output.resolve()),
        "checkpoint_sha256": _sha256(output),
        "candidate": candidate.name,
        "parameters": parameters,
        "trainable_parameters": trainable,
        **payload["initialization"],
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


@torch.inference_mode()
def modernize_cyclic_memory_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Persist the FiLM, remaining-L and factorized-head schema exactly.

    The old cyclic checkpoint remains the controller source.  Every new route
    into policy/value is initialized as an identity or zero residual; auxiliary
    heads exist in the saved artifact but are deliberately marked untrained.
    A separate protected curriculum must train and admit them.
    """
    candidate = next(item for item in candidates() if item.name == "s-cyclic-tape8-192")
    legacy_candidate = replace(
        candidate,
        objective_budget_channel=False,
        auxiliary_solve_backprop_to_encoder=False,
        auxiliary_budget_monotonic_weight=0.0,
        auxiliary_budget_conditioning=False,
        use_auxiliary_value=False,
    )
    stage = ("R(3,12)#0", 0)
    legacy_config = _config(legacy_candidate, stage, seed, device, selfplay_games=1)
    modern_config = _config(candidate, stage, seed, device, selfplay_games=1)
    legacy = make_braid_network(legacy_config.game, legacy_config.model).to(device).eval()
    modern = make_braid_network(modern_config.game, modern_config.model).to(device).eval()
    if not isinstance(legacy, CyclicMemoryBraidNet) or not isinstance(
        modern, CyclicMemoryBraidNet
    ):
        raise TypeError("cyclic modernization requires CyclicMemoryBraidNet")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("network", payload)
    load_policy_value_state_dict(legacy, state)
    migrated = load_policy_value_state_dict(modern, state)
    if not migrated:
        raise ValueError("checkpoint already matches the modern cyclic schema")

    legacy_game = make_game(legacy_config.game)
    modern_game = make_game(modern_config.game)
    maxima = {"policy": 0.0, "value": 0.0}
    probes = (
        ([1, -1], 2),
        ([1, 2, -1, -2, 1], 3),
        ([1, 2, 3, -2, -1, 2, -3], 4),
    )
    for word, strands in probes:
        for ratio in (10.0, 1000.0):
            old_transition = legacy_game.from_word(word, strands, math.log(ratio))
            new_transition = modern_game.from_word(word, strands, math.log(ratio))
            old_observation = (
                torch.from_numpy(np.asarray(old_transition.observation))
                .permute(2, 0, 1)[None]
                .float()
                .to(device)
            )
            new_observation = (
                torch.from_numpy(np.asarray(new_transition.observation))
                .permute(2, 0, 1)[None]
                .float()
                .to(device)
            )
            old_policy, old_value = legacy(old_observation)
            new_policy, new_value = modern(new_observation)
            maxima["policy"] = max(
                maxima["policy"], float((old_policy - new_policy).abs().max().item())
            )
            maxima["value"] = max(
                maxima["value"], float((old_value - new_value).abs().max().item())
            )
    if max(maxima.values()) > 1e-6:
        raise AssertionError(f"cyclic modernization changed controller outputs: {maxima}")

    modern_payload = {
        **payload,
        "network": modern.state_dict(),
        "candidate": candidate.name,
        "candidate_spec": asdict(candidate),
        "modernization": {
            "kind": "function-preserving-film-budget-factorized-v1",
            "source_checkpoint": str(checkpoint.resolve()),
            "source_sha256": _sha256(checkpoint),
            "max_abs_difference": maxima,
            "film_trained": False,
            "factorized_heads_trained": False,
        },
    }
    modern_payload.pop("optimizer", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(modern_payload, temporary)
    os.replace(temporary, output)
    report = {
        "checkpoint": str(output.resolve()),
        "checkpoint_sha256": _sha256(output),
        "candidate": candidate.name,
        **modern_payload["modernization"],
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
