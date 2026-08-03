"""Function-preserving initialization for the cyclic-memory scientist."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from pgx_mcts_bench.ladder import _config, candidates
from pgx_mcts_bench.networks import CyclicMemoryBraidNet, make_braid_network


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
