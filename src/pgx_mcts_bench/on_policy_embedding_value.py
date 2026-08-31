"""Exact, replayable rows for on-policy full-braid value adaptation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from pgx_mcts_bench.embedding_value_adapter import _full_braid, _serial_head
from pgx_mcts_bench.search import _observation_batch

SCHEMA = "pgx-on-policy-embedding-value-data-v0"


def state_row(
    state: Any,
    observation: np.ndarray,
    *,
    game: Any,
    player: int,
    root_value: float,
    episode_id: int,
    episode_seed: int,
    stage_index: int,
    split: str,
    position_index: int,
) -> dict[str, Any]:
    """Capture every input needed by the value adapter, but no opaque JAX state."""
    word, strands = _full_braid(state, game)
    return {
        "observation": np.asarray(observation, dtype=np.float32).copy(),
        "word": list(word),
        "strands": strands,
        "head": _serial_head(state) % max(len(word), 1),
        "player": int(player),
        "root_value": float(root_value),
        "episode_id": int(episode_id),
        "episode_seed": int(episode_seed),
        "stage_index": int(stage_index),
        "split": str(split),
        "position_index": int(position_index),
    }


def attach_terminal_outcome(
    rows: list[dict[str, Any]],
    rewards: np.ndarray,
    *,
    solved: bool,
) -> None:
    """Label a played trajectory from each recorded side-to-move perspective."""
    values = np.asarray(rewards, dtype=np.float32)
    for row in rows:
        row["target"] = float(values[row["player"]])
        row["solved"] = bool(solved)


def tensor_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert variable-length semantic rows to a compact, portable payload."""
    if not rows:
        raise ValueError("on-policy dataset contains no positions")
    required = {"target", "solved"}
    if any(not required <= row.keys() for row in rows):
        raise ValueError("all on-policy rows must have terminal labels")
    return {
        "observation": _observation_batch(
            [row["observation"] for row in rows], torch.device("cpu")
        ),
        "words": [row["word"] for row in rows],
        "strands": torch.tensor([row["strands"] for row in rows], dtype=torch.long),
        "heads": torch.tensor([row["head"] for row in rows], dtype=torch.long),
        "players": torch.tensor([row["player"] for row in rows], dtype=torch.long),
        "root_values": torch.tensor([row["root_value"] for row in rows]),
        "targets": torch.tensor([row["target"] for row in rows]),
        "solved": torch.tensor([row["solved"] for row in rows], dtype=torch.bool),
        "episode_ids": torch.tensor([row["episode_id"] for row in rows], dtype=torch.long),
        "episode_seeds": torch.tensor([row["episode_seed"] for row in rows], dtype=torch.long),
        "stage_indexes": torch.tensor([row["stage_index"] for row in rows], dtype=torch.long),
        "splits": [row["split"] for row in rows],
        "position_indexes": torch.tensor([row["position_index"] for row in rows], dtype=torch.long),
    }


def split_mask(payload: dict[str, Any], split: str) -> Tensor:
    return torch.tensor([value == split for value in payload["splits"]], dtype=torch.bool)
