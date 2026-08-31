from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pgx_mcts_bench.on_policy_embedding_value import (
    attach_terminal_outcome,
    split_mask,
    state_row,
    tensor_payload,
)


class FakeGame:
    @staticmethod
    def unwrap(state):
        return state.pgx


def test_on_policy_rows_preserve_exact_word_head_and_perspective_target() -> None:
    pgx = SimpleNamespace(_word=np.asarray([1, -2, 0, 0]), _n=np.asarray(3))
    state = SimpleNamespace(pgx=pgx, head=5)
    row = state_row(
        state,
        np.zeros((2, 3, 4), dtype=np.float32),
        game=FakeGame(),
        player=1,
        root_value=0.25,
        episode_id=7,
        episode_seed=11,
        stage_index=2,
        split="validation",
        position_index=0,
    )
    attach_terminal_outcome([row], np.asarray([-0.75, 0.75]), solved=True)
    payload = tensor_payload([row])

    assert payload["words"] == [[1, -2]]
    assert payload["heads"].tolist() == [1]
    assert payload["targets"].tolist() == [0.75]
    assert payload["observation"].shape == (1, 4, 2, 3)
    assert torch.equal(split_mask(payload, "validation"), torch.tensor([True]))


def test_tensor_payload_rejects_unlabelled_rows() -> None:
    row = {
        "observation": np.zeros((1, 1, 1), dtype=np.float32),
        "word": [],
        "strands": 1,
        "head": 0,
        "player": 0,
        "root_value": 0.0,
        "episode_id": 0,
        "episode_seed": 0,
        "stage_index": 0,
        "split": "train",
        "position_index": 0,
    }
    try:
        tensor_payload([row])
    except ValueError as error:
        assert "terminal labels" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unlabelled row was accepted")
