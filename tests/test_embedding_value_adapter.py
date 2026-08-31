from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from pgx_mcts_bench.embedding_value_adapter import FrozenEmbeddingValueAdapter
from pgx_mcts_bench.networks import PolicyValueNet


class _State(NamedTuple):
    pgx: object
    head: int


class _Game:
    @staticmethod
    def unwrap(state):
        return state.pgx


class _Parent(PolicyValueNet):
    def forward(self, observation):
        batch = observation.shape[0]
        return observation.new_full((batch, 5), 0.25), observation.new_full((batch,), -0.2)


class _Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, word, lengths, strands):
        self.calls += 1
        rows = torch.arange(64, device=word.device, dtype=torch.float32).repeat(len(word), 1)
        rows = rows + lengths[:, None] + strands[:, None] + self.anchor
        return {"knot": rows}


def _state(word, head=0, strands=3):
    return _State(
        SimpleNamespace(_word=np.asarray([*word, 0, 0], dtype=np.int32), _n=np.int32(strands)),
        head,
    )


def test_zero_adapter_preserves_policy_and_value_exactly() -> None:
    model = FrozenEmbeddingValueAdapter(
        _Parent(),
        _Embedding(),
        observation_channels=4,
    )
    observation = torch.randn(2, 4, 1, 7)
    states = [_state([1, -2, 1], head=0), _state([1, -2, 1], head=2)]
    base_policy, base_value = model.parent(observation)
    policy, value = model.forward_with_states(observation, states, _Game())
    assert torch.equal(policy, base_policy)
    assert torch.equal(value, base_value)


def test_head_motion_reuses_embedding_but_changes_position_context() -> None:
    embedding = _Embedding()
    model = FrozenEmbeddingValueAdapter(_Parent(), embedding, observation_channels=2)
    states = [_state([1, 2, -1], head=0), _state([1, 2, -1], head=1)]
    context = model.state_context(states, _Game(), device=torch.device("cpu"))
    assert embedding.calls == 1
    assert model.embedding_cache_stats == {"hits": 1, "misses": 1, "evictions": 0}
    assert torch.equal(context[0, :64], context[1, :64])
    assert not torch.equal(context[0, 64:], context[1, 64:])
    assert model.inference_context_key(states[0], _Game()) != model.inference_context_key(
        states[1], _Game()
    )


def test_semantic_word_change_recomputes_embedding() -> None:
    embedding = _Embedding()
    model = FrozenEmbeddingValueAdapter(_Parent(), embedding, observation_channels=2)
    model.state_context([_state([1, 2], head=0)], _Game(), device=torch.device("cpu"))
    model.state_context([_state([1, -2], head=0)], _Game(), device=torch.device("cpu"))
    assert embedding.calls == 2
    assert model.embedding_cache_stats["misses"] == 2


def test_head_only_control_never_evaluates_or_exposes_embedding() -> None:
    embedding = _Embedding()
    model = FrozenEmbeddingValueAdapter(
        _Parent(), embedding, observation_channels=2, use_embedding=False
    )
    states = [_state([1, 2, -1], head=0), _state([1, 2, -1], head=1)]
    context = model.state_context(states, _Game(), device=torch.device("cpu"))
    assert embedding.calls == 0
    assert torch.equal(context[:, :64], torch.zeros_like(context[:, :64]))
    assert not torch.equal(context[0, 64:], context[1, 64:])


def test_only_adapter_parameters_are_trainable() -> None:
    model = FrozenEmbeddingValueAdapter(_Parent(), _Embedding(), observation_channels=3)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith(("local.", "global_context.", "residual.")) for name in trainable)
