"""Frozen full-braid embedding as a zero-residual value-only MCTS adapter."""

from __future__ import annotations

import hashlib
import math
import struct
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from pgx_mcts_bench.networks import PolicyValueNet


def _full_braid(state: Any, game: Any) -> tuple[tuple[int, ...], int]:
    raw = game.unwrap(state)
    word = tuple(int(letter) for letter in np.asarray(raw._word) if int(letter))
    return word, int(np.asarray(raw._n))


def _serial_head(state: Any) -> int:
    current = state
    for _ in range(8):
        if hasattr(current, "head"):
            return int(current.head)
        if hasattr(current, "base_state"):
            current = current.base_state
            continue
        break
    return 0


def head_position_features(head: int, length: int, *, device: torch.device) -> Tensor:
    """Match the frozen encoder's absolute cyclic positional basis at the head."""
    denominator = max(length, 1)
    position = float(head % denominator)
    angle = 2.0 * math.pi * position / denominator
    return torch.tensor(
        [
            math.sin(angle),
            math.cos(angle),
            math.sin(2.0 * angle),
            math.cos(2.0 * angle),
            math.sin(4.0 * angle),
            math.cos(4.0 * angle),
            position / denominator,
            1.0,
        ],
        dtype=torch.float32,
        device=device,
    )


class FrozenEmbeddingValueAdapter(PolicyValueNet):
    """Add a small global correction to value while preserving policy exactly.

    The expensive 64D knot embedding is cached by ``(word, strands)``. Moving
    the serial head therefore computes only eight cheap position features; a
    semantic edit invalidates the cache naturally because it changes the word.
    The final residual layer is all zero, so attaching this adapter changes
    neither policy logits nor values before it is trained.
    """

    def __init__(
        self,
        parent: PolicyValueNet,
        embedding_model: nn.Module,
        *,
        observation_channels: int,
        embedding_dim: int = 64,
        width: int = 32,
        maximum_cache_entries: int = 65_536,
        maximum_value_residual: float = 0.5,
        embedding_head: str = "knot",
        use_embedding: bool = True,
    ):
        super().__init__()
        if width < 8:
            raise ValueError("adapter width must be at least 8")
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        if maximum_cache_entries < 1:
            raise ValueError("maximum_cache_entries must be positive")
        if not 0.0 < maximum_value_residual <= 2.0:
            raise ValueError("maximum_value_residual must be in (0, 2]")
        self.parent = parent
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.embedding_head = embedding_head
        self.use_embedding = bool(use_embedding)
        self.maximum_cache_entries = maximum_cache_entries
        self.maximum_value_residual = maximum_value_residual
        self.local = nn.Sequential(nn.Linear(3 * observation_channels + 1, width), nn.SiLU())
        self.global_context = nn.Sequential(
            nn.Linear(embedding_dim + 8, width),
            nn.SiLU(),
        )
        self.residual = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self._embedding_cache: OrderedDict[tuple[tuple[int, ...], int], Tensor] = OrderedDict()
        self.embedding_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.parent.requires_grad_(False)
        self.embedding_model.requires_grad_(False)

    def train(self, mode: bool = True) -> FrozenEmbeddingValueAdapter:
        super().train(mode)
        # Frozen modules must keep dropout and normalization deterministic even
        # while the small residual branch is being fitted.
        self.parent.eval()
        self.embedding_model.eval()
        return self

    @staticmethod
    def _semantic_key(state: Any, game: Any) -> tuple[tuple[int, ...], int]:
        return _full_braid(state, game)

    def inference_context_key(self, state: Any, game: Any) -> bytes:
        word, strands = self._semantic_key(state, game)
        head = _serial_head(state) % max(len(word), 1)
        payload = struct.pack("<III", strands, len(word), head)
        if word:
            payload += struct.pack(f"<{len(word)}i", *word)
        return hashlib.blake2b(payload, digest_size=16).digest()

    def _encode_missing(
        self,
        keys: list[tuple[tuple[int, ...], int]],
    ) -> None:
        if not keys:
            return
        model_device = next(self.embedding_model.parameters()).device
        maximum_length = max(max(len(word), 1) for word, _ in keys)
        words = torch.zeros((len(keys), maximum_length), dtype=torch.long, device=model_device)
        lengths = torch.tensor([len(word) for word, _ in keys], device=model_device)
        strands = torch.tensor([strands for _, strands in keys], device=model_device)
        for row, (word, _) in enumerate(keys):
            if word:
                words[row, : len(word)] = torch.tensor(word, dtype=torch.long, device=model_device)
        with torch.inference_mode():
            encoded = self.embedding_model(words, lengths, strands)
        if isinstance(encoded, dict):
            encoded = encoded[self.embedding_head]
        if encoded.shape != (len(keys), self.embedding_dim):
            raise ValueError(
                f"embedding model returned {tuple(encoded.shape)}, expected "
                f"({len(keys)}, {self.embedding_dim})"
            )
        for key, row in zip(keys, encoded.detach().cpu(), strict=True):
            self._embedding_cache[key] = row.clone()
            if len(self._embedding_cache) > self.maximum_cache_entries:
                self._embedding_cache.popitem(last=False)
                self.embedding_cache_stats["evictions"] += 1

    def state_context(self, states: list[Any], game: Any, *, device: torch.device) -> Tensor:
        semantic_keys = [self._semantic_key(state, game) for state in states]
        if not self.use_embedding:
            rows = []
            for state, key in zip(states, semantic_keys, strict=True):
                word, _ = key
                embedding = torch.zeros(self.embedding_dim, device=device)
                position = head_position_features(_serial_head(state), len(word), device=device)
                rows.append(torch.cat((embedding, position)))
            return torch.stack(rows)
        missing = []
        seen = set()
        for key in semantic_keys:
            if key in self._embedding_cache:
                self._embedding_cache.move_to_end(key)
                self.embedding_cache_stats["hits"] += 1
            elif key in seen:
                self.embedding_cache_stats["hits"] += 1
            else:
                seen.add(key)
                missing.append(key)
                self.embedding_cache_stats["misses"] += 1
        self._encode_missing(missing)
        rows = []
        for state, key in zip(states, semantic_keys, strict=True):
            word, _ = key
            embedding = self._embedding_cache[key].to(device=device)
            position = head_position_features(_serial_head(state), len(word), device=device)
            rows.append(torch.cat((embedding, position)))
        return torch.stack(rows)

    def value_residual(self, observation: Tensor, base_value: Tensor, context: Tensor) -> Tensor:
        mean = observation.mean(dim=(2, 3))
        maximum = observation.amax(dim=(2, 3))
        centre = observation[:, :, observation.shape[2] // 2, observation.shape[3] // 2]
        local = self.local(torch.cat((mean, maximum, centre, base_value[:, None]), dim=1))
        global_context = self.global_context(context)
        return self.maximum_value_residual * torch.tanh(
            self.residual(torch.cat((local, global_context), dim=1)).squeeze(-1)
        )

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Observation-only callers retain the frozen parent's exact behavior."""
        return self.parent(observation)

    def forward_with_states(
        self,
        observation: Tensor,
        states: list[Any],
        game: Any,
    ) -> tuple[Tensor, Tensor]:
        policy, base_value = self.parent(observation)
        context = self.state_context(states, game, device=observation.device)
        corrected = torch.clamp(
            base_value + self.value_residual(observation, base_value, context),
            -1.0,
            1.0,
        )
        return policy, corrected
