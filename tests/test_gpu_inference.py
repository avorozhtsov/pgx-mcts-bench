from __future__ import annotations

import threading
import time

import pytest
import torch
from torch import nn

from pgx_mcts_bench.gpu_inference import (
    CoordinatedPolicyValueNet,
    InferenceMetadata,
    PersistentInferenceCoordinator,
)
from pgx_mcts_bench.networks import PolicyValueNet


class _CountingNetwork(PolicyValueNet):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(6, 4)
        self.calls: list[int] = []

    def forward(self, observation: torch.Tensor):
        self.calls.append(observation.shape[0])
        flat = observation.flatten(1)
        return self.linear(flat), flat.mean(dim=1)

    def forward_with_auxiliary(self, observation: torch.Tensor):
        policy, value = self(observation)
        flat = observation.flatten(1)
        members = torch.stack([flat.mean(dim=1), flat.amax(dim=1)], dim=1)
        return policy, value, (members, members.abs(), members.square())


def test_coordinator_batches_concurrent_requests_and_preserves_outputs() -> None:
    torch.manual_seed(3)
    network = _CountingNetwork().eval()
    observations = [torch.randn(1, 1, 2, 3) for _ in range(4)]
    expected = [network(observation) for observation in observations]
    network.calls.clear()
    with PersistentInferenceCoordinator(
        network, device="cpu", max_batch_rows=8, max_wait_ms=25
    ) as coordinator:
        barrier = threading.Barrier(len(observations))
        output: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(observations)

        def run(index: int) -> None:
            barrier.wait()
            result = coordinator.infer(observations[index])
            output[index] = result  # type: ignore[assignment]

        threads = [threading.Thread(target=run, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert network.calls == [4]
        assert coordinator.snapshot()["max_batch_rows"] == 4
    for observed, wanted in zip(output, expected, strict=True):
        assert observed is not None
        torch.testing.assert_close(observed[0], wanted[0])
        torch.testing.assert_close(observed[1], wanted[1])


def test_coordinator_keeps_registered_buckets_separate() -> None:
    network = _CountingNetwork().eval()
    observation = torch.randn(1, 1, 2, 3)
    with PersistentInferenceCoordinator(
        network, device="cpu", max_batch_rows=8, max_wait_ms=15
    ) as coordinator:
        first = coordinator.submit(
            observation,
            metadata=InferenceMetadata(strand_count=6, dose=32),
        )
        second = coordinator.submit(
            observation,
            metadata=InferenceMetadata(strand_count=7, dose=32),
        )
        first.result(timeout=1)
        second.result(timeout=1)
    assert network.calls == [1, 1]


def test_auxiliary_proxy_and_explicit_deadline() -> None:
    network = _CountingNetwork().eval()
    observation = torch.randn(2, 1, 2, 3)
    with PersistentInferenceCoordinator(
        network, device="cpu", max_batch_rows=8, max_wait_ms=0
    ) as coordinator:
        proxy = CoordinatedPolicyValueNet(coordinator, strand_count=5, dose=64)
        policy, value, auxiliary = proxy.forward_with_auxiliary(observation)
        assert policy.shape == (2, 4)
        assert value.shape == (2,)
        assert auxiliary is not None and auxiliary[0].shape == (2, 2)
        expired = coordinator.submit(
            observation[:1],
            metadata=InferenceMetadata(deadline_monotonic=time.monotonic() - 1.0),
        )
        with pytest.raises(TimeoutError):
            expired.result(timeout=1)
