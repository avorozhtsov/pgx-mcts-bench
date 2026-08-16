"""Persistent, dynamically batched inference for CPU MCTS actors.

The coordinator owns the accelerator-facing network for its whole lifetime.
Actors submit already constructed observations and block on their own future;
the worker continuously refills compatible batches.  Training continues to use
the original network directly and must not run concurrently with this service.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import Tensor

from pgx_mcts_bench.networks import PolicyValueNet

InferenceMode = Literal["policy-value", "auxiliary"]
PolicyValueOutput: TypeAlias = tuple[Tensor, Tensor]
AuxiliaryOutput: TypeAlias = tuple[
    Tensor, Tensor, tuple[Tensor, Tensor, Tensor] | None
]
InferenceOutput: TypeAlias = PolicyValueOutput | AuxiliaryOutput


@dataclass(frozen=True)
class InferenceMetadata:
    """Deterministic batching hints; they never alter model semantics."""

    lane_id: int = -1
    strand_count: int = -1
    dose: int = -1
    deadline_monotonic: float | None = None


@dataclass
class _Request:
    sequence: int
    observation: Tensor
    mode: InferenceMode
    metadata: InferenceMetadata
    submitted: float
    future: Future[InferenceOutput]

    @property
    def rows(self) -> int:
        return int(self.observation.shape[0])

    @property
    def bucket(self) -> tuple[object, ...]:
        # Word length, channel schema, strand count, and dose are the registered
        # bucketing dimensions. Dtype is included to prevent implicit casts.
        return (
            self.mode,
            tuple(self.observation.shape[1:]),
            self.observation.dtype,
            self.metadata.strand_count,
            self.metadata.dose,
        )


class PersistentInferenceCoordinator:
    """One long-lived device worker with deterministic FIFO dynamic batching."""

    def __init__(
        self,
        network: PolicyValueNet,
        *,
        device: str | torch.device,
        max_batch_rows: int = 128,
        max_wait_ms: float = 2.0,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if max_batch_rows < 1:
            raise ValueError("max_batch_rows must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self.network = network
        self.device = torch.device(device)
        self.max_batch_rows = int(max_batch_rows)
        self.max_wait_seconds = float(max_wait_ms) / 1000.0
        self.amp_dtype = amp_dtype
        self._condition = threading.Condition()
        self._pending: list[_Request] = []
        self._closed = False
        self._sequence = 0
        self._worker_error: BaseException | None = None
        self.stats = {
            "requests": 0,
            "rows": 0,
            "batches": 0,
            "max_batch_rows": 0,
            "expired": 0,
            "failures": 0,
        }
        self.network.to(self.device).eval()
        self._thread = threading.Thread(
            target=self._worker,
            name="persistent-inference-coordinator",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        observation: Tensor,
        *,
        mode: InferenceMode = "policy-value",
        metadata: InferenceMetadata | None = None,
    ) -> Future[InferenceOutput]:
        if observation.ndim != 4 or observation.shape[0] < 1:
            raise ValueError("observation must have shape (batch, channels, height, width)")
        if mode not in ("policy-value", "auxiliary"):
            raise ValueError(f"unknown inference mode {mode!r}")
        metadata = metadata or InferenceMetadata()
        future: Future[InferenceOutput] = Future()
        with self._condition:
            if self._worker_error is not None:
                raise RuntimeError("inference worker failed") from self._worker_error
            if self._closed:
                raise RuntimeError("inference coordinator is closed")
            request = _Request(
                sequence=self._sequence,
                observation=observation.detach(),
                mode=mode,
                metadata=metadata,
                submitted=time.monotonic(),
                future=future,
            )
            self._sequence += 1
            self._pending.append(request)
            self.stats["requests"] += 1
            self.stats["rows"] += request.rows
            self._condition.notify()
        return future

    def infer(
        self,
        observation: Tensor,
        *,
        mode: InferenceMode = "policy-value",
        metadata: InferenceMetadata | None = None,
        timeout: float | None = None,
    ) -> InferenceOutput:
        return self.submit(observation, mode=mode, metadata=metadata).result(timeout=timeout)

    def close(self, *, wait: bool = True) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if wait and threading.current_thread() is not self._thread:
            self._thread.join()

    def __enter__(self) -> PersistentInferenceCoordinator:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def _expire(self, now: float) -> None:
        live = []
        for request in self._pending:
            deadline = request.metadata.deadline_monotonic
            if deadline is not None and deadline <= now:
                request.future.set_exception(TimeoutError("inference request deadline expired"))
                self.stats["expired"] += 1
            elif request.future.cancelled():
                continue
            else:
                live.append(request)
        self._pending = live

    def _take_batch(self) -> list[_Request]:
        first = self._pending[0]
        selected = []
        rows = 0
        remaining = []
        for request in self._pending:
            if request.bucket == first.bucket and rows + request.rows <= self.max_batch_rows:
                selected.append(request)
                rows += request.rows
            else:
                remaining.append(request)
        self._pending = remaining
        return selected

    def _worker(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._closed:
                        self._condition.wait()
                    self._expire(time.monotonic())
                    if not self._pending:
                        if self._closed:
                            return
                        continue
                    first = self._pending[0]
                    compatible_rows = sum(
                        request.rows
                        for request in self._pending
                        if request.bucket == first.bucket
                    )
                    age = time.monotonic() - first.submitted
                    if compatible_rows < self.max_batch_rows and age < self.max_wait_seconds:
                        self._condition.wait(timeout=self.max_wait_seconds - age)
                        continue
                    requests = self._take_batch()
                self._execute(requests)
        except BaseException as error:
            with self._condition:
                self._worker_error = error
                pending, self._pending = self._pending, []
                self.stats["failures"] += 1
            for request in pending:
                if not request.future.done():
                    request.future.set_exception(error)

    @staticmethod
    def _split_tensor(tensor: Tensor, sizes: list[int]) -> tuple[Tensor, ...]:
        return tuple(part.detach() for part in torch.split(tensor, sizes, dim=0))

    def _execute(self, requests: list[_Request]) -> None:
        if not requests:
            return
        sizes = [request.rows for request in requests]
        observation = torch.cat(
            [request.observation.to(self.device, non_blocking=True) for request in requests],
            dim=0,
        )
        use_amp = self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=use_amp,
        ):
            if requests[0].mode == "auxiliary":
                output = self.network.forward_with_auxiliary(observation)
            else:
                output = self.network(observation)
        policy_parts = self._split_tensor(output[0], sizes)
        value_parts = self._split_tensor(output[1], sizes)
        auxiliary_parts: tuple[
            tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]
        ] | None = None
        if len(output) == 3 and output[2] is not None:
            auxiliary_parts = (
                self._split_tensor(output[2][0], sizes),
                self._split_tensor(output[2][1], sizes),
                self._split_tensor(output[2][2], sizes),
            )
        for index, request in enumerate(requests):
            if request.future.cancelled():
                continue
            if requests[0].mode == "auxiliary":
                auxiliary = (
                    None
                    if auxiliary_parts is None
                    else tuple(parts[index] for parts in auxiliary_parts)
                )
                request.future.set_result(
                    (policy_parts[index], value_parts[index], auxiliary)  # type: ignore[arg-type]
                )
            else:
                request.future.set_result((policy_parts[index], value_parts[index]))
        rows = sum(sizes)
        self.stats["batches"] += 1
        self.stats["max_batch_rows"] = max(self.stats["max_batch_rows"], rows)

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            pending_by_bucket: dict[str, int] = defaultdict(int)
            for request in self._pending:
                pending_by_bucket[str(request.bucket)] += request.rows
            return {
                **self.stats,
                "device": str(self.device),
                "amp_dtype": str(self.amp_dtype),
                "pending_rows_by_bucket": dict(pending_by_bucket),
                "closed": self._closed,
                "worker_alive": self._thread.is_alive(),
            }


class CoordinatedPolicyValueNet(PolicyValueNet):
    """MCTS-compatible blocking client for a persistent coordinator."""

    def __init__(
        self,
        coordinator: PersistentInferenceCoordinator,
        *,
        strand_count: int = -1,
        dose: int = -1,
        timeout: float | None = None,
    ) -> None:
        super().__init__()
        # Registering the underlying model keeps NeuralMCTS checkpoint-version
        # invalidation exact while all forward calls still route to the worker.
        self.network = coordinator.network
        self.coordinator = coordinator
        self.strand_count = strand_count
        self.dose = dose
        self.timeout = timeout

    def _metadata(self) -> InferenceMetadata:
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        return InferenceMetadata(
            strand_count=self.strand_count,
            dose=self.dose,
            deadline_monotonic=deadline,
        )

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        output = self.coordinator.infer(
            observation,
            metadata=self._metadata(),
            timeout=self.timeout,
        )
        return output  # type: ignore[return-value]

    def forward_with_auxiliary(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor] | None]:
        output = self.coordinator.infer(
            observation,
            mode="auxiliary",
            metadata=self._metadata(),
            timeout=self.timeout,
        )
        return output  # type: ignore[return-value]
