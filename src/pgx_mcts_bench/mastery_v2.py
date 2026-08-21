"""Operational safeguards and scientific accounting for mastery program v2.

This module is deliberately additive.  It contains no service management and
never mutates a v1 run: callers opt into the v2 protocol and persist every
decision beside the copied v1 checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rf_knots.evidence import BraidState, UnknotWitness
from rf_knots.lower_bounds import computed_claims, tabulated_claims


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class AttemptDeadlineExceeded(TimeoutError):
    """Raised when one MCTS call exceeds its durable v2 wall-clock budget."""


@contextmanager
def hard_deadline(seconds: float | None) -> Iterator[None]:
    """Interrupt a stuck CPU MCTS call on POSIX while preserving resumability.

    The mastery services call this from their main thread on Linux.  Tests and
    non-POSIX callers fall back to no signal rather than pretending that a
    thread timer can interrupt native PyTorch work.
    """

    if (
        seconds is None
        or seconds <= 0
        or not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def expired(_signum: int, _frame: Any) -> None:
        raise AttemptDeadlineExceeded(f"MCTS attempt exceeded {seconds:.3f}s")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def outcome_class(
    *, solved: bool, reason: str, attempt_target_u: int | None, scientific_target_u: int
) -> str:
    """Return a mutually exclusive, denominator-safe search outcome class."""

    if solved:
        return (
            "strict_challenge_success"
            if attempt_target_u == scientific_target_u
            else "relaxed_training_success"
        )
    normalized = reason.lower().replace("-", "_")
    if "timeout" in normalized:
        return "hard_timeout"
    if "capacity" in normalized or "unsupported" in normalized:
        return "unsupported_capacity"
    if "invalid" in normalized or "verify" in normalized:
        return "invalid_witness"
    return "supported_search_failure"


def admissible_negative(reason: str) -> bool:
    """Only a completed, supported search can become a negative solve label."""

    normalized = reason.lower().replace("-", "_")
    excluded = ("timeout", "capacity", "unsupported", "invalid", "verify", "error")
    return not any(token in normalized for token in excluded)


@dataclass
class DoseCalibrationTable:
    """Small beta-binomial calibrator for P(solver succeeds | p-head, dose)."""

    bins: int = 10
    prior_strength: float = 4.0
    observations: dict[str, list[list[int]]] = field(default_factory=dict)
    wall_seconds: dict[str, list[float]] = field(default_factory=dict)

    def _rows(self, dose: int) -> list[list[int]]:
        key = str(int(dose))
        if key not in self.observations:
            self.observations[key] = [[0, 0] for _ in range(self.bins)]
        return self.observations[key]

    def sample_count(self, dose: int, raw_probability: float) -> int:
        index = min(int(max(raw_probability, 0.0) * self.bins), self.bins - 1)
        solved, total = self._rows(dose)[index]
        del solved
        return int(total)

    def calibrate(self, dose: int, raw_probability: float) -> tuple[float, int, float]:
        raw = min(max(float(raw_probability), 0.0), 1.0)
        index = min(int(raw * self.bins), self.bins - 1)
        solved, total = self._rows(dose)[index]
        calibrated = (solved + self.prior_strength * raw) / (total + self.prior_strength)
        uncertainty = math.sqrt(calibrated * (1.0 - calibrated) / (total + 3.0))
        return float(calibrated), int(total), float(uncertainty)

    def observe(self, dose: int, raw_probability: float, solved: bool, wall_seconds: float) -> None:
        raw = min(max(float(raw_probability), 0.0), 1.0)
        index = min(int(raw * self.bins), self.bins - 1)
        row = self._rows(dose)[index]
        row[0] += int(bool(solved))
        row[1] += 1
        if wall_seconds > 0:
            bucket = self.wall_seconds.setdefault(str(int(dose)), [])
            bucket.append(float(wall_seconds))
            del bucket[:-256]

    def expected_wall_seconds(self, dose: int) -> float | None:
        values = self.wall_seconds.get(str(int(dose)), [])
        return sum(values) / len(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bins": self.bins,
            "prior_strength": self.prior_strength,
            "observations": self.observations,
            "wall_seconds": self.wall_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> DoseCalibrationTable:
        payload = payload or {}
        return cls(
            bins=int(payload.get("bins", 10)),
            prior_strength=float(payload.get("prior_strength", 4.0)),
            observations={
                str(key): [[int(a), int(b)] for a, b in rows]
                for key, rows in payload.get("observations", {}).items()
            },
            wall_seconds={
                str(key): [float(value) for value in values]
                for key, values in payload.get("wall_seconds", {}).items()
            },
        )


@dataclass(frozen=True)
class EvidenceSnapshot:
    path: Path
    sha256: str
    schema: str
    best_upper_by_knot: dict[str, int]
    replayable_rows: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, path: Path) -> EvidenceSnapshot:
        payload = json.loads(path.read_text())
        verified = payload.get("verified") or {}
        representation_pool = verified.get("best_by_representation", [])
        if isinstance(representation_pool, dict):
            representation_pool = representation_pool.values()
        rows = tuple(dict(row) for row in representation_pool)
        best_by_knot = verified.get("best_by_knot", {})
        if isinstance(best_by_knot, dict):
            values = {
                str(name): int(row["crossing_changes"] if isinstance(row, dict) else row)
                for name, row in best_by_knot.items()
            }
        else:
            values = {
                str(row["knot_name"]): int(row["crossing_changes"])
                for row in best_by_knot
                if row.get("knot_name") is not None
            }
        return cls(path, sha256_file(path), str(payload.get("schema", "unknown")), values, rows)

    def best_upper(self, knot_name: str, default: int) -> int:
        return min(int(default), self.best_upper_by_knot.get(knot_name, int(default)))

    def manifest(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "schema": self.schema,
            "replayable_representations": len(self.replayable_rows),
            "mapped_knots": len(self.best_upper_by_knot),
        }


class CertificationEngine:
    """Write an independent certificate after a replay-verified improvement."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def certify(
        self,
        *,
        evidence_id: str,
        knot_name: str,
        root: BraidState,
        witness: UnknotWitness,
        declared_lower_bound: int,
    ) -> dict[str, Any]:
        witness.verify()
        computed = computed_claims(root.word, root.strands)
        tabulated = tabulated_claims(knot_name)
        claims = [*computed, *tabulated]
        strongest = max((int(claim.value) for claim in claims), default=None)
        exact = strongest is not None and strongest >= witness.crossing_changes
        report = {
            "schema": "mastery-improvement-certificate-v1",
            "evidence_id": evidence_id,
            "knot_name": knot_name,
            "instance_id": witness.instance_id,
            "upper_bound": witness.crossing_changes,
            "moves": witness.moves,
            "l1000": 1000 * witness.crossing_changes + witness.moves,
            "witness_replay_verified": True,
            "declared_lower_bound": int(declared_lower_bound),
            "independent_lower_bound": strongest,
            "status": "exact-certified" if exact else "upper-bound-certified",
            "lower_bound_claims": [claim.to_dict() for claim in claims],
        }
        atomic_json(self.root / f"{evidence_id}.json", report)
        return report


def parameter_capacity_report(
    checkpoints: Sequence[tuple[str, Path]], retention: dict[str, float | None] | None = None
) -> dict[str, Any]:
    """Count checkpoint tensors; parameter count is diagnostic, never a proof."""

    import torch

    rows = []
    for name, checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("network", payload)
        parameters = sum(int(value.numel()) for value in state.values() if hasattr(value, "numel"))
        rows.append(
            {
                "scientist": name,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "network_parameters": parameters,
                "state_tensors": sum(hasattr(value, "numel") for value in state.values()),
                "observed_retention": (retention or {}).get(name),
            }
        )
    largest = max((row["network_parameters"] for row in rows), default=0)
    for row in rows:
        row["fraction_of_largest"] = row["network_parameters"] / largest if largest else None
        relatively_small = (
            row["fraction_of_largest"] is not None and row["fraction_of_largest"] < 0.5
        )
        retention_failed = (
            row["observed_retention"] is not None and row["observed_retention"] < 0.80
        )
        if relatively_small and retention_failed:
            interpretation = "capacity-risk-needs-controlled-width-ablation"
        elif relatively_small:
            interpretation = "smaller-model-but-observed-retention-is-healthy"
        else:
            interpretation = "parameter-count-alone-does-not-show-capacity-failure"
        row["interpretation"] = interpretation
    return {
        "schema": "mastery-network-capacity-audit-v1",
        "warning": (
            "Parameter count cannot establish representational sufficiency; combine this audit "
            "with frozen-panel train/eval retention and a controlled wider-model ablation."
        ),
        "scientists": rows,
    }
