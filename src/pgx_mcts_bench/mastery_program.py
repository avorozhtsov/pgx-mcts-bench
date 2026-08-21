"""Persistent multi-knot mastery with a bounded cross-knot science heap.

Each scientist owns one independent network, optimizer, replay buffer, sequence
cursor, and 200-entry heap.  Heap entries are exact braid representations (or
one-crossing-change descendants) ranked by the operational solve head under
their remaining crossing-change and randomized L1000 budget.  Verified
witnesses are written atomically to a shared evidence inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rf_knots.evidence import BraidState, UnknotWitness, braid_instance_id

from pgx_mcts_bench.adaptive_scientists import load_scientist
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.mastery_v2 import (
    AttemptDeadlineExceeded,
    CertificationEngine,
    DoseCalibrationTable,
    EvidenceSnapshot,
    admissible_negative,
    atomic_json,
    hard_deadline,
    outcome_class,
)
from pgx_mcts_bench.single_knot_mastery import (
    AdaptiveOutcomeController,
    AttemptResult,
    DistillationExample,
    FairRefreshScheduler,
    MasteryConfig,
    MutableProbabilityHeap,
    RepresentationNode,
    ScientistMasteryBackend,
    SimulationDoseController,
    equivalent_representations,
    load_distillation,
    one_crossing_change_children,
)

SCHEMA_V1 = "multi-knot-mastery-program-v1"
SCHEMA = "multi-knot-mastery-program-v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_id(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True)
class ProgramConfig:
    sequence_name: str
    scientist: str
    max_heap: int = 200
    group_size: int = 20
    bootstrap_challenges: int = 4
    max_active_challenges: int = 16
    admit_interval_steps: int = 2
    parallel_searches: int = 2
    attempts_per_visit: int = 1
    task_attempt_limit: int = 6
    challenge_attempt_limit: int = 24
    challenge_search_seconds_limit: float = 900.0
    cooldown_steps: int = 2
    min_child_probability: float = 0.0
    expansion_diagrams: int = 8
    expansion_children: int = 32
    expansion_node_budget: int = 250
    refresh_top: int = 12
    refresh_fair: int = 4
    negative_confirmations: int = 3
    target_positive_fraction: float = 0.50
    positive_fraction_tolerance: float = 0.08
    outcome_window: int = 64
    outcome_warmup: int = 8
    strict_search_fraction: float = 0.50
    max_training_budget_slack: int = 3
    min_rehearsal_fraction: float = 0.05
    initial_rehearsal_fraction: float = 0.10
    max_rehearsal_fraction: float = 0.50
    rehearsal_fraction_step: float = 0.05
    retention_target: float = 0.80
    retention_probe_interval: int = 20
    improvement_priority_bonus: float = 0.10
    reservoir_probe_interval: int = 10
    reservoir_probe_batch: int = 16
    reservoir_injections_per_group: int = 1
    seed: int = 20260815
    protocol_version: int = 2
    attempt_wall_seconds_limit: float = 900.0
    simulation_levels: tuple[int, ...] = ()
    simulation_probe_interval: int = 20
    simulation_probe_lanes: int = 2
    simulation_probe_min_pairs: int = 12
    simulation_success_margin: float = 0.05
    simulation_l1000_tolerance: float = 5.0
    heap_uncertainty_bonus: float = 0.10
    heap_age_bonus: float = 0.02
    heap_cost_penalty: float = 0.01
    max_snapshot_distillation_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.protocol_version not in (1, 2):
            raise ValueError("protocol_version must be 1 or 2")
        if self.max_heap < self.parallel_searches:
            raise ValueError("max_heap must cover every search lane")
        if self.group_size < 1 or self.bootstrap_challenges < 1:
            raise ValueError("group and bootstrap sizes must be positive")
        if self.max_active_challenges < self.bootstrap_challenges:
            raise ValueError("max_active_challenges must cover bootstrap challenges")
        if self.challenge_attempt_limit < self.task_attempt_limit:
            raise ValueError("challenge limit must cover one task visit")
        if not (
            0.0
            <= self.min_rehearsal_fraction
            <= self.initial_rehearsal_fraction
            <= self.max_rehearsal_fraction
            <= 0.50
        ):
            raise ValueError("rehearsal fractions must satisfy 0 <= min <= initial <= max <= 0.5")
        levels = tuple(int(value) for value in self.simulation_levels)
        if levels and (levels != tuple(sorted(set(levels))) or min(levels) < 1):
            raise ValueError("simulation_levels must be sorted unique positive integers")
        if self.attempt_wall_seconds_limit <= 0:
            raise ValueError("attempt wall-clock limit must be positive")
        if not 0.0 <= self.max_snapshot_distillation_fraction < 0.5:
            raise ValueError("snapshot distillation fraction must be in [0, 0.5)")


@dataclass(frozen=True)
class SequenceChallenge:
    sequence_index: int
    challenge_id: str
    knot_name: str
    representation_id: str
    root: BraidState
    certified_lower_bound: int
    declared_upper_bound: int
    target_u: int
    acs10: int
    provenance: dict[str, Any]


@dataclass
class ChallengeRuntime:
    challenge_id: str
    knot_name: str
    representation_id: str
    sequence_index: int
    root_id: str
    certified_lower_bound: int
    initial_upper_bound: int
    current_upper_bound: int
    target_u: int
    status: str = "active"
    attempts: int = 0
    search_seconds: float = 0.0
    best_evidence_id: str | None = None
    best_l1000: int | None = None
    admitted_step: int = 0
    completed_step: int | None = None


def _load_challenges(
    path: Path, *, expected_schema: str, minimum: int
) -> tuple[str, list[SequenceChallenge]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != expected_schema:
        raise ValueError(f"unsupported sequence schema in {path}")
    challenges = []
    for index, row in enumerate(payload["challenges"]):
        rep = row["stored_representation"]
        lower, upper = (int(value) for value in row["bound_interval"])
        if upper <= lower:
            raise ValueError(f"{row['canonical_name']} has no improvable bound gap")
        root = BraidState(tuple(int(value) for value in rep["word"]), int(rep["strands"]))
        challenge_id = str(
            row.get(
                "challenge_id",
                _stable_id(
                    {
                        "sequence": payload["name"],
                        "index": index,
                        "representation": row["representation_id"],
                    }
                ),
            )
        )
        challenges.append(
            SequenceChallenge(
                index,
                challenge_id,
                str(row["canonical_name"]),
                str(row["representation_id"]),
                root,
                lower,
                upper,
                upper - 1,
                int(row["acs10"]),
                dict(row.get("provenance", {})),
            )
        )
    if len(challenges) < minimum:
        raise ValueError(f"challenge list must contain at least {minimum} rows")
    if len({row.challenge_id for row in challenges}) != len(challenges):
        raise ValueError("sequence challenge ids are not unique")
    return str(payload["name"]), challenges


def load_sequence(path: Path) -> tuple[str, list[SequenceChallenge]]:
    return _load_challenges(path, expected_schema="multi-knot-mastery-sequence-v1", minimum=200)


def load_short_ablation_sequence(path: Path) -> tuple[str, list[SequenceChallenge]]:
    """Load the registered 20-task process comparison without weakening SKM-240."""

    return _load_challenges(
        path,
        expected_schema="q-skm-short-ablation-sequence-v1",
        minimum=20,
    )


def load_reservoir(path: Path) -> tuple[str, list[SequenceChallenge]]:
    return _load_challenges(path, expected_schema="multi-knot-mastery-reservoir-v1", minimum=1)


class EvidenceInventory:
    """Concurrency-safe append-only witness directory shared by scientists."""

    def __init__(self, root: Path, *, snapshot: EvidenceSnapshot | None = None) -> None:
        self.root = root
        self.snapshot = snapshot
        self.witnesses = root / "witnesses"
        self.witnesses.mkdir(parents=True, exist_ok=True)

    def best_upper(self, knot_name: str, default: int) -> int:
        best = (
            self.snapshot.best_upper(knot_name, default)
            if self.snapshot is not None
            else int(default)
        )
        for path in self.witnesses.glob("*.json"):
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if row.get("knot_name") == knot_name:
                best = min(best, int(row["crossing_changes"]))
        return best

    def record(
        self,
        *,
        witness: UnknotWitness,
        knot_name: str,
        representation_id: str,
        scientist: str,
        sequence_name: str,
        challenge_id: str,
        previous_upper_bound: int,
        solver_metadata: dict[str, Any] | None = None,
        search_parameters: dict[str, Any] | None = None,
    ) -> str:
        witness.verify()
        evidence_id = hashlib.sha256(
            json.dumps(witness.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payload = {
            "schema": "mastery-evidence-inventory-row-v2",
            "evidence_id": evidence_id,
            "knot_name": knot_name,
            "representation_id": representation_id,
            "scientist": scientist,
            "sequence_name": sequence_name,
            "challenge_id": challenge_id,
            "previous_upper_bound": int(previous_upper_bound),
            "crossing_changes": witness.crossing_changes,
            "moves": witness.moves,
            "l1000": 1000 * witness.crossing_changes + witness.moves,
            "witness": witness.to_dict(),
            "solver_metadata": dict(solver_metadata or {}),
            "search_parameters": dict(search_parameters or {}),
            "evidence_snapshot": (self.snapshot.manifest() if self.snapshot is not None else None),
        }
        path = self.witnesses / f"{evidence_id}.json"
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            existing = json.loads(path.read_text())
            if existing["witness"] != payload["witness"]:
                raise ValueError(f"evidence id collision at {path}") from None
            return evidence_id
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return evidence_id


class MasteryProgram:
    def __init__(
        self,
        config: ProgramConfig,
        backend: ScientistMasteryBackend,
        sequence: Sequence[SequenceChallenge],
        inventory: EvidenceInventory,
        *,
        sequence_sha256: str,
        reservoir: Sequence[SequenceChallenge] = (),
        reservoir_sha256: str | None = None,
        runtime_root: Path | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.sequence = tuple(sequence)
        self.inventory = inventory
        self.sequence_sha256 = sequence_sha256
        self.reservoir = tuple(reservoir)
        self.reservoir_sha256 = reservoir_sha256
        self.runtime_root = runtime_root
        self.certification = (
            CertificationEngine(runtime_root / "certificates")
            if runtime_root is not None and config.protocol_version >= 2
            else None
        )
        self.reservoir_cursor = 0
        self.reservoir_used: set[str] = set()
        self.reservoir_injections_by_group: dict[int, int] = defaultdict(int)
        self.step_index = 0
        self.next_sequence_index = 0
        self.introduced_count = 0
        self.native_train_steps = 0
        self.rehearsal_fraction = config.initial_rehearsal_fraction
        self.nodes: dict[str, RepresentationNode] = {}
        self.node_challenge: dict[str, str] = {}
        self.roots: dict[str, BraidState] = {}
        self.challenges: dict[str, ChallengeRuntime] = {}
        self.heap = MutableProbabilityHeap()
        self.refresh = FairRefreshScheduler()
        self.cooldown_until: dict[str, int] = {}
        self.negative_trials: dict[str, set[int]] = defaultdict(set)
        self.admitted_negative_keys: set[str] = set()
        self.outcome = AdaptiveOutcomeController(
            target=config.target_positive_fraction,
            tolerance=config.positive_fraction_tolerance,
            window=config.outcome_window,
            warmup=config.outcome_warmup,
            max_slack=config.max_training_budget_slack,
        )
        levels = config.simulation_levels or (int(backend.config.simulations),)
        self.dose_controller = SimulationDoseController(
            levels,
            current=int(backend.config.simulations),
            min_pairs=config.simulation_probe_min_pairs,
            success_margin=config.simulation_success_margin,
            l1000_tolerance=config.simulation_l1000_tolerance,
        )
        self.dose_calibration = DoseCalibrationTable()
        self.outcome_totals: dict[str, int] = defaultdict(int)
        self.snapshot_distilled_train_steps = 0
        self.consumed_snapshot_evidence: set[str] = set()
        self.events: list[dict[str, Any]] = []
        for _ in range(config.bootstrap_challenges):
            if not self._admit_next():
                break

    @classmethod
    def from_saved(
        cls,
        path: Path,
        backend: ScientistMasteryBackend,
        sequence: Sequence[SequenceChallenge],
        inventory: EvidenceInventory,
        *,
        sequence_sha256: str,
        reservoir: Sequence[SequenceChallenge] = (),
        reservoir_sha256: str | None = None,
        runtime_root: Path | None = None,
    ) -> MasteryProgram:
        payload = json.loads(path.read_text())
        if payload.get("schema") not in (SCHEMA_V1, SCHEMA):
            raise ValueError(f"unsupported mastery program state in {path}")
        if payload["sequence_sha256"] != sequence_sha256:
            raise ValueError("saved state belongs to a different sequence")
        if payload.get("reservoir_sha256") != reservoir_sha256:
            raise ValueError("saved state belongs to a different challenge reservoir")
        saved_snapshot = payload.get("evidence_snapshot_sha256")
        live_snapshot = inventory.snapshot.sha256 if inventory.snapshot is not None else None
        if saved_snapshot != live_snapshot:
            raise ValueError("saved state belongs to a different evidence snapshot")
        program = cls.__new__(cls)
        saved_config = dict(payload["config"])
        if payload.get("schema") == SCHEMA_V1:
            saved_config.setdefault("protocol_version", 1)
        program.config = ProgramConfig(**saved_config)
        program.backend = backend
        program.sequence = tuple(sequence)
        program.inventory = inventory
        program.sequence_sha256 = sequence_sha256
        program.reservoir = tuple(reservoir)
        program.reservoir_sha256 = reservoir_sha256
        program.runtime_root = runtime_root
        program.certification = (
            CertificationEngine(runtime_root / "certificates")
            if runtime_root is not None and program.config.protocol_version >= 2
            else None
        )
        program.reservoir_cursor = int(payload.get("reservoir_cursor", 0))
        program.reservoir_used = set(payload.get("reservoir_used", []))
        program.reservoir_injections_by_group = defaultdict(
            int,
            {
                int(key): int(value)
                for key, value in payload.get("reservoir_injections_by_group", {}).items()
            },
        )
        program.step_index = int(payload["step_index"])
        program.next_sequence_index = int(payload["next_sequence_index"])
        program.introduced_count = int(payload["introduced_count"])
        program.native_train_steps = int(payload["native_train_steps"])
        program.rehearsal_fraction = float(payload["rehearsal_fraction"])
        program.roots = {
            key: BraidState.from_dict(value) for key, value in payload["roots"].items()
        }
        program.challenges = {
            key: ChallengeRuntime(**value) for key, value in payload["challenges"].items()
        }
        program.nodes = {}
        for row in payload["nodes"]:
            row = dict(row)
            row["word"] = tuple(int(value) for value in row["word"])
            row["semantic_path"] = tuple(int(value) for value in row["semantic_path"])
            node = RepresentationNode(**row)
            program.nodes[node.node_id] = node
        program.node_challenge = dict(payload["node_challenge"])
        program.heap = MutableProbabilityHeap()
        program.heap.rebuild(program.nodes)
        program.refresh = FairRefreshScheduler()
        program.cooldown_until = {
            key: int(value) for key, value in payload.get("cooldown_until", {}).items()
        }
        program.negative_trials = defaultdict(
            set,
            {
                key: {int(seed) for seed in seeds}
                for key, seeds in payload.get("negative_trials", {}).items()
            },
        )
        program.admitted_negative_keys = set(payload.get("admitted_negative_keys", []))
        outcome = payload.get("outcome", {})
        program.outcome = AdaptiveOutcomeController(
            target=program.config.target_positive_fraction,
            tolerance=program.config.positive_fraction_tolerance,
            window=program.config.outcome_window,
            warmup=program.config.outcome_warmup,
            max_slack=program.config.max_training_budget_slack,
            slack=int(outcome.get("slack", 0)),
            outcomes=outcome.get("outcomes", ()),
        )
        levels = program.config.simulation_levels or (int(backend.config.simulations),)
        dose = payload.get("dose_controller", {})
        program.dose_controller = SimulationDoseController(
            levels,
            current=int(dose.get("current", backend.config.simulations)),
            min_pairs=program.config.simulation_probe_min_pairs,
            success_margin=program.config.simulation_success_margin,
            l1000_tolerance=program.config.simulation_l1000_tolerance,
            observations=dose.get("observations", {}),
        )
        program.dose_calibration = DoseCalibrationTable.from_dict(payload.get("dose_calibration"))
        program.outcome_totals = defaultdict(
            int,
            {str(key): int(value) for key, value in payload.get("outcome_totals", {}).items()},
        )
        program.snapshot_distilled_train_steps = int(
            payload.get("snapshot_distilled_train_steps", 0)
        )
        program.consumed_snapshot_evidence = set(payload.get("consumed_snapshot_evidence", []))
        program.events = list(payload.get("recent_events", []))
        return program

    @property
    def finished(self) -> bool:
        return self.next_sequence_index >= len(self.sequence) and not self.nodes

    def _active_challenges(self) -> int:
        return sum(row.status == "active" for row in self.challenges.values())

    def _add_node(self, node: RepresentationNode, challenge_id: str) -> bool:
        challenge = self.challenges[challenge_id]
        if node.crossing_distance > challenge.target_u:
            return False
        if any(
            current.root_id == node.root_id
            and current.word == node.word
            and current.strands == node.strands
            and current.crossing_distance == node.crossing_distance
            for current in self.nodes.values()
        ):
            return False
        self.nodes[node.node_id] = node
        self.node_challenge[node.node_id] = challenge_id
        return True

    def _admit_challenge(self, item: SequenceChallenge, *, origin: str) -> bool:
        if item.challenge_id in self.challenges:
            return False
        known_upper = self.inventory.best_upper(item.knot_name, item.declared_upper_bound)
        target = known_upper - 1
        root_id = (
            f"{item.challenge_id}:{braid_instance_id(item.root.word, item.root.strands)[6:18]}"
        )
        status = "already-at-lower-bound" if known_upper <= item.certified_lower_bound else "active"
        runtime = ChallengeRuntime(
            item.challenge_id,
            item.knot_name,
            item.representation_id,
            item.sequence_index,
            root_id,
            item.certified_lower_bound,
            item.declared_upper_bound,
            known_upper,
            target,
            status=status,
            admitted_step=self.step_index,
            completed_step=self.step_index if status != "active" else None,
        )
        self.challenges[item.challenge_id] = runtime
        self.roots[root_id] = item.root
        self.introduced_count += 1
        if status == "active":
            equivalents = equivalent_representations(
                item.root,
                self.backend.spec,
                count=4,
                node_budget=120,
                growth=2,
            )
            new_ids = []
            for state, path in equivalents:
                node = RepresentationNode.create(
                    root_id=root_id,
                    word=state.word,
                    strands=state.strands,
                    crossing_distance=0,
                    semantic_path=path,
                    provenance={
                        **item.provenance,
                        "program_origin": origin,
                        "challenge_id": item.challenge_id,
                        "knot_name": item.knot_name,
                        "representation_id": item.representation_id,
                        "original_knot_name": item.knot_name,
                        "knotinfo_lower_bound": item.certified_lower_bound,
                        "knotinfo_upper_bound": item.declared_upper_bound,
                        "original_target_u": item.target_u,
                        "current_target_u": target,
                        "remaining_target_u": target,
                        "subtask_depth": 0,
                        "subtask_lineage": [],
                        "upper_bound_improvement_challenge": (
                            item.target_u < item.declared_upper_bound
                        ),
                        "equivalent_seed": True,
                    },
                    priority_bonus=(
                        self.config.improvement_priority_bonus
                        if item.target_u < item.declared_upper_bound
                        else 0.0
                    ),
                )
                if self._add_node(node, item.challenge_id):
                    new_ids.append(node.node_id)
            self._rescore(new_ids)
        return True

    def _admit_next(self) -> bool:
        if self.next_sequence_index >= len(self.sequence):
            return False
        item = self.sequence[self.next_sequence_index]
        self.next_sequence_index += 1
        return self._admit_challenge(item, origin="predefined-sequence")

    def _rescore(self, node_ids: Sequence[str]) -> None:
        grouped: dict[int, list[RepresentationNode]] = defaultdict(list)
        for node_id in dict.fromkeys(node_ids):
            node = self.nodes.get(node_id)
            if node is None:
                continue
            challenge = self.challenges[self.node_challenge[node_id]]
            if challenge.status == "active":
                grouped[challenge.target_u].append(node)
        for target, nodes in grouped.items():
            scores = self.backend.score(nodes, target)
            for node, score in zip(nodes, scores, strict=True):
                raw_probability = float(np.clip(score.probability, 0.0, 1.0))
                dose = self.dose_controller.current
                calibrated, samples, uncertainty = self.dose_calibration.calibrate(
                    dose, raw_probability
                )
                node.raw_probability = raw_probability
                node.probability = calibrated
                node.calibration_samples = samples
                node.score_uncertainty = uncertainty
                node.expected_cpu_seconds = float(
                    self.dose_calibration.expected_wall_seconds(dose) or 0.0
                )
                node.predicted_crossings = float(score.predicted_crossings)
                node.predicted_moves = float(score.predicted_moves)
                age = max(self.step_index - node.last_scored_step, 0)
                node.scheduling_priority = float(
                    calibrated
                    + node.priority_bonus
                    + self.config.heap_uncertainty_bonus * uncertainty
                    + self.config.heap_age_bonus * min(age / max(self.config.refresh_fair, 1), 1.0)
                    - self.config.heap_cost_penalty * math.log1p(node.expected_cpu_seconds)
                )
                node.last_scored_step = self.step_index
                self.heap.update(node)

    def _maybe_inject_reservoir(self) -> dict[str, Any] | None:
        """Admit the closest harder KnotInfo-only challenge below heap rank 10.

        Raw solve probabilities define the frontier.  The scientific +0.1
        bonus is applied only after admission, so it cannot make an otherwise
        ineligible catalogue candidate pass this gate.
        """
        if (
            not self.reservoir
            or self.step_index % self.config.reservoir_probe_interval
            or len(self.nodes) < 10
        ):
            return None
        group = self.introduced_count // self.config.group_size + 1
        if self.reservoir_injections_by_group[group] >= self.config.reservoir_injections_per_group:
            return None
        p10 = sorted((node.probability for node in self.nodes.values()), reverse=True)[9]
        existing_knots = {row.knot_name for row in self.challenges.values()}
        candidates: list[tuple[SequenceChallenge, int, RepresentationNode]] = []
        inspected = 0
        while (
            inspected < len(self.reservoir) and len(candidates) < self.config.reservoir_probe_batch
        ):
            item = self.reservoir[self.reservoir_cursor % len(self.reservoir)]
            self.reservoir_cursor += 1
            inspected += 1
            if item.challenge_id in self.reservoir_used or item.knot_name in existing_knots:
                continue
            known_upper = self.inventory.best_upper(item.knot_name, item.declared_upper_bound)
            if known_upper <= item.certified_lower_bound:
                self.reservoir_used.add(item.challenge_id)
                continue
            target = known_upper - 1
            node = RepresentationNode.create(
                root_id=f"reservoir-probe:{item.challenge_id}",
                word=item.root.word,
                strands=item.root.strands,
                crossing_distance=0,
                semantic_path=(),
                provenance={
                    **item.provenance,
                    "program_origin": "knotinfo-frontier-reservoir-probe",
                    "original_knot_name": item.knot_name,
                    "knotinfo_lower_bound": item.certified_lower_bound,
                    "knotinfo_upper_bound": item.declared_upper_bound,
                    "original_target_u": item.target_u,
                    "current_target_u": target,
                    "remaining_target_u": target,
                },
            )
            candidates.append((item, target, node))
        if not candidates:
            return None
        grouped: dict[int, list[tuple[SequenceChallenge, RepresentationNode]]] = defaultdict(list)
        for item, target, node in candidates:
            grouped[target].append((item, node))
        scored: list[tuple[float, SequenceChallenge]] = []
        for target, rows in grouped.items():
            scores = self.backend.score([node for _, node in rows], target)
            for (item, _), score in zip(rows, scores, strict=True):
                scored.append((float(np.clip(score.probability, 0.0, 1.0)), item))
        eligible = [(probability, item) for probability, item in scored if probability <= p10]
        if not eligible:
            return None
        probability, item = max(eligible, key=lambda row: (row[0], -row[1].acs10))
        if not self._admit_challenge(item, origin="knotinfo-frontier-reservoir"):
            self.reservoir_used.add(item.challenge_id)
            return None
        self.reservoir_used.add(item.challenge_id)
        self.reservoir_injections_by_group[group] += 1
        return {
            "type": "LOUD_KNOTINFO_FRONTIER_INJECTION",
            "challenge_id": item.challenge_id,
            "knot_name": item.knot_name,
            "representation_id": item.representation_id,
            "target_u": self.challenges[item.challenge_id].target_u,
            "raw_p_solve": probability,
            "heap_rank_10_raw_p_solve": p10,
            "science_priority_bonus": self.config.improvement_priority_bonus,
            "science_key": probability + self.config.improvement_priority_bonus,
            "knotinfo_evidence_l10": item.provenance.get("knotinfo_evidence_l10"),
            "group": group,
        }

    def _pop_eligible(self, count: int) -> list[RepresentationNode]:
        selected = []
        deferred = []
        while len(selected) < count and self.nodes:
            try:
                node = self.heap.pop(self.nodes)
            except IndexError:
                break
            challenge = self.challenges[self.node_challenge[node.node_id]]
            if challenge.status != "active":
                self._remove_node(node.node_id)
                continue
            if self.cooldown_until.get(node.node_id, 0) > self.step_index:
                deferred.append(node)
                if len(deferred) >= len(self.nodes):
                    break
                continue
            selected.append(node)
        for node in deferred:
            if node.node_id in self.nodes:
                self.heap.update(node)
        return selected

    def _remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.node_challenge.pop(node_id, None)
        self.cooldown_until.pop(node_id, None)

    def _remove_challenge_nodes(self, challenge_id: str) -> None:
        for node_id in [
            node_id for node_id, owner in self.node_challenge.items() if owner == challenge_id
        ]:
            self._remove_node(node_id)
        self.heap.rebuild(self.nodes)

    def _run_attempt_batch(
        self,
        nodes: Sequence[RepresentationNode],
        target: int,
        seeds: Sequence[int],
        dose: int,
        *,
        purpose: str,
    ) -> list[AttemptResult]:
        started = time.monotonic()
        journal = {
            "schema": "mastery-attempt-journal-v1",
            "status": "in-flight",
            "step": self.step_index,
            "purpose": purpose,
            "node_ids": [node.node_id for node in nodes],
            "target_u": int(target),
            "seeds": [int(seed) for seed in seeds],
            "simulations": int(dose),
            "deadline_seconds": self.config.attempt_wall_seconds_limit,
        }
        journal_path = (
            self.runtime_root / "attempt-journal" / "current.json"
            if self.runtime_root is not None
            else None
        )
        if journal_path is not None:
            atomic_json(journal_path, journal)
        try:
            with hard_deadline(
                self.config.attempt_wall_seconds_limit
                if self.config.protocol_version >= 2
                else None
            ):
                rows = self.backend.attempt_batch(
                    nodes,
                    target,
                    seeds,
                    simulations=dose,
                )
            status = "completed"
        except AttemptDeadlineExceeded:
            elapsed = time.monotonic() - started
            rows = [
                AttemptResult(
                    node.node_id,
                    False,
                    reason="hard_timeout",
                    attempt_target_u=target,
                    simulations=dose,
                    seed=int(seed),
                    wall_seconds=elapsed / max(len(nodes), 1),
                )
                for node, seed in zip(nodes, seeds, strict=True)
            ]
            status = "hard-timeout"
        if len(rows) != len(nodes):
            raise ValueError("backend returned the wrong number of attempts")
        rows = [
            replace(
                row,
                attempt_target_u=int(target),
                simulations=int(dose),
                seed=int(seed),
            )
            for row, seed in zip(rows, seeds, strict=True)
        ]
        elapsed = time.monotonic() - started
        if journal_path is not None:
            atomic_json(
                journal_path,
                {
                    **journal,
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "outcomes": [
                        row.reason or ("solved" if row.solved else "failed") for row in rows
                    ],
                },
            )
            history = (
                self.runtime_root / "attempt-journal" / f"{self.step_index:08d}-{purpose}.json"
            )
            atomic_json(history, json.loads(journal_path.read_text()))
        return rows

    def _attempt(self, selected: Sequence[RepresentationNode]) -> list[AttemptResult]:
        output: list[AttemptResult | None] = [None] * len(selected)
        strict = max(1, math.ceil(len(selected) * self.config.strict_search_fraction))
        relaxed = [index >= strict for index in range(len(selected))]
        if relaxed:
            shift = self.step_index % len(relaxed)
            relaxed = relaxed[shift:] + relaxed[:shift]
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, node in enumerate(selected):
            challenge = self.challenges[self.node_challenge[node.node_id]]
            # One relaxed lane always supplies recent native positives at
            # target+1; the controller may increase that slack when the rolling
            # positive fraction remains too low.
            relaxed_slack = max(self.outcome.slack, 1)
            target = challenge.target_u + (relaxed_slack if relaxed[index] else 0)
            grouped[target].append(index)
        for target, indexes in grouped.items():
            seeds = [
                self.config.seed + self.step_index * 1_000_003 + index * 100_003
                for index in indexes
            ]
            dose = self.dose_controller.current
            rows = self._run_attempt_batch(
                [selected[index] for index in indexes],
                target,
                seeds,
                dose,
                purpose=f"search-target-{target}",
            )
            for index, seed, result in zip(indexes, seeds, rows, strict=True):
                output[index] = replace(
                    result,
                    attempt_target_u=target,
                    simulations=dose,
                    seed=seed,
                )
        return [row for row in output if row is not None]

    def _dose_probe(
        self, selected: Sequence[RepresentationNode]
    ) -> tuple[list[AttemptResult], dict[str, Any] | None]:
        pair = self.dose_controller.probe_pair
        if (
            self.config.protocol_version < 2
            or pair is None
            or self.step_index % self.config.simulation_probe_interval
            or not selected
        ):
            return [], None
        first_target = self.challenges[self.node_challenge[selected[0].node_id]].target_u
        nodes = [
            node
            for node in selected
            if self.challenges[self.node_challenge[node.node_id]].target_u == first_target
        ][: self.config.simulation_probe_lanes]
        if not nodes:
            return [], None
        low_dose, high_dose = pair
        target = first_target
        seeds = [
            self.config.seed + 700_000_000 + self.step_index * 1009 + index
            for index in range(len(nodes))
        ]
        low = self._run_attempt_batch(
            nodes, target, seeds, low_dose, purpose=f"dose-probe-{low_dose}"
        )
        high = self._run_attempt_batch(
            nodes, target, seeds, high_dose, purpose=f"dose-probe-{high_dose}"
        )
        for rows, dose in ((low, low_dose), (high, high_dose)):
            for node, result in zip(nodes, rows, strict=True):
                self.dose_calibration.observe(
                    dose,
                    (
                        node.raw_probability
                        if np.isfinite(node.raw_probability)
                        else node.probability
                    ),
                    result.solved,
                    result.wall_seconds,
                )
        return [*low, *high], self.dose_controller.observe(low_dose, high_dose, low, high)

    def _verified_witness(self, node: RepresentationNode, result: AttemptResult) -> UnknotWitness:
        root = self.roots[node.root_id]
        witness = UnknotWitness.from_actions(
            root.word,
            root.strands,
            self.backend.spec,
            (*node.semantic_path, *result.semantic_actions),
        )
        witness.verify()
        return witness

    @staticmethod
    def _negative_key(result: AttemptResult) -> str:
        return f"{result.node_id}:{result.attempt_target_u}:{result.simulations}"

    def _admit_training(self, attempts: Sequence[AttemptResult]) -> list[AttemptResult]:
        admitted = []
        for result in attempts:
            if result.solved:
                admitted.append(result)
                continue
            if not admissible_negative(result.reason):
                continue
            key = self._negative_key(result)
            if result.seed is not None:
                self.negative_trials[key].add(int(result.seed))
            if key not in self.admitted_negative_keys and (
                len(self.negative_trials[key]) >= self.config.negative_confirmations
            ):
                self.admitted_negative_keys.add(key)
                admitted.append(result)
        return admitted

    def _distill_snapshot_after_give_up(self, challenge: ChallengeRuntime) -> dict[str, Any] | None:
        snapshot = self.inventory.snapshot
        fraction = self.config.max_snapshot_distillation_fraction
        if snapshot is None or fraction <= 0.0 or self.native_train_steps <= 0:
            return None
        allowed_total = math.floor(fraction * self.native_train_steps / (1.0 - fraction))
        if self.snapshot_distilled_train_steps >= allowed_total:
            return None
        candidates = [
            row
            for row in snapshot.replayable_rows
            if row.get("mapped_knot") == challenge.knot_name
            and row.get("evidence_id") not in self.consumed_snapshot_evidence
            and row.get("witness") is not None
        ]
        candidates.sort(
            key=lambda row: (
                int(row.get("l10", 10**12)),
                int(row.get("l1000", 10**12)),
                str(row.get("evidence_id")),
            )
        )
        for row in candidates:
            try:
                witness = UnknotWitness.from_dict(row["witness"])
                witness.verify()
            except (KeyError, TypeError, ValueError):
                continue
            if witness.start.strands > self.backend.scientist.config.game.max_strands:
                continue
            evidence_id = str(row["evidence_id"])
            trained = self.backend.distill(
                [DistillationExample(f"evidence-snapshot:{evidence_id}", witness)], 1
            )
            if not trained:
                return None
            self.snapshot_distilled_train_steps += trained
            self.consumed_snapshot_evidence.add(evidence_id)
            return {
                "challenge_id": challenge.challenge_id,
                "knot_name": challenge.knot_name,
                "evidence_id": evidence_id,
                "l10": int(row.get("l10", 10 * witness.crossing_changes + witness.moves)),
                "l1000": int(row.get("l1000", 1000 * witness.crossing_changes + witness.moves)),
                "train_steps": trained,
                "snapshot_sha256": snapshot.sha256,
                "trigger": "scientist-gave-up-before-distillation",
            }
        return None

    def _prune(self) -> int:
        excess = len(self.nodes) - self.config.max_heap
        if excess <= 0:
            return 0
        ordered = sorted(
            self.nodes.values(),
            key=lambda node: (
                -node.science_priority,
                1000.0 * node.predicted_crossings + node.predicted_moves
                if np.isfinite([node.predicted_crossings, node.predicted_moves]).all()
                else float("inf"),
                node.attempts,
                node.node_id,
            ),
        )
        keep = {node.node_id for node in ordered[: self.config.max_heap]}
        for node_id in list(self.nodes):
            if node_id not in keep:
                self._remove_node(node_id)
        self.heap.rebuild(self.nodes)
        return excess

    def _close_empty_challenges(self) -> list[str]:
        live_owners = set(self.node_challenge.values())
        closed = []
        for challenge in self.challenges.values():
            if challenge.status == "active" and challenge.challenge_id not in live_owners:
                challenge.status = "heap-evicted"
                challenge.completed_step = self.step_index
                closed.append(challenge.challenge_id)
        return closed

    def step(self) -> dict[str, Any]:
        if self.finished:
            return {"finished": True}
        introduced_at_step_start = self.introduced_count
        self.step_index += 1
        admitted = []
        if (
            self.next_sequence_index < len(self.sequence)
            and self._active_challenges() < self.config.max_active_challenges
            and self.step_index % self.config.admit_interval_steps == 0
        ):
            item = self.sequence[self.next_sequence_index]
            if self._admit_next():
                admitted.append(item.challenge_id)

        refresh_ids = self.refresh.choose(
            self.nodes,
            self.heap,
            top=self.config.refresh_top,
            fair=self.config.refresh_fair,
        )
        self._rescore(refresh_ids)
        frontier_injection = None
        if (
            self.introduced_count // self.config.group_size
            == introduced_at_step_start // self.config.group_size
        ):
            frontier_injection = self._maybe_inject_reservoir()
        selected = self._pop_eligible(self.config.parallel_searches)
        if not selected:
            if self._admit_next():
                admitted.append(self.sequence[self.next_sequence_index - 1].challenge_id)
                selected = self._pop_eligible(self.config.parallel_searches)
        attempts = self._attempt(selected) if selected else []
        probe_attempts, dose_probe = self._dose_probe(selected) if selected else ([], None)
        self.outcome.observe([result.solved for result in attempts])
        for node, result in zip(selected, attempts, strict=True):
            dose = int(result.simulations or self.dose_controller.current)
            raw = node.raw_probability if np.isfinite(node.raw_probability) else node.probability
            self.dose_calibration.observe(dose, raw, result.solved, result.wall_seconds)

        classified_outcomes = []
        for result in [*attempts, *probe_attempts]:
            if result.node_id in self.node_challenge:
                challenge = self.challenges[self.node_challenge[result.node_id]]
                scientific_target = challenge.target_u
            else:
                scientific_target = int(result.attempt_target_u or 0)
            category = outcome_class(
                solved=result.solved,
                reason=result.reason,
                attempt_target_u=result.attempt_target_u,
                scientific_target_u=scientific_target,
            )
            classified_outcomes.append(category)
            self.outcome_totals[category] += 1

        improved = []
        completed = []
        touched = set()
        node_lookup = {node.node_id: node for node in selected}
        owner_lookup = {
            node.node_id: self.node_challenge[node.node_id]
            for node in selected
            if node.node_id in self.node_challenge
        }
        for result in [*attempts, *probe_attempts]:
            node = node_lookup.get(result.node_id)
            challenge_id = owner_lookup.get(result.node_id)
            if node is None or challenge_id is None:
                continue
            challenge = self.challenges[challenge_id]
            touched.add(challenge_id)
            node.attempts += 1
            challenge.attempts += 1
            challenge.search_seconds += result.wall_seconds
            if node.node_id in self.nodes:
                self.cooldown_until[node.node_id] = self.step_index + self.config.cooldown_steps
            if result.solved:
                witness = self._verified_witness(node, result)
                evidence_id = self.inventory.record(
                    witness=witness,
                    knot_name=challenge.knot_name,
                    representation_id=challenge.representation_id,
                    scientist=self.config.scientist,
                    sequence_name=self.config.sequence_name,
                    challenge_id=challenge_id,
                    previous_upper_bound=challenge.current_upper_bound,
                    solver_metadata=dict(getattr(self.backend, "solver_metadata", {})),
                    search_parameters={
                        "simulations": result.simulations,
                        "action_horizon": getattr(self.backend.config, "move_allowance", None),
                        "attempt_target_u": result.attempt_target_u,
                        "seed": result.seed,
                        "scheduled_network_evaluations": result.scheduled_network_evaluations,
                        "wall_seconds": result.wall_seconds,
                        "protocol_version": self.config.protocol_version,
                    },
                )
                if witness.crossing_changes < challenge.current_upper_bound:
                    previous = challenge.current_upper_bound
                    challenge.current_upper_bound = witness.crossing_changes
                    challenge.target_u = witness.crossing_changes - 1
                    challenge.best_evidence_id = evidence_id
                    challenge.best_l1000 = 1000 * witness.crossing_changes + witness.moves
                    improved.append(
                        {
                            "challenge_id": challenge_id,
                            "knot_name": challenge.knot_name,
                            "old_upper": previous,
                            "new_upper": witness.crossing_changes,
                            "moves": witness.moves,
                            "evidence_id": evidence_id,
                        }
                    )
                    if self.certification is not None:
                        certificate = self.certification.certify(
                            evidence_id=evidence_id,
                            knot_name=challenge.knot_name,
                            root=self.roots[node.root_id],
                            witness=witness,
                            declared_lower_bound=challenge.certified_lower_bound,
                        )
                        improved[-1]["certification"] = certificate
                    if witness.crossing_changes <= challenge.certified_lower_bound:
                        challenge.status = "solved-to-lower-bound"
                        challenge.completed_step = self.step_index
                        completed.append(challenge_id)
                        self._remove_challenge_nodes(challenge_id)
                    else:
                        for other_id in [
                            current.node_id
                            for current in self.nodes.values()
                            if self.node_challenge[current.node_id] == challenge_id
                            and current.crossing_distance > challenge.target_u
                        ]:
                            self._remove_node(other_id)
                        for current in self.nodes.values():
                            if self.node_challenge[current.node_id] == challenge_id:
                                current.provenance["current_target_u"] = challenge.target_u
                                current.provenance["remaining_target_u"] = (
                                    challenge.target_u - current.crossing_distance
                                )
                        self.heap.rebuild(self.nodes)
                        self._rescore(list(self.nodes))

        training_attempts = self._admit_training([*attempts, *probe_attempts])
        trained = self.backend.train_native(
            training_attempts, rehearsal_fraction=self.rehearsal_fraction
        )
        self.native_train_steps += trained
        # Training changes operational solve probabilities.  Every searched
        # task must receive a fresh key and return to the science heap.
        self._rescore([node.node_id for node in selected if node.node_id in self.nodes])

        new_nodes = []
        for node, result in zip(selected, attempts, strict=True):
            if result.solved or node.node_id not in self.nodes:
                continue
            challenge_id = self.node_challenge[node.node_id]
            challenge = self.challenges[challenge_id]
            if node.crossing_distance < challenge.target_u:
                children = one_crossing_change_children(
                    node,
                    self.backend.spec,
                    node_budget=self.config.expansion_node_budget,
                    growth=2,
                    diagram_limit=self.config.expansion_diagrams,
                    child_limit=self.config.expansion_children,
                )
                for child in children:
                    if self._add_node(child, challenge_id):
                        new_nodes.append(child.node_id)
        self._rescore(new_nodes)
        for node_id in list(new_nodes):
            node = self.nodes.get(node_id)
            if node is not None and node.probability < self.config.min_child_probability:
                self._remove_node(node_id)

        gave_up = []
        snapshot_distillation = []
        for challenge_id in touched:
            challenge = self.challenges[challenge_id]
            if challenge.status != "active":
                continue
            node_attempt_exhausted = all(
                node.attempts >= self.config.task_attempt_limit
                for node in self.nodes.values()
                if self.node_challenge[node.node_id] == challenge_id
            )
            if (
                challenge.attempts >= self.config.challenge_attempt_limit
                or challenge.search_seconds >= self.config.challenge_search_seconds_limit
                or node_attempt_exhausted
            ):
                challenge.status = "gave-up-threshold"
                challenge.completed_step = self.step_index
                gave_up.append(challenge_id)
                self._remove_challenge_nodes(challenge_id)
                distilled_row = self._distill_snapshot_after_give_up(challenge)
                if distilled_row is not None:
                    snapshot_distillation.append(distilled_row)

        pruned = self._prune()
        evicted = self._close_empty_challenges()
        retention = None
        retention_timeout = False
        retention_report = None
        rehearsal_adjustment = "not-probed"
        if self.step_index % self.config.retention_probe_interval == 0:
            try:
                with hard_deadline(
                    self.config.attempt_wall_seconds_limit
                    if self.config.protocol_version >= 2
                    else None
                ):
                    retention = self.backend.retention_rate(
                        simulations=self.dose_controller.current,
                        seed=self.config.seed + 800_000_000 + self.step_index,
                    )
                retention_report = getattr(self.backend, "last_retention_report", None)
            except AttemptDeadlineExceeded:
                retention_timeout = True
                rehearsal_adjustment = "retention-probe-hard-timeout"
            if retention is not None and retention < self.config.retention_target:
                self.rehearsal_fraction = min(
                    self.config.max_rehearsal_fraction,
                    self.rehearsal_fraction + self.config.rehearsal_fraction_step,
                )
                rehearsal_adjustment = "increased-below-retention-target"
            elif retention is not None:
                self.rehearsal_fraction = max(
                    self.config.min_rehearsal_fraction,
                    self.rehearsal_fraction - self.config.rehearsal_fraction_step,
                )
                rehearsal_adjustment = "decreased-retention-healthy"

        while (
            (completed or gave_up or evicted)
            and self.next_sequence_index < len(self.sequence)
            and self._active_challenges() < self.config.max_active_challenges
            and len(admitted) < self.config.parallel_searches
            and (
                self.introduced_count // self.config.group_size
                == introduced_at_step_start // self.config.group_size
            )
        ):
            item = self.sequence[self.next_sequence_index]
            if not self._admit_next():
                break
            admitted.append(item.challenge_id)

        event = {
            "schema": "multi-knot-mastery-event-v1",
            "step": self.step_index,
            "introduced": self.introduced_count,
            "sequence_total": len(self.sequence),
            "active_challenges": self._active_challenges(),
            "heap_size": len(self.nodes),
            "selected": [node.node_id for node in selected],
            "selected_tasks": [
                {
                    "node_id": node.node_id,
                    "challenge_id": self.node_challenge.get(node.node_id),
                    "raw_p_solve": node.probability,
                    "priority_bonus": node.priority_bonus,
                    "science_key": node.science_priority,
                    "provenance": node.provenance,
                }
                for node in selected
            ],
            "attempts": len(attempts),
            "probe_attempts": len(probe_attempts),
            "solve_evaluations": len(attempts) + len(probe_attempts),
            "verified_solutions": sum(result.solved for result in [*attempts, *probe_attempts]),
            "outcomes": {
                category: classified_outcomes.count(category)
                for category in sorted(set(classified_outcomes))
            },
            "outcome_totals": dict(sorted(self.outcome_totals.items())),
            "strict_challenge_solutions": classified_outcomes.count("strict_challenge_success"),
            "relaxed_training_solutions": classified_outcomes.count("relaxed_training_success"),
            "genuine_upper_bound_improvements": len(improved),
            "improvements": improved,
            "completed": completed,
            "gave_up": gave_up,
            "snapshot_distillation": snapshot_distillation,
            "heap_evicted_challenges": evicted,
            "admitted": admitted,
            "frontier_injections": ([frontier_injection] if frontier_injection is not None else []),
            "new_subtasks": len(new_nodes),
            "pruned_tasks": pruned,
            "native_train_steps": trained,
            "total_native_train_steps": self.native_train_steps,
            "snapshot_distilled_train_steps": self.snapshot_distilled_train_steps,
            "positive_fraction": self.outcome.positive_fraction,
            "training_budget_slack": self.outcome.slack,
            "recent_positive_budget_slack": max(self.outcome.slack, 1),
            "rehearsal_fraction": self.rehearsal_fraction,
            "retention_rate": retention,
            "retention_report": retention_report,
            "retention_probe_timeout": retention_timeout,
            "rehearsal_adjustment": rehearsal_adjustment,
            "retention_capacity_alert": bool(
                retention is not None
                and retention < self.config.retention_target
                and self.rehearsal_fraction >= self.config.max_rehearsal_fraction
            ),
            "simulations": self.dose_controller.current,
            "simulation_probe": dose_probe,
            "dose_calibration": self.dose_calibration.to_dict(),
            "evidence_snapshot": (
                self.inventory.snapshot.manifest() if self.inventory.snapshot is not None else None
            ),
        }
        self.events.append(event)
        self.events = self.events[-200:]
        return event

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA if self.config.protocol_version >= 2 else SCHEMA_V1,
            "config": asdict(self.config),
            "sequence_sha256": self.sequence_sha256,
            "reservoir_sha256": self.reservoir_sha256,
            "evidence_snapshot_sha256": (
                self.inventory.snapshot.sha256 if self.inventory.snapshot is not None else None
            ),
            "reservoir_cursor": self.reservoir_cursor,
            "reservoir_used": sorted(self.reservoir_used),
            "reservoir_injections_by_group": dict(
                sorted(self.reservoir_injections_by_group.items())
            ),
            "step_index": self.step_index,
            "next_sequence_index": self.next_sequence_index,
            "introduced_count": self.introduced_count,
            "native_train_steps": self.native_train_steps,
            "snapshot_distilled_train_steps": self.snapshot_distilled_train_steps,
            "consumed_snapshot_evidence": sorted(self.consumed_snapshot_evidence),
            "rehearsal_fraction": self.rehearsal_fraction,
            "outcome": self.outcome.to_dict(),
            "outcome_totals": dict(sorted(self.outcome_totals.items())),
            "dose_controller": self.dose_controller.to_dict(),
            "dose_calibration": self.dose_calibration.to_dict(),
            "roots": {key: value.to_dict() for key, value in sorted(self.roots.items())},
            "challenges": {key: asdict(value) for key, value in sorted(self.challenges.items())},
            "nodes": [
                asdict(node) for node in sorted(self.nodes.values(), key=lambda row: row.node_id)
            ],
            "node_challenge": dict(sorted(self.node_challenge.items())),
            "cooldown_until": dict(sorted(self.cooldown_until.items())),
            "negative_trials": {
                key: sorted(value) for key, value in sorted(self.negative_trials.items())
            },
            "admitted_negative_keys": sorted(self.admitted_negative_keys),
            "recent_events": self.events,
        }

    def save(self, output: Path, event: dict[str, Any] | None = None) -> None:
        output.mkdir(parents=True, exist_ok=True)
        state_tmp = output / "program-state.json.tmp"
        state_tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(state_tmp, output / "program-state.json")
        if event is not None:
            events = output / "events"
            events.mkdir(exist_ok=True)
            event_path = events / f"{self.step_index:08d}.json"
            if not event_path.exists():
                event_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
        solved = [
            asdict(row) for row in self.challenges.values() if row.best_evidence_id is not None
        ]
        solved_tmp = output / "solved-pool.json.tmp"
        solved_tmp.write_text(
            json.dumps(
                {"schema": "mastery-solved-pool-v1", "rows": solved},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.replace(solved_tmp, output / "solved-pool.json")
        self.backend.save(output / "scientist-state.pt.gz")

        if self.introduced_count and self.introduced_count % self.config.group_size == 0:
            checkpoint = output / "checkpoints" / f"group-{self.introduced_count:03d}"
            if not checkpoint.exists():
                checkpoints = output / "checkpoints"
                checkpoints.mkdir(exist_ok=True)
                staging = checkpoints / f".{checkpoint.name}.tmp-{os.getpid()}"
                staging.mkdir()
                shutil.copy2(output / "program-state.json", staging / "program-state.json")
                shutil.copy2(output / "scientist-state.pt.gz", staging / "scientist-state.pt.gz")
                (staging / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema": (
                                "mastery-group-checkpoint-v2"
                                if self.config.protocol_version >= 2
                                else "mastery-group-checkpoint-v1"
                            ),
                            "introduced": self.introduced_count,
                            "step": self.step_index,
                            "program_state_sha256": _sha256(staging / "program-state.json"),
                            "scientist_state_sha256": _sha256(staging / "scientist-state.pt.gz"),
                            "evidence_snapshot": (
                                self.inventory.snapshot.manifest()
                                if self.inventory.snapshot is not None
                                else None
                            ),
                            "simulation_dose": self.dose_controller.current,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                os.replace(staging, checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--challenge-reservoir", type=Path)
    parser.add_argument("--scientist", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-inventory", type=Path, required=True)
    parser.add_argument("--evidence-snapshot", type=Path)
    parser.add_argument("--rehearsal-panel", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--parallel-searches", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--simulation-levels", default="")
    parser.add_argument("--simulation-probe-interval", type=int, default=20)
    parser.add_argument("--attempt-wall-seconds-limit", type=float, default=900.0)
    parser.add_argument("--protocol-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--action-horizon", type=int, default=256)
    parser.add_argument("--max-heap", type=int, default=200)
    parser.add_argument("--task-attempt-limit", type=int, default=6)
    parser.add_argument("--challenge-attempt-limit", type=int, default=24)
    parser.add_argument("--challenge-seconds-limit", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--short-ablation",
        action="store_true",
        help="require the registered 20-task Q-vs-SKM ablation schema",
    )
    args = parser.parse_args()

    if args.protocol_version >= 2 and args.evidence_snapshot is None:
        raise ValueError("mastery v2 requires a pinned --evidence-snapshot")
    evidence_snapshot = (
        EvidenceSnapshot.load(args.evidence_snapshot)
        if args.evidence_snapshot is not None
        else None
    )
    simulation_levels = tuple(
        sorted(
            set(
                [args.simulations]
                + [int(value) for value in args.simulation_levels.split(",") if value.strip()]
            )
        )
    )
    if simulation_levels[0] != args.simulations:
        raise ValueError("--simulations must be the minimum simulation dose")

    sequence_name, sequence = (
        load_short_ablation_sequence(args.sequence)
        if args.short_ablation
        else load_sequence(args.sequence)
    )
    sequence_hash = _sha256(args.sequence)
    reservoir: list[SequenceChallenge] = []
    reservoir_hash = None
    if args.challenge_reservoir is not None:
        _, reservoir = load_reservoir(args.challenge_reservoir)
        reservoir_hash = _sha256(args.challenge_reservoir)
    all_inputs = [*sequence, *reservoir]
    if max(row.root.strands for row in all_inputs) > 12:
        raise ValueError("sequence exceeds the declared 12-strand program capacity")
    backend_config = MasteryConfig(
        knot_name=f"program:{sequence_name}",
        initial_target_u=max(row.target_u for row in sequence),
        certified_lower_bound=0,
        move_allowance=args.action_horizon,
        objective_move_base=200,
        objective_move_jitter=max(args.action_horizon - 200, 0),
        parallel_searches=args.parallel_searches,
        torch_threads=args.torch_threads,
        simulations=args.simulations,
        train_steps_per_batch=4,
        train_batch_size=64,
        seed=args.seed,
        max_live_nodes=args.max_heap,
        initial_rehearsal_fraction=0.10,
        max_rehearsal_fraction=0.50,
        retention_target=0.80,
        negative_confirmations=3,
        target_positive_fraction=0.50,
        strict_search_fraction=0.50,
        protocol_version=args.protocol_version,
        simulation_levels=simulation_levels if args.protocol_version >= 2 else (args.simulations,),
        simulation_probe_interval=args.simulation_probe_interval,
    )
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    scientist = load_scientist(
        args.scientist,
        args.checkpoint,
        seed=args.seed,
        device=args.device,
        simulations=args.simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    game_config = replace(scientist.config.game, simplify_budget=args.action_horizon)
    scientist.config = replace(scientist.config, game=game_config)
    scientist.game = make_game(game_config)
    if max(row.root.strands for row in all_inputs) > scientist.config.game.max_strands:
        raise ValueError(
            f"{args.scientist} supports {scientist.config.game.max_strands} strands, "
            f"but the inputs require {max(row.root.strands for row in all_inputs)}"
        )
    rehearsal = load_distillation(args.rehearsal_panel)
    backend = ScientistMasteryBackend(scientist, backend_config, rehearsal=rehearsal)
    backend.solver_metadata = {
        "scientist": args.scientist,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
        },
    }
    inventory = EvidenceInventory(args.evidence_inventory, snapshot=evidence_snapshot)
    state_path = args.output / "program-state.json"
    scientist_state = args.output / "scientist-state.pt.gz"
    if args.resume:
        if not state_path.exists() or not scientist_state.exists():
            raise ValueError("--resume requires both program and scientist states")
        backend.restore(scientist_state)
        program = MasteryProgram.from_saved(
            state_path,
            backend,
            sequence,
            inventory,
            sequence_sha256=sequence_hash,
            reservoir=reservoir,
            reservoir_sha256=reservoir_hash,
            runtime_root=args.output,
        )
    else:
        if state_path.exists() or scientist_state.exists():
            raise ValueError("output already contains state; use --resume")
        config = ProgramConfig(
            sequence_name=sequence_name,
            scientist=args.scientist,
            max_heap=args.max_heap,
            parallel_searches=args.parallel_searches,
            task_attempt_limit=args.task_attempt_limit,
            challenge_attempt_limit=args.challenge_attempt_limit,
            challenge_search_seconds_limit=args.challenge_seconds_limit,
            seed=args.seed,
            protocol_version=args.protocol_version,
            attempt_wall_seconds_limit=args.attempt_wall_seconds_limit,
            simulation_levels=(
                simulation_levels if args.protocol_version >= 2 else (args.simulations,)
            ),
            simulation_probe_interval=args.simulation_probe_interval,
        )
        program = MasteryProgram(
            config,
            backend,
            sequence,
            inventory,
            sequence_sha256=sequence_hash,
            reservoir=reservoir,
            reservoir_sha256=reservoir_hash,
            runtime_root=args.output,
        )
        program.save(args.output)
    for _ in range(args.steps):
        if program.finished:
            break
        event = program.step()
        print(json.dumps(event, sort_keys=True), flush=True)
        program.save(args.output, event)
    print(
        json.dumps(
            {
                "finished": program.finished,
                "introduced": program.introduced_count,
                "output": str(args.output),
                "steps": program.step_index,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
