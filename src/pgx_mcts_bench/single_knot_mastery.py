"""Resumable, proof-carrying mastery search for one knot.

The coordinator maintains a mutable priority queue of representations reached
from one or more verified starting braids.  A node at crossing-change distance
``d`` is scored by the scientist's factorized head under the remaining L1000
budget.  The head is interpreted operationally as
``P(this solver succeeds | encoded budget, representation)``; it is not a
claim that a witness exists.  The heap stores ``-P`` so its smallest keys are
the most promising nodes.

Search remains representation aware: a one-crossing-change child is only a
candidate reached from one exact diagram.  A knot-type upper bound is emitted
only after the complete semantic path from a declared starting representation
to the unknot has been replay-verified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import math
import os
import tempfile
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from rf_knots.actions import CROSSING_CHANGE, ActionSpec
from rf_knots.evidence import BraidState, UnknotWitness, braid_instance_id
from rf_knots.reference import successors

from pgx_mcts_bench.adaptive_scientists import (
    FixedWordGame,
    KnotItem,
    Scientist,
    calibrated_solve_probability,
    load_scientist,
)
from pgx_mcts_bench.collaborative_scientists import (
    translate_semantic_record,
    verified_record_cost,
)
from pgx_mcts_bench.data import GameRecord, Position
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.gpu_inference import (
    CoordinatedPolicyValueNet,
    PersistentInferenceCoordinator,
)
from pgx_mcts_bench.mastery_v2 import admissible_negative
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import train_alphazero_step

SCHEMA_V1 = "single-knot-mastery-v1"
SCHEMA = "single-knot-mastery-v2"


def _stable_id(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True)
class MasteryConfig:
    knot_name: str
    initial_target_u: int
    protocol_version: int = 2
    certified_lower_bound: int = 0
    ratio: float = 1000.0
    move_allowance: int = 128
    objective_move_base: int | None = None
    objective_move_jitter: int = 0
    parallel_searches: int = 4
    torch_threads: int = 4
    attempts_per_node: int = 1
    simulations: int = 128
    train_steps_per_batch: int = 4
    train_batch_size: int = 64
    seed: int = 20260815
    initial_equivalent_representations: int = 8
    equivalent_node_budget: int = 500
    expansion_node_budget: int = 500
    expansion_growth: int = 2
    expansion_diagrams: int = 24
    expansion_children: int = 96
    max_live_nodes: int = 20_000
    refresh_top: int = 8
    refresh_fair: int = 2
    distill_after_native_attempts: int = 200
    max_distillation_fraction: float = 0.10
    target_positive_fraction: float = 0.50
    positive_fraction_tolerance: float = 0.08
    outcome_window: int = 64
    outcome_warmup: int = 8
    strict_search_fraction: float = 0.50
    max_training_budget_slack: int = 3
    negative_confirmations: int = 3
    initial_rehearsal_fraction: float = 0.25
    max_rehearsal_fraction: float = 0.50
    rehearsal_fraction_step: float = 0.05
    retention_target: float = 0.80
    retention_probe_interval: int = 20
    simulation_levels: tuple[int, ...] = ()
    simulation_probe_interval: int = 20
    simulation_probe_lanes: int = 2
    simulation_probe_min_pairs: int = 12
    simulation_success_margin: float = 0.05
    simulation_l1000_tolerance: float = 5.0

    def __post_init__(self) -> None:
        if self.protocol_version not in (1, 2):
            raise ValueError("protocol_version must be 1 or 2")
        if self.initial_target_u < 0:
            raise ValueError("initial_target_u must be non-negative")
        if self.certified_lower_bound < 0:
            raise ValueError("certified_lower_bound must be non-negative")
        if self.objective_move_base is not None and self.objective_move_base < 0:
            raise ValueError("objective_move_base must be non-negative")
        if self.objective_move_jitter < 0:
            raise ValueError("objective_move_jitter must be non-negative")
        if self.parallel_searches < 1 or self.attempts_per_node < 1:
            raise ValueError("parallel searches and attempts must be positive")
        if self.torch_threads < 1:
            raise ValueError("torch_threads must be positive")
        if self.max_live_nodes < self.parallel_searches:
            raise ValueError("max_live_nodes must cover every parallel search lane")
        if not 0.0 <= self.max_distillation_fraction < 1.0:
            raise ValueError("max_distillation_fraction must be in [0, 1)")
        if not 0.0 < self.target_positive_fraction < 1.0:
            raise ValueError("target_positive_fraction must be in (0, 1)")
        if not 0.0 <= self.positive_fraction_tolerance < 0.5:
            raise ValueError("positive_fraction_tolerance must be in [0, 0.5)")
        if self.outcome_window < 1 or not 1 <= self.outcome_warmup <= self.outcome_window:
            raise ValueError("outcome warmup/window are inconsistent")
        if not 0.0 < self.strict_search_fraction <= 1.0:
            raise ValueError("strict_search_fraction must be in (0, 1]")
        if self.max_training_budget_slack < 0:
            raise ValueError("max_training_budget_slack must be non-negative")
        if self.negative_confirmations < 1:
            raise ValueError("negative_confirmations must be positive")
        if not 0.0 <= self.initial_rehearsal_fraction <= self.max_rehearsal_fraction <= 0.5:
            raise ValueError("rehearsal fractions must satisfy 0 <= initial <= max <= 0.5")
        if self.rehearsal_fraction_step <= 0.0:
            raise ValueError("rehearsal_fraction_step must be positive")
        if not 0.0 <= self.retention_target <= 1.0:
            raise ValueError("retention_target must be in [0, 1]")
        if self.retention_probe_interval < 1:
            raise ValueError("retention_probe_interval must be positive")
        levels = tuple(int(value) for value in self.simulation_levels)
        if levels and (levels != tuple(sorted(set(levels))) or min(levels) < 1):
            raise ValueError("simulation_levels must be sorted unique positive integers")
        if levels and self.simulations != levels[0]:
            raise ValueError("simulations must be the minimum simulation_levels dose")
        if self.simulation_probe_interval < 1 or self.simulation_probe_lanes < 1:
            raise ValueError("simulation probe interval and lanes must be positive")
        if self.simulation_probe_min_pairs < 1:
            raise ValueError("simulation_probe_min_pairs must be positive")


def _saved_config(payload: dict[str, Any]) -> MasteryConfig:
    """Load a state config without changing the protocol of legacy v1 runs."""
    row = dict(payload["config"])
    if payload.get("schema") == SCHEMA_V1 and "protocol_version" not in row:
        simulations = int(row.get("simulations", 128))
        row.update(
            {
                "protocol_version": 1,
                "strict_search_fraction": 1.0,
                "max_training_budget_slack": 0,
                "negative_confirmations": 1,
                "initial_rehearsal_fraction": 0.0,
                "max_rehearsal_fraction": 0.0,
                "simulation_levels": (simulations,),
            }
        )
    return MasteryConfig(**row)


@dataclass
class RepresentationNode:
    node_id: str
    root_id: str
    word: tuple[int, ...]
    strands: int
    crossing_distance: int
    semantic_path: tuple[int, ...]
    parent_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    probability: float = 0.0
    predicted_crossings: float = float("nan")
    predicted_moves: float = float("nan")
    score_version: int = 0
    last_scored_step: int = -1
    attempts: int = 0
    priority_bonus: float = 0.0
    raw_probability: float = float("nan")
    calibration_samples: int = 0
    score_uncertainty: float = 0.0
    expected_cpu_seconds: float = 0.0
    scheduling_priority: float = float("nan")

    @property
    def science_priority(self) -> float:
        """Scheduling key without corrupting calibrated solve probability."""
        if np.isfinite(self.scheduling_priority):
            return float(self.scheduling_priority)
        return float(self.probability + self.priority_bonus)

    @classmethod
    def create(
        cls,
        *,
        root_id: str,
        word: Iterable[int],
        strands: int,
        crossing_distance: int,
        semantic_path: Iterable[int],
        parent_id: str | None = None,
        provenance: dict[str, Any] | None = None,
        priority_bonus: float = 0.0,
    ) -> RepresentationNode:
        compact = tuple(int(value) for value in word if int(value))
        path = tuple(int(action) for action in semantic_path)
        identity = _stable_id(
            {
                "root_id": root_id,
                "word": compact,
                "strands": int(strands),
                "crossing_distance": int(crossing_distance),
                "semantic_path": path,
            }
        )
        return cls(
            identity,
            root_id,
            compact,
            int(strands),
            int(crossing_distance),
            path,
            parent_id,
            dict(provenance or {}),
            priority_bonus=float(priority_bonus),
        )

    @property
    def state_key(self) -> tuple[str, tuple[int, ...], int, int]:
        return self.root_id, self.word, self.strands, self.crossing_distance


@dataclass(frozen=True)
class NodeScore:
    probability: float
    predicted_crossings: float
    predicted_moves: float


@dataclass(frozen=True)
class AttemptResult:
    node_id: str
    solved: bool
    native_record: GameRecord | None = None
    semantic_actions: tuple[int, ...] = ()
    crossing_changes: int | None = None
    moves: int | None = None
    reason: str = ""
    attempt_target_u: int | None = None
    simulations: int | None = None
    seed: int | None = None
    scheduled_network_evaluations: int = 0
    wall_seconds: float = 0.0


@dataclass(frozen=True)
class DistillationExample:
    source: str
    witness: UnknotWitness

    @property
    def l10(self) -> int:
        return 10 * self.witness.crossing_changes + self.witness.moves


class MasteryBackend(Protocol):
    spec: ActionSpec

    def score(self, nodes: Sequence[RepresentationNode], target_u: int) -> list[NodeScore]: ...

    def attempt_batch(
        self,
        nodes: Sequence[RepresentationNode],
        target_u: int,
        seeds: Sequence[int],
        *,
        simulations: int | None = None,
    ) -> list[AttemptResult]: ...

    def train_native(
        self,
        attempts: Sequence[AttemptResult],
        *,
        rehearsal_fraction: float,
    ) -> int: ...

    def retention_rate(self, *, simulations: int, seed: int) -> float | None: ...

    def distill(self, examples: Sequence[DistillationExample], max_steps: int) -> int: ...

    def save(self, path: Path) -> None: ...


class AdaptiveOutcomeController:
    """Keep solve-head replay near a target outcome mix via budget slack.

    The controller changes only the encoded training target.  The coordinator's
    scientific target is stored separately and can move only after witness
    replay verification.
    """

    def __init__(
        self,
        *,
        target: float,
        tolerance: float,
        window: int,
        warmup: int,
        max_slack: int,
        slack: int = 0,
        outcomes: Sequence[bool] = (),
    ) -> None:
        self.target = float(target)
        self.tolerance = float(tolerance)
        self.window = int(window)
        self.warmup = int(warmup)
        self.max_slack = int(max_slack)
        self.slack = int(slack)
        self.outcomes: deque[bool] = deque((bool(value) for value in outcomes), maxlen=self.window)

    @property
    def positive_fraction(self) -> float | None:
        if not self.outcomes:
            return None
        return sum(self.outcomes) / len(self.outcomes)

    def observe(self, outcomes: Sequence[bool]) -> int:
        self.outcomes.extend(bool(value) for value in outcomes)
        rate = self.positive_fraction
        if rate is None or len(self.outcomes) < self.warmup:
            return self.slack
        if rate < self.target - self.tolerance:
            self.slack = min(self.max_slack, self.slack + 1)
        elif rate > self.target + self.tolerance:
            self.slack = max(0, self.slack - 1)
        return self.slack

    def targets(self, scientific_target: int, lanes: int, strict_fraction: float) -> list[int]:
        strict = max(1, math.ceil(lanes * strict_fraction))
        return [
            scientific_target if index < strict else scientific_target + self.slack
            for index in range(lanes)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"slack": self.slack, "outcomes": list(self.outcomes)}


class SimulationDoseController:
    """Tune MCTS simulations with paired, representation-matched probes."""

    def __init__(
        self,
        levels: Sequence[int],
        *,
        current: int,
        min_pairs: int,
        success_margin: float,
        l1000_tolerance: float,
        observations: dict[str, list[dict[str, float | bool | None]]] | None = None,
    ) -> None:
        self.levels = tuple(sorted(set(int(value) for value in levels)))
        if current not in self.levels:
            raise ValueError("current simulation dose is absent from levels")
        self.current = int(current)
        self.min_pairs = int(min_pairs)
        self.success_margin = float(success_margin)
        self.l1000_tolerance = float(l1000_tolerance)
        self.observations = dict(observations or {})

    @property
    def probe_pair(self) -> tuple[int, int] | None:
        index = self.levels.index(self.current)
        if index + 1 >= len(self.levels):
            return None
        return self.current, self.levels[index + 1]

    @staticmethod
    def _cost(result: AttemptResult) -> float | None:
        if not result.solved or result.crossing_changes is None or result.moves is None:
            return None
        return 1000.0 * result.crossing_changes + result.moves

    def observe(
        self,
        low_dose: int,
        high_dose: int,
        low: Sequence[AttemptResult],
        high: Sequence[AttemptResult],
    ) -> dict[str, Any]:
        if len(low) != len(high):
            raise ValueError("paired dose probes returned unequal counts")
        key = f"{low_dose}:{high_dose}"
        bucket = self.observations.setdefault(key, [])
        for left, right in zip(low, high, strict=True):
            bucket.append(
                {
                    "low_solved": bool(left.solved),
                    "high_solved": bool(right.solved),
                    "low_l1000": self._cost(left),
                    "high_l1000": self._cost(right),
                    "low_wall_seconds": float(left.wall_seconds),
                    "high_wall_seconds": float(right.wall_seconds),
                }
            )
        rows = bucket[-max(self.min_pairs * 4, self.min_pairs) :]
        n = len(rows)
        low_rate = sum(bool(row["low_solved"]) for row in rows) / n
        high_rate = sum(bool(row["high_solved"]) for row in rows) / n
        success_deltas = np.asarray(
            [int(bool(row["high_solved"])) - int(bool(row["low_solved"])) for row in rows],
            dtype=np.float64,
        )
        success_gain = float(success_deltas.mean())
        success_se = float(success_deltas.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        success_lower_95 = success_gain - 1.96 * success_se
        paired_costs = [
            float(row["low_l1000"]) - float(row["high_l1000"])
            for row in rows
            if bool(row["low_solved"])
            and bool(row["high_solved"])
            and row["low_l1000"] is not None
            and row["high_l1000"] is not None
            and np.isfinite([row["low_l1000"], row["high_l1000"]]).all()
        ]
        cost_gain = float(np.mean(paired_costs)) if paired_costs else 0.0
        cost_se = (
            float(np.std(paired_costs, ddof=1) / math.sqrt(len(paired_costs)))
            if len(paired_costs) > 1
            else 0.0
        )
        cost_lower_95 = cost_gain - 1.96 * cost_se
        low_wall = sum(float(row.get("low_wall_seconds", 0.0)) for row in rows)
        high_wall = sum(float(row.get("high_wall_seconds", 0.0)) for row in rows)
        low_success_per_second = (
            sum(bool(row["low_solved"]) for row in rows) / low_wall if low_wall else None
        )
        high_success_per_second = (
            sum(bool(row["high_solved"]) for row in rows) / high_wall if high_wall else None
        )
        throughput_not_worse = (
            low_success_per_second is None
            or high_success_per_second is None
            or high_success_per_second >= low_success_per_second
        )
        promoted = False
        if (
            n >= self.min_pairs
            and (success_lower_95 > self.success_margin or cost_lower_95 > self.l1000_tolerance)
            and throughput_not_worse
        ):
            self.current = high_dose
            promoted = True
        return {
            "pair": [low_dose, high_dose],
            "pairs": n,
            "low_success_rate": low_rate,
            "high_success_rate": high_rate,
            "paired_success_gain": success_gain,
            "paired_success_gain_lower_95": success_lower_95,
            "paired_l1000_gain": cost_gain,
            "paired_l1000_gain_lower_95": cost_lower_95,
            "low_successes_per_cpu_second": low_success_per_second,
            "high_successes_per_cpu_second": high_success_per_second,
            "throughput_not_worse": throughput_not_worse,
            "low_scheduled_network_evaluations": sum(
                row.scheduled_network_evaluations for row in low
            ),
            "high_scheduled_network_evaluations": sum(
                row.scheduled_network_evaluations for row in high
            ),
            "low_wall_seconds": sum(row.wall_seconds for row in low),
            "high_wall_seconds": sum(row.wall_seconds for row in high),
            "promoted": promoted,
            "current_simulations": self.current,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current,
            "observations": self.observations,
        }


class MutableProbabilityHeap:
    """Versioned mutable heap; stale entries are discarded on access."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, float, int, str, int]] = []

    def update(self, node: RepresentationNode) -> None:
        node.score_version += 1
        predicted_l1000 = (
            1000.0 * node.predicted_crossings + node.predicted_moves
            if np.isfinite([node.predicted_crossings, node.predicted_moves]).all()
            else float("inf")
        )
        heapq.heappush(
            self._heap,
            (
                -node.science_priority,
                predicted_l1000,
                node.last_scored_step,
                node.node_id,
                node.score_version,
            ),
        )

    def pop(self, nodes: dict[str, RepresentationNode]) -> RepresentationNode:
        while self._heap:
            _, _, _, node_id, version = heapq.heappop(self._heap)
            node = nodes.get(node_id)
            if node is not None and node.score_version == version:
                return node
        raise IndexError("priority heap is empty")

    def peek_ids(self, nodes: dict[str, RepresentationNode], count: int) -> list[str]:
        chosen: list[RepresentationNode] = []
        for _ in range(min(count, len(nodes))):
            try:
                chosen.append(self.pop(nodes))
            except IndexError:
                break
        for node in chosen:
            self.update(node)
        return [node.node_id for node in chosen]

    def rebuild(self, nodes: dict[str, RepresentationNode]) -> None:
        self._heap.clear()
        for node in nodes.values():
            self.update(node)


class FairRefreshScheduler:
    """Refresh heap leaders often and every live node eventually."""

    def __init__(self) -> None:
        self._cursor = 0

    def choose(
        self,
        nodes: dict[str, RepresentationNode],
        heap: MutableProbabilityHeap,
        *,
        top: int,
        fair: int,
    ) -> list[str]:
        selected = heap.peek_ids(nodes, top)
        ordered = sorted(nodes)
        if ordered and fair:
            for _ in range(min(fair, len(ordered))):
                selected.append(ordered[self._cursor % len(ordered)])
                self._cursor += 1
        return list(dict.fromkeys(selected))


def equivalent_representations(
    start: BraidState,
    spec: ActionSpec,
    *,
    count: int,
    node_budget: int,
    growth: int = 2,
) -> list[tuple[BraidState, tuple[int, ...]]]:
    """Return diverse short diagrams connected by type-preserving moves only."""
    cap = len(start.word) + growth
    queue = deque([(start, ())])
    seen = {(start.word, start.strands)}
    reached: list[tuple[BraidState, tuple[int, ...]]] = []
    while queue and len(seen) <= node_budget:
        state, path = queue.popleft()
        reached.append((state, path))
        for action, word, strands in successors(spec, state.word, state.strands, False):
            if len(word) > cap or (word, strands) in seen:
                continue
            seen.add((word, strands))
            queue.append((BraidState(word, strands), (*path, action)))
    reached.sort(key=lambda item: (len(item[0].word), item[0].strands, item[0].word, item[1]))
    if count >= len(reached):
        return reached
    # Keep the canonical state plus evenly spaced short alternatives instead of
    # taking near-duplicates from one tiny BFS neighbourhood.
    canonical = next(item for item in reached if item[0] == start)
    alternatives = [item for item in reached if item[0] != start]
    if count == 1 or not alternatives:
        return [canonical]
    indices = np.linspace(0, len(alternatives) - 1, count - 1, dtype=int)
    return [canonical, *(alternatives[int(index)] for index in indices)]


def one_crossing_change_children(
    node: RepresentationNode,
    spec: ActionSpec,
    *,
    node_budget: int,
    growth: int,
    diagram_limit: int,
    child_limit: int,
) -> list[RepresentationNode]:
    """Generate one-CC children from short type-preserving variants of a node."""
    start = BraidState(node.word, node.strands)
    diagrams = equivalent_representations(
        start,
        spec,
        count=max(diagram_limit, 1),
        node_budget=node_budget,
        growth=growth,
    )
    children: list[RepresentationNode] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    for diagram, prefix in diagrams:
        for position in range(len(diagram.word)):
            action = spec.encode(CROSSING_CHANGE, position)
            word = (
                diagram.word[:position] + (-diagram.word[position],) + diagram.word[position + 1 :]
            )
            state = (word, diagram.strands)
            if state in seen:
                continue
            seen.add(state)
            lineage = list(node.provenance.get("subtask_lineage", []))
            lineage.append(
                {
                    "parent_node_id": node.node_id,
                    "parent_crossing_distance": node.crossing_distance,
                    "type_preserving_prefix": [int(value) for value in prefix],
                    "crossing_change_action": action,
                    "crossing_change_position": position,
                }
            )
            child_provenance = {
                **node.provenance,
                "operation": "type-preserving-prefix-then-one-crossing-change",
                "parent": node.node_id,
                "subtask_depth": node.crossing_distance + 1,
                "subtask_lineage": lineage,
            }
            if "current_target_u" in child_provenance:
                child_provenance["remaining_target_u"] = int(
                    child_provenance["current_target_u"]
                ) - (node.crossing_distance + 1)
            children.append(
                RepresentationNode.create(
                    root_id=node.root_id,
                    word=word,
                    strands=diagram.strands,
                    crossing_distance=node.crossing_distance + 1,
                    semantic_path=(*node.semantic_path, *prefix, action),
                    parent_id=node.node_id,
                    provenance=child_provenance,
                    priority_bonus=node.priority_bonus,
                )
            )
    children.sort(key=lambda child: (len(child.word), child.strands, child.word, child.node_id))
    return children[:child_limit]


class MasteryCoordinator:
    def __init__(
        self,
        config: MasteryConfig,
        backend: MasteryBackend,
        roots: Sequence[tuple[BraidState, dict[str, Any]]],
        *,
        distillation: Sequence[DistillationExample] = (),
    ) -> None:
        self.config = config
        self.schema = SCHEMA if config.protocol_version == 2 else SCHEMA_V1
        self.backend = backend
        self.target_u = config.initial_target_u
        self.step_index = 0
        self.native_attempts_at_target = 0
        self.native_train_steps = 0
        self.distilled_train_steps = 0
        self.rehearsal_fraction = config.initial_rehearsal_fraction
        self.best_witness: UnknotWitness | None = None
        self.nodes: dict[str, RepresentationNode] = {}
        self.root_states: dict[str, BraidState] = {}
        self.seen_states: set[tuple[str, tuple[int, ...], int, int]] = set()
        self.heap = MutableProbabilityHeap()
        self.refresh = FairRefreshScheduler()
        self.distillation = sorted(distillation, key=lambda item: (item.l10, item.source))
        self.events: list[dict[str, Any]] = []
        self.outcome_controller = AdaptiveOutcomeController(
            target=config.target_positive_fraction,
            tolerance=config.positive_fraction_tolerance,
            window=config.outcome_window,
            warmup=config.outcome_warmup,
            max_slack=config.max_training_budget_slack,
        )
        levels = config.simulation_levels or (
            config.simulations,
            config.simulations * 2,
            config.simulations * 4,
        )
        self.dose_controller = SimulationDoseController(
            levels,
            current=config.simulations,
            min_pairs=config.simulation_probe_min_pairs,
            success_margin=config.simulation_success_margin,
            l1000_tolerance=config.simulation_l1000_tolerance,
        )
        self.negative_trials: dict[str, set[int]] = defaultdict(set)
        self.admitted_negative_keys: set[str] = set()

        for root_index, (root, provenance) in enumerate(roots):
            root_id = f"root-{root_index:02d}-{braid_instance_id(root.word, root.strands)[6:18]}"
            self.root_states[root_id] = root
            equivalents = equivalent_representations(
                root,
                backend.spec,
                count=config.initial_equivalent_representations,
                node_budget=config.equivalent_node_budget,
                growth=config.expansion_growth,
            )
            for state, path in equivalents:
                self._add_node(
                    RepresentationNode.create(
                        root_id=root_id,
                        word=state.word,
                        strands=state.strands,
                        crossing_distance=0,
                        semantic_path=path,
                        provenance={**provenance, "equivalent_seed": True},
                    )
                )
        self._rescore(list(self.nodes))

    @classmethod
    def from_saved(
        cls,
        state_path: Path,
        backend: MasteryBackend,
        *,
        distillation: Sequence[DistillationExample] = (),
    ) -> MasteryCoordinator:
        payload = json.loads(state_path.read_text())
        if payload.get("schema") not in (SCHEMA_V1, SCHEMA):
            raise ValueError(f"unsupported mastery state schema in {state_path}")
        coordinator = cls.__new__(cls)
        coordinator.config = _saved_config(payload)
        coordinator.schema = str(payload["schema"])
        coordinator.backend = backend
        coordinator.target_u = int(payload["target_u"])
        coordinator.step_index = int(payload["step_index"])
        coordinator.native_attempts_at_target = int(payload["native_attempts_at_target"])
        coordinator.native_train_steps = int(payload["native_train_steps"])
        coordinator.distilled_train_steps = int(payload["distilled_train_steps"])
        coordinator.rehearsal_fraction = float(
            payload.get("rehearsal_fraction", coordinator.config.initial_rehearsal_fraction)
        )
        coordinator.best_witness = (
            UnknotWitness.from_dict(payload["best_witness"])
            if payload.get("best_witness")
            else None
        )
        if coordinator.best_witness is not None:
            coordinator.best_witness.verify()
        coordinator.root_states = {
            root_id: BraidState.from_dict(row) for root_id, row in payload["roots"].items()
        }
        coordinator.nodes = {}
        for row in payload["nodes"]:
            row = dict(row)
            row["word"] = tuple(int(value) for value in row["word"])
            row["semantic_path"] = tuple(int(value) for value in row["semantic_path"])
            node = RepresentationNode(**row)
            coordinator.nodes[node.node_id] = node
        coordinator.seen_states = {node.state_key for node in coordinator.nodes.values()}
        coordinator.heap = MutableProbabilityHeap()
        coordinator.heap.rebuild(coordinator.nodes)
        coordinator.refresh = FairRefreshScheduler()
        coordinator.distillation = sorted(distillation, key=lambda item: (item.l10, item.source))
        coordinator.events = list(payload.get("events", []))
        outcome = payload.get("outcome_controller", {})
        coordinator.outcome_controller = AdaptiveOutcomeController(
            target=coordinator.config.target_positive_fraction,
            tolerance=coordinator.config.positive_fraction_tolerance,
            window=coordinator.config.outcome_window,
            warmup=coordinator.config.outcome_warmup,
            max_slack=coordinator.config.max_training_budget_slack,
            slack=int(outcome.get("slack", 0)),
            outcomes=outcome.get("outcomes", ()),
        )
        levels = coordinator.config.simulation_levels or (
            coordinator.config.simulations,
            coordinator.config.simulations * 2,
            coordinator.config.simulations * 4,
        )
        dose = payload.get("dose_controller", {})
        coordinator.dose_controller = SimulationDoseController(
            levels,
            current=int(dose.get("current", coordinator.config.simulations)),
            min_pairs=coordinator.config.simulation_probe_min_pairs,
            success_margin=coordinator.config.simulation_success_margin,
            l1000_tolerance=coordinator.config.simulation_l1000_tolerance,
            observations=dose.get("observations", {}),
        )
        coordinator.negative_trials = defaultdict(
            set,
            {
                str(key): {int(seed) for seed in seeds}
                for key, seeds in payload.get("negative_trials", {}).items()
            },
        )
        coordinator.admitted_negative_keys = set(payload.get("admitted_negative_keys", []))
        return coordinator

    @property
    def finished(self) -> bool:
        return self.target_u < self.config.certified_lower_bound or not self.nodes

    def _add_node(self, node: RepresentationNode) -> bool:
        if node.crossing_distance > self.target_u or node.state_key in self.seen_states:
            return False
        self.nodes[node.node_id] = node
        self.seen_states.add(node.state_key)
        return True

    def _rescore(self, node_ids: Sequence[str]) -> None:
        live = [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
        if not live:
            return
        scores = self.backend.score(live, self.target_u)
        if len(scores) != len(live):
            raise ValueError("backend returned the wrong number of scores")
        for node, score in zip(live, scores, strict=True):
            node.probability = float(np.clip(score.probability, 0.0, 1.0))
            node.predicted_crossings = float(score.predicted_crossings)
            node.predicted_moves = float(score.predicted_moves)
            node.last_scored_step = self.step_index
            self.heap.update(node)

    def _verified_full_witness(self, result: AttemptResult) -> UnknotWitness:
        node = self.nodes[result.node_id]
        root = self.root_states[node.root_id]
        actions = (*node.semantic_path, *result.semantic_actions)
        witness = UnknotWitness.from_actions(root.word, root.strands, self.backend.spec, actions)
        witness.verify()
        return witness

    def _prune_live_nodes(self) -> int:
        excess = len(self.nodes) - self.config.max_live_nodes
        if excess <= 0:
            return 0
        ordered = sorted(
            self.nodes.values(),
            key=lambda node: (
                -node.probability,
                (
                    1000.0 * node.predicted_crossings + node.predicted_moves
                    if np.isfinite([node.predicted_crossings, node.predicted_moves]).all()
                    else float("inf")
                ),
                node.crossing_distance,
                len(node.word),
                node.node_id,
            ),
        )
        keep = {node.node_id for node in ordered[: self.config.max_live_nodes]}
        self.nodes = {node_id: node for node_id, node in self.nodes.items() if node_id in keep}
        # Removed states remain in ``seen_states``. Eviction is a bounded-search
        # decision, not permission to regenerate the same low-priority branch.
        self.heap.rebuild(self.nodes)
        return excess

    def _ratchet(self, witness: UnknotWitness) -> None:
        incumbent = self.best_witness
        if incumbent is not None and (
            witness.crossing_changes,
            1000 * witness.crossing_changes + witness.moves,
        ) >= (
            incumbent.crossing_changes,
            1000 * incumbent.crossing_changes + incumbent.moves,
        ):
            return
        self.best_witness = witness
        old_target = self.target_u
        self.target_u = witness.crossing_changes - 1
        self.native_attempts_at_target = 0
        self.events.append(
            {
                "type": "verified_upper_bound",
                "step": self.step_index,
                "old_target_u": old_target,
                "crossing_changes": witness.crossing_changes,
                "moves": witness.moves,
                "l1000": 1000 * witness.crossing_changes + witness.moves,
                "new_target_u": self.target_u,
                "root_instance_id": witness.instance_id,
            }
        )
        self.nodes = {
            node_id: node
            for node_id, node in self.nodes.items()
            if node.crossing_distance <= self.target_u
        }
        self.heap.rebuild(self.nodes)
        self._rescore(list(self.nodes))

    def _distillation_allowance(self) -> int:
        fraction = self.config.max_distillation_fraction
        if fraction <= 0.0 or not self.distillation:
            return 0
        # d / (native + d) <= fraction
        total_allowed = math.floor(fraction * self.native_train_steps / (1.0 - fraction))
        return max(total_allowed - self.distilled_train_steps, 0)

    @staticmethod
    def _negative_key(result: AttemptResult) -> str:
        return ":".join(
            (
                result.node_id,
                str(result.attempt_target_u),
                str(result.simulations),
            )
        )

    def _admit_training_attempts(
        self, attempts: Sequence[AttemptResult]
    ) -> tuple[list[AttemptResult], int]:
        if self.config.protocol_version == 1:
            return list(attempts), 0
        admitted: list[AttemptResult] = []
        pending = 0
        for result in attempts:
            if result.solved:
                admitted.append(result)
                continue
            if not admissible_negative(result.reason):
                continue
            key = self._negative_key(result)
            if result.seed is not None:
                self.negative_trials[key].add(int(result.seed))
            if key in self.admitted_negative_keys:
                continue
            if len(self.negative_trials[key]) >= self.config.negative_confirmations:
                self.admitted_negative_keys.add(key)
                admitted.append(result)
            else:
                pending += 1
        return admitted, pending

    def _attempt_grouped(
        self,
        nodes: Sequence[RepresentationNode],
        targets: Sequence[int],
        seeds: Sequence[int],
        simulations: Sequence[int],
    ) -> list[AttemptResult]:
        if not (len(nodes) == len(targets) == len(seeds) == len(simulations)):
            raise ValueError("attempt lanes have inconsistent lengths")
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (target, dose) in enumerate(zip(targets, simulations, strict=True)):
            grouped[(int(target), int(dose))].append(index)
        output: list[AttemptResult | None] = [None] * len(nodes)
        for (target, dose), indexes in grouped.items():
            rows = self.backend.attempt_batch(
                [nodes[index] for index in indexes],
                target,
                [seeds[index] for index in indexes],
                simulations=dose,
            )
            if len(rows) != len(indexes):
                raise ValueError("backend returned the wrong number of attempts")
            for index, result in zip(indexes, rows, strict=True):
                output[index] = replace(
                    result,
                    attempt_target_u=target,
                    simulations=dose,
                    seed=int(seeds[index]),
                )
        if any(result is None for result in output):
            raise RuntimeError("attempt grouping left an empty lane")
        return [result for result in output if result is not None]

    def _dose_probe(
        self, selected: Sequence[RepresentationNode]
    ) -> tuple[list[AttemptResult], dict[str, Any] | None]:
        pair = self.dose_controller.probe_pair
        if pair is None or self.step_index % self.config.simulation_probe_interval:
            return [], None
        nodes = list(selected[: self.config.simulation_probe_lanes])
        if not nodes:
            return [], None
        seeds = [
            self.config.seed + 700_000_000 + self.step_index * 1009 + index
            for index in range(len(nodes))
        ]
        target = [self.target_u] * len(nodes)
        low_dose, high_dose = pair
        low = self._attempt_grouped(nodes, target, seeds, [low_dose] * len(nodes))
        high = self._attempt_grouped(nodes, target, seeds, [high_dose] * len(nodes))
        report = self.dose_controller.observe(low_dose, high_dose, low, high)
        return [*low, *high], report

    def step(self) -> dict[str, Any]:
        if self.finished:
            return {"finished": True, "target_u": self.target_u}
        self.step_index += 1
        refresh_ids = self.refresh.choose(
            self.nodes,
            self.heap,
            top=self.config.refresh_top,
            fair=self.config.refresh_fair,
        )
        self._rescore(refresh_ids)

        selected: list[RepresentationNode] = []
        for _ in range(min(self.config.parallel_searches, len(self.nodes))):
            try:
                selected.append(self.heap.pop(self.nodes))
            except IndexError:
                break
        if not selected:
            return {"finished": True, "target_u": self.target_u}
        seeds = [
            self.config.seed + self.step_index * 1_000_003 + index * 100_003
            for index in range(len(selected) * self.config.attempts_per_node)
        ]
        lanes = [node for node in selected for _ in range(self.config.attempts_per_node)]
        scientific_target = self.target_u
        targets = self.outcome_controller.targets(
            scientific_target,
            len(lanes),
            self.config.strict_search_fraction,
        )
        # Rotate the strict/relaxed assignment so representation rank is not a
        # hidden treatment variable.
        if len(targets) > 1:
            shift = self.step_index % len(targets)
            targets = targets[shift:] + targets[:shift]
        doses = [self.dose_controller.current] * len(lanes)
        attempts = self._attempt_grouped(lanes, targets, seeds, doses)
        probe_attempts, dose_probe = self._dose_probe(selected)
        all_attempts = [*attempts, *probe_attempts]
        self.native_attempts_at_target += sum(
            result.attempt_target_u == scientific_target for result in all_attempts
        )
        for node in selected:
            node.attempts += self.config.attempts_per_node
            self.heap.update(node)

        verified: list[UnknotWitness] = []
        for result in all_attempts:
            if result.solved:
                verified.append(self._verified_full_witness(result))
        ratchet_eligible = [
            witness for witness in verified if witness.crossing_changes <= scientific_target
        ]
        if ratchet_eligible:
            best = min(
                ratchet_eligible,
                key=lambda witness: (
                    witness.crossing_changes,
                    1000 * witness.crossing_changes + witness.moves,
                ),
            )
            self._ratchet(best)

        self.outcome_controller.observe([result.solved for result in attempts])
        admitted_attempts, pending_negatives = self._admit_training_attempts(all_attempts)
        trained = self.backend.train_native(
            admitted_attempts,
            rehearsal_fraction=self.rehearsal_fraction,
        )
        self.native_train_steps += trained

        retention_rate = None
        if self.step_index % self.config.retention_probe_interval == 0:
            retention_rate = self.backend.retention_rate(
                simulations=self.config.simulations,
                seed=self.config.seed + 800_000_000,
            )
            if retention_rate is not None and retention_rate < self.config.retention_target:
                self.rehearsal_fraction = min(
                    self.config.max_rehearsal_fraction,
                    self.rehearsal_fraction + self.config.rehearsal_fraction_step,
                )

        children: list[str] = []
        pruned_nodes = 0
        if not self.finished:
            for node in selected:
                if node.node_id not in self.nodes or node.crossing_distance >= self.target_u:
                    continue
                for child in one_crossing_change_children(
                    node,
                    self.backend.spec,
                    node_budget=self.config.expansion_node_budget,
                    growth=self.config.expansion_growth,
                    diagram_limit=self.config.expansion_diagrams,
                    child_limit=self.config.expansion_children,
                ):
                    if self._add_node(child):
                        children.append(child.node_id)
            self._rescore(children)
            pruned_nodes = self._prune_live_nodes()

        distilled = 0
        if (
            self.native_attempts_at_target >= self.config.distill_after_native_attempts
            and not ratchet_eligible
        ):
            allowance = self._distillation_allowance()
            if allowance:
                distilled = self.backend.distill(self.distillation, allowance)
                self.distilled_train_steps += distilled
                self._rescore(refresh_ids)

        event = {
            "type": "search_step",
            "step": self.step_index,
            "target_u": self.target_u,
            "scientific_target_before_step": scientific_target,
            "attempt_target_us": targets,
            "selected": [node.node_id for node in selected],
            "attempts": len(attempts),
            "probe_attempts": len(probe_attempts),
            "strict_attempts": sum(
                result.attempt_target_u == scientific_target for result in all_attempts
            ),
            "relaxed_attempts": sum(
                result.attempt_target_u != scientific_target for result in all_attempts
            ),
            "verified_solutions": len(verified),
            "strict_verified_solutions": len(ratchet_eligible),
            "training_only_solutions": len(verified) - len(ratchet_eligible),
            "admitted_training_attempts": len(admitted_attempts),
            "pending_unconfirmed_negatives": pending_negatives,
            "positive_fraction": self.outcome_controller.positive_fraction,
            "training_budget_slack": self.outcome_controller.slack,
            "simulations": self.dose_controller.current,
            "simulation_probe": dose_probe,
            "scheduled_network_evaluations": sum(
                result.scheduled_network_evaluations for result in all_attempts
            ),
            "search_wall_seconds": sum(result.wall_seconds for result in all_attempts),
            "retention_rate": retention_rate,
            "rehearsal_fraction": self.rehearsal_fraction,
            "new_nodes": len(children),
            "pruned_nodes": pruned_nodes,
            "live_nodes": len(self.nodes),
            "native_train_steps": trained,
            "training_metrics": dict(getattr(self.backend, "last_train_metrics", {})),
            "distilled_train_steps": distilled,
        }
        self.events.append(event)
        return event

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "config": asdict(self.config),
            "target_u": self.target_u,
            "step_index": self.step_index,
            "native_attempts_at_target": self.native_attempts_at_target,
            "native_train_steps": self.native_train_steps,
            "distilled_train_steps": self.distilled_train_steps,
            "rehearsal_fraction": self.rehearsal_fraction,
            "outcome_controller": self.outcome_controller.to_dict(),
            "dose_controller": self.dose_controller.to_dict(),
            "negative_trials": {
                key: sorted(seeds) for key, seeds in sorted(self.negative_trials.items())
            },
            "admitted_negative_keys": sorted(self.admitted_negative_keys),
            "roots": {
                root_id: state.to_dict() for root_id, state in sorted(self.root_states.items())
            },
            "nodes": [
                asdict(node) for node in sorted(self.nodes.values(), key=lambda node: node.node_id)
            ],
            "best_witness": self.best_witness.to_dict() if self.best_witness else None,
            "events": self.events,
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        temporary = directory / "state.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, directory / "state.json")
        if self.best_witness is not None:
            witness_path = directory / (
                f"witness-u{self.best_witness.crossing_changes}-"
                f"l1000-{1000 * self.best_witness.crossing_changes + self.best_witness.moves}.json"
            )
            if not witness_path.exists():
                witness_path.write_text(
                    json.dumps(self.best_witness.to_dict(), indent=2, sort_keys=True) + "\n"
                )
        self.backend.save(directory / "scientist-state.pt.gz")


def _observation_tensor(observations: list[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack(observations)).permute(0, 3, 1, 2).float().to(device)


class ScientistMasteryBackend:
    """Factorized-head scorer, batched MCTS lanes, and online trainer."""

    def __init__(
        self,
        scientist: Scientist,
        config: MasteryConfig,
        *,
        rehearsal: Sequence[DistillationExample] = (),
        inference_coordinator: PersistentInferenceCoordinator | None = None,
        inference_timeout_seconds: float | None = None,
    ) -> None:
        if not scientist.prediction_source.startswith("factorized"):
            raise ValueError("single-knot mastery requires a trained factorized p-head")
        if not scientist.config.game.objective_budget_channel:
            raise ValueError("single-knot mastery requires the objective-budget channel")
        if config.protocol_version == 2:
            if not hasattr(scientist.network, "auxiliary_solve_backprop"):
                raise ValueError("factorized checkpoint cannot expose solve loss to its encoder")
            # Failure policies remain masked, but the balanced negative solve
            # loss intentionally shapes the shared representation in v2.
            scientist.network.auxiliary_solve_backprop = True
            if float(getattr(scientist.network, "auxiliary_loss_weight", 0.0)) <= 0.0:
                raise ValueError("single-knot mastery requires a positive solve-head loss weight")
        self.scientist = scientist
        self.config = config
        self.spec = scientist.game.config._spec
        self.rehearsal = tuple(rehearsal)
        self.inference_coordinator = inference_coordinator
        self.inference_timeout_seconds = inference_timeout_seconds
        self.rehearsal_ids: set[str] = set()
        self.last_train_metrics: dict[str, float] = {}
        self.last_retention_report: dict[str, Any] | None = None
        self._seed_rehearsal()

    def _seed_rehearsal(self) -> None:
        """Put verified inherited solutions in replay without taking an update."""
        for index, example in enumerate(self.rehearsal):
            witness = example.witness
            witness.verify()
            identity = f"rehearsal:{example.source}:{witness.instance_id}"
            knot = KnotItem(
                identity,
                len(witness.start.word),
                witness.start.word,
                witness.start.strands,
            )
            actions = [step.action.to_flat(self.spec) for step in witness.steps]
            record = translate_semantic_record(
                self.scientist,
                knot,
                self.config.ratio,
                actions,
                seed=self.config.seed + 850_000_000 + index,
            )
            if record is None:
                continue
            self.scientist.replay.add(
                record,
                representation_id=identity,
                objective_ratio=self.config.ratio,
            )
            self.rehearsal_ids.add(identity)

    def _item(self, node: RepresentationNode) -> KnotItem:
        return KnotItem(node.node_id, len(node.word), node.word, node.strands)

    def _objective_cap(self, remaining: int, seed: int | None = None) -> float:
        base = (
            self.config.move_allowance
            if self.config.objective_move_base is None
            else self.config.objective_move_base
        )
        jitter = 0
        if self.config.objective_move_jitter:
            if seed is None:
                jitter = self.config.objective_move_jitter // 2
            else:
                jitter = int(
                    np.random.default_rng(int(seed) ^ 0x5A17).integers(
                        0, self.config.objective_move_jitter + 1
                    )
                )
        return self.config.ratio * remaining + base + jitter

    @torch.inference_mode()
    def score(self, nodes: Sequence[RepresentationNode], target_u: int) -> list[NodeScore]:
        observations = []
        live = []
        for node in nodes:
            remaining = target_u - node.crossing_distance
            if remaining < 0:
                live.append(False)
                observations.append(None)
                continue
            cap = self._objective_cap(remaining)
            game = FixedWordGame(
                self.scientist.game,
                self._item(node),
                self.config.ratio,
                objective_cap=cap,
                cap_type="remaining-crossing-budget",
            )
            observations.append(game.reset(0).observation)
            live.append(True)
        active = [observation for observation in observations if observation is not None]
        if not active:
            return [NodeScore(0.0, float("inf"), float("inf")) for _ in nodes]
        device = torch.device(self.scientist.config.train.device)
        self.scientist.network.eval()
        tensor = _observation_tensor(active, device)
        if self.inference_coordinator is None:
            _, _, auxiliary = self.scientist.network.forward_with_auxiliary(tensor)
        else:
            _, _, auxiliary = self.inference_coordinator.infer(
                tensor, mode="auxiliary"
            )
        if auxiliary is None:
            raise ValueError("checkpoint has no factorized auxiliary outputs")
        solve_logits, crossings, moves = auxiliary
        probability = calibrated_solve_probability(self.scientist, solve_logits).cpu().numpy()
        predicted_crossings = crossings.mean(dim=1).cpu().numpy()
        predicted_moves = moves.mean(dim=1).cpu().numpy()
        result = []
        cursor = 0
        for is_live in live:
            if not is_live:
                result.append(NodeScore(0.0, float("inf"), float("inf")))
            else:
                result.append(
                    NodeScore(
                        float(probability[cursor]),
                        float(predicted_crossings[cursor]),
                        float(predicted_moves[cursor]),
                    )
                )
                cursor += 1
        return result

    def attempt_batch(
        self,
        nodes: Sequence[RepresentationNode],
        target_u: int,
        seeds: Sequence[int],
        *,
        simulations: int | None = None,
    ) -> list[AttemptResult]:
        if len(nodes) != len(seeds):
            raise ValueError("one seed is required per MCTS lane")
        fixed_games = []
        transitions = []
        rngs = []
        records: list[GameRecord] = [[] for _ in nodes]
        for node, seed in zip(nodes, seeds, strict=True):
            remaining = target_u - node.crossing_distance
            cap = self._objective_cap(remaining, int(seed))
            fixed = FixedWordGame(
                self.scientist.game,
                self._item(node),
                self.config.ratio,
                objective_cap=cap,
                cap_type="remaining-crossing-budget",
            )
            fixed_games.append(fixed)
            transitions.append(fixed.reset(int(seed)))
            rngs.append(np.random.default_rng(seed))
        search_config = self.scientist.config.search
        search_config = type(search_config)(
            **{
                **asdict(search_config),
                "simulations": int(simulations or self.config.simulations),
            }
        )
        inference_network = (
            self.scientist.network
            if self.inference_coordinator is None
            else CoordinatedPolicyValueNet(
                self.inference_coordinator,
                dose=int(simulations or self.config.simulations),
                timeout=self.inference_timeout_seconds,
            )
        )
        search = NeuralMCTS(
            fixed_games[0],
            inference_network,
            search_config,
            self.scientist.config.train.device,
        )
        moves = [0] * len(nodes)
        first_roles = [
            game.first_role_player(transition.state)
            for game, transition in zip(fixed_games, transitions, strict=True)
        ]
        started = time.perf_counter()
        while True:
            active = [
                index for index, transition in enumerate(transitions) if not transition.terminated
            ]
            if not active:
                break
            searched = search.run_batch(
                states=[transitions[index].state for index in active],
                observations=[transitions[index].observation for index in active],
                legal_actions=[transitions[index].legal_actions for index in active],
                rngs=[rngs[index] for index in active],
                temperatures=[
                    1.0 if moves[index] < self.scientist.config.train.temperature_moves else 0.0
                    for index in active
                ],
                add_root_noise=True,
            )
            for index, result in zip(active, searched, strict=True):
                transition = transitions[index]
                game = fixed_games[index]
                raw = game.unwrap(transition.state)
                position = Position(
                    observation=transition.observation,
                    legal_actions=transition.legal_actions,
                    policy=result.policy.astype(np.float32),
                    action=int(result.action),
                    player=transition.player,
                    role=0 if transition.player == first_roles[index] else 1,
                    episode_seed=int(seeds[index]),
                    value_potential=game.value_potential(transition.state, transition.player),
                    representation_id=nodes[index].node_id,
                    objective_ratio=self.config.ratio,
                    objective_cap=float(game.objective_cap),
                    action_horizon=int(game.config.simplify_budget),
                    residual_word_length=int(np.count_nonzero(np.asarray(raw._word))),
                    mcts_root_value=float(result.root_value),
                    mcts_visit_count=int(np.asarray(result.visits).sum()),
                    episode_position_index=len(records[index]),
                )
                nxt = game.step(transition.state, result.action)
                position.reward = search.edge_reward(transition.state, transition.player, nxt)
                position.next_terminated = nxt.terminated
                records[index].append(position)
                transitions[index] = nxt
                moves[index] += 1

        batch_wall_seconds = time.perf_counter() - started
        output = []
        dose = int(simulations or self.config.simulations)
        for node, game, transition, record in zip(
            nodes, fixed_games, transitions, records, strict=True
        ):
            raw = game.unwrap(transition.state)
            solved = bool((np.asarray(raw._word) == 0).all()) and int(np.asarray(raw._n)) == 1
            rewards = game.final_rewards(transition.state)
            crossing_changes = int(np.asarray(raw._crossing_changes))
            final_moves = game.semantic_move_count(transition.state)
            reason = transition.termination_reason
            for position in record:
                position.outcome = float(rewards[position.player])
                position.solved = float(solved)
                position.final_crossing_changes = float(crossing_changes)
                position.final_moves = float(final_moves)
                position.final_native_plies = float(game.native_ply_count(transition.state))
                position.final_internal_plies = float(game.internal_ply_count(transition.state))
                position.objective_censored = reason == "objective_budget_exhausted"
                position.termination_reason = reason
            verified = verified_record_cost(
                self.scientist.game,
                self._item(node),
                self.config.ratio,
                record,
            )
            if verified is None:
                output.append(
                    AttemptResult(
                        node.node_id,
                        False,
                        record,
                        reason=reason,
                        scheduled_network_evaluations=len(record) * (dose + 1),
                        wall_seconds=batch_wall_seconds / len(nodes),
                    )
                )
            else:
                cc, semantic_moves, actions = verified
                output.append(
                    AttemptResult(
                        node.node_id,
                        True,
                        record,
                        tuple(actions),
                        cc,
                        semantic_moves,
                        "verified",
                        scheduled_network_evaluations=len(record) * (dose + 1),
                        wall_seconds=batch_wall_seconds / len(nodes),
                    )
                )
        return output

    def _train(self, steps: int) -> int:
        completed = 0
        for _ in range(steps):
            if self.scientist.replay.position_count < 1:
                break
            self.last_train_metrics = train_alphazero_step(
                self.scientist.network,
                self.scientist.optimizer,
                self.scientist.replay,
                min(self.config.train_batch_size, self.scientist.replay.position_count),
                torch.device(self.scientist.config.train.device),
                policy_value_success_only=True,
            )
            completed += 1
        return completed

    def train_native(
        self,
        attempts: Sequence[AttemptResult],
        *,
        rehearsal_fraction: float,
    ) -> int:
        for attempt in attempts:
            if attempt.native_record:
                self.scientist.replay.add(
                    attempt.native_record,
                    representation_id=attempt.node_id,
                    objective_ratio=self.config.ratio,
                )
        if self.config.protocol_version == 1:
            return self._train(self.config.train_steps_per_batch)
        completed = 0
        current = attempts[0].node_id if attempts else ""
        for _ in range(self.config.train_steps_per_batch):
            if self.scientist.replay.position_count < 1:
                break
            self.last_train_metrics = train_alphazero_step(
                self.scientist.network,
                self.scientist.optimizer,
                self.scientist.replay,
                min(self.config.train_batch_size, self.scientist.replay.position_count),
                torch.device(self.scientist.config.train.device),
                policy_value_success_only=True,
                continual_replay=True,
                replay_current_representation=current,
                replay_rehearsal_representations=self.rehearsal_ids,
                replay_rehearsal_fraction=rehearsal_fraction,
                replay_positions_per_episode=4,
            )
            completed += 1
        return completed

    def retention_rate(self, *, simulations: int, seed: int) -> float | None:
        if not self.rehearsal:
            self.last_retention_report = None
            return None
        solved = 0
        rows = []
        for index, example in enumerate(self.rehearsal):
            witness = example.witness
            node = RepresentationNode.create(
                root_id=f"retention-{index}",
                word=witness.start.word,
                strands=witness.start.strands,
                crossing_distance=0,
                semantic_path=(),
                provenance={"source": example.source, "frozen_rehearsal": True},
            )
            result = self.attempt_batch(
                [node],
                witness.crossing_changes,
                [seed + index * 1009],
                simulations=simulations,
            )[0]
            solved += int(result.solved)
            rows.append(
                {
                    "source": example.source,
                    "instance_id": witness.instance_id,
                    "strands": witness.start.strands,
                    "historical_target_u": witness.crossing_changes,
                    "solved": bool(result.solved),
                    "reason": result.reason,
                }
            )
        by_strands = {}
        for strands in sorted({int(row["strands"]) for row in rows}):
            group = [row for row in rows if int(row["strands"]) == strands]
            by_strands[str(strands)] = {
                "solved": sum(bool(row["solved"]) for row in group),
                "total": len(group),
                "rate": sum(bool(row["solved"]) for row in group) / len(group),
            }
        rate = solved / len(self.rehearsal)
        self.last_retention_report = {
            "rate": rate,
            "solved": solved,
            "total": len(rows),
            "by_strands": by_strands,
            "rows": rows,
        }
        return rate

    def distill(self, examples: Sequence[DistillationExample], max_steps: int) -> int:
        admitted = 0
        for example in examples:
            witness = example.witness
            witness.verify()
            knot = KnotItem(
                witness.instance_id,
                len(witness.start.word),
                witness.start.word,
                witness.start.strands,
            )
            actions = [step.action.to_flat(self.spec) for step in witness.steps]
            record = translate_semantic_record(
                self.scientist,
                knot,
                self.config.ratio,
                actions,
                seed=self.config.seed + 900_000_000 + admitted,
            )
            if record is None:
                continue
            self.scientist.replay.add(
                record,
                representation_id=knot.name,
                objective_ratio=self.config.ratio,
            )
            admitted += 1
            if admitted >= max_steps:
                break
        return self._train(min(max_steps, admitted))

    def save(self, path: Path) -> None:
        payload = {
            "schema": "single-knot-mastery-scientist-v1",
            "scientist": self.scientist.name,
            "network": self.scientist.network.state_dict(),
            "optimizer": self.scientist.optimizer.state_dict(),
            "replay": self.scientist.replay,
            "rehearsal_ids": sorted(self.rehearsal_ids),
            "source_checkpoint": str(self.scientist.checkpoint),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with gzip.open(temporary_path, "wb", compresslevel=1) as handle:
                torch.save(payload, handle)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def restore(self, path: Path) -> None:
        with gzip.open(path, "rb") as handle:
            payload = torch.load(
                handle,
                map_location=self.scientist.config.train.device,
                weights_only=False,
            )
        if payload.get("schema") != "single-knot-mastery-scientist-v1":
            raise ValueError(f"unsupported scientist state schema in {path}")
        if payload.get("scientist") != self.scientist.name:
            raise ValueError("saved scientist name does not match the requested scientist")
        self.scientist.network.load_state_dict(payload["network"])
        self.scientist.optimizer.load_state_dict(payload["optimizer"])
        self.scientist.replay = payload["replay"]
        self.rehearsal_ids = set(payload.get("rehearsal_ids", self.rehearsal_ids))


def load_distillation(path: Path | None) -> list[DistillationExample]:
    if path is None:
        return []
    payload = json.loads(path.read_text())
    rows = payload.get("witnesses", payload if isinstance(payload, list) else [])
    examples = []
    for index, row in enumerate(rows):
        witness_row = row.get("witness", row)
        witness = UnknotWitness.from_dict(witness_row)
        witness.verify()
        examples.append(DistillationExample(str(row.get("source", f"row-{index}")), witness))
    return sorted(examples, key=lambda example: (example.l10, example.source))


def load_catalogue_target(path: Path, knot_name: str) -> tuple[MasteryConfig, BraidState, dict]:
    payload = json.loads(path.read_text())
    row = next(
        (item for item in payload["candidates"] if item["canonical_name"] == knot_name),
        None,
    )
    if row is None:
        raise ValueError(f"{knot_name!r} is not in {path}")
    if row["representation_status"] != "available":
        raise ValueError(f"{knot_name} has no stored local braid representation")
    rep = row["stored_representation"]
    interval = row.get("bound_interval", row.get("knotinfo_interval_at_snapshot"))
    if interval is None or len(interval) != 2:
        raise ValueError(f"{knot_name} has no two-endpoint bound interval")
    certified_lower_bound = int(row.get("certified_lower_bound", interval[0]))
    config = MasteryConfig(
        knot_name=knot_name,
        initial_target_u=int(row["strict_upper_bound_target"]),
        certified_lower_bound=certified_lower_bound,
    )
    provenance = {
        "catalogue": str(path),
        "catalogue_schema": payload["schema"],
        "bound_interval": interval,
        "bound_provenance": row.get("bound_provenance"),
        "instance_id": rep["instance_id"],
    }
    if "knotinfo_interval_at_snapshot" in row:
        provenance["knotinfo_interval"] = row["knotinfo_interval_at_snapshot"]
    return config, BraidState(tuple(rep["word"]), int(rep["strands"])), provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--knot", required=True)
    parser.add_argument("--scientist", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distillation", type=Path)
    parser.add_argument(
        "--rehearsal-panel",
        type=Path,
        help="verified inherited witnesses used for replay and frozen retention probes",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--parallel-searches", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument(
        "--simulation-levels",
        default="",
        help="comma-separated paired MCTS doses; defaults to S,2S,4S",
    )
    parser.add_argument("--negative-confirmations", type=int, default=3)
    parser.add_argument("--target-positive-fraction", type=float, default=0.5)
    parser.add_argument("--strict-search-fraction", type=float, default=0.5)
    parser.add_argument("--action-horizon", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base, root, provenance = load_catalogue_target(args.catalogue, args.knot)
    state_path = args.output / "state.json"
    scientist_state_path = args.output / "scientist-state.pt.gz"
    if args.resume:
        if not state_path.exists() or not scientist_state_path.exists():
            raise ValueError("--resume requires state.json and scientist-state.pt.gz")
        saved = json.loads(state_path.read_text())
        config = _saved_config(saved)
        if config.knot_name != args.knot:
            raise ValueError("saved mastery state belongs to a different knot")
    else:
        simulation_levels = tuple(
            int(value) for value in args.simulation_levels.split(",") if value.strip()
        )
        config = MasteryConfig(
            **{
                **asdict(base),
                "parallel_searches": args.parallel_searches,
                "torch_threads": args.torch_threads,
                "simulations": args.simulations,
                "simulation_levels": simulation_levels,
                "move_allowance": args.action_horizon,
                "seed": args.seed,
                "negative_confirmations": args.negative_confirmations,
                "target_positive_fraction": args.target_positive_fraction,
                "strict_search_fraction": args.strict_search_fraction,
            }
        )
    torch.set_num_threads(config.torch_threads)
    torch.set_num_interop_threads(1)
    scientist = load_scientist(
        args.scientist,
        args.checkpoint,
        seed=config.seed,
        device=args.device,
        simulations=config.simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    game_config = replace(scientist.config.game, simplify_budget=config.move_allowance)
    scientist.config = replace(scientist.config, game=game_config)
    scientist.game = make_game(game_config)
    if root.strands > scientist.config.game.max_strands:
        raise ValueError(
            f"{args.knot} needs {root.strands} strands but {args.scientist} supports "
            f"{scientist.config.game.max_strands}"
        )
    distillation = load_distillation(args.distillation)
    rehearsal = load_distillation(args.rehearsal_panel)
    backend = ScientistMasteryBackend(scientist, config, rehearsal=rehearsal)
    if args.resume:
        backend.restore(scientist_state_path)
        coordinator = MasteryCoordinator.from_saved(
            state_path,
            backend,
            distillation=distillation,
        )
    else:
        if state_path.exists() or scientist_state_path.exists():
            raise ValueError("output already contains state; pass --resume or choose a new output")
        coordinator = MasteryCoordinator(
            config,
            backend,
            [(root, provenance)],
            distillation=distillation,
        )
        coordinator.save(args.output)
    for _ in range(args.steps):
        if coordinator.finished:
            break
        event = coordinator.step()
        print(json.dumps(event, sort_keys=True), flush=True)
        coordinator.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "target_u": coordinator.target_u,
                "finished": coordinator.finished,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
