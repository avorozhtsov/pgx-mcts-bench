"""Resumable, proof-carrying mastery search for one knot.

The coordinator maintains a mutable priority queue of representations reached
from one or more verified starting braids.  A node at crossing-change distance
``d`` is scored by the scientist's factorized head under the remaining L1000
budget, approximating ``P(remaining crossing changes <= target_u - d)``.  The
heap stores ``-P`` so its smallest keys are the most promising nodes.

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
from collections import deque
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
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import train_alphazero_step

SCHEMA = "single-knot-mastery-v1"


def _stable_id(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True)
class MasteryConfig:
    knot_name: str
    initial_target_u: int
    certified_lower_bound: int = 0
    ratio: float = 1000.0
    move_allowance: int = 128
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

    def __post_init__(self) -> None:
        if self.initial_target_u < 0:
            raise ValueError("initial_target_u must be non-negative")
        if self.certified_lower_bound < 0:
            raise ValueError("certified_lower_bound must be non-negative")
        if self.parallel_searches < 1 or self.attempts_per_node < 1:
            raise ValueError("parallel searches and attempts must be positive")
        if self.torch_threads < 1:
            raise ValueError("torch_threads must be positive")
        if self.max_live_nodes < self.parallel_searches:
            raise ValueError("max_live_nodes must cover every parallel search lane")
        if not 0.0 <= self.max_distillation_fraction < 1.0:
            raise ValueError("max_distillation_fraction must be in [0, 1)")


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
    ) -> list[AttemptResult]: ...

    def train_native(self, attempts: Sequence[AttemptResult]) -> int: ...

    def distill(self, examples: Sequence[DistillationExample], max_steps: int) -> int: ...

    def save(self, path: Path) -> None: ...


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
            (-float(node.probability), predicted_l1000, node.last_scored_step, node.node_id,
             node.score_version),
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
                diagram.word[:position]
                + (-diagram.word[position],)
                + diagram.word[position + 1 :]
            )
            state = (word, diagram.strands)
            if state in seen:
                continue
            seen.add(state)
            children.append(
                RepresentationNode.create(
                    root_id=node.root_id,
                    word=word,
                    strands=diagram.strands,
                    crossing_distance=node.crossing_distance + 1,
                    semantic_path=(*node.semantic_path, *prefix, action),
                    parent_id=node.node_id,
                    provenance={
                        "operation": "type-preserving-prefix-then-one-crossing-change",
                        "parent": node.node_id,
                    },
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
        self.backend = backend
        self.target_u = config.initial_target_u
        self.step_index = 0
        self.native_attempts_at_target = 0
        self.native_train_steps = 0
        self.distilled_train_steps = 0
        self.best_witness: UnknotWitness | None = None
        self.nodes: dict[str, RepresentationNode] = {}
        self.root_states: dict[str, BraidState] = {}
        self.seen_states: set[tuple[str, tuple[int, ...], int, int]] = set()
        self.heap = MutableProbabilityHeap()
        self.refresh = FairRefreshScheduler()
        self.distillation = sorted(distillation, key=lambda item: (item.l10, item.source))
        self.events: list[dict[str, Any]] = []

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
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"unsupported mastery state schema in {state_path}")
        coordinator = cls.__new__(cls)
        coordinator.config = MasteryConfig(**payload["config"])
        coordinator.backend = backend
        coordinator.target_u = int(payload["target_u"])
        coordinator.step_index = int(payload["step_index"])
        coordinator.native_attempts_at_target = int(payload["native_attempts_at_target"])
        coordinator.native_train_steps = int(payload["native_train_steps"])
        coordinator.distilled_train_steps = int(payload["distilled_train_steps"])
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
        coordinator.distillation = sorted(
            distillation, key=lambda item: (item.l10, item.source)
        )
        coordinator.events = list(payload.get("events", []))
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
        self.nodes = {
            node_id: node for node_id, node in self.nodes.items() if node_id in keep
        }
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
        lanes = [
            node
            for node in selected
            for _ in range(self.config.attempts_per_node)
        ]
        attempts = self.backend.attempt_batch(lanes, self.target_u, seeds)
        self.native_attempts_at_target += len(attempts)
        for node in selected:
            node.attempts += self.config.attempts_per_node
            self.heap.update(node)

        verified: list[UnknotWitness] = []
        for result in attempts:
            if result.solved:
                verified.append(self._verified_full_witness(result))
        if verified:
            best = min(
                verified,
                key=lambda witness: (
                    witness.crossing_changes,
                    1000 * witness.crossing_changes + witness.moves,
                ),
            )
            self._ratchet(best)

        trained = self.backend.train_native(attempts)
        self.native_train_steps += trained

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
            and not verified
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
            "selected": [node.node_id for node in selected],
            "attempts": len(attempts),
            "verified_solutions": len(verified),
            "new_nodes": len(children),
            "pruned_nodes": pruned_nodes,
            "live_nodes": len(self.nodes),
            "native_train_steps": trained,
            "distilled_train_steps": distilled,
        }
        self.events.append(event)
        return event

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "target_u": self.target_u,
            "step_index": self.step_index,
            "native_attempts_at_target": self.native_attempts_at_target,
            "native_train_steps": self.native_train_steps,
            "distilled_train_steps": self.distilled_train_steps,
            "roots": {
                root_id: state.to_dict() for root_id, state in sorted(self.root_states.items())
            },
            "nodes": [
                asdict(node)
                for node in sorted(self.nodes.values(), key=lambda node: node.node_id)
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

    def __init__(self, scientist: Scientist, config: MasteryConfig) -> None:
        if not scientist.prediction_source.startswith("factorized"):
            raise ValueError("single-knot mastery requires a trained factorized p-head")
        if not scientist.config.game.objective_budget_channel:
            raise ValueError("single-knot mastery requires the objective-budget channel")
        self.scientist = scientist
        self.config = config
        self.spec = scientist.game.config._spec

    def _item(self, node: RepresentationNode) -> KnotItem:
        return KnotItem(node.node_id, len(node.word), node.word, node.strands)

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
            cap = self.config.ratio * remaining + self.config.move_allowance
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
        _, _, auxiliary = self.scientist.network.forward_with_auxiliary(
            _observation_tensor(active, device)
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
    ) -> list[AttemptResult]:
        if len(nodes) != len(seeds):
            raise ValueError("one seed is required per MCTS lane")
        fixed_games = []
        transitions = []
        rngs = []
        records: list[GameRecord] = [[] for _ in nodes]
        for node, seed in zip(nodes, seeds, strict=True):
            remaining = target_u - node.crossing_distance
            cap = self.config.ratio * remaining + self.config.move_allowance
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
            **{**asdict(search_config), "simulations": self.config.simulations}
        )
        search = NeuralMCTS(
            fixed_games[0],
            self.scientist.network,
            search_config,
            self.scientist.config.train.device,
        )
        moves = [0] * len(nodes)
        first_roles = [
            game.first_role_player(transition.state)
            for game, transition in zip(fixed_games, transitions, strict=True)
        ]
        while True:
            active = [
                index
                for index, transition in enumerate(transitions)
                if not transition.terminated
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

        output = []
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
                output.append(AttemptResult(node.node_id, False, record, reason=reason))
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
                    )
                )
        return output

    def _train(self, steps: int) -> int:
        completed = 0
        for _ in range(steps):
            if self.scientist.replay.position_count < 1:
                break
            train_alphazero_step(
                self.scientist.network,
                self.scientist.optimizer,
                self.scientist.replay,
                min(self.config.train_batch_size, self.scientist.replay.position_count),
                torch.device(self.scientist.config.train.device),
                policy_value_success_only=True,
            )
            completed += 1
        return completed

    def train_native(self, attempts: Sequence[AttemptResult]) -> int:
        for attempt in attempts:
            if attempt.native_record:
                self.scientist.replay.add(
                    attempt.native_record,
                    representation_id=attempt.node_id,
                    objective_ratio=self.config.ratio,
                )
        return self._train(self.config.train_steps_per_batch)

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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--parallel-searches", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=128)
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
        config = MasteryConfig(**saved["config"])
        if config.knot_name != args.knot:
            raise ValueError("saved mastery state belongs to a different knot")
    else:
        config = MasteryConfig(
            **{
                **asdict(base),
                "parallel_searches": args.parallel_searches,
                "torch_threads": args.torch_threads,
                "simulations": args.simulations,
                "move_allowance": args.action_horizon,
                "seed": args.seed,
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
    backend = ScientistMasteryBackend(scientist, config)
    distillation = load_distillation(args.distillation)
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
    print(json.dumps({"output": str(args.output), "target_u": coordinator.target_u,
                      "finished": coordinator.finished}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
