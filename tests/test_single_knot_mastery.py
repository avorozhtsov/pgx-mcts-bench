from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from rf_knots.actions import DESTABILIZE, ActionSpec
from rf_knots.evidence import BraidState, UnknotWitness

from pgx_mcts_bench.single_knot_mastery import (
    AttemptResult,
    DistillationExample,
    FairRefreshScheduler,
    MasteryConfig,
    MasteryCoordinator,
    MutableProbabilityHeap,
    NodeScore,
    RepresentationNode,
    equivalent_representations,
    one_crossing_change_children,
)


class FakeBackend:
    def __init__(self, *, solve_unknot_representation: bool = False):
        self.spec = ActionSpec(16, 5)
        self.solve_unknot_representation = solve_unknot_representation
        self.native_training = 0
        self.distilled = 0

    def score(self, nodes, target_u):
        del target_u
        return [
            NodeScore(0.9 if node.word == (1,) else 0.1, 1.0, float(len(node.word)))
            for node in nodes
        ]

    def attempt_batch(self, nodes, target_u, seeds):
        del target_u, seeds
        rows = []
        for node in nodes:
            if self.solve_unknot_representation and node.word == (1,) and node.strands == 2:
                rows.append(
                    AttemptResult(
                        node.node_id,
                        True,
                        semantic_actions=(self.spec.encode(DESTABILIZE),),
                        crossing_changes=0,
                        moves=1,
                        reason="verified",
                    )
                )
            else:
                rows.append(AttemptResult(node.node_id, False, reason="budget_exhausted"))
        return rows

    def train_native(self, attempts):
        self.native_training += bool(attempts)
        return int(bool(attempts))

    def distill(self, examples, max_steps):
        used = min(len(examples), max_steps)
        self.distilled += used
        return used

    def save(self, path: Path):
        del path


def config(**changes):
    base = MasteryConfig(
        knot_name="test",
        initial_target_u=2,
        parallel_searches=1,
        initial_equivalent_representations=1,
        equivalent_node_budget=10,
        expansion_node_budget=20,
        expansion_diagrams=2,
        expansion_children=0,
        refresh_top=1,
        refresh_fair=1,
        distill_after_native_attempts=100,
    )
    return replace(base, **changes)


def test_mutable_heap_moves_changed_probability_both_directions():
    heap = MutableProbabilityHeap()
    left = RepresentationNode.create(
        root_id="root", word=(1,), strands=2, crossing_distance=0, semantic_path=()
    )
    right = RepresentationNode.create(
        root_id="root", word=(1, 1, 1), strands=2, crossing_distance=0, semantic_path=()
    )
    nodes = {left.node_id: left, right.node_id: right}
    left.probability, right.probability = 0.8, 0.2
    heap.update(left)
    heap.update(right)
    assert heap.pop(nodes).node_id == left.node_id

    left.probability, right.probability = 0.1, 0.9
    heap.update(left)
    heap.update(right)
    assert heap.pop(nodes).node_id == right.node_id


def test_fair_refresh_eventually_visits_every_node():
    heap = MutableProbabilityHeap()
    scheduler = FairRefreshScheduler()
    nodes = {}
    for index in range(7):
        node = RepresentationNode.create(
            root_id="root",
            word=(1,) * (2 * index + 1),
            strands=2,
            crossing_distance=0,
            semantic_path=(),
        )
        node.probability = 1.0 if index == 0 else 0.0
        nodes[node.node_id] = node
        heap.update(node)
    visited = set()
    for _ in range(7):
        visited.update(scheduler.choose(nodes, heap, top=1, fair=1))
    assert visited == set(nodes)


def test_equivalent_seeds_and_one_cc_children_have_replayable_paths():
    spec = ActionSpec(16, 5)
    start = BraidState((1, 1, 1), 2)
    equivalents = equivalent_representations(start, spec, count=4, node_budget=100)
    assert len(equivalents) == 4
    for state, path in equivalents:
        current = start
        for action in path:
            from rf_knots.reference import apply

            word, strands = apply(spec, current.word, current.strands, action)
            current = BraidState(word, strands)
        assert current == state

    node = RepresentationNode.create(
        root_id="root",
        word=start.word,
        strands=start.strands,
        crossing_distance=0,
        semantic_path=(),
    )
    children = one_crossing_change_children(
        node,
        spec,
        node_budget=100,
        growth=2,
        diagram_limit=4,
        child_limit=20,
    )
    assert children
    assert all(child.crossing_distance == 1 for child in children)
    assert all(
        sum(spec.decode(action)[0] == 8 for action in child.semantic_path) == 1
        for child in children
    )


def test_verified_solution_ratchets_target_and_stops_at_lower_bound():
    backend = FakeBackend(solve_unknot_representation=True)
    coordinator = MasteryCoordinator(
        config(initial_target_u=1, certified_lower_bound=0),
        backend,
        [(BraidState((1,), 2), {"source": "test"})],
    )

    event = coordinator.step()

    assert event["verified_solutions"] == 1
    assert coordinator.best_witness is not None
    assert coordinator.best_witness.crossing_changes == 0
    assert coordinator.target_u == -1
    assert coordinator.finished
    assert coordinator.events[0]["type"] == "verified_upper_bound"


def test_distillation_is_delayed_and_fraction_limited():
    backend = FakeBackend()
    witness = UnknotWitness.from_actions(
        (1,), 2, backend.spec, [backend.spec.encode(DESTABILIZE)]
    )
    coordinator = MasteryCoordinator(
        config(
            distill_after_native_attempts=2,
            max_distillation_fraction=0.5,
        ),
        backend,
        [(BraidState((1, 1, 1), 2), {"source": "test"})],
        distillation=[DistillationExample("simple-first", witness)],
    )

    first = coordinator.step()
    second = coordinator.step()

    assert first["distilled_train_steps"] == 0
    assert second["distilled_train_steps"] == 1
    assert coordinator.distilled_train_steps <= coordinator.native_train_steps


def test_saved_coordinator_resumes_heap_and_counters(tmp_path):
    backend = FakeBackend()
    coordinator = MasteryCoordinator(
        config(),
        backend,
        [(BraidState((1, 1, 1), 2), {"source": "test"})],
    )
    coordinator.step()
    payload = coordinator.to_dict()
    state = tmp_path / "state.json"
    import json

    state.write_text(json.dumps(payload))

    resumed = MasteryCoordinator.from_saved(state, FakeBackend())

    assert resumed.step_index == coordinator.step_index
    assert resumed.native_train_steps == coordinator.native_train_steps
    assert resumed.target_u == coordinator.target_u
    assert set(resumed.nodes) == set(coordinator.nodes)
    assert resumed.step()["attempts"] == 1


def test_live_heap_is_bounded_by_probability():
    backend = FakeBackend()
    coordinator = MasteryCoordinator(
        config(
            expansion_children=20,
            expansion_diagrams=4,
            max_live_nodes=5,
        ),
        backend,
        [(BraidState((1, 1, 1), 2), {"source": "test"})],
    )

    event = coordinator.step()

    assert event["pruned_nodes"] > 0
    assert event["live_nodes"] == 5
    assert len(coordinator.nodes) == 5
