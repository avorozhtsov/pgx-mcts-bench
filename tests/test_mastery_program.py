from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from rf_knots.actions import DESTABILIZE, ActionSpec
from rf_knots.evidence import BraidState

from pgx_mcts_bench.mastery_program import (
    EvidenceInventory,
    MasteryProgram,
    ProgramConfig,
    SequenceChallenge,
    load_sequence,
    load_short_ablation_sequence,
)
from pgx_mcts_bench.single_knot_mastery import AttemptResult, NodeScore


class FakeBackend:
    def __init__(self, *, solve: bool = False) -> None:
        self.spec = ActionSpec(16, 12)
        self.config = SimpleNamespace(simulations=8)
        self.solve = solve
        self.train_steps = 0

    def score(self, nodes, target_u):
        return [
            NodeScore(
                max(
                    0.01,
                    (0.9 if node.word == (1,) else 0.1) - 0.1 * node.attempts,
                ),
                float(max(target_u - node.crossing_distance, 0)),
                float(len(node.word)),
            )
            for node in nodes
        ]

    def attempt_batch(self, nodes, target_u, seeds, *, simulations=None):
        del target_u, simulations
        rows = []
        for node, seed in zip(nodes, seeds, strict=True):
            solved = self.solve and node.word == (1,) and node.strands == 2
            rows.append(
                AttemptResult(
                    node.node_id,
                    solved,
                    semantic_actions=(self.spec.encode(DESTABILIZE),) if solved else (),
                    crossing_changes=0 if solved else None,
                    moves=1 if solved else None,
                    reason="verified" if solved else "budget_exhausted",
                    seed=seed,
                    wall_seconds=0.01,
                )
            )
        return rows

    def train_native(self, attempts, *, rehearsal_fraction):
        assert 0.05 <= rehearsal_fraction <= 0.50
        trained = int(bool(attempts))
        self.train_steps += trained
        return trained

    def retention_rate(self, *, simulations, seed):
        del simulations, seed
        return None

    def save(self, path: Path):
        path.write_bytes(b"fake-scientist-state")


def challenge(index: int, *, upper: int = 2) -> SequenceChallenge:
    return SequenceChallenge(
        index,
        f"challenge-{index:03d}",
        f"K{index:03d}",
        f"representation-{index:03d}",
        BraidState((1,), 2),
        0,
        upper,
        upper - 1,
        20 + index,
        {"source": "test"},
    )


def program(tmp_path: Path, **changes) -> MasteryProgram:
    config = replace(
        ProgramConfig(
            "test-sequence",
            "test-scientist",
            bootstrap_challenges=1,
            max_active_challenges=2,
            parallel_searches=1,
            expansion_children=0,
            negative_confirmations=1,
            task_attempt_limit=3,
            challenge_attempt_limit=4,
        ),
        **changes,
    )
    return MasteryProgram(
        config,
        FakeBackend(),
        [challenge(index) for index in range(40)],
        EvidenceInventory(tmp_path / "inventory"),
        sequence_sha256="a" * 64,
    )


def test_load_sequence_requires_a_long_unique_curriculum(tmp_path: Path):
    rows = []
    for index in range(200):
        rows.append(
            {
                "challenge_id": f"c-{index}",
                "canonical_name": f"K{index}",
                "representation_id": f"r-{index}",
                "stored_representation": {"word": [1], "strands": 2},
                "bound_interval": [0, 2],
                "acs10": index,
            }
        )
    path = tmp_path / "sequence.json"
    path.write_text(
        json.dumps(
            {
                "schema": "multi-knot-mastery-sequence-v1",
                "name": "long-test",
                "challenges": rows,
            }
        )
    )
    name, loaded = load_sequence(path)
    assert name == "long-test"
    assert len(loaded) == 200
    rows[-1]["challenge_id"] = "c-0"
    path.write_text(
        json.dumps(
            {
                "schema": "multi-knot-mastery-sequence-v1",
                "name": "long-test",
                "challenges": rows,
            }
        )
    )
    try:
        load_sequence(path)
    except ValueError as error:
        assert "not unique" in str(error)
    else:
        raise AssertionError("duplicate challenge ids were accepted")


def test_short_ablation_sequence_has_distinct_fail_closed_schema(tmp_path: Path):
    rows = [
        {
            "challenge_id": f"short-{index}",
            "canonical_name": f"K{index}",
            "representation_id": f"short-r-{index}",
            "stored_representation": {"word": [1], "strands": 2},
            "bound_interval": [0, 2],
            "acs10": index,
        }
        for index in range(20)
    ]
    path = tmp_path / "short.json"
    path.write_text(
        json.dumps(
            {
                "schema": "q-skm-short-ablation-sequence-v1",
                "name": "paired-short-test",
                "challenges": rows,
            }
        )
    )
    assert len(load_short_ablation_sequence(path)[1]) == 20
    try:
        load_sequence(path)
    except ValueError as error:
        assert "unsupported sequence schema" in str(error)
    else:
        raise AssertionError("short ablation leaked into the SKM-240 loader")


def test_searched_task_is_rekeyed_and_can_be_visited_again(tmp_path: Path):
    mastery = program(tmp_path)
    first = mastery.step()
    second = mastery.step()
    assert first["attempts"] == 1
    assert second["attempts"] == 1
    assert sum(row.attempts for row in mastery.challenges.values()) == 2


def test_give_up_advances_sequence_without_erasing_state(tmp_path: Path):
    mastery = program(
        tmp_path,
        task_attempt_limit=1,
        challenge_attempt_limit=1,
        cooldown_steps=0,
    )
    event = mastery.step()
    assert event["gave_up"] == ["challenge-000"]
    assert mastery.challenges["challenge-000"].status == "gave-up-threshold"
    assert mastery.introduced_count >= 2


def test_verified_solution_enters_shared_inventory_and_ratchets(tmp_path: Path):
    config = ProgramConfig(
        "test-sequence",
        "solving-scientist",
        bootstrap_challenges=1,
        max_active_challenges=1,
        parallel_searches=1,
        expansion_children=0,
        negative_confirmations=1,
    )
    inventory = EvidenceInventory(tmp_path / "inventory")
    mastery = MasteryProgram(
        config,
        FakeBackend(solve=True),
        [challenge(0), challenge(1)],
        inventory,
        sequence_sha256="b" * 64,
    )
    event = mastery.step()
    assert event["improvements"][0]["new_upper"] == 0
    assert mastery.challenges["challenge-000"].status == "solved-to-lower-bound"
    evidence = list(inventory.witnesses.glob("*.json"))
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text())["crossing_changes"] == 0


def test_exact_group_checkpoint_and_resume(tmp_path: Path):
    mastery = program(tmp_path)
    # Direct admission is deterministic and isolates checkpoint durability from
    # search policy in this unit test.
    while mastery.introduced_count < 20:
        assert mastery._admit_next()
    output = tmp_path / "run"
    mastery.save(output)
    checkpoint = output / "checkpoints" / "group-020"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["introduced"] == 20
    restored = MasteryProgram.from_saved(
        output / "program-state.json",
        FakeBackend(),
        [challenge(index) for index in range(40)],
        EvidenceInventory(tmp_path / "inventory"),
        sequence_sha256="a" * 64,
    )
    assert restored.introduced_count == 20
    assert restored.next_sequence_index == 20


def test_rehearsal_fraction_tracks_old_target_retention(tmp_path: Path):
    class RetentionBackend(FakeBackend):
        retention = 1.0

        def retention_rate(self, *, simulations, seed):
            del simulations, seed
            return self.retention

    config = ProgramConfig(
        "test-sequence",
        "retention-scientist",
        bootstrap_challenges=1,
        max_active_challenges=2,
        parallel_searches=1,
        expansion_children=0,
        negative_confirmations=1,
        retention_probe_interval=1,
        min_rehearsal_fraction=0.05,
        initial_rehearsal_fraction=0.10,
        max_rehearsal_fraction=0.50,
    )
    backend = RetentionBackend()
    mastery = MasteryProgram(
        config,
        backend,
        [challenge(index) for index in range(40)],
        EvidenceInventory(tmp_path / "inventory"),
        sequence_sha256="e" * 64,
    )
    healthy = mastery.step()
    assert healthy["retention_rate"] == 1.0
    assert healthy["rehearsal_fraction"] == 0.05
    assert healthy["rehearsal_adjustment"] == "decreased-retention-healthy"
    backend.retention = 0.5
    degraded = mastery.step()
    assert degraded["retention_rate"] == 0.5
    assert degraded["rehearsal_fraction"] == 0.10
    assert degraded["rehearsal_adjustment"] == "increased-below-retention-target"


def test_knotinfo_frontier_injection_preserves_provenance_and_bonus(tmp_path: Path):
    config = ProgramConfig(
        "test-sequence",
        "frontier-scientist",
        bootstrap_challenges=10,
        max_active_challenges=16,
        parallel_searches=1,
        expansion_children=0,
        reservoir_probe_interval=1,
        reservoir_probe_batch=1,
    )
    candidate = replace(
        challenge(300),
        challenge_id="knotinfo-only-300",
        provenance={
            "evidence_source": "knotinfo-only",
            "knotinfo_evidence_l10": 31,
        },
    )
    mastery = MasteryProgram(
        config,
        FakeBackend(),
        [challenge(index) for index in range(40)],
        EvidenceInventory(tmp_path / "inventory"),
        sequence_sha256="c" * 64,
        reservoir=[candidate],
        reservoir_sha256="d" * 64,
    )
    event = mastery.step()
    injection = event["frontier_injections"][0]
    assert injection["type"] == "LOUD_KNOTINFO_FRONTIER_INJECTION"
    assert injection["raw_p_solve"] <= injection["heap_rank_10_raw_p_solve"]
    nodes = [
        node
        for node in mastery.nodes.values()
        if mastery.node_challenge[node.node_id] == candidate.challenge_id
    ]
    assert nodes
    assert all(node.priority_bonus == 0.10 for node in nodes)
    assert all(node.provenance["original_knot_name"] == candidate.knot_name for node in nodes)
    assert all(node.provenance["original_target_u"] == 1 for node in nodes)


def test_v2_hard_timeout_commits_an_outcome_and_attempt_journal(tmp_path: Path):
    class SleepingBackend(FakeBackend):
        def attempt_batch(self, nodes, target_u, seeds, *, simulations=None):
            del nodes, target_u, seeds, simulations
            time.sleep(0.2)
            raise AssertionError("deadline did not interrupt the backend")

    config = ProgramConfig(
        "test-sequence",
        "sleeping-scientist",
        bootstrap_challenges=1,
        max_active_challenges=1,
        parallel_searches=1,
        expansion_children=0,
        attempt_wall_seconds_limit=0.02,
        challenge_search_seconds_limit=1.0,
    )
    output = tmp_path / "run"
    mastery = MasteryProgram(
        config,
        SleepingBackend(),
        [challenge(0), challenge(1)],
        EvidenceInventory(tmp_path / "inventory"),
        sequence_sha256="f" * 64,
        runtime_root=output,
    )
    event = mastery.step()
    assert event["outcomes"] == {"hard_timeout": 1}
    journal = json.loads((output / "attempt-journal" / "current.json").read_text())
    assert journal["status"] == "hard-timeout"
