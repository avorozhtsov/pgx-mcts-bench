import pickle

import numpy as np
import pytest

from pgx_mcts_bench.data import Position, ReplayBuffer


def _position(terminal: bool) -> Position:
    return Position(
        observation=np.zeros((6, 6, 11), dtype=np.float32),
        legal_actions=np.ones(37, dtype=bool),
        policy=np.full(37, 1 / 37, dtype=np.float32),
        action=36,
        player=0,
        next_terminated=terminal,
    )


def _episode(
    representation: str,
    solved: float,
    *,
    length: int = 4,
    seed: int = 0,
) -> list[Position]:
    result = []
    for index in range(length):
        position = _position(index == length - 1)
        position.representation_id = representation
        position.solved = solved
        position.episode_seed = seed
        position.episode_position_index = index
        position.policy = np.zeros(37, dtype=np.float32)
        position.policy[: index + 1] = 1.0 / (index + 1)
        result.append(position)
    return result


def _solution(
    representation: str,
    ratio: float,
    crossing_changes: float,
    moves: float,
    *,
    shared: bool,
) -> list[Position]:
    record = _episode(representation, 1.0, length=2)
    for position in record:
        position.objective_ratio = ratio
        position.final_crossing_changes = crossing_changes
        position.final_moves = moves
        position.shared_witness = shared
        position.option_state = object() if shared else None
        position.target_external_action = 1 if shared else -1
    return record


def test_sequence_sampling_includes_terminal_transitions() -> None:
    replay = ReplayBuffer(100, np.random.default_rng(0))
    replay.add([_position(False) for _ in range(9)] + [_position(True)])
    sequences = replay.sample_sequences(8, unroll_steps=3, terminal_fraction=0.5)
    assert sum(sequence[-1].next_terminated for sequence in sequences) >= 4


def test_partial_games_are_not_treated_as_terminal_samples() -> None:
    replay = ReplayBuffer(100, np.random.default_rng(0))
    replay.add([_position(False) for _ in range(4)])
    sequences = replay.sample_sequences(4, unroll_steps=2, terminal_fraction=1.0)
    assert not any(sequence[-1].next_terminated for sequence in sequences)


def test_collaboration_sampling_balances_success_and_includes_cap_failures() -> None:
    replay = ReplayBuffer(100, np.random.default_rng(4))
    for solved in (0.0, 1.0):
        for _ in range(3):
            position = _position(False)
            position.solved = solved
            replay.add([position])
    shared = _position(False)
    shared.solved = 1.0
    shared.shared_witness = True
    replay.add([shared])
    censored = _position(False)
    censored.solved = 0.0
    censored.objective_censored = True
    replay.add([censored])

    batch = replay.sample_collaboration_positions(20, shared_fraction=0.25)

    assert sum(position.shared_witness for position in batch) == 5
    assert sum(position.solved > 0.5 and not position.shared_witness for position in batch) == 7
    assert sum(position.solved == 0.0 for position in batch) == 8

    censored_only = ReplayBuffer(10, np.random.default_rng(5))
    censored_only.add([censored])
    assert all(
        position.objective_censored for position in censored_only.sample_collaboration_positions(4)
    )
    assert censored_only.has_trainable_collaboration_positions()


def test_small_positive_shared_fraction_still_schedules_one_witness_episode() -> None:
    replay = ReplayBuffer(100, np.random.default_rng(6))
    replay.add(_episode("native", 1.0, length=8), representation_id="native")
    shared = _episode("donated", 1.0, length=8)
    for position in shared:
        position.shared_witness = True
    replay.add(shared, representation_id="donated")

    batch = replay.sample_collaboration_positions(
        32,
        shared_fraction=0.05,
        positions_per_episode=4,
    )

    assert sum(position.shared_witness for position in batch) == 4


def test_replay_retains_all_attempts_for_last_m_representations() -> None:
    replay = ReplayBuffer(100, np.random.default_rng(7), representation_capacity=2)
    replay.add(_episode("a", 1.0), representation_id="a")
    replay.add(_episode("b", 0.0), representation_id="b")
    replay.add(_episode("a", 0.0), representation_id="a")
    replay.add(_episode("c", 1.0), representation_id="c")

    assert replay.representation_order == ["a", "c"]
    assert {game[0].representation_id for game in replay.games} == {"a", "c"}
    assert sum(game[0].representation_id == "a" for game in replay.games) == 2


def test_collaboration_replay_uses_current_similar_and_global_quotas() -> None:
    replay = ReplayBuffer(1_000, np.random.default_rng(11))
    embeddings = {
        "current": np.asarray([1.0, 0.0]),
        "similar": np.asarray([0.99, 0.01]),
        "far-a": np.asarray([0.0, 1.0]),
        "far-b": np.asarray([-1.0, 0.0]),
    }
    for representation, embedding in embeddings.items():
        replay.set_representation_embedding(representation, embedding)
        replay.add(_episode(representation, 1.0), representation_id=representation)
        replay.add(_episode(representation, 0.0), representation_id=representation)

    batch = replay.sample_collaboration_positions(
        40,
        shared_fraction=0.0,
        current_representation="current",
        current_fraction=0.25,
        similar_fraction=0.25,
        similar_representation_count=1,
    )
    identities = [position.representation_id for position in batch]

    assert identities.count("current") >= 10
    assert identities.count("similar") >= 10
    assert sum(position.solved > 0.5 for position in batch) == 20
    assert (
        sum(
            row["positions"]
            for row in replay.last_collaboration_sample_trace
            if row["requested_representation_group"] == "current"
        )
        == 10
    )
    assert (
        sum(
            row["positions"]
            for row in replay.last_collaboration_sample_trace
            if row["requested_representation_group"] == "similar"
        )
        == 10
    )
    assert all(row["fallback"] == "none" for row in replay.last_collaboration_sample_trace)


def test_collaboration_replay_spreads_positions_and_persists_exposure() -> None:
    replay = ReplayBuffer(100, np.random.default_rng(13))
    replay.add(_episode("knot", 1.0, length=6), representation_id="knot")

    batch = replay.sample_collaboration_positions(4, positions_per_episode=4)

    assert {0, 5}.issubset({position.episode_position_index for position in batch})
    assert replay.games[0][0].replay_episode_uses == 1
    assert sum(position.replay_position_uses for position in replay.games[0]) == 4

    resumed = pickle.loads(pickle.dumps(replay))
    assert resumed.games[0][0].replay_episode_uses == 1
    assert sum(position.replay_position_uses for position in resumed.games[0]) == 4


def test_collaboration_replay_enforces_position_exposure_cap() -> None:
    replay = ReplayBuffer(100, np.random.default_rng(9))
    replay.add(_episode("a", 1.0), representation_id="a")
    replay.sample_collaboration_positions(4, positions_per_episode=4, max_position_uses=1)
    assert max(position.replay_position_uses for position in replay.games[0]) == 1
    with pytest.raises(RuntimeError, match="reached max_position_uses"):
        replay.sample_collaboration_positions(1, positions_per_episode=1, max_position_uses=1)


def test_collaboration_replay_prefers_less_exposed_attempts() -> None:
    replay = ReplayBuffer(100, np.random.default_rng(17))
    overused = _episode("same", 1.0, seed=1)
    fresh = _episode("same", 1.0, seed=2)
    overused[0].replay_episode_uses = 100
    replay.add(overused, representation_id="same")
    replay.add(fresh, representation_id="same")

    seeds = [replay.sample_collaboration_positions(1)[0].episode_seed for _ in range(40)]

    assert seeds.count(2) > seeds.count(1)


def test_continual_replay_balances_outcomes_and_new_old_successes() -> None:
    replay = ReplayBuffer(1_000, np.random.default_rng(29))
    replay.add(_episode("current", 1.0), representation_id="current")
    replay.add(_episode("old-a", 1.0), representation_id="old-a")
    replay.add(_episode("old-b", 1.0), representation_id="old-b")
    replay.add(_episode("failed-a", 0.0), representation_id="failed-a")
    replay.add(_episode("failed-b", 0.0), representation_id="failed-b")

    batch = replay.sample_continual_positions(
        32,
        current_representation="current",
        rehearsal_representations={"old-a", "old-b"},
        positions_per_episode=4,
    )

    assert sum(position.solved > 0.5 for position in batch) == 16
    success_ids = [position.representation_id for position in batch if position.solved > 0.5]
    assert success_ids.count("current") == 8
    assert sum(identity.startswith("old-") for identity in success_ids) == 8
    strata = [row["requested_stratum"] for row in replay.last_collaboration_sample_trace]
    assert strata.count("current-success") == 2
    assert strata.count("rehearsal-success") == 2
    assert strata.count("ordinary-failure") == 4


def test_continual_replay_can_raise_rehearsal_to_the_positive_half() -> None:
    replay = ReplayBuffer(1_000, np.random.default_rng(31))
    replay.add(_episode("current", 1.0), representation_id="current")
    inherited = _episode("old", 1.0)
    for position in inherited:
        position.shared_witness = True
    replay.add(inherited, representation_id="old")
    replay.add(_episode("failed", 0.0), representation_id="failed")

    batch = replay.sample_continual_positions(
        32,
        current_representation="current",
        rehearsal_representations={"old"},
        rehearsal_fraction=0.5,
        positions_per_episode=4,
    )

    successes = [position for position in batch if position.solved > 0.5]
    assert len(successes) == 16
    assert {position.representation_id for position in successes} == {"old"}


def test_native_solution_bank_survives_replay_eviction_and_returns_a_copy() -> None:
    replay = ReplayBuffer(3, np.random.default_rng(31))
    best = _solution("old", 10.0, 2.0, 3.0, shared=False)
    replay.add(best)
    replay.add(_episode("new", 0.0, length=3), representation_id="new")

    assert all(game[0].representation_id != "old" for game in replay.games)
    archived = replay.best_native_solution_record("old", 10.0)
    assert archived is not None
    assert archived is not best
    assert archived[0].representation_id == "old"
    archived[0].action = 0
    assert replay.best_native_solution_record("old", 10.0)[0].action != 0


def test_distillation_uses_only_best_donation_strictly_better_than_native_archive() -> None:
    replay = ReplayBuffer(1_000, np.random.default_rng(19))
    replay.add(_solution("knot", 10.0, 5.0, 50.0, shared=False))  # L10 = 100
    replay.add(_solution("knot", 10.0, 5.0, 50.0, shared=True))  # equal
    replay.add(_solution("knot", 10.0, 6.0, 50.0, shared=True))  # worse
    assert replay.active_distillation_records() == []

    better = _solution("knot", 10.0, 4.0, 50.0, shared=True)  # L10 = 90
    best = _solution("knot", 10.0, 3.0, 50.0, shared=True)  # L10 = 80
    replay.add(better)
    replay.add(best)
    assert replay.active_distillation_records() == [best]

    # A new equal native incumbent makes the donated policy target stale. The
    # shared record stays in replay for one-sided critic upper-bound training.
    native_best = _solution("knot", 10.0, 3.0, 50.0, shared=False)
    replay.add(native_best)
    assert replay.best_native_objective("knot", 10.0) == 80.0
    assert replay.active_distillation_records() == []
    assert any(record is best for record in replay.games)

    # Objective ratios have independent incumbents.
    ratio_1000 = _solution("knot", 1000.0, 0.0, 70.0, shared=True)
    replay.add(ratio_1000)
    assert replay.active_distillation_records() == [ratio_1000]

    resumed = pickle.loads(pickle.dumps(replay))
    assert resumed.best_native_objective("knot", 10.0) == 80.0
    assert resumed.active_distillation_records()[0][0].objective_ratio == 1000.0


def test_evaluation_objective_can_make_existing_donation_stale() -> None:
    replay = ReplayBuffer(100, np.random.default_rng(23))
    donation = _solution("knot", 10.0, 4.0, 50.0, shared=True)
    replay.add(donation)
    assert replay.active_distillation_records() == [donation]

    replay.record_native_objective("knot", 10.0, 90.0)
    assert replay.active_distillation_records() == []

    with pytest.raises(ValueError, match="must be finite"):
        replay.record_native_objective("knot", 10.0, float("nan"))
