import numpy as np

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
        position.objective_censored
        for position in censored_only.sample_collaboration_positions(4)
    )
    assert censored_only.has_trainable_collaboration_positions()
