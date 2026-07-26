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
