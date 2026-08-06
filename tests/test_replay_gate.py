import numpy as np

from pgx_mcts_bench.data import Position, ReplayBuffer
from pgx_mcts_bench.replay_gate import _exposure_snapshot, _metadata_failures


def _position(index: int) -> Position:
    return Position(
        observation=np.zeros((2, 2, 2), dtype=np.float32),
        legal_actions=np.ones(2, dtype=bool),
        policy=np.full(2, 0.5, dtype=np.float32),
        action=0,
        player=1,
        role=1,
        solved=1.0,
        representation_id="knot",
        termination_reason="solved",
        action_horizon=64,
        residual_word_length=2 - index,
        mcts_visit_count=32,
        episode_position_index=index,
    )


def test_replay_gate_metadata_and_exposure_snapshot() -> None:
    replay = ReplayBuffer(100, np.random.default_rng(1), representation_capacity=10)
    replay.add([_position(0), _position(1)], representation_id="knot")

    assert _metadata_failures(replay) == []
    assert _exposure_snapshot(replay) == [
        {
            "representation": "knot",
            "episode_seed": 0,
            "episode_uses": 0,
            "position_uses": [0, 0],
        }
    ]


def test_replay_gate_reports_missing_search_metadata() -> None:
    replay = ReplayBuffer(100, np.random.default_rng(2))
    position = _position(0)
    position.mcts_visit_count = 0
    replay.add([position], representation_id="knot")

    assert _metadata_failures(replay) == ["game 0 position 0: missing MCTS visits"]
