import numpy as np

from pgx_mcts_bench.config import GameConfig
from pgx_mcts_bench.game import Go6x6


def test_go_6x6_shapes_and_pass() -> None:
    game = Go6x6(GameConfig())
    state = game.reset(0)
    assert state.observation.shape == (6, 6, 9)
    assert state.legal_actions.shape == (37,)
    assert state.legal_actions[-1]


def test_two_passes_end_game() -> None:
    game = Go6x6(GameConfig())
    state = game.reset(0)
    state = game.step(state.state, 36)
    assert not state.terminated
    state = game.step(state.state, 36)
    assert state.terminated
    assert set(np.asarray(game.final_rewards(state.state)).tolist()) == {-1.0, 1.0}
