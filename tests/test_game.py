import numpy as np

from pgx_mcts_bench.config import GameConfig
from pgx_mcts_bench.game import Go6x6


def test_go_6x6_shapes_and_pass() -> None:
    game = Go6x6(GameConfig())
    state = game.reset(0)
    assert state.observation.shape == (6, 6, 11)
    assert state.legal_actions.shape == (37,)
    assert not state.legal_actions[-1]
    assert np.all(state.observation[..., -2] == 0.0)
    assert np.all(state.observation[..., -1] == 0.0)


def test_two_passes_end_game() -> None:
    game = Go6x6(GameConfig(min_moves_before_pass=0))
    state = game.reset(0)
    state = game.step(state.state, 36)
    assert not state.terminated
    assert np.all(state.observation[..., -2] == 1.0)
    np.testing.assert_allclose(state.observation[..., -1], 1 / 72)
    state = game.step(state.state, 36)
    assert state.terminated
    assert set(np.asarray(game.final_rewards(state.state)).tolist()) == {-1.0, 1.0}


def test_pass_opens_after_configured_minimum_move_count() -> None:
    game = Go6x6(GameConfig(min_moves_before_pass=1))
    state = game.reset(0)
    assert not state.legal_actions[-1]
    state = game.step(state.state, 0)
    assert state.legal_actions[-1]
