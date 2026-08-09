from __future__ import annotations

import math

import numpy as np

from pgx_mcts_bench.config import BraidGameConfig
from pgx_mcts_bench.game import make_game


def _game():
    return make_game(
        BraidGameConfig(
            max_len=16,
            max_strands=4,
            simplify_budget=16,
            allow_crossing_change=True,
            multi_objective=True,
            serial_window=7,
            serial_act_width=7,
            serial_internal_horizon=5,
            serial_internal_budget_remaining=True,
            objective_budget_channel=True,
        )
    )


def test_internal_shifts_do_not_change_terminal_scientific_reward() -> None:
    game = _game()
    start = game.from_word([1, 1, -1], strands=2, log_ratio=math.log(1000.0))
    reduce_at_one = (game._act_origin + 1) * game._per_offset  # noqa: SLF001
    destabilize = game._singleton_base  # noqa: SLF001
    direct = game.step(start.state, reduce_at_one)
    direct = game.step(direct.state, destabilize)
    shifted = start
    shift_action = game._shift_base  # noqa: SLF001
    for _ in range(3):
        shifted = game.step(shifted.state, shift_action)
    shifted = game.step(shifted.state, reduce_at_one)
    shifted = game.step(shifted.state, destabilize)

    direct_reward = game.final_rewards(direct.state)
    shifted_reward = game.final_rewards(shifted.state)
    np.testing.assert_allclose(shifted_reward, direct_reward)
    assert direct.terminated and shifted.terminated
    assert game.semantic_move_count(shifted.state) == 2
    assert game.internal_ply_count(shifted.state) == 3


def test_terminal_reward_uses_ratio_times_crossings_plus_semantic_moves() -> None:
    game = _game()
    for ratio in (10.0, 1000.0):
        direct = game.from_word([1], strands=2, log_ratio=math.log(ratio))
        direct = game.step(direct.state, game._singleton_base)  # noqa: SLF001
        transition = game.from_word([1], strands=2, log_ratio=math.log(ratio))
        crossing_action = game._act_origin * game._per_offset + game._per_offset - 1  # noqa: SLF001
        changed = game.step(transition.state, crossing_action)
        terminal = game.step(changed.state, game._singleton_base)  # noqa: SLF001
        assert terminal.terminated
        assert game.semantic_move_count(terminal.state) == 2
        reward = float(game.final_rewards(terminal.state).max())
        expected = 1.0 - 2.0 * (ratio + 2.0) / ((ratio + 1.0) * 16)
        assert np.isclose(reward, expected)
        assert reward < float(game.final_rewards(direct.state).max())
