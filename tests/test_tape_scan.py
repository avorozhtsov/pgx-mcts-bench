from dataclasses import replace

import numpy as np
import torch

from pgx_mcts_bench.config import BraidGameConfig, ModelConfig
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import candidates, tape_scan_arms
from pgx_mcts_bench.networks import SequenceBraidNet, make_braid_network
from pgx_mcts_bench.serial_braid import (
    SERIAL_BRAID,
    SERIAL_COMMUTE,
    SERIAL_INSERT,
    SERIAL_REDUCE,
)

BASE = BraidGameConfig(
    max_len=12,
    max_strands=4,
    scramble_budget=1,
    simplify_budget=16,
    allow_crossing_change=True,
    multi_objective=True,
    serial_window=7,
    serial_act_width=1,
)


def _tape_game(symbols: int = 4):
    from pgx_mcts_bench.game import make_game

    return make_game(replace(BASE, serial_tape_symbols=symbols))


def _marked(game, word: list[int], strands: int, marks: list[int]):
    transition = game.from_word(word, strands)
    tape = np.zeros(game.config.max_len, dtype=np.int64)
    tape[: len(marks)] = marks
    return transition.state._replace(tape=tape)


def test_serial_legal_action_translation_is_cached_by_head_and_length() -> None:
    game = _tape_game()
    state = game.from_word([1, 2, 1], 3).state
    original = game.underlying_action
    calls = 0

    def counted(action, head, length):
        nonlocal calls
        calls += 1
        return original(action, head, length)

    game.underlying_action = counted
    game._underlying_action_cache.clear()
    first = game._legal(state.pgx, state.head, state.internal_steps)
    first_calls = calls
    second = game._legal(state.pgx, state.head, state.internal_steps)

    assert first_calls == game._shift_base
    assert calls == first_calls
    np.testing.assert_array_equal(second, first)


def test_shift_writes_symbol_before_moving() -> None:
    game = _tape_game(2)
    state = game.from_word([1, 2, 1], 3).state
    action = next(
        index
        for index in range(game.num_actions)
        if game.describe(index) == "SHIFT_RIGHT(1,WRITE(1))"
    )
    after = game.step(state, action).state
    assert after.head == 1
    assert after.tape[0] == 1


def test_rewrites_transport_the_aligned_tape() -> None:
    game = _tape_game()

    reduced = game.step(_marked(game, [1, -1, 2], 3, [1, 2, 3]), SERIAL_REDUCE).state
    assert reduced.tape[:3].tolist() == [3, 0, 0]

    commuted = game.step(_marked(game, [1, 3, 2], 4, [1, 2, 3]), SERIAL_COMMUTE).state
    assert commuted.tape[:3].tolist() == [2, 1, 3]

    braided = game.step(_marked(game, [1, 2, 1], 3, [1, 2, 3]), SERIAL_BRAID).state
    assert braided.tape[:3].tolist() == [3, 2, 1]

    inserted = game.step(_marked(game, [1, 2], 3, [1, 2]), SERIAL_INSERT).state
    assert inserted.tape[:4].tolist() == [0, 0, 1, 2]

    destab = game.step(
        _marked(game, [1, 2, 1], 3, [1, 2, 3]), game._singleton_base
    ).state
    assert destab.tape[:3].tolist() == [1, 3, 0]

    stabilized = game.step(
        _marked(game, [1], 2, [3]), game._singleton_base + 1
    ).state
    assert stabilized.tape[:2].tolist() == [3, 0]


def test_tape_is_visible_as_one_hot_and_increases_only_shift_family() -> None:
    plain = _tape_game(0)
    tape = _tape_game(4)
    transition = tape.from_word([1, 2, 1], 3)
    assert tape.num_actions == plain.num_actions + 2 * len(tape.strides) * 3
    assert transition.observation.shape[-1] == tape.config.observation_channels
    # Tape one-hot channels precede the mandatory internal-budget plane.
    assert np.allclose(transition.observation[0, :, -5:-1].sum(axis=1), 1.0)


def test_scan_gru_uses_local_window_sequence_and_advanced_heads() -> None:
    config = replace(
        BASE,
        serial_encoder="scan-gru",
        serial_encoder_states=16,
        serial_tape_symbols=2,
    )
    network = make_braid_network(
        config,
        ModelConfig(
            channels=8,
            auxiliary_value_members=4,
            use_auxiliary_value=True,
        ),
    )
    assert isinstance(network, SequenceBraidNet)
    observation = torch.randn(3, config.observation_channels, 1, config.max_len)
    policy, value, auxiliary = network.forward_with_auxiliary(observation)
    assert policy.shape == (3, config.action_size)
    assert value.shape == (3,)
    assert all(part.shape == (3, 4) for part in auxiliary)


def test_recommended_arms_are_matched_and_use_advanced_value() -> None:
    arms = tape_scan_arms()
    assert [arm.name for arm in arms] == [
        "s-tape2",
        "s-tape4",
        "s-scan-gru",
        "s-scan-gru-tape2",
    ]
    assert all(arm.use_auxiliary_value for arm in arms)
    assert len({(arm.simulations, arm.train_steps, arm.serial_window) for arm in arms}) == 1


def test_tape4_h5_exposes_remaining_internal_budget() -> None:
    candidate = next(candidate for candidate in candidates() if candidate.name == "s-tape4-h5")
    assert candidate.serial_internal_horizon == 5
    assert candidate.serial_internal_budget_remaining
    game = make_game(
        replace(
            BASE,
            serial_tape_symbols=4,
            serial_internal_horizon=5,
            serial_internal_budget_remaining=True,
        )
    )
    transition = game.from_word([1, 2, 1], 3)
    internal_channel = -1
    assert np.allclose(transition.observation[..., internal_channel], 1.0)
    shift = next(
        action
        for action in range(game.num_actions)
        if game.describe(action) == "SHIFT_RIGHT(1,WRITE(1))"
    )
    after = game.step(transition.state, shift)
    assert np.allclose(after.observation[..., internal_channel], 0.8)
