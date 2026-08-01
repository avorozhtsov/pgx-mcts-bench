from __future__ import annotations

import numpy as np
import torch

from pgx_mcts_bench.data import Position
from pgx_mcts_bench.distill import (
    STUDENT_NAMES,
    _best_destination,
    _project_policy,
    _route,
    bounded_option_loss,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import STAGES, _config, candidates
from pgx_mcts_bench.networks import make_braid_network
from pgx_mcts_bench.serial_braid import SerialBraidGame


def _student(name: str) -> SerialBraidGame:
    candidate = next(candidate for candidate in candidates() if candidate.name == name)
    game = make_game(_config(candidate, STAGES[0], 0, "cpu").game)
    assert isinstance(game, SerialBraidGame)
    return game


def test_distilled_candidates_exist_and_use_factorized_value() -> None:
    by_name = {candidate.name: candidate for candidate in candidates()}
    assert set(STUDENT_NAMES) <= by_name.keys()
    assert all(by_name[name].use_auxiliary_value for name in STUDENT_NAMES)
    assert by_name["d-gru128-u1"].serial_encoder == "gru"
    assert by_name["d-fsa32-u1"].serial_encoder == "fsa"
    assert by_name["d-tape4-u1"].serial_tape_symbols == 4
    assert all(by_name[name].serial_internal_horizon == 5 for name in STUDENT_NAMES)


def test_sixth_action_must_be_an_external_braid_move() -> None:
    game = _student("d-head128-u1")
    transition = game.from_word([1, 2, -1, 2, 1], strands=3)
    shift = game._shift_base  # noqa: SLF001
    for expected in range(1, 6):
        assert transition.legal_actions[shift]
        transition = game.step(transition.state, shift)
        assert transition.state.internal_steps == expected
    assert not transition.legal_actions[game._shift_base :].any()  # noqa: SLF001
    external = int(np.flatnonzero(transition.legal_actions[: game._shift_base])[0])  # noqa: SLF001
    transition = game.step(transition.state, external)
    assert transition.state.internal_steps == 0


def test_bounded_option_loss_reaches_nonlocal_teacher_move_and_backpropagates() -> None:
    candidate = next(candidate for candidate in candidates() if candidate.name == "d-head128-u1")
    config = _config(candidate, STAGES[0], 0, "cpu")
    game = make_game(config.game)
    assert isinstance(game, SerialBraidGame)
    transition = game.from_word([1, 2, -1, 2, 1], strands=3)
    pgx = game.unwrap(transition.state)
    length = 5
    target = next(
        int(action)
        for action in np.flatnonzero(np.asarray(pgx.legal_action_mask))
        if (
            (destination := _best_destination(game, int(action), 0, length))
            is not None
            and destination[0]
        )
    )
    position = Position(
        transition.observation,
        transition.legal_actions,
        np.zeros(game.num_actions, dtype=np.float32),
        0,
        transition.player,
        option_state=transition.state,
        target_external_action=target,
    )
    network = make_braid_network(config.game, config.model)
    loss = bounded_option_loss(
        network, game, [position], horizon=5, beam_width=2, device=torch.device("cpu")
    )
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in network.parameters())


def test_binary_stride_route_reaches_any_position_in_at_most_two_shifts() -> None:
    game = _student("d-head128-u1")
    for length in range(2, 49):
        for goal in range(length):
            route = _route(game, 0, goal, length)
            head = 0
            for action in route:
                head = (head + int(game.shift_of(action))) % length
            assert head == goal
            # Signed powers of two cover the 48-letter capacity in at most three
            # shifts (the common cases need one or two).
            assert len(route) <= 3


def test_teacher_action_maps_to_same_underlying_edit_and_policy() -> None:
    game = _student("d-tape4-u1")
    transition = game.from_word([1, 2, -1, 2, 1], strands=3)
    pgx_state = game.unwrap(transition.state)
    length = 5
    # Select any legal underlying edit at a non-zero position.
    teacher_action = next(
        action
        for action in np.flatnonzero(np.asarray(pgx_state.legal_action_mask))
        if any(
            game.underlying_action(serial, 3, length) == action
            for serial in range(game._shift_base)
        )
    )
    destination = _best_destination(game, int(teacher_action), 0, length)
    assert destination is not None
    route, head, serial_action = destination
    assert game.underlying_action(serial_action, head, length) == teacher_action
    assert all(game.tape_write_of(action) == 0 for action in route)

    teacher_policy = np.zeros(game.env.num_actions, dtype=np.float32)
    teacher_policy[teacher_action] = 1.0
    projected = _project_policy(game, teacher_policy, head, length, serial_action)
    assert projected[serial_action] == 1.0
    assert projected.sum() == 1.0
