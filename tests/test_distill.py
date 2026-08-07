from __future__ import annotations

import numpy as np
import torch

from pgx_mcts_bench.data import Position, ReplayBuffer
from pgx_mcts_bench.distill import (
    STUDENT_NAMES,
    _best_destination,
    _project_policy,
    _route,
    bounded_option_loss,
    stable_option_route_loss,
    train_bounded_option_step,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import STAGES, _config, candidates
from pgx_mcts_bench.networks import (
    OptionPolicyAdapter,
    load_policy_value_state_dict,
    make_braid_network,
)
from pgx_mcts_bench.serial_braid import SerialBraidGame
from pgx_mcts_bench.training import attach_policy_value_preservation_teacher


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


def test_option_adapter_can_condition_on_the_head_cell_not_only_global_pools() -> None:
    adapter = OptionPolicyAdapter(
        observation_channels=1,
        action_size=1,
        internal_budget_channel=None,
        width=16,
        residual_blocks=0,
    )
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
        adapter.input.weight[0, 0, 1] = 1.0
        adapter.readout[0].weight[0, 32] = 1.0
        adapter.readout[2].weight[0, 0] = 1.0
        adapter.readout[4].weight[0, 0] = 1.0
    at_head = torch.zeros(1, 1, 1, 7)
    away_from_head = torch.zeros_like(at_head)
    at_head[0, 0, 0, 3] = 1.0
    away_from_head[0, 0, 0, 2] = 1.0

    assert adapter(at_head).item() > adapter(away_from_head).item()


def test_old_global_pool_option_checkpoint_migrates_without_changing_outputs() -> None:
    torch.manual_seed(41)
    candidate = next(candidate for candidate in candidates() if candidate.name == "s-tape4")
    config = _config(candidate, STAGES[0], 0, "cpu")
    source = make_braid_network(config.game, config.model).eval()
    adapter = source.attach_option_policy_adapter()
    gate = source.attach_option_policy_gate(initial_probability=0.1)
    with torch.no_grad():
        for module in (adapter, gate):
            module.readout[0].weight[:, 64:96].zero_()
            module.readout[-1].weight.normal_(std=0.05)
    observation = torch.randn(3, config.game.observation_channels, 1, config.game.width)
    with torch.inference_mode():
        expected = source(observation)
    historical = dict(source.state_dict())
    for key in (
        "option_policy_adapter.readout.0.weight",
        "option_policy_gate.readout.0.weight",
    ):
        weight = historical[key]
        historical[key] = torch.cat([weight[:, :64], weight[:, -1:]], dim=1)

    restored = make_braid_network(config.game, config.model).eval()
    assert load_policy_value_state_dict(restored, historical)
    with torch.inference_mode():
        actual = restored(observation)

    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output)


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


def test_bounded_option_step_learns_with_frozen_starting_teacher() -> None:
    torch.manual_seed(7)
    candidate = next(candidate for candidate in candidates() if candidate.name == "d-head128-u1")
    config = _config(candidate, STAGES[0], 0, "cpu")
    game = make_game(config.game)
    assert isinstance(game, SerialBraidGame)
    transition = game.from_word([1, 2, -1, 2, 1], strands=3)
    raw = game.unwrap(transition.state)
    target = next(
        int(action)
        for action in np.flatnonzero(np.asarray(raw.legal_action_mask))
        if (
            (destination := _best_destination(game, int(action), 0, 5)) is not None
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
    replay = ReplayBuffer(100, np.random.default_rng(11))
    replay.add([position])
    network = make_braid_network(config.game, config.model)
    network.policy_value_preservation_weight = 0.1
    teacher = attach_policy_value_preservation_teacher(network)
    teacher_before = {name: value.clone() for name, value in teacher.state_dict().items()}
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    initial = float(
        bounded_option_loss(
            network, game, [position], horizon=5, beam_width=2, device=torch.device("cpu")
        ).item()
    )

    train_bounded_option_step(
        network,
        optimizer,
        game,
        replay,
        replay.rng,
        batch_size=1,
        horizon=5,
        beam_width=2,
        device=torch.device("cpu"),
        positions=[position],
        learning_rate_scale=0.1,
    )
    assert optimizer.param_groups[0]["lr"] == 1e-3

    for _ in range(12):
        train_bounded_option_step(
            network,
            optimizer,
            game,
            replay,
            replay.rng,
            batch_size=1,
            horizon=5,
            beam_width=2,
            device=torch.device("cpu"),
        )

    final = float(
        bounded_option_loss(
            network, game, [position], horizon=5, beam_width=2, device=torch.device("cpu")
        ).item()
    )
    assert final < initial
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name])


def test_zero_initialized_option_adapter_is_exact_and_updates_in_isolation() -> None:
    torch.manual_seed(13)
    candidate = next(candidate for candidate in candidates() if candidate.name == "d-head128-u1")
    config = _config(candidate, STAGES[0], 0, "cpu")
    game = make_game(config.game)
    assert isinstance(game, SerialBraidGame)
    transition = game.from_word([1, 2, -1, 2, 1], strands=3)
    raw = game.unwrap(transition.state)
    target = next(
        int(action)
        for action in np.flatnonzero(np.asarray(raw.legal_action_mask))
        if (
            (destination := _best_destination(game, int(action), 0, 5)) is not None
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
        solved=1.0,
        final_crossing_changes=0.0,
        final_moves=10.0,
        shared_witness=True,
        representation_id="route",
        objective_ratio=10.0,
    )
    replay = ReplayBuffer(10, np.random.default_rng(17))
    replay.add([position])
    network = make_braid_network(config.game, config.model).eval()
    observation = torch.from_numpy(transition.observation[None]).permute(0, 3, 1, 2)
    with torch.inference_mode():
        logits_before, value_before = network(observation)
    adapter = network.attach_option_policy_adapter()
    with torch.inference_mode():
        logits_attached, value_attached = network(observation)
    torch.testing.assert_close(logits_attached, logits_before, rtol=0, atol=0)
    torch.testing.assert_close(value_attached, value_before, rtol=0, atol=0)

    base_before = {
        name: value.clone()
        for name, value in network.state_dict().items()
        if not name.startswith("option_policy_adapter.")
    }
    adapter_before = {name: value.clone() for name, value in adapter.state_dict().items()}
    option_optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    initial_option_loss = float(
        stable_option_route_loss(
            network, game, [position], horizon=5, device=torch.device("cpu")
        ).item()
    )
    diagnostics: dict[str, float | str] = {}
    train_bounded_option_step(
        network,
        option_optimizer,
        game,
        replay,
        replay.rng,
        batch_size=1,
        horizon=5,
        beam_width=2,
        device=torch.device("cpu"),
        positions=[position],
        adapter_only=True,
        diagnostics=diagnostics,
    )
    final_option_loss = float(
        stable_option_route_loss(
            network, game, [position], horizon=5, device=torch.device("cpu")
        ).item()
    )
    assert final_option_loss < initial_option_loss
    assert diagnostics["target"] == "canonical-shortest-route"
    assert diagnostics["loss_before"] == initial_option_loss
    assert diagnostics["loss_after"] == final_option_loss
    assert float(diagnostics["loss_delta"]) < 0.0
    for name, value in network.state_dict().items():
        if name.startswith("option_policy_adapter."):
            continue
        torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)
    assert all(
        parameter.grad is None
        for name, parameter in network.named_parameters()
        if not name.startswith("option_policy_adapter.")
    )
    assert any(
        not torch.equal(value, adapter_before[name])
        for name, value in adapter.state_dict().items()
    )
    replay.record_native_objective("route", 10.0, 10.0)
    stale_adapter = {name: value.clone() for name, value in adapter.state_dict().items()}
    assert (
        train_bounded_option_step(
            network,
            option_optimizer,
            game,
            replay,
            replay.rng,
            batch_size=1,
            horizon=5,
            beam_width=2,
            device=torch.device("cpu"),
            adapter_only=True,
        )
        == 0.0
    )
    for name, value in adapter.state_dict().items():
        torch.testing.assert_close(value, stale_adapter[name], rtol=0, atol=0)
    with torch.inference_mode():
        logits_after, value_after = network(observation)
    torch.testing.assert_close(value_after, value_before, rtol=0, atol=0)
    restored = make_braid_network(config.game, config.model).eval()
    load_policy_value_state_dict(restored, network.state_dict())
    assert restored.option_policy_adapter is not None
    with torch.inference_mode():
        restored_logits, restored_value = restored(observation)
    torch.testing.assert_close(restored_logits, logits_after, rtol=0, atol=0)
    torch.testing.assert_close(restored_value, value_after, rtol=0, atol=0)


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


def test_gated_option_adapter_trains_route_and_off_route_losses_in_isolation() -> None:
    torch.manual_seed(29)
    candidate = next(candidate for candidate in candidates() if candidate.name == "d-head128-u1")
    config = _config(candidate, STAGES[0], 0, "cpu")
    game = make_game(config.game)
    assert isinstance(game, SerialBraidGame)
    transition = game.from_word([1, 2, -1, 2, 1], strands=3)
    raw = game.unwrap(transition.state)
    target = next(
        int(action)
        for action in np.flatnonzero(np.asarray(raw.legal_action_mask))
        if (
            (destination := _best_destination(game, int(action), 0, 5)) is not None
            and destination[0]
        )
    )
    route_position = Position(
        transition.observation,
        transition.legal_actions,
        np.zeros(game.num_actions, dtype=np.float32),
        0,
        transition.player,
        option_state=transition.state,
        target_external_action=target,
    )
    off_route_transition = game.from_word([1, -2, 1, -2], strands=3)
    off_route = Position(
        off_route_transition.observation,
        off_route_transition.legal_actions,
        np.zeros(game.num_actions, dtype=np.float32),
        0,
        off_route_transition.player,
    )
    replay = ReplayBuffer(10, np.random.default_rng(31))
    replay.add([route_position])
    network = make_braid_network(config.game, config.model).eval()
    adapter = network.attach_option_policy_adapter()
    gate = network.attach_option_policy_gate(initial_probability=0.1)
    tensor = torch.from_numpy(transition.observation[None]).permute(0, 3, 1, 2)
    with torch.inference_mode():
        residual, initial_gate = network.option_policy_components(tensor)
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)
    torch.testing.assert_close(initial_gate, torch.full_like(initial_gate, 0.1))
    base_before = {
        name: value.clone()
        for name, value in network.state_dict().items()
        if not name.startswith(("option_policy_adapter.", "option_policy_gate."))
    }
    optimizer = torch.optim.AdamW(
        [*adapter.parameters(), *gate.parameters()], lr=1e-3
    )
    diagnostics: dict[str, float | str] = {}
    for _ in range(4):
        train_bounded_option_step(
            network,
            optimizer,
            game,
            replay,
            replay.rng,
            batch_size=1,
            horizon=5,
            beam_width=2,
            device=torch.device("cpu"),
            positions=[route_position],
            adapter_only=True,
            preservation_positions=[off_route],
            route_gate_weight=0.1,
            off_route_kl_weight=1.0,
            off_route_gate_weight=0.1,
            diagnostics=diagnostics,
        )
    assert float(diagnostics["route_gate_loss"]) > 0.0
    assert float(diagnostics["off_route_kl"]) >= 0.0
    assert float(diagnostics["off_route_gate"]) > 0.0
    assert diagnostics["off_route_positions"] == 1.0
    for name, value in network.state_dict().items():
        if name.startswith(("option_policy_adapter.", "option_policy_gate.")):
            continue
        torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)
    restored = make_braid_network(config.game, config.model).eval()
    load_policy_value_state_dict(restored, network.state_dict())
    assert restored.option_policy_adapter is not None
    assert restored.option_policy_gate is not None


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
