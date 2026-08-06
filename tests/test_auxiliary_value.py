from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from pgx_mcts_bench.config import BraidGameConfig, ModelConfig
from pgx_mcts_bench.data import Position, ReplayBuffer
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.training import train_alphazero_step


def _observation(game: BraidGameConfig, batch: int = 2) -> torch.Tensor:
    observation = torch.zeros(batch, game.observation_channels, 1, game.width, dtype=torch.float32)
    observation[:, 2 * (game.max_strands - 1), :, :] = 1.0
    return observation


@pytest.mark.parametrize(
    "game",
    [
        BraidGameConfig(max_len=16, max_strands=3, simplify_budget=12),
        BraidGameConfig(max_len=16, max_strands=3, simplify_budget=12, serial_window=7),
        BraidGameConfig(
            max_len=16,
            max_strands=3,
            simplify_budget=12,
            serial_window=7,
            serial_encoder="gru",
            serial_encoder_states=8,
        ),
    ],
)
def test_all_braid_architectures_expose_four_shadow_members(
    game: BraidGameConfig,
) -> None:
    network = make_braid_network(
        game, ModelConfig(channels=8, latent_channels=8, residual_blocks=1)
    )
    observation = _observation(game)

    policy, legacy, auxiliary = network.forward_with_auxiliary(observation)
    solve_logits, crossings, moves = auxiliary
    composed = network.composed_auxiliary_value(observation, auxiliary)

    assert policy.shape == (2, game.action_size)
    assert legacy.shape == (2,)
    assert solve_logits.shape == crossings.shape == moves.shape == (2, 4)
    assert torch.all(crossings > 0) and torch.all(moves > 0)
    assert composed.shape == (2,) and torch.isfinite(composed).all()


def test_old_checkpoint_loads_without_auxiliary_parameters() -> None:
    game = BraidGameConfig(max_len=16, max_strands=3, simplify_budget=12)
    model = ModelConfig(channels=8, latent_channels=8, residual_blocks=1)
    source = make_braid_network(game, model).eval()
    old_state = {
        key: value for key, value in source.state_dict().items() if not key.startswith("auxiliary.")
    }
    target = make_braid_network(game, model).eval()

    migrated = load_policy_value_state_dict(target, old_state)
    observation = _observation(game)
    with torch.inference_mode():
        source_policy, source_value = source(observation)
        target_policy, target_value = target(observation)

    assert migrated
    assert torch.equal(source_policy, target_policy)
    assert torch.equal(source_value, target_value)


@pytest.mark.parametrize(
    ("game", "expanded_keys"),
    [
        (
            BraidGameConfig(max_len=16, max_strands=3, simplify_budget=12),
            ("representation.net.0.weight",),
        ),
        (
            BraidGameConfig(
                max_len=16,
                max_strands=3,
                simplify_budget=12,
                serial_window=7,
                serial_encoder="gru",
                serial_encoder_states=8,
            ),
            ("gru.weight_ih_l0", "body.0.weight"),
        ),
    ],
)
def test_checkpoint_migration_ignores_one_new_input_feature(
    game: BraidGameConfig, expanded_keys: tuple[str, ...]
) -> None:
    model = ModelConfig(channels=8, latent_channels=8, residual_blocks=1)
    target = make_braid_network(game, model)
    old_state = {key: value.clone() for key, value in target.state_dict().items()}
    for key in expanded_keys:
        old_state[key] = old_state[key][:, :-1, ...].clone()

    migrated = load_policy_value_state_dict(target, old_state)
    loaded = target.state_dict()

    assert migrated
    for key in expanded_keys:
        assert torch.equal(loaded[key][:, :-1, ...], old_state[key])
        assert torch.count_nonzero(loaded[key][:, -1, ...]) == 0


def test_budget_prototype_migration_is_function_preserving_before_training() -> None:
    old_game = BraidGameConfig(
        max_len=16, max_strands=3, simplify_budget=12, serial_window=7
    )
    new_game = BraidGameConfig(
        max_len=16,
        max_strands=3,
        simplify_budget=12,
        serial_window=7,
        objective_budget_channel=True,
    )
    old_model = ModelConfig(channels=8, latent_channels=8, residual_blocks=1)
    new_model = ModelConfig(
        channels=8,
        latent_channels=8,
        residual_blocks=1,
        auxiliary_solve_backprop_to_encoder=True,
        auxiliary_budget_monotonic_weight=0.25,
        auxiliary_budget_conditioning=True,
    )
    source = make_braid_network(old_game, old_model).eval()
    target = make_braid_network(new_game, new_model).eval()
    historical_state = {
        key: value
        for key, value in source.state_dict().items()
        if not any(
            part in key
            for part in (
                ".cost_budget.",
                ".solve_conditioning.",
                "auxiliary.body_budget_skip.",
                "auxiliary.legacy_budget_skip.",
            )
        )
    }
    migrated = load_policy_value_state_dict(target, historical_state)
    old_observation = _observation(old_game)
    new_observation = _observation(new_game)
    new_observation[0, -1] = 0.2
    new_observation[1, -1] = 0.9

    with torch.inference_mode():
        old_policy, old_value, old_auxiliary = source.forward_with_auxiliary(old_observation)
        new_policy, new_value, new_auxiliary = target.forward_with_auxiliary(new_observation)

    assert migrated
    torch.testing.assert_close(new_policy, old_policy)
    torch.testing.assert_close(new_value, old_value)
    for new_output, old_output in zip(new_auxiliary, old_auxiliary, strict=True):
        torch.testing.assert_close(new_output, old_output)


def test_cutover_switch_uses_composed_ensemble_value() -> None:
    game = BraidGameConfig(max_len=16, max_strands=3, simplify_budget=12)
    model = ModelConfig(
        channels=8,
        latent_channels=8,
        residual_blocks=1,
        use_auxiliary_value=True,
    )
    network = make_braid_network(game, model).eval()
    observation = _observation(game)

    with torch.inference_mode():
        _, search_value = network(observation)
        _, _, auxiliary = network.forward_with_auxiliary(observation)
        composed = network.composed_auxiliary_value(observation, auxiliary)

    assert torch.equal(search_value, composed)


def test_auxiliary_losses_use_solve_labels_and_mask_unsolved_costs() -> None:
    game = BraidGameConfig(max_len=16, max_strands=3, simplify_budget=12)
    network = make_braid_network(
        game, ModelConfig(channels=8, latent_channels=8, residual_blocks=1)
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    replay = ReplayBuffer(100, np.random.default_rng(0))
    observation = _observation(game, batch=1)[0].permute(1, 2, 0).numpy()
    policy = np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32)
    legal = np.ones(game.action_size, dtype=bool)
    positions = []
    for index, solved in enumerate((0.0, 1.0)):
        positions.append(
            Position(
                observation=observation.copy(),
                legal_actions=legal,
                policy=policy,
                action=0,
                player=1,
                role=1,
                outcome=1.0 if solved else -1.0,
                solved=solved,
                final_crossing_changes=2.0 if solved else float("nan"),
                final_moves=7.0 if solved else float("nan"),
                episode_seed=index + 1,
            )
        )
    replay.add(positions)

    metrics = train_alphazero_step(network, optimizer, replay, 8, torch.device("cpu"))

    for key in ("auxiliary", "solve", "crossings", "moves", "solve_brier", "shadow_mae"):
        assert math.isfinite(metrics[key])
    assert metrics["solve"] > 0.0
    assert metrics["crossings"] >= 0.0
    assert metrics["moves"] >= 0.0


def test_success_only_controller_masks_failed_policy_and_value_but_trains_solve() -> None:
    game = BraidGameConfig(max_len=16, max_strands=3, simplify_budget=12)
    network = make_braid_network(
        game,
        ModelConfig(
            channels=8,
            latent_channels=8,
            residual_blocks=1,
            auxiliary_value_loss_weight=1.0,
            auxiliary_solve_backprop_to_encoder=True,
        ),
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    replay = ReplayBuffer(100, np.random.default_rng(29))
    observation = _observation(game, batch=1)[0].permute(1, 2, 0).numpy()
    replay.add(
        [
            Position(
                observation=observation,
                legal_actions=np.ones(game.action_size, dtype=bool),
                policy=np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32),
                action=0,
                player=1,
                role=1,
                outcome=-1.0,
                solved=0.0,
                episode_seed=30,
            )
        ]
    )

    metrics = train_alphazero_step(
        network,
        optimizer,
        replay,
        4,
        torch.device("cpu"),
        policy_value_success_only=True,
    )

    assert metrics["policy"] == 0.0
    assert metrics["value"] == 0.0
    assert metrics["policy_value_targets"] == 0.0
    assert metrics["solve"] > 0.0
    assert metrics["solve_targets"] == 4.0


def test_shared_auxiliary_only_masks_policy_and_scalar_value_targets() -> None:
    game = BraidGameConfig(max_len=16, max_strands=3, simplify_budget=12)
    network = make_braid_network(
        game, ModelConfig(channels=8, latent_channels=8, residual_blocks=1)
    )
    network.shared_auxiliary_only = True
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    replay = ReplayBuffer(100, np.random.default_rng(3))
    observation = _observation(game, batch=1)[0].permute(1, 2, 0).numpy()
    replay.add(
        [
            Position(
                observation=observation.copy(),
                legal_actions=np.ones(game.action_size, dtype=bool),
                policy=np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32),
                action=0,
                player=1,
                role=1,
                outcome=1.0,
                solved=1.0,
                final_crossing_changes=2.0,
                final_moves=7.0,
                episode_seed=4,
                shared_witness=True,
            )
        ]
    )

    metrics = train_alphazero_step(network, optimizer, replay, 4, torch.device("cpu"))

    assert metrics["policy"] == 0.0
    assert metrics["value"] == 0.0
    assert metrics["solve"] > 0.0


def test_objective_censored_positions_train_only_conditional_solve() -> None:
    game = BraidGameConfig(
        max_len=16,
        max_strands=3,
        simplify_budget=12,
        objective_budget_channel=True,
    )
    network = make_braid_network(
        game, ModelConfig(channels=8, latent_channels=8, residual_blocks=1)
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    replay = ReplayBuffer(100, np.random.default_rng(5))
    observation = _observation(game, batch=1)[0].permute(1, 2, 0).numpy()
    replay.add(
        [
            Position(
                observation=observation,
                legal_actions=np.ones(game.action_size, dtype=bool),
                policy=np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32),
                action=0,
                player=1,
                role=1,
                outcome=-1.0,
                solved=0.0,
                final_crossing_changes=2.0,
                final_moves=7.0,
                objective_censored=True,
            )
        ]
    )

    metrics = train_alphazero_step(network, optimizer, replay, 4, torch.device("cpu"))

    assert metrics["policy"] == 0.0
    assert metrics["value"] == 0.0
    assert metrics["solve"] > 0.0
    assert metrics["crossings"] == 0.0
    assert metrics["moves"] == 0.0


@pytest.mark.parametrize(
    "serial_options",
    [
        {"serial_window": 7, "serial_act_width": 7},
        {"serial_window": 7, "serial_act_width": 1, "serial_tape_symbols": 4},
        {"serial_window": 11, "serial_act_width": 11},
    ],
)
def test_roster_solve_loss_reaches_encoder_but_cost_loss_does_not(
    serial_options: dict[str, int],
) -> None:
    game = BraidGameConfig(
        max_len=16,
        max_strands=3,
        simplify_budget=12,
        objective_budget_channel=True,
        **serial_options,
    )
    network = make_braid_network(
        game,
        ModelConfig(
            channels=8,
            latent_channels=8,
            residual_blocks=1,
            auxiliary_solve_backprop_to_encoder=True,
        ),
    ).train()
    observation = torch.randn(4, game.observation_channels, 1, game.width)
    encoder_weight = network.representation.net[0].weight

    auxiliary = network.forward_with_auxiliary(observation)[2]
    (auxiliary[1].sum() + auxiliary[2].sum()).backward()
    assert encoder_weight.grad is None or torch.count_nonzero(encoder_weight.grad) == 0

    network.zero_grad(set_to_none=True)
    solve_logits = network.forward_with_auxiliary(observation)[2][0]
    solve_logits.sum().backward()
    assert encoder_weight.grad is not None
    assert torch.count_nonzero(encoder_weight.grad) > 0


def test_budget_monotonic_training_updates_migrated_budget_input() -> None:
    game = BraidGameConfig(
        max_len=16,
        max_strands=3,
        simplify_budget=12,
        serial_window=7,
        objective_budget_channel=True,
    )
    network = make_braid_network(
        game,
        ModelConfig(
            channels=8,
            latent_channels=8,
            residual_blocks=1,
            auxiliary_solve_backprop_to_encoder=True,
            auxiliary_budget_monotonic_weight=1.0,
        ),
    )
    budget_weight = network.representation.net[0].weight[:, -1]
    with torch.no_grad():
        budget_weight.zero_()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    replay = ReplayBuffer(100, np.random.default_rng(6))
    observation = _observation(game, batch=1)[0].permute(1, 2, 0).numpy()
    observation[..., -1] = 0.5
    replay.add(
        [
            Position(
                observation=observation,
                legal_actions=np.ones(game.action_size, dtype=bool),
                policy=np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32),
                action=0,
                player=1,
                role=1,
                solved=0.0,
                objective_censored=True,
                episode_seed=7,
            )
        ]
    )

    metrics = train_alphazero_step(network, optimizer, replay, 8, torch.device("cpu"))

    assert metrics["budget_monotonic"] > 0.0
    assert torch.count_nonzero(budget_weight) > 0


def test_budget_finetuning_freezes_batchnorm_running_statistics() -> None:
    game = BraidGameConfig(
        max_len=16,
        max_strands=3,
        simplify_budget=12,
        serial_window=7,
        objective_budget_channel=True,
    )
    network = make_braid_network(
        game,
        ModelConfig(
            channels=8,
            latent_channels=8,
            residual_blocks=1,
            auxiliary_solve_backprop_to_encoder=True,
            auxiliary_budget_conditioning=True,
            freeze_batchnorm_stats=True,
        ),
    )
    batchnorm = network.representation.net[1]
    before_mean = batchnorm.running_mean.clone()
    before_variance = batchnorm.running_var.clone()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    replay = ReplayBuffer(100, np.random.default_rng(8))
    observation = torch.randn(game.observation_channels, 1, game.width).permute(1, 2, 0).numpy()
    replay.add(
        [
            Position(
                observation=observation,
                legal_actions=np.ones(game.action_size, dtype=bool),
                policy=np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32),
                action=0,
                player=1,
                role=1,
                solved=0.0,
                objective_censored=True,
                episode_seed=9,
            )
        ]
    )

    train_alphazero_step(network, optimizer, replay, 8, torch.device("cpu"))

    torch.testing.assert_close(batchnorm.running_mean, before_mean)
    torch.testing.assert_close(batchnorm.running_var, before_variance)


def test_policy_value_preservation_teacher_is_used_without_registration() -> None:
    game = BraidGameConfig(
        max_len=16,
        max_strands=3,
        simplify_budget=12,
        serial_window=7,
        objective_budget_channel=True,
    )
    network = make_braid_network(
        game,
        ModelConfig(
            channels=8,
            latent_channels=8,
            residual_blocks=1,
            policy_value_preservation_weight=1.0,
        ),
    )
    teacher = copy.deepcopy(network).eval()
    object.__setattr__(network, "_policy_value_preservation_teacher", teacher)
    with torch.no_grad():
        network.global_policy.bias.add_(0.5 * torch.randn_like(network.global_policy.bias))
        network.value[0].bias.add_(0.5)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    replay = ReplayBuffer(100, np.random.default_rng(18))
    observation = _observation(game, batch=1)[0].permute(1, 2, 0).numpy()
    replay.add(
        [
            Position(
                observation=observation,
                legal_actions=np.ones(game.action_size, dtype=bool),
                policy=np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32),
                action=0,
                player=1,
                role=1,
                outcome=0.0,
                solved=0.0,
                episode_seed=19,
            )
        ]
    )

    metrics = train_alphazero_step(network, optimizer, replay, 4, torch.device("cpu"))

    assert metrics["preservation_policy"] > 0.0
    assert metrics["preservation_value"] > 0.0
    assert not any("preservation_teacher" in key for key in network.state_dict())


def test_solve_conditioning_receives_budget_costs_and_manual_objective() -> None:
    game = BraidGameConfig(
        max_len=16,
        max_strands=3,
        simplify_budget=12,
        serial_window=7,
        objective_budget_channel=True,
    )
    network = make_braid_network(
        game,
        ModelConfig(
            channels=8,
            latent_channels=8,
            residual_blocks=1,
            auxiliary_solve_backprop_to_encoder=True,
            auxiliary_budget_conditioning=True,
        ),
    ).eval()
    member = network.auxiliary.members[0]
    with torch.no_grad():
        for parameter in member.cost.parameters():
            parameter.zero_()
        member.cost[-1].bias.copy_(
            torch.tensor(
                [
                    math.log(math.expm1(2.0)),
                    math.log(math.expm1(3.0)),
                ]
            )
        )
    captured = []
    hook = member.solve_conditioning[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    observation = _observation(game, batch=1)
    observation[:, network.ratio_channel] = math.log(10.0) / 5.0
    observation[:, -1] = 0.4

    with torch.inference_mode():
        network.forward_with_auxiliary(observation)
    hook.remove()

    expected = torch.tensor(
        [0.4, 2.0 / 12.0, 3.0 / 12.0, (10.0 * 2.0 + 3.0) / (11.0 * 12.0)]
    )
    torch.testing.assert_close(captured[0][0, -4:], expected)
