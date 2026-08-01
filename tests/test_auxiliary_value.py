from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pgx_mcts_bench.config import BraidGameConfig, ModelConfig
from pgx_mcts_bench.data import Position, ReplayBuffer
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.training import train_alphazero_step


def _observation(game: BraidGameConfig, batch: int = 2) -> torch.Tensor:
    observation = torch.zeros(
        batch, game.observation_channels, 1, game.width, dtype=torch.float32
    )
    observation[:, 2 * (game.max_strands - 1), :, :] = 1.0
    return observation


@pytest.mark.parametrize(
    "game",
    [
        BraidGameConfig(max_len=16, max_strands=3, simplify_budget=12),
        BraidGameConfig(
            max_len=16, max_strands=3, simplify_budget=12, serial_window=7
        ),
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
