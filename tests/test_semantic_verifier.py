from __future__ import annotations

import pytest
import torch

from pgx_mcts_bench.config import BraidGameConfig, GameConfig, ModelConfig
from pgx_mcts_bench.game import Go6x6, make_game
from pgx_mcts_bench.networks import AlphaZeroNet, MuZeroNet, make_braid_network
from pgx_mcts_bench.semantic_verifier import SemanticBraidVerifier
from pgx_mcts_bench.serial_braid import SerialBraidGame


def test_go6x6_and_its_two_network_families_remain_available() -> None:
    config = GameConfig()
    assert isinstance(make_game(config), Go6x6)
    observation = torch.zeros(1, config.observation_channels, 6, 6)
    policy, _ = AlphaZeroNet(config, ModelConfig(channels=4))(observation)
    _, muzero_policy, *_ = MuZeroNet(
        config, ModelConfig(channels=4, latent_channels=4)
    ).initial_inference(observation)
    assert policy.shape == muzero_policy.shape == (1, 37)


def test_braid_factory_exposes_only_bounded_serial_game() -> None:
    config = BraidGameConfig(generator_max_crossings=3, generator_max_scramble=1)
    assert isinstance(make_game(config), SerialBraidGame)
    transition = make_game(config).reset(0)
    observation = torch.from_numpy(transition.observation).permute(2, 0, 1)[None]
    policy, _ = make_braid_network(config, ModelConfig(channels=4))(observation)
    assert policy.shape == (1, config.action_size)
    with pytest.raises(ValueError, match="serial_window"):
        BraidGameConfig(serial_window=0)
    with pytest.raises(ValueError, match="serial_internal_horizon"):
        BraidGameConfig(serial_internal_horizon=0)


def test_semantic_verifier_replays_without_a_trainable_parallel_game() -> None:
    verifier = SemanticBraidVerifier.from_config(BraidGameConfig())
    state = verifier.state((1,), 2)
    destabilize = next(
        action
        for action in verifier.legal_actions(state, allow_crossing_change=False)
        if verifier.spec.describe(action).startswith("DESTABILIZE")
    )
    final = verifier.apply(state, destabilize, allow_crossing_change=False)
    witness = verifier.verify_states((state, final))
    witness.verify()
    assert final.word == ()
    assert witness.moves == 1
