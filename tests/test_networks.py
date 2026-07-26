import torch

from pgx_mcts_bench.config import GameConfig, ModelConfig
from pgx_mcts_bench.networks import AlphaZeroNet, MuZeroNet


def test_alphazero_shapes() -> None:
    game = GameConfig()
    network = AlphaZeroNet(game, ModelConfig(channels=4, residual_blocks=1))
    policy, value = network(torch.zeros(2, game.observation_channels, 6, 6))
    assert policy.shape == (2, 37)
    assert value.shape == (2,)


def test_muzero_initial_and_recurrent_shapes() -> None:
    game = GameConfig()
    model = ModelConfig(channels=4, latent_channels=4, residual_blocks=1)
    network = MuZeroNet(game, model)
    observation = torch.zeros(2, game.observation_channels, 6, 6)
    hidden, policy, value, legal, terminal = network.initial_inference(observation)
    assert hidden.shape == (2, 4, 6, 6)
    assert policy.shape == legal.shape == (2, 37)
    assert value.shape == terminal.shape == (2,)
    assert torch.all(torch.sigmoid(terminal) < 0.5)
    next_hidden, reward, next_policy, next_value, next_legal, next_terminal = (
        network.recurrent_inference(hidden, torch.tensor([0, 36]))
    )
    assert next_hidden.shape == hidden.shape
    assert reward.shape == next_value.shape == next_terminal.shape == (2,)
    assert next_policy.shape == next_legal.shape == (2, 37)
