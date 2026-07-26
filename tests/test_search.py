import numpy as np

from pgx_mcts_bench.config import GameConfig, ModelConfig, SearchConfig
from pgx_mcts_bench.game import Go6x6
from pgx_mcts_bench.networks import AlphaZeroNet, MuZeroNet
from pgx_mcts_bench.search import NeuralMCTS


def _assert_search_returns_legal(network) -> None:
    game = Go6x6(GameConfig())
    root = game.reset(0)
    search = NeuralMCTS(game, network, SearchConfig(simulations=2))
    result = search.run(
        root.state,
        root.observation,
        root.legal_actions,
        np.random.default_rng(0),
        temperature=0.0,
        add_root_noise=False,
    )
    assert root.legal_actions[result.action]
    assert np.isclose(result.policy.sum(), 1.0)
    assert result.visits.sum() == 2


def test_alphazero_search() -> None:
    _assert_search_returns_legal(
        AlphaZeroNet(GameConfig(), ModelConfig(channels=4, residual_blocks=1))
    )


def test_muzero_search() -> None:
    _assert_search_returns_legal(
        MuZeroNet(
            GameConfig(),
            ModelConfig(channels=4, latent_channels=4, residual_blocks=1),
        )
    )


def test_muzero_learned_rules_ablation_runs() -> None:
    game = Go6x6(GameConfig())
    root = game.reset(0)
    network = MuZeroNet(
        GameConfig(),
        ModelConfig(channels=4, latent_channels=4, residual_blocks=1),
    )
    search = NeuralMCTS(
        game,
        network,
        SearchConfig(simulations=2, muzero_exact_rules=False),
    )
    result = search.run(
        root.state,
        root.observation,
        root.legal_actions,
        np.random.default_rng(0),
        temperature=0.0,
        add_root_noise=False,
    )
    assert root.legal_actions[result.action]


def test_batched_search_returns_one_result_per_root() -> None:
    game = Go6x6(GameConfig())
    roots = [game.reset(0), game.reset(1), game.reset(2)]
    network = AlphaZeroNet(
        GameConfig(),
        ModelConfig(channels=4, residual_blocks=1),
    )
    search = NeuralMCTS(game, network, SearchConfig(simulations=3))
    results = search.run_batch(
        states=[root.state for root in roots],
        observations=[root.observation for root in roots],
        legal_actions=[root.legal_actions for root in roots],
        rngs=[np.random.default_rng(seed) for seed in range(3)],
        temperatures=[0.0, 0.0, 0.0],
        add_root_noise=False,
    )
    assert len(results) == 3
    for root, result in zip(roots, results, strict=True):
        assert root.legal_actions[result.action]
        assert result.visits.sum() == 3
