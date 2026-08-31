import numpy as np
import torch

from pgx_mcts_bench.config import GameConfig, ModelConfig, SearchConfig
from pgx_mcts_bench.game import Go6x6
from pgx_mcts_bench.networks import AlphaZeroNet, MuZeroNet, PolicyValueNet
from pgx_mcts_bench.search import NeuralMCTS, Node, _masked_softmax


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


def test_alphazero_inference_cache_deduplicates_and_invalidates() -> None:
    game = Go6x6(GameConfig())
    root = game.reset(0)
    network = AlphaZeroNet(GameConfig(), ModelConfig(channels=4, residual_blocks=1))
    search = NeuralMCTS(game, network, SearchConfig(simulations=1))
    original = network.forward
    inferred_batch_sizes = []

    def counted(observation):
        inferred_batch_sizes.append(len(observation))
        return original(observation)

    network.forward = counted
    search._ensure_inference_cache_fresh()
    for _ in range(2):
        search._expand_alphazero_batch(
            [Node(1.0, state=root.state) for _ in range(3)],
            [root.observation] * 3,
            [root.legal_actions] * 3,
        )
    assert inferred_batch_sizes == [1]
    assert search.inference_cache_stats == {"hits": 5, "misses": 1, "evictions": 0}

    with torch.no_grad():
        next(network.parameters()).add_(1.0)
    search._ensure_inference_cache_fresh()
    search._expand_alphazero_batch(
        [Node(1.0, state=root.state)],
        [root.observation],
        [root.legal_actions],
    )
    assert inferred_batch_sizes == [1, 1]


def test_state_conditioned_cache_does_not_merge_identical_local_observations() -> None:
    game = Go6x6(GameConfig())
    root = game.reset(0)

    class StateAware(PolicyValueNet):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.forwarded_states = []

        def inference_context_key(self, state, _game):
            return bytes([int(state._step_count)])

        def forward_with_states(self, observation, states, _game):
            self.forwarded_states.extend(states)
            return (
                observation.new_zeros((len(states), 37)) + self.anchor,
                observation.new_zeros((len(states),)) + self.anchor,
            )

        def forward(self, observation):
            raise AssertionError("search must use state-conditioned inference")

    network = StateAware()
    search = NeuralMCTS(game, network, SearchConfig(simulations=1))
    alternate = root.state.replace(_step_count=root.state._step_count + 1)
    search._expand_alphazero_batch(
        [Node(1.0, state=root.state), Node(1.0, state=alternate)],
        [root.observation, root.observation],
        [root.legal_actions, root.legal_actions],
    )
    assert len(network.forwarded_states) == 2
    assert search.inference_cache_stats["misses"] == 2


def test_masked_softmax_does_not_exponentiate_illegal_logits() -> None:
    probabilities = _masked_softmax(
        np.array([0.0, 1_000.0]),
        np.array([True, False]),
    )
    np.testing.assert_array_equal(probabilities, [1.0, 0.0])
