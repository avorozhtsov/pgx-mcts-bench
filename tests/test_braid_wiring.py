"""Tests for wiring the rf_knots braid environment into this benchmark."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgx_mcts_bench.braid_progress import BraidProgress
from pgx_mcts_bench.config import (
    BraidGameConfig,
    ExperimentConfig,
    GameConfig,
    ModelConfig,
    SearchConfig,
    TrainConfig,
)
from pgx_mcts_bench.game import BraidUnknotGame, Go6x6, make_game
from pgx_mcts_bench.networks import BraidAlphaZeroNet, BraidPolicyHead, MuZeroNet, PolicyValueNet
from pgx_mcts_bench.search import NeuralMCTS, Node
from pgx_mcts_bench.training import evaluate_against_random, train_agent

SMALL = BraidGameConfig(
    max_len=12, max_strands=4, scramble_budget=3, simplify_budget=6
)


def _experiment(**train_overrides) -> ExperimentConfig:
    train = {
        "iterations": 1,
        "selfplay_games": 2,
        "train_steps": 1,
        "batch_size": 2,
        "seed": 0,
        "device": "cpu",
    }
    train.update(train_overrides)
    return ExperimentConfig(
        game=SMALL,
        search=SearchConfig(simulations=2),
        model=ModelConfig(channels=4, latent_channels=4),
        train=TrainConfig(**train),
    )


# -- configuration -------------------------------------------------------------


def test_braid_config_matches_the_environment() -> None:
    from rf_knots.actions import ActionSpec
    from rf_knots.env import BraidUnknot

    env = BraidUnknot(SMALL.to_braid_config())
    assert SMALL.action_size == ActionSpec(SMALL.max_len, SMALL.max_strands).num_actions
    assert SMALL.action_size == env.num_actions
    assert SMALL.observation_channels == env.num_channels
    assert (SMALL.height, SMALL.width) == (1, SMALL.max_len)
    assert SMALL.cells == SMALL.max_len


def test_experiment_config_round_trips_through_json() -> None:
    config = _experiment()
    restored = ExperimentConfig.from_dict(config.to_dict())
    assert restored == config
    assert isinstance(restored.game, BraidGameConfig)


def test_go_config_without_kind_still_loads() -> None:
    """Checkpoints written before the braid environment existed have no `kind`."""
    payload = ExperimentConfig().to_dict()
    del payload["game"]["kind"]
    restored = ExperimentConfig.from_dict(payload)
    assert isinstance(restored.game, GameConfig)
    assert restored == ExperimentConfig()


def test_make_game_dispatches() -> None:
    assert isinstance(make_game(SMALL), BraidUnknotGame)
    assert isinstance(make_game(GameConfig()), Go6x6)


# -- adapter -------------------------------------------------------------------


def test_adapter_shapes_and_roles() -> None:
    game = BraidUnknotGame(SMALL)
    transition = game.reset(0)
    assert transition.observation.shape == (1, SMALL.max_len, SMALL.observation_channels)
    assert transition.legal_actions.shape == (SMALL.action_size,)
    assert not transition.terminated
    assert transition.player in (0, 1)
    assert game.first_role_player(transition.state) == transition.player
    info = game.state_info(transition.state)
    assert set(info) == {"player", "move_count", "consecutive_passes"}
    assert info["player"] == transition.player


def test_adapter_rejects_illegal_actions() -> None:
    game = BraidUnknotGame(SMALL)
    transition = game.reset(0)
    illegal = int(np.flatnonzero(~transition.legal_actions)[0])
    with pytest.raises(ValueError):
        game.step(transition.state, illegal)


def test_players_do_not_alternate() -> None:
    """The property that forced the value-backup change."""
    game = BraidUnknotGame(SMALL)
    transition = game.reset(0)
    players = [transition.player]
    for _ in range(SMALL.scramble_budget):
        action = int(np.flatnonzero(transition.legal_actions)[0])
        transition = game.step(transition.state, action)
        players.append(transition.player)
    scrambler = players[0]
    assert players[: SMALL.scramble_budget] == [scrambler] * SMALL.scramble_budget
    assert players[-1] == 1 - scrambler


def test_from_word_starts_the_simplifier() -> None:
    game = BraidUnknotGame(SMALL)
    transition = game.from_word([1, 2], 3)
    assert not transition.terminated
    assert int(np.asarray(transition.state._phase)) == 1


# -- value backup --------------------------------------------------------------


def test_perspective_flips_only_on_player_change() -> None:
    assert NeuralMCTS._perspective(Node(1.0, player=0), Node(1.0, player=1)) == -1.0
    assert NeuralMCTS._perspective(Node(1.0, player=1), Node(1.0, player=0)) == -1.0
    assert NeuralMCTS._perspective(Node(1.0, player=0), Node(1.0, player=0)) == 1.0
    # Unevaluated children have value 0, so the sign is immaterial; -1 keeps the
    # historical Go behaviour byte-for-byte.
    assert NeuralMCTS._perspective(Node(1.0, player=0), Node(1.0)) == -1.0
    assert NeuralMCTS._perspective(Node(1.0), Node(1.0, player=0)) == -1.0


def test_backup_within_a_phase_does_not_negate() -> None:
    game = BraidUnknotGame(SMALL)
    search = NeuralMCTS(game, BraidAlphaZeroNet(SMALL, ModelConfig(channels=4)), SearchConfig())
    same = [Node(1.0, player=0), Node(1.0, player=0), Node(1.0, player=0)]
    search._backup(same, 0.5)
    assert same[0].value == pytest.approx(0.5)

    alternating = [Node(1.0, player=0), Node(1.0, player=1), Node(1.0, player=0)]
    search._backup(alternating, 0.5)
    assert alternating[0].value == pytest.approx(0.5)
    assert alternating[1].value == pytest.approx(-0.5)


# -- policy head ---------------------------------------------------------------


def test_policy_head_blocks_align_with_the_action_space() -> None:
    """The head's output order must match `rf_knots`' legal-action mask exactly.

    A misalignment here would be invisible: training would still run, the loss
    would still fall, and the agent would be learning a permuted action space.
    So the alignment is checked block by block, by zeroing the head and lighting
    up one output channel at a time.
    """
    from rf_knots.actions import (
        BRAID,
        COMMUTE,
        CROSSING_CHANGE,
        DESTABILIZE,
        INSERT,
        PASS,
        REDUCE,
        ActionSpec,
    )

    spec = ActionSpec(SMALL.max_len, SMALL.max_strands)
    head = BraidPolicyHead(4, SMALL)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    hidden = torch.ones(1, 4, 1, SMALL.max_len)

    expected_blocks = [(REDUCE, 1, 1), (COMMUTE, 1, 1), (BRAID, 1, 1)]
    for generator in range(1, SMALL.max_strands):
        for sign in (1, -1):
            expected_blocks.append((INSERT, generator, sign))

    for channel, (kind, generator, sign) in enumerate(expected_blocks):
        with torch.no_grad():
            head.positional.bias.zero_()
            head.positional.bias[channel] = 1.0
        logits = head(hidden)[0].detach().numpy()
        lit = set(np.flatnonzero(logits == 1.0).tolist())
        wanted = {
            spec.encode(kind, position=p, generator=generator, sign=sign)
            for p in range(SMALL.max_len)
        }
        assert lit == wanted, f"channel {channel} does not cover block {kind}"

    with torch.no_grad():
        head.positional.bias.zero_()
        head.crossing.bias[0] = 1.0
    logits = head(hidden)[0].detach().numpy()
    assert set(np.flatnonzero(logits == 1.0).tolist()) == {
        spec.encode(CROSSING_CHANGE, position=p) for p in range(SMALL.max_len)
    }

    with torch.no_grad():
        head.crossing.bias.zero_()
        head.singletons.bias[0] = 1.0
    logits = head(hidden)[0].detach().numpy()
    assert set(np.flatnonzero(logits == 1.0).tolist()) == {spec.encode(DESTABILIZE)}

    with torch.no_grad():
        head.singletons.bias.zero_()
        head.singletons.bias[-1] = 1.0
    logits = head(hidden)[0].detach().numpy()
    assert set(np.flatnonzero(logits == 1.0).tolist()) == {spec.encode(PASS)}


def test_braid_network_shapes_and_dispatch() -> None:
    network = BraidAlphaZeroNet(SMALL, ModelConfig(channels=8))
    assert isinstance(network, PolicyValueNet)
    assert not isinstance(network, MuZeroNet)
    observation = torch.zeros(3, SMALL.observation_channels, 1, SMALL.max_len)
    policy, value = network(observation)
    assert policy.shape == (3, SMALL.action_size)
    assert value.shape == (3,)
    assert torch.all(value.abs() <= 1.0)


def test_muzero_is_refused_for_braid() -> None:
    with pytest.raises(NotImplementedError):
        MuZeroNet(SMALL, ModelConfig(channels=4))


# -- end to end ----------------------------------------------------------------


def test_training_runs_end_to_end(tmp_path) -> None:
    config = _experiment()
    seen: list[int] = []
    agent = train_agent(
        "alphazero",
        config,
        checkpoint_dir=tmp_path / "checkpoints",
        iteration_hook=lambda iteration, network: seen.append(iteration) or "ok",
    )
    assert seen == [1]
    assert isinstance(agent.network, BraidAlphaZeroNet)
    assert len(agent.history) == 1
    # A full game is K scramble plies plus at least one Simplifier ply. Anything
    # near 1 would mean a player is forfeiting immediately.
    assert agent.history[0]["mean_game_length"] > config.game.scramble_budget

    baseline = evaluate_against_random(agent, 2, seed=1)
    assert 0.0 <= baseline["first_role_win_rate"] <= 1.0
    assert 0.0 <= baseline["second_role_win_rate"] <= 1.0


def test_progress_tracker_produces_a_report(tmp_path) -> None:
    config = _experiment()
    progress = BraidProgress(config, tmp_path / "run", anchors=2, bfs_depth=4)
    assert len(progress.instances) == 2
    for word, strands in progress.instances:
        assert word != ()
        assert 1 <= strands <= SMALL.max_strands

    network = BraidAlphaZeroNet(SMALL, config.model)
    report = progress.evaluate(1, network)
    assert 0.0 <= report.solve_rate <= 1.0
    assert len(report.attempts) == 2
    markdown = (tmp_path / "run" / "progress.md").read_text()
    assert "Braid unknotting progress" in markdown
    assert "<svg" in markdown
    assert (tmp_path / "run" / "progress.json").exists()
    caches = list(tmp_path.glob(".anchor-optima-*.json"))
    assert len(caches) == 1, "anchor optima must be cached, not recomputed per run"
    second = BraidProgress(config, tmp_path / "run2", anchors=2, bfs_depth=4)
    assert second.optimal == progress.optimal


# -- the fixes -----------------------------------------------------------------


def test_replay_balances_roles() -> None:
    """The braid buffer is ~85% Simplifier positions without this."""
    from pgx_mcts_bench.data import Position, ReplayBuffer

    rng = np.random.default_rng(0)
    replay = ReplayBuffer(1000, rng)
    obs = np.zeros((1, 4, 3), dtype=np.float32)
    legal = np.ones(5, dtype=bool)
    policy = np.ones(5, dtype=np.float32) / 5
    scrambler = [Position(obs, legal, policy, 0, player=0, role=0) for _ in range(3)]
    simplifier = [Position(obs, legal, policy, 0, player=1, role=1) for _ in range(97)]
    replay.add(scrambler + simplifier)

    plain = replay.sample_positions(400)
    assert sum(p.role == 0 for p in plain) < 100  # tracks the 3% population share

    balanced = replay.sample_positions(400, balanced=True)
    assert sum(p.role == 0 for p in balanced) == 200


def test_replay_balanced_falls_back_when_one_role_is_missing() -> None:
    from pgx_mcts_bench.data import Position, ReplayBuffer

    replay = ReplayBuffer(100, np.random.default_rng(0))
    obs = np.zeros((1, 4, 3), dtype=np.float32)
    replay.add([Position(obs, np.ones(5, bool), np.ones(5, np.float32) / 5, 0, 0, role=0)])
    assert len(replay.sample_positions(8, balanced=True)) == 8


def test_top_k_expansion_limits_and_renormalises() -> None:
    game = BraidUnknotGame(SMALL)
    search = NeuralMCTS(
        game,
        BraidAlphaZeroNet(SMALL, ModelConfig(channels=4)),
        SearchConfig(max_children=4),
    )
    legal = np.zeros(SMALL.action_size, dtype=bool)
    legal[:20] = True
    logits = np.linspace(0.0, 5.0, SMALL.action_size)
    node = Node(1.0)
    search._expand_children(node, logits, legal)
    assert len(node.children) == 4
    assert set(node.children) == {16, 17, 18, 19}  # the four highest logits
    assert sum(child.prior for child in node.children.values()) == pytest.approx(1.0)


def test_unlimited_expansion_is_the_default() -> None:
    game = BraidUnknotGame(SMALL)
    search = NeuralMCTS(game, BraidAlphaZeroNet(SMALL, ModelConfig(channels=4)), SearchConfig())
    legal = np.zeros(SMALL.action_size, dtype=bool)
    legal[:20] = True
    node = Node(1.0)
    search._expand_children(node, np.zeros(SMALL.action_size), legal)
    assert len(node.children) == 20


def test_root_noise_can_be_set_per_root() -> None:
    """Needed so exploration can be applied to one role and not the other."""
    game = BraidUnknotGame(SMALL)
    network = BraidAlphaZeroNet(SMALL, ModelConfig(channels=4))
    search = NeuralMCTS(game, network, SearchConfig(simulations=2))
    transitions = [game.reset(seed) for seed in (0, 1)]
    rngs = [np.random.default_rng(0), np.random.default_rng(0)]
    results = search.run_batch(
        states=[t.state for t in transitions],
        observations=[t.observation for t in transitions],
        legal_actions=[t.legal_actions for t in transitions],
        rngs=rngs,
        temperatures=[0.0, 0.0],
        add_root_noise=[True, False],
    )
    assert len(results) == 2

    with pytest.raises(ValueError):
        search.run_batch(
            states=[t.state for t in transitions],
            observations=[t.observation for t in transitions],
            legal_actions=[t.legal_actions for t in transitions],
            rngs=rngs,
            temperatures=[0.0, 0.0],
            add_root_noise=[True],
        )


def test_scaled_dirichlet_alpha_tracks_the_branching_factor() -> None:
    """A fixed alpha cannot fit both tiers: branching varies by an order of
    magnitude, and AlphaZero scales alpha inversely with it."""

    class SpyRng:
        def __init__(self) -> None:
            self.alphas: list[float] = []
            self.inner = np.random.default_rng(0)

        def dirichlet(self, alphas):
            self.alphas.append(float(alphas[0]))
            return self.inner.dirichlet(alphas)

    game = BraidUnknotGame(SMALL)
    network = BraidAlphaZeroNet(SMALL, ModelConfig(channels=4))

    scaled = NeuralMCTS(game, network, SearchConfig(root_dirichlet_scale=10.0))
    rng = SpyRng()
    root = Node(1.0)
    root.children = {action: Node(0.25) for action in range(20)}
    scaled._add_root_noise(root, rng)  # type: ignore[arg-type]
    assert rng.alphas == [pytest.approx(10.0 / 20)]

    fixed = NeuralMCTS(game, network, SearchConfig(root_dirichlet_alpha=0.3))
    rng = SpyRng()
    root = Node(1.0)
    root.children = {action: Node(0.25) for action in range(20)}
    fixed._add_root_noise(root, rng)  # type: ignore[arg-type]
    assert rng.alphas == [pytest.approx(0.3)]


def test_scrambler_difficulty_beats_or_matches_the_random_baseline() -> None:
    from pgx_mcts_bench.braid_progress import evaluate_scrambler_difficulty
    from pgx_mcts_bench.training import TrainedAgent

    config = _experiment()
    agent = TrainedAgent("alphazero", BraidAlphaZeroNet(SMALL, config.model), [], config)
    stats = evaluate_scrambler_difficulty(agent, 4, seed=0, bfs_depth=4)
    assert stats["games"] == 4
    assert 0.0 <= stats["beyond_cutoff"] <= 1.0
    assert stats["mean_word_length"] >= 0.0


def test_the_whole_braid_network_is_length_agnostic() -> None:
    """Weights trained at one word capacity must load and run at any other.

    Every parameter depends on the receptive field (11 letters), not on L. A
    `Flatten -> Linear(L, ...)` anywhere breaks that, which is exactly what the
    value head used to do.
    """
    small = BraidGameConfig(max_len=32, max_strands=5)
    large = BraidGameConfig(max_len=128, max_strands=5)
    model = ModelConfig(channels=16, latent_channels=16)
    assert model.braid_value_head == "masked", "the default must be length-agnostic"

    trained = BraidAlphaZeroNet(small, model)
    grown = BraidAlphaZeroNet(large, model)
    shapes_small = {k: v.shape for k, v in trained.state_dict().items()}
    shapes_large = {k: v.shape for k, v in grown.state_dict().items()}
    mismatched = {k for k in shapes_small if shapes_small[k] != shapes_large[k]}
    assert not mismatched, f"length-dependent parameters: {mismatched}"

    grown.load_state_dict(trained.state_dict())
    policy, value = grown(torch.zeros(2, large.observation_channels, 1, large.max_len))
    assert policy.shape == (2, large.action_size)
    assert value.shape == (2,)
    assert large.action_size > small.action_size


def test_flattened_value_head_is_length_dependent() -> None:
    """The A/B alternative, and the reason it is not the default: it reads every
    position directly, so its weights are tied to one word capacity."""
    model = ModelConfig(channels=16, latent_channels=16, braid_value_head="flat")
    small = BraidAlphaZeroNet(BraidGameConfig(max_len=32, max_strands=5), model)
    large = BraidAlphaZeroNet(BraidGameConfig(max_len=128, max_strands=5), model)
    shapes_small = {k: v.shape for k, v in small.state_dict().items()}
    shapes_large = {k: v.shape for k, v in large.state_dict().items()}
    assert {k for k in shapes_small if shapes_small[k] != shapes_large[k]}

    game = BraidGameConfig(max_len=32, max_strands=5)
    policy, value = small(torch.zeros(2, game.observation_channels, 1, game.max_len))
    assert policy.shape == (2, game.action_size)
    assert value.shape == (2,)


def test_masked_value_head_is_length_agnostic_and_finite() -> None:
    """Masked pooling keeps the head independent of L while ignoring padding.

    Plain mean pooling averages a 5-letter word over all L slots; the A/B showed
    that ~6x dilution is enough to make runs collapse.
    """
    model = ModelConfig(channels=16, latent_channels=16, braid_value_head="masked")
    small = BraidAlphaZeroNet(BraidGameConfig(max_len=32, max_strands=5), model)
    large = BraidAlphaZeroNet(BraidGameConfig(max_len=128, max_strands=5), model)
    assert {k: v.shape for k, v in small.state_dict().items()} == {
        k: v.shape for k, v in large.state_dict().items()
    }
    large.load_state_dict(small.state_dict())

    game = BraidGameConfig(max_len=32, max_strands=5)
    from rf_knots.env import BraidUnknot

    env = BraidUnknot(game.to_braid_config())
    obs = np.asarray(env.init_from_word([1, 2, 1], n=3).observation)
    x = torch.from_numpy(obs).float().permute(1, 0)[None, :, None, :]
    small.eval()
    with torch.inference_mode():
        policy, value = small(x)
    assert policy.shape == (1, game.action_size)
    assert torch.isfinite(value).all() and value.abs().item() <= 1.0


def test_curriculum_starts_low_and_climbs_only_on_wins() -> None:
    """A run whose Simplifier never wins has every target reading "it lost", so
    nothing points toward solving. Four of 42 runs died that way at fixed K=6."""
    from dataclasses import replace as _replace

    game = _replace(SMALL, scramble_budget=5)
    config = ExperimentConfig(
        game=game,
        search=SearchConfig(simulations=2),
        model=ModelConfig(channels=4, latent_channels=4),
        train=TrainConfig(
            iterations=3,
            selfplay_games=2,
            train_steps=1,
            batch_size=2,
            device="cpu",
            curriculum_start_k=1,
            curriculum_promote_at=0.0,  # promote every iteration
        ),
    )
    agent = train_agent("alphazero", config)
    ks = [row["scramble_k"] for row in agent.history]
    assert ks == [2.0, 3.0, 4.0], f"K should climb one step per iteration, got {ks}"
    assert all("simplifier_wins" in row for row in agent.history)


def test_curriculum_is_off_by_default_and_reports_the_target() -> None:
    config = _experiment()
    assert config.train.curriculum_start_k == 0
    agent = train_agent("alphazero", config)
    assert agent.history[0]["scramble_k"] == float(SMALL.scramble_budget)
