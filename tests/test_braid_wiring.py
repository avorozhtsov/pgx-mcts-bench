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


def test_potential_shaping_preserves_solved_and_failed_returns() -> None:
    from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE

    config = BraidGameConfig(
        max_len=12,
        max_strands=4,
        scramble_budget=1,
        simplify_budget=8,
        allow_crossing_change=True,
        multi_objective=True,
        log_ratio_range=(float(np.log(9.0)), float(np.log(9.0))),
    )
    game = BraidUnknotGame(config)
    network = BraidAlphaZeroNet(config, ModelConfig(channels=4))
    search = NeuralMCTS(
        game,
        network,
        SearchConfig(potential_cost_shaping=True),
    )

    def shaped_return(actions: list[int]) -> tuple[float, float]:
        transition = game.from_word([1], 2, log_ratio=float(np.log(9.0)))
        actor = transition.player
        shaped = 0.0
        for action in actions:
            next_transition = game.step(transition.state, action)
            shaped += search.edge_reward(transition.state, actor, next_transition)
            transition = next_transition
        assert transition.terminated
        terminal = float(game.final_rewards(transition.state)[actor])
        return shaped, terminal

    crossing = config._spec.encode(CROSSING_CHANGE, position=0)
    destabilize = config._spec.encode(DESTABILIZE)
    shaped, terminal = shaped_return([crossing, destabilize])
    assert shaped == pytest.approx(terminal)

    passed, failure = shaped_return([crossing] * config.simplify_budget)
    assert failure == -1.0
    assert passed == pytest.approx(failure)


def test_shaped_value_target_is_remaining_return_and_stays_bounded() -> None:
    from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE

    ratio = 9.0
    config = BraidGameConfig(
        max_len=12,
        max_strands=4,
        scramble_budget=1,
        simplify_budget=8,
        allow_crossing_change=True,
        multi_objective=True,
        log_ratio_range=(float(np.log(ratio)), float(np.log(ratio))),
    )
    game = BraidUnknotGame(config)
    transition = game.from_word([1], 2, log_ratio=float(np.log(ratio)))
    player = transition.player
    assert game.value_potential(transition.state, player) == 0.0
    transition = game.step(
        transition.state,
        config._spec.encode(CROSSING_CHANGE, position=0),
    )
    potential = game.value_potential(transition.state, player)
    assert potential == pytest.approx(-2.0 * (ratio + 1.0) / ((ratio + 1.0) * 8))
    solved = game.step(transition.state, config._spec.encode(DESTABILIZE))
    solved_outcome = float(game.final_rewards(solved.state)[player])
    assert solved_outcome - potential == pytest.approx(0.975)
    # If this same prefix eventually exhausts its budget, the shaped failure
    # target is also bounded and says only that the *remaining* return is bad.
    assert -1.0 - potential == pytest.approx(-0.75)


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




def test_the_whole_braid_network_is_length_agnostic() -> None:
    """Weights trained at one word capacity must load and run at any other.

    Every parameter depends on the receptive field (11 letters), not on L. A
    `Flatten -> Linear(L, ...)` anywhere breaks that, which is exactly what the
    value head used to do.
    """
    small = BraidGameConfig(max_len=32, max_strands=5)
    large = BraidGameConfig(max_len=128, max_strands=5)
    model = ModelConfig(channels=16, latent_channels=16)

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



def test_masked_value_head_is_length_agnostic_and_finite() -> None:
    """Masked pooling keeps the head independent of L while ignoring padding.

    Plain mean pooling averages a 5-letter word over all L slots; the A/B showed
    that ~6x dilution is enough to make runs collapse.
    """
    model = ModelConfig(channels=16, latent_channels=16)
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


# -- serial (moving-window) formulation ----------------------------------------

SERIAL = BraidGameConfig(
    max_len=12, max_strands=4, scramble_budget=3, simplify_budget=18, serial_window=5
)


def test_serial_action_space_does_not_depend_on_L() -> None:
    """The whole point: the serial action space is invariant to word capacity,
    while the parallel one grows linearly with it."""
    from dataclasses import replace as _replace

    serial = [_replace(SERIAL, max_len=length).action_size for length in (12, 64, 256)]
    # head-only: (3 rewrites + 2G inserts + crossing) + 4 singletons + a left and
    # a right action per stride. The stride set is fixed, not derived from
    # max_len -- deriving it would make the count grow like log L.
    expected = (2 * SERIAL.max_strands + 2) + 4 + 2 * len(SERIAL.serial_strides)
    assert serial == [expected] * 3, serial

    parallel = [
        _replace(SERIAL, max_len=length, serial_window=0).action_size
        for length in (12, 64, 256)
    ]
    assert parallel[0] < parallel[1] < parallel[2]
    assert parallel[2] > 40 * serial[2]


def test_serial_game_dispatches_and_observes_a_window() -> None:
    from pgx_mcts_bench.game import make_game
    from pgx_mcts_bench.serial_braid import SerialBraidGame

    game = make_game(SERIAL)
    assert isinstance(game, SerialBraidGame)
    transition = game.reset(0)
    assert transition.observation.shape == (1, SERIAL.serial_window, SERIAL.observation_channels)
    assert transition.legal_actions.shape == (SERIAL.action_size,)
    assert transition.legal_actions.any()
    pgx_state, head = transition.state[0], transition.state[1]
    assert head == 0
    assert game.unwrap(transition.state) is pgx_state


def test_serial_shifts_cost_a_ply() -> None:
    """Free shifts would make the game non-terminating and would hide the very
    cost the serial formulation is being measured for."""
    from pgx_mcts_bench.game import make_game

    game = make_game(SERIAL)
    transition = game.from_word([1, 2, 1, 2, 1], strands=3)
    budget_before = int(np.asarray(game.unwrap(transition.state)._budget))
    shift = SERIAL.action_size - 1
    assert transition.legal_actions[shift]
    after = game.step(transition.state, shift)
    assert int(np.asarray(game.unwrap(after.state)._budget)) == budget_before - 1
    assert after.state[1] != transition.state[1], "the head must actually move"


def test_serial_edits_agree_with_the_parallel_environment() -> None:
    """A serial edit at the head must be the same move as the parallel action."""
    from pgx_mcts_bench.game import make_game
    from pgx_mcts_bench.serial_braid import SERIAL_REDUCE

    serial = make_game(SERIAL)
    transition = serial.from_word([1, -1, 2, 3], strands=4)
    assert transition.legal_actions[SERIAL_REDUCE], "s1 s1^-1 at the head reduces"
    after = serial.step(transition.state, SERIAL_REDUCE)
    word = [int(x) for x in np.asarray(serial.unwrap(after.state)._word) if int(x)]
    assert word == [2, 3]


def test_serial_trains_end_to_end(tmp_path) -> None:
    from pgx_mcts_bench.networks import SerialBraidNet

    config = ExperimentConfig(
        game=SERIAL,
        search=SearchConfig(simulations=4),
        model=ModelConfig(channels=8, latent_channels=8),
        train=TrainConfig(
            iterations=1, selfplay_games=2, train_steps=1, batch_size=2, device="cpu"
        ),
    )
    agent = train_agent("alphazero", config, checkpoint_dir=tmp_path)
    assert isinstance(agent.network, SerialBraidNet)
    assert agent.history[0]["mean_game_length"] > 1


def test_acting_anywhere_in_the_window_stays_independent_of_L() -> None:
    """Two readings of "windowed": act only at the head, or act anywhere visible.

    Both keep the action space independent of L -- the position never appears in
    the action index, only an offset bounded by the window.
    """
    from dataclasses import replace as _replace

    head_only = [
        _replace(SERIAL, max_len=length, serial_act_width=1).action_size
        for length in (12, 64, 256)
    ]
    in_window = [
        _replace(SERIAL, max_len=length, serial_act_width=SERIAL.serial_window).action_size
        for length in (12, 64, 256)
    ]
    assert len(set(head_only)) == 1, head_only
    assert len(set(in_window)) == 1, in_window
    assert in_window[0] > head_only[0]
    # w offsets x (3 rewrites + 2G inserts + crossing) + 4 singletons + 2 per stride
    generators = SERIAL.max_strands - 1
    assert in_window[0] == SERIAL.serial_window * (3 + 2 * generators + 1) + 4 + 2 * len(
        SERIAL.serial_strides
    )


def test_in_window_offsets_address_the_right_positions() -> None:
    from dataclasses import replace as _replace

    from pgx_mcts_bench.game import make_game

    game = make_game(_replace(SERIAL, serial_act_width=SERIAL.serial_window))
    transition = game.from_word([2, 3, 1, -1, 2], strands=4)
    # Offsets are centred on the head, so with act_width 5 the blocks address
    # head-2 .. head+2. s1 s1^-1 sits at positions 2,3 and the head is at 0, so
    # the reducible pair is block 4 -- offset +2.
    per_offset = 3 + 2 * (SERIAL.max_strands - 1) + 1
    reduce_at_2 = 4 * per_offset + 0
    assert game.describe(reduce_at_2).startswith("REDUCE@+2")
    assert transition.legal_actions[reduce_at_2]
    after = game.step(transition.state, reduce_at_2)
    word = [int(x) for x in np.asarray(game.unwrap(after.state)._word) if int(x)]
    assert word == [2, 3, 2]


def test_serial_window_is_centred_so_both_shifts_are_informed() -> None:
    """A forward-looking window `[head, head+w)` makes one shift direction blind.

    The first ladder run measured the consequence: in the episodes it failed,
    `serial-w7-head` played 118 `SHIFT_LEFT` against 2 `SHIFT_RIGHT` -- it walked
    into tape it had never seen, because nothing distinguished the direction that
    was visible from the one that was not.
    """
    from pgx_mcts_bench.game import make_game

    game = make_game(SERIAL)
    transition = game.from_word([1, 2, 3, 1, 2, 3, 1, 2], strands=4)
    state, head = transition.state[0], transition.state[1]
    observed = np.asarray(state.observation)
    window = transition.observation[0]
    assert np.allclose(window[SERIAL.serial_window // 2], observed[head]), (
        "the head must sit at the centre of its own window"
    )
    # Both neighbours of the head are visible before the agent commits a ply.
    length = int((np.asarray(state._word) != 0).sum())
    assert np.allclose(window[SERIAL.serial_window // 2 - 1], observed[(head - 1) % length])
    assert np.allclose(window[SERIAL.serial_window // 2 + 1], observed[(head + 1) % length])


def test_serial_policy_head_reads_the_cell_it_acts_on() -> None:
    """The regression that put every serial candidate at stage 0.

    The first `SerialBraidNet` pooled the window with mean+max and read every
    logit off that vector, so the readout was near-invariant to *where* in the
    window a feature sat -- the one question this formulation exists to answer.
    A trained checkpoint moved its policy by 0.14 when the window was cyclically
    rolled, against 0.40 for a genuinely different state. `REDUCE` at an offset
    must depend on that offset's cell and its neighbour, not on the average.
    """
    from dataclasses import replace as _replace

    from pgx_mcts_bench.networks import SerialBraidNet

    config = _replace(SERIAL, serial_act_width=1)
    net = SerialBraidNet(config, ModelConfig(channels=8, latent_channels=8)).eval()
    x = torch.randn(1, config.observation_channels, 1, config.serial_window,
                    requires_grad=True)
    logits, _ = net(x)
    logits[0, 0].backward()                      # REDUCE at the head
    per_cell = x.grad.abs().sum(dim=(0, 1))[0].numpy()

    head = config.serial_window // 2
    # REDUCE acts on the pair (head, head+1), and the trunk's receptive field is
    # three taps, so the mass belongs on the head and its immediate neighbours.
    local = per_cell[head - 1 : head + 2].sum()
    assert local > 0.75 * per_cell.sum(), per_cell / per_cell.max()


def test_serial_strides_reposition_the_head_in_a_couple_of_plies() -> None:
    """Every ply of repositioning is a ply of MCTS depth, and depth is
    exponentially expensive -- a head that can only step puts distant sites past
    the search horizon. Powers of two reach anything on the tape by binary
    decomposition."""
    from pgx_mcts_bench.game import make_game

    game = make_game(SERIAL)
    length = SERIAL.max_len
    strides = sorted(game.strides, reverse=True)
    for target in range(length):
        remaining, plies = min(target, length - target), 0
        for stride in strides:
            plies += remaining // stride
            remaining %= stride
        assert remaining == 0
        assert plies <= 2, (target, plies)


def test_film_amplifies_the_ratio_beyond_the_input_channel() -> None:
    """log(A/B) reaches the network two ways, and they are not equivalent.

    As an input channel it is one of 18 planes, and the trunk is free to give it
    almost no weight -- collapsing the Pareto front to a single compromise
    policy. FiLM applies a per-channel gain generated from the ratio, so the same
    observation can produce genuinely different policies at the two ends. This
    checks the second mechanism does more than the first, with the trunk held
    identical between the two networks.
    """
    from dataclasses import replace as _replace

    game = _replace(SMALL, multi_objective=True, log_ratio_range=(-5.0, 5.0))
    torch.manual_seed(0)
    without = BraidAlphaZeroNet(
        game, ModelConfig(channels=16, latent_channels=16, film_on_ratio=False)
    ).eval()
    torch.manual_seed(0)
    with_film = BraidAlphaZeroNet(
        game, ModelConfig(channels=16, latent_channels=16, film_on_ratio=True)
    ).eval()
    assert without.film is None and with_film.film is not None
    # identical trunk and heads; the only difference is the modulator
    with torch.no_grad():
        for name, parameter in without.named_parameters():
            dict(with_film.named_parameters())[name].copy_(parameter)
        for parameter in with_film.film.net[-1].parameters():
            parameter.normal_(0.0, 0.5)

    def spread(net) -> float:
        outputs = []
        for log_ratio in (-5.0, 5.0):
            obs = torch.zeros(1, game.observation_channels, 1, game.max_len)
            obs[0, : game.max_strands - 1, 0, :3] = 1.0
            obs[0, net.ratio_channel] = log_ratio / 5.0
            with torch.inference_mode():
                outputs.append(net(obs)[0])
        return (outputs[0] - outputs[1]).abs().max().item()

    channel_only = spread(without)
    modulated = spread(with_film)
    assert modulated > channel_only, (
        f"FiLM should widen the gap between the ends of the front: "
        f"{modulated:.4f} vs {channel_only:.4f}"
    )


def test_film_can_be_switched_off() -> None:
    model = ModelConfig(channels=16, latent_channels=16, film_on_ratio=False)
    net = BraidAlphaZeroNet(SMALL, model)
    assert net.film is None
    obs = torch.zeros(2, SMALL.observation_channels, 1, SMALL.max_len)
    policy, value = net(obs)
    assert policy.shape == (2, SMALL.action_size)
