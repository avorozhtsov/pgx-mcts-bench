from __future__ import annotations

import math

import numpy as np
import torch

from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, experimental_capacity_arms
from pgx_mcts_bench.networks import StrandGraphBraidNet, make_braid_network
from pgx_mcts_bench.serial_braid import SerialBraidGame


def _candidate():
    return next(
        candidate
        for candidate in experimental_capacity_arms()
        if candidate.name == "s-strand-graph-128"
    )


def _tensor(*observations):
    return torch.from_numpy(np.stack(observations)).permute(0, 3, 1, 2).float()


def test_compiled_graph_links_closed_physical_strands() -> None:
    word = np.asarray([1, 2, 3, -2, -1, 3, 2, -3, 1], dtype=np.int32)
    planes = SerialBraidGame._strand_graph_planes(word, 4, observed_width=12)
    edges = np.rint(planes[: len(word)] * 11).astype(int)
    assert np.all((0 <= edges) & (edges < len(word)))
    for position in range(len(word)):
        for incidence in (0, 1):
            previous, following = edges[position, 2 * incidence : 2 * incidence + 2]
            # The same physical strand must point back through one of the two
            # incidences at the neighbouring crossing, including across closure.
            assert position in edges[following, (0, 2)]
            assert position in edges[previous, (1, 3)]


def test_strand_router_runs_compulsory_scan_and_factorized_heads() -> None:
    config = _config(_candidate(), ("R(5,12)#0", 0), 0, "cpu", selfplay_games=1)
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model)
    assert isinstance(network, StrandGraphBraidNet)
    observation = game.from_word(
        [1, 2, 3, -2, -1, 3, 2, -3, 1], 4, math.log(1000.0)
    ).observation
    policy, value, auxiliary = network.forward_with_auxiliary(_tensor(observation))
    assert policy.shape == (1, config.game.action_size)
    assert value.shape == (1,)
    assert all(part.shape == (1, 4) for part in auxiliary)
    assert config.game.objective_budget_channel
    assert network.auxiliary_solve_backprop


def test_remote_crossing_changes_head_routing_logits() -> None:
    torch.manual_seed(7)
    config = _config(_candidate(), ("R(5,12)#0", 0), 0, "cpu", selfplay_games=1)
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model).eval()
    first = game.from_word(
        [1, 2, 3, 2, 1, 2, -3, -2, 1, 3, 2], 4, math.log(10.0)
    ).observation
    second = game.from_word(
        [1, 2, 3, 2, 1, -2, -3, -2, 1, 3, 2], 4, math.log(10.0)
    ).observation
    with torch.no_grad():
        policy, _ = network(_tensor(first, second))
    route_start = network.per_offset + 4
    assert not torch.allclose(policy[0, route_start:], policy[1, route_start:])


def test_policy_gradient_reaches_strand_scan() -> None:
    config = _config(_candidate(), ("R(5,12)#0", 0), 0, "cpu", selfplay_games=1)
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model)
    observation = game.from_word(
        [1, 2, 3, -1, 2, -3, 1, 3, -2], 4, math.log(1000.0)
    ).observation
    policy, _ = network(_tensor(observation))
    loss = -torch.log_softmax(policy, dim=1)[0, -1]
    loss.backward()
    gradient = network.routing_blocks[0].body[0].weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0


def test_capacity_variants_apply_their_depth_and_optimizer_settings() -> None:
    variants = {
        candidate.name: candidate
        for candidate in experimental_capacity_arms()
        if candidate.name.startswith("s-strand-graph")
    }
    assert set(variants) == {
        "s-strand-graph-compact-128",
        "s-strand-graph-128",
        "s-strand-graph-wide-128",
        "s-strand-graph-local-128",
    }
    wide = _config(
        variants["s-strand-graph-wide-128"],
        ("R(5,12)#0", 0),
        0,
        "cpu",
        selfplay_games=1,
    )
    network = make_braid_network(wide.game, wide.model)
    assert isinstance(network, StrandGraphBraidNet)
    assert network.width == 160
    assert len(network.routing_blocks) == 8
    assert wide.train.batch_size == 64
    assert wide.train.learning_rate == 5e-4
    assert wide.train.weight_decay == 5e-5


def test_local_tower_scaffold_is_exact_at_initialization() -> None:
    candidate = next(
        candidate
        for candidate in experimental_capacity_arms()
        if candidate.name == "s-strand-graph-local-128"
    )
    config = _config(candidate, ("R(5,12)#0", 0), 0, "cpu", selfplay_games=1)
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model).eval()
    assert isinstance(network, StrandGraphBraidNet)
    assert network.local_tower is not None
    observation = game.from_word(
        [1, 2, 3, -2, -1, 3, 2, -3, 1], 4, math.log(1000.0)
    ).observation
    batch = _tensor(observation)
    with torch.no_grad():
        _, lengths, _ = network.encode(batch)
        local_observation = network._local_view(batch, lengths)
        expected_policy, expected_value, _ = network.local_tower._forward_core(
            local_observation
        )
        actual_policy, actual_value, _ = network._forward_core(batch)
    torch.testing.assert_close(actual_policy, expected_policy)
    torch.testing.assert_close(actual_value, expected_value)
