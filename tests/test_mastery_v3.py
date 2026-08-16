from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, mastery_v3_arms, strand12_arms
from pgx_mcts_bench.mastery_v3 import (
    CyclicGraphDualV3,
    CyclicMemoryDeepV3,
    migration_max_abs_difference,
)
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network


def _tensor(observation: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(observation)).permute(2, 0, 1)[None].float()


@pytest.fixture(scope="module")
def parent():
    candidate = next(
        arm for arm in strand12_arms() if arm.name == "cyclic-memory-12"
    )
    config = _config(candidate, ("R(3,12)#0", 0), 7, "cpu", selfplay_games=1)
    return config, make_braid_network(config.game, config.model).eval()


@pytest.mark.parametrize(
    ("candidate_name", "network_type"),
    [
        ("cyclic-memory-deep-v3", CyclicMemoryDeepV3),
        ("cyclic-graph-dual-v3", CyclicGraphDualV3),
    ],
)
def test_v3_checkpoint_fork_is_function_preserving(
    parent, candidate_name: str, network_type: type[CyclicMemoryDeepV3]
) -> None:
    parent_config, parent_network = parent
    candidate = next(arm for arm in mastery_v3_arms() if arm.name == candidate_name)
    child_config = _config(
        candidate, ("R(3,12)#0", 0), 7, "cpu", selfplay_games=1
    )
    child = make_braid_network(child_config.game, child_config.model).eval()
    assert isinstance(child, network_type)
    child.load_parent_state_dict(parent_network.state_dict())

    word = [1, 2, 1, 2]
    parent_observation = make_game(parent_config.game).from_word(
        word, 3, math.log(1000.0)
    ).observation
    child_observation = make_game(child_config.game).from_word(
        word, 3, math.log(1000.0)
    ).observation
    differences = migration_max_abs_difference(
        parent_network,
        child,
        _tensor(parent_observation),
        _tensor(child_observation),
    )
    assert differences == {key: 0.0 for key in differences}


def test_safe_candidate_meets_registered_size_and_depth(parent) -> None:
    candidate = next(
        arm for arm in mastery_v3_arms() if arm.name == "cyclic-memory-deep-v3"
    )
    config = _config(candidate, ("R(3,12)#0", 0), 7, "cpu", selfplay_games=1)
    network = make_braid_network(config.game, config.model)
    assert isinstance(network, CyclicMemoryDeepV3)
    assert len(network.blocks) == 10
    assert [block.dilation for block in network.blocks] == [1, 2, 4, 8, 16] * 2
    assert all(torch.count_nonzero(block.layer_scale) == 0 for block in network.blocks)
    assert 1_500_000 <= network.parameter_report()["total"] <= 3_000_000


def test_graph_candidate_exposes_12_strand_shared_heads(parent) -> None:
    parent_config, parent_network = parent
    candidate = next(
        arm for arm in mastery_v3_arms() if arm.name == "cyclic-graph-dual-v3"
    )
    config = _config(candidate, ("R(3,12)#0", 0), 7, "cpu", selfplay_games=1)
    network = make_braid_network(config.game, config.model).eval()
    assert isinstance(network, CyclicGraphDualV3)
    network.load_parent_state_dict(parent_network.state_dict())
    observation = make_game(config.game).from_word(
        list(range(1, 12)), 12, math.log(10.0)
    ).observation
    diagnostics = network.diagnostics(_tensor(observation))
    assert diagnostics.row_pair_logits.shape == (1, 11)
    assert diagnostics.invalid_logit.shape == (1,)
    assert diagnostics.capacity_logit.shape == (1,)
    assert torch.count_nonzero(network.row_pair_gate) == 0
    assert parent_config.game.action_size == config.game.action_size


def test_migrated_v3_checkpoint_reloads_nested_parent_adapter(parent) -> None:
    _parent_config, parent_network = parent
    parent_network.attach_option_policy_adapter()
    candidate = next(
        arm for arm in mastery_v3_arms() if arm.name == "cyclic-memory-deep-v3"
    )
    config = _config(candidate, ("R(3,12)#0", 0), 7, "cpu", selfplay_games=1)
    migrated = make_braid_network(config.game, config.model)
    assert isinstance(migrated, CyclicMemoryDeepV3)
    migrated.load_parent_state_dict(parent_network.state_dict())
    resumed = make_braid_network(config.game, config.model)
    load_policy_value_state_dict(resumed, migrated.state_dict())
    assert isinstance(resumed, CyclicMemoryDeepV3)
    assert resumed.parent.option_policy_adapter is not None
