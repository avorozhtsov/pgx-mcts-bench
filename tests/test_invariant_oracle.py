from __future__ import annotations

import numpy as np
import pytest
import torch

from pgx_mcts_bench.invariant_features import invariant_feature_size, invariant_features
from pgx_mcts_bench.ladder import (
    _config,
    invariant_oracle_arms,
    invariant_oracle_depth_dose_arms,
    invariant_oracle_mutation_arms,
)
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.serial_braid import SerialBraidGame


@pytest.mark.parametrize(
    ("name", "size"),
    [("classical", 6), ("alexander", 18), ("jones", 18), ("combined", 30)],
)
def test_invariant_feature_sets_have_fixed_finite_shapes(name: str, size: int) -> None:
    features = invariant_features([1, 1, 1], 2, name, 48)
    terminal = invariant_features([], 1, name, 48)
    assert invariant_feature_size(name) == size
    assert features.shape == terminal.shape == (size,)
    assert np.isfinite(features).all()
    assert np.isfinite(terminal).all()


def test_invariants_are_carried_across_internal_head_moves() -> None:
    candidate = invariant_oracle_arms()[2]
    config = _config(candidate, ("T(2,3)", 0), 1, "cpu").game
    game = SerialBraidGame(config)
    transition = game.from_word([1, 1, 1], 2)
    vector = transition.state.invariant_vector
    shift = next(action for action in range(game.num_actions) if game.shift_of(action))
    shifted = game.step(transition.state, shift)
    assert shifted.state.invariant_vector is vector
    np.testing.assert_array_equal(shifted.state.invariant_vector, vector)


@pytest.mark.parametrize("feature_set", ["classical", "alexander", "jones", "combined"])
def test_crossing_change_recomputes_current_knot_invariants(feature_set: str) -> None:
    candidate = next(
        arm for arm in invariant_oracle_arms() if arm.invariant_features == feature_set
    )
    config = _config(candidate, ("T(2,3)", 0), 1, "cpu").game
    game = SerialBraidGame(config)
    transition = game.from_word([1, 1, 1], 2)
    before = transition.state.invariant_vector
    crossing_change = next(
        int(action)
        for action in np.flatnonzero(transition.legal_actions)
        if "CROSSING_CHANGE" in game.describe(int(action))
    )

    changed = game.step(transition.state, crossing_change)
    expected = invariant_features(
        changed.state.pgx._word,
        int(np.asarray(changed.state.pgx._n)),
        feature_set,
        config.max_len,
    )

    assert changed.state.invariant_vector is not before
    np.testing.assert_array_equal(changed.state.invariant_vector, expected)
    assert not np.array_equal(changed.state.invariant_vector, before)


@pytest.mark.parametrize("candidate", invariant_oracle_arms(), ids=lambda arm: arm.name)
def test_all_oracle_architectures_forward_and_backpropagate(candidate) -> None:
    experiment = _config(candidate, ("T(2,3)", 0), 1, "cpu")
    transition = SerialBraidGame(experiment.game).from_word([1, 1, 1], 2)
    observation = torch.from_numpy(transition.observation).permute(2, 0, 1).unsqueeze(0)
    network = make_braid_network(experiment.game, experiment.model)
    policy, value = network(observation)
    (policy.square().mean() + value.square().mean()).backward()
    invariant_gradients = [
        parameter.grad for name, parameter in network.named_parameters() if "invariant_" in name
    ]
    assert policy.shape == (1, experiment.game.action_size)
    assert value.shape == (1,)
    assert invariant_gradients and all(gradient is not None for gradient in invariant_gradients)


def test_dual_tower_invariants_can_change_positional_policy() -> None:
    candidate = invariant_oracle_arms()[-1]
    experiment = _config(candidate, ("T(2,3)", 0), 1, "cpu")
    transition = SerialBraidGame(experiment.game).from_word([1, 1, 1], 2)
    observation = torch.from_numpy(transition.observation).permute(2, 0, 1).unsqueeze(0)
    changed = observation.clone()
    end = experiment.game.observation_channels - 2
    changed[:, end - 30 : end] += 0.5
    network = make_braid_network(experiment.game, experiment.model).eval()
    with torch.no_grad():
        original_policy, _ = network(observation)
        changed_policy, _ = network(changed)
    positional = candidate.serial_act_width * (3 + 2 * experiment.game.generator_capacity + 1)
    assert not torch.equal(original_policy[:, :positional], changed_policy[:, :positional])


@pytest.mark.parametrize(
    "child", invariant_oracle_mutation_arms(), ids=lambda candidate: candidate.name
)
def test_dual_architecture_mutations_preserve_parent_outputs(child) -> None:
    parent = invariant_oracle_arms()[-1]
    parent_experiment = _config(parent, ("T(2,3)", 0), 1, "cpu")
    child_experiment = _config(child, ("T(2,3)", 0), 1, "cpu")
    transition = SerialBraidGame(parent_experiment.game).from_word([1, 1, 1], 2)
    observation = torch.from_numpy(transition.observation).permute(2, 0, 1).unsqueeze(0)
    parent_network = make_braid_network(parent_experiment.game, parent_experiment.model).eval()
    child_network = make_braid_network(child_experiment.game, child_experiment.model).eval()

    assert load_policy_value_state_dict(child_network, parent_network.state_dict())
    with torch.no_grad():
        parent_outputs = parent_network.forward_with_auxiliary(observation)
        child_outputs = child_network.forward_with_auxiliary(observation)

    torch.testing.assert_close(child_outputs[0], parent_outputs[0], rtol=0, atol=0)
    torch.testing.assert_close(child_outputs[1], parent_outputs[1], rtol=0, atol=0)
    for actual, expected in zip(child_outputs[2], parent_outputs[2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "child", invariant_oracle_depth_dose_arms(), ids=lambda candidate: candidate.name
)
def test_invariant_depth_doses_preserve_parent_outputs(child) -> None:
    parent_name = (
        "raster-invariant-combined-film"
        if "-film-" in child.name
        else "raster-invariant-combined-dual"
    )
    parent = next(
        candidate for candidate in invariant_oracle_arms() if candidate.name == parent_name
    )
    parent_experiment = _config(parent, ("T(2,3)", 0), 1, "cpu")
    child_experiment = _config(child, ("T(2,3)", 0), 1, "cpu")
    transition = SerialBraidGame(parent_experiment.game).from_word([1, 1, 1], 2)
    observation = torch.from_numpy(transition.observation).permute(2, 0, 1).unsqueeze(0)
    parent_network = make_braid_network(parent_experiment.game, parent_experiment.model).eval()
    child_network = make_braid_network(child_experiment.game, child_experiment.model).eval()

    assert load_policy_value_state_dict(child_network, parent_network.state_dict())
    with torch.no_grad():
        parent_outputs = parent_network.forward_with_auxiliary(observation)
        child_outputs = child_network.forward_with_auxiliary(observation)

    torch.testing.assert_close(child_outputs[0], parent_outputs[0], rtol=0, atol=0)
    torch.testing.assert_close(child_outputs[1], parent_outputs[1], rtol=0, atol=0)
    for actual, expected in zip(child_outputs[2], parent_outputs[2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
