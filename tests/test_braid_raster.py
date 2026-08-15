from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from pgx_mcts_bench.config import BraidGameConfig, ModelConfig
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.networks import (
    RasterSerialBraidNet,
    RasterWindowRepresentation,
    RoutedRasterResidualBlock,
    ScalableRasterSerialBraidNet,
    load_policy_value_state_dict,
    make_braid_network,
)
from pgx_mcts_bench.serial_braid import SerialBraidGame


def _config(max_strands: int = 5) -> BraidGameConfig:
    return BraidGameConfig(
        max_len=32,
        max_strands=max_strands,
        simplify_budget=64,
        allow_crossing_change=True,
        serial_window=7,
        serial_act_width=7,
        serial_raster="joint",
    )


def _tensor(observation: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(observation[None]).permute(0, 3, 1, 2).float()


def test_raster_is_an_exact_paired_encoding_of_an_artin_word() -> None:
    word = np.asarray([1, -2, 3, -1, 2], dtype=np.int32)
    raster = SerialBraidGame.braid_raster_planes(word, strands=4, max_strands=6)

    assert raster.shape == (5, 6, 4)
    np.testing.assert_array_equal(raster[0, 0, :3], (0, 1, 1))
    np.testing.assert_array_equal(raster[0, 1, :3], (1, 0, 0))
    assert np.all(raster[:, :4, 3] == 1)
    assert np.all(raster[:, 4:] == 0)
    np.testing.assert_array_equal(SerialBraidGame.word_from_braid_raster(raster, strands=4), word)

    identity = np.zeros((1, 6, 4), dtype=np.float32)
    identity[:, :4, 1] = 1.0
    identity[:, :4, 3] = 1.0
    expanded = np.concatenate([raster[:2], identity, raster[2:]], axis=0)
    np.testing.assert_array_equal(SerialBraidGame.word_from_braid_raster(expanded, strands=4), word)


def test_serial_observation_appends_the_head_centred_raster() -> None:
    config = _config()
    game = make_game(config)
    transition = game.from_word([1, -2, 3, -1, 2], strands=4)
    observation = transition.observation[0]
    end = config.observation_channels - 1  # mandatory internal-budget plane
    start = end - 4 * config.max_strands
    raster = observation[:, start:end].reshape(config.serial_window, 5, 4)

    expected_letters = np.asarray([3, -1, 2, 1, -2, 3, -1], dtype=np.int32)
    expected = SerialBraidGame.braid_raster_planes(expected_letters, 4, 5)
    np.testing.assert_array_equal(raster, expected)


def test_raster_checkpoint_capacity_migration_preserves_old_policy_parameters() -> None:
    source_config = replace(
        _config(5),
        serial_raster="axial",
        serial_raster_masked_norm=True,
        serial_raster_identity_padding=True,
    )
    target_config = replace(source_config, max_strands=12)
    model = ModelConfig(channels=16, residual_blocks=2)
    source = make_braid_network(source_config, model).eval()
    target = make_braid_network(target_config, model).eval()

    assert load_policy_value_state_dict(target, source.state_dict())
    source_state = source.state_dict()
    target_state = target.state_dict()
    torch.testing.assert_close(
        target_state["positional.weight"][:11],
        source_state["positional.weight"][:11],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        target_state["positional.weight"][-1],
        source_state["positional.weight"][-1],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        target_state["positional.bias"][:11],
        source_state["positional.bias"][:11],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        target_state["positional.bias"][-1],
        source_state["positional.bias"][-1],
        rtol=0,
        atol=0,
    )
    for strands, config, network in (
        (4, source_config, source),
        (12, target_config, target),
    ):
        word = [1, -2, 3, -1, 2] if strands == 4 else list(range(1, 12))
        observation = _tensor(make_game(config).from_word(word, strands=strands).observation)
        with torch.no_grad():
            policy, value = network(observation)
        assert torch.isfinite(policy).all()
        assert torch.isfinite(value).all()


@pytest.mark.parametrize("identity_padding", [False, True])
def test_scalable_raster_uses_full_canvas_with_optional_identity_columns(
    identity_padding: bool,
) -> None:
    config = replace(
        _config(max_strands=6),
        serial_raster="scalable",
        serial_raster_identity_padding=identity_padding,
        cyclic_band_generators=True,
    )
    word = np.asarray([1, -2, 3, -1, 2], dtype=np.int32)
    observation = make_game(config).from_word(word.tolist(), strands=4).observation[0]
    assert observation.shape[0] == config.max_len
    start = config.observation_channels - 1 - 4 * config.max_strands
    raster = observation[:, start:-1].reshape(config.max_len, config.max_strands, 4)

    np.testing.assert_array_equal(
        SerialBraidGame.word_from_braid_raster(
            raster if identity_padding else raster[: len(word)],
            strands=4,
            cyclic_band_generators=True,
        ),
        word,
    )
    padding = raster[len(word) :]
    if identity_padding:
        assert np.all(padding[:, :4, 1] == 1)
        assert np.all(padding[:, :4, 3] == 1)
        assert np.all(padding[:, 4:] == 0)
    else:
        assert np.all(padding == 0)


@pytest.mark.parametrize("sign", [-1, 1])
def test_initial_extra_row_is_a_reversible_markov_stabilization(sign: int) -> None:
    config = replace(
        _config(max_strands=6),
        serial_raster="scalable",
        serial_raster_identity_padding=True,
        serial_initial_markov_stabilizations=1,
        serial_initial_markov_sign=sign,
    )
    game = make_game(config)
    transition = game.from_word([1, -1], strands=2)
    raw = game.unwrap(transition.state)
    assert int(raw._n) == 3
    np.testing.assert_array_equal(np.asarray(raw._word)[:3], [1, -1, 2 * sign])

    destabilize = game._singleton_base  # noqa: SLF001
    assert transition.legal_actions[destabilize]
    restored = game.step(transition.state, destabilize)
    restored_raw = game.unwrap(restored.state)
    assert int(restored_raw._n) == 2
    np.testing.assert_array_equal(np.asarray(restored_raw._word)[:2], [1, -1])


@pytest.mark.parametrize("variant", ["joint", "axial", "recurrent"])
def test_raster_window_network_runs_and_trunk_is_capacity_agnostic(variant: str) -> None:
    model = ModelConfig(channels=16, residual_blocks=2, latent_channels=16)
    small = replace(_config(5), serial_raster=variant)
    large = replace(_config(8), serial_raster=variant)
    network = make_braid_network(small, model)
    assert isinstance(network, RasterSerialBraidNet)
    observation = make_game(small).from_word([1, -2, 3, 2, -1], strands=4).observation
    policy, value = network(_tensor(observation))
    assert policy.shape == (1, small.action_size)
    assert value.shape == (1,)

    small_trunk = RasterWindowRepresentation(small, model).state_dict()
    large_trunk = RasterWindowRepresentation(large, model).state_dict()
    assert {key: value.shape for key, value in small_trunk.items()} == {
        key: value.shape for key, value in large_trunk.items()
    }


def test_masked_axial_raster_runs_on_padded_canvas() -> None:
    config = replace(
        _config(max_strands=8),
        serial_raster="axial",
        serial_raster_masked_norm=True,
        serial_raster_identity_padding=True,
    )
    model = ModelConfig(channels=16, residual_blocks=2, latent_channels=16)
    network = make_braid_network(config, model)
    observation = make_game(config).from_word([1, -2, 3, 2, -1], strands=4).observation
    policy, value = network(_tensor(observation))
    assert policy.shape == (1, config.action_size)
    assert value.shape == (1,)


def test_raster_trunk_is_equivariant_to_word_rotation() -> None:
    torch.manual_seed(4)
    config = _config()
    game = make_game(config)
    observation = _tensor(game.from_word([1, -2, 3, 2, -1, 2, 3], 4).observation)
    trunk = RasterWindowRepresentation(
        config, ModelConfig(channels=8, residual_blocks=2, latent_channels=8)
    ).eval()
    with torch.no_grad():
        original = trunk(observation)
        shifted = trunk(torch.roll(observation, 2, dims=-1))
    torch.testing.assert_close(shifted, torch.roll(original, 2, dims=-1))


def test_full_torus_is_not_silently_assumed() -> None:
    """First and last strand are boundaries, not neighbouring Artin strands."""
    config = _config()
    game = make_game(config)
    observation = _tensor(game.from_word([1, 1, -1, 1, -1], 2).observation)
    trunk = RasterWindowRepresentation(
        config, ModelConfig(channels=8, residual_blocks=1, latent_channels=8)
    )
    raster = observation[:, trunk.raster_start : trunk.raster_end, 0]
    raster = raster.reshape(1, config.max_strands, 4, -1).permute(0, 2, 1, 3)
    padded = trunk.blocks[0]._pad(raster)
    # Vertical halo is zero even though the word halo is circular.
    assert torch.count_nonzero(padded[:, :, 0]) == 0
    torch.testing.assert_close(padded[:, :, 1:-1, 0], raster[:, :, :, -1])


def test_explicit_bstar_torus_wraps_the_strand_axis() -> None:
    config = replace(
        _config(),
        serial_raster="scalable",
        serial_raster_wrap_strands=True,
        cyclic_band_generators=True,
    )
    trunk = RasterWindowRepresentation(
        config, ModelConfig(channels=8, residual_blocks=1, latent_channels=8)
    )
    block = trunk.blocks[0]
    features = torch.arange(5.0).reshape(1, 1, 5, 1).expand(2, 8, 5, 3).clone()
    features[1] += 10
    active = torch.zeros((2, 1, 5, 3))
    active[0, :, :4] = 1
    active[1, :, :2] = 1
    with torch.no_grad():
        block.vertical.weight.zero_()
        for channel in range(8):
            block.vertical.weight[channel, channel, 2, 0] = 1
    convolved = block._vertical_convolve(features, active)
    # The next neighbour of active row 4 is row 1, not inactive capacity row 5.
    torch.testing.assert_close(convolved[0, :, 3], features[0, :, 0])
    torch.testing.assert_close(convolved[1, :, 1], features[1, :, 0])
    assert torch.count_nonzero(convolved[0, :, 4]) == 0
    assert torch.count_nonzero(convolved[1, :, 2:]) == 0


def test_bstar_seam_crossing_wraps_last_row_to_first_and_round_trips() -> None:
    word = np.asarray([4, -4, 1, -3], dtype=np.int32)
    raster = SerialBraidGame.braid_raster_planes(
        word, strands=4, max_strands=6, cyclic_band_generators=True
    )
    # Positive seam: last row moves right/over through the wrap, first row
    # moves left/under.  It is one local torus edge, not an affine assumption.
    np.testing.assert_array_equal(raster[0, 3, :3], (0, 1, 1))
    np.testing.assert_array_equal(raster[0, 0, :3], (1, 0, 0))
    np.testing.assert_array_equal(
        SerialBraidGame.word_from_braid_raster(raster, strands=4, cyclic_band_generators=True),
        word,
    )


@pytest.mark.parametrize(
    "name",
    [
        "s-window-128-bstar",
        "conv-window-axial-128-bstar",
        "conv-window-recurrent-128-bstar",
        "conv-cylinder-recurrent-128-bstar",
        "conv-torus-recurrent-idcols-128-bstar",
    ],
)
def test_bstar_candidate_action_and_observation_shapes(name: str) -> None:
    from pgx_mcts_bench.ladder import _config as ladder_config
    from pgx_mcts_bench.ladder import experimental_capacity_arms

    candidate = next(item for item in experimental_capacity_arms() if item.name == name)
    config = ladder_config(candidate, ("unknot", 2), 0, "cpu", selfplay_games=1)
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model)
    transition = game.from_word([4, 1, -4], strands=4)
    policy, value = network(_tensor(transition.observation))
    assert policy.shape == (1, config.game.action_size)
    assert value.shape == (1,)
    assert config.game.generator_capacity == config.game.max_strands


def test_scalable_raster_reuses_its_block_for_every_recurrent_step() -> None:
    config = replace(
        _config(),
        serial_raster="scalable",
        cyclic_band_generators=True,
    )
    trunk = RasterWindowRepresentation(
        config, ModelConfig(channels=8, residual_blocks=3, latent_channels=8)
    )
    calls = 0

    def count_calls(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = trunk.blocks[0].register_forward_hook(count_calls)
    try:
        observation = make_game(config).from_word([1, -2, 3], strands=4).observation
        trunk(_tensor(observation))
    finally:
        handle.remove()

    assert calls == 4  # max(4, residual_blocks)


def test_routed_raster_uses_dilations_and_starts_as_an_exact_residual() -> None:
    config = replace(
        _config(),
        serial_raster="scalable",
        serial_raster_wrap_strands=True,
        serial_raster_masked_norm=True,
        serial_raster_identity_padding=True,
        cyclic_band_generators=True,
        objective_budget_channel=True,
    )
    trunk = RasterWindowRepresentation(
        config, ModelConfig(channels=8, residual_blocks=4, latent_channels=8)
    )
    block = trunk.blocks[0]
    assert isinstance(block, RoutedRasterResidualBlock)
    assert block.residual_gate.item() == 0.0

    dilations: list[int] = []
    original = block._horizontal_convolve

    def record_dilation(x: torch.Tensor, dilation: int) -> torch.Tensor:
        dilations.append(dilation)
        return original(x, dilation)

    block._horizontal_convolve = record_dilation  # type: ignore[method-assign]
    observation = make_game(config).from_word([1, -2, 3, 2, -1], strands=4).observation
    raster = _tensor(observation)
    with torch.no_grad():
        initial = torch.relu(
            trunk.input(
                raster[:, trunk.raster_start : trunk.raster_end, 0]
                .reshape(1, config.max_strands, 4, config.max_len)
                .permute(0, 2, 1, 3)
            )
        )
        spatial, active, _ = trunk.encode_spatial(raster)
    assert dilations == [1, 2, 4, 8]
    torch.testing.assert_close(spatial, initial * active)


def test_routed_raster_has_distinct_content_and_identity_workspace_masks() -> None:
    config = replace(
        _config(max_strands=6),
        serial_raster="scalable",
        serial_raster_wrap_strands=True,
        serial_raster_masked_norm=True,
        serial_raster_identity_padding=True,
        cyclic_band_generators=True,
        objective_budget_channel=True,
    )
    trunk = RasterWindowRepresentation(
        config, ModelConfig(channels=8, residual_blocks=1, latent_channels=8)
    )
    block = trunk.blocks[0]
    assert isinstance(block, RoutedRasterResidualBlock)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(_module, args, _kwargs, _output):
        seen.append((args[1].detach().clone(), args[2].detach().clone()))

    handle = block.register_forward_hook(capture, with_kwargs=True)
    try:
        word = [1, -2, 3, 2, -1]
        observation = make_game(config).from_word(word, strands=4).observation
        trunk.encode_spatial(_tensor(observation))
    finally:
        handle.remove()

    workspace, content = seen[0]
    assert torch.all(workspace[:, :, :4, len(word) :] == 1)
    assert torch.count_nonzero(content[:, :, :, len(word) :]) == 0
    assert torch.all(content[:, :, :4, : len(word)] == 1)


def test_routed_critic_changes_when_only_remote_braid_content_changes() -> None:
    torch.manual_seed(17)
    config = replace(
        _config(max_strands=6),
        serial_raster="scalable",
        serial_raster_wrap_strands=True,
        serial_raster_masked_norm=True,
        serial_raster_identity_padding=True,
        cyclic_band_generators=True,
        objective_budget_channel=True,
    )
    network = make_braid_network(
        config, ModelConfig(channels=16, residual_blocks=4, latent_channels=16)
    )
    assert isinstance(network, ScalableRasterSerialBraidNet)
    with torch.no_grad():
        network.representation.blocks[0].residual_gate.fill_(1.0)
    prefix = [1, -2, 3, 1, -2, 3, 1]
    first = make_game(config).from_word(prefix + [2, -3], strands=4).observation
    second = make_game(config).from_word(prefix + [-3, 2], strands=4).observation
    # The head-local seven letters are identical. Only the full-raster path can
    # make the task-level critic features differ.
    np.testing.assert_array_equal(first[:, :7], second[:, :7])
    with torch.no_grad():
        _, _, first_features = network._forward_core(_tensor(first))
        _, _, second_features = network._forward_core(_tensor(second))
    assert not torch.allclose(first_features, second_features)


def test_scalable_bstar_scores_the_active_seam_not_the_capacity_seam() -> None:
    config = replace(
        _config(max_strands=6),
        serial_raster="scalable",
        cyclic_band_generators=True,
    )
    network = make_braid_network(
        config, ModelConfig(channels=8, residual_blocks=1, latent_channels=8)
    )
    assert isinstance(network, ScalableRasterSerialBraidNet)

    columns = config.serial_window
    spatial = torch.zeros(2, 8, config.max_strands, columns)
    # Make each row recognizable.  The two batch members have respectively
    # three and five active strands inside the same six-row capacity.
    for row in range(config.max_strands):
        spatial[:, :, row] = float(row + 1)
    active = torch.zeros(2, 1, config.max_strands, columns)
    active[0, :, :3] = 1
    active[1, :, :5] = 1
    column_features = torch.zeros(2, 8, columns)

    seen_pairs: list[torch.Tensor] = []

    def record_input(_module, inputs, _output):
        seen_pairs.append(inputs[0].detach().clone())

    handle = network.insert_policy.register_forward_hook(record_input)
    try:
        logits = network._insert_logits(spatial, active, column_features)
    finally:
        handle.remove()

    seam_input = seen_pairs[-1]
    # First eight channels are the active last row: row 2 (value 3) and row 4
    # (value 5), never the inactive capacity row 5 (value 6).
    torch.testing.assert_close(seam_input[0, :8], torch.full_like(seam_input[0, :8], 3))
    torch.testing.assert_close(seam_input[1, :8], torch.full_like(seam_input[1, :8], 5))
    assert logits.shape[2] == config.generator_capacity


def test_scalable_torus_checkpoint_shapes_do_not_depend_on_strand_capacity() -> None:
    model = ModelConfig(
        channels=16,
        residual_blocks=4,
        latent_channels=16,
        use_auxiliary_value=True,
        auxiliary_solve_backprop_to_encoder=True,
        auxiliary_budget_conditioning=True,
    )
    small = replace(
        _config(max_strands=5),
        serial_raster="scalable",
        serial_raster_wrap_strands=True,
        serial_raster_identity_padding=True,
        cyclic_band_generators=True,
        objective_budget_channel=True,
    )
    large = replace(small, max_strands=9)
    small_shapes = {
        name: parameter.shape
        for name, parameter in make_braid_network(small, model).state_dict().items()
    }
    large_shapes = {
        name: parameter.shape
        for name, parameter in make_braid_network(large, model).state_dict().items()
    }
    assert small_shapes == large_shapes


def test_fresh_budget_raster_starts_with_a_zero_budget_input_weight() -> None:
    config = replace(
        _config(),
        serial_raster="scalable",
        objective_budget_channel=True,
    )
    trunk = RasterWindowRepresentation(
        config, ModelConfig(channels=8, residual_blocks=2, latent_channels=8)
    )
    assert torch.count_nonzero(trunk.metadata.weight[:, -1:]) == 0
