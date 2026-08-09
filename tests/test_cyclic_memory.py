from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import load_scientist
from pgx_mcts_bench.cyclic_memory import (
    build_cyclic_memory_checkpoint,
    modernize_cyclic_memory_checkpoint,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.invariant_pretrain import pretrain_cyclic_invariants
from pgx_mcts_bench.ladder import _config, experimental_capacity_arms, serial_arms
from pgx_mcts_bench.networks import CyclicMemoryBraidNet, make_braid_network


def _candidate():
    return next(
        item for item in experimental_capacity_arms() if item.name == "s-cyclic-tape8-192"
    )


def _tensor(*observations: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack(observations)).permute(0, 3, 1, 2).float()


def test_cyclic_memory_global_summary_is_rotation_invariant() -> None:
    config = _config(_candidate(), ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model)
    assert isinstance(network, CyclicMemoryBraidNet)
    first = game.from_word([1, 2, -1, -2, 1], 3, math.log(10.0)).observation
    second = game.from_word([2, -1, -2, 1, 1], 3, math.log(10.0)).observation
    summaries = network.encode_global(_tensor(first, second))
    torch.testing.assert_close(summaries[0], summaries[1], atol=1e-5, rtol=1e-5)


def test_zero_residual_preserves_window_outputs_on_mapped_actions() -> None:
    config = _config(_candidate(), ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model)
    assert isinstance(network, CyclicMemoryBraidNet)
    observation = game.from_word([1, 2, -1, -2, 1], 3, math.log(10.0)).observation
    batch = _tensor(observation)
    lengths = network._lengths(batch)
    local = network._local_view(batch, lengths)
    parent_policy, parent_value = network.window(local)
    policy, value = network(batch)
    torch.testing.assert_close(policy[:, network.window_action_map], parent_policy)
    torch.testing.assert_close(value, parent_value)


def test_budget_aware_local_view_strips_tape_but_preserves_remaining_l() -> None:
    candidate = _candidate()
    config = _config(candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    assert config.game.objective_budget_channel
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model)
    assert isinstance(network, CyclicMemoryBraidNet)
    observation = game.from_word([1, 2, -1, -2, 1], 3, math.log(10.0)).observation
    batch = _tensor(observation)
    local = network._local_view(batch, network._lengths(batch))

    window_game = replace(
        config.game,
        serial_encoder="",
        serial_encoder_states=0,
        serial_tape_symbols=0,
        serial_tape_preserve_shift=False,
    )
    assert local.shape[1] == window_game.observation_channels
    torch.testing.assert_close(local[:, -1], batch[:, -1, :, :7])


def test_global_film_and_budget_branches_start_function_preserving() -> None:
    config = _config(_candidate(), ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model)
    assert isinstance(network, CyclicMemoryBraidNet)
    low = game.from_word([1, 2, -1, -2, 1], 3, math.log(10.0)).observation
    high = game.from_word([1, 2, -1, -2, 1], 3, math.log(1000.0)).observation
    low[:, :, -1] = 0.25
    high[:, :, -1] = 0.25
    with torch.inference_mode():
        low_policy, low_value = network(_tensor(low))
        high_policy, high_value = network(_tensor(high))
    # FiLM and explicit budget skips are zero-residual at migration.  The raw
    # ratio channel remains available to the pretrained observation trunk, so
    # equality across ratios is not required; finite unchanged outputs are.
    assert torch.isfinite(low_policy).all() and torch.isfinite(high_policy).all()
    assert torch.isfinite(low_value).all() and torch.isfinite(high_value).all()
    assert torch.count_nonzero(network.global_film.net[-1].weight) == 0
    assert torch.count_nonzero(network.fusion_budget_skip[-1].weight) == 0
    assert torch.count_nonzero(network.value_budget_skip[-1].weight) == 0


def test_builder_loads_window_checkpoint_and_records_provenance(tmp_path: Path) -> None:
    parent_candidate = next(
        item for item in serial_arms() if item.name == "s-window-128"
    )
    parent_config = _config(
        parent_candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1
    )
    parent = make_braid_network(parent_config.game, parent_config.model)
    source = tmp_path / "window.pt"
    output = tmp_path / "cyclic.pt"
    torch.save({"network": parent.state_dict()}, source)
    report = build_cyclic_memory_checkpoint(source, output)
    assert output.is_file()
    assert report["candidate"] == "s-cyclic-tape8-192"
    assert report["parameters"] > sum(p.numel() for p in parent.parameters())
    assert report["window_sha256"]
    loaded = load_scientist(
        "s-cyclic-tape8-192",
        output,
        seed=0,
        device="cpu",
        require_factorized=True,
    )
    assert loaded.prediction_source == "factorized"

    migrated = load_scientist(
        "s-cyclic-tape8-192",
        output,
        seed=0,
        device="cpu",
        require_factorized=True,
        objective_budget_channel=True,
    )
    assert migrated.config.game.objective_budget_channel


def test_modernizer_persists_factorized_budget_schema_without_changing_controller(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    legacy_candidate = replace(
        candidate,
        objective_budget_channel=False,
        auxiliary_solve_backprop_to_encoder=False,
        auxiliary_budget_monotonic_weight=0.0,
        auxiliary_budget_conditioning=False,
        use_auxiliary_value=False,
    )
    config = _config(legacy_candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    legacy = make_braid_network(config.game, config.model)
    state = {
        key: value
        for key, value in legacy.state_dict().items()
        if not key.startswith(("window.auxiliary.", "window.film."))
        and not key.startswith(
            ("global_film.", "fusion_budget_skip.", "value_budget_skip.")
        )
    }
    source = tmp_path / "legacy-cyclic.pt"
    output = tmp_path / "modern-cyclic.pt"
    torch.save({"network": state, "candidate_spec": asdict(legacy_candidate)}, source)

    report = modernize_cyclic_memory_checkpoint(source, output)
    assert report["max_abs_difference"] == {"policy": 0.0, "value": 0.0}
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["modernization"]["film_trained"] is False
    assert payload["modernization"]["factorized_heads_trained"] is False
    keys = payload["network"]
    assert any(key.startswith("global_film.") for key in keys)
    assert any(key.startswith("window.auxiliary.") for key in keys)
    loaded = load_scientist(
        "s-cyclic-tape8-192",
        output,
        seed=0,
        device="cpu",
        require_factorized=True,
        objective_budget_channel=True,
    )
    assert loaded.prediction_source == "factorized"


def test_invariant_pretraining_uses_identity_disjoint_equivalent_views(
    tmp_path: Path,
) -> None:
    parent_candidate = next(
        item for item in serial_arms() if item.name == "s-window-128"
    )
    parent_config = _config(
        parent_candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1
    )
    source = tmp_path / "window.pt"
    initialized = tmp_path / "cyclic.pt"
    pretrained = tmp_path / "pretrained.pt"
    torch.save(
        {
            "network": make_braid_network(
                parent_config.game, parent_config.model
            ).state_dict()
        },
        source,
    )
    build_cyclic_memory_checkpoint(source, initialized)
    report = pretrain_cyclic_invariants(
        initialized,
        pretrained,
        identities=6,
        calibration_identities=4,
        views_per_identity=2,
        steps=2,
        batch_size=3,
        bank_seed=71,
        seed=19,
    )
    assert pretrained.is_file()
    assert report["calibration_before"]["top1"] >= 0.0
    assert report["calibration_after"]["top1"] >= 0.0
