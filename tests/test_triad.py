from __future__ import annotations

from pathlib import Path

import torch

from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import STAGES, _config, ensemble_arms
from pgx_mcts_bench.networks import TriadBraidNet, make_braid_network
from pgx_mcts_bench.triad import build_triad_checkpoint


def _triad():
    candidate = ensemble_arms()[0]
    config = _config(candidate, STAGES[0], seed=0, device="cpu")
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model)
    assert isinstance(network, TriadBraidNet)
    return candidate, config, game, network


def _observation(transition) -> torch.Tensor:
    return torch.from_numpy(transition.observation).permute(2, 0, 1).unsqueeze(0)


def test_triad_union_has_preserve_and_write_zero_as_distinct_actions() -> None:
    _, _, game, _ = _triad()
    names = [game.describe(index) for index in range(game.num_actions)]
    preserve = names.index("SHIFT_LEFT(1,PRESERVE)")
    erase = names.index("SHIFT_LEFT(1,WRITE(0))")
    assert preserve != erase
    assert game.tape_write_of(preserve) is None
    assert game.tape_write_of(erase) == 0


def test_zero_initialized_router_averages_only_supporting_normalized_logits() -> None:
    _, _, game, network = _triad()
    observation = _observation(game.from_word([1, 1, 1], strands=2))
    views = network._views(observation)
    outputs = [
        tower._forward_core(view)
        for tower, view in zip(network.towers, views, strict=True)
    ]
    normalized = [network._normalize_logits(output[0]) for output in outputs]
    policy, _ = network(observation)

    for action in range(network.action_size):
        opinions = []
        for tower in range(3):
            mapping = getattr(network, f"action_map_{tower}")
            parent = torch.nonzero(mapping == action).flatten()
            if len(parent):
                opinions.append(normalized[tower][0, parent.item()])
        assert opinions
        assert torch.allclose(policy[0, action], torch.stack(opinions).mean(), atol=1e-5)


def test_triad_towers_are_frozen_and_stay_in_eval_mode() -> None:
    _, _, _, network = _triad()
    network.train()
    assert all(
        not parameter.requires_grad
        for tower in network.towers
        for parameter in tower.parameters()
    )
    assert all(not tower.training for tower in network.towers)
    assert network.router.training


def test_builder_pins_candidate_rungs_and_writes_resumable_checkpoint(tmp_path: Path) -> None:
    _, _, _, triad = _triad()
    paths = []
    for tower, (candidate, rung) in zip(
        triad.towers,
        (("s-window-128", 18), ("s-scan-gru", 10), ("s-tape4", 8)),
        strict=True,
    ):
        path = tmp_path / f"{candidate}.pt"
        torch.save(
            {
                "network": tower.state_dict(),
                "candidate": candidate,
                "stage": rung,
                "source": STAGES[rung][0],
                "scramble": STAGES[rung][1],
                "when": "after",
            },
            path,
        )
        paths.append(path)

    output = tmp_path / "run" / "checkpoints" / "s-triad-wst.pt"
    report = build_triad_checkpoint(*paths, output)
    saved = torch.load(output, map_location="cpu", weights_only=False)
    assert [parent.rung for parent in report.parents] == [18, 10, 8]
    assert saved["candidate"] == "s-triad-wst"
    assert saved["stages"] == []
    assert len(saved["triad_parents"]) == 3


def test_all_union_actions_have_at_least_one_parent() -> None:
    _, _, _, network = _triad()
    assert bool(network.action_support.any(dim=1).all())
    # Tape writes are private to the tape expert; centred edits have all three votes.
    assert int(network.action_support.sum(dim=1).min()) == 1
    assert int(network.action_support.sum(dim=1).max()) == 3
