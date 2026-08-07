from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rf_knots.actions import DESTABILIZE, REDUCE

from pgx_mcts_bench.adaptive_scientists import FixedWordGame, KnotItem, load_scientist
from pgx_mcts_bench.collaboration_eval import (
    compare_collaboration_evaluations,
    evaluate_collaboration,
    export_collaboration_scientist,
)
from pgx_mcts_bench.collaborative_scientists import (
    _bank_payload,
    _commit_round,
    _refresh_schedule,
    _strict_shared_improvement,
    common_structural_objective_cap,
    play_with_common_objective_restarts,
    play_with_objective_restarts,
    run_collaborative_scientists,
    stratified_banks,
    translate_semantic_record,
    verified_record_cost,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, candidates, serial_arms
from pgx_mcts_bench.networks import make_braid_network
from pgx_mcts_bench.sharing_gate import (
    _active_training_records,
    _balanced_witness_positions,
    _paired_evaluation_summary,
    _primary_sharing_gate_passed,
    _routable_training_schedule,
    summarize_sharing_multiseed,
)


def _window_candidate():
    return next(candidate for candidate in serial_arms() if candidate.name == "s-window-128")


def test_stratified_banks_are_identity_disjoint_and_span_quartiles() -> None:
    bank, anchors = stratified_banks(20, 8, seed=17)
    assert len(bank) == 20
    assert len(anchors) == 8
    assert {item.id for item in bank}.isdisjoint(item.id for item in anchors)
    assert {item.difficulty_quartile for item in bank} == {0, 1, 2, 3}
    assert bank == sorted(bank, key=lambda item: (item.cheap_score, item.id))


def test_pilot_banks_are_not_unknotting_number_one_only() -> None:
    bank, anchors = stratified_banks(200, 70, seed=20260802)
    assert sum((item.certified_unknotting_lower_bound or 0) >= 2 for item in bank) == 87
    assert sum((item.certified_unknotting_lower_bound or 0) >= 2 for item in anchors) == 27
    assert {
        item.id: item.known_unknotting_number
        for item in bank
        if item.known_unknotting_number is not None
    } == {"10_124": 4, "7_5": 2}
    assert {
        item.id: item.known_unknotting_number
        for item in anchors
        if item.known_unknotting_number is not None
    } == {"8_19": 3}


def test_semantic_destabilization_translates_to_native_serial_record() -> None:
    candidate = _window_candidate()
    config = _config(candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    game = make_game(config.game)
    receiver = SimpleNamespace(game=game)
    knot = KnotItem("stabilized-unknot", 1, (1,), 2)
    destabilize = config.game._spec.encode(DESTABILIZE)
    record = translate_semantic_record(receiver, knot, 10.0, [destabilize], seed=9)
    assert record is not None
    assert record[0].shared_witness
    assert record[0].objective_ratio == 10.0
    assert record[0].option_state is not None
    assert record[0].target_external_action == destabilize
    assert verified_record_cost(game, knot, 10.0, record)[:2] == (0, 1)
    rescued, failed_cost, shared_cost = _strict_shared_improvement(receiver, knot, 10.0, [], record)
    assert rescued
    assert failed_cost is None
    assert shared_cost == 1.0
    duplicate, own_cost, duplicate_cost = _strict_shared_improvement(
        receiver, knot, 10.0, record, record
    )
    assert not duplicate
    assert own_cost == duplicate_cost == 1.0
    stale, archived_cost, stale_cost = _strict_shared_improvement(
        receiver,
        knot,
        10.0,
        [],
        record,
        best_native_objective=0.5,
    )
    assert not stale
    assert archived_cost == 0.5
    assert stale_cost == 1.0


def test_verified_cost_separates_semantic_moves_from_receiver_head_route() -> None:
    candidate = next(candidate for candidate in candidates() if candidate.name == "s-tape4")
    config = _config(candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    game = make_game(config.game)
    knot = KnotItem("routed-unknot", 3, (1, 1, -1), 2)
    semantic = [
        config.game._spec.encode(REDUCE, position=1),
        config.game._spec.encode(DESTABILIZE),
    ]

    record = translate_semantic_record(
        SimpleNamespace(game=game), knot, 10.0, semantic, seed=13
    )

    assert record is not None
    assert len(record) == 3  # shift, reduction, destabilization
    assert len(semantic) == 2  # portable proof omits the receiver-only shift
    assert verified_record_cost(game, knot, 10.0, record)[:2] == (0, 2)
    assert record[0].final_moves == 2
    assert record[0].final_native_plies == 3
    assert record[0].final_internal_plies == 1

    window_config = _config(
        _window_candidate(), ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1
    )
    window_game = make_game(window_config.game)
    direct = translate_semantic_record(
        SimpleNamespace(game=window_game), knot, 10.0, semantic, seed=13
    )
    assert direct is not None
    assert len(direct) == 2
    assert verified_record_cost(window_game, knot, 10.0, direct)[:2] == (0, 2)


def test_internal_controller_action_does_not_spend_semantic_L_budget() -> None:
    candidate = next(candidate for candidate in candidates() if candidate.name == "s-tape4")
    config = _config(candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    config = replace(config, game=replace(config.game, objective_budget_channel=True))
    game = make_game(config.game)
    fixed = FixedWordGame(
        game, KnotItem("route", 3, (1, 1, -1), 2), 10.0, objective_cap=20.0
    )
    before = fixed.reset(0)
    shift = int(
        game._shift_base  # noqa: SLF001
        + np.flatnonzero(before.legal_actions[game._shift_base :])[0]  # noqa: SLF001
    )

    after = fixed.step(before.state, shift)

    assert np.array_equal(after.observation[..., -1], before.observation[..., -1])
    assert fixed.semantic_move_count(after.state) == 0
    assert fixed.native_ply_count(after.state) == 1
    assert fixed.internal_ply_count(after.state) == 1


def test_translation_rejects_more_than_five_internal_actions(monkeypatch) -> None:
    candidate = _window_candidate()
    config = _config(candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    game = make_game(config.game)
    knot = KnotItem("stabilized-unknot", 1, (1,), 2)
    destabilize = config.game._spec.encode(DESTABILIZE)
    monkeypatch.setattr(
        "pgx_mcts_bench.collaborative_scientists._best_destination",
        lambda *_args: ([0] * 6, 0, 0),
    )

    record = translate_semantic_record(
        SimpleNamespace(game=game),
        knot,
        10.0,
        [destabilize],
        seed=9,
        internal_action_cap=5,
    )

    assert record is None


def test_solo_compute_matched_requires_one_scientist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one scientist"):
        run_collaborative_scientists(
            {"a": tmp_path / "a.pt", "b": tmp_path / "b.pt"},
            tmp_path / "run",
            arm="solo-compute-matched",
            rounds=0,
        )


def test_remaining_budget_channel_is_independent_of_hard_objective_cap(
    tmp_path: Path,
) -> None:
    candidate = _window_candidate()
    config = _config(candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    checkpoint = tmp_path / "initial.pt"
    torch.save(
        {"network": make_braid_network(config.game, config.model).state_dict()},
        checkpoint,
    )

    report = run_collaborative_scientists(
        {candidate.name: checkpoint},
        tmp_path / "run",
        arm="static-no-sharing",
        rounds=0,
        pool_size=2,
        anchor_size=1,
        remaining_budget_channel=True,
        objective_budget=False,
        action_horizon=128,
    )

    assert report["remaining_budget_channel"] is True
    assert report["objective_budget"] is False
    assert report["solution_definition"]["native_action_horizon"] == 128


def test_runner_freezes_external_identity_disjoint_banks(tmp_path: Path) -> None:
    candidate = _window_candidate()
    config = _config(candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    checkpoint = tmp_path / "initial.pt"
    torch.save(
        {"network": make_braid_network(config.game, config.model).state_dict()},
        checkpoint,
    )
    bank, anchor = stratified_banks(4, 2, seed=19)
    bank_path = tmp_path / "bank.json"
    anchor_path = tmp_path / "anchor.json"
    bank_path.write_text(json.dumps(_bank_payload(bank)))
    anchor_path.write_text(json.dumps(_bank_payload(anchor)))

    report = run_collaborative_scientists(
        {candidate.name: checkpoint},
        tmp_path / "run",
        arm="static-no-sharing",
        rounds=0,
        input_bank=bank_path,
        input_anchor_bank=anchor_path,
    )

    assert report["bank_sha256"]
    assert report["anchor_sha256"]
    assert report["input_banks"]["base"]["path"] == str(bank_path.resolve())
    assert json.loads((tmp_path / "run/base.json").read_text()) == _bank_payload(bank)


def test_objective_budget_accepts_exact_cap_and_marks_censoring() -> None:
    candidate = _window_candidate()
    config = _config(candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    config = replace(config, game=replace(config.game, objective_budget_channel=True))
    game = make_game(config.game)
    knot = KnotItem("stabilized-unknot", 1, (1,), 2)
    destabilize = config.game._spec.encode(DESTABILIZE)
    native = translate_semantic_record(
        SimpleNamespace(game=game), knot, 10.0, [destabilize], seed=7
    )
    assert native is not None

    exact = FixedWordGame(game, knot, 10.0, objective_cap=1.0).reset(0)
    solved = FixedWordGame(game, knot, 10.0, objective_cap=1.0).step(exact.state, native[0].action)
    assert solved.terminated
    assert solved.termination_reason == "solved"

    censored = FixedWordGame(game, knot, 10.0, objective_cap=0.0).reset(0)
    assert censored.terminated
    assert censored.termination_reason == "objective_budget_exhausted"

    small = FixedWordGame(game, knot, 10.0, objective_cap=10.0).reset(0)
    large = FixedWordGame(game, knot, 10.0, objective_cap=20.0).reset(0)
    assert np.allclose(large.observation[..., -1], 2 * small.observation[..., -1])
    assert float(large.observation[0, 0, -1]) < 1.0


def test_budget_channel_without_explicit_cap_is_soft() -> None:
    candidate = _window_candidate()
    config = _config(candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    config = replace(config, game=replace(config.game, objective_budget_channel=True))
    game = make_game(config.game)
    knot = KnotItem("over-budget-unsolved", 1, (1,), 2)
    fixed = FixedWordGame(game, knot, 10.0)
    transition = fixed.reset(0)

    # A negative soft remainder is an input feature, not a terminal event.
    serial_state = transition.state.base_state
    raw = game.unwrap(serial_state)
    raw = raw.replace(_crossing_changes=np.int32(2 * config.game.simplify_budget))
    updated_state = (raw, *serial_state[1:])
    soft = fixed._budgeted(
        replace(transition, state=updated_state), fixed._global_cap()
    )

    assert float(soft.observation[0, 0, -1]) < 0.0
    assert not soft.terminated
    assert soft.termination_reason != "objective_budget_exhausted"


def test_old_checkpoint_ignores_new_objective_budget_channel(tmp_path: Path) -> None:
    by_name = {candidate.name: candidate for candidate in candidates()}
    knot = KnotItem("stabilized-unknot", 1, (1,), 2)
    for name in ("s-window-128", "s-tape4", "s-w11-128"):
        candidate = by_name[name]
        config = _config(candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
        checkpoint = tmp_path / f"{name}.pt"
        old_network = make_braid_network(config.game, config.model).eval()
        torch.save({"network": old_network.state_dict()}, checkpoint)
        scientist = load_scientist(
            name,
            checkpoint,
            seed=0,
            device="cpu",
            objective_budget_channel=True,
        )
        old_observation = FixedWordGame(make_game(config.game), knot, 10.0).reset(0).observation
        new_observation = (
            FixedWordGame(scientist.game, knot, 10.0, objective_cap=12.0).reset(0).observation
        )
        old_tensor = torch.from_numpy(old_observation).permute(2, 0, 1)[None]
        new_tensor = torch.from_numpy(new_observation).permute(2, 0, 1)[None]
        with torch.inference_mode():
            old_outputs = old_network(old_tensor)
            new_outputs = scientist.network.eval()(new_tensor)
            old_auxiliary = old_network.forward_with_auxiliary(old_tensor)[2]
            new_auxiliary = scientist.network.forward_with_auxiliary(new_tensor)[2]
        assert new_observation.shape[-1] == old_observation.shape[-1] + 1
        torch.testing.assert_close(new_outputs[0], old_outputs[0])
        torch.testing.assert_close(new_outputs[1], old_outputs[1])
        for new_head, old_head in zip(new_auxiliary, old_auxiliary, strict=True):
            torch.testing.assert_close(new_head, old_head)
        assert scientist.config.model.auxiliary_budget_conditioning
        assert scientist.config.model.auxiliary_solve_backprop_to_encoder
        assert scientist.config.model.freeze_batchnorm_stats
        expected_learning_rate = 5e-5 if name == "s-tape4" else 2.5e-4
        assert scientist.config.train.learning_rate == expected_learning_rate
        expected_auxiliary_rate = 1e-3 if name == "s-tape4" else 2.5e-4
        assert [group["lr"] for group in scientist.optimizer.param_groups] == [
            expected_learning_rate,
            expected_auxiliary_rate,
        ]
        expected_preservation = {
            "s-window-128": 1.0,
            "s-tape4": 20.0,
            "s-w11-128": 5.0,
        }[name]
        assert scientist.config.model.policy_value_preservation_weight == expected_preservation


def test_tape4_h5_migrates_two_trailing_budget_channels_exactly(tmp_path: Path) -> None:
    by_name = {candidate.name: candidate for candidate in candidates()}
    source = by_name["s-tape4"]
    source_config = _config(source, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    source_network = make_braid_network(source_config.game, source_config.model).eval()
    checkpoint = tmp_path / "s-tape4.pt"
    torch.save({"network": source_network.state_dict()}, checkpoint)

    scientist = load_scientist(
        "s-tape4-h5",
        checkpoint,
        seed=0,
        device="cpu",
        objective_budget_channel=True,
    )
    knot = KnotItem("stabilized-unknot", 1, (1,), 2)
    old_observation = FixedWordGame(
        make_game(source_config.game), knot, 10.0
    ).reset(0).observation
    new_observation = FixedWordGame(
        scientist.game, knot, 10.0, objective_cap=12.0
    ).reset(0).observation
    old_tensor = torch.from_numpy(old_observation).permute(2, 0, 1)[None]
    new_tensor = torch.from_numpy(new_observation).permute(2, 0, 1)[None]

    with torch.inference_mode():
        old_outputs = source_network.forward_with_auxiliary(old_tensor)
        new_outputs = scientist.network.eval().forward_with_auxiliary(new_tensor)

    assert new_observation.shape[-1] == old_observation.shape[-1] + 2
    torch.testing.assert_close(new_outputs[0], old_outputs[0])
    torch.testing.assert_close(new_outputs[1], old_outputs[1])
    for new_head, old_head in zip(new_outputs[2], old_outputs[2], strict=True):
        torch.testing.assert_close(new_head, old_head)


def test_objective_budget_underestimate_restarts_geometrically(monkeypatch) -> None:
    calls = []

    def fake_play(*args, objective_cap=None, **kwargs):
        del args, kwargs
        calls.append(objective_cap)
        return [
            SimpleNamespace(
                solved=float(len(calls) == 2),
                final_moves=float(len(calls)),
                final_native_plies=float(len(calls)),
                objective_censored=len(calls) == 1,
            )
        ]

    monkeypatch.setattr("pgx_mcts_bench.collaborative_scientists._play", fake_play)
    scientist = SimpleNamespace(
        config=SimpleNamespace(
            game=SimpleNamespace(
                objective_budget_channel=True,
                simplify_budget=20,
            )
        )
    )
    retained = []
    record, budget = play_with_objective_restarts(
        scientist,
        KnotItem("x", 3, (1, 1, 1), 2),
        10.0,
        predicted_objective=1.0,
        simulations=1,
        seed=5,
        retained_records=retained,
    )
    assert record[0].solved == 1.0
    assert calls == [2, 4]
    assert budget["restart_count"] == 1
    assert budget["attempts"][0]["objective_budget_exhausted"]
    assert len(retained) == 2
    assert retained[0][0].solved == 0.0
    assert retained[1] is record


def test_common_structural_cap_uses_complete_representation_not_scientist_prediction() -> None:
    knot = KnotItem("x", 3, (1, -2, 1, -1, 2), 3)

    # ceil(5 observed intersections / 2) crossing changes plus the full
    # 20-semantic-move allowance. No network output enters this calculation.
    assert common_structural_objective_cap(knot, 10.0, 20) == 50.0
    assert common_structural_objective_cap(knot, 1000.0, 20) == 3020.0


def test_common_structural_cap_restarts_every_censored_failure_at_global_cap(
    monkeypatch,
) -> None:
    calls = []

    def fake_play(*args, objective_cap=None, cap_type="", **kwargs):
        del args, kwargs
        calls.append((objective_cap, cap_type))
        return [
            SimpleNamespace(
                solved=float(len(calls) == 2),
                final_moves=float(len(calls)),
                final_native_plies=float(len(calls)),
                objective_censored=len(calls) == 1,
            )
        ]

    monkeypatch.setattr("pgx_mcts_bench.collaborative_scientists._play", fake_play)
    scientist = SimpleNamespace(
        config=SimpleNamespace(
            game=SimpleNamespace(
                objective_budget_channel=True,
                simplify_budget=20,
            )
        )
    )
    retained = []
    record, budget = play_with_common_objective_restarts(
        scientist,
        KnotItem("x", 3, (1, -2, 1, -1, 2), 3),
        10.0,
        simulations=1,
        seed=5,
        retained_records=retained,
    )

    assert record[0].solved == 1.0
    assert calls == [(50.0, "structural"), (220.0, "common-global-restart")]
    assert budget["restart_count"] == 1
    assert budget["scientist_prediction_used"] is False
    assert budget["attempts"][0]["objective_budget_exhausted"]
    assert len(retained) == 2


def test_common_structural_cap_does_not_repeat_an_ordinary_horizon_failure(
    monkeypatch,
) -> None:
    calls = []

    def fake_play(*args, objective_cap=None, **kwargs):
        del args, kwargs
        calls.append(objective_cap)
        return [
            SimpleNamespace(
                solved=0.0,
                final_moves=0.0,
                final_native_plies=20.0,
                objective_censored=False,
            )
        ]

    monkeypatch.setattr("pgx_mcts_bench.collaborative_scientists._play", fake_play)
    scientist = SimpleNamespace(
        config=SimpleNamespace(
            game=SimpleNamespace(
                objective_budget_channel=True,
                simplify_budget=20,
            )
        )
    )
    _, budget = play_with_common_objective_restarts(
        scientist,
        KnotItem("x", 3, (1, -2, 1, -1, 2), 3),
        10.0,
        simulations=1,
        seed=5,
    )

    assert calls == [50.0]
    assert budget["restart_count"] == 0


def test_round_commit_is_immutable_and_schedule_is_rebuilt(tmp_path: Path) -> None:
    event = {"round": 0, "selected": "x"}
    _commit_round(tmp_path, 0, event, {"value": torch.tensor(3)})
    assert _refresh_schedule(tmp_path) == [event]
    assert (tmp_path / "schedule.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "rounds/000000/state.pt.gz").is_file()
    assert not (tmp_path / "rounds/000000/state.pt").exists()


def test_minimal_run_resumes_from_committed_round(tmp_path: Path) -> None:
    candidate = _window_candidate()
    config = _config(candidate, ("R(3,12)#0", 0), 0, "cpu", selfplay_games=1)
    checkpoint = tmp_path / "initial.pt"
    torch.save(
        {"network": make_braid_network(config.game, config.model).state_dict()},
        checkpoint,
    )
    output = tmp_path / "run"
    kwargs = dict(
        checkpoints={candidate.name: checkpoint},
        output=output,
        arm="static-sharing",
        pool_size=4,
        anchor_size=2,
        frontier=2,
        ratios=(10.0,),
        qualification_simulations=1,
        simulations=1,
        train_every=10,
        train_steps=0,
        attempt_workers=2,
        objective_budget=True,
        bank_seed=11,
        seed=23,
        device="cpu",
    )
    first = run_collaborative_scientists(rounds=1, **kwargs)
    resumed = run_collaborative_scientists(rounds=2, resume=True, **kwargs)
    assert first["completed_rounds"] == 1
    assert resumed["completed_rounds"] == 2
    assert first["schema"] == "collaborative-scientists-v6-soft-budget-horizon"
    assert first["objective_budget_attempt_protocol"]["scientist_prediction_used"] is False
    assert first["solution_definition"]["native_action_horizon"] == 64
    assert first["solution_definition"]["moves"].startswith("verified portable semantic")
    assert "excluded from L_A:B" in first["solution_definition"]["internal_plies"]
    assert "state gate" in first["sharing_policy_adapter"]["initialization"]
    assert len((output / "schedule.jsonl").read_text().splitlines()) == 2
    exported = export_collaboration_scientist(output, candidate.name, tmp_path / "exported.pt")
    evaluated = evaluate_collaboration(
        output,
        tmp_path / "evaluation",
        state="final",
        split="new70",
        simulations=1,
        limit=1,
        seed=29,
    )
    assert exported["checkpoint_sha256"]
    assert evaluated["completed_items"] == 1
    assert evaluated["attempts_per_representation"] == 4
    assert len(json.loads((tmp_path / "evaluation/items/0000.json").read_text())["attempts"]) == 4


def test_primary_sharing_gate_uses_aggregate_objective_not_exact_retention() -> None:
    sharing_blocks = [
        {
            "distinct_witnesses": 10,
            # Route imitation is diagnostic: heterogeneous witnesses may
            # legitimately disagree with one scientist's preferred method.
            "route_loss_relative_reduction": {"a": 0.2, "b": -0.1},
        }
    ]

    assert _primary_sharing_gate_passed(100.0, 120.0, sharing_blocks)
    assert not _primary_sharing_gate_passed(121.0, 120.0, sharing_blocks)
    assert not _primary_sharing_gate_passed(100.0, 120.0, [])


def test_witness_position_sampling_is_representation_balanced() -> None:
    def record(identity: str, size: int):
        return [
            SimpleNamespace(
                representation_id=identity,
                option_state=object(),
                target_external_action=1,
            )
            for _ in range(size)
        ]

    selected = _balanced_witness_positions(
        [record("short", 2), record("long", 20)],
        np.random.default_rng(7),
        positions_per_witness=4,
    )
    identities = [position.representation_id for position in selected]
    assert identities.count("short") == 4
    assert identities.count("long") == 4


def test_simulation_dose_summary_preserves_paired_sets_objectives_and_compute() -> None:
    def evaluation(costs: dict[str, float | None], compute: int) -> dict[str, object]:
        rows = []
        for item, cost in costs.items():
            solved = cost is not None
            rows.append(
                {
                    "item": item,
                    "solved": solved,
                    "best_objective": cost,
                    "attempts": [
                        {
                            "compute": {
                                "scheduled_network_evaluations": compute,
                                "wall_seconds": 0.5,
                            }
                        }
                    ],
                }
            )
        return {"rows": rows}

    summary = _paired_evaluation_summary(
        evaluation({"a": 10.0, "b": None}, 100),
        evaluation({"a": 12.0, "b": 20.0}, 120),
        failure=264.0,
    )

    assert summary["intersection"] == ["a"]
    assert summary["sharing_only"] == []
    assert summary["control_only"] == ["b"]
    assert summary["sharing_capped_loss"] == 274.0
    assert summary["control_capped_loss"] == 32.0
    assert summary["intersection_objective_sharing"] == 10.0
    assert summary["intersection_objective_control"] == 12.0
    assert summary["sharing_scheduled_network_evaluations"] == 200
    assert summary["control_scheduled_network_evaluations"] == 240


def test_multiseed_sharing_summary_uses_mean_and_median_paired_delta(
    tmp_path: Path,
) -> None:
    common = {
        "schema": "semantic-cost-block-balanced-option-adapter-sharing-v11",
        "bank_sha256": "bank",
        "items": ["a", "b"],
        "target_items": ["a"],
        "generalization_items": [],
        "ratio": 10.0,
        "training_simulations": 64,
        "evaluation_simulations": 128,
        "evaluation_games": 8,
        "update_cycles": 10,
        "batch_size": 16,
        "option_learning_rate_scale": 1.0,
        "sharing_block_size": 10,
        "sharing_interval_cycles": 10,
        "adapter_steps_per_block": 16,
        "option_positions_per_witness": 4,
        "option_adapter_base_learning_rate": 1e-3,
        "option_adapter_effective_learning_rate": 1e-3,
        "native_refresh_games": 4,
        "gated_adapter": True,
        "route_gate_weight": 0.1,
        "off_route_kl_weight": 1.0,
        "off_route_gate_weight": 0.1,
        "off_route_batch_size": 32,
        "receivers": [{"name": "student", "checkpoint": "x", "sha256": "y"}],
    }

    def paired(sharing_loss: float, control_loss: float, sharing_only: list[str]):
        return {
            "sharing_capped_loss": sharing_loss,
            "control_capped_loss": control_loss,
            "sharing_solved": ["a", *sharing_only],
            "control_solved": ["a"],
            "sharing_only": sharing_only,
            "control_only": [],
            "intersection": ["a"],
        }

    runs = []
    for index, delta in enumerate((-10.0, 5.0, -3.0)):
        root = tmp_path / f"seed-{index}"
        root.mkdir()
        manifest = {
            **common,
            "seed": 100 + index,
            "protocol_sha256": f"protocol-{index}",
        }
        final = paired(100.0 + delta, 100.0, ["b"] if index == 0 else [])
        report = {
            "receivers": {
                "student": {
                    "completed_sharing_blocks": 1,
                    "paired_final": final,
                    "paired_training_targets": final,
                    "paired_non_target_canaries": final,
                    "paired_generalization": final,
                    "lost_from_before": [],
                    "sharing_blocks": [
                        {"mean_route_loss_relative_reduction": -0.2}
                    ],
                }
            }
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "report.json").write_text(json.dumps(report))
        runs.append(root)

    summary = summarize_sharing_multiseed(tuple(runs), tmp_path / "summary")
    receiver = summary["receivers"]["student"]
    assert receiver["mean_capped_loss_delta_sharing_minus_control"] < 0.0
    assert receiver["median_capped_loss_delta_sharing_minus_control"] == -3.0
    assert receiver["sharing_wins"] == 2
    assert receiver["control_wins"] == 1
    assert receiver["training_targets_summary"]["sharing_wins"] == 2
    assert (
        receiver["non_target_canary_summary"][
            "mean_delta_sharing_minus_control"
        ]
        < 0.0
    )
    assert receiver["passed"]
    assert receiver["generalization_summary"]["sharing_wins"] == 2
    assert summary["decision"]["passed"]


def test_active_training_records_exclude_non_target_canaries() -> None:
    records = [
        [SimpleNamespace(representation_id="training")],
        [SimpleNamespace(representation_id="canary")],
    ]

    selected = _active_training_records(records, {"training"})

    assert [record[0].representation_id for record in selected] == ["training"]


def test_routable_schedule_excludes_failed_translations() -> None:
    schedule = _routable_training_schedule({"translated-b": 20.0, "translated-a": 10.0})

    assert schedule == ["translated-a", "translated-b"]
    with pytest.raises(ValueError, match="no target witness is routable"):
        _routable_training_schedule({})


def test_paired_evaluation_reports_exact_solved_sets(tmp_path: Path) -> None:
    def write(name: str, solved: list[str], objectives: dict[str, float], capped: float):
        root = tmp_path / name
        root.mkdir()
        (root / "report.json").write_text(
            json.dumps(
                {
                    "split_sha256": "same-bank",
                    "completed_items": 3,
                    "summary": {
                        "10.0": {
                            "portfolio_solved": len(solved),
                            "capped_objective_sum": capped,
                            "solved_items": solved,
                            "best_by_item": {
                                item: {"objective": objectives[item]} for item in solved
                            },
                            "compute": {
                                "scheduled_network_evaluations": 100,
                                "wall_seconds": 2.0,
                            },
                        }
                    },
                }
            )
        )
        return root

    treatment = write("treatment", ["a", "b"], {"a": 7.0, "b": 9.0}, 30.0)
    control = write("control", ["a", "c"], {"a": 10.0, "c": 8.0}, 35.0)
    paired = compare_collaboration_evaluations(treatment, control)
    ratio = paired["comparisons"]["10.0"]
    assert ratio["intersection"] == ["a"]
    assert ratio["treatment_only"] == ["b"]
    assert ratio["control_only"] == ["c"]
    assert ratio["common_objective_delta_sum"] == -3.0
    assert ratio["capped_objective_delta"] == -5.0
