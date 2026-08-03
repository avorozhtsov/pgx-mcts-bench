from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from rf_knots.actions import DESTABILIZE

from pgx_mcts_bench.adaptive_scientists import FixedWordGame, KnotItem, load_scientist
from pgx_mcts_bench.collaboration_eval import (
    compare_collaboration_evaluations,
    evaluate_collaboration,
    export_collaboration_scientist,
)
from pgx_mcts_bench.collaborative_scientists import (
    _commit_round,
    _refresh_schedule,
    _strict_shared_improvement,
    play_with_objective_restarts,
    run_collaborative_scientists,
    stratified_banks,
    translate_semantic_record,
    verified_record_cost,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, candidates, serial_arms
from pgx_mcts_bench.networks import make_braid_network


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
    assert sum(
        (item.certified_unknotting_lower_bound or 0) >= 2 for item in bank
    ) == 87
    assert sum(
        (item.certified_unknotting_lower_bound or 0) >= 2 for item in anchors
    ) == 27
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
    assert verified_record_cost(game, knot, 10.0, record)[:2] == (0, 1)
    rescued, failed_cost, shared_cost = _strict_shared_improvement(
        receiver, knot, 10.0, [], record
    )
    assert rescued
    assert failed_cost is None
    assert shared_cost == 1.0
    duplicate, own_cost, duplicate_cost = _strict_shared_improvement(
        receiver, knot, 10.0, record, record
    )
    assert not duplicate
    assert own_cost == duplicate_cost == 1.0


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
    solved = FixedWordGame(game, knot, 10.0, objective_cap=1.0).step(
        exact.state, native[0].action
    )
    assert solved.terminated
    assert solved.termination_reason == "solved"

    censored = FixedWordGame(game, knot, 10.0, objective_cap=0.0).reset(0)
    assert censored.terminated
    assert censored.termination_reason == "objective_budget_exhausted"


def test_old_checkpoint_ignores_new_objective_budget_channel(tmp_path: Path) -> None:
    by_name = {candidate.name: candidate for candidate in candidates()}
    knot = KnotItem("stabilized-unknot", 1, (1,), 2)
    for name in ("s-window-128", "d-tape4-u1", "s-w11-128"):
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
        old_observation = FixedWordGame(make_game(config.game), knot, 10.0).reset(
            0
        ).observation
        new_observation = FixedWordGame(
            scientist.game, knot, 10.0, objective_cap=12.0
        ).reset(0).observation
        old_tensor = torch.from_numpy(old_observation).permute(2, 0, 1)[None]
        new_tensor = torch.from_numpy(new_observation).permute(2, 0, 1)[None]
        with torch.inference_mode():
            old_outputs = old_network(old_tensor)
            new_outputs = scientist.network.eval()(new_tensor)
        assert new_observation.shape[-1] == old_observation.shape[-1] + 1
        torch.testing.assert_close(new_outputs[0], old_outputs[0])
        torch.testing.assert_close(new_outputs[1], old_outputs[1])


def test_objective_budget_underestimate_restarts_geometrically(monkeypatch) -> None:
    calls = []

    def fake_play(*args, objective_cap=None, **kwargs):
        del args, kwargs
        calls.append(objective_cap)
        return [
            SimpleNamespace(
                solved=float(len(calls) == 2),
                final_moves=float(len(calls)),
            )
        ]

    monkeypatch.setattr(
        "pgx_mcts_bench.collaborative_scientists._play", fake_play
    )
    scientist = SimpleNamespace(
        config=SimpleNamespace(
            game=SimpleNamespace(
                objective_budget_channel=True,
                simplify_budget=20,
            )
        )
    )
    record, budget = play_with_objective_restarts(
        scientist,
        KnotItem("x", 3, (1, 1, 1), 2),
        10.0,
        predicted_objective=1.0,
        simulations=1,
        seed=5,
    )
    assert record[0].solved == 1.0
    assert calls == [2, 4]
    assert budget["restart_count"] == 1
    assert budget["attempts"][0]["objective_budget_exhausted"]


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
        bank_seed=11,
        seed=23,
        device="cpu",
    )
    first = run_collaborative_scientists(rounds=1, **kwargs)
    resumed = run_collaborative_scientists(rounds=2, resume=True, **kwargs)
    assert first["completed_rounds"] == 1
    assert resumed["completed_rounds"] == 2
    assert len((output / "schedule.jsonl").read_text().splitlines()) == 2
    exported = export_collaboration_scientist(
        output, candidate.name, tmp_path / "exported.pt"
    )
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
