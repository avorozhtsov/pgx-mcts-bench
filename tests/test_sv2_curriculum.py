from pathlib import Path

import numpy as np

from pgx_mcts_bench.adaptive_scientists import KnotItem
from pgx_mcts_bench.data import Position
from pgx_mcts_bench.sv2_curriculum import (
    F_NATIVE_LEVELS,
    F_OLD_LEVELS,
    SIMULATION_LEVELS,
    _assert_native_commit,
    _commit_native_event,
    _coordinated_name,
    _donation_is_still_eligible,
    _initial_controller_values,
    _portfolio_summary,
    adapt_donation_dose,
    auditable_complexity,
    build_prefix24,
    build_r200,
    coordinated_block_report,
    next_compute_dose,
    next_rehearsal_dose,
    run_static_no_sharing,
    write_prefix24,
)


def test_native_event_is_immutable_and_hash_verified(tmp_path: Path) -> None:
    payload = {
        "schema": "semantic-v2-native-event-v1",
        "phase": "native-committed",
        "round": 0,
        "scientists": {"a": {"evaluation": {"1000.0": {"best_objective": 1007}}}},
    }
    reference = _commit_native_event(tmp_path, 0, payload)
    committed = tmp_path / reference["path"]
    assert committed.is_file()
    _assert_native_commit(tmp_path, reference)

    import pytest

    with pytest.raises(RuntimeError, match="different results"):
        _commit_native_event(tmp_path, 0, {**payload, "round": 1})

    committed.write_text("{}\n")
    with pytest.raises(RuntimeError, match="hash changed"):
        _assert_native_commit(tmp_path, reference)


def test_group_continuation_restores_adaptive_controller_values() -> None:
    payloads = {
        "a": {
            "f_native": 12,
            "simulations": 256,
            "donation_dose": 2,
            "donation_healthy_streak": 1,
        },
        "b": {
            "f_native": 8,
            "simulations": 128,
            "donation_dose": 2,
            "donation_healthy_streak": 1,
        },
    }
    assert _initial_controller_values(
        payloads,
        ["a", "b"],
        default_f_native=5,
        default_simulations=64,
    ) == (2, 1, {"a": 12, "b": 8}, {"a": 256, "b": 128})


def test_group_continuation_rejects_disagreeing_sharing_controller() -> None:
    import pytest

    with pytest.raises(ValueError, match="disagree"):
        _initial_controller_values(
            {"a": {"donation_dose": 1}, "b": {"donation_dose": 2}},
            ["a", "b"],
            default_f_native=5,
            default_simulations=64,
        )


def test_r200_uses_original_acs_with_presentation_length(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    upper_bounds = tmp_path / "upper.json"
    rows = [
        {
            "id": f"k{index:03d}",
            "name": f"k{index:03d}",
            "word": [1, -1, 1][: 1 + index % 3],
            "strands": 2 + index % 2,
            "crossings": 1,
            "certified_unknotting_lower_bound": 1 + index % 2,
            "difficulty_quartile": index // 50,
            "cheap_score": 0.0,
        }
        for index in range(200)
    ]
    source.write_text(__import__("json").dumps(rows))
    upper_bounds.write_text(
        __import__("json").dumps(
            {
                "values": {
                    row["id"]: {"upper_bound": row["certified_unknotting_lower_bound"] + 1}
                    for row in rows
                }
            }
        )
    )
    converted = build_r200(source, upper_bounds)
    assert len(converted) == 200
    assert [row["acs"] for row in converted] == sorted(row["acs"] for row in converted)
    for row in converted:
        assert row["source_minimal_crossings"] == 1
        assert row["presentation_crossings"] == len(row["word"])
        assert row["crossings"] == len(row["word"])
        assert row["acs"] == (
            10 * row["strands"] + 5 * row["certified_unknotting_upper_bound"] + len(row["word"])
        )


def test_compute_dose_only_rises_after_deficient_block() -> None:
    assert next_compute_dose(5, levels=F_NATIVE_LEVELS, observed_rate=0.80, target=0.80) == 5
    assert next_compute_dose(5, levels=F_NATIVE_LEVELS, observed_rate=0.79, target=0.80) == 8
    assert next_compute_dose(16, levels=F_NATIVE_LEVELS, observed_rate=0.0, target=0.80) == 16
    assert next_compute_dose(64, levels=SIMULATION_LEVELS, observed_rate=0.69, target=0.70) == 128


def test_coordinated_block_report_counts_declared_work() -> None:
    iteration = {
        "selfplay_solved": 3,
        "selfplay_games": 4,
        "scheduled_network_evaluations": 50,
    }
    attempt = {"solved": True, "scheduled_network_evaluations": 7}
    events = []
    for round_index in range(2):
        events.append(
            {
                "round": round_index,
                "arm": "adaptive-sharing",
                "selected": f"k{round_index}",
                "translations": [{"admitted": round_index == 1}],
                "donation_guard": None,
                "scientists": {
                    "s": {
                        "iterations": [iteration],
                        "evaluation": {"10.0": {"attempts": [attempt]}},
                    }
                },
            }
        )
    events[-1]["donation_guard"] = {"accepted": True}
    events[-1]["scientists"]["s"]["rehearsal"] = {
        "F_old": 1,
        "after": {"solved": 4, "attempts": 4, "capped_cost": 11.0},
    }
    report = coordinated_block_report(events, block_size=2, f_old={"s": 2}, donation_dose=1)
    assert report["completed_rungs"] == 2
    assert report["selected"] == ["k0", "k1"]
    assert report["scientists"]["s"]["native_selfplay_solved"] == 6
    assert report["scientists"]["s"]["evaluation_solved"] == 2
    assert report["scientists"]["s"]["next_F_old"] == 2
    assert report["sharing"]["translated"] == 2
    assert report["sharing"]["admitted"] == 1


def test_final_partial_block_does_not_repeat_prior_rounds() -> None:
    events = [
        {
            "round": index,
            "arm": "adaptive-no-sharing",
            "selected": f"k{index}",
            "translations": [],
            "donation_guard": None,
            "scientists": {
                "s": {
                    "iterations": [],
                    "evaluation": {"10.0": {"attempts": []}},
                    **(
                        {
                            "rehearsal": {
                                "F_old": 1,
                                "after": {
                                    "solved": 24,
                                    "attempts": 24,
                                    "capped_cost": 24.0,
                                },
                            }
                        }
                        if index == 23
                        else {}
                    ),
                }
            },
        }
        for index in range(24)
    ]
    report = coordinated_block_report(events, block_size=10, f_old={"s": 1}, donation_dose=1)
    assert report["rounds"] == [20, 23]
    assert report["selected"] == ["k20", "k21", "k22", "k23"]


def test_prefix24_has_declared_phase_shape_and_order() -> None:
    rows = build_prefix24(seed=123)
    assert len(rows) == len({row["id"] for row in rows}) == 24
    assert [sum(row["phase"] == phase for row in rows) for phase in range(6)] == [
        6,
        6,
        3,
        3,
        3,
        3,
    ]
    assert sorted(row["scramble_moves"] for row in rows) == (
        [0] * 6 + [1] * 3 + [2] * 9 + [3] * 3 + [4] * 3
    )
    assert [row["acs"] for row in rows] == sorted(row["acs"] for row in rows)
    for row in rows:
        assert row["presentation_crossings"] == len(row["word"])
        assert row["acs"] == auditable_complexity(
            strands=row["strands"],
            unknotting_number=row["known_unknotting_number"],
            word_length=len(row["word"]),
        )


def test_rehearsal_dose_only_rises_after_unhealthy_block() -> None:
    assert next_rehearsal_dose(1, retention_solve_rate=0.8, capped_cost_worsened=False) == 1
    assert next_rehearsal_dose(1, retention_solve_rate=0.79, capped_cost_worsened=False) == 2
    assert next_rehearsal_dose(2, retention_solve_rate=1.0, capped_cost_worsened=True) == 4
    assert next_rehearsal_dose(8, retention_solve_rate=0.0, capped_cost_worsened=True) == 8
    assert F_OLD_LEVELS == (1, 2, 4, 8)


def test_donation_dose_needs_evidence_and_two_healthy_blocks() -> None:
    insufficient = adapt_donation_dose(
        1,
        healthy_streak=1,
        eligible_donations=9,
        donation_only_regression=False,
        portfolio_noninferior=True,
    )
    assert (insufficient.dose, insufficient.healthy_streak) == (1, 0)
    first = adapt_donation_dose(
        1,
        healthy_streak=0,
        eligible_donations=10,
        donation_only_regression=False,
        portfolio_noninferior=True,
    )
    assert (first.dose, first.healthy_streak) == (1, 1)
    second = adapt_donation_dose(
        1,
        healthy_streak=first.healthy_streak,
        eligible_donations=12,
        donation_only_regression=False,
        portfolio_noninferior=True,
    )
    assert (second.dose, second.healthy_streak) == (2, 0)
    regression = adapt_donation_dose(
        3,
        healthy_streak=1,
        eligible_donations=12,
        donation_only_regression=True,
        portfolio_noninferior=False,
    )
    assert (regression.dose, regression.healthy_streak) == (2, 0)


def test_static_manifest_freezes_learning_dose(tmp_path: Path, monkeypatch) -> None:
    bank = tmp_path / "bank.json"
    write_prefix24(bank, seed=5)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "pgx_mcts_bench.sv2_curriculum.source_provenance",
        lambda: {"base_commit": "test", "dirty": False},
    )
    monkeypatch.setattr(
        "pgx_mcts_bench.sv2_curriculum._run_scientist",
        lambda payload: {
            "scientist": payload["scientist"],
            "completed_rungs": len(payload["bank_rows"]),
        },
    )
    report = run_static_no_sharing(
        {"raster-axial": checkpoint},
        bank,
        tmp_path / "run",
        workers=1,
    )
    assert report["name"] == "SV2-3S-R24-SIM64-F10-AR-EV4-NO-SHARING"
    assert report["representations"] == 24
    assert report["F_native"] == 10
    assert report["simulations"] == 64
    assert report["evaluation_attempts_per_objective"] == 4
    assert report["evaluation_root_noise"] is True
    assert report["evaluation_attempt_protocol"].endswith("batched")
    assert report["adaptive_compute"] is False
    assert report["adaptive_rehearsal_only"] is True


def test_coordinated_arm_names_are_unambiguous() -> None:
    common = {
        "scientists": 3,
        "representations": 24,
        "simulations": 64,
        "f_native": 10,
        "evaluation_attempts": 4,
    }
    assert _coordinated_name("adaptive-no-sharing", **common).endswith("ADAPTIVE-NO-SHARING")
    assert _coordinated_name("static-sharing", **common).endswith("EV4-SHARING")
    assert _coordinated_name("adaptive-sharing", **common).endswith("ADAPTIVE-SHARING")


def test_portfolio_summary_takes_best_scientist_and_caps_failures() -> None:
    knot = KnotItem("x", 3, (1, -1, 1), 2)
    summaries = {
        "a": {
            "cells": {
                "x": {
                    "10.0": {"best_objective": 25.0},
                    "1000.0": {"best_objective": None},
                }
            }
        },
        "b": {
            "cells": {
                "x": {
                    "10.0": {"best_objective": 17.0},
                    "1000.0": {"best_objective": None},
                }
            }
        },
    }
    result = _portfolio_summary(summaries, [knot], (10.0, 1000.0), action_horizon=128)
    assert result["solved"] == 1
    assert result["capped_cost"] == 17.0 + 20_128.0
    assert result["objectives"]["10.0"] == {
        "attempts": 1,
        "solved": 1,
        "capped_cost": 17.0,
    }
    assert result["objectives"]["1000.0"] == {
        "attempts": 1,
        "solved": 0,
        "capped_cost": 20_128.0,
    }
    assert result["cells"]["x|10"]["scientist"] == "b"


def test_donation_eligibility_is_strict_and_ratio_specific() -> None:
    class Replay:
        def __init__(self, objective: float | None):
            self.objective = objective

        def best_native_objective(self, representation: str, ratio: float) -> float | None:
            assert representation == "x"
            assert ratio == 10.0
            return self.objective

    class Scientist:
        def __init__(self, objective: float | None):
            self.replay = Replay(objective)

    position = Position(
        observation=np.zeros((1,), dtype=np.float32),
        legal_actions=np.ones((1,), dtype=bool),
        policy=np.ones((1,), dtype=np.float32),
        action=0,
        player=0,
        shared_witness=True,
        representation_id="x",
        objective_ratio=10.0,
        final_crossing_changes=1.0,
        final_moves=3.0,
    )
    assert _donation_is_still_eligible(Scientist(None), [position])
    assert _donation_is_still_eligible(Scientist(14.0), [position])
    assert not _donation_is_still_eligible(Scientist(13.0), [position])
    assert not _donation_is_still_eligible(Scientist(12.0), [position])
