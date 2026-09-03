import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pgx_mcts_bench.sv2_curriculum as curriculum
from pgx_mcts_bench.adaptive_scientists import KnotItem
from pgx_mcts_bench.collaborative_scientists import BankItem
from pgx_mcts_bench.data import Position
from pgx_mcts_bench.sv2_curriculum import (
    F_NATIVE_LEVELS,
    F_OLD_LEVELS,
    SIMULATION_LEVELS,
    _assert_native_commit,
    _commit_native_event,
    _coordinated_name,
    _donation_is_still_eligible,
    _fixed_no_sharing_order,
    _frozen_static_random_order,
    _initial_controller_values,
    _portfolio_summary,
    adapt_donation_dose,
    auditable_complexity,
    build_prefix24,
    build_r200,
    coordinated_block_report,
    curriculum_skip_event,
    deterministic_rehearsal_panel,
    deterministic_rehearsal_task_order,
    next_compute_dose,
    next_rehearsal_dose,
    rehearsal_cumulative_timeout_seconds,
    rehearsal_timeout_debt,
    run_static_no_sharing,
    write_prefix24,
)


def test_expanding_round_robin_panel_uses_exact_order_and_durable_absolute_cursor() -> None:
    items = [
        BankItem(f"r{index}", KnotItem(f"k{index}", 3, (1, -1), 2), float(index), 0)
        for index in range(7)
    ]
    first, cursor, metadata = deterministic_rehearsal_panel(items, panel_size=4, cursor=0)
    second, cursor2, metadata2 = deterministic_rehearsal_panel(
        [*items, BankItem("r7", KnotItem("k7", 3, (1, -1), 2), 7.0, 0)],
        panel_size=4,
        cursor=cursor,
    )
    assert [item.id for item in first] == ["r0", "r1", "r2", "r3"]
    assert [item.id for item in second] == ["r4", "r5", "r6", "r7"]
    assert (cursor, cursor2) == (4, 8)
    assert metadata["policy"] == "exact-bank-order-expanding-round-robin-v1"
    assert metadata2["population_size"] == 8


def test_rehearsal_panel_can_be_reused_without_double_wrapping() -> None:
    items = [
        BankItem(f"r{index}", KnotItem(f"k{index}", 3, (1, -1), 2), float(index), 0)
        for index in range(3)
    ]
    first, cursor, _metadata = deterministic_rehearsal_panel(items, panel_size=2, cursor=0)

    reused, next_cursor, metadata = deterministic_rehearsal_panel(
        first,
        panel_size=2,
        cursor=0,
    )

    assert reused == first
    assert (cursor, next_cursor) == (2, 2)
    assert metadata["representations"] == ["r0", "r1"]


def test_seeded_rehearsal_task_order_interleaves_outcomes_and_is_replayable() -> None:
    items = [
        BankItem(f"r{index}", KnotItem(f"k{index}", 3, (1, -1), 2), float(index), 0)
        for index in range(4)
    ]
    cells = {}
    signatures = ((True, True), (True, False), (False, True), (False, False))
    for item, signature in zip(items, signatures, strict=True):
        cells[item.id] = {
            "10.0": {"best_objective": 10.0 if signature[0] else None},
            "1000.0": {"best_objective": 1000.0 if signature[1] else None},
        }
    first, metadata = deterministic_rehearsal_task_order(
        items,
        retention={"cells": cells},
        ratios=(10.0, 1000.0),
        exposure={},
        seed=1234,
    )
    second, metadata2 = deterministic_rehearsal_task_order(
        items,
        retention={"cells": cells},
        ratios=(10.0, 1000.0),
        exposure={},
        seed=1234,
    )
    assert [item.id for item in first] == [item.id for item in second]
    assert metadata == metadata2
    assert len(set(metadata["exposure_tiers"][0]["stratum_order"])) == 4
    assert metadata["outcome_signature_deficits"] == []
    assert metadata["policy"] == "seeded-outcome-interleaved-exposure-v1"


def test_rehearsal_timeout_debt_counts_only_missing_censored_iterations() -> None:
    events = [
        {
            "scientists": {
                "s": {"rehearsal": {"F_old": 8, "iterations": [{}, {}], "hard_timeout": {}}}
            }
        },
        {"scientists": {"s": {"rehearsal": {"F_old": 4, "iterations": [{}]}}}},
        {
            "scientists": {
                "s": {
                    "rehearsal": {
                        "F_old": 4,
                        "iterations": [{}, {}, {}],
                        "hard_timeout": {},
                    }
                }
            }
        },
    ]
    assert rehearsal_timeout_debt(events, "s") == 7


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


def test_native_phase_separates_training_and_evaluation_objectives(
    monkeypatch,
) -> None:
    calls = []

    class Replay:
        def set_representation_embedding(self, *args) -> None:
            pass

        def record_native_objective(self, *args) -> None:
            pass

    scientist = SimpleNamespace(replay=Replay())
    selected = SimpleNamespace(
        id="x",
        knot=KnotItem("x", 3, (1, -1, 1), 2),
    )

    def iteration(*args, ratios, **kwargs):
        calls.append(("train", ratios))
        return {"selfplay_games": 8}

    def evaluate(*args, ratios, **kwargs):
        calls.append(("evaluate", ratios))
        return {"1000.0": {"best_objective": None, "attempts": []}}

    monkeypatch.setattr(curriculum, "_iteration", iteration)
    monkeypatch.setattr(curriculum, "_evaluate", evaluate)
    monkeypatch.setattr(
        curriculum,
        "_native_witnesses",
        lambda *args: {"1000.0": None},
    )
    curriculum._sv2_phase_operation(
        scientist,
        "native",
        {
            "selected": selected,
            "ratios": (1000.0,),
            "training_ratios": (10.0, 1000.0),
            "f_native": 1,
            "simulations": 64,
            "selfplay_games": 8,
            "train_steps": 1,
            "batch_size": 1,
            "evaluation_attempts": 4,
            "seed": 1,
            "static_index": 0,
        },
    )
    assert calls == [
        ("train", (10.0, 1000.0)),
        ("evaluate", (1000.0,)),
    ]


def test_native_timeout_is_unsolved_and_retained_in_denominators() -> None:
    result = curriculum._phase_timeout_result(
        "raster-invariant-combined-dual-12",
        "native",
        {
            "ratios": (10.0, 1000.0),
            "evaluation_attempts": 4,
        },
        timeout_seconds=3600,
    )
    event = result["scientist_event"]
    assert event["hard_timeout"] == {
        "phase": "native",
        "seconds": 3600.0,
        "retained_in_denominators": True,
        "state_advanced": False,
    }
    assert result["native_witnesses"] == {"10.0": None, "1000.0": None}
    assert all(
        len(cell["attempts"]) == 4
        and not any(row["solved"] for row in cell["attempts"])
        and all(row["hard_timeout"] for row in cell["attempts"])
        for cell in event["evaluation"].values()
    )


def test_rehearsal_timeout_uses_declared_failure_caps_and_holds_controller() -> None:
    items = [
        BankItem(
            id="k1",
            knot=KnotItem("k1", 3, (1, -1, 1), 2),
            cheap_score=1.0,
            difficulty_quartile=0,
        ),
        BankItem(
            id="k2",
            knot=KnotItem("k2", 3, (1, 1, -1), 2),
            cheap_score=2.0,
            difficulty_quartile=0,
        ),
    ]
    result = curriculum._phase_timeout_result(
        "raster-invariant-combined-dual-12",
        "rehearse",
        {
            "processed_items": items,
            "ratios": (10.0, 1000.0),
            "action_horizon": 128,
            "f_old": 1,
            "rehearsal_exposure": {"k1": 1},
            "retention_target": 0.8,
        },
        timeout_seconds=3600,
    )
    after = result["retention_after"]
    assert after["attempts"] == 4
    assert after["solved"] == 0
    assert after["capped_cost"] == 2 * ((10 * 20 + 128) + (1000 * 20 + 128))
    assert result["next_F_old"] == 1
    assert result["event"]["controller_update"] == "held-censored-timeout"
    assert result["event"]["hard_timeout"]["controller_update"] == ("held-censored-timeout")
    assert result["rehearsal_exposure"] == {"k1": 1}


def test_rehearsal_resumes_from_single_atomic_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    items = [
        BankItem(
            id="k1",
            knot=KnotItem("k1", 3, (1, -1, 1), 2),
            cheap_score=1.0,
            difficulty_quartile=0,
        ),
        BankItem(
            id="k2",
            knot=KnotItem("k2", 3, (1, 1, -1), 2),
            cheap_score=2.0,
            difficulty_quartile=0,
        ),
    ]
    checkpoint_path = tmp_path / "rehearsal.pt.gz"
    before = {"solve_rate": 1.0, "capped_cost": 10.0}
    curriculum._save_state(
        checkpoint_path,
        {
            "schema": "semantic-v2-rehearsal-checkpoint-v1",
            "scientist": "scientist",
            "round_index": 4,
            "checkpointed_at_unix": 1.0,
            "selected_order": ["k1", "k2"],
            "selected": ["k1"],
            "iterations": [{"representation": "k1", "train_steps": 24}],
            "completed_iterations": 1,
            "before": before,
            "rehearsal_exposure": {"k1": 1},
            "scientist_state": {"version": 1},
        },
    )

    restored = []
    trained = []
    monkeypatch.setattr(
        curriculum,
        "_restore_scientist",
        lambda scientist, state: restored.append(state),
    )
    monkeypatch.setattr(curriculum, "_scientist_state", lambda scientist: {"version": 2})
    monkeypatch.setattr(
        curriculum,
        "_iteration",
        lambda _scientist, knot, **_kwargs: trained.append(knot.name) or {"train_steps": 24},
    )
    monkeypatch.setattr(
        curriculum,
        "_retention_summary_resumable",
        lambda *args, **kwargs: {
            "solve_rate": 1.0,
            "capped_cost": 9.0,
            "cells": {},
        },
    )

    result = curriculum._sv2_phase_operation(
        SimpleNamespace(name="scientist"),
        "rehearse",
        {
            "scientist": "scientist",
            "processed_items": items,
            "ratios": (10.0, 1000.0),
            "training_ratios": (10.0, 1000.0),
            "identity_indices": {"k1": 0, "k2": 1},
            "seed": 1,
            "round_index": 4,
            "simulations": 40,
            "f_old": 2,
            "rehearsal_exposure": {},
            "selfplay_games": 4,
            "train_steps": 24,
            "batch_size": 64,
            "retention_target": 0.8,
            "action_horizon": 128,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_interval_seconds": 600.0,
        },
    )

    assert restored == [{"version": 1}]
    assert trained == ["k2"]
    assert [row["representation"] for row in result["event"]["iterations"]] == [
        "k1",
        "k2",
    ]
    checkpoint = curriculum._load_state(checkpoint_path)
    assert checkpoint["schema"] == "semantic-v2-rehearsal-checkpoint-v2"
    assert checkpoint["phase"] == "complete"
    assert checkpoint["completed_iterations"] == 2
    assert checkpoint["selected"] == ["k1", "k2"]
    assert checkpoint["rehearsal_exposure"] == {"k1": 1, "k2": 1}
    assert checkpoint["scientist_state"] == {"version": 2}
    assert list(tmp_path.glob("rehearsal.pt.gz*")) == [checkpoint_path]


def test_rehearsal_timeout_installs_latest_checkpoint_in_coordinator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    item = BankItem(
        id="k1",
        knot=KnotItem("k1", 3, (1, -1, 1), 2),
        cheap_score=1.0,
        difficulty_quartile=0,
    )
    checkpoint_path = tmp_path / "rehearsal.pt.gz"
    partial_state = {"version": 7}
    curriculum._save_state(
        checkpoint_path,
        {
            "schema": "semantic-v2-rehearsal-checkpoint-v1",
            "scientist": "scientist",
            "round_index": 3,
            "checkpointed_at_unix": 1.0,
            "selected_order": ["k1"],
            "selected": ["k1"],
            "iterations": [{"representation": "k1", "train_steps": 24}],
            "completed_iterations": 1,
            "before": {"solve_rate": 0.0, "capped_cost": 328.0},
            "rehearsal_exposure": {"k1": 1},
            "scientist_state": partial_state,
        },
    )
    payload = {
        "scientist": "scientist",
        "processed_items": [item],
        "ratios": (10.0,),
        "action_horizon": 128,
        "f_old": 1,
        "rehearsal_exposure": {},
        "retention_target": 0.8,
        "round_index": 3,
        "checkpoint_path": str(checkpoint_path),
    }
    coordinator = object.__new__(curriculum._ScientistPhaseCoordinator)
    coordinator.parallel = True
    coordinator.names = ["scientist"]
    coordinator.states = {"scientist": {"version": 0}}
    coordinator._restore_next = {"scientist": False}
    monkeypatch.setattr(coordinator, "submit", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        coordinator,
        "collect",
        lambda *args, **kwargs: (_ for _ in ()).throw(curriculum.FutureTimeoutError()),
    )
    monkeypatch.setattr(coordinator, "_reset_executor", lambda name: None)

    result = coordinator.run(
        "rehearse",
        {"scientist": payload},
        timeout_seconds=3600,
    )["scientist"]

    assert coordinator.states["scientist"] == partial_state
    assert coordinator._restore_next["scientist"] is True
    assert result["event"]["hard_timeout"] == {
        "phase": "rehearse",
        "seconds": 3600.0,
        "retained_in_denominators": True,
        "state_advanced": True,
        "checkpoint_recovered": True,
        "completed_rehearsal_iterations": 1,
        "checkpoint_stage": "train",
        "completed_retention_before_cells": 0,
        "completed_retention_after_cells": 0,
        "controller_update": "held-censored-timeout",
    }
    assert result["next_F_old"] == 1
    assert result["event"]["iterations"] == [{"representation": "k1", "train_steps": 24}]
    assert result["rehearsal_exposure"] == {"k1": 1}


def test_rehearsal_cumulative_timeout_scales_with_history_and_compute() -> None:
    assert (
        rehearsal_cumulative_timeout_seconds(
            3600,
            processed_items=104,
            ratios=2,
            simulations=80,
            f_old=8,
        )
        == 8 * 3600
    )
    assert (
        rehearsal_cumulative_timeout_seconds(
            3600,
            processed_items=154,
            ratios=2,
            simulations=80,
            f_old=8,
        )
        == 11 * 3600
    )
    assert (
        rehearsal_cumulative_timeout_seconds(
            3600,
            processed_items=154,
            ratios=2,
            simulations=128,
            f_old=8,
        )
        == 17 * 3600
    )
    assert (
        rehearsal_cumulative_timeout_seconds(
            7200,
            processed_items=20,
            ratios=2,
            simulations=80,
            f_old=1,
            training_seconds_per_iteration_at_reference=7200,
        )
        == 2 * 7200
    )


def test_legacy_resume_protocol_accepts_only_neutral_default_spellings() -> None:
    current = {
        "arm": "scheduled-no-sharing",
        "sharing": "none",
        "resumable_rehearsal_segments": False,
        "strict_own_budget_rehearsal": False,
        "rehearsal_budget_policy": "global",
        "terminal_full_retention_audit": False,
        "simulations": 40,
        "protocol_sha256": "current",
    }
    frozen = {
        "arm": "scheduled-no-sharing",
        "sharing": False,
        "resumable_rehearsal_segments": None,
        "strict_own_budget_rehearsal": None,
        "rehearsal_budget_policy": None,
        "terminal_full_retention_audit": None,
        "simulations": 40,
        "protocol_sha256": "frozen",
    }
    assert curriculum._legacy_resume_protocol_is_equivalent(frozen, current)
    frozen["simulations"] = 80
    assert not curriculum._legacy_resume_protocol_is_equivalent(frozen, current)


def test_verified_timeout_resume_accepts_only_gated_wall_time_increase(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    frozen = {
        "protocol_sha256": "frozen-hash",
        "scientist_task_timeout_seconds": 7200,
        "rehearsal_segment_timeout_seconds": 7200,
        "seed": 17,
    }
    current = {
        "protocol_sha256": "current-hash",
        "scientist_task_timeout_seconds": 21600,
        "rehearsal_segment_timeout_seconds": 21600,
        "seed": 17,
    }
    transition = tmp_path / "transition.json"
    transition.write_text(
        json.dumps(
            {
                "schema": "semantic-v2-timeout-extension-v1",
                "passed": True,
                "output": str(output),
                "frozen_protocol_sha256": "frozen-hash",
                "old_timeout_seconds": 7200,
                "new_timeout_seconds": 21600,
                "allowed_protocol_fields": [
                    "scientist_task_timeout_seconds",
                    "rehearsal_segment_timeout_seconds",
                ],
            }
        )
    )
    assert curriculum._verified_timeout_resume_protocol_is_equivalent(
        frozen, current, transition, output
    )
    current["seed"] = 18
    assert not curriculum._verified_timeout_resume_protocol_is_equivalent(
        frozen, current, transition, output
    )


def test_completed_rehearsal_repair_report_prevents_replaying_debt(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    debt = {"scientist": 38}
    report.write_text(
        json.dumps(
            {
                "schema": "semantic-v2-q104-rehearsal-repair-report-v1",
                "source_debt": debt,
                "completed_iterations": 38,
            }
        )
    )
    assert curriculum._completed_rehearsal_repair_matches(report, debt)
    assert not curriculum._completed_rehearsal_repair_matches(report, {"scientist": 39})


def test_rehearsal_segment_timeout_resumes_same_phase_until_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    item = BankItem(
        id="k1",
        knot=KnotItem("k1", 3, (1, -1, 1), 2),
        cheap_score=1.0,
        difficulty_quartile=0,
    )
    checkpoint_path = tmp_path / "rehearsal.pt.gz"
    partial_state = {"version": 7}
    curriculum._save_state(
        checkpoint_path,
        {
            "schema": "semantic-v2-rehearsal-checkpoint-v1",
            "scientist": "scientist",
            "round_index": 3,
            "checkpointed_at_unix": 1.0,
            "selected_order": ["k1"],
            "selected": ["k1"],
            "iterations": [{"representation": "k1", "train_steps": 24}],
            "completed_iterations": 1,
            "before": {"solve_rate": 0.0, "capped_cost": 328.0},
            "rehearsal_exposure": {"k1": 1},
            "scientist_state": partial_state,
        },
    )
    payload = {
        "scientist": "scientist",
        "processed_items": [item],
        "ratios": (10.0,),
        "action_horizon": 128,
        "f_old": 1,
        "rehearsal_exposure": {},
        "retention_target": 0.8,
        "round_index": 3,
        "checkpoint_path": str(checkpoint_path),
    }
    completed = {
        "next_F_old": 1,
        "rehearsal_exposure": {"k1": 1},
        "retention_after": {"attempts": 1},
        "event": {"F_old": 1, "next_F_old": 1},
    }
    coordinator = object.__new__(curriculum._ScientistPhaseCoordinator)
    coordinator.parallel = True
    coordinator.names = ["scientist"]
    coordinator.states = {"scientist": {"version": 0}}
    coordinator._restore_next = {"scientist": False}
    submitted = []
    monkeypatch.setattr(
        coordinator,
        "submit",
        lambda *args, **kwargs: submitted.append((args, kwargs)) or object(),
    )
    collects = iter([curriculum.FutureTimeoutError(), completed])

    def collect(*_args, **_kwargs):
        value = next(collects)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(coordinator, "collect", collect)
    monkeypatch.setattr(coordinator, "_reset_executor", lambda name: None)

    result = coordinator.run(
        "rehearse",
        {"scientist": payload},
        timeout_seconds=3600,
        cumulative_timeout_seconds={"scientist": 10800},
    )["scientist"]

    assert len(submitted) == 2
    assert coordinator.states["scientist"] == partial_state
    assert result["event"]["rehearsal_segments"] == {
        "segment_timeout_seconds": 3600.0,
        "cumulative_timeout_seconds": 10800.0,
        "segment_expirations": 1,
        "checkpoint_resumes": 1,
        "completed": True,
    }
    assert not checkpoint_path.exists()


def test_rehearsal_segments_censor_only_after_cumulative_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    item = BankItem(
        id="k1",
        knot=KnotItem("k1", 3, (1, -1, 1), 2),
        cheap_score=1.0,
        difficulty_quartile=0,
    )
    checkpoint_path = tmp_path / "rehearsal.pt.gz"
    curriculum._save_state(
        checkpoint_path,
        {
            "schema": "semantic-v2-rehearsal-checkpoint-v1",
            "scientist": "scientist",
            "round_index": 3,
            "checkpointed_at_unix": 1.0,
            "selected_order": ["k1"],
            "selected": [],
            "iterations": [],
            "completed_iterations": 0,
            "before": None,
            "rehearsal_exposure": {},
            "scientist_state": {"version": 7},
        },
    )
    payload = {
        "scientist": "scientist",
        "processed_items": [item],
        "ratios": (10.0,),
        "action_horizon": 128,
        "f_old": 1,
        "rehearsal_exposure": {},
        "retention_target": 0.8,
        "round_index": 3,
        "checkpoint_path": str(checkpoint_path),
    }
    coordinator = object.__new__(curriculum._ScientistPhaseCoordinator)
    coordinator.parallel = True
    coordinator.names = ["scientist"]
    coordinator.states = {"scientist": {"version": 0}}
    coordinator._restore_next = {"scientist": False}
    monkeypatch.setattr(coordinator, "submit", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        coordinator,
        "collect",
        lambda *args, **kwargs: (_ for _ in ()).throw(curriculum.FutureTimeoutError()),
    )
    monkeypatch.setattr(coordinator, "_reset_executor", lambda name: None)

    result = coordinator.run(
        "rehearse",
        {"scientist": payload},
        timeout_seconds=3600,
        cumulative_timeout_seconds={"scientist": 7200},
    )["scientist"]

    timeout = result["event"]["hard_timeout"]
    assert timeout["seconds"] == 7200.0
    assert timeout["segment_timeout_seconds"] == 3600.0
    assert timeout["segment_expirations"] == 2
    assert timeout["controller_update"] == "held-censored-timeout"
    assert result["next_F_old"] == 1


def test_rehearsal_segment_requires_atomic_checkpoint(monkeypatch) -> None:
    coordinator = object.__new__(curriculum._ScientistPhaseCoordinator)
    coordinator.parallel = True
    coordinator.names = ["scientist"]
    coordinator.states = {"scientist": {"version": 0}}
    coordinator._restore_next = {"scientist": False}
    monkeypatch.setattr(coordinator, "submit", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        coordinator,
        "collect",
        lambda *args, **kwargs: (_ for _ in ()).throw(curriculum.FutureTimeoutError()),
    )
    monkeypatch.setattr(coordinator, "_reset_executor", lambda name: None)

    with pytest.raises(RuntimeError, match="without an atomic checkpoint"):
        coordinator.run(
            "rehearse",
            {
                "scientist": {
                    "scientist": "scientist",
                    "round_index": 3,
                    "checkpoint_path": None,
                }
            },
            timeout_seconds=3600,
            cumulative_timeout_seconds={"scientist": 7200},
        )


def test_resumable_retention_skips_completed_cells(monkeypatch) -> None:
    items = [
        BankItem(
            id="k1",
            knot=KnotItem("k1", 3, (1, -1, 1), 2),
            cheap_score=1.0,
            difficulty_quartile=0,
        ),
        BankItem(
            id="k2",
            knot=KnotItem("k2", 3, (1, 1, -1), 2),
            cheap_score=2.0,
            difficulty_quartile=0,
        ),
    ]
    completed = {
        "k1": {
            "10.0": {
                "solve_rate": 0.0,
                "best_objective": None,
                "best_witness": None,
                "attempts": [],
            }
        }
    }
    calls = []

    def evaluate(_scientist, knots, ratio, simulations, seeds, **_kwargs):
        calls.append((knots[0].name, ratio, simulations, seeds[0]))
        return [(None, {"scheduled_network_evaluations": simulations})]

    monkeypatch.setattr(curriculum, "_evaluation_tasks", evaluate)
    snapshots = []
    scientist = SimpleNamespace(config=SimpleNamespace(game=SimpleNamespace(simplify_budget=128)))
    result = curriculum._retention_summary_resumable(
        scientist,
        items,
        ratios=(10.0, 1000.0),
        simulations=40,
        seed=7,
        identity_indices={"k1": 3, "k2": 4},
        partial_cells=completed,
        progress=lambda cells: snapshots.append(curriculum._retention_cell_count(cells)),
    )

    assert [(name, ratio) for name, ratio, _simulations, _seed in calls] == [
        ("k2", 10.0),
        ("k1", 1000.0),
        ("k2", 1000.0),
    ]
    assert snapshots == [2, 3, 4]
    assert result["attempts"] == 4
    assert curriculum._retention_cell_count(result["cells"]) == 4


def test_rehearsal_timeout_retains_completed_retention_cells() -> None:
    item = BankItem(
        id="k1",
        knot=KnotItem("k1", 3, (1, -1, 1), 2),
        cheap_score=1.0,
        difficulty_quartile=0,
    )
    solved_cell = {
        "solve_rate": 1.0,
        "best_objective": 2.0,
        "best_witness": {"objective": 2.0},
        "attempts": [{"attempt": 0, "solved": True}],
    }
    result = curriculum._phase_timeout_result(
        "scientist",
        "rehearse",
        {
            "processed_items": [item],
            "ratios": (10.0, 1000.0),
            "action_horizon": 128,
            "f_old": 4,
            "rehearsal_exposure": {},
            "retention_target": 0.8,
        },
        timeout_seconds=3600,
        partial_checkpoint={
            "phase": "retention_after",
            "iterations": [],
            "rehearsal_exposure": {},
            "retention_before_cells": {},
            "retention_after_cells": {"k1": {"10.0": solved_cell}},
            "before": None,
        },
    )

    after = result["retention_after"]
    assert after["solved"] == 1
    assert after["cells"]["k1"]["10.0"] == solved_cell
    assert after["cells"]["k1"]["1000.0"]["attempts"][0]["hard_timeout"] is True
    assert result["next_F_old"] == 4


def test_iteration_reports_games_by_training_objective(monkeypatch) -> None:
    class Replay:
        games = [object()]

        def set_representation_embedding(self, *args) -> None:
            pass

        def add(self, *args, **kwargs) -> None:
            pass

    class Record(list):
        pass

    scientist = SimpleNamespace(
        replay=Replay(),
        game=object(),
        network=object(),
        optimizer=object(),
        config=SimpleNamespace(search=object(), train=SimpleNamespace(device="cpu")),
    )
    monkeypatch.setattr(curriculum, "replace", lambda value, **kwargs: value)
    monkeypatch.setattr(curriculum, "FixedWordGame", lambda *args: object())
    monkeypatch.setattr(curriculum, "NeuralMCTS", lambda *args: object())
    monkeypatch.setattr(
        curriculum,
        "play_selfplay_games",
        lambda *args: [Record() for _ in args[3]],
    )
    monkeypatch.setattr(curriculum, "train_alphazero_step", lambda *args, **kwargs: {})
    result = curriculum._iteration(
        scientist,
        KnotItem("x", 3, (1, -1, 1), 2),
        ratios=(10.0, 1000.0),
        simulations=2,
        selfplay_games=8,
        train_steps=1,
        batch_size=1,
        seed=1,
    )
    assert result["selfplay_games_by_ratio"] == {"10.0": 4, "1000.0": 4}


def test_strict_rehearsal_uses_only_lineage_local_caps(monkeypatch) -> None:
    caps = []

    class Replay:
        games = [object()]
        last_collaboration_sample_trace = []

        def set_representation_embedding(self, *args) -> None:
            pass

        def best_native_objective(self, representation: str, ratio: float) -> float:
            assert representation == "exact-id"
            return ratio + 3.0

        def add(self, *args, **kwargs) -> None:
            pass

    scientist = SimpleNamespace(
        replay=Replay(),
        game=object(),
        network=object(),
        optimizer=object(),
        config=SimpleNamespace(search=object(), train=SimpleNamespace(device="cpu")),
    )
    monkeypatch.setattr(curriculum, "replace", lambda value, **kwargs: value)

    def fixed(*args, **kwargs):
        caps.append((args[2], kwargs["objective_cap"], kwargs["cap_type"]))
        return object()

    monkeypatch.setattr(curriculum, "FixedWordGame", fixed)
    monkeypatch.setattr(curriculum, "NeuralMCTS", lambda *args: object())
    monkeypatch.setattr(curriculum, "play_selfplay_games", lambda *args: [[]])
    monkeypatch.setattr(curriculum, "train_alphazero_step", lambda *args, **kwargs: {})

    result = curriculum._iteration(
        scientist,
        KnotItem("coarse-knot", 3, (1, -1, 1), 2),
        representation_id="exact-id",
        ratios=(10.0, 1000.0),
        simulations=2,
        selfplay_games=4,
        train_steps=1,
        batch_size=1,
        seed=1,
        use_own_budget_caps=True,
    )

    assert caps == [
        (10.0, 13.0, "own"),
        (10.0, 13.0, "own"),
        (1000.0, 1003.0, "own"),
        (1000.0, 1003.0, "own"),
    ]
    assert result["selfplay_games_by_budget_source"] == {"own": 4}


def test_iteration_resumes_after_safe_game_and_optimizer_boundaries(monkeypatch) -> None:
    class Replay:
        games = [object()]

        def set_representation_embedding(self, *args) -> None:
            pass

        def add(self, *args, **kwargs) -> None:
            pass

    scientist = SimpleNamespace(
        replay=Replay(),
        game=object(),
        network=object(),
        optimizer=object(),
        config=SimpleNamespace(search=object(), train=SimpleNamespace(device="cpu")),
    )
    played = []
    optimized = []
    monkeypatch.setattr(curriculum, "replace", lambda value, **kwargs: value)
    monkeypatch.setattr(curriculum, "FixedWordGame", lambda *args, **kwargs: object())
    monkeypatch.setattr(curriculum, "NeuralMCTS", lambda *args: object())

    def play(*args):
        played.append(len(played))
        return [[]]

    def optimize(*args, **kwargs):
        optimized.append(len(optimized))
        return {"step": len(optimized)}

    monkeypatch.setattr(curriculum, "play_selfplay_games", play)
    monkeypatch.setattr(curriculum, "train_alphazero_step", optimize)
    saved = None

    class SegmentExpired(Exception):
        pass

    def interrupt_after_two_games(row):
        nonlocal saved
        saved = row
        if len(row["completed_games"]) == 2:
            raise SegmentExpired

    kwargs = dict(
        ratios=(10.0, 1000.0),
        simulations=2,
        selfplay_games=4,
        train_steps=6,
        batch_size=1,
        seed=1,
    )
    with pytest.raises(SegmentExpired):
        curriculum._iteration(
            scientist,
            KnotItem("x", 3, (1, -1, 1), 2),
            progress=interrupt_after_two_games,
            **kwargs,
        )
    assert saved is not None
    assert len(saved["completed_games"]) == 2

    def interrupt_after_three_steps(row):
        nonlocal saved
        saved = row
        if row["completed_optimizer_steps"] == 3:
            raise SegmentExpired

    with pytest.raises(SegmentExpired):
        curriculum._iteration(
            scientist,
            KnotItem("x", 3, (1, -1, 1), 2),
            resume_progress=saved,
            progress=interrupt_after_three_steps,
            **kwargs,
        )
    assert len(played) == 4
    assert len(optimized) == 3

    result = curriculum._iteration(
        scientist,
        KnotItem("x", 3, (1, -1, 1), 2),
        resume_progress=saved,
        **kwargs,
    )
    assert len(played) == 4
    assert len(optimized) == 6
    assert result["selfplay_games"] == 4
    assert result["train_steps"] == 6
    assert result["last_loss"] == {"step": 6}


def test_pipelined_native_block_advances_fast_scientist_and_resumes(
    tmp_path: Path,
) -> None:
    timeline = []

    class Coordinator:
        names = ["fast", "slow"]
        parallel = True

        def __init__(self) -> None:
            self.pool = ThreadPoolExecutor(max_workers=2)
            self.states = {name: {"step": -1} for name in self.names}
            self.submitted = []

        def submit(self, name, operation, payload):
            assert operation == "native"
            round_index = payload["round"]
            self.submitted.append((name, round_index))

            def work():
                timeline.append(("start", name, round_index, time.monotonic()))
                time.sleep(0.01 if name == "fast" else 0.08)
                timeline.append(("finish", name, round_index, time.monotonic()))
                return {"name": name, "round": round_index}

            return self.pool.submit(work)

        def collect(self, name, future):
            row = future.result()
            self.states[name] = {"step": row["round"]}
            return {
                "scientist_event": {
                    "iterations": [],
                    "evaluation": {},
                    "native_best": {},
                    "rehearsal": None,
                },
                "native_witnesses": {},
            }

        def restore_full_state(self, name, state):
            self.states[name] = state

    payloads = {
        round_index: {name: {"round": round_index} for name in ("fast", "slow")}
        for round_index in range(3)
    }
    selected = {round_index: f"k{round_index}" for round_index in range(3)}
    coordinator = Coordinator()
    first = curriculum._run_pipelined_native_block(
        coordinator,
        output=tmp_path,
        protocol_sha256="a" * 64,
        start_round=0,
        selected_by_round=selected,
        payloads_by_round=payloads,
    )
    coordinator.pool.shutdown()
    fast_round_one_started = next(row[3] for row in timeline if row[:3] == ("start", "fast", 1))
    slow_round_zero_finished = next(row[3] for row in timeline if row[:3] == ("finish", "slow", 0))
    assert fast_round_one_started < slow_round_zero_finished
    assert all(set(first[index]) == {"fast", "slow"} for index in range(3))

    resumed = Coordinator()
    second = curriculum._run_pipelined_native_block(
        resumed,
        output=tmp_path,
        protocol_sha256="a" * 64,
        start_round=0,
        selected_by_round=selected,
        payloads_by_round=payloads,
    )
    resumed.pool.shutdown()
    assert resumed.submitted == []
    assert resumed.states == {"fast": {"step": 2}, "slow": {"step": 2}}
    assert second == first


def test_pipelined_execution_rejects_adaptive_or_sharing_arms(tmp_path: Path) -> None:
    import pytest

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    bank = tmp_path / "bank.json"
    bank.write_text("[]")
    for arm in ("adaptive-no-sharing", "static-sharing"):
        with pytest.raises(ValueError, match="fixed-order no-sharing"):
            curriculum.run_coordinated_arm(
                {"s": checkpoint},
                bank,
                tmp_path / arm,
                arm=arm,
                pipelined_static_no_sharing=True,
            )


def test_trajectory_tournament_requires_exact_scheduled_configuration(tmp_path: Path) -> None:
    import pytest

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    bank = tmp_path / "bank.json"
    bank.write_text("[]")
    with pytest.raises(ValueError, match="must equal"):
        curriculum.run_coordinated_arm(
            {"s": checkpoint},
            bank,
            tmp_path / "wrong-size",
            arm="scheduled-no-sharing",
            trajectory_tournament_size=10,
            selfplay_games=8,
        )
    with pytest.raises(ValueError, match="scheduled-no-sharing"):
        curriculum.run_coordinated_arm(
            {"s": checkpoint},
            bank,
            tmp_path / "wrong-arm",
            arm="static-no-sharing",
            trajectory_tournament_size=10,
            selfplay_games=10,
        )
    with pytest.raises(ValueError, match="requires a trajectory tournament"):
        curriculum.run_coordinated_arm(
            {"s": checkpoint},
            bank,
            tmp_path / "weight-only",
            arm="scheduled-no-sharing",
            relative_trajectory_weight=1.0,
        )


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


def test_curriculum_skip_is_bounded_and_retained_in_denominators() -> None:
    failed = {"10.0": {"best_objective": None}, "1000.0": {"best_objective": None}}
    solved = {"10.0": {"best_objective": 12.0}, "1000.0": {"best_objective": None}}
    assert curriculum_skip_event(failed, prior_skips=0, limit=1) == {
        "reason": "budget_exhausted",
        "token": 1,
        "limit": 1,
        "retained_in_denominators": True,
    }
    assert curriculum_skip_event(failed, prior_skips=1, limit=1) is None
    assert curriculum_skip_event(solved, prior_skips=0, limit=1) is None


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
    assert next_compute_dose(10, levels=F_NATIVE_LEVELS, observed_rate=0.80, target=0.80) == 10
    assert next_compute_dose(10, levels=F_NATIVE_LEVELS, observed_rate=0.79, target=0.80) == 12


def test_compute_dose_supports_aligned_q_protocol_levels() -> None:
    assert next_compute_dose(4, levels=(4, 6, 8, 12, 16), observed_rate=0.79, target=0.80) == 6
    assert (
        next_compute_dose(
            40,
            levels=(40, 64, 80, 128, 256),
            observed_rate=0.69,
            target=0.70,
        )
        == 64
    )


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
    assert (
        next_rehearsal_dose(
            6,
            retention_solve_rate=0.0,
            capped_cost_worsened=True,
            repair_chunk=True,
        )
        == 6
    )
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
    assert _coordinated_name("static-random-no-sharing", **common).endswith("RANDOM-NO-SHARING")
    assert _coordinated_name("scheduled-no-sharing", **common).endswith("SCHEDULED-NO-SHARING")
    assert _coordinated_name("static-sharing", **common).endswith("EV4-SHARING")
    assert _coordinated_name("adaptive-sharing", **common).endswith("ADAPTIVE-SHARING")


def test_static_random_order_is_frozen_and_input_order_independent() -> None:
    identities = [f"knot-{index:02d}" for index in range(20)]
    expected = _frozen_static_random_order(identities, 2026081401)
    assert expected == _frozen_static_random_order(list(reversed(identities)), 2026081401)
    assert expected != sorted(identities)
    assert expected != _frozen_static_random_order(identities, 2026081402)
    assert sorted(expected) == sorted(identities)


def test_scheduled_no_sharing_preserves_registered_bank_order() -> None:
    items = [
        BankItem(
            id=item_id,
            knot=KnotItem(item_id, 3, (1, -1, 1), 1),
            cheap_score=score,
            difficulty_quartile=0,
        )
        for item_id, score in (("third", 1.0), ("first", 99.0), ("second", 2.0))
    ]
    observed = _fixed_no_sharing_order(
        list(reversed(items)),
        arm="scheduled-no-sharing",
        bank_order=[item.id for item in items],
        static_random_order=None,
    )
    assert [item.id for item in observed] == ["third", "first", "second"]


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


def test_portfolio_summary_keeps_duplicate_knot_names_as_distinct_representations() -> None:
    knot = KnotItem("11n_142", 11, (1, -2, 1), 3)
    items = [
        BankItem("11n_142::presentation-a", knot, 1.0, 0),
        BankItem("11n_142::presentation-b", knot, 2.0, 0),
    ]
    summaries = {
        "scientist": {
            "cells": {
                "11n_142::presentation-a": {"10.0": {"best_objective": 17.0}},
                "11n_142::presentation-b": {"10.0": {"best_objective": None}},
            }
        }
    }

    result = _portfolio_summary(summaries, items, (10.0,), action_horizon=128)

    assert result["attempts"] == 2
    assert result["solved"] == 1
    assert set(result["cells"]) == {
        "11n_142::presentation-a|10",
        "11n_142::presentation-b|10",
    }


def test_retention_uses_representation_ids_for_cells_and_seeds(monkeypatch) -> None:
    knot = KnotItem("11n_142", 11, (1, -2, 1), 3)
    items = [
        BankItem("11n_142::presentation-a", knot, 1.0, 0),
        BankItem("11n_142::presentation-b", knot, 2.0, 0),
    ]
    observed_seeds = []

    def evaluate(*args, **kwargs):
        observed_seeds.extend(args[4])
        return [(None, {"scheduled_network_evaluations": 0}) for _ in args[1]]

    monkeypatch.setattr(curriculum, "_evaluation_tasks", evaluate)
    scientist = SimpleNamespace(config=SimpleNamespace(game=SimpleNamespace(simplify_budget=128)))

    result = curriculum._retention_summary(
        scientist,
        items,
        ratios=(10.0,),
        simulations=2,
        seed=7,
        identity_indices={
            "11n_142::presentation-a": 3,
            "11n_142::presentation-b": 5,
        },
    )

    assert set(result["cells"]) == {
        "11n_142::presentation-a",
        "11n_142::presentation-b",
    }
    assert observed_seeds == [300_007, 500_007]


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
