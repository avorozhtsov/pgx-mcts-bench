from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "rank_sv2_innovation.py"
    spec = importlib.util.spec_from_file_location("rank_sv2_innovation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(cost: float | None, *, solved_attempts: int = 0, evaluations: int = 100):
    attempts = [
        {"solved": index < solved_attempts, "scheduled_network_evaluations": 10}
        for index in range(4)
    ]
    return {
        "evaluation": {"10.0": {"best_objective": cost, "attempts": attempts}},
        "native_network_evaluations": evaluations,
        "selection_probe_network_evaluations": 0,
        "round": 0,
    }


def test_innovation_rewards_solo_solve_and_portfolio_delta() -> None:
    module = _module()
    observations = {
        "run/alice": {
            "x": _observation(20, solved_attempts=4),
            "y": _observation(40, solved_attempts=3),
        },
        "run/bob": {
            "x": _observation(None),
            "y": _observation(50, solved_attempts=2),
        },
        "run/carol": {
            "x": _observation(None),
            "y": _observation(40, solved_attempts=4),
        },
    }
    report = module.rank_innovation(observations, caps={"10": 100})
    rows = {row["scientist"]: row for row in report["objectives"]["10.0"]["ranking"]}
    alice = rows["run/alice"]
    carol = rows["run/carol"]
    assert alice["sole_solves"] == 1
    assert alice["native_win_credit"] == 1.5
    assert carol["native_win_credit"] == 0.5
    assert alice["innovation_margin"] == 85.0  # 80 solo margin + half of 10.
    assert carol["innovation_margin"] == 5.0
    assert alice["portfolio_marginal_contribution"] == 80.0
    assert carol["portfolio_marginal_contribution"] == 0.0
    assert alice["innovation_score"] == 1.85


def test_intersection_excludes_unpaired_representations() -> None:
    module = _module()
    observations = {
        "run/alice": {"x": _observation(20), "only-alice": _observation(10)},
        "run/bob": {"x": _observation(30)},
    }
    report = module.rank_innovation(observations, caps={"10": 100})
    assert report["representations"] == ["x"]
    assert all(row["representations"] == 1 for row in report["objectives"]["10.0"]["ranking"])


def test_all_scientist_tie_has_no_innovation_margin() -> None:
    module = _module()
    observations = {
        "run/alice": {"x": _observation(20)},
        "run/bob": {"x": _observation(20)},
    }
    rows = module.rank_innovation(observations, caps={"10": 100})["objectives"]["10.0"]["ranking"]
    assert [row["native_win_credit"] for row in rows] == [0.5, 0.5]
    assert all(row["innovation_margin"] == 0 for row in rows)


def test_incumbent_bank_counts_only_strict_record_improvements() -> None:
    module = _module()
    observations = {
        "run/alice": {"x": _observation(20), "y": _observation(40)},
        "run/bob": {"x": _observation(30), "y": _observation(None)},
    }
    report = module.rank_innovation(
        observations,
        caps={"10": 100},
        incumbent_bank={"x": {"10.0": 25}, "y": {"10.0": 40}},
    )
    rows = {row["scientist"]: row for row in report["objectives"]["10.0"]["ranking"]}
    assert rows["run/alice"]["record_improvements"] == 1
    assert rows["run/alice"]["record_improvement_margin"] == 5
    assert rows["run/bob"]["record_improvements"] == 0


def test_loader_exposes_native_commit_until_completed_event_exists(tmp_path: Path) -> None:
    module = _module()
    native = tmp_path / "native-events"
    completed = tmp_path / "events"
    native.mkdir()
    completed.mkdir()
    native_event = {
        "round": 0,
        "selected": "x",
        "scientists": {"alice": _observation(20)},
    }
    (native / "000.json").write_text(json.dumps(native_event))
    assert module._load_events(tmp_path, None) == [native_event]

    completed_event = {**native_event, "phase": "complete"}
    (completed / "000.json").write_text(json.dumps(completed_event))
    assert module._load_events(tmp_path, None) == [completed_event]
