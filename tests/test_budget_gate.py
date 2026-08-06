import pytest

from pgx_mcts_bench.budget_gate import calibration, gate_decision, roc_auc


def test_roc_auc_handles_ordering_ties_and_single_class() -> None:
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.5, 0.5], [1, 0]) == 0.5
    assert roc_auc([0.9, 0.8], [1, 1]) is None


def test_calibration_reports_brier_auc_and_bins() -> None:
    report = calibration([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert report["attempts"] == 4
    assert report["positives"] == 2
    assert report["brier"] == pytest.approx(0.025)
    assert report["brier_skill"] == pytest.approx(0.9)
    assert report["auc"] == 1.0
    assert report["ece_5"] == pytest.approx(0.15)


def test_gate_requires_calibration_coverage_and_retention() -> None:
    trained = {
        "attempts": 20,
        "positives": 10,
        "brier": 0.12,
        "brier_skill": 0.52,
        "ece_5": 0.08,
        "auc": 0.82,
    }
    baseline = {
        "attempts": 20,
        "positives": 10,
        "brier": 0.30,
        "brier_skill": -0.2,
        "ece_5": 0.2,
        "auc": 0.5,
    }
    passed = gate_decision(
        items=10,
        monotone=10,
        informative_items=4,
        informative_sensitive=4,
        never_solved_items=6,
        never_solved_low_probability=6,
        trained_calibration=trained,
        baseline_calibration=baseline,
        trained_solves=11,
        baseline_solves=10,
        trained_rung_rate=1.0,
        baseline_rung_rate=1.0,
    )
    failed = gate_decision(
        items=10,
        monotone=10,
        informative_items=4,
        informative_sensitive=3,
        never_solved_items=6,
        never_solved_low_probability=6,
        trained_calibration=trained,
        baseline_calibration=baseline,
        trained_solves=10,
        baseline_solves=10,
        trained_rung_rate=1.0,
        baseline_rung_rate=1.0,
    )

    assert passed["passed"]
    assert not failed["passed"]
    assert not failed["checks"]["all_informative_items_sensitive"]
