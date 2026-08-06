from pgx_mcts_bench.readiness_gates import (
    ARCHIVED_FIRST_TEN,
    analyze_rung_parity,
    solution_horizon_contract,
)


def _result(highest: int, iterations: int, solve_rate: float) -> dict:
    return {
        "highest_stage": highest,
        "stages": [
            {
                "stage": 0,
                "source": "unknot",
                "scramble": 2,
                "iterations": iterations,
                "solve_rate": solve_rate,
                "promoted": True,
            }
        ],
    }


def test_solution_horizon_does_not_claim_unbounded_completeness() -> None:
    contract = solution_horizon_contract(64)

    assert contract["learned_objective_cap"] is None
    assert contract["crossing_change_cap"] is None
    assert contract["native_action_horizon"] == 64
    assert contract["completeness_claim"] is False


def test_archived_first_ten_is_not_the_mutable_current_prefix() -> None:
    assert ARCHIVED_FIRST_TEN[3] == ("T(2,3)", 4)
    assert ARCHIVED_FIRST_TEN[6] == ("T(2,7)", 0)


def test_rung_parity_requires_progress_quality_and_iteration_noninferiority() -> None:
    report = analyze_rung_parity(
        _result(0, 10, 0.9),
        0,
    )

    assert report["decision"]["passed"] is True
    assert report["reproduction"]["solve_rate_below_0_8"] == []
