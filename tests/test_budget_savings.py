from pgx_mcts_bench.budget_savings import savings_decision


def test_savings_gate_requires_identical_sets_quality_and_compute() -> None:
    passed = savings_decision(
        full_solved={"a", "b"},
        budgeted_solved={"a", "b"},
        full_objective=50.0,
        budgeted_objective=48.0,
        full_evaluations=1_000,
        budgeted_evaluations=750,
    )
    changed_set = savings_decision(
        full_solved={"a", "b"},
        budgeted_solved={"a"},
        full_objective=50.0,
        budgeted_objective=20.0,
        full_evaluations=1_000,
        budgeted_evaluations=500,
    )
    more_compute = savings_decision(
        full_solved={"a"},
        budgeted_solved={"a"},
        full_objective=20.0,
        budgeted_objective=20.0,
        full_evaluations=1_000,
        budgeted_evaluations=900,
    )

    assert passed["passed"]
    assert not changed_set["passed"]
    assert not changed_set["checks"]["identical_final_solved_set"]
    assert not more_compute["passed"]
    assert not more_compute["checks"]["scheduled_evaluation_savings"]
