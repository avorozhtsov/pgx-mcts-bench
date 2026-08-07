from pgx_mcts_bench.structural_budget_audit import structural_audit_decision


def portfolio(solved, solve_rate, capped):
    return {
        "solved": sorted(solved),
        "solve_rate": solve_rate,
        "capped_objective": capped,
    }


def test_structural_audit_requires_coverage_and_exact_equivalence() -> None:
    decision = structural_audit_decision(
        direct=portfolio({"a", "b"}, 0.75, 100.0),
        structural=portfolio({"a", "b"}, 0.75, 100.0),
        exact_pair_outcomes=True,
        exact_pair_actions=True,
        minimum_solve_rate=0.70,
        direct_evaluations=1_000,
        structural_evaluations=900,
    )
    assert decision["passed"]
    assert decision["economical"]


def test_structural_audit_rejects_changed_set_even_with_lower_cost() -> None:
    decision = structural_audit_decision(
        direct=portfolio({"a", "b"}, 0.75, 100.0),
        structural=portfolio({"a"}, 0.75, 90.0),
        exact_pair_outcomes=False,
        exact_pair_actions=False,
        minimum_solve_rate=0.70,
        direct_evaluations=1_000,
        structural_evaluations=800,
    )
    assert not decision["passed"]
    assert not decision["checks"]["identical_portfolio_solved_set"]


def test_structural_audit_reports_equivalence_separately_from_economy() -> None:
    decision = structural_audit_decision(
        direct=portfolio({"a"}, 1.0, 10.0),
        structural=portfolio({"a"}, 1.0, 10.0),
        exact_pair_outcomes=True,
        exact_pair_actions=False,
        minimum_solve_rate=0.70,
        direct_evaluations=1_000,
        structural_evaluations=1_200,
    )
    assert decision["passed"]
    assert not decision["economical"]
    assert not decision["diagnostics"]["exact_paired_final_actions"]
