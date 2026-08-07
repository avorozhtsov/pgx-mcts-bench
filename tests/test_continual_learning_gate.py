from pgx_mcts_bench.continual_learning_gate import (
    analyze_paired_smoke,
    analyze_portfolio_progress_smoke,
    cycle_certificate,
    empirical_objective_cap,
    frontier_splits,
    portfolio_progress_decision,
    select_rehearsal_identity,
)


def test_frontier_splits_are_disjoint_and_keep_u_diversity() -> None:
    rows = [
        {
            "id": f"knot-{index:02d}",
            "cheap_score": float(index // 5),
            "certified_unknotting_lower_bound": index % 5 + 1,
        }
        for index in range(45)
    ]
    splits = frontier_splits(rows)
    assert {name: len(values) for name, values in splits.items()} == {
        "calibration": 10,
        "new": 20,
        "heldout": 10,
    }
    identities = [row["id"] for values in splits.values() for row in values]
    assert len(identities) == len(set(identities))
    assert len({row["certified_unknotting_lower_bound"] for row in splits["new"]}) == 5


def test_rehearsal_selection_prefers_decline_before_exposure() -> None:
    selected = select_rehearsal_identity(
        ["stable", "declined", "fresh"],
        initial_rates={"stable": 1.0, "declined": 1.0, "fresh": 0.5},
        latest_rates={"stable": 1.0, "declined": 0.25, "fresh": 0.5},
        solve_probabilities={"stable": 0.9, "declined": 0.8, "fresh": 0.1},
        exposures={"stable": 0, "declined": 20, "fresh": 0},
    )
    assert selected == "declined"


def test_cycle_certificate_reports_realized_balance_and_targets() -> None:
    losses = [
        {
            "policy_value_targets": 16,
            "solve_targets": 32,
            "crossing_targets": 16,
            "move_targets": 16,
            "replay_current_success_positions": 8,
            "replay_rehearsal_success_positions": 8,
            "replay_ordinary_failure_positions": 15,
            "replay_budget_censored_failure_positions": 1,
        }
    ]
    certificate = cycle_certificate(losses)
    assert certificate["positive_fraction"] == 0.5
    assert certificate["budget_censored_fraction_of_failures"] == 1 / 16
    assert certificate["policy_value_targets"] == 16


def test_empirical_cap_and_portfolio_progress_use_complete_capped_loss() -> None:
    calibration = {
        "rows": {
            "a": {"attempts": [{"solved": True, "objective": 70.0}]},
            "b": {
                "attempts": [
                    {"solved": False},
                    {"solved": True, "objective": 95.0},
                ]
            },
        }
    }
    assert empirical_objective_cap(calibration) == 95.0

    before = {"solved": ["a"], "capped_objective": 285.0}
    progress = portfolio_progress_decision(
        before, {"solved": ["a", "b"], "capped_objective": 250.0}
    )
    forgetting = portfolio_progress_decision(
        before, {"solved": ["b"], "capped_objective": 290.0}
    )
    assert progress["passed"]
    assert progress["solved_delta"] == 1
    assert not forgetting["passed"]
    assert forgetting["needs_recovery"]
    assert forgetting["lost"] == ["a"]


def _arm(name: str, *, solved: set[str], new_capped: float, heldout_capped: float):
    evaluation = {
        "solved": sorted(solved),
        "representation_solve_rate": len(solved) / 10,
        "capped_objective": new_capped,
    }
    return {
        "arm": name,
        "completed": True,
        "compute": {"allocated": 1000, "scheduled": 900},
        "rounds": [{"cycle_certificate": {"optimizer_updates": 1}}],
        "final": {
            "new": evaluation,
            "heldout": {
                "solved": [],
                "representation_solve_rate": 0.0,
                "capped_objective": heldout_capped,
            },
            "rehearsal": {
                "solved": list(range(8)),
                "representation_solve_rate": 0.8,
                "capped_objective": 10.0,
            },
        },
    }


def test_smoke_analysis_reports_exact_solved_set_differences() -> None:
    continual = _arm(
        "continual",
        solved={"a", "b", "c", "d", "e", "f", "g"},
        new_capped=90,
        heldout_capped=50,
    )
    transactional = _arm(
        "transactional",
        solved={"a", "b", "c", "d", "e", "f", "h"},
        new_capped=100,
        heldout_capped=50,
    )
    decision = analyze_paired_smoke([continual, transactional])
    assert decision["passed"]
    assert decision["continual_only"] == ["g"]
    assert decision["transactional_only"] == ["h"]
    assert decision["new_solved_intersection"] == ["a", "b", "c", "d", "e", "f"]


def _portfolio_arm(
    name: str,
    *,
    solved: set[str],
    capped_l10: float,
    progress_delta: float | None,
) -> dict:
    progress = (
        None
        if progress_delta is None
        else {
            "needs_recovery": False,
            "solved_delta": 0,
            "capped_objective_delta": progress_delta,
        }
    )
    portfolio = {
        "solved": sorted(solved),
        "representations": 10,
        "representation_solve_rate": len(solved) / 10,
        "capped_objective": capped_l10,
    }
    return {
        "arm": name,
        "completed": True,
        "compute": {"allocated": 1000},
        "rounds": [
            {
                "block_progress": progress,
                "restored_block_start": False,
            }
        ],
        "final": {
            "portfolio": portfolio,
            "new": {"representation_solve_rate": 0.7},
            "heldout": {"capped_objective": 50.0},
            "lifetime_solution_bank": {
                "solved": sorted(solved),
                "capped_objective": capped_l10,
            },
        },
    }


def test_portfolio_gate_uses_block_progress_not_exact_control_retention() -> None:
    treatment = _portfolio_arm(
        "block-progress",
        solved={"a", "b", "c", "d", "e", "f", "g"},
        capped_l10=80.0,
        progress_delta=-5.0,
    )
    control = _portfolio_arm(
        "transactional",
        solved={"a", "b", "c", "d", "e", "f", "g", "h"},
        capped_l10=70.0,
        progress_delta=None,
    )
    decision = analyze_portfolio_progress_smoke([treatment, control])
    assert decision["passed"]
    assert decision["control_only"] == ["h"]
    assert decision["secondary_control_comparison"][
        "treatment_solved_minus_control"
    ] == -1
