from pgx_mcts_bench.budget_head_audit import budget_head_accepted


def test_budget_audit_acceptance_requires_negative_labels_and_sensitivity() -> None:
    assert budget_head_accepted(
        items=20,
        monotone_items=20,
        sensitive_items=16,
        brier=0.2,
        negative_positions=100,
    )
    assert not budget_head_accepted(
        items=20,
        monotone_items=20,
        sensitive_items=0,
        brier=0.2,
        negative_positions=100,
    )
    assert not budget_head_accepted(
        items=20,
        monotone_items=20,
        sensitive_items=20,
        brier=0.2,
        negative_positions=0,
    )
