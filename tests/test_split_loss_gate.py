from pgx_mcts_bench.split_loss_gate import analyze_split_loss_gate


def _row(item: str, solved: bool, *, after_training: bool = True) -> dict:
    return {
        "item": item,
        "solved": solved,
        "best_objective": 20.0 if solved else None,
        "first_solve": (
            {
                "phase": "selfplay",
                "before_any_training": not after_training,
            }
            if solved
            else None
        ),
        "iterations": [
            {
                "kind": "current",
                "selfplay_solved": int(solved),
                "train_steps_without_policy_value_targets": 24,
            }
        ],
    }


def test_split_loss_gate_requires_replicated_rescue_and_no_control_only() -> None:
    rows_by_seed = {}
    for seed in ("1", "2", "3"):
        frozen = [_row("12a_146", True), _row("11n_107", False)]
        split = [_row("12a_146", True), _row("11n_107", True)]
        rows_by_seed[seed] = {"frozen": frozen, "split-success-24": split}

    report = analyze_split_loss_gate(rows_by_seed, ratio=10.0, move_budget=64)

    assert report["decision"]["passed"] is True
    assert report["decision"]["replicated_frozen_never_rescues"] == ["11n_107"]
    assert report["by_seed"]["1"]["control_only"] == []


def test_split_loss_gate_fails_if_treatment_loses_a_frozen_solution() -> None:
    rows_by_seed = {
        "1": {
            "frozen": [_row("12a_146", True), _row("11n_107", False)],
            "split-success-24": [_row("12a_146", False), _row("11n_107", True)],
        }
    }

    report = analyze_split_loss_gate(rows_by_seed, ratio=10.0, move_budget=64)

    assert report["decision"]["passed"] is False
    assert report["by_seed"]["1"]["control_only"] == ["12a_146"]
