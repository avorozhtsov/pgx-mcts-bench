from pgx_mcts_bench.policy_update_diagnostic import ARMS, analyze_policy_update_rows


def _row(before: int, after: int, *, steps: int, selfplay: int = 0) -> dict:
    def probe(solved: int) -> dict:
        return {
            "games": 2,
            "solved": solved,
            "rows": [
                {"seed": 1, "solved": bool(solved), "objective": 20.0 if solved else None}
            ],
        }

    return {
        "diagnostic_evaluation": {"before": probe(before), "after": probe(after)},
        "iterations": [
            {
                "kind": "current",
                "selfplay_solved": selfplay,
                "train_steps_applied": steps,
                "train_skipped_no_success": steps == 0,
                "last_loss": (
                    {"preservation_policy": 0.1} if steps else None
                ),
            }
        ],
    }


def test_policy_update_analysis_reports_destructive_legacy_update() -> None:
    rows = {
        arm: [_row(1, 1, steps=0)]
        for arm in ARMS
    }
    rows["legacy-96"] = [_row(1, 0, steps=96)]
    rows["guarded-96"] = [_row(1, 1, steps=96)]
    rows["guarded-24"] = [_row(1, 1, steps=24)]

    report = analyze_policy_update_rows(rows)

    assert report["decision"]["baseline_bitwise_consistent"] is True
    assert report["decision"]["legacy_update_degrades_paired_solve_count"] is True
    assert report["arms"]["legacy-96"]["task_selfplay_solves_by_ordinal"] == [0]
    assert report["decision"]["guarded_lower_dose_noninferior_to_frozen"] is True
