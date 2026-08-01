from __future__ import annotations

import torch

from pgx_mcts_bench.ladder import STAGES, VALUE_EVAL_STAGES
from pgx_mcts_bench.value_eval import auxiliary_statistics, render, summarize_candidate


def test_value_eval_rungs_are_appended_as_one_block() -> None:
    assert tuple(STAGES[-10:]) == VALUE_EVAL_STAGES
    assert VALUE_EVAL_STAGES[0] == ("R(3,22)#0", 0)
    assert VALUE_EVAL_STAGES[-1] == ("R(5,26)#0", 4)


def test_summary_and_render_include_ratio_and_scramble_comparisons() -> None:
    predictions = [
        {
            "source": "R(3,22)#0",
            "scramble": scramble,
            "values": {"1000.0": value, "10.0": value, "0.1": value},
        }
        for scramble, value in ((0, 0.4), (4, 0.1))
    ]
    row = {"candidate": "arm", "highest_stage": 3, "predictions": predictions}
    summary = summarize_candidate(row)

    assert summary["mean_10"] == 0.25
    assert round(summary["scramble_delta_10"], 10) == -0.3
    report = render(
        {
            "instances": [
                {"source": prediction["source"], "scramble": prediction["scramble"]}
                for prediction in predictions
            ],
            "candidates": [row],
        }
    )
    assert "mean v(10:1)" in report
    assert "-0.300" in report


def test_auxiliary_statistics_preserve_memberwise_penalized_cost() -> None:
    auxiliary = (
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[2.0, 4.0, 6.0, 8.0]]),
        torch.tensor([[5.0, 5.0, 5.0, 5.0]]),
    )

    statistics = auxiliary_statistics(auxiliary, failure_crossings=20.0)

    assert statistics["solve_probability"]["mean"] == 0.5
    assert statistics["conditional_crossings"]["mean"] == 5.0
    assert statistics["penalized_crossings"]["members"] == [11.0, 12.0, 13.0, 14.0]
