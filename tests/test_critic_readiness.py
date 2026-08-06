import json
from pathlib import Path

from pgx_mcts_bench.critic_readiness import build_critic_readiness_report


def test_critic_readiness_accepts_calibrated_unrebalanced_evidence(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation.json"
    calibration = tmp_path / "calibration.json"
    output = tmp_path / "readiness.json"
    validation.write_text(
        json.dumps(
            {
                "scientist": "s-window-128",
                "trained": {
                    "attempts": [
                        {"cap": 4, "solved": False},
                        {"cap": 704, "solved": True},
                    ],
                    "curves": [{"monotone": True, "spread": 0.8}],
                    "promoted_rung": {"solve_rate": 1.0},
                },
                "baseline": {"promoted_rung": {"solve_rate": 1.0}},
                "paired": {
                    "baseline_only": [],
                    "shared_solved": ["knot@704#0"],
                    "trained_only": [],
                },
            }
        )
    )
    calibration.write_text(
        json.dumps(
            {
                "fitted": {
                    "auc": 1.0,
                    "brier": 0.02,
                    "brier_skill": 0.8,
                    "ece_5": 0.01,
                }
            }
        )
    )

    report = build_critic_readiness_report(validation, calibration, output)

    assert report["decision"]["passed"] is True
    assert report["evaluation_distribution"]["positive_fraction"] == 0.5
    assert json.loads(output.read_text()) == report
