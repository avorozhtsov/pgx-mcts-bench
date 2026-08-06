import json
from pathlib import Path

from pgx_mcts_bench.rapid_gate import analyze_paired_gate


def _row(item: str, solved: bool, *, before: bool = False, retention: bool = True) -> dict:
    return {
        "item": item,
        "solved": solved,
        "best_objective": 20.0 if solved else None,
        "first_solve": (
            {"phase": "selfplay", "iteration": 0 if before else 1, "before_any_training": before}
            if solved
            else None
        ),
        "old_retention": [{"item": "old", "solved": retention}],
        "compute": {
            "adaptation_scheduled_network_evaluations": 1,
            "final_scheduled_network_evaluations": 1,
            "retention_scheduled_network_evaluations": 1,
            "training_seconds": 0.0,
            "wall_seconds": 1.0,
        },
    }


def test_paired_gate_requires_post_training_rescues_in_two_seeds(tmp_path: Path) -> None:
    for seed_index in range(3):
        for arm in ("frozen-8+1", "train-8+0", "train-8+1"):
            directory = tmp_path / f"seed-{seed_index}" / arm
            directory.mkdir(parents=True)
            rows = [_row("a", True, before=True), _row("b", arm == "train-8+1")]
            for index, row in enumerate(rows):
                (directory / f"{index:04d}.json").write_text(json.dumps(row))

    report = analyze_paired_gate(
        tmp_path,
        seeds=(1, 2, 3),
        ratio=10.0,
        move_budget=64,
    )

    assert report["decision"]["passed"] is True
    assert report["decision"]["seeds_with_post_training_rescue"] == 3
    assert report["by_seed"]["1"]["comparisons"]["frozen-8+1"]["treatment_only"] == ["b"]
