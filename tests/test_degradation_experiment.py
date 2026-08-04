from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pgx_mcts_bench.data import Position
from pgx_mcts_bench.degradation_experiment import (
    _summarize,
    analyze_degradation_experiment,
    matched_batches,
    select_shared_games,
)


def _position(
    tag: int,
    *,
    shared: bool = False,
    crossings: float = 1.0,
    moves: float = 1.0,
) -> Position:
    return Position(
        observation=np.asarray([tag]),
        legal_actions=np.asarray([True]),
        policy=np.asarray([1.0]),
        action=tag,
        player=0,
        final_crossing_changes=crossings,
        final_moves=moves,
        episode_seed=tag,
        shared_witness=shared,
    )


def test_shared_selection_is_deterministic_and_cost_ordered() -> None:
    games = [
        [_position(1)],
        [_position(30, shared=True, crossings=3, moves=0)],
        [_position(12, shared=True, crossings=1, moves=2)],
        [_position(11, shared=True, crossings=1, moves=1)],
    ]
    assert [game[0].action for game in select_shared_games(games, 2, 10.0)] == [
        11,
        12,
    ]


def test_matched_batches_replace_only_tail_and_are_reproducible() -> None:
    native = [[_position(tag)] for tag in range(20)]
    shared = [[_position(100, shared=True), _position(101, shared=True)]]
    control, treatment = matched_batches(
        native + shared,
        shared,
        train_steps=3,
        batch_size=10,
        shared_fraction=0.2,
        seed=7,
    )
    control_again, treatment_again = matched_batches(
        native + shared,
        shared,
        train_steps=3,
        batch_size=10,
        shared_fraction=0.2,
        seed=7,
    )
    for left, right, repeated_left, repeated_right in zip(
        control, treatment, control_again, treatment_again, strict=True
    ):
        assert [p.action for p in left[:8]] == [p.action for p in right[:8]]
        assert all(p.shared_witness for p in right[8:])
        assert [p.action for p in left] == [p.action for p in repeated_left]
        assert [p.action for p in right] == [p.action for p in repeated_right]


def test_summary_exposes_individual_degradation_hidden_by_portfolio() -> None:
    rows = [
        {
            "item": "a",
            "attempts": [
                {
                    "scientist": "one",
                    "ratio": 10.0,
                    "solved": True,
                    "crossing_changes": 1,
                    "moves": 2,
                    "objective": 12.0,
                },
                {"scientist": "two", "ratio": 10.0, "solved": False},
            ],
        }
    ]
    summary = _summarize(rows, ["one", "two"], 10.0, 50)
    assert summary["portfolio"]["capped_objective_sum"] == 12.0
    assert summary["scientists"]["one"]["portfolio_solved"] == 1
    assert summary["scientists"]["two"]["capped_objective_sum"] == 250.0


def test_analysis_rejects_any_treatment_with_a_seed_regression(tmp_path: Path) -> None:
    variants = ("pre", "rl0", "d1-full", "d10-full", "d1-aux", "d10-aux")
    for seed in range(3):
        for variant in variants:
            if variant == "pre" and seed:
                continue
            for split in ("base50", "new70"):
                solved = 2
                loss = 100.0
                if variant.startswith("d") and seed == 2 and split == "base50":
                    solved = 1
                    loss = 200.0
                report = {
                    "summary": {
                        "portfolio": {
                            "portfolio_solved": solved,
                            "capped_objective_sum": loss,
                            "solved_items": ["a", "b"][:solved],
                        },
                        "scientists": {
                            "one": {
                                "portfolio_solved": solved,
                                "capped_objective_sum": loss,
                                "solved_items": ["a", "b"][:solved],
                            }
                        },
                    }
                }
                target = tmp_path / f"seed-{seed}" / variant / split
                target.mkdir(parents=True)
                (target / "report.json").write_text(json.dumps(report))
    result = analyze_degradation_experiment(tmp_path)
    assert not result["treatments_vs_rl0"]["d1-aux"]["accepted"]
    assert result["decision"] == "stop_current_training_and_distillation"
