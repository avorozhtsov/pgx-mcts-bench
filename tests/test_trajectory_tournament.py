import numpy as np
import pytest

from pgx_mcts_bench.data import Position
from pgx_mcts_bench.sv2_curriculum import _iteration
from pgx_mcts_bench.trajectory_tournament import (
    apply_divergence_tournament_advantages,
    apply_tournament_advantages,
    split_trajectory_tournament,
    split_trajectory_tournament_trimmed,
)


def record(*, solved: bool, cc: int = 0, moves: int = 20, residual: int = 10):
    return [
        Position(
            observation=np.zeros((1, 1, 1), dtype=np.float32),
            legal_actions=np.ones(2, dtype=np.float32),
            policy=np.array([0.5, 0.5], dtype=np.float32),
            action=0,
            player=0,
            representation_id="same-root",
            objective_ratio=1000.0,
            solved=float(solved),
            final_crossing_changes=float(cc),
            final_moves=float(moves),
            residual_word_length=residual,
            best_residual_word_length=residual,
        )
    ]


def test_solved_unsolved_is_the_primary_boundary_and_is_zero_sum() -> None:
    records = [record(solved=index < 3, cc=1) for index in range(10)]
    split = split_trajectory_tournament(records)
    assert split is not None
    assert split.boundary == "solved-vs-unsolved"
    assert split.positive_indexes == (0, 1, 2)
    apply_tournament_advantages(records, split)
    assert sum(row[0].relative_trajectory_advantage for row in records) == pytest.approx(0.0)
    assert records[0][0].relative_trajectory_advantage == pytest.approx(1 / 3)
    assert records[-1][0].relative_trajectory_advantage == pytest.approx(-1 / 7)


def test_largest_crossing_change_gap_separates_solved_routes() -> None:
    records = [record(solved=True, cc=value) for value in (1, 1, 1, 2, 2, 5, 5, 5, 5, 5)]
    split = split_trajectory_tournament(records)
    assert split is not None
    assert split.boundary == "crossing-change-gap"
    assert split.positive_indexes == (0, 1, 2, 3, 4)
    assert split.negative_indexes == (5, 6, 7, 8, 9)


def test_small_unique_improvement_uses_low_confidence_best_vs_rest() -> None:
    records = [record(solved=True, cc=1, moves=20 + index) for index in range(10)]
    split = split_trajectory_tournament(records)
    assert split is not None
    assert split.boundary == "unique-best-vs-rest"
    assert split.confidence == 0.25
    apply_tournament_advantages(records, split)
    assert records[0][0].relative_trajectory_advantage == 0.25
    assert records[1][0].relative_trajectory_advantage == pytest.approx(-0.25 / 9)


def test_invalid_trajectory_is_ignored_not_negative() -> None:
    records = [record(solved=index == 0) for index in range(10)]
    records[-1][0].trajectory_replay_valid = False
    split = split_trajectory_tournament(records)
    assert split is not None
    assert split.ignored_indexes == (9,)
    apply_tournament_advantages(records, split)
    assert records[-1][0].relative_trajectory_advantage == 0.0


def test_trimmed_split_uses_extremes_and_ignores_ambiguous_middle() -> None:
    records = [record(solved=True, cc=value) for value in (1, 1, 2, 2, 3, 3, 5, 5, 6, 6)]
    split = split_trajectory_tournament_trimmed(records)
    assert split is not None
    assert split.boundary == "trimmed-crossing-change-margin"
    assert split.positive_indexes == (0, 1, 2)
    assert split.negative_indexes == (7, 8, 9)
    assert split.ignored_indexes == (3, 4, 5, 6)


def test_divergence_credit_does_not_label_post_divergence_states() -> None:
    records = [record(solved=index < 3, cc=1) for index in range(10)]
    for index, game in enumerate(records):
        game[0].action = 0 if index < 3 else 1
        tail = record(solved=index < 3, cc=1)[0]
        tail.observation = np.full((1, 1, 1), index + 1, dtype=np.float32)
        tail.action = index % 2
        game.append(tail)
    split = split_trajectory_tournament_trimmed(records)
    assert split is not None
    annotated = apply_divergence_tournament_advantages(records, split)
    assert annotated == 6
    assert all(records[index][0].relative_trajectory_advantage > 0 for index in range(3))
    assert all(records[index][0].relative_trajectory_advantage < 0 for index in range(7, 10))
    assert all(game[1].relative_trajectory_advantage == 0.0 for game in records)


def test_tournament_rejects_mixed_roots() -> None:
    records = [record(solved=False) for _ in range(10)]
    records[-1][0].representation_id = "other"
    with pytest.raises(ValueError, match="one explicit representation"):
        split_trajectory_tournament(records)


def test_curriculum_iteration_exposes_only_zero_or_ten_tournament_size() -> None:
    with pytest.raises(ValueError, match="zero or ten"):
        _iteration(
            object(),
            object(),
            ratios=(1000.0,),
            simulations=1,
            selfplay_games=3,
            train_steps=1,
            batch_size=1,
            seed=1,
            trajectory_tournament_size=3,
        )
