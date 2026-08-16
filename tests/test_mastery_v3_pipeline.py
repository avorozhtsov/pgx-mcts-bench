from __future__ import annotations

from pgx_mcts_bench.mastery_v3_pretrain import _rotations
from pgx_mcts_bench.mastery_v3_screening import _retention, _upper


def test_pretraining_views_are_replayable_cyclic_rotations() -> None:
    word = [1, -2, 3, -1]
    views = _rotations(word, 4, 17)
    assert len(views) == 4
    assert all(sorted(view) == sorted(word) for view in views)
    assert len({tuple(view) for view in views}) == 4


def test_screening_upper_uses_registered_certified_bound() -> None:
    assert _upper({"certified_unknotting_upper_bound": 3}) == 3
    assert _upper({"known_unknotting_number": 2}) == 2


def test_retention_is_fail_closed_when_control_has_no_solved_strand() -> None:
    control = [
        {"representation_id": "a", "seed": 1, "strands": 6, "solved": True},
        {"representation_id": "b", "seed": 1, "strands": 7, "solved": False},
    ]
    candidate = [
        {"representation_id": "a", "seed": 1, "strands": 6, "solved": True},
        {"representation_id": "b", "seed": 1, "strands": 7, "solved": True},
    ]
    overall, by_strand = _retention(control, candidate)
    assert overall == 1.0
    assert by_strand["6"] == 1.0
    assert by_strand["7"] is None
