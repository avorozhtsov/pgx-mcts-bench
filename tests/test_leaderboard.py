from __future__ import annotations

import os
from pathlib import Path

import torch

from pgx_mcts_bench.ladder import STAGES
from pgx_mcts_bench.leaderboard import discover_checkpoints, leaderboard, render


def _stage(
    index: int,
    iterations: int,
    *,
    crossings: float = 1.0,
    moves_10: float = 10.0,
    crossings_10: float | None = None,
) -> dict:
    source, scramble = STAGES[index]
    return {
        "stage": index,
        "source": source,
        "scramble": scramble,
        "iterations": iterations,
        "promoted": True,
        "optimal_crossings": 0 if index == 0 else -1,
        "by_ratio": {
            "1000.0": {
                "crossings": crossings,
                "solved": 0.5,
                "expected_crossings": crossings / 0.5,
            },
            "10.0": {
                "crossings": crossings if crossings_10 is None else crossings_10,
                "moves": moves_10,
            },
        },
    }


def _save(root: Path, name: str, stages: list[dict]) -> Path:
    path = root / name / "checkpoints" / f"{name}.pt"
    path.parent.mkdir(parents=True)
    torch.save({"candidate": name, "stages": stages}, path)
    return path


def test_discovers_resume_checkpoints_but_not_stage_snapshots(tmp_path: Path) -> None:
    resume = _save(tmp_path, "arm", [_stage(0, 2)])
    snapshot = resume.parent / "arm" / "stage00-after.pt"
    snapshot.parent.mkdir()
    torch.save({"candidate": "arm"}, snapshot)

    assert discover_checkpoints([tmp_path]) == [resume]


def test_discovers_checkpoint_files_in_a_flat_server_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "server.pt"
    torch.save({"candidate": "server"}, snapshot)

    assert discover_checkpoints([tmp_path]) == [snapshot]


def test_retired_negative_result_is_not_ranked(tmp_path: Path) -> None:
    _save(tmp_path, "s-ff4-p5", [_stage(0, 2)])
    _save(tmp_path, "active-arm", [_stage(0, 2)])

    rows, warnings = leaderboard([tmp_path])

    assert warnings == []
    assert [row.name for row in rows] == ["active-arm"]


def test_leaderboard_uses_current_rungs_and_counts_repeated_work(tmp_path: Path) -> None:
    obsolete = {
        **_stage(0, 99),
        "source": "P(2,11)#0",
        "scramble": 0,
    }
    _save(
        tmp_path,
        "repeat-arm",
        [_stage(0, 2), _stage(1, 4), _stage(1, 6), obsolete],
    )

    rows, warnings = leaderboard([tmp_path])

    assert warnings == []
    assert len(rows) == 1
    row = rows[0]
    assert row.highest_stage == 1
    assert row.rungs_cleared == 2
    assert row.total_iterations == 12
    assert row.iterations_per_rung == 6.0


def test_newest_snapshot_wins_and_render_has_iteration_columns(tmp_path: Path) -> None:
    local = _save(tmp_path / "local", "same-arm", [_stage(0, 20)])
    server = _save(tmp_path / "server-snapshot", "same-arm", [_stage(1, 4)])
    os.utime(local, ns=(1, 1))
    os.utime(server, ns=(2, 2))

    rows, warnings = leaderboard([tmp_path / "local", tmp_path / "server-snapshot"])

    assert warnings == []
    assert len(rows) == 1
    assert rows[0].checkpoint == server
    table = render(rows)
    assert "avgΔ(n)" in table
    assert "avgΔmv(n)" in table
    assert "avgΔL10:1(n)" in table
    assert "it/r" in table
    assert table.rstrip().endswith("4   4.00")


def test_average_gap_uses_theorem_then_achieved_min_for_unknown_rungs(
    tmp_path: Path,
) -> None:
    low = _stage(17, 2, crossings=3.0)
    high = _stage(17, 2, crossings=5.0)
    _save(tmp_path, "low", [_stage(0, 2, crossings=0.0), low])
    _save(tmp_path, "high", [_stage(0, 2, crossings=1.0), high])

    rows, warnings = leaderboard([tmp_path])
    by_name = {row.name: row for row in rows}

    assert warnings == []
    assert by_name["low"].average_gap == 0.0
    assert by_name["low"].top_gap == 0.0
    assert by_name["low"].gap_rungs == 2
    assert by_name["high"].average_gap == 1.5  # mean of labelled +1 and unknown +2
    assert by_name["high"].top_gap == 2.0
    assert by_name["high"].gap_rungs == 2


def test_average_move_delta_uses_common_rungs_and_u1_reference(tmp_path: Path) -> None:
    _save(
        tmp_path,
        "u1-puct",
        [_stage(0, 2, moves_10=10.0), _stage(17, 2, moves_10=20.0)],
    )
    _save(
        tmp_path,
        "other",
        [_stage(0, 2, moves_10=12.0), _stage(17, 2, moves_10=17.0)],
    )

    rows, warnings = leaderboard([tmp_path])
    by_name = {row.name: row for row in rows}

    assert warnings == []
    assert by_name["u1-puct"].average_move_delta == 0.0
    assert by_name["u1-puct"].move_delta_rungs == 2
    assert by_name["other"].average_move_delta == -0.5
    assert by_name["other"].move_delta_rungs == 2


def test_average_loss_10_delta_uses_paired_ratio_10_costs(tmp_path: Path) -> None:
    _save(
        tmp_path,
        "u1-puct",
        [
            _stage(0, 2, crossings_10=2.0, moves_10=10.0),
            _stage(17, 2, crossings_10=3.0, moves_10=20.0),
        ],
    )
    _save(
        tmp_path,
        "other",
        [
            _stage(0, 2, crossings_10=1.0, moves_10=15.0),
            _stage(17, 2, crossings_10=4.0, moves_10=12.0),
        ],
    )

    rows, warnings = leaderboard([tmp_path])
    by_name = {row.name: row for row in rows}

    assert warnings == []
    assert by_name["u1-puct"].average_loss_10_delta == 0.0
    assert by_name["u1-puct"].loss_10_delta_rungs == 2
    # Rung deltas are (25 - 30) and (52 - 50), averaged over common rungs.
    assert by_name["other"].average_loss_10_delta == -1.5
    assert by_name["other"].loss_10_delta_rungs == 2
