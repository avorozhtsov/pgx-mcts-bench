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
    solve_rate: float = 0.5,
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
                "solved": solve_rate,
                "expected_crossings": crossings / solve_rate,
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
    assert " n  avg_sr   avgΔ  avgΔmv" in table
    assert "avgΔ(n)" not in table
    assert "avgΔmv(n)" not in table
    assert "avg_sr" in table
    assert "cc/sr" not in table
    assert "avgΔL10:1(win)" in table
    assert "avgΔL10:1(n)" not in table
    assert "it/r" in table
    assert table.rstrip().endswith("4   4.00")


def test_average_solve_rate_uses_all_available_rungs(tmp_path: Path) -> None:
    _save(
        tmp_path,
        "arm",
        [
            _stage(0, 2, solve_rate=1.0),
            _stage(1, 2, solve_rate=0.5),
            _stage(2, 2, solve_rate=0.75),
        ],
    )

    rows, warnings = leaderboard([tmp_path])

    assert warnings == []
    assert rows[0].solve_rate == 0.75
    assert rows[0].average_solve_rate == 0.75
    assert rows[0].solve_rate_rungs == 3


def test_newer_shallow_benchmark_does_not_hide_deeper_resume_checkpoint(
    tmp_path: Path,
) -> None:
    deep = _save(tmp_path / "ladder", "same-arm", [_stage(0, 2), _stage(1, 4)])
    shallow = _save(tmp_path / "device-benchmark", "same-arm", [_stage(0, 1)])
    os.utime(deep, ns=(1, 1))
    os.utime(shallow, ns=(2, 2))

    rows, warnings = leaderboard([tmp_path])

    assert warnings == []
    assert len(rows) == 1
    assert rows[0].checkpoint == deep


def test_average_gap_backfills_scheduled_knot_theorem_into_old_checkpoints(
    tmp_path: Path,
) -> None:
    low = _stage(17, 2, crossings=3.0)
    high = _stage(17, 2, crossings=5.0)
    _save(tmp_path, "low", [_stage(0, 2, crossings=0.0), low])
    _save(tmp_path, "high", [_stage(0, 2, crossings=1.0), high])

    rows, warnings = leaderboard([tmp_path])
    by_name = {row.name: row for row in rows}

    assert warnings == []
    assert by_name["low"].average_gap == 1.0  # stage 0 gap 0, stage 17 gap 2
    assert by_name["low"].top_gap == 2.0
    assert by_name["low"].gap_rungs == 2
    assert by_name["high"].average_gap == 2.5  # stage 0 gap 1, stage 17 gap 4
    assert by_name["high"].top_gap == 4.0
    assert by_name["high"].gap_rungs == 2


def test_rung_27_displays_database_unknotting_number(tmp_path: Path) -> None:
    record = _stage(27, 2, crossings=2.0, solve_rate=1.0)
    assert record["source"] == "R(3,18)#0"
    assert record["optimal_crossings"] == -1
    _save(tmp_path, "arm", [_stage(0, 2), record])

    rows, warnings = leaderboard([tmp_path])

    assert warnings == []
    row = rows[0]
    assert row.rung_scores[-1].optimal_crossings == 2
    assert render([row]).splitlines()[2].split()[3] == "2"


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
        "s-window-128",
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
    assert by_name["s-window-128"].average_loss_10_delta == 0.0
    assert by_name["s-window-128"].loss_10_delta_rungs == 2
    # Rung deltas are (25 - 30) and (52 - 50), averaged over common rungs.
    assert by_name["other"].average_loss_10_delta == -1.5
    assert by_name["other"].loss_10_delta_rungs == 2
