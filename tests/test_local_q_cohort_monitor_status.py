from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from scripts import write_local_q_cohort_monitor_status as monitor


def test_parse_cpu_time_and_process_rows() -> None:
    processes = monitor.parse_ps(
        "10 1 02:03 python launcher.py\n"
        "11 10 01:02:03 python branch.py\n"
        "12 11 1-01:02:03.50 python scientist.py\n"
    )
    assert processes[10].cpu_seconds == 123
    assert processes[11].cpu_seconds == 3723
    assert processes[12].cpu_seconds == 90123.5
    assert monitor.descendants(processes, 10) == {11, 12}


def test_build_status_collapses_wrappers_and_verifies_activity(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    launcher = tmp_path / "launcher.py"
    gate = tmp_path / "gate.json"
    launcher.write_text("launcher\n")
    gate.write_text("{}\n")
    output = root / "branch"
    before = monitor.parse_ps(
        f"10 1 00:01 python {launcher}\n"
        f"11 10 00:00 uv run pgx-mcts-bench braid-sv2-coordinated --output {output}\n"
        f"12 11 00:00 pgx-mcts-bench braid-sv2-coordinated --output {output}\n"
        "13 12 10:00 python -c from multiprocessing.spawn import spawn_main\n"
    )
    after = monitor.parse_ps(
        f"10 1 00:01 python {launcher}\n"
        f"11 10 00:00 uv run pgx-mcts-bench braid-sv2-coordinated --output {output}\n"
        f"12 11 00:00 pgx-mcts-bench braid-sv2-coordinated --output {output}\n"
        "13 12 10:06 python -c from multiprocessing.spawn import spawn_main\n"
    )
    status = monitor.build_status(
        label="example",
        cohort="slow-4",
        phase="Q154",
        artifact_root=root,
        launcher_script=launcher,
        gate=gate,
        min_scientists=1,
        max_scientists=1,
        launcher_pid=10,
        before=before,
        after=after,
        disk_free_bytes=9 * 1024**3,
        sampled_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    assert status["state"] == "VERIFIED ACTIVE"
    assert status["scientists"]["pids"] == [13]
    assert status["writers"]["leaf_pids"] == [12]
    assert status["writers"]["duplicate_roots"] == {}


def test_build_status_refuses_duplicate_leaf_writers(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    launcher = tmp_path / "launcher.py"
    gate = tmp_path / "gate.json"
    launcher.write_text("launcher\n")
    gate.write_text("{}\n")
    output = root / "branch"
    text = (
        f"10 1 00:01 python {launcher}\n"
        f"11 10 00:00 python run_local_q204_fast6_branch.py --output {output}\n"
        f"12 10 00:00 python run_local_q204_fast6_branch.py --output {output}\n"
        "13 11 10:00 python -c from multiprocessing.spawn import spawn_main\n"
    )
    before = monitor.parse_ps(text)
    after = dict(before)
    after[13] = monitor.Process(13, 11, before[13].cpu_seconds + 5, before[13].command)
    status = monitor.build_status(
        label="example",
        cohort="fast-6",
        phase="Q204",
        artifact_root=root,
        launcher_script=launcher,
        gate=gate,
        min_scientists=1,
        max_scientists=1,
        launcher_pid=10,
        before=before,
        after=after,
        disk_free_bytes=9 * 1024**3,
        sampled_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    assert status["state"] == "LAUNCHED"
    assert status["checks"]["duplicate_writers_absent"] is False
