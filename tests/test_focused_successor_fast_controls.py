from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_focused_control_sources_fail_closed_and_preserve_protocol() -> None:
    preparation = (REPO / "scripts/prepare_focused_successor_fast_controls.py").read_text()
    branch = (REPO / "scripts/run_focused_successor_branch.py").read_text()
    launcher = (REPO / "scripts/run_focused_successor_fast_controls.py").read_text()
    assert '"strand-graph-12-rl-control"' in preparation
    assert '"raster-axial-12-control"' in preparation
    assert '"proof_distilled": "QUEUED"' in preparation
    assert '"proof_embedding": "QUEUED"' in preparation
    assert '"diverse_slow_selection": "QUEUED"' in preparation
    assert 'arm="scheduled-no-sharing"' in branch
    assert "selfplay_games=4" in branch
    assert "train_steps=24" in branch
    assert "terminal_full_retention_audit=True" in branch
    assert "ThreadPoolExecutor(max_workers=2)" in launcher
    assert "flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)" in launcher


def test_focused_control_seeds_are_distinct() -> None:
    preparation = (REPO / "scripts/prepare_focused_successor_fast_inputs.py").read_text()
    assert "202608310101" in preparation
    assert "202608310104" in preparation


def test_recovery_v2_extracts_single_scientist_carry() -> None:
    repair = (REPO / "scripts/prepare_focused_successor_fast_controls_recovery_v2.py").read_text()
    launcher = (REPO / "scripts/run_focused_successor_fast_controls_recovery_v2.py").read_text()
    assert '"scientist": state["scientists"][scientist]' in repair
    assert '"f_old": int(state["f_old"][scientist])' in repair
    assert '"rehearsal_panel_cursor": int(state["rehearsal_panel_cursor"][scientist])' in repair
    assert 'base.STAGE = "q50-4-updated-scheduled-no-sharing-bounded-recovery-v2"' in launcher


def test_recovery_v3_uses_spawn_importable_runtime_name() -> None:
    branch = (REPO / "scripts/run_focused_successor_branch.py").read_text()
    launcher = (REPO / "scripts/run_focused_successor_fast_controls_recovery_v3.py").read_text()
    assert 'name = "pgx_mcts_bench.q254_sv2_curriculum_runtime"' in branch
    assert 'base.STAGE = "q50-4-updated-scheduled-no-sharing-bounded-recovery-v3"' in launcher
