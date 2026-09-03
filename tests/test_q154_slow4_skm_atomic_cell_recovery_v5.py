from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path("scripts/run_local_q154_slow4_skm_atomic_cell_recovery_v5.py")
    spec = importlib.util.spec_from_file_location("slow4_skm_atomic_v5", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent.resolve()))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_atomic_cell_recovery_changes_only_timeout() -> None:
    module = _module()
    old = module.ORIGINAL_COMMAND()
    new = module._command()
    old_index = old.index("--scientist-task-timeout-seconds")
    new_index = new.index("--scientist-task-timeout-seconds")
    assert old[old_index + 1] == "7200"
    assert new[new_index + 1] == "21600"
    assert old[: old_index + 1] == new[: new_index + 1]
    assert new[-3:] == ["--resume-timeout-transition", str(module.TRANSITION), "--resume"]
    assert old[old_index + 2 : -1] == new[new_index + 2 : -3]


def test_atomic_cell_recovery_uses_main_and_same_lock() -> None:
    module = _module()
    assert str(module.base.REPO) == "/Users/artemvorozhtsov/projects/pgx-mcts-bench"
    assert module.base.LOCK.name == "slow4-orchestrator.lock"
    assert module.GATE.name == "SLOW4_Q154_SKM_ATOMIC_CELL_RECOVERY_V5_VERIFIED.json"
    assert module.STATUS.name == "slow4-skm-atomic-cell-recovery-v5-status.json"
