from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path("scripts/run_local_q154_slow4_skm_debt_recovery_v4.py")
    spec = importlib.util.spec_from_file_location("slow4_skm_recovery_v4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_targets_only_skm_with_frozen_protocol() -> None:
    module = _module()
    command = module._command()
    joined = " ".join(command)
    assert str(module.REPO) == "/Users/artemvorozhtsov/projects/pgx-mcts-bench"
    assert "skm-v2-high-combined-dual" in str(module.OUTPUT)
    assert "q-grown-raster-invariant-combined-dual-12" not in joined
    assert "--rehearsal-repair-debt raster-invariant-combined-dual-12=38" in joined
    assert "--pause-after-rungs 30" in joined
    assert "--scientist-task-timeout-seconds 7200" in joined
    assert "--resumable-rehearsal-segments" in command
    assert "--strict-own-budget-rehearsal" in command
    assert "--resume" in command


def test_recovery_uses_one_worker_and_existing_root() -> None:
    module = _module()
    assert module.LOCK.name == "slow4-orchestrator.lock"
    assert module.STATUS.name == "slow4-skm-debt-recovery-v4-status.json"
    assert module.GATE.name == "SLOW4_Q154_SKM_DEBT_RECOVERY_V4_VERIFIED.json"
