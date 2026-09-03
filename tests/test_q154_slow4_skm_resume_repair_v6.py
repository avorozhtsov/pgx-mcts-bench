from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path("scripts/run_local_q154_slow4_skm_resume_repair_v6.py")
    spec = importlib.util.spec_from_file_location("slow4_skm_resume_v6", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent.resolve()))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_resume_repair_keeps_v5_command_and_one_worker() -> None:
    module = _module()
    command = module.prior._command()
    assert command[command.index("--scientist-task-timeout-seconds") + 1] == "21600"
    assert command[-1] == "--resume"
    assert module.prior.base.LOCK.name == "slow4-orchestrator.lock"
    assert module.GATE.name == "SLOW4_Q154_SKM_RESUME_REPAIR_V6_VERIFIED.json"
    assert module.STATUS.name == "slow4-skm-resume-repair-v6-status.json"
