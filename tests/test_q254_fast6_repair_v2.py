from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_v2_runtime_accepts_seeded_to_seeded_transition() -> None:
    repair = load_script("prepare_local_q254_fast6_repair_v2")
    transformed = repair.transform_runtime(repair.base.SV2_SOURCE.read_text())
    assert transformed.count(
        '"from_policy": "seeded-outcome-interleaved-exposure-v1",'
    ) == 1
    assert '"boundary_completed_rungs": 0,' in transformed
    assert repair.base.RUNTIME.name.endswith("repair_v2.py")


def test_repair_v2_launcher_uses_isolated_stage_and_branch() -> None:
    repair = load_script("run_local_q254_fast6_repair_v2")
    command = repair.branch_command("q-grown-raster-axial-12", "raster-axial-12", Path("/carry"))
    assert command[1].endswith("run_local_q254_fast6_branch_repair_v2.py")
    assert repair.base.STAGE.endswith("repair-v2")
