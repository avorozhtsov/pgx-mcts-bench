from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load(name: str):
    path = ROOT / "scripts" / f"{name}.py"; spec = importlib.util.spec_from_file_location(f"{name}_test", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def test_q304_runtime_is_seeded_to_seeded_and_isolated() -> None:
    prepare = load("prepare_local_q304_fast6"); transformed = prepare.transform_runtime(prepare.SV2_SOURCE.read_text())
    assert '"boundary_completed_rungs": 0,' in transformed
    assert transformed.count('"from_policy": "seeded-outcome-interleaved-exposure-v1",') == 1
    assert '"semantic-v2-q304-first-block-seeded-order-v1",' in transformed
    assert prepare.RUNTIME not in prepare.SOURCE_FILES


def test_q304_launcher_preserves_seed_and_repair_v2_boundary() -> None:
    launcher = load("run_local_q304_fast6"); label = "skm-v1-simple-strand-graph"
    command = launcher.branch_command(label, "strand-graph-12", Path("/carry"))
    assert command[command.index("--seed") + 1] == str(202608240500 + launcher.base.q154.PRIMARY_8_SEED_INDEX[label])
    assert launcher.Q254_STAGE.endswith("repair-v2")
    assert launcher.base.STAGE == "q50-4-updated-scheduled-no-sharing-bounded"
