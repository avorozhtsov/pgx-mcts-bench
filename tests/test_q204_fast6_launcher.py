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


def test_q204_runtime_isolation_starts_seeded_order_at_boundary_zero() -> None:
    prepare = load_script("prepare_local_q204_fast6")
    source = prepare.SV2_SOURCE.read_text()
    transformed = prepare.transform_runtime(source)
    assert source != transformed
    assert source.count('"boundary_completed_rungs": 30,') == 1
    assert '"boundary_completed_rungs": 0,' in transformed
    assert '"semantic-v2-q204-first-block-seeded-order-v1",' in transformed
    assert prepare.SV2_SOURCE.read_text() == source


def test_q204_gate_binds_new_runtime_without_mutating_active_q154_sources() -> None:
    prepare = load_script("prepare_local_q204_fast6")
    assert prepare.RUNTIME not in prepare.SOURCE_FILES
    assert prepare.SV2_SOURCE in prepare.SOURCE_FILES
    assert ROOT / "scripts/run_local_q204_fast6.py" in prepare.SOURCE_FILES
    assert ROOT / "scripts/run_local_q204_fast6_branch.py" in prepare.SOURCE_FILES


def test_q204_fast_launcher_preserves_exact_primary8_seed_indices_and_dose() -> None:
    launcher = load_script("run_local_q204_fast6")
    label = "skm-v1-simple-strand-graph"
    command = launcher.branch_command(label, "strand-graph-12", Path("/carry.pt.gz"))
    assert command[command.index("--seed") + 1] == str(
        202608220400 + launcher.q154.PRIMARY_8_SEED_INDEX[label]
    )
    assert command[command.index("--initial-state") + 1] == "/carry.pt.gz"
    assert launcher._status["maximum_experiment_workers"] == 6
