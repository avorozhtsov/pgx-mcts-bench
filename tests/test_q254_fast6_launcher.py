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


def test_q254_runtime_is_isolated_and_seeded_from_boundary_zero() -> None:
    prepare = load_script("prepare_local_q254_fast6")
    source = prepare.SV2_SOURCE.read_text()
    transformed = prepare.transform_runtime(source)
    assert source != transformed
    assert source.count('"boundary_completed_rungs": 30,') == 1
    assert '"boundary_completed_rungs": 0,' in transformed
    assert '"semantic-v2-q254-first-block-seeded-order-v1",' in transformed
    assert prepare.RUNTIME not in prepare.SOURCE_FILES
    assert prepare.SV2_SOURCE in prepare.SOURCE_FILES
    assert prepare.SV2_SOURCE.read_text() == source


def test_q254_launcher_preserves_lineage_seed_indices_and_isolated_paths() -> None:
    launcher = load_script("run_local_q254_fast6")
    label = "skm-v1-simple-strand-graph"
    command = launcher.branch_command(label, "strand-graph-12", Path("/carry.pt.gz"))
    assert command[command.index("--seed") + 1] == str(
        202608230500 + launcher.q154.PRIMARY_8_SEED_INDEX[label]
    )
    assert command[command.index("--initial-state") + 1] == "/carry.pt.gz"
    assert launcher.Q204_ROOT != launcher.ROOT
    assert launcher._status["maximum_experiment_workers"] == 6
