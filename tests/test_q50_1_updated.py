from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "build_q50_1_updated.py"
    spec = importlib.util.spec_from_file_location("build_q50_1_updated", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bridge_mix_is_balanced_across_strands_7_through_11() -> None:
    module = _module()
    mix = module.BRIDGE_MIX
    assert mix == {7: 2, 8: 2, 9: 2, 10: 2, 11: 2}
    assert sum(mix.values()) == 10
    assert set(mix) == set(range(7, 12))


def test_bridge_mix_is_not_mutated_by_copy() -> None:
    module = _module()
    selected = dict(module.BRIDGE_MIX)
    selected[7] -= 1
    assert module.BRIDGE_MIX[7] == 2


def test_report_paths_follow_explicit_cohort_not_glob(tmp_path: Path) -> None:
    module = _module()
    required = ["primary-a", "primary-b"]
    for label in [*required, "deferred-v3"]:
        report = tmp_path / "branches" / label / module.Q104_STAGE / "report.json"
        report.parent.mkdir(parents=True)
        report.write_text("{}\n")

    paths = module._report_paths(tmp_path, required)

    assert [path.parts[-3] for path in paths] == required


def test_report_paths_reject_missing_required_lineage(tmp_path: Path) -> None:
    module = _module()
    try:
        module._report_paths(tmp_path, ["missing"])
    except ValueError as error:
        assert "missing required Q104 reports" in str(error)
    else:
        raise AssertionError("missing cohort report was accepted")
