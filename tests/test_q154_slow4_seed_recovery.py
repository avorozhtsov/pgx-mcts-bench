import importlib.util
from pathlib import Path


def _load(name: str, relative: str):
    path = Path(__file__).parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slow4_recovery_adds_unique_v3_seed_indices() -> None:
    recovery = _load(
        "q154_slow4_seed_recovery_for_test",
        "scripts/run_local_q154_slow4_seed_recovery.py",
    )
    launcher = _load(
        "q154_launcher_for_seed_recovery_test",
        "scripts/run_local_q154_updated_continuation.py",
    )

    indices = recovery._effective_seed_indices(launcher)

    assert indices["q-grown-raster-invariant-combined-dual-12"] == 1
    assert indices["skm-v2-high-combined-dual"] == 7
    assert indices["cyclic-memory-deep-v3"] == 8
    assert indices["cyclic-graph-dual-v3"] == 9
    assert len(indices.values()) == len(set(indices.values()))
