import importlib.util
from pathlib import Path


def test_recovery_v2_binds_homebrew_uv_path() -> None:
    path = (
        Path(__file__).parents[1]
        / "scripts/run_local_q154_slow4_seed_recovery_v2.py"
    )
    spec = importlib.util.spec_from_file_location("q154_seed_recovery_v2_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.LAUNCHD_PATH.split(":")[0] == "/opt/homebrew/bin"
    assert Path("/opt/homebrew/bin/uv").is_file()
