import importlib.util
from pathlib import Path


def _load(name: str, relative: str):
    path = Path(__file__).parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_recovery_v3_uses_isolated_new_gates() -> None:
    recovery = _load(
        "q154_contract_recovery_v3_test",
        "scripts/run_local_q154_slow4_contract_recovery_v3.py",
    )
    prepare = _load(
        "q154_contract_recovery_v3_prepare_test",
        "scripts/prepare_local_q154_slow4_contract_recovery_v3.py",
    )

    assert recovery.GATE == prepare.NEW_RECOVERY_GATE
    assert prepare.NEW_SPLIT_GATE.parent == recovery.ROOT
    assert prepare.NEW_SPLIT_GATE != prepare.OLD_SPLIT_GATE
    assert recovery.LAUNCHD_PATH.split(":")[0] == "/opt/homebrew/bin"
