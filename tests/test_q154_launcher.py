import gzip
import importlib.util
from pathlib import Path

import numpy as np
import torch

from pgx_mcts_bench.data import ReplayBuffer


def _launcher_module():
    path = Path(__file__).parents[1] / "scripts/run_local_q154_updated_continuation.py"
    spec = importlib.util.spec_from_file_location("q154_launcher_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rehearsal_debt_is_lineage_local_and_idempotent(tmp_path: Path) -> None:
    launcher = _launcher_module()
    launcher.Q104_ROOT = tmp_path / "q104"
    launcher.Q104_STAGE = "stage"
    launcher.REHEARSAL_DEBT = tmp_path / "q154/protocol/debt.json"
    launcher.BRANCHES = (("lineage", "scientist", 40, True),)

    replay = ReplayBuffer(100, np.random.default_rng(7))
    state = {
        "scientists": {"scientist": {"replay": replay}},
        "events": [
            {
                "scientists": {
                    "scientist": {
                        "rehearsal": {
                            "F_old": 8,
                            "iterations": [{}, {}, {}],
                            "hard_timeout": {"checkpoint_recovered": True},
                        },
                    }
                },
            }
        ],
    }
    state_path = launcher.Q104_ROOT / "branches/lineage/stage/state.pt.gz"
    state_path.parent.mkdir(parents=True)
    with gzip.open(state_path, "wb") as handle:
        torch.save(state, handle)

    assert launcher._build_rehearsal_debt() == {"lineage": 5}
    first_hash = launcher._sha256(launcher.REHEARSAL_DEBT)
    assert launcher._build_rehearsal_debt() == {"lineage": 5}
    assert launcher._sha256(launcher.REHEARSAL_DEBT) == first_hash
