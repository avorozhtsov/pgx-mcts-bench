import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pgx_mcts_bench.data import ReplayBuffer


def _launcher_module():
    path = Path(__file__).parents[1] / "scripts/run_local_q154_updated_continuation.py"
    spec = importlib.util.spec_from_file_location("q154_launcher_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slow_launcher_module():
    path = Path(__file__).parents[1] / "scripts/run_local_q_slow4_continuation.py"
    spec = importlib.util.spec_from_file_location("q_slow4_launcher_for_test", path)
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


def test_fast_6_launcher_excludes_slow_combined_and_v3() -> None:
    launcher = _launcher_module()
    labels = [label for label, _scientist, _simulations, _timeout in launcher.BRANCHES]
    assert len(labels) == 6
    assert "q-grown-raster-invariant-combined-dual-12" not in labels
    assert "skm-v2-high-combined-dual" not in labels
    assert "cyclic-graph-dual-v3" not in labels
    assert "cyclic-memory-deep-v3" not in labels
    assert launcher.Q104_MARKER.name == "PRIMARY_8_LINEAGES_Q104_COMPLETE.json"
    assert launcher.COMPLETION_MARKER.name == "ALL_FAST_6_LINEAGES_Q154_COMPLETE"


def test_slow_4_launcher_is_serial_and_uses_two_hour_segments() -> None:
    slow = _slow_launcher_module()
    assert [row[0] for row in slow.SLOW_BRANCHES] == [
        "q-grown-raster-invariant-combined-dual-12",
        "skm-v2-high-combined-dual",
        "cyclic-memory-deep-v3",
        "cyclic-graph-dual-v3",
    ]
    assert all(row[3] for row in slow.SLOW_BRANCHES)
    assert slow.MAX_EXPERIMENT_WORKERS == 1
    assert slow.SLOW_TIMEOUT_SECONDS == 7200
    assert slow.SLOW_TRAINING_SECONDS_PER_ITERATION_AT_REFERENCE == 7200


def test_primary_8_marker_binds_report_and_state_hashes(tmp_path: Path) -> None:
    launcher = _launcher_module()
    launcher.Q104_ROOT = tmp_path / "q104"
    launcher.Q104_STAGE = "stage"
    launcher.Q104_MARKER = launcher.Q104_ROOT / "PRIMARY_8_LINEAGES_Q104_COMPLETE.json"
    launcher.STATUS = tmp_path / "q154/launcher-status.json"
    launcher.BRANCHES = (("lineage", "scientist", 40, False),)
    launcher.PRIMARY_8_BRANCHES = launcher.BRANCHES
    branch = launcher.Q104_ROOT / "branches/lineage/stage"
    branch.mkdir(parents=True)
    report = branch / "report.json"
    state = branch / "state.pt.gz"
    report.write_text("{}\n")
    state.write_bytes(b"state")
    launcher.Q104_MARKER.write_text(
        json.dumps(
            {
                "schema": "q104-primary-8-completion-v1",
                "lineages": ["lineage"],
                "artifacts": {
                    "lineage": {
                        "report_sha256": launcher._sha256(report),
                        "state_sha256": launcher._sha256(state),
                    }
                },
            }
        )
    )

    launcher._wait_for_q104()

    marker = json.loads(launcher.Q104_MARKER.read_text())
    marker["artifacts"]["lineage"]["state_sha256"] = "bad"
    launcher.Q104_MARKER.write_text(json.dumps(marker))
    with pytest.raises(RuntimeError, match="state hash changed"):
        launcher._wait_for_q104()
