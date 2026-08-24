from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _builder_module():
    path = ROOT / "scripts/build_q50_3_updated.py"
    spec = importlib.util.spec_from_file_location("q50_3_builder_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q50_3_rotates_remainder_bridges_and_preserves_exact_schedule() -> None:
    builder = _builder_module()
    assert builder.BRIDGE_TARGET_ORDER == (11, 12, 9, 10, 11, 12, 9, 10, 11, 12)
    assert Counter(builder.q50_2.BRIDGE_TARGET_ORDER + builder.BRIDGE_TARGET_ORDER) == {
        9: 5,
        10: 5,
        11: 5,
        12: 5,
    }
    easy = [{"id": f"easy-{index}", "strands": 3} for index in range(32)]
    medium = [{"id": f"medium-{index}", "strands": 6} for index in range(8)]
    bridges = [
        {"id": f"bridge-{index}", "strands": target, "capacity_bridge": True}
        for index, target in enumerate(builder.BRIDGE_TARGET_ORDER)
    ]
    scheduled = builder.q50_2.schedule_rows(easy, medium, bridges)
    assert [
        index for index, row in enumerate(scheduled, start=1) if row.get("capacity_bridge")
    ] == list(range(5, 51, 5))


def test_q50_3_policy_is_isolated_and_strict() -> None:
    policy = json.loads(
        (ROOT / "research/local-q-skm-ablation/q50-3-updated-policy.json").read_text()
    )
    assert policy["status"] == "PREPARED"
    assert policy["output"]["low_strand_rows"] == 40
    assert policy["output"]["simple_bridge_rows"] == 10
    assert policy["bridge_schedule"]["two_batch_balance"] == {
        "9": 5,
        "10": 5,
        "11": 5,
        "12": 5,
    }
    assert policy["safety"]["active_q204_files_are_immutable"] is True
    assert policy["safety"]["strict_no_sharing"] is True
