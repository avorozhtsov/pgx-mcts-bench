from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _builder_module():
    path = Path(__file__).parents[1] / "scripts/build_q50_2_updated.py"
    spec = importlib.util.spec_from_file_location("q50_2_builder_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(identity: str, strands: int, *, bridge: bool = False) -> dict:
    return {"id": identity, "strands": strands, "capacity_bridge": bridge}


def test_q50_2_exact_schedule_mixes_low_rows_and_bridges() -> None:
    builder = _builder_module()
    easy = [_row(f"easy-{index}", 3) for index in range(32)]
    medium = [_row(f"medium-{index}", 6) for index in range(8)]
    bridges = [
        _row(f"bridge-{index}", strands, bridge=True)
        for index, strands in enumerate(builder.BRIDGE_TARGET_ORDER)
    ]

    scheduled = builder.schedule_rows(easy, medium, bridges)

    assert len(scheduled) == 50
    assert [
        index for index, row in enumerate(scheduled, start=1) if row["capacity_bridge"]
    ] == list(range(5, 51, 5))
    assert [
        row["strands"] for row in scheduled if row["capacity_bridge"]
    ] == [9, 10, 9, 11, 10, 12, 9, 11, 10, 12]
    assert [
        index
        for index, row in enumerate(scheduled, start=1)
        if not row["capacity_bridge"] and row["strands"] == 6
    ] == [9, 14, 19, 24, 29, 34, 39, 44]


def test_q50_2_policy_preserves_twenty_percent_bridge_fraction() -> None:
    root = Path(__file__).parents[1]
    policy = json.loads(
        (root / "research/local-q-skm-ablation/q50-2-updated-policy.json").read_text()
    )

    assert policy["status"] == "PREPARED"
    assert policy["output"]["low_strand_rows"] == 40
    assert policy["output"]["simple_bridge_rows"] == 10
    assert policy["output"]["bridge_fraction"] == 0.2
    assert policy["bridge_schedule"]["target_order"] == [
        9,
        10,
        9,
        11,
        10,
        12,
        9,
        11,
        10,
        12,
    ]
    assert policy["exact_schedule"]["bridge_positions_one_based"] == list(
        range(5, 51, 5)
    )
