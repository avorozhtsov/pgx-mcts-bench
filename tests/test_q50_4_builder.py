from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]


def module():
    path = ROOT / "scripts/build_q50_4_updated.py"
    spec = importlib.util.spec_from_file_location("q50_4_builder_for_test", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def test_q50_4_rotates_bridge_remainders_and_balances_pair() -> None:
    builder = module()
    assert builder.BRIDGE_TARGET_ORDER == (9, 10, 9, 11, 10, 12, 9, 11, 10, 12)
    assert Counter(builder.q50_3.BRIDGE_TARGET_ORDER + builder.BRIDGE_TARGET_ORDER) == {9: 5, 10: 5, 11: 5, 12: 5}


def test_q50_4_policy_is_strict_and_isolated() -> None:
    policy = json.loads((ROOT / "research/local-q-skm-ablation/q50-4-updated-policy.json").read_text())
    assert policy["status"] == "PREPARED"
    assert policy["output"]["low_strand_rows"] == 40
    assert policy["output"]["simple_bridge_rows"] == 10
    assert policy["source_pool"]["bank"].endswith("q200-2.json")
    assert policy["safety"]["active_q254_files_are_immutable"] is True
    assert policy["safety"]["strict_no_sharing"] is True
