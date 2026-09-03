from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]


def module():
    path = ROOT / "scripts/build_q50_5_tournament.py"
    spec = importlib.util.spec_from_file_location("q50_5_tournament_builder_for_test", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_q50_5_rotates_bridge_remainders() -> None:
    builder = module()
    assert builder.BRIDGE_TARGET_ORDER == (12, 11, 12, 10, 12, 11, 12, 11, 12, 12)
    assert Counter(builder.BRIDGE_TARGET_ORDER) == {10: 1, 11: 3, 12: 6}


def test_q50_5_policy_registers_same_root_tournament() -> None:
    policy = json.loads(
        (ROOT / "research/local-q-skm-ablation/q50-5-tournament-policy.json").read_text()
    )
    assert policy["status"] == "PREPARED"
    assert policy["output"]["first_cumulative_index"] == 305
    assert policy["training"]["trajectories_per_root_and_ratio"] == 10
    assert policy["training"]["invalid_or_absent"] == "ignored-not-negative"
    assert policy["safety"]["predecessor_representation_reuse"] is False
