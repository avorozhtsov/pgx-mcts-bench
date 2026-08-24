from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
POLICY = ROOT / "research/local-q-skm-ablation/dkt-inflation-frontier-policy.json"


def load_builder():
    path = ROOT / "scripts/build_dkt_inflation_frontier.py"
    spec = importlib.util.spec_from_file_location("dkt_frontier_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner():
    path = ROOT / "scripts/run_dkt_inflation_frontier.py"
    spec = importlib.util.spec_from_file_location("dkt_frontier_runner_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frontier_is_deterministic_replayable_and_inflating(tmp_path: Path) -> None:
    builder = load_builder()
    first = tmp_path / "first"
    second = tmp_path / "second"
    audit1 = builder.build(POLICY, first)
    audit2 = builder.build(POLICY, second)
    bank = json.loads((first / "dkt-11a14-inflation-frontier.json").read_text())

    assert audit1["status"] == "passed"
    assert audit1["bank_byte_sha256"] == audit2["bank_byte_sha256"]
    assert audit1["frontier_rows"] == 23
    assert audit1["depth_counts"] == {"2": 7, "4": 8, "6": 8}
    assert all(audit1["checks"].values())
    assert bank["target"] == "11a_14"
    assert bank["published_upper_bound"] == 2
    assert bank["paper_reports_exact_value"] is True
    assert all(row["prefix_depth"] in {2, 4, 6} for row in bank["rows"])
    assert all(
        "CROSSING_CHANGE" not in description
        for row in bank["rows"]
        for description in row["prefix_action_descriptions"]
    )


def test_policy_freezes_cheap_two_stage_search_without_learning() -> None:
    policy = json.loads(POLICY.read_text())
    assert policy["status"] == "PREPARED"
    assert policy["representative"]["lineage"] == "q-grown-raster-axial-12"
    assert policy["evaluation"]["learning"] is False
    assert policy["evaluation"]["screening"] == {
        "attempts_per_frontier": 1,
        "simulations": 64,
        "root_noise": True,
    }
    assert policy["evaluation"]["promotion"] == {
        "count": 4,
        "ranking": "representative-network-p_solve-at-L1000",
        "attempts_per_frontier": 4,
        "simulations": 256,
        "root_noise": True,
    }
    assert policy["acceptance"]["may_enter_training_before_clean_dkt_evaluation"] is False


def test_runner_summary_requires_replay_certificate_and_unchanged_network() -> None:
    runner = load_runner()
    rows = [
        {
            "screening_attempts": [{"solved": False}],
            "promotion_attempts": [{"solved": True}],
            "certificates": [
                {"crossing_changes": 2, "semantic_moves": 37},
            ],
        },
        {
            "screening_attempts": [{"solved": False}],
            "promotion_attempts": [],
            "certificates": [],
        },
    ]
    summary = runner.summarize(rows, network_unchanged=True)
    assert summary["passed"] is True
    assert summary["best_crossing_changes"] == 2
    assert summary["best_semantic_moves"] == 37
    assert summary["screening_attempts"] == 2
    assert summary["promotion_attempts"] == 1
    assert summary["blocking_reasons"] == []

    changed = runner.summarize(rows, network_unchanged=False)
    assert changed["passed"] is False
    assert changed["blocking_reasons"] == ["network_changed_during_evaluation"]


def test_runner_requires_explicit_confirmation(tmp_path: Path) -> None:
    runner = load_runner()
    try:
        runner.run(
            tmp_path / "policy.json",
            tmp_path / "bank.json",
            tmp_path / "result.json",
        )
    except RuntimeError as error:
        assert str(error) == "DKT inflation-frontier evaluation requires explicit confirmation"
    else:
        raise AssertionError("unconfirmed DKT frontier evaluation was accepted")


def test_prepare_writes_exact_evaluation_only_gate(tmp_path: Path) -> None:
    path = ROOT / "scripts/prepare_dkt_inflation_frontier.py"
    spec = importlib.util.spec_from_file_location("dkt_frontier_prepare_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    gate = module.prepare(POLICY, tmp_path / "protocol", verified_tests=999)

    assert gate["status"] == "PREPARED"
    assert gate["prepared"] is True
    assert gate["verified_tests"] == 999
    assert gate["scientific_role"] == "evaluation-only DKT solution mining; no learning"
    assert gate["launch"]["arguments"][-1] == "--confirm-evaluate"
    assert Path(gate["bank"]).is_file()
    assert Path(gate["audit"]).is_file()
