from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
POLICY = ROOT / "research/local-q-skm-ablation/dkt-shadow4-raster-axial-policy.json"


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result_row(
    identity: str, *, upper: int = 2, best: int | None = None
) -> dict[str, object]:
    return {
        "id": identity,
        "registered_upper_bound": upper,
        "best_crossing_changes": best,
        "evaluation": {
            "10.0": {
                "attempts": [{"attempt": index} for index in range(4)],
            }
        },
    }


def test_shadow4_builder_is_deterministic_minimum_acs10_and_disjoint(tmp_path: Path) -> None:
    builder = load_script("build_dkt_shadow4_raster_axial")
    audit = builder.build(POLICY, tmp_path)
    bank = json.loads((tmp_path / "dkt-disjoint-shadow4.json").read_text())

    assert audit["status"] == "passed"
    assert audit["dkt_profile"]["minimum_acs10"] == 63
    assert audit["eligible_minimum_l10_candidates"] == 44
    assert audit["selected_ids"] == ["11a_109", "11a_112", "11a_113", "11a_129"]
    assert all(audit["checks"].values())
    assert bank["name"] == "dkt-disjoint-shadow4-minimum-acs10"
    assert bank["role"] == "evaluation-only; reserved from all training and selection"
    assert len(bank["rows"]) == 4
    assert all(row["acs10"] == 63 for row in bank["rows"])
    assert all(str(row["representation_id"]).startswith("braid:") for row in bank["rows"])


def test_shadow4_policy_freezes_the_low_cost_protocol() -> None:
    policy = json.loads(POLICY.read_text())
    assert policy["status"] == "PREPARED"
    assert policy["representative"] == {
        "lineage": "q-grown-raster-axial-12",
        "scientist": "raster-axial-12",
        "checkpoint_stage": "Q204",
    }
    assert policy["evaluation"] == {
        "learning": False,
        "sharing": False,
        "objective_ratio": 10.0,
        "attempts_per_example": 4,
        "simulations": 256,
        "action_horizon": 128,
        "root_noise": True,
        "temperature": 0.0,
        "base_seed": 2026082304,
    }
    assert policy["selection"]["exclude_q_training_through"] == 254
    assert policy["selection"]["exclude_dkt_knot_identities"] is True
    assert policy["selection"]["exclude_dkt_exact_representations"] is True
    assert policy["selection"]["reserve_from_future_training"] is True


def test_shadow4_summary_requires_complete_repeat_or_improve_and_no_mutation() -> None:
    runner = load_script("run_dkt_shadow4_raster_axial")
    rows = [
        _result_row("a", best=3),
        _result_row("b", best=2),
        _result_row("c"),
        _result_row("d", best=4),
    ]
    summary = runner.summarize(rows, network_unchanged=True)
    assert summary["passed"] is True
    assert summary["attempts"] == 16
    assert summary["repeat_or_improve_ids"] == ["b"]
    assert summary["blocking_reasons"] == []

    mutated = runner.summarize(rows, network_unchanged=False)
    assert mutated["passed"] is False
    assert mutated["blocking_reasons"] == ["network_changed_during_evaluation"]


def test_shadow4_summary_rejects_incomplete_or_nonrepeating_results() -> None:
    runner = load_script("run_dkt_shadow4_raster_axial")
    rows = [_result_row(identity, best=3) for identity in ("a", "b", "c")]
    summary = runner.summarize(rows, network_unchanged=True)
    assert summary["passed"] is False
    assert summary["blocking_reasons"] == [
        "incomplete_4x4_evaluation",
        "no_replay_verified_repeat_or_improvement",
    ]


def test_shadow4_runner_requires_explicit_evaluation_confirmation(tmp_path: Path) -> None:
    runner = load_script("run_dkt_shadow4_raster_axial")
    try:
        runner.run(tmp_path / "gate.json", tmp_path / "result.json")
    except RuntimeError as error:
        assert str(error) == "shadow evaluation requires explicit confirmation"
    else:
        raise AssertionError("unconfirmed shadow evaluation was accepted")
