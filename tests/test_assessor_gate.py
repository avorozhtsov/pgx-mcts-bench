from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from pgx_mcts_bench.adaptive_scientists import KnotItem
from pgx_mcts_bench.assessor_gate import (
    EVIDENCE_SCHEMA,
    build_assessor_gate,
    validate_assessor_gate,
)
from pgx_mcts_bench.collaborative_scientists import (
    BankItem,
    expected_capped_scores,
    prediction_details,
    run_collaborative_scientists,
)
from pgx_mcts_bench.config import BraidGameConfig, ModelConfig
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.networks import make_braid_network


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _evidence(checkpoint_sha256: str, *, representations: int = 100) -> dict:
    rows = []
    for index in range(representations):
        solved = index < 75
        for attempt in range(2):
            rows.append(
                {
                    "representation": f"k{index:03d}",
                    "ratio": 10.0,
                    "attempt": attempt,
                    "p_solve": 0.95 if solved else 0.05,
                    "predicted_crossing_changes": float(index) / 20.0,
                    "predicted_moves": float(index),
                    "solved": solved,
                    "actual_crossing_changes": float(index) / 20.0 if solved else None,
                    "actual_moves": float(index) if solved else None,
                    "failure_cost": 1_000.0,
                    "scan_coverage": 1.0,
                }
            )
    return {
        "schema": EVIDENCE_SCHEMA,
        "identity_disjoint_from_training": True,
        "scan_protocol": {
            "scoring_rule": "mean-head-sweep-v1",
            "steps": 5,
            "native_only": True,
            "semantic_actions": 0,
            "predictions_after_scan": True,
        },
        "scientists": [
            {
                "name": "s-window-128",
                "checkpoint_sha256": checkpoint_sha256,
                "rows": rows,
            }
        ],
    }


def test_assessor_gate_certifies_exact_checkpoint_bound_roster(tmp_path) -> None:
    checkpoint = tmp_path / "scientist.pt"
    checkpoint.write_bytes(b"scientist")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_evidence(_digest(b"scientist"))))
    gate = tmp_path / "gate.json"

    report = build_assessor_gate(evidence, gate)

    assert report["passed"]
    objective = report["scientists"][0]["objectives"]["10.0"]
    assert objective["representations"] == 100
    assert objective["top_quartile_solve_rate"] == 1.0
    assert objective["solve_auc"] == 1.0
    assert validate_assessor_gate(gate, {"s-window-128": checkpoint})["passed"]


def test_assessor_gate_rejects_short_panels_and_checkpoint_drift(tmp_path) -> None:
    checkpoint = tmp_path / "scientist.pt"
    checkpoint.write_bytes(b"scientist")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_evidence(_digest(b"scientist"), representations=20)))
    gate = tmp_path / "gate.json"

    report = build_assessor_gate(evidence, gate)
    assert not report["passed"]
    with pytest.raises(ValueError, match="passed assessor gate"):
        validate_assessor_gate(gate, {"s-window-128": checkpoint})

    evidence.write_text(json.dumps(_evidence(_digest(b"scientist"))))
    build_assessor_gate(evidence, gate)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash_mismatch"):
        validate_assessor_gate(gate, {"s-window-128": checkpoint})


def test_adaptive_scheduler_requires_assessor_certificate(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires --assessor-gate"):
        run_collaborative_scientists(
            {"s-window-128": tmp_path / "missing.pt"},
            tmp_path / "run",
            arm="adaptive-no-sharing",
            rounds=0,
        )


def test_scheduler_uses_five_view_read_only_assessor() -> None:
    config = BraidGameConfig(
        max_len=32,
        max_strands=3,
        simplify_budget=64,
        allow_crossing_change=True,
        serial_window=7,
        serial_act_width=7,
    )
    game = make_game(config)
    knot = KnotItem("T(2,19)", 19, (1,) * 19, 2)
    views, coverage = game.assessment_scan(list(knot.word), knot.strands, steps=5)
    assert len(views) == 5
    assert coverage == 1.0
    assert all((view[0, :, -1] == 1.0).all() for view in views)

    scientist = SimpleNamespace(
        game=game,
        network=make_braid_network(config, ModelConfig(channels=4)),
        config=SimpleNamespace(
            game=config,
            train=SimpleNamespace(device="cpu"),
        ),
        prediction_source="factorized",
        solve_calibration_scale=1.0,
        solve_calibration_bias=0.0,
    )
    item = BankItem("T(2,19)#r0", knot, 19.0, 0)
    detail = prediction_details(scientist, item, (10.0,))[0]
    score = expected_capped_scores(scientist, [item], (10.0,))[0]
    assert detail["assessment_scan_steps"] == 5
    assert detail["assessment_scan_coverage"] == 1.0
    assert detail["assessment_scoring_rule"] == "mean-head-sweep-v1"
    assert score == pytest.approx(detail["normalized_expected_capped_loss"])
