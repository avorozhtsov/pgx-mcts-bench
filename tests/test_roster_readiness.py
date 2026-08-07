from __future__ import annotations

from pathlib import Path

import torch

from pgx_mcts_bench.collaborative_scientists import _cheap_score, _compatible_table
from pgx_mcts_bench.roster_readiness import (
    _calibrate_checkpoint,
    _critic_checks,
    select_frontier_panels,
)


def test_frontier_panels_are_stratified_disjoint_and_respect_exclusions() -> None:
    excluded = {"3_1", "4_1", "5_1"}
    calibration, confirmation = select_frontier_panels(
        12,
        20,
        seed=7,
        excluded=excluded,
        frontier_pool_size=64,
    )

    calibration_ids = {item.id for item in calibration}
    confirmation_ids = {item.id for item in confirmation}
    assert len(calibration_ids) == 12
    assert len(confirmation_ids) == 20
    assert not calibration_ids & confirmation_ids
    assert not (calibration_ids | confirmation_ids) & excluded
    assert {item.difficulty_quartile for item in calibration} == {0, 1, 2, 3}
    assert {item.difficulty_quartile for item in confirmation} == {0, 1, 2, 3}
    eligible = sorted(
        (knot for knot in _compatible_table() if knot.name not in excluded),
        key=lambda knot: (_cheap_score(knot), knot.crossings, len(knot.word), knot.name),
    )
    assert calibration_ids | confirmation_ids <= {knot.name for knot in eligible[:64]}


def test_calibration_composes_with_existing_checkpoint_transform(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "calibrated.pt"
    torch.save(
        {
            "network": {"weight": torch.tensor([1.0])},
            "solve_calibration": {"scale": 2.0, "bias": 3.0},
        },
        source,
    )
    attempts = [
        {"p_solve": score, "solved": solved}
        for score, solved in ((0.1, False), (0.2, False), (0.8, True), (0.9, True))
    ]
    evaluation = {"evaluation": {"rows": {"x": {"attempts": attempts}}}}

    report = _calibrate_checkpoint("scientist", source, evaluation, output)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    layer = report["calibration"]["calibration_layer"]

    assert report["network_weights_unchanged"]
    torch.testing.assert_close(payload["network"]["weight"], torch.tensor([1.0]))
    assert payload["solve_calibration"]["scale"] == layer["scale"] * 2.0
    assert payload["solve_calibration"]["bias"] == layer["scale"] * 3.0 + layer["bias"]


def test_critic_checks_require_rank_and_calibration_quality() -> None:
    good = {
        "attempts": 40,
        "positives": 20,
        "auc": 0.8,
        "brier": 0.15,
        "brier_skill": 0.2,
        "ece_5": 0.05,
    }
    assert all(_critic_checks(good).values())
    assert not _critic_checks({**good, "auc": 0.7})["auc_at_least_0_75"]
