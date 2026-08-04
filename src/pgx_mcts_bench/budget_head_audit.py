"""Audit whether a migrated solve head actually conditions on objective budget."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from pgx_mcts_bench.adaptive_scientists import FixedWordGame, load_scientist
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _observation_tensor,
    _sha256,
)


def budget_head_accepted(
    *,
    items: int,
    monotone_items: int,
    sensitive_items: int,
    brier: float | None,
    negative_positions: int | None,
) -> bool:
    return (
        monotone_items == items
        and sensitive_items >= math.ceil(0.8 * items)
        and brier is not None
        and brier <= 0.25
        and negative_positions is not None
        and negative_positions > 0
    )


def audit_budget_head(
    checkpoint: Path,
    bank: Path,
    output: Path,
    *,
    outcomes: Path | None = None,
    training_metrics: Path | None = None,
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    caps: tuple[float, ...] = (16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 704.0),
    device: str = "cpu",
) -> dict[str, Any]:
    items = _bank_from_payload(json.loads(bank.read_text()))
    model = load_scientist(
        scientist,
        checkpoint,
        seed=0,
        device=device,
        require_factorized=True,
        objective_budget_channel=True,
    )
    observations = []
    channel_values = []
    for item in items:
        item_observations = [
            FixedWordGame(model.game, item.knot, ratio, objective_cap=cap).reset(0).observation
            for cap in caps
        ]
        observations.extend(item_observations)
        channel_values.append([float(observation[0, 0, -1]) for observation in item_observations])
    tensor = _observation_tensor(observations, torch.device(device))
    with torch.inference_mode():
        auxiliary = model.network.eval().forward_with_auxiliary(tensor)[2]
        probability = auxiliary[0].sigmoid().mean(dim=1).reshape(len(items), len(caps))
    probabilities = probability.cpu().tolist()
    tolerance = 1e-7
    rows = []
    for item, channels, values in zip(items, channel_values, probabilities, strict=True):
        monotone = all(
            values[index + 1] + tolerance >= values[index] for index in range(len(values) - 1)
        )
        spread = max(values) - min(values)
        rows.append(
            {
                "item": item.id,
                "caps": list(caps),
                "channel_values": channels,
                "p_solve": values,
                "monotone_nondecreasing": monotone,
                "probability_spread": spread,
                "strictly_budget_sensitive": spread > 1e-4,
            }
        )

    calibration = None
    if outcomes is not None:
        labels_by_item = {
            row["item"]: float(row["solved"])
            for path in outcomes.glob("*.json")
            for row in [json.loads(path.read_text())]
        }
        pairs = [
            (values[-1], labels_by_item[item.id])
            for item, values in zip(items, probabilities, strict=True)
            if item.id in labels_by_item
        ]
        calibration = {
            "labels": len(pairs),
            "positive_labels": int(sum(label for _, label in pairs)),
            "mean_p_solve": sum(value for value, _ in pairs) / len(pairs),
            "observed_solve_rate": sum(label for _, label in pairs) / len(pairs),
            "brier": sum((value - label) ** 2 for value, label in pairs) / len(pairs),
            "log_loss": -sum(
                label * math.log(max(value, 1e-9))
                + (1.0 - label) * math.log(max(1.0 - value, 1e-9))
                for value, label in pairs
            )
            / len(pairs),
        }

    label_balance = None
    if training_metrics is not None:
        metrics = json.loads(training_metrics.read_text())
        factorized = metrics["post"]["factorized"]
        eligible = int(factorized["eligible_positions"])
        solved = int(factorized["solved_positions"])
        label_balance = {
            "eligible_positions": eligible,
            "positive_positions": solved,
            "negative_positions": eligible - solved,
            "positive_fraction": solved / eligible if eligible else None,
        }

    sensitive = sum(row["strictly_budget_sensitive"] for row in rows)
    monotone = sum(row["monotone_nondecreasing"] for row in rows)
    accepted = budget_head_accepted(
        items=len(rows),
        monotone_items=monotone,
        sensitive_items=sensitive,
        brier=calibration["brier"] if calibration is not None else None,
        negative_positions=(
            label_balance["negative_positions"] if label_balance is not None else None
        ),
    )
    report = {
        "schema": "budget-head-audit-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "bank": str(bank.resolve()),
        "scientist": scientist,
        "ratio": ratio,
        "caps": list(caps),
        "rows": rows,
        "calibration": calibration,
        "training_label_balance": label_balance,
        "summary": {
            "items": len(rows),
            "monotone_nondecreasing_items": monotone,
            "strictly_budget_sensitive_items": sensitive,
            "maximum_probability_spread": max(row["probability_spread"] for row in rows),
            "accepted": accepted,
            "decision": (
                "budget-conditioned caps may be tested" if accepted else "do not use p_solve caps"
            ),
        },
    }
    _atomic_json(output, report)
    return report
