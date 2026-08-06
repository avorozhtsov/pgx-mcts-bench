"""Held-out monotone calibration for a scientist's solve probability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from pgx_mcts_bench.budget_gate import calibration
from pgx_mcts_bench.collaborative_scientists import _atomic_json, _sha256


def fit_solve_calibration(
    checkpoint: Path,
    validation_report: Path,
    output_checkpoint: Path,
    output_report: Path,
) -> dict[str, Any]:
    """Fit positive-scale Platt calibration without changing network weights."""
    report = json.loads(validation_report.read_text())
    attempts = report["trained"]["attempts"]
    scores = torch.tensor(
        [float(row["p_solve"]) for row in attempts], dtype=torch.float64
    ).clamp(1e-6, 1.0 - 1e-6)
    labels = torch.tensor(
        [float(row["solved"]) for row in attempts], dtype=torch.float64
    )
    if not bool((labels > 0.5).any()) or not bool((labels < 0.5).any()):
        raise ValueError("solve calibration requires both positive and negative labels")
    logits = torch.logit(scores)
    log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_scale, bias], max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        calibrated_logits = log_scale.exp() * logits + bias
        loss = F.binary_cross_entropy_with_logits(calibrated_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = float(log_scale.exp().detach())
    fitted_bias = float(bias.detach())
    calibrated = torch.sigmoid(scale * logits + fitted_bias).tolist()
    raw = calibration(scores.tolist(), labels.int().tolist())
    fitted = calibration(calibrated, labels.int().tolist())

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    calibration_payload = {
        "method": "positive-scale-platt-v1",
        "scale": scale,
        "bias": fitted_bias,
        "validation_report": str(validation_report.resolve()),
        "validation_report_sha256": _sha256(validation_report),
        "simulations": int(report["protocol"]["simulations"]),
    }
    payload["solve_calibration"] = calibration_payload
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_checkpoint)
    result = {
        "schema": "solve-calibration-v1",
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "output_checkpoint": str(output_checkpoint.resolve()),
        "validation_report": str(validation_report.resolve()),
        "scientist": report["scientist"],
        "attempts": len(attempts),
        "calibration": calibration_payload,
        "raw": raw,
        "fitted": fitted,
        "network_weights_changed": False,
    }
    _atomic_json(output_report, result)
    return result
