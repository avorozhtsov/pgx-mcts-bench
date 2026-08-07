"""Held-out monotone calibration for a scientist's solve probability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from pgx_mcts_bench.budget_gate import calibration
from pgx_mcts_bench.collaborative_scientists import _atomic_json, _sha256


def positive_scale_platt(scores: list[float], labels: list[int]) -> dict[str, Any]:
    """Fit a monotone affine transform in logit space."""
    score_tensor = torch.tensor(scores, dtype=torch.float64).clamp(1e-6, 1.0 - 1e-6)
    label_tensor = torch.tensor(labels, dtype=torch.float64)
    if not bool((label_tensor > 0.5).any()) or not bool((label_tensor < 0.5).any()):
        raise ValueError("solve calibration requires both positive and negative labels")
    logits = torch.logit(score_tensor)
    log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_scale, bias], max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        calibrated_logits = log_scale.exp() * logits + bias
        loss = F.binary_cross_entropy_with_logits(calibrated_logits, label_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = float(log_scale.exp().detach())
    fitted_bias = float(bias.detach())
    calibrated = torch.sigmoid(scale * logits + fitted_bias).tolist()
    return {
        "scale": scale,
        "bias": fitted_bias,
        "scores": calibrated,
        "raw": calibration(score_tensor.tolist(), labels),
        "fitted": calibration(calibrated, labels),
    }


def fit_solve_calibration(
    checkpoint: Path,
    validation_report: Path,
    output_checkpoint: Path,
    output_report: Path,
) -> dict[str, Any]:
    """Fit positive-scale Platt calibration without changing network weights."""
    report = json.loads(validation_report.read_text())
    attempts = report["trained"]["attempts"]
    scores = [float(row["p_solve"]) for row in attempts]
    labels = [int(row["solved"]) for row in attempts]
    fitted_platt = positive_scale_platt(scores, labels)
    scale = float(fitted_platt["scale"])
    fitted_bias = float(fitted_platt["bias"])

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
        "raw": fitted_platt["raw"],
        "fitted": fitted_platt["fitted"],
        "network_weights_changed": False,
    }
    _atomic_json(output_report, result)
    return result
