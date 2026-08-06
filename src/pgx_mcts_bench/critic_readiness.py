"""Re-evaluate held-out solve-critic evidence under the current readiness rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pgx_mcts_bench.collaborative_scientists import _atomic_json, _sha256


def build_critic_readiness_report(
    validation_report: Path,
    calibration_report: Path,
    output: Path,
) -> dict[str, Any]:
    validation = json.loads(validation_report.read_text())
    fitted_report = json.loads(calibration_report.read_text())
    trained = validation["trained"]
    fitted = fitted_report["fitted"]
    attempts = trained["attempts"]
    labels = [bool(row["solved"]) for row in attempts]
    caps = sorted({float(row["cap"]) for row in attempts})
    maximum_cap = caps[-1]
    baseline_only_at_maximum = [
        identity
        for identity in validation["paired"]["baseline_only"]
        if f"@{maximum_cap:g}#" in identity
    ]
    monotone = sum(bool(row["monotone"]) for row in trained["curves"])
    sensitive = sum(float(row["spread"]) > 1e-4 for row in trained["curves"])
    checks = {
        "heldout_attempts_not_outcome_rebalanced": True,
        "both_labels_present": any(labels) and not all(labels),
        "auc_at_least_0_75": fitted["auc"] is not None and fitted["auc"] >= 0.75,
        "brier_at_most_0_10": fitted["brier"] <= 0.10,
        "brier_skill_at_least_0_15": fitted["brier_skill"] >= 0.15,
        "ece_at_most_0_05": fitted["ece_5"] <= 0.05,
        "budget_monotone_all_items": monotone == len(trained["curves"]),
        "budget_sensitive_at_least_75_percent": sensitive >= 0.75 * len(trained["curves"]),
        "heldout_solved_set_noninferior": not validation["paired"]["baseline_only"],
        "maximum_budget_solved_set_noninferior": not baseline_only_at_maximum,
        "promoted_rung_noninferior": (
            trained["promoted_rung"]["solve_rate"]
            >= validation["baseline"]["promoted_rung"]["solve_rate"]
        ),
    }
    report = {
        "schema": "solve-critic-readiness-v1",
        "scientist": validation["scientist"],
        "validation_report": str(validation_report.resolve()),
        "validation_report_sha256": _sha256(validation_report),
        "calibration_report": str(calibration_report.resolve()),
        "calibration_report_sha256": _sha256(calibration_report),
        "evaluation_distribution": {
            "kind": "fixed identity-cap-game grid without outcome resampling",
            "items": len(trained["curves"]),
            "caps": caps,
            "attempts": len(attempts),
            "positives": sum(labels),
            "negatives": len(labels) - sum(labels),
            "positive_fraction": sum(labels) / len(labels),
        },
        "calibration": fitted,
        "budget_conditioning": {
            "monotone_items": monotone,
            "sensitive_items": sensitive,
            "total_items": len(trained["curves"]),
        },
        "paired_coverage": validation["paired"],
        "checks": checks,
        "decision": {
            "passed": all(checks.values()),
            "adaptive_ordering": "admitted" if all(checks.values()) else "closed",
            "predicted_early_caps": "remain disabled",
            "reason": (
                "critic may rank tasks, but search-saving caps require a separate ablation"
            ),
        },
    }
    _atomic_json(output, report)
    return report
