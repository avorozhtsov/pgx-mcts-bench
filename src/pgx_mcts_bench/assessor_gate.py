"""Held-out qualification gate for neural adaptive-schedule assessors.

The scheduler may use a scientist's ordering only after an identity-disjoint
panel demonstrates that its post-scan predictions rank genuinely easier tasks.
Evidence is aggregated by representation, so repeated attempts improve the
label rather than multiplying one task's statistical weight.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pgx_mcts_bench.budget_gate import roc_auc

EVIDENCE_SCHEMA = "local-window-assessor-evidence-v1"
GATE_SCHEMA = "adaptive-schedule-assessor-gate-v1"
SCORING_RULE = "mean-head-sweep-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    left_rank = _ranks(left)
    right_rank = _ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, float | str | int | bool]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        representation = str(row["representation"])
        ratio = float(row["ratio"])
        grouped[(representation, ratio)].append(row)
    aggregated = []
    for (representation, ratio), attempts in sorted(grouped.items()):
        failure_costs = {float(row["failure_cost"]) for row in attempts}
        if len(failure_costs) != 1:
            raise ValueError(f"{representation}@{ratio} has inconsistent failure costs")
        solved_attempts = [row for row in attempts if bool(row["solved"])]
        actual_cost = min(
            (
                ratio * float(row["actual_crossing_changes"])
                + float(row["actual_moves"])
                for row in solved_attempts
            ),
            default=failure_costs.pop(),
        )
        probability = float(np.mean([float(row["p_solve"]) for row in attempts]))
        predicted_cost = float(
            np.mean(
                [
                    ratio * float(row["predicted_crossing_changes"])
                    + float(row["predicted_moves"])
                    for row in attempts
                ]
            )
        )
        failure = float(attempts[0]["failure_cost"])
        aggregated.append(
            {
                "representation": representation,
                "ratio": ratio,
                "attempts": len({int(row["attempt"]) for row in attempts}),
                "p_solve": probability,
                "predicted_expected_cost": probability * predicted_cost
                + (1.0 - probability) * failure,
                "solved": bool(solved_attempts),
                "actual_capped_cost": actual_cost,
                "scan_coverage": float(
                    np.mean([float(row["scan_coverage"]) for row in attempts])
                ),
            }
        )
    return aggregated


def _objective_metrics(
    rows: list[dict[str, float | str | int | bool]],
    *,
    minimum_attempts: int,
    minimum_scan_coverage: float,
    minimum_auc: float,
    maximum_brier_ratio: float,
    minimum_top_quartile_solve_rate: float,
    minimum_cost_spearman: float,
) -> dict[str, Any]:
    labels = np.asarray([bool(row["solved"]) for row in rows], dtype=np.float64)
    probabilities = np.asarray([float(row["p_solve"]) for row in rows])
    predicted = np.asarray([float(row["predicted_expected_cost"]) for row in rows])
    actual = np.asarray([float(row["actual_capped_cost"]) for row in rows])
    prevalence = float(labels.mean()) if len(labels) else 0.0
    brier = float(np.mean((probabilities - labels) ** 2)) if len(labels) else float("inf")
    baseline_brier = prevalence * (1.0 - prevalence)
    brier_ratio = brier / baseline_brier if baseline_brier > 0 else float("inf")
    auc = roc_auc(probabilities.tolist(), labels.astype(np.int32).tolist())
    quartile = max(1, int(np.ceil(len(rows) / 4)))
    easiest = np.argsort(predicted, kind="stable")[:quartile]
    top_rate = float(labels[easiest].mean()) if len(labels) else 0.0
    spearman = _spearman(predicted, actual)
    checks = {
        "attempts_per_representation": min(int(row["attempts"]) for row in rows)
        >= minimum_attempts,
        "scan_coverage": min(float(row["scan_coverage"]) for row in rows)
        >= minimum_scan_coverage,
        "positive_and_negative_labels": 0 < int(labels.sum()) < len(labels),
        "solve_auc": bool(auc is not None and auc >= minimum_auc),
        "brier_skill": brier_ratio <= maximum_brier_ratio,
        "top_quartile_solve_rate": top_rate >= minimum_top_quartile_solve_rate,
        "cost_rank": bool(np.isfinite(spearman) and spearman >= minimum_cost_spearman),
    }
    return {
        "representations": len(rows),
        "positive_representations": int(labels.sum()),
        "negative_representations": int(len(labels) - labels.sum()),
        "minimum_attempts_per_representation": min(int(row["attempts"]) for row in rows),
        "minimum_scan_coverage": min(float(row["scan_coverage"]) for row in rows),
        "solve_rate": prevalence,
        "solve_auc": auc,
        "brier": brier,
        "constant_prevalence_brier": baseline_brier,
        "brier_ratio": brier_ratio,
        "top_quartile_solve_rate": top_rate,
        "cost_spearman": spearman,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_assessor_gate(
    evidence_path: Path,
    output: Path,
    *,
    minimum_representations: int = 100,
    minimum_attempts: int = 2,
    minimum_scan_steps: int = 5,
    minimum_scan_coverage: float = 0.90,
    minimum_auc: float = 0.70,
    maximum_brier_ratio: float = 1.0,
    minimum_top_quartile_solve_rate: float = 0.70,
    minimum_cost_spearman: float = 0.20,
) -> dict[str, Any]:
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError(f"expected {EVIDENCE_SCHEMA} evidence")
    scan = evidence.get("scan_protocol", {})
    scan_checks = {
        "exact_scoring_rule": str(scan.get("scoring_rule", "")) == SCORING_RULE,
        "post_scan_predictions": bool(scan.get("predictions_after_scan", False)),
        "native_only_scan": bool(scan.get("native_only", False)),
        "no_semantic_edits": int(scan.get("semantic_actions", -1)) == 0,
        "scan_steps": int(scan.get("steps", 0)) >= minimum_scan_steps,
        "identity_disjoint": bool(evidence.get("identity_disjoint_from_training", False)),
    }
    scientists = []
    for scientist in evidence.get("scientists", []):
        rows = _aggregate(list(scientist.get("rows", [])))
        ratios = sorted({float(row["ratio"]) for row in rows})
        objectives = {}
        for ratio in ratios:
            objective_rows = [row for row in rows if float(row["ratio"]) == ratio]
            objectives[str(ratio)] = _objective_metrics(
                objective_rows,
                minimum_attempts=minimum_attempts,
                minimum_scan_coverage=minimum_scan_coverage,
                minimum_auc=minimum_auc,
                maximum_brier_ratio=maximum_brier_ratio,
                minimum_top_quartile_solve_rate=minimum_top_quartile_solve_rate,
                minimum_cost_spearman=minimum_cost_spearman,
            )
        representation_count = len({str(row["representation"]) for row in rows})
        checks = {
            "checkpoint_hash_present": len(str(scientist.get("checkpoint_sha256", ""))) == 64,
            "representation_count": representation_count >= minimum_representations,
            "objectives_present": bool(objectives),
            "every_objective_passed": bool(objectives)
            and all(row["passed"] for row in objectives.values()),
        }
        scientists.append(
            {
                "name": str(scientist["name"]),
                "checkpoint_sha256": str(scientist.get("checkpoint_sha256", "")),
                "representations": representation_count,
                "objectives": objectives,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    thresholds = {
        "minimum_representations": minimum_representations,
        "minimum_attempts_per_representation": minimum_attempts,
        "minimum_scan_steps": minimum_scan_steps,
        "minimum_scan_coverage": minimum_scan_coverage,
        "minimum_solve_auc": minimum_auc,
        "maximum_brier_ratio": maximum_brier_ratio,
        "minimum_top_quartile_solve_rate": minimum_top_quartile_solve_rate,
        "minimum_cost_spearman": minimum_cost_spearman,
    }
    report = {
        "schema": GATE_SCHEMA,
        "source_evidence": {
            "path": str(evidence_path.resolve()),
            "sha256": _sha256(evidence_path),
        },
        "thresholds": thresholds,
        "scan_protocol": scan,
        "scan_checks": scan_checks,
        "scientists": scientists,
        "passed": bool(scientists)
        and all(scan_checks.values())
        and all(scientist["passed"] for scientist in scientists),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def validate_assessor_gate(
    gate_path: Path | None,
    checkpoints: dict[str, Path],
) -> dict[str, Any]:
    if gate_path is None:
        raise ValueError("adaptive scheduling requires --assessor-gate")
    report = json.loads(gate_path.read_text())
    if report.get("schema") != GATE_SCHEMA or not report.get("passed", False):
        raise ValueError("adaptive scheduling requires a passed assessor gate")
    certified = {
        str(row["name"]): str(row["checkpoint_sha256"])
        for row in report.get("scientists", [])
        if row.get("passed", False)
    }
    expected = {name: _sha256(path) for name, path in checkpoints.items()}
    if certified != expected:
        missing = sorted(set(expected) - set(certified))
        extra = sorted(set(certified) - set(expected))
        mismatched = sorted(
            name for name in set(expected) & set(certified) if expected[name] != certified[name]
        )
        raise ValueError(
            "assessor gate does not certify the exact adaptive roster: "
            f"missing={missing}, extra={extra}, hash_mismatch={mismatched}"
        )
    return report


def collect_assessor_evidence(
    checkpoints: dict[str, Path],
    output: Path,
    *,
    exclude_banks: tuple[Path, ...] = (),
    representation_count: int = 100,
    frontier_pool_size: int = 400,
    attempts: int = 2,
    ratios: tuple[float, ...] = (10.0, 1000.0),
    simulations: int = 128,
    action_horizon: int = 128,
    seed: int = 20261880,
    workers: int = 4,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    """Collect paired post-scan evidence and build its checkpoint-bound gate."""
    from pgx_mcts_bench.collaborative_scientists import (
        _atomic_json,
        _bank_payload,
        _json_hash,
    )
    from pgx_mcts_bench.roster_readiness import (
        _excluded_ids,
        _run_workers,
        select_frontier_panels,
    )

    if not checkpoints:
        raise ValueError("at least one scientist checkpoint is required")
    if representation_count < 4 or attempts < 1 or simulations < 1:
        raise ValueError("invalid assessor evidence allocation")
    if not ratios:
        raise ValueError("at least one objective ratio is required")
    excluded = _excluded_ids(exclude_banks)
    panel, _reserved = select_frontier_panels(
        representation_count,
        4,
        seed=seed,
        excluded=excluded,
        frontier_pool_size=frontier_pool_size,
    )
    from pgx_mcts_bench.foundation_pretraining import source_provenance

    protocol = {
        "schema": "adaptive-schedule-assessor-evidence-run-v1",
        "source_provenance": source_provenance(),
        "checkpoints": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in checkpoints.items()
        },
        "excluded_banks": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in exclude_banks
        ],
        "representation_count": representation_count,
        "frontier_pool_size": frontier_pool_size,
        "attempts_per_representation": attempts,
        "ratios": list(ratios),
        "simulations": simulations,
        "action_horizon": action_horizon,
        "seed": seed,
        "device": device,
        "panel_sha256": _json_hash(_bank_payload(panel)),
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; pass resume=True")
        previous = json.loads(manifest_path.read_text())
        if previous.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("assessor evidence resume protocol differs")
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume without {manifest_path}")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest_path, protocol)
        _atomic_json(output / "panel.json", _bank_payload(panel))

    jobs = []
    for name, checkpoint in checkpoints.items():
        for ratio in ratios:
            item_path = output / "items" / f"{name}-L{ratio:g}.json"
            item_path.parent.mkdir(parents=True, exist_ok=True)
            jobs.append(
                (
                    item_path,
                    {
                        "scientist": name,
                        "checkpoint": str(checkpoint),
                        "panel": _bank_payload(panel),
                        "ratio": ratio,
                        "simulations": simulations,
                        "attempts": attempts,
                        "action_horizon": action_horizon,
                        "seed": seed,
                        "namespace": f"assessor-evidence-L{ratio:g}",
                        "device": device,
                    },
                )
            )
    results = _run_workers(jobs, workers=workers)
    by_scientist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        name = str(result["scientist"])
        evaluation = result["evaluation"]
        ratio = float(next(iter(evaluation["rows"].values()))["prediction"]["ratio"])
        failure = float(evaluation["failure_cap"])
        for representation, row in evaluation["rows"].items():
            prediction = row["prediction"]
            for attempt_index, attempt in enumerate(row["attempts"]):
                by_scientist[name].append(
                    {
                        "representation": representation,
                        "ratio": ratio,
                        "attempt": attempt_index,
                        "p_solve": float(prediction["p_solve"]),
                        "predicted_crossing_changes": float(
                            prediction["predicted_crossing_changes"]
                        ),
                        "predicted_moves": float(prediction["predicted_moves"]),
                        "scan_coverage": float(prediction["assessment_scan_coverage"]),
                        "solved": bool(attempt["solved"]),
                        "actual_crossing_changes": attempt.get("crossing_changes"),
                        "actual_moves": attempt.get("moves"),
                        "failure_cost": failure,
                        "seed": int(attempt["seed"]),
                    }
                )
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "identity_disjoint_from_training": True,
        "identity_disjoint_basis": (
            "external table-representation IDs selected after foundation pretraining; "
            "protected banks excluded before sampling"
        ),
        "scan_protocol": {
            "scoring_rule": SCORING_RULE,
            "predictions_after_scan": True,
            "native_only": True,
            "semantic_actions": 0,
            "steps": 5,
        },
        "scientists": [
            {
                "name": name,
                "checkpoint_sha256": _sha256(checkpoints[name]),
                "rows": by_scientist[name],
            }
            for name in checkpoints
        ],
    }
    evidence_path = output / "evidence.json"
    _atomic_json(evidence_path, evidence)
    gate = build_assessor_gate(
        evidence_path,
        output / "gate.json",
        minimum_representations=representation_count,
        minimum_attempts=attempts,
    )
    report = {
        **protocol,
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": _sha256(evidence_path),
        "gate_path": str((output / "gate.json").resolve()),
        "gate_passed": bool(gate["passed"]),
        "scientists": gate["scientists"],
    }
    _atomic_json(output / "report.json", report)
    return report
