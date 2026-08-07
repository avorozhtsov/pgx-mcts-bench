"""Frontier-dose selection and critic qualification for a scientist roster."""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import load_scientist
from pgx_mcts_bench.budget_gate import calibration
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _bank_item,
    _bank_payload,
    _compatible_table,
    _json_hash,
    _sha256,
    prediction_details,
)
from pgx_mcts_bench.continual_learning_gate import (
    _evaluation_scientist,
    panel_evaluation,
)
from pgx_mcts_bench.solve_calibration import positive_scale_platt


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _excluded_ids(paths: tuple[Path, ...]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else payload.get("items", [])
        if not isinstance(rows, list):
            raise ValueError(f"excluded bank is not a row list: {path}")
        excluded.update(str(row["id"]) for row in rows)
    return excluded


def select_frontier_panels(
    calibration_size: int,
    confirmation_size: int,
    *,
    seed: int,
    excluded: set[str] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Select difficulty-stratified, identity-disjoint table-knot panels."""
    if calibration_size < 4 or confirmation_size < 4:
        raise ValueError("frontier panels need at least four identities each")
    compatible = sorted(_compatible_table(), key=lambda knot: (knot.crossings, knot.name))
    quartiles = np.array_split(np.asarray(compatible, dtype=object), 4)
    excluded = set(excluded or set())
    calibration_items = []
    confirmation_items = []
    for quartile, items in enumerate(quartiles):
        ranked = sorted(
            (item for item in items if item.name not in excluded),
            key=lambda item: hashlib.sha256(
                f"frontier-readiness:{seed}:{quartile}:{item.name}".encode()
            ).digest(),
        )
        calibration_take = calibration_size // 4 + int(quartile < calibration_size % 4)
        confirmation_take = confirmation_size // 4 + int(quartile < confirmation_size % 4)
        if len(ranked) < calibration_take + confirmation_take:
            raise ValueError(f"quartile {quartile} has too few unprotected identities")
        calibration_items.extend(_bank_item(item, quartile) for item in ranked[:calibration_take])
        confirmation_items.extend(
            _bank_item(item, quartile)
            for item in ranked[calibration_take : calibration_take + confirmation_take]
        )
    if {item.id for item in calibration_items} & {item.id for item in confirmation_items}:
        raise AssertionError("frontier panels are not identity-disjoint")
    return calibration_items, confirmation_items


def _evaluate_worker(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    scientist = load_scientist(
        payload["scientist"],
        Path(payload["checkpoint"]),
        seed=int(payload["seed"]),
        device=str(payload["device"]),
        simulations=int(payload["simulations"]),
        require_factorized=True,
        objective_budget_channel=True,
    )
    items = _bank_from_payload(payload["panel"])
    configured = _evaluation_scientist(scientist, int(payload["action_horizon"]))
    probabilities = {
        item.id: float(
            prediction_details(configured, item, (float(payload["ratio"]),))[0]["p_solve"]
        )
        for item in items
    }
    evaluation = panel_evaluation(
        scientist,
        items,
        ratio=float(payload["ratio"]),
        simulations=int(payload["simulations"]),
        attempts=int(payload["attempts"]),
        action_horizon=int(payload["action_horizon"]),
        seed=int(payload["seed"]),
        namespace=str(payload["namespace"]),
        objective_cap=None,
    )
    scores = []
    labels = []
    for identity, row in evaluation["rows"].items():
        row["p_solve"] = probabilities[identity]
        for attempt in row["attempts"]:
            attempt["p_solve"] = probabilities[identity]
            scores.append(probabilities[identity])
            labels.append(int(attempt["solved"]))
    return {
        "scientist": payload["scientist"],
        "checkpoint": str(Path(payload["checkpoint"]).resolve()),
        "checkpoint_sha256": _sha256(Path(payload["checkpoint"])),
        "simulations": int(payload["simulations"]),
        "evaluation": evaluation,
        "calibration": calibration(scores, labels),
    }


def _run_workers(jobs: list[tuple[Path, dict[str, Any]]], *, workers: int) -> list[dict[str, Any]]:
    rows = []
    pending = []
    for path, payload in jobs:
        if path.exists():
            rows.append(json.loads(path.read_text()))
        else:
            pending.append((path, payload))
    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {
                executor.submit(_evaluate_worker, payload): path for path, payload in pending
            }
            for future in as_completed(futures):
                row = future.result()
                _atomic_json(futures[future], row)
                rows.append(row)
    return sorted(rows, key=lambda row: str(row["scientist"]))


def _attempt_vectors(row: dict[str, Any]) -> tuple[list[float], list[int]]:
    attempts = [
        attempt for item in row["evaluation"]["rows"].values() for attempt in item["attempts"]
    ]
    return (
        [float(attempt["p_solve"]) for attempt in attempts],
        [int(attempt["solved"]) for attempt in attempts],
    )


def _calibrate_checkpoint(
    name: str,
    checkpoint: Path,
    evaluation: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    scores, labels = _attempt_vectors(evaluation)
    fitted = positive_scale_platt(scores, labels)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    old = payload.get("solve_calibration", {})
    old_scale = float(old.get("scale", 1.0))
    old_bias = float(old.get("bias", 0.0))
    layer_scale = float(fitted["scale"])
    layer_bias = float(fitted["bias"])
    combined = {
        "method": "composed-positive-scale-platt-v1",
        "scale": layer_scale * old_scale,
        "bias": layer_scale * old_bias + layer_bias,
        "calibration_layer": {"scale": layer_scale, "bias": layer_bias},
        "source": str(checkpoint.resolve()),
        "source_sha256": _sha256(checkpoint),
    }
    payload["solve_calibration"] = combined
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "scientist": name,
        "source_checkpoint": str(checkpoint.resolve()),
        "output_checkpoint": str(output.resolve()),
        "network_weights_unchanged": True,
        "calibration": combined,
        "raw": fitted["raw"],
        "fitted": fitted["fitted"],
    }


def _critic_checks(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "both_labels_present": 0 < metrics["positives"] < metrics["attempts"],
        "auc_at_least_0_75": metrics["auc"] is not None and metrics["auc"] >= 0.75,
        "brier_at_most_0_25": metrics["brier"] <= 0.25,
        "brier_skill_at_least_0_15": metrics["brier_skill"] >= 0.15,
        "ece_at_most_0_10": metrics["ece_5"] <= 0.10,
    }


def _scientist_only(solved_sets: dict[str, set[str]], name: str) -> list[str]:
    others = [values for other, values in solved_sets.items() if other != name]
    other_union = set.union(*others) if others else set()
    return sorted(solved_sets[name] - other_union)


def run_roster_readiness(
    checkpoints: dict[str, Path],
    output: Path,
    *,
    exclude_banks: tuple[Path, ...] = (),
    calibration_size: int = 12,
    confirmation_size: int = 24,
    attempts: int = 4,
    ratio: float = 1000.0,
    simulation_doses: tuple[int, ...] = (64, 128, 256),
    action_horizon: int = 128,
    minimum_coverage: float = 0.70,
    seed: int = 20261840,
    workers: int = 4,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    """Select a common dose, calibrate critics, and confirm a roster."""
    if not checkpoints:
        raise ValueError("at least one scientist checkpoint is required")
    if not simulation_doses or any(dose < 1 for dose in simulation_doses):
        raise ValueError("simulation doses must be positive")
    excluded = _excluded_ids(exclude_banks)
    calibration_panel, confirmation_panel = select_frontier_panels(
        calibration_size,
        confirmation_size,
        seed=seed,
        excluded=excluded,
    )
    protocol = {
        "schema": "frontier-roster-readiness-v1",
        "git_revision": _git_revision(),
        "checkpoints": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in checkpoints.items()
        },
        "excluded_banks": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in exclude_banks
        ],
        "excluded_identities": len(excluded),
        "calibration_size": calibration_size,
        "confirmation_size": confirmation_size,
        "attempts_per_representation": attempts,
        "ratio": ratio,
        "simulation_doses": list(simulation_doses),
        "action_horizon": action_horizon,
        "minimum_coverage": minimum_coverage,
        "objective_cap": None,
        "remaining_L_feature": "soft global remainder; never terminates search",
        "seed": seed,
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{manifest} exists; pass resume=True")
        if json.loads(manifest.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("roster-readiness resume protocol differs")
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume without {manifest}")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, protocol)
        _atomic_json(output / "calibration-panel.json", _bank_payload(calibration_panel))
        _atomic_json(output / "confirmation-panel.json", _bank_payload(confirmation_panel))

    calibration_by_dose = []
    selected = None
    selected_rows = None
    for dose in simulation_doses:
        jobs = []
        for scientist_index, (name, checkpoint) in enumerate(checkpoints.items()):
            path = output / "dose-calibration" / f"sim{dose}-{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            jobs.append(
                (
                    path,
                    {
                        "scientist": name,
                        "checkpoint": str(checkpoint),
                        "panel": _bank_payload(calibration_panel),
                        "ratio": ratio,
                        "simulations": dose,
                        "attempts": attempts,
                        "action_horizon": action_horizon,
                        "seed": seed + scientist_index * 100_000_003,
                        "namespace": f"dose-calibration-{dose}",
                        "device": device,
                    },
                )
            )
        rows = _run_workers(jobs, workers=workers)
        coverage = {
            row["scientist"]: row["evaluation"]["representation_solve_rate"] for row in rows
        }
        calibration_by_dose.append({"simulations": dose, "coverage": coverage})
        if all(value >= minimum_coverage for value in coverage.values()):
            selected = dose
            selected_rows = rows
            break

    if selected is None or selected_rows is None:
        report = {
            **protocol,
            "calibration_by_dose": calibration_by_dose,
            "decision": {
                "passed": False,
                "reason": "no common dose gave every scientist at least 70% coverage",
                "selected_simulations": None,
            },
        }
        _atomic_json(output / "report.json", report)
        return report

    calibration_reports = []
    calibrated_paths = {}
    by_name = {row["scientist"]: row for row in selected_rows}
    for name, checkpoint in checkpoints.items():
        calibrated_path = output / "checkpoints" / f"{name}-calibrated.pt"
        calibration_report = _calibrate_checkpoint(
            name,
            checkpoint,
            by_name[name],
            calibrated_path,
        )
        calibrated_paths[name] = calibrated_path
        calibration_reports.append(calibration_report)
    _atomic_json(output / "calibration.json", {"scientists": calibration_reports})

    confirmation_jobs = []
    for scientist_index, (name, checkpoint) in enumerate(calibrated_paths.items()):
        path = output / "confirmation" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        confirmation_jobs.append(
            (
                path,
                {
                    "scientist": name,
                    "checkpoint": str(checkpoint),
                    "panel": _bank_payload(confirmation_panel),
                    "ratio": ratio,
                    "simulations": selected,
                    "attempts": attempts,
                    "action_horizon": action_horizon,
                    "seed": seed + 500_000_000 + scientist_index * 100_000_003,
                    "namespace": "source-disjoint-confirmation",
                    "device": device,
                },
            )
        )
    confirmation = _run_workers(confirmation_jobs, workers=workers)
    roster_checks = {}
    solved_sets = {}
    for row in confirmation:
        checks = {
            "coverage_at_least_minimum": (
                row["evaluation"]["representation_solve_rate"] >= minimum_coverage
            ),
            **_critic_checks(row["calibration"]),
        }
        roster_checks[row["scientist"]] = {
            "passed": all(checks.values()),
            "checks": checks,
            "coverage": row["evaluation"]["representation_solve_rate"],
            "calibration": row["calibration"],
        }
        solved_sets[row["scientist"]] = set(row["evaluation"]["solved"])
    names = sorted(solved_sets)
    intersection = set.intersection(*(solved_sets[name] for name in names)) if names else set()
    union = set.union(*(solved_sets[name] for name in names)) if names else set()
    passed = all(row["passed"] for row in roster_checks.values())
    report = {
        **protocol,
        "selected_simulations": selected,
        "calibration_by_dose": calibration_by_dose,
        "calibrated_checkpoints": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in calibrated_paths.items()
        },
        "confirmation": roster_checks,
        "solved_sets": {
            "intersection": sorted(intersection),
            "union": sorted(union),
            "scientist_only": {name: _scientist_only(solved_sets, name) for name in names},
        },
        "decision": {
            "passed": passed,
            "selected_simulations": selected,
            "qualified": sorted(name for name, row in roster_checks.items() if row["passed"]),
            "rejected": sorted(name for name, row in roster_checks.items() if not row["passed"]),
            "next": (
                "run the paired no-sharing schedule gate"
                if passed
                else "use only the qualified subset and keep K4 sensitivity closed"
            ),
        },
    }
    _atomic_json(output / "report.json", report)
    return report
