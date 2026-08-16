"""Resumable paired 20-knot screening for the mastery-v3 candidates."""

from __future__ import annotations

import gzip
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from pgx_mcts_bench.adaptive_scientists import load_scientist
from pgx_mcts_bench.gpu_inference import PersistentInferenceCoordinator
from pgx_mcts_bench.mastery_v3_curriculum import audit_promotion, file_sha256
from pgx_mcts_bench.single_knot_mastery import (
    MasteryConfig,
    RepresentationNode,
    ScientistMasteryBackend,
)

CANDIDATES = ("cyclic-memory-deep-v3", "cyclic-graph-dual-v3")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _upper(row: dict[str, Any]) -> int:
    value = row.get("certified_unknotting_upper_bound", row.get("known_unknotting_number"))
    if value is None:
        raise ValueError(f"screening row {row.get('id')} has no registered upper bound")
    return int(value)


def _plain_source(path: Path, directory: Path) -> Path:
    if path.suffix != ".gz":
        return path
    target = directory / "parent-source.pt"
    if target.exists():
        return target
    with gzip.open(path, "rb") as source:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    torch.save(payload, target)
    return target


def _backend(
    name: str,
    checkpoint: Path,
    *,
    device: str,
    simulations: int,
    seed: int,
    coordinated: bool,
) -> tuple[ScientistMasteryBackend, PersistentInferenceCoordinator | None]:
    scientist = load_scientist(
        name,
        checkpoint,
        seed=seed,
        device=device,
        require_factorized=True,
        objective_budget_channel=True,
    )
    coordinator = (
        PersistentInferenceCoordinator(
            scientist.network,
            device=device,
            max_batch_rows=128,
            max_wait_ms=2.0,
            amp_dtype=torch.bfloat16,
        )
        if coordinated
        else None
    )
    config = MasteryConfig(
        knot_name="mastery-v3-screening",
        initial_target_u=0,
        ratio=1000.0,
        move_allowance=128,
        parallel_searches=32,
        torch_threads=8,
        attempts_per_node=1,
        simulations=simulations,
        train_steps_per_batch=1,
        seed=seed,
    )
    return (
        ScientistMasteryBackend(
            scientist,
            config,
            inference_coordinator=coordinator,
            inference_timeout_seconds=30.0 if coordinated else None,
        ),
        coordinator,
    )


def _run_phase(
    backend: ScientistMasteryBackend,
    rows: list[dict[str, Any]],
    seeds: list[int],
    journal: Path,
    *,
    simulations: int,
    strict: bool,
) -> list[dict[str, Any]]:
    items = json.loads(journal.read_text()) if journal.exists() else []
    completed = {(str(item["representation_id"]), int(item["seed"])) for item in items}
    for seed in seeds:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (str(row["representation_id"]), int(seed))
            if key in completed:
                continue
            upper = _upper(row)
            target = max(0, upper - 1) if strict else upper
            grouped[target].append(row)
        for target, group in sorted(grouped.items()):
            nodes = [
                RepresentationNode.create(
                    root_id=f"screen:{row['representation_id']}",
                    word=row["word"],
                    strands=int(row["strands"]),
                    crossing_distance=0,
                    semantic_path=(),
                    provenance={"mastery_v3_screening": True, "strict": strict},
                )
                for row in group
            ]
            lane_seeds = [
                int(seed) + int(str(row["representation_id"])[-8:], 16) % 10_000_000
                for row in group
            ]
            started = time.perf_counter()
            try:
                results = backend.attempt_batch(nodes, target, lane_seeds, simulations=simulations)
                error = None
            except Exception as exception:  # fail closed, but durably account every denominator
                results = [None] * len(group)
                error = f"{type(exception).__name__}: {exception}"
            elapsed = time.perf_counter() - started
            for row, lane_seed, result in zip(group, lane_seeds, results, strict=True):
                upper = _upper(row)
                solved = bool(result is not None and result.solved)
                crossing_changes = None if result is None else result.crossing_changes
                moves = None if result is None else result.moves
                capped = (
                    1000 * int(crossing_changes) + int(moves)
                    if solved and crossing_changes is not None and moves is not None
                    else 1000 * upper + 128
                )
                item = {
                    "representation_id": str(row["representation_id"]),
                    "identity": str(row.get("name") or row.get("id")),
                    "strands": int(row["strands"]),
                    "seed": int(seed),
                    "lane_seed": lane_seed,
                    "current_upper": upper,
                    "attempt_target_u": target,
                    "simulations": simulations,
                    "strict_phase": strict,
                    "solved": solved,
                    "crossing_changes": crossing_changes,
                    "moves": moves,
                    "l1000_capped": capped,
                    "reason": error if result is None else result.reason,
                    "semantic_actions": [] if result is None else list(result.semantic_actions),
                    "replay_verified": solved,
                    "scheduled_network_evaluations": (
                        0 if result is None else result.scheduled_network_evaluations
                    ),
                    "wall_seconds": elapsed / len(group),
                    "capacity_exception": error is not None,
                }
                items.append(item)
                completed.add((item["representation_id"], item["seed"]))
            _atomic_json(journal, items)
    return items


def _retention(
    control: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> tuple[float | None, dict[str, float | None]]:
    candidate_by_key = {
        (item["representation_id"], item["seed"]): item for item in candidate
    }
    parent_solved = [item for item in control if item["solved"]]

    def rate(rows: list[dict[str, Any]]) -> float | None:
        if not rows:
            return None
        return sum(
            bool(candidate_by_key[(item["representation_id"], item["seed"])]["solved"])
            for item in rows
        ) / len(rows)

    by_strand = {
        str(strands): rate([item for item in parent_solved if int(item["strands"]) == strands])
        for strands in range(6, 13)
    }
    return rate(parent_solved), by_strand


def run_screening(
    curriculum_path: Path,
    source_checkpoint: Path,
    deep_checkpoint: Path,
    graph_checkpoint: Path,
    output: Path,
    *,
    device: str = "cuda",
    simulations: int = 128,
    benchmark_simulations: int = 32,
) -> dict[str, Any]:
    curriculum = json.loads(curriculum_path.read_text())
    rows = list(curriculum["stages"]["screening"]["rows"])
    seeds = [int(seed) for seed in curriculum["paired_screening_seeds"]]
    output.mkdir(parents=True, exist_ok=True)
    plain_parent = _plain_source(source_checkpoint, output)
    torch.set_num_threads(8)

    control_backend, control_coordinator = _backend(
        "cyclic-memory-12",
        plain_parent,
        device=device,
        simulations=simulations,
        seed=seeds[0],
        coordinated=True,
    )
    try:
        control_retention = _run_phase(
            control_backend,
            rows,
            seeds,
            output / "control-retention.json",
            simulations=simulations,
            strict=False,
        )
    finally:
        if control_coordinator is not None:
            control_coordinator.close()

    reports: dict[str, Any] = {}
    for name, checkpoint in zip(CANDIDATES, (deep_checkpoint, graph_checkpoint), strict=True):
        backend, coordinator = _backend(
            name,
            checkpoint,
            device=device,
            simulations=simulations,
            seed=seeds[0],
            coordinated=True,
        )
        try:
            strict_items = _run_phase(
                backend,
                rows,
                seeds,
                output / f"{name}-strict.json",
                simulations=simulations,
                strict=True,
            )
            retention_items = _run_phase(
                backend,
                rows,
                seeds,
                output / f"{name}-retention.json",
                simulations=simulations,
                strict=False,
            )
            gpu_started = time.perf_counter()
            _run_phase(
                backend,
                rows[:1],
                [seeds[0] + 99_000_000],
                output / f"{name}-gpu-benchmark.json",
                simulations=benchmark_simulations,
                strict=False,
            )
            gpu_seconds = time.perf_counter() - gpu_started
            coordinator_stats = coordinator.snapshot() if coordinator is not None else {}
        finally:
            if coordinator is not None:
                coordinator.close()

        cpu_backend, _ = _backend(
            name,
            checkpoint,
            device="cpu",
            simulations=benchmark_simulations,
            seed=seeds[0],
            coordinated=False,
        )
        cpu_started = time.perf_counter()
        _run_phase(
            cpu_backend,
            rows[:1],
            [seeds[0] + 99_000_000],
            output / f"{name}-cpu-benchmark.json",
            simulations=benchmark_simulations,
            strict=False,
        )
        cpu_seconds = time.perf_counter() - cpu_started
        retention, by_strand = _retention(control_retention, retention_items)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        migration = payload.get("mastery_v3_migration") or {}
        reports[name] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "migration_max_abs_error": max(
                (float(value) for value in (migration.get("max_abs_difference") or {}).values()),
                default=None,
            ),
            "migration_tolerance": migration.get("tolerance", 1e-6),
            "screening_items": strict_items,
            "screening_retention": retention,
            "retention_by_strand": by_strand,
            "capacity_exceptions": sum(bool(item["capacity_exception"]) for item in strict_items),
            "strict_successes": sum(bool(item["solved"]) for item in strict_items),
            "genuine_improvements": sum(
                bool(item["solved"])
                and item["crossing_changes"] is not None
                and int(item["crossing_changes"]) < int(item["current_upper"])
                for item in strict_items
            ),
            "capped_l1000": sum(float(item["l1000_capped"]) for item in strict_items)
            / len(strict_items),
            "accelerator_seconds": sum(float(item["wall_seconds"]) for item in strict_items),
            "strict_evidence_per_accelerator_second": sum(
                bool(item["solved"]) for item in strict_items
            )
            / max(1e-9, sum(float(item["wall_seconds"]) for item in strict_items)),
            "gpu_end_to_end_speedup": cpu_seconds / max(gpu_seconds, 1e-9),
            "benchmark": {
                "simulations": benchmark_simulations,
                "cpu_seconds": cpu_seconds,
                "gpu_seconds": gpu_seconds,
            },
            "coordinator": coordinator_stats,
        }
        _atomic_json(output / f"{name}-report.json", reports[name])

    report = {
        "schema": "mastery-v3-paired-screening-v1",
        "curriculum": str(curriculum_path.resolve()),
        "curriculum_sha256": file_sha256(curriculum_path),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "simulations": simulations,
        "device": device,
        "arms": reports,
        "full_240_launched": False,
    }
    report["promotion_audit"] = audit_promotion(curriculum, report)
    _atomic_json(output / "screening-report.json", report)
    _atomic_json(output / "promotion-audit.json", report["promotion_audit"])
    return report
