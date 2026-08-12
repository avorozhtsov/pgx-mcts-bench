"""Resumable adaptive foundation pretraining for the semantic-v1 roster."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pgx_mcts_bench.strand_architecture_gate import EARLY_MIXED_STRAND_STAGES


def source_provenance(repository: Path | None = None) -> dict[str, Any]:
    """Hash the executable source tree and record its Git base/status.

    Foundation pretraining may legitimately run before a local commit is made,
    but the paper must still identify the exact code that produced a checkpoint.
    Tests are excluded from the executable hash; package code, scripts, and the
    dependency lock/config are included.
    """
    root = repository or Path(__file__).resolve().parents[2]
    files = [
        path
        for directory in (root / "src", root / "scripts")
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    ]
    files.extend(path for path in (root / "pyproject.toml", root / "uv.lock") if path.exists())
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "repository": str(root),
        "base_commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "executable_source_sha256": digest.hexdigest(),
        "executable_files": len(files),
    }


def next_dose(
    *,
    promoted: bool,
    solve_rate: float,
    native_index: int,
    simulation_index: int,
    native_levels: tuple[int, ...],
    simulation_levels: tuple[int, ...],
    evaluation_target: float,
) -> tuple[int, int, bool]:
    """Return the next adaptive dose and whether every useful dose is exhausted."""
    if promoted:
        return native_index, simulation_index, False
    next_native = min(native_index + 1, len(native_levels) - 1)
    next_simulation = simulation_index
    if solve_rate < evaluation_target:
        next_simulation = min(simulation_index + 1, len(simulation_levels) - 1)
    changed = next_native != native_index or next_simulation != simulation_index
    return next_native, next_simulation, not changed


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    from pgx_mcts_bench import ladder
    from pgx_mcts_bench.worker_runtime import worker_init

    worker_init()
    output = Path(payload["output"]) / payload["candidate"] / f"seed-{payload['seed']}"
    result_path = output / "foundation-result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    by_name = {candidate.name: candidate for candidate in ladder.foundation_arms()}
    base_candidate = by_name[payload["candidate"]]
    native_levels = tuple(int(value) for value in payload["native_levels"])
    simulation_levels = tuple(int(value) for value in payload["simulation_levels"])
    stages = EARLY_MIXED_STRAND_STAGES[: int(payload["stage_limit"])]
    state_path = output / "foundation-state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {
            "stage_index": 0,
            "native_index": 0,
            "simulation_index": 0,
            "retry_capped": False,
            "attempts": [],
        }
    historical_stages = ladder.STAGES
    ladder.STAGES = list(stages)
    final_result = None
    try:
        while int(state["stage_index"]) < len(stages):
            stage_index = int(state["stage_index"])
            candidate = replace(
                base_candidate,
                simulations=simulation_levels[int(state["simulation_index"])],
            )
            _atomic_json(state_path, state)
            final_result = ladder.run_ladder(
                candidate,
                seed=int(payload["seed"]),
                device=str(payload["device"]),
                checkpoint_dir=output / "checkpoints",
                max_iterations_per_stage=native_levels[int(state["native_index"])],
                selfplay_games=int(payload["selfplay_games"]),
                checkpoint_every=1,
                eval_every=int(payload["eval_every"]),
                eval_games=int(payload["eval_games"]),
                promote_at=float(payload["promote_at"]),
                mix_decay=0.5,
                max_consecutive_caps=len(stages) + 1,
                plateau_on_known_objective=False,
                rehearsal_games_per_cleared_stage=int(payload["initial_old_cycles"]),
                adaptive_rehearsal=True,
                rehearsal_target=float(payload["retention_target"]),
                max_rehearsal_games_per_stage=int(payload["max_old_cycles"]),
                stop_after=stage_index,
                retro_games=int(payload["retro_games"]),
                balanced_replay=True,
                policy_value_success_only=True,
                retry_capped_on_resume=bool(state["retry_capped"]),
                log=lambda *args, **kwargs: None,
            )
            stage = final_result.stages[-1]
            attempt = {
                "stage_index": stage_index,
                "source": stage.source,
                "scramble": stage.scramble,
                "F_native": native_levels[int(state["native_index"])],
                "simulations": simulation_levels[int(state["simulation_index"])],
                "solve_rate": stage.solve_rate,
                "crossings": stage.crossings,
                "optimal_crossings": stage.optimal_crossings,
                "promoted": stage.promoted,
                "reason": stage.reason,
                "iterations": stage.iterations,
                "certified_value_stats": stage.certified_value_stats,
            }
            state["attempts"].append(attempt)
            if stage.promoted:
                state["stage_index"] = stage_index + 1
                state["retry_capped"] = False
            else:
                native_index, simulation_index, exhausted = next_dose(
                    promoted=False,
                    solve_rate=float(stage.solve_rate),
                    native_index=int(state["native_index"]),
                    simulation_index=int(state["simulation_index"]),
                    native_levels=native_levels,
                    simulation_levels=simulation_levels,
                    evaluation_target=float(payload["evaluation_target"]),
                )
                state["native_index"] = native_index
                state["simulation_index"] = simulation_index
                state["retry_capped"] = not exhausted
                if exhausted:
                    state["stage_index"] = stage_index + 1
            _atomic_json(state_path, state)
    finally:
        ladder.STAGES = historical_stages

    promoted_checkpoint = output / "checkpoints" / f"{base_candidate.name}.pt"
    progress_checkpoint = output / "checkpoints" / base_candidate.name / "progress.pt"
    if final_result is None:
        if not promoted_checkpoint.exists():
            raise RuntimeError("completed foundation state has no final checkpoint")
        saved = __import__("torch").load(
            promoted_checkpoint, map_location="cpu", weights_only=False
        )
        stages_payload = saved["stages"]
        highest_stage = int(saved["highest_stage"])
    else:
        stages_payload = [asdict(stage) for stage in final_result.stages]
        highest_stage = final_result.highest_stage
    # Publish exactly one compact initialization checkpoint. Prefer the latest
    # promoted state: training on a subsequently capped rung is evidence, not an
    # automatic reason to replace the last weights that passed a gate.
    source_checkpoint = promoted_checkpoint if promoted_checkpoint.exists() else progress_checkpoint
    if not source_checkpoint.exists():
        raise RuntimeError("foundation run produced no recoverable checkpoint")
    import torch

    saved = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    final_checkpoint = output / "checkpoint.pt"
    temporary_checkpoint = final_checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "network": saved["network"],
            "candidate": base_candidate.name,
            "candidate_spec": asdict(base_candidate),
            "seed": int(payload["seed"]),
            "highest_stage": highest_stage,
            "checkpoint_basis": (
                "latest-promoted" if source_checkpoint == promoted_checkpoint else "latest-progress"
            ),
        },
        temporary_checkpoint,
    )
    temporary_checkpoint.replace(final_checkpoint)
    result = {
        "schema": "semantic-v1-foundation-pretrain-run-v1",
        "candidate": base_candidate.name,
        "seed": int(payload["seed"]),
        "highest_stage": highest_stage,
        "stages": stages_payload,
        "attempts": state["attempts"],
        "final_schedule": {
            "F_native": native_levels[int(state["native_index"])],
            "simulations": simulation_levels[int(state["simulation_index"])],
        },
        "checkpoint": str(final_checkpoint),
        "checkpoint_basis": (
            "latest-promoted" if source_checkpoint == promoted_checkpoint else "latest-progress"
        ),
    }
    _atomic_json(result_path, result)
    return result


def run_foundation_pretraining(
    output: Path,
    *,
    candidate_names: list[str],
    seeds: list[int],
    workers: int = 4,
    native_levels: tuple[int, ...] = (5, 8, 12, 16),
    simulation_levels: tuple[int, ...] = (64, 128, 256, 512),
    initial_old_cycles: int = 1,
    max_old_cycles: int = 8,
    evaluation_target: float = 0.70,
    retention_target: float = 0.80,
    selfplay_games: int = 2,
    eval_games: int = 10,
    eval_every: int = 2,
    retro_games: int = 24,
    promote_at: float = 0.80,
    stage_limit: int = len(EARLY_MIXED_STRAND_STAGES),
    device: str = "cpu",
) -> dict[str, Any]:
    from pgx_mcts_bench.ladder import foundation_arms

    roster = {candidate.name: candidate for candidate in foundation_arms()}
    known = set(roster)
    unknown = sorted(set(candidate_names) - known)
    if unknown:
        raise ValueError(f"unknown foundation candidates: {', '.join(unknown)}")
    if not candidate_names or not seeds:
        raise ValueError("at least one candidate and seed are required")
    if not native_levels or any(value < 1 for value in native_levels):
        raise ValueError("native levels must be positive")
    if tuple(sorted(set(native_levels))) != native_levels:
        raise ValueError("native levels must be strictly increasing")
    if not simulation_levels or any(value < 1 for value in simulation_levels):
        raise ValueError("simulation levels must be positive")
    if tuple(sorted(set(simulation_levels))) != simulation_levels:
        raise ValueError("simulation levels must be strictly increasing")
    if not 1 <= stage_limit <= len(EARLY_MIXED_STRAND_STAGES):
        raise ValueError("invalid foundation stage limit")
    settings = {
        "schema": "semantic-v1-foundation-pretrain-protocol-v2",
        "candidate_names": candidate_names,
        "candidate_specs": [asdict(roster[name]) for name in candidate_names],
        "source_provenance": source_provenance(),
        "seeds": seeds,
        "native_levels": list(native_levels),
        "simulation_levels": list(simulation_levels),
        "initial_old_cycles": initial_old_cycles,
        "max_old_cycles": max_old_cycles,
        "evaluation_target": evaluation_target,
        "retention_target": retention_target,
        "selfplay_games": selfplay_games,
        "eval_games": eval_games,
        "eval_every": eval_every,
        "retro_games": retro_games,
        "promote_at": promote_at,
        "stage_limit": stage_limit,
        "stages": [list(stage) for stage in EARLY_MIXED_STRAND_STAGES[:stage_limit]],
        "device": device,
    }
    # JSON turns every tuple in a candidate spec into a list. Compare and reuse
    # that canonical representation too, otherwise every interrupted run with a
    # tuple-valued field rejects its own manifest on resume.
    settings = json.loads(json.dumps(settings))
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError("foundation pretraining protocol differs from existing manifest")
    _atomic_json(manifest, settings)
    jobs = [
        {**settings, "candidate": candidate, "seed": seed, "output": str(output)}
        for seed in seeds
        for candidate in candidate_names
    ]
    if workers == 1:
        rows = [_run_one(job) for job in jobs]
    else:
        # Linux defaults to ``fork``. JAX has already initialized background
        # threads by the time the CLI enters here, so forked PyTorch/JAX workers
        # can deadlock before their first MCTS step. Spawn is slower only during
        # startup and is safe on every supported platform.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            rows = list(pool.map(_run_one, jobs))
    report = {**settings, "runs": rows}
    _atomic_json(output / "report.json", report)
    return report
