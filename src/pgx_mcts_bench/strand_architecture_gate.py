"""Bounded early-curriculum gate for strand-aware serial architectures."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

# A deliberately short progression that introduces four strands before a large
# model can specialize on the historical two-strand prefix.  Positive-braid
# sources have exact u=(crossings-strands+1)/2, so promotion remains objective.
EARLY_MIXED_STRAND_STAGES: tuple[tuple[str, int], ...] = (
    ("unknot", 2),
    ("T(2,3)", 0),
    ("P(3,4)#0", 0),
    ("P(4,5)#0", 0),
    ("T(2,5)", 0),
    ("P(4,7)#0", 0),
)


def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    from pgx_mcts_bench import ladder
    from pgx_mcts_bench.worker_runtime import worker_init

    worker_init()
    by_name = {candidate.name: candidate for candidate in ladder.candidates()}
    candidate = replace(
        by_name[payload["candidate"]], simulations=int(payload["simulations"])
    )
    output = Path(payload["output"]) / candidate.name / f"seed-{payload['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    stages = EARLY_MIXED_STRAND_STAGES[: int(payload["stage_limit"])]
    historical_stages = ladder.STAGES
    ladder.STAGES = list(stages)
    try:
        result = ladder.run_ladder(
            candidate,
            seed=int(payload["seed"]),
            device=str(payload["device"]),
            checkpoint_dir=output,
            max_iterations_per_stage=int(payload["max_iterations"]),
            selfplay_games=int(payload["selfplay_games"]),
            checkpoint_every=1,
            eval_every=int(payload["eval_every"]),
            eval_games=int(payload["eval_games"]),
            promote_at=float(payload["promote_at"]),
            mix_decay=0.5,
            max_consecutive_caps=int(payload["max_consecutive_caps"]),
            plateau_on_known_objective=False,
            rehearsal_games_per_cleared_stage=int(
                payload["rehearsal_games_per_cleared_stage"]
            ),
            adaptive_rehearsal=bool(payload["adaptive_rehearsal"]),
            rehearsal_target=float(payload["rehearsal_target"]),
            max_rehearsal_games_per_stage=int(payload["max_rehearsal_games_per_stage"]),
            stop_after=len(stages) - 1,
            retro_games=int(payload["retro_games"]),
            balanced_replay=True,
            policy_value_success_only=True,
            retry_capped_on_resume=bool(payload["retry_capped_on_resume"]),
            log=lambda *args, **kwargs: None,
        )
    finally:
        ladder.STAGES = historical_stages
    row = asdict(result)
    row["seed"] = int(payload["seed"])
    row["candidate_spec"] = asdict(candidate)
    row["four_strand_promoted"] = [
        stage["source"]
        for stage in row["stages"]
        if stage["source"].startswith("P(4,") and stage["promoted"]
    ]
    row["all_stage_solve_rates"] = [stage["solve_rate"] for stage in row["stages"]]
    (output / "gate-result.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )
    return row


def run_strand_architecture_gate(
    output: Path,
    *,
    candidate_names: list[str],
    seeds: list[int],
    workers: int = 1,
    simulations: int = 32,
    max_iterations: int = 4,
    selfplay_games: int = 2,
    eval_games: int = 5,
    eval_every: int = 2,
    rehearsal_games_per_cleared_stage: int = 1,
    adaptive_rehearsal: bool = True,
    rehearsal_target: float = 0.8,
    max_rehearsal_games_per_stage: int = 8,
    max_consecutive_caps: int = 1,
    retry_capped_on_resume: bool = False,
    retro_games: int = 24,
    promote_at: float = 0.8,
    stage_limit: int = len(EARLY_MIXED_STRAND_STAGES),
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the same early mixed-strand curriculum for every candidate/seed."""
    from pgx_mcts_bench.ladder import candidates

    known = {candidate.name for candidate in candidates()}
    unknown = sorted(set(candidate_names) - known)
    if unknown:
        raise ValueError(f"unknown candidates: {', '.join(unknown)}")
    if not candidate_names or not seeds:
        raise ValueError("at least one candidate and seed are required")
    if not 1 <= stage_limit <= len(EARLY_MIXED_STRAND_STAGES):
        raise ValueError(
            f"stage_limit must be between 1 and {len(EARLY_MIXED_STRAND_STAGES)}"
        )
    if eval_every < 1:
        raise ValueError("eval_every must be positive")
    if rehearsal_games_per_cleared_stage < 0:
        raise ValueError("rehearsal_games_per_cleared_stage must be non-negative")
    if adaptive_rehearsal and rehearsal_games_per_cleared_stage < 1:
        raise ValueError("adaptive rehearsal requires a positive initial rehearsal dose")
    if not 0.0 <= rehearsal_target <= 1.0:
        raise ValueError("rehearsal_target must be in 0..1")
    if max_rehearsal_games_per_stage < max(1, rehearsal_games_per_cleared_stage):
        raise ValueError("maximum rehearsal dose is smaller than its initial dose")
    if max_consecutive_caps < 1:
        raise ValueError("maximum consecutive caps must be positive")
    if retro_games < 1:
        raise ValueError("retro_games must be positive")
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "candidate": candidate,
            "seed": seed,
            "output": str(output),
            "simulations": simulations,
            "max_iterations": max_iterations,
            "selfplay_games": selfplay_games,
            "eval_games": eval_games,
            "eval_every": eval_every,
            "rehearsal_games_per_cleared_stage": (
                rehearsal_games_per_cleared_stage
            ),
            "adaptive_rehearsal": adaptive_rehearsal,
            "rehearsal_target": rehearsal_target,
            "max_rehearsal_games_per_stage": max_rehearsal_games_per_stage,
            "max_consecutive_caps": max_consecutive_caps,
            "retry_capped_on_resume": retry_capped_on_resume,
            "retro_games": retro_games,
            "policy_value_success_only": True,
            "promote_at": promote_at,
            "stage_limit": stage_limit,
            "device": device,
        }
        for candidate in candidate_names
        for seed in seeds
    ]
    if workers == 1:
        rows = [_run_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, jobs))
    report = {
        "schema": "strand-architecture-early-mixed-gate-v2",
        "stages": [list(stage) for stage in EARLY_MIXED_STRAND_STAGES[:stage_limit]],
        "settings": {
            "simulations": simulations,
            "max_iterations": max_iterations,
            "selfplay_games": selfplay_games,
            "eval_games": eval_games,
            "eval_every": eval_every,
            "rehearsal_games_per_cleared_stage": (
                rehearsal_games_per_cleared_stage
            ),
            "adaptive_rehearsal": adaptive_rehearsal,
            "rehearsal_target": rehearsal_target,
            "max_rehearsal_games_per_stage": max_rehearsal_games_per_stage,
            "max_consecutive_caps": max_consecutive_caps,
            "retry_capped_on_resume": retry_capped_on_resume,
            "retro_games": retro_games,
            "policy_value_success_only": True,
            "promote_at": promote_at,
            "stage_limit": stage_limit,
            "workers": workers,
            "device": device,
        },
        "runs": rows,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
