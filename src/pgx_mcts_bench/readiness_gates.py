"""Pre-flight gates before the long scientist-collaboration comparison."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pgx_mcts_bench import ladder as ladder_module
from pgx_mcts_bench.collaborative_scientists import _atomic_json, _json_hash
from pgx_mcts_bench.ladder import candidates, run_ladder

ARCHIVED_FIRST_TEN = [
    ("unknot", 2),
    ("unknot", 6),
    ("T(2,3)", 0),
    ("T(2,3)", 4),
    ("T(2,5)", 0),
    ("T(2,5)", 4),
    ("T(2,7)", 0),
    ("T(2,7)", 4),
    ("T(3,4)", 0),
    ("T(3,4)", 4),
]

RUNG_REFERENCE = {
    "candidate": "s-window-128",
    "recipe": {
        "selfplay_games": 8,
        "train_steps_per_iteration": 96,
        "simulations": 128,
        "policy_value_update": "historical AlphaZero wins-and-losses",
        "crossing_tolerance": 0.5,
        "collapse_floor": 0.45,
        "max_consecutive_caps": 99,
    },
    "archived_first_ten": {
        "stage_identities": [
            "unknot+2",
            "unknot+6",
            "T(2,3)+0",
            "T(2,3)+4",
            "T(2,5)+0",
            "T(2,5)+4",
            "T(2,7)+0",
            "T(2,7)+4",
            "T(3,4)+0",
            "T(3,4)+4",
        ],
        "iterations": [2, 20, 2, 4, 2, 2, 2, 2, 2, 58],
        "solve_rates": [1.0, 5 / 6, 1.0, 11 / 12, 1.0, 11 / 12, 1.0, 11 / 12, 1.0, 5 / 6],
        "note": "historical stages changed; these are evidence, not row-wise targets",
    },
}


def solution_horizon_contract(native_action_horizon: int) -> dict[str, Any]:
    """State exactly what an admitted solution means in finite search."""
    return {
        "terminal_condition": "exactly replayed braid state has empty word and one strand",
        "learned_objective_cap": None,
        "crossing_change_cap": None,
        "native_action_horizon": native_action_horizon,
        "completeness_claim": False,
        "interpretation": (
            "failure means no solution was found within this finite action/search horizon; "
            "it is not evidence that the knot has no solution at a larger horizon"
        ),
    }


def analyze_rung_parity(result: dict[str, Any], stop_after: int) -> dict[str, Any]:
    """Check reproduction of the successful curriculum, not a new update ablation.

    Success-gated policy imitation is a safety rule for adapting a pretrained
    scientist.  Applying it during training from scratch changes the rung-18
    algorithm and removes the negative examples needed by its critic.
    """
    stages = [
        {
            "stage": int(stage["stage"]),
            "identity": f"{stage['source']}+{stage['scramble']}",
            "iterations": int(stage["iterations"]),
            "solve_rate": float(stage["solve_rate"]),
            "promoted": bool(stage["promoted"]),
        }
        for stage in result["stages"]
        if int(stage["stage"]) <= stop_after
    ]
    failed_rungs = [row["identity"] for row in stages if not row["promoted"]]
    low_solve_rate_rungs = [
        row["identity"] for row in stages if row["solve_rate"] < 0.8
    ]
    passed = (
        int(result["highest_stage"]) >= stop_after
        and len(stages) == stop_after + 1
        and not failed_rungs
        and not low_solve_rate_rungs
    )
    return {
        "reproduction": {
            "highest_stage": int(result["highest_stage"]),
            "total_iterations": sum(row["iterations"] for row in stages),
            "stages": stages,
            "failed_rungs": failed_rungs,
            "solve_rate_below_0_8": low_solve_rate_rungs,
        },
        "decision": {
            "passed": passed,
            "next_step": (
                "run matched five-action witness distillation"
                if passed
                else "repair curriculum parity before collaboration"
            ),
        },
    }


def _run_rung_arm(payload: dict[str, Any]) -> dict[str, Any]:
    base = next(candidate for candidate in candidates() if candidate.name == "s-window-128")
    candidate = replace(
        base,
        name="rung18-reproduction",
        rationale="readiness reproduction of the successful rung-18 recipe",
    )
    previous_stages = ladder_module.STAGES
    ladder_module.STAGES = ARCHIVED_FIRST_TEN
    try:
        result = run_ladder(
            candidate,
            seed=payload["seed"],
            device=payload["device"],
            checkpoint_dir=Path(payload["output"]) / "checkpoints",
            max_iterations_per_stage=payload["max_iterations"],
            selfplay_games=8,
            checkpoint_every=1,
            eval_every=2,
            eval_games=payload["eval_games"],
            promote_at=0.8,
            mix_decay=0.5,
            crossing_tolerance=0.5,
            plateau_window=3,
            collapse_floor=0.45,
            max_consecutive_caps=99,
            stop_after=payload["stop_after"],
            min_iterations_per_rung=9.0,
            min_iterations_from=10,
            retro_games=payload["retro_games"],
            policy_value_success_only=False,
            policy_value_success_gated=False,
            log=lambda _message: None,
        )
        return asdict(result)
    finally:
        ladder_module.STAGES = previous_stages


def run_rung_parity_gate(
    output: Path,
    *,
    seed: int = 0,
    stop_after: int = 9,
    max_iterations: int = 100,
    eval_games: int = 12,
    retro_games: int = 4,
    workers: int = 2,
    device: str = "cpu",
) -> dict[str, Any]:
    protocol = {
        "schema": "rung-parity-gate-v2",
        "candidate": "s-window-128",
        "reference": RUNG_REFERENCE,
        "rungs": [
            f"{source}+{scramble}"
            for source, scramble in ARCHIVED_FIRST_TEN[: stop_after + 1]
        ],
        "seed": seed,
        "stop_after": stop_after,
        "max_iterations": max_iterations,
        "selfplay_games": 8,
        "train_steps_per_iteration": 96,
        "simulations": 128,
        "eval_games": eval_games,
        "retro_games": retro_games,
        "crossing_tolerance": 0.5,
        "collapse_floor": 0.45,
        "max_consecutive_caps": 99,
        "min_iterations_per_rung": 9.0,
        "min_iterations_from": 10,
        "solution_horizon": solution_horizon_contract(64),
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "manifest.json", protocol)
    payload = {
        "seed": seed,
        "output": str(output),
        "stop_after": stop_after,
        "max_iterations": max_iterations,
        "eval_games": eval_games,
        "retro_games": retro_games,
        "device": device,
    }
    result = _run_rung_arm(payload)
    _atomic_json(output / "result.json", result)
    report = {**protocol, **analyze_rung_parity(result, stop_after)}
    _atomic_json(output / "report.json", report)
    return report
