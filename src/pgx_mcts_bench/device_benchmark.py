"""End-to-end CPU/CUDA gate for the braid ladder.

The networks are small enough that a forward-only microbenchmark is misleading:
game stepping, tree traversal, training, and held-out evaluation all matter.  This
module times those phases separately and normalizes them to one standard ladder
iteration (8 self-play games, the candidate's full optimizer budget, and half of
a 12-game-per-ratio evaluation because the ladder evaluates every two iterations).
"""

from __future__ import annotations

import json
import platform
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from pgx_mcts_bench.data import ReplayBuffer
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import (
    STAGES,
    Candidate,
    LadderResult,
    _config,
    _save_ladder_progress,
    evaluate_stage,
)
from pgx_mcts_bench.networks import make_braid_network
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import parameter_count, play_selfplay_games, train_alphazero_step


def available_device(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return False


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark_point(
    candidate: Candidate,
    *,
    device: str,
    actor_batch: int,
    stage_index: int,
    eval_games: int,
    measured_train_steps: int,
    seed: int,
) -> dict:
    """Measure one fresh network; no benchmark point can warm-start another."""
    if not available_device(device):
        raise ValueError(f"Device {device!r} is not available")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    config = _config(
        candidate,
        STAGES[stage_index],
        seed,
        device,
        selfplay_games=actor_batch,
    )
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model).to(torch.device(device))
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
    replay = ReplayBuffer(20_000, rng)
    search = NeuralMCTS(game, network, config.search, device)
    seeds = [seed + index for index in range(actor_batch)]

    # Exclude one-time JAX/CUDA initialization. Without this, whichever device
    # and batch happens to run first pays compilation and is systematically
    # understated. Warm the exact actor batch because tiny networks are sensitive
    # to launch shape even when their weights are identical.
    warm_transitions = [game.reset(value + 900_000) for value in seeds]
    warm_rngs = [np.random.default_rng(value + 900_007) for value in seeds]
    warm_results = search.run_batch(
        states=[transition.state for transition in warm_transitions],
        observations=[transition.observation for transition in warm_transitions],
        legal_actions=[transition.legal_actions for transition in warm_transitions],
        rngs=warm_rngs,
        temperatures=[0.0] * actor_batch,
        add_root_noise=False,
    )
    for transition, result in zip(warm_transitions, warm_results, strict=True):
        game.step(transition.state, result.action)
    _sync(device)

    _sync(device)
    started = time.perf_counter()
    records = play_selfplay_games(
        game,
        search,
        [np.random.default_rng(value + 7) for value in seeds],
        seeds,
        temperature_moves=12,
    )
    _sync(device)
    selfplay_seconds = time.perf_counter() - started
    positions = sum(len(record) for record in records)
    for record in records:
        replay.add(record)

    steps = min(measured_train_steps, candidate.train_steps)
    _sync(device)
    started = time.perf_counter()
    for _ in range(steps):
        train_alphazero_step(network, optimizer, replay, 32, torch.device(device))
    _sync(device)
    training_seconds = time.perf_counter() - started

    _sync(device)
    started = time.perf_counter()
    evaluate_stage(
        game,
        network,
        config,
        eval_games,
        seed + 500_000,
    )
    _sync(device)
    evaluation_seconds = time.perf_counter() - started

    # Preemptible-safe operation checkpoints every iteration, and CUDA tensors
    # have to be serialized too. Include that real storage/device transfer cost.
    with tempfile.TemporaryDirectory(prefix="braid-device-gate-") as directory:
        _sync(device)
        started = time.perf_counter()
        _save_ladder_progress(
            Path(directory) / "progress.pt",
            candidate=candidate,
            stage_index=stage_index,
            iteration=1,
            stage_complete=False,
            network=network,
            optimizer=optimizer,
            replay=replay,
            rng=rng,
            result=LadderResult(candidate.name, candidate.rationale, -1, 0.0),
            history=[],
            by_ratio={},
            solve_rate=float("nan"),
            crossings=float("nan"),
            consecutive_caps=0,
        )
        _sync(device)
        checkpoint_seconds = time.perf_counter() - started

    # Compare equal work even when a large GPU actor batch is used. Evaluation
    # contains three ratios, and the production ladder performs it every second
    # iteration. The result is an estimate, not a claim that changing actor batch
    # leaves the learning trajectory invariant.
    normalized_selfplay = selfplay_seconds * 8 / actor_batch
    normalized_training = training_seconds * candidate.train_steps / max(steps, 1)
    normalized_evaluation = evaluation_seconds * 12 / eval_games / 2
    return {
        "candidate": candidate.name,
        "candidate_spec": asdict(candidate),
        "device": device,
        "actor_batch": actor_batch,
        "stage": stage_index,
        "source": STAGES[stage_index][0],
        "scramble": STAGES[stage_index][1],
        "parameters": parameter_count(network),
        "simulations": candidate.simulations,
        "positions": positions,
        "measured_train_steps": steps,
        "eval_games_per_ratio": eval_games,
        "selfplay_seconds": selfplay_seconds,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "selfplay_games_per_second": actor_batch / selfplay_seconds,
        "positions_per_second": positions / selfplay_seconds,
        "normalized_iteration_seconds": (
            normalized_selfplay + normalized_training + normalized_evaluation + checkpoint_seconds
        ),
        "normalized_selfplay_seconds": normalized_selfplay,
        "normalized_training_seconds": normalized_training,
        "normalized_evaluation_seconds": normalized_evaluation,
        "normalized_checkpoint_seconds": checkpoint_seconds,
    }


def run_device_benchmark(
    candidates: list[Candidate],
    *,
    devices: list[str],
    actor_batches: list[int],
    stage_index: int,
    eval_games: int,
    measured_train_steps: int,
    seed: int,
    output: Path,
    simulations: int | None = None,
    torch_threads: int = 1,
    cpu_hourly: float = 0.0248,
    gpu_hourly: float = 1.5484,
    log=print,
) -> dict:
    torch.set_num_threads(torch_threads)
    rows: list[dict] = []
    skipped: list[str] = []
    for device in devices:
        if not available_device(device):
            skipped.append(device)
            log(f"skip {device}: unavailable")
            continue
        for original in candidates:
            candidate = (
                replace(original, simulations=simulations) if simulations is not None else original
            )
            for actor_batch in actor_batches:
                log(f"benchmark {candidate.name} {device} actors={actor_batch}")
                row = benchmark_point(
                    candidate,
                    device=device,
                    actor_batch=actor_batch,
                    stage_index=stage_index,
                    eval_games=eval_games,
                    measured_train_steps=measured_train_steps,
                    seed=seed,
                )
                rows.append(row)
                log(
                    f"  normalized {row['normalized_iteration_seconds']:.2f}s/iteration; "
                    f"{row['positions_per_second']:.1f} positions/s"
                )

    decisions: list[dict] = []
    for candidate in {row["candidate"] for row in rows}:
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        cpu = [row for row in candidate_rows if row["device"] == "cpu"]
        cuda = [row for row in candidate_rows if row["device"] == "cuda"]
        if not cpu or not cuda:
            continue
        cpu_best = min(cpu, key=lambda row: row["normalized_iteration_seconds"])
        cuda_best = min(cuda, key=lambda row: row["normalized_iteration_seconds"])
        speedup = (
            cpu_best["normalized_iteration_seconds"] / cuda_best["normalized_iteration_seconds"]
        )
        cpu_cost = cpu_best["normalized_iteration_seconds"] / 3600 * cpu_hourly
        gpu_cost = cuda_best["normalized_iteration_seconds"] / 3600 * gpu_hourly
        cost_ratio = gpu_cost / cpu_cost
        decisions.append(
            {
                "candidate": candidate,
                "cpu_actor_batch": cpu_best["actor_batch"],
                "cuda_actor_batch": cuda_best["actor_batch"],
                "speedup": speedup,
                "technical_gpu_gate": speedup >= 3.0,
                "cpu_hourly": cpu_hourly,
                "gpu_hourly": gpu_hourly,
                "cpu_cost_per_normalized_iteration": cpu_cost,
                "gpu_cost_per_normalized_iteration": gpu_cost,
                "gpu_to_cpu_cost_ratio": cost_ratio,
                "use_gpu": speedup >= 3.0 and cost_ratio < 1.0,
                "gate": "CUDA must be at least 3x faster and cheaper per equal-work iteration",
            }
        )

    payload = {
        "schema": 1,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "cpu_threads": torch.get_num_threads(),
        },
        "skipped_devices": skipped,
        "hourly_prices": {"cpu_worker": cpu_hourly, "gpu_vm": gpu_hourly},
        "rows": rows,
        "decisions": decisions,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "device-benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "device-benchmark.md").write_text(render(payload))
    return payload


def render(payload: dict) -> str:
    lines = [
        "# Braid ladder device gate",
        "",
        "Times are normalized to 8 self-play games, the candidate's configured",
        "optimizer steps, and half of a 12-game-per-ratio evaluation.",
        "",
        "| candidate | device | actors | params | selfplay | train | eval/2 | "
        "ckpt | normalized | pos/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['candidate']} | {row['device']} | {row['actor_batch']} | "
            f"{row['parameters']} | {row['normalized_selfplay_seconds']:.2f}s | "
            f"{row['normalized_training_seconds']:.2f}s | "
            f"{row['normalized_evaluation_seconds']:.2f}s | "
            f"{row['normalized_checkpoint_seconds']:.2f}s | "
            f"{row['normalized_iteration_seconds']:.2f}s | "
            f"{row['positions_per_second']:.1f} |"
        )
    lines += ["", "## Decision", ""]
    if payload["decisions"]:
        for decision in payload["decisions"]:
            verdict = "GPU" if decision["use_gpu"] else "CPU"
            lines.append(
                f"- `{decision['candidate']}`: {decision['speedup']:.2f}x speedup, "
                f"GPU/CPU cost {decision['gpu_to_cpu_cost_ratio']:.2f}x; "
                f"choose **{verdict}**."
            )
    else:
        lines.append("Run both `cpu` and `cuda` on the same machine to obtain a decision.")
    lines += [
        "",
        "> Larger actor batches change the amount of fresh self-play data per optimizer",
        "> iteration. Normalization measures throughput; it does not assert identical",
        "> learning trajectories.",
        "",
    ]
    return "\n".join(lines)
