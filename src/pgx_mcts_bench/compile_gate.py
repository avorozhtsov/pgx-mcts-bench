"""Opt-in torch.compile gate for braid training; never used by production runs."""

from __future__ import annotations

import copy
import json
import platform
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.data import ReplayBuffer
from pgx_mcts_bench.device_benchmark import benchmark_point
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import STAGES, Candidate, _config
from pgx_mcts_bench.networks import make_braid_network
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games, train_alphazero_step


def _replay(records, seed: int) -> ReplayBuffer:
    replay = ReplayBuffer(20_000, np.random.default_rng(seed))
    for record in copy.deepcopy(records):
        replay.add(record)
    return replay


def _max_parameter_delta(left, right) -> float:
    return max(
        float((a.detach() - b.detach()).abs().max())
        for a, b in zip(left.parameters(), right.parameters(), strict=True)
    )


def run_compile_gate(
    candidate: Candidate,
    output: Path,
    *,
    stage_index: int = 1,
    simulations: int = 8,
    actor_batch: int = 8,
    measured_train_steps: int = 12,
    seed: int = 0,
    torch_threads: int = 3,
    minimum_end_to_end_speedup: float = 1.10,
) -> dict[str, Any]:
    """Measure speed, numerical agreement, and eager checkpoint portability."""
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build has no torch.compile")
    torch.set_num_threads(torch_threads)
    candidate = replace(candidate, simulations=simulations)
    config = _config(
        candidate, STAGES[stage_index], seed, "cpu", selfplay_games=actor_batch
    )
    game = make_game(config.game)
    torch.manual_seed(seed)
    initial = make_braid_network(config.game, config.model)
    search = NeuralMCTS(game, initial, config.search, "cpu")
    seeds = [seed + index for index in range(actor_batch)]
    records = play_selfplay_games(
        game,
        search,
        [np.random.default_rng(value + 7) for value in seeds],
        seeds,
        temperature_moves=12,
    )
    eager = copy.deepcopy(initial)
    compiled = copy.deepcopy(initial)
    eager_optimizer = torch.optim.AdamW(
        eager.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    compiled_optimizer = torch.optim.AdamW(
        compiled.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    eager_replay = _replay(records, seed + 50_000)
    compiled_replay = _replay(records, seed + 50_000)

    original_state = copy.deepcopy(compiled.state_dict())
    compiled.forward_with_auxiliary = torch.compile(  # type: ignore[method-assign]
        compiled.forward_with_auxiliary, mode="reduce-overhead"
    )
    started = time.perf_counter()
    warm_loss = train_alphazero_step(
        compiled, compiled_optimizer, compiled_replay, 32, torch.device("cpu")
    )
    compile_warm_seconds = time.perf_counter() - started
    compiled.load_state_dict(original_state)
    compiled_optimizer = torch.optim.AdamW(
        compiled.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    compiled_replay = _replay(records, seed + 50_000)

    eager_losses = []
    started = time.perf_counter()
    for _ in range(measured_train_steps):
        eager_losses.append(
            train_alphazero_step(
                eager, eager_optimizer, eager_replay, 32, torch.device("cpu")
            )
        )
    eager_seconds = time.perf_counter() - started

    compiled_losses = []
    started = time.perf_counter()
    for _ in range(measured_train_steps):
        compiled_losses.append(
            train_alphazero_step(
                compiled,
                compiled_optimizer,
                compiled_replay,
                32,
                torch.device("cpu"),
            )
        )
    compiled_seconds = time.perf_counter() - started

    finite_losses = all(
        np.isfinite(float(row["loss"]))
        for row in [*eager_losses, *compiled_losses, warm_loss]
    )
    max_loss_delta = max(
        abs(float(a["loss"]) - float(b["loss"]))
        for a, b in zip(eager_losses, compiled_losses, strict=True)
    )
    max_parameter_delta = _max_parameter_delta(eager, compiled)

    with tempfile.TemporaryDirectory(prefix="braid-compile-gate-") as directory:
        checkpoint = Path(directory) / "compiled-trained.pt"
        torch.save(
            {
                "network": compiled.state_dict(),
                "optimizer": compiled_optimizer.state_dict(),
            },
            checkpoint,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        resumed = make_braid_network(config.game, config.model)
        resumed.load_state_dict(payload["network"])
        resumed_optimizer = torch.optim.AdamW(
            resumed.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        resumed_optimizer.load_state_dict(payload["optimizer"])
        checkpoint_delta = _max_parameter_delta(compiled, resumed)
        resume_loss = train_alphazero_step(
            resumed,
            resumed_optimizer,
            _replay(records, seed + 60_000),
            32,
            torch.device("cpu"),
        )
    resume_finite = np.isfinite(float(resume_loss["loss"]))

    baseline = benchmark_point(
        candidate,
        device="cpu",
        actor_batch=actor_batch,
        stage_index=stage_index,
        eval_games=1,
        measured_train_steps=min(measured_train_steps, candidate.train_steps),
        seed=seed + 900_000,
    )
    nontraining = (
        float(baseline["normalized_selfplay_seconds"])
        + float(baseline["normalized_evaluation_seconds"])
        + float(baseline["normalized_checkpoint_seconds"])
    )
    scale = candidate.train_steps / measured_train_steps
    projected_eager = nontraining + eager_seconds * scale
    projected_compiled = nontraining + compiled_seconds * scale
    speedup = projected_eager / projected_compiled
    numerically_close = max_loss_delta <= 1e-4 and max_parameter_delta <= 1e-5
    passed = bool(
        finite_losses
        and resume_finite
        and checkpoint_delta == 0.0
        and numerically_close
        and speedup >= minimum_end_to_end_speedup
    )
    report = {
        "schema": "torch-compile-training-gate-v1",
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch_threads,
        },
        "candidate": asdict(candidate),
        "stage_index": stage_index,
        "actor_batch": actor_batch,
        "measured_train_steps": measured_train_steps,
        "compile_warm_seconds": compile_warm_seconds,
        "eager_train_seconds": eager_seconds,
        "compiled_train_seconds": compiled_seconds,
        "training_speedup": eager_seconds / compiled_seconds,
        "projected_eager_iteration_seconds": projected_eager,
        "projected_compiled_iteration_seconds": projected_compiled,
        "projected_end_to_end_speedup": speedup,
        "minimum_end_to_end_speedup": minimum_end_to_end_speedup,
        "finite_losses": finite_losses,
        "max_paired_loss_delta": max_loss_delta,
        "max_parameter_delta": max_parameter_delta,
        "checkpoint_parameter_delta": checkpoint_delta,
        "resume_finite": bool(resume_finite),
        "passed": passed,
        "decision": "enable only after a passing gate" if passed else "keep disabled",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "compile-gate.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "compile-gate.md").write_text(
        "# torch.compile training gate\n\n"
        f"- Candidate: `{candidate.name}`\n"
        f"- Training speedup: {report['training_speedup']:.3f}x\n"
        f"- Projected end-to-end speedup: {speedup:.3f}x\n"
        f"- Compile warm-up: {compile_warm_seconds:.2f}s\n"
        f"- Max paired loss delta: {max_loss_delta:.3g}\n"
        f"- Max parameter delta: {max_parameter_delta:.3g}\n"
        f"- Eager checkpoint delta: {checkpoint_delta:.3g}\n"
        f"- Resume finite: {bool(resume_finite)}\n"
        f"- Decision: **{report['decision']}**\n"
    )
    return report
