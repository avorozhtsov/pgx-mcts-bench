"""Acquire one verified positive trajectory without training on failed policies."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.config import pick_stage
from pgx_mcts_bench.data import GameRecord, ReplayBuffer
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, candidates, evaluate_stage
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games, train_alphazero_step


def _record_objective(record: GameRecord, ratio: float) -> float:
    if not record or float(record[0].solved) <= 0.5:
        return float("inf")
    return ratio * float(record[0].final_crossing_changes) + float(record[0].final_moves)


def _evaluation_aligned_rng(game: Any, episode_seed: int) -> np.random.Generator:
    """Return the search RNG state used by ``evaluate_stage`` for one episode.

    Ladder evaluation uses one generator first to construct the representation
    and then to drive MCTS. ``play_selfplay_games`` reconstructs the same
    representation through ``game.reset(seed)``, so its separately supplied
    search generator must consume those construction draws too. Without this,
    a solved evaluation seed cannot be replayed as a donor witness.
    """
    rng = np.random.default_rng(episode_seed)
    if game.generator is None:
        return rng
    source, moves = pick_stage(game.config, game.generator, rng)
    game.generator.generate(source, moves, rng)
    low, high = game.config.log_ratio_range
    if high > low:
        rng.uniform(low, high)
    return rng


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_positive_acquisition(
    donor_checkpoint: Path,
    receiver_checkpoint: Path,
    output_checkpoint: Path,
    output_report: Path,
    *,
    candidate_name: str,
    stage_source: str,
    stage_scramble: int = 0,
    ratio: float = 1000.0,
    donor_simulations: int = 256,
    donor_attempts: int = 4,
    train_steps: int = 96,
    batch_size: int = 32,
    evaluation_simulations: int = 256,
    evaluation_games: int = 12,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train a receiver on only the best exactly solved donor episode.

    Failed donor searches are counted in the report but never enter replay.
    The output preserves ladder stages and optimizer state from the receiver so
    it can be installed as a resumable frontier checkpoint.
    """
    by_name = {candidate.name: candidate for candidate in candidates()}
    if candidate_name not in by_name:
        raise ValueError(f"unknown candidate {candidate_name!r}")
    if donor_attempts < 1 or train_steps < 1 or batch_size < 1:
        raise ValueError("attempts, train steps, and batch size must be positive")
    candidate = replace(by_name[candidate_name], simulations=donor_simulations)
    config = _config(
        candidate,
        (stage_source, stage_scramble),
        seed,
        device,
        selfplay_games=donor_attempts,
    )
    game = make_game(config.game)
    donor = make_braid_network(config.game, config.model).to(device)
    receiver = make_braid_network(config.game, config.model).to(device)
    donor_payload = torch.load(donor_checkpoint, map_location=device, weights_only=False)
    receiver_payload = torch.load(receiver_checkpoint, map_location=device, weights_only=False)
    load_policy_value_state_dict(
        donor,
        donor_payload.get("network", donor_payload),
    )
    load_policy_value_state_dict(
        receiver,
        receiver_payload.get("network", receiver_payload),
    )
    donor.use_auxiliary_value = True
    receiver.use_auxiliary_value = True
    receiver.auxiliary_solve_backprop = True

    search = NeuralMCTS(game, donor, config.search, device)
    # Match evaluate_stage's seed schedule exactly. This lets a reported solved
    # attempt become a reproducible donor rather than an unrelated fresh search.
    attempt_seeds = [seed + 100_003 * (index + 1) for index in range(donor_attempts)]
    records = play_selfplay_games(
        game,
        search,
        [_evaluation_aligned_rng(game, value) for value in attempt_seeds],
        attempt_seeds,
        # A donor is a verified best route, not an exploratory self-play sample.
        # Match evaluate_stage's deterministic root-action selection.
        0,
    )
    solved = [record for record in records if np.isfinite(_record_objective(record, ratio))]
    if not solved:
        raise RuntimeError("donor produced no exactly solved trajectory")
    best = min(solved, key=lambda record: _record_objective(record, ratio))

    evaluation_candidate = replace(candidate, simulations=evaluation_simulations)
    evaluation_config = _config(
        evaluation_candidate,
        (stage_source, stage_scramble),
        seed,
        device,
        selfplay_games=1,
    )
    evaluation_game = make_game(evaluation_config.game)
    before = evaluate_stage(
        evaluation_game,
        receiver,
        evaluation_config,
        evaluation_games,
        seed + 10_000_000,
        ratios=(ratio,),
        include_attempts=True,
    )

    optimizer = torch.optim.AdamW(
        receiver.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    if isinstance(receiver_payload, dict) and "optimizer" in receiver_payload:
        optimizer.load_state_dict(receiver_payload["optimizer"])
    replay = ReplayBuffer(20_000, np.random.default_rng(seed + 20_000_000))
    replay.add(best, representation_id=stage_source, objective_ratio=ratio)
    losses = None
    for _ in range(train_steps):
        losses = train_alphazero_step(
            receiver,
            optimizer,
            replay,
            batch_size,
            torch.device(device),
            collaboration_replay=True,
            shared_fraction=0.0,
            policy_value_success_only=True,
            replay_positions_per_episode=4,
        )
    after = evaluate_stage(
        evaluation_game,
        receiver,
        evaluation_config,
        evaluation_games,
        seed + 10_000_000,
        ratios=(ratio,),
        include_attempts=True,
    )

    checkpoint = dict(receiver_payload) if isinstance(receiver_payload, dict) else {}
    checkpoint["network"] = receiver.state_dict()
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["positive_acquisition"] = {
        "candidate": candidate_name,
        "stage": [stage_source, stage_scramble],
        "ratio": ratio,
        "donor_checkpoint": str(donor_checkpoint.resolve()),
        "receiver_checkpoint": str(receiver_checkpoint.resolve()),
        "attempts": donor_attempts,
        "solved_attempts": len(solved),
        "best_objective": _record_objective(best, ratio),
        "train_steps": train_steps,
    }
    _atomic_torch_save(output_checkpoint, checkpoint)
    report = {
        "schema": "braid-positive-acquisition-v1",
        "checkpoint": str(output_checkpoint.resolve()),
        "candidate": candidate_name,
        "stage": [stage_source, stage_scramble],
        "ratio": ratio,
        "donor": {
            "attempts": donor_attempts,
            "attempt_seeds": attempt_seeds,
            "temperature_moves": 0,
            "solved": len(solved),
            "best_objective": _record_objective(best, ratio),
            "best_crossing_changes": float(best[0].final_crossing_changes),
            "best_moves": float(best[0].final_moves),
        },
        "training": {
            "episodes": 1,
            "failed_policy_episodes": 0,
            "steps": train_steps,
            "positions_per_episode": 4,
            "last_losses": losses,
        },
        "before": before,
        "after": after,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
