#!/usr/bin/env python3
"""Collect played MCTS root trajectories with exact full-braid context."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import STAGES, Candidate, _config
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.on_policy_embedding_value import (
    SCHEMA,
    attach_terminal_outcome,
    state_row,
    tensor_payload,
)
from pgx_mcts_bench.search import NeuralMCTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    torch.save(payload, temporary)
    temporary.replace(path)


def collect_stage(
    *,
    candidate: Candidate,
    parent,
    stage_index: int,
    split: str,
    games: int,
    simulations: int,
    ratio: float,
    seed: int,
    device: str,
    first_episode_id: int,
) -> tuple[list[dict], list[dict]]:
    config = _config(candidate, STAGES[stage_index], seed, device, selfplay_games=1)
    config = replace(config, search=replace(config.search, simulations=simulations))
    game = make_game(config.game)
    search = NeuralMCTS(game, parent, config.search, device)
    source_name, scramble = STAGES[stage_index]
    source = next(item for item in game.generator.sources if item.name == source_name)
    rows: list[dict] = []
    summaries: list[dict] = []
    episodes = []
    for attempt in range(games):
        episode_id = first_episode_id + attempt
        episode_seed = seed + 10_000_019 * stage_index + 100_003 * (attempt + 1)
        rng = np.random.default_rng(episode_seed)
        instance = game.generator.generate(source, scramble, rng)
        episodes.append(
            {
                "episode_id": episode_id,
                "episode_seed": episode_seed,
                "attempt": attempt,
                "rng": rng,
                "instance": instance,
                "transition": game.from_word(
                    list(instance.word), instance.strands, math.log(ratio)
                ),
                "rows": [],
            }
        )

    while True:
        active = [
            index for index, episode in enumerate(episodes) if not episode["transition"].terminated
        ]
        if not active:
            break
        results = search.run_batch(
            states=[episodes[index]["transition"].state for index in active],
            observations=[episodes[index]["transition"].observation for index in active],
            legal_actions=[episodes[index]["transition"].legal_actions for index in active],
            rngs=[episodes[index]["rng"] for index in active],
            temperatures=[0.0] * len(active),
            add_root_noise=False,
        )
        for index, result in zip(active, results, strict=True):
            episode = episodes[index]
            transition = episode["transition"]
            episode["rows"].append(
                state_row(
                    transition.state,
                    transition.observation,
                    game=game,
                    player=transition.player,
                    root_value=result.root_value,
                    episode_id=episode["episode_id"],
                    episode_seed=episode["episode_seed"],
                    stage_index=stage_index,
                    split=split,
                    position_index=len(episode["rows"]),
                )
            )
            episode["transition"] = game.step(transition.state, result.action)

    for episode in episodes:
        transition = episode["transition"]
        episode_rows = episode["rows"]
        instance = episode["instance"]
        final = game.unwrap(transition.state)
        solved = bool((np.asarray(final._word) == 0).all()) and int(np.asarray(final._n)) == 1
        rewards = game.final_rewards(transition.state)
        attach_terminal_outcome(episode_rows, rewards, solved=solved)
        rows.extend(episode_rows)
        summaries.append(
            {
                "episode_id": episode["episode_id"],
                "seed": episode["episode_seed"],
                "split": split,
                "stage_index": stage_index,
                "stage": STAGES[stage_index],
                "initial_word": [int(letter) for letter in instance.word],
                "initial_strands": int(instance.strands),
                "positions": len(episode_rows),
                "solved": solved,
                "target": float(rewards[episode_rows[0]["player"]]),
                "crossings": int(np.asarray(final._crossing_changes)) if solved else None,
                "semantic_moves": game.semantic_move_count(transition.state) if solved else None,
            }
        )
    return rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-stage", type=int, action="append", required=True)
    parser.add_argument("--validation-stage", type=int, action="append", required=True)
    parser.add_argument("--train-games", type=int, default=8)
    parser.add_argument("--validation-games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--ratio", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.train_games, args.validation_games, args.simulations) < 1 or args.ratio <= 0:
        raise ValueError("games, simulations, and ratio must be positive")
    if set(args.train_stage) & set(args.validation_stage):
        raise ValueError("train and validation stages must be disjoint")
    for stage_index in args.train_stage + args.validation_stage:
        if not 0 <= stage_index < len(STAGES):
            raise ValueError(f"stage {stage_index} is outside 0..{len(STAGES) - 1}")

    manifest = json.loads(args.parent_manifest.read_text())
    candidate = Candidate(**manifest["candidate"])
    base_config = _config(candidate, STAGES[0], args.seed, args.device, selfplay_games=1)
    parent = make_braid_network(base_config.game, base_config.model).to(args.device)
    saved = torch.load(args.parent_checkpoint, map_location=args.device, weights_only=False)
    load_policy_value_state_dict(parent, saved.get("network", saved))
    parent.eval()

    rows: list[dict] = []
    episodes: list[dict] = []
    next_episode_id = 0
    for split, stages, games in (
        ("train", args.train_stage, args.train_games),
        ("validation", args.validation_stage, args.validation_games),
    ):
        for stage_index in stages:
            stage_rows, stage_episodes = collect_stage(
                candidate=candidate,
                parent=parent,
                stage_index=stage_index,
                split=split,
                games=games,
                simulations=args.simulations,
                ratio=args.ratio,
                seed=args.seed,
                device=args.device,
                first_episode_id=next_episode_id,
            )
            rows.extend(stage_rows)
            episodes.extend(stage_episodes)
            next_episode_id += games
            solved = sum(item["solved"] for item in stage_episodes)
            print(f"{split} stage {stage_index}: {solved}/{games} solved, {len(stage_rows)} rows")

    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "inputs": {
            "parent_manifest": str(args.parent_manifest.resolve()),
            "parent_checkpoint": str(args.parent_checkpoint.resolve()),
            "parent_checkpoint_sha256": sha256(args.parent_checkpoint),
        },
        "protocol": {
            "target": "actual terminal semantic payoff from each played root state",
            "states": "played MCTS trajectory roots only; no unplayed branch relabeling",
            "policy": "frozen parent",
            "root_noise": False,
            "temperature": 0.0,
            "simulations": args.simulations,
            "ratio": args.ratio,
            "seed": args.seed,
            "train_stages": args.train_stage,
            "validation_stages": args.validation_stage,
            "stage_disjoint_validation": True,
            "test_split_used": False,
        },
        "episodes": episodes,
        "data": tensor_payload(rows),
    }
    atomic_torch_save(payload, args.output)
    print(f"wrote {len(rows)} rows from {len(episodes)} episodes to {args.output}")


if __name__ == "__main__":
    main()
