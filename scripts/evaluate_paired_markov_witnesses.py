#!/usr/bin/env python3
"""Evaluate one braid-ladder checkpoint on paired equivalent representations.

Each generated representation is evaluated once at every requested objective
ratio.  Complete semantic witnesses are written for solved trajectories, and
the report identifies within-representation Pareto dominance inversions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from rf_knots.evidence import BraidState, UnknotWitness

from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import _config, candidates
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.search import NeuralMCTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", default="d-tape4-u1")
    parser.add_argument("--source", default="R(3,12)#0")
    parser.add_argument("--scramble-depths", default="1,2,4,8")
    parser.add_argument("--seeds-per-depth", type=int, default=2)
    parser.add_argument("--ratios", default="1000,10,0.1")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--use-auxiliary-value", action="store_true")
    parser.add_argument("--disable-film", action="store_true")
    parser.add_argument("--potential-cost-shaping", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def braid_state(game, state) -> BraidState:
    raw = game.unwrap(state)
    word = tuple(int(value) for value in np.asarray(raw._word) if int(value))
    return BraidState(word, int(np.asarray(raw._n)))


def main() -> None:
    args = parse_args()
    candidate = next(item for item in candidates() if item.name == args.candidate)
    if args.simulations is not None:
        candidate = type(candidate)(**{**asdict(candidate), "simulations": args.simulations})

    config = _config(candidate, (args.source, 0), args.seed, args.device, selfplay_games=1)
    if args.potential_cost_shaping:
        config = replace(
            config,
            search=replace(config.search, potential_cost_shaping=True),
        )
    game = make_game(config.game)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    network = make_braid_network(config.game, config.model).to(args.device)
    load_policy_value_state_dict(network, payload.get("network", payload))
    if args.use_auxiliary_value:
        network.use_auxiliary_value = True
    if args.disable_film:
        network.film = None
    network.eval()
    search = NeuralMCTS(game, network, config.search, args.device)

    source = next(item for item in game.generator.sources if item.name == args.source)
    ratios = tuple(float(value) for value in args.ratios.split(","))
    depths = tuple(int(value) for value in args.scramble_depths.split(","))
    representations: list[dict] = []
    for depth in depths:
        # Depth zero is the unchanged canonical representation, so multiple
        # seeds would only duplicate it.
        sample_count = 1 if depth == 0 else args.seeds_per_depth
        for sample in range(sample_count):
            representation_seed = args.seed + depth * 1_000_003 + sample * 100_003
            generated = game.generator.generate(
                source, depth, np.random.default_rng(representation_seed)
            )
            representations.append(
                {
                    "id": f"depth{depth:02d}-sample{sample:02d}",
                    "seed": representation_seed,
                    "scramble_depth": depth,
                    "word": list(generated.word),
                    "strands": int(generated.strands),
                }
            )

    episodes: list[dict] = []
    for representation in representations:
        for ratio in ratios:
            transition = game.from_word(
                representation["word"], representation["strands"], math.log(ratio)
            )
            episodes.append(
                {
                    "representation": representation,
                    "ratio": ratio,
                    "transition": transition,
                    # The identical RNG seed across ratios preserves pairing. At
                    # temperature zero without root noise it affects only ties.
                    "rng": np.random.default_rng(representation["seed"] + 77_777),
                    "controller_actions": [],
                    "braid_states": [braid_state(game, transition.state)],
                }
            )

    while True:
        active = [
            index
            for index, episode in enumerate(episodes)
            if not episode["transition"].terminated
        ]
        if not active:
            break
        searched = search.run_batch(
            states=[episodes[index]["transition"].state for index in active],
            observations=[episodes[index]["transition"].observation for index in active],
            legal_actions=[episodes[index]["transition"].legal_actions for index in active],
            rngs=[episodes[index]["rng"] for index in active],
            temperatures=[0.0] * len(active),
            add_root_noise=False,
        )
        for index, result in zip(active, searched, strict=True):
            episode = episodes[index]
            episode["controller_actions"].append(int(result.action))
            episode["transition"] = game.step(episode["transition"].state, result.action)
            episode["braid_states"].append(braid_state(game, episode["transition"].state))

    rows: list[dict] = []
    for episode in episodes:
            representation = episode["representation"]
            ratio = episode["ratio"]
            final = game.unwrap(episode["transition"].state)
            solved = bool((np.asarray(final._word) == 0).all()) and int(final._n) == 1
            row = {
                "representation_id": representation["id"],
                "scramble_depth": representation["scramble_depth"],
                "representation_seed": representation["seed"],
                "start_word": representation["word"],
                "start_strands": representation["strands"],
                "ratio": ratio,
                "solved": solved,
                "crossing_changes": int(np.asarray(final._crossing_changes)),
                "moves": config.game.simplify_budget - int(np.asarray(final._budget)),
                "controller_actions": episode["controller_actions"],
                "controller_action_descriptions": [
                    game.describe(action) for action in episode["controller_actions"]
                ],
            }
            if solved:
                witness = UnknotWitness.from_states(episode["braid_states"], config.game._spec)
                witness.verify()
                row["witness"] = witness.to_dict()
            rows.append(row)

    inversions: list[dict] = []
    for representation in representations:
        paired = [row for row in rows if row["representation_id"] == representation["id"]]
        for left in paired:
            for right in paired:
                if left["ratio"] <= right["ratio"] or not left["solved"] or not right["solved"]:
                    continue
                if (
                    left["crossing_changes"] >= right["crossing_changes"]
                    and left["moves"] >= right["moves"]
                    and (
                        left["crossing_changes"] > right["crossing_changes"]
                        or left["moves"] > right["moves"]
                    )
                ):
                    inversions.append(
                        {
                            "representation_id": representation["id"],
                            "dominated_ratio": left["ratio"],
                            "dominating_ratio": right["ratio"],
                            "dominated_cost": [left["crossing_changes"], left["moves"]],
                            "dominating_cost": [right["crossing_changes"], right["moves"]],
                        }
                    )

    report = {
        "schema": "paired-markov-witness-evaluation-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "candidate": args.candidate,
        "source": args.source,
        "ratios": ratios,
        "simulations": config.search.simulations,
        "temperature": 0.0,
        "root_noise": False,
        "use_auxiliary_value": args.use_auxiliary_value,
        "film_enabled": not args.disable_film,
        "potential_cost_shaping": args.potential_cost_shaping,
        "representations": representations,
        "results": rows,
        "dominance_inversions": inversions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "rows": len(rows), "inversions": len(inversions)}
        )
    )


if __name__ == "__main__":
    main()
