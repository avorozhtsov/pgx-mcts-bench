#!/usr/bin/env python3
"""Paired held-out evaluation of shortest-route and bounded-option students."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import smallest_crossing_pool
from pgx_mcts_bench.distill import STUDENT_NAMES, _best_destination, _candidate
from pgx_mcts_bench.game import BraidUnknotGame, make_game
from pgx_mcts_bench.ladder import STAGES, _config
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.serial_braid import SerialBraidGame


@dataclass(frozen=True)
class Outcome:
    knot: str
    target_action: int
    chosen_action: int
    chosen_underlying_action: int | None
    success: bool
    internal_actions: int
    forced_commit: bool
    oracle_reachable: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_paired_p(improvements: int, regressions: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant pairs."""
    discordant = improvements + regressions
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(improvements, regressions) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def teacher_targets(args: argparse.Namespace, pool) -> list[int]:
    candidate = _candidate("u1-puct")
    config = _config(candidate, STAGES[0], args.seed, args.device)
    game = make_game(config.game)
    assert isinstance(game, BraidUnknotGame)
    network = make_braid_network(config.game, config.model).to(args.device)
    saved = torch.load(args.teacher, map_location=args.device, weights_only=False)
    load_policy_value_state_dict(network, saved["network"])
    network.eval()
    search = NeuralMCTS(game, network, config.search, args.device)
    targets: list[int] = []
    batch_size = 25
    for offset in range(0, len(pool), batch_size):
        chunk = pool[offset : offset + batch_size]
        transitions = [
            game.from_word(list(knot.word), knot.strands, log_ratio=math.log(1000.0))
            for knot in chunk
        ]
        results = search.run_batch(
            [item.state for item in transitions],
            [item.observation for item in transitions],
            [item.legal_actions for item in transitions],
            [np.random.default_rng(args.seed + offset + i) for i in range(len(chunk))],
            temperatures=[0.0] * len(chunk),
            add_root_noise=False,
        )
        targets.extend(int(result.action) for result in results)
    return targets


def load_student(name: str, checkpoint: Path, args: argparse.Namespace):
    saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
    base = _candidate(name)
    saved_spec = saved.get("candidate_spec", {})
    native_spec = {
        key: value for key, value in saved_spec.items() if hasattr(base, key)
    }
    native_spec["serial_internal_horizon"] = saved_spec.get("serial_internal_horizon", 0)
    candidate = replace(base, **native_spec)
    config = _config(candidate, STAGES[0], args.seed, args.device)
    game = make_game(config.game)
    assert isinstance(game, SerialBraidGame)
    network = make_braid_network(config.game, config.model).to(args.device)
    load_policy_value_state_dict(network, saved["network"])
    network.eval()
    return game, network


@torch.inference_mode()
def evaluate_student(name: str, checkpoint: Path, args, pool, targets) -> list[Outcome]:
    game, network = load_student(name, checkpoint, args)
    transitions = [
        game.from_word(list(knot.word), knot.strands, log_ratio=math.log(1000.0))
        for knot in pool
    ]
    states = [item.state for item in transitions]
    active = list(range(len(pool)))
    outcomes: list[Outcome | None] = [None] * len(pool)
    counts = [0] * len(pool)
    forced = [False] * len(pool)

    for _decision in range(args.horizon + 1):
        observations = np.stack(
            [
                game._view(  # noqa: SLF001 - evaluator intentionally exposes memory state
                    states[i].pgx,
                    states[i].head,
                    states[i].registers,
                    states[i].colours,
                    states[i].colour,
                    states[i].tape,
                    reward=0.0,
                    internal_steps=states[i].internal_steps,
                ).observation
                for i in active
            ]
        )
        tensor = torch.from_numpy(observations).permute(0, 3, 1, 2).float().to(args.device)
        logits, _ = network(tensor)
        next_active: list[int] = []
        for row, index in enumerate(active):
            state = states[index]
            length = int(np.asarray(state.pgx._word).astype(bool).sum())
            legal = game._legal(state.pgx, state.head, state.internal_steps)  # noqa: SLF001
            if counts[index] == args.horizon:
                legal[game._shift_base :] = False  # noqa: SLF001 - common evaluation cap
            scores = logits[row].detach().cpu().numpy().copy()
            scores[~legal] = -np.inf
            action = int(np.argmax(scores))
            if counts[index] == args.horizon:
                unconstrained = game._legal(state.pgx, state.head, 0)  # noqa: SLF001
                unconstrained_scores = logits[row].detach().cpu().numpy().copy()
                unconstrained_scores[~unconstrained] = -np.inf
                forced[index] = int(np.argmax(unconstrained_scores)) >= game._shift_base  # noqa: SLF001
            if action >= game._shift_base:  # noqa: SLF001
                counts[index] += 1
                step = game.step(state, action)
                states[index] = step.state
                next_active.append(index)
                continue
            underlying = game.underlying_action(action, state.head, length)
            destination = _best_destination(game, targets[index], 0, length)
            outcomes[index] = Outcome(
                knot=pool[index].name,
                target_action=targets[index],
                chosen_action=action,
                chosen_underlying_action=underlying,
                success=underlying == targets[index],
                internal_actions=counts[index],
                forced_commit=forced[index],
                oracle_reachable=destination is not None and len(destination[0]) <= args.horizon,
            )
        active = next_active
        if not active:
            break
    if active:
        raise RuntimeError(f"{name}: {len(active)} policies did not commit by the cap")
    return [item for item in outcomes if item is not None]


def summarize(rows: list[Outcome]) -> dict[str, object]:
    return {
        "knots": len(rows),
        "successes": sum(row.success for row in rows),
        "success_rate": sum(row.success for row in rows) / len(rows),
        "mean_internal_actions": float(np.mean([row.internal_actions for row in rows])),
        "forced_commits": sum(row.forced_commit for row in rows),
        "oracle_reachable": sum(row.oracle_reachable for row in rows),
        "internal_action_histogram": {
            str(count): sum(row.internal_actions == count for row in rows)
            for count in range(6)
        },
    }


def main() -> None:
    args = parse_args()
    pool = smallest_crossing_pool(args.pool_size)
    targets = teacher_targets(args, pool)
    report: dict[str, object] = {
        "design": {
            "pool": "smallest-crossing compatible knot-table entries",
            "pool_size": args.pool_size,
            "teacher": str(args.teacher),
            "teacher_sha256": sha256(args.teacher),
            "old_root": str(args.old),
            "new_root": str(args.new),
            "teacher_simulations": _candidate("u1-puct").simulations,
            "internal_action_cap": args.horizon,
            "selection": "greedy policy; external actions forced after cap",
        },
        "students": {},
    }
    for name in STUDENT_NAMES:
        old_checkpoint = args.old / name / "checkpoints" / f"{name}.pt"
        new_checkpoint = args.new / name / "checkpoints" / f"{name}.pt"
        old_rows = evaluate_student(name, old_checkpoint, args, pool, targets)
        new_rows = evaluate_student(name, new_checkpoint, args, pool, targets)
        paired = [
            {
                "knot": old.knot,
                "old": asdict(old),
                "new": asdict(new),
            }
            for old, new in zip(old_rows, new_rows, strict=True)
        ]
        improvements = sum(
            not old.success and new.success
            for old, new in zip(old_rows, new_rows, strict=True)
        )
        regressions = sum(
            old.success and not new.success
            for old, new in zip(old_rows, new_rows, strict=True)
        )
        report["students"][name] = {
            "old": summarize(old_rows),
            "new": summarize(new_rows),
            "improvements": improvements,
            "regressions": regressions,
            "net_change": improvements - regressions,
            "exact_paired_p": exact_paired_p(improvements, regressions),
            "paired_outcomes": paired,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        name: {key: value for key, value in rows.items() if key != "paired_outcomes"}
        for name, rows in report["students"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
