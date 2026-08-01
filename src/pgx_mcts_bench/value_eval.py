"""Raw critic evaluation on deterministic held-out ladder rungs."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import (
    RATIOS,
    STAGES,
    VALUE_EVAL_STAGES,
    _config,
    candidates,
)
from pgx_mcts_bench.leaderboard import leaderboard
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network


def _observation_tensor(observation: np.ndarray, device: str) -> torch.Tensor:
    return (
        torch.from_numpy(np.array(observation[None], copy=True))
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )


def auxiliary_statistics(
    auxiliary: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    failure_crossings: float = 20.0,
) -> dict[str, Any]:
    """Ensemble mean and quartiles for schedule decisions."""
    solve_logits, crossings, moves = auxiliary
    probability = solve_logits.sigmoid()[0]
    conditional_crossings = crossings[0]
    conditional_moves = moves[0]
    penalized_crossings = (
        probability * conditional_crossings
        + failure_crossings * (1.0 - probability)
    )

    def stats(values: torch.Tensor) -> dict[str, Any]:
        return {
            "members": [float(value) for value in values.tolist()],
            "mean": float(values.mean().item()),
            "q25": float(torch.quantile(values, 0.25).item()),
            "q75": float(torch.quantile(values, 0.75).item()),
        }

    return {
        "solve_probability": stats(probability),
        "conditional_crossings": stats(conditional_crossings),
        "conditional_moves": stats(conditional_moves),
        "penalized_crossings": stats(penalized_crossings),
        "failure_crossings": failure_crossings,
    }


def _held_out_instances(seed: int) -> list[dict[str, Any]]:
    """Generate each held-out word once so every critic sees identical states."""
    reference = next(candidate for candidate in candidates() if candidate.name == "u1-puct")
    config = _config(reference, VALUE_EVAL_STAGES[0], seed, "cpu")
    game = make_game(config.game)
    instances = []
    for offset, stage in enumerate(VALUE_EVAL_STAGES):
        source = next(source for source in game.generator.sources if source.name == stage[0])
        rng = np.random.default_rng(seed + 100_003 * (offset + 1))
        instance = game.generator.generate(source, stage[1], rng)
        instances.append(
            {
                "stage": STAGES.index(stage),
                "source": stage[0],
                "scramble": stage[1],
                "word": [int(letter) for letter in instance.word],
                "strands": int(instance.strands),
            }
        )
    return instances


def evaluate_value_heads(
    roots: list[Path], *, seed: int = 1_337_000, device: str = "cpu"
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate newest discovered candidate critics without running MCTS.

    Values are predictions, not realized returns.  The generated words and ratios
    are identical across candidates; only the observation formulation and critic
    weights differ.
    """
    standings, warnings = leaderboard(roots)
    by_name = {candidate.name: candidate for candidate in candidates()}
    instances = _held_out_instances(seed)
    rows = []

    for standing in standings:
        candidate = by_name.get(standing.name)
        if candidate is None:
            warnings.append(f"No candidate configuration for {standing.name}")
            continue
        saved = torch.load(standing.checkpoint, map_location=device, weights_only=False)
        auxiliary_trained = any(
            key.startswith("auxiliary.") for key in saved["network"]
        )
        initial = _config(candidate, STAGES[0], seed, device)
        network = make_braid_network(initial.game, initial.model).to(device)
        load_policy_value_state_dict(network, saved["network"])
        network.eval()
        inference_game = replace(
            initial.game,
            generator_max_crossings=0,
            generator_positive_braids=0,
            generator_random_crossings=(),
        )
        game = make_game(inference_game)

        predictions = []
        with torch.inference_mode():
            for instance in instances:
                values = {}
                for ratio in RATIOS:
                    transition = game.from_word(
                        instance["word"], instance["strands"], float(np.log(ratio))
                    )
                    observation = _observation_tensor(transition.observation, device)
                    if auxiliary_trained:
                        _, value, auxiliary = network.forward_with_auxiliary(observation)
                    else:
                        _, value = network(observation)
                        auxiliary = None
                    values[str(ratio)] = float(value.item())
                    if auxiliary is not None:
                        values[f"{ratio}:auxiliary"] = auxiliary_statistics(auxiliary)
                predictions.append(
                    {
                        "stage": instance["stage"],
                        "source": instance["source"],
                        "scramble": instance["scramble"],
                        "values": values,
                    }
                )
        rows.append(
            {
                "candidate": standing.name,
                "checkpoint": str(standing.checkpoint),
                "highest_stage": standing.highest_stage,
                "auxiliary_trained": auxiliary_trained,
                "predictions": predictions,
            }
        )

    return {
        "seed": seed,
        "ratios": list(RATIOS),
        "instances": instances,
        "candidates": rows,
    }, warnings


def save(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    output.with_suffix(".md").write_text(render(result))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _number(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:+.3f}"


def summarize_candidate(row: dict[str, Any]) -> dict[str, float]:
    by_ratio = {
        str(ratio): [prediction["values"][str(ratio)] for prediction in row["predictions"]]
        for ratio in RATIOS
    }
    ratio_10 = by_ratio["10.0"]
    scramble_deltas = []
    by_source = {
        (prediction["source"], prediction["scramble"]): prediction
        for prediction in row["predictions"]
    }
    for source, scramble in by_source:
        if scramble != 4 or (source, 0) not in by_source:
            continue
        scramble_deltas.append(
            by_source[(source, 4)]["values"]["10.0"]
            - by_source[(source, 0)]["values"]["10.0"]
        )
    return {
        "mean_1000": _mean(by_ratio["1000.0"]),
        "mean_10": _mean(ratio_10),
        "mean_0.1": _mean(by_ratio["0.1"]),
        "sd_10": float(np.std(ratio_10)),
        "scramble_delta_10": _mean(scramble_deltas),
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Held-out value-head evaluation",
        "",
        "Raw initial-state predictions; no MCTS was run, so these are not realized returns.",
        "",
        "| candidate | trained through | mean v(1000:1) | mean v(10:1) | "
        "mean v(1:10) | sd v(10:1) | mean Δv(+4, 10:1) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["candidates"]:
        summary = summarize_candidate(row)
        lines.append(
            f"| `{row['candidate']}` | {row['highest_stage']} | "
            f"{_number(summary['mean_1000'])} | {_number(summary['mean_10'])} | "
            f"{_number(summary['mean_0.1'])} | {summary['sd_10']:.3f} | "
            f"{_number(summary['scramble_delta_10'])} |"
        )
    lines += [
        "",
        "`Δv(+4)` is value on the scrambled word minus value on its matching `+0` source; "
        "negative means the critic predicts scrambling makes the state worse.",
        "",
        "## Per-rung values at A:B=10:1",
        "",
    ]
    labels = [f"{item['source']}+{item['scramble']}" for item in result["instances"]]
    lines.append("| candidate | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---:|" * len(labels))
    for row in result["candidates"]:
        values = [prediction["values"]["10.0"] for prediction in row["predictions"]]
        lines.append(f"| `{row['candidate']}` | " + " | ".join(_number(v) for v in values) + " |")
    return "\n".join(lines) + "\n"
