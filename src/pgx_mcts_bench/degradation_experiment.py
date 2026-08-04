"""Matched offline forks for measuring degradation from shared witnesses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal

import numpy as np
import torch

from pgx_mcts_bench.adaptive_scientists import load_scientist
from pgx_mcts_bench.collaboration_eval import _evaluation_record
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _json_hash,
    _restore_scientist,
    load_round_state,
)
from pgx_mcts_bench.data import GameRecord, Position
from pgx_mcts_bench.training import train_alphazero_step

Variant = Literal["pre", "rl0", "d1-full", "d10-full", "d1-aux", "d10-aux"]
VARIANTS: tuple[Variant, ...] = (
    "pre",
    "rl0",
    "d1-full",
    "d10-full",
    "d1-aux",
    "d10-aux",
)


@dataclass
class FixedBatchReplay:
    batches: list[list[Position]]
    cursor: int = 0

    def sample_positions(self, batch_size: int) -> list[Position]:
        batch = self.batches[self.cursor]
        self.cursor += 1
        if len(batch) != batch_size:
            raise ValueError(f"expected batch size {batch_size}, got {len(batch)}")
        return batch


def _is_shared(game: GameRecord) -> bool:
    return bool(game and getattr(game[0], "shared_witness", False))


def _objective(game: GameRecord, ratio: float) -> float:
    if not game:
        return float("inf")
    return ratio * float(game[0].final_crossing_changes) + float(game[0].final_moves)


def select_shared_games(games: list[GameRecord], dose: int, ratio: float) -> list[GameRecord]:
    """Select the lowest-cost receiver-native witnesses deterministically."""
    shared = [
        game
        for game in games
        if _is_shared(game)
        and any(not getattr(position, "objective_censored", False) for position in game)
    ]
    shared.sort(
        key=lambda game: (
            _objective(game, ratio),
            int(getattr(game[0], "episode_seed", 0)),
            len(game),
        )
    )
    return shared[:dose]


def matched_batches(
    games: list[GameRecord],
    selected_shared: list[GameRecord],
    *,
    train_steps: int,
    batch_size: int,
    shared_fraction: float,
    seed: int,
) -> tuple[list[list[Position]], list[list[Position]]]:
    """Return native control and treatment batches with common native prefixes.

    Shared positions are sampled episode-uniformly, then position-uniformly, so a
    long translated trajectory does not receive more weight merely because it is
    long. Treatment slots replace the tail of the matched native control batch.
    """
    native = [
        position
        for game in games
        if not _is_shared(game)
        for position in game
        if not getattr(position, "objective_censored", False)
    ]
    if not native:
        raise ValueError("no native replay positions")
    rng = np.random.default_rng(seed)
    shared_count = (
        min(batch_size - 1, max(1, round(batch_size * shared_fraction))) if selected_shared else 0
    )
    controls: list[list[Position]] = []
    treatments: list[list[Position]] = []
    for _ in range(train_steps):
        native_indexes = rng.integers(0, len(native), size=batch_size)
        control = [native[int(index)] for index in native_indexes]
        treatment = list(control[: batch_size - shared_count])
        for _ in range(shared_count):
            game = selected_shared[int(rng.integers(0, len(selected_shared)))]
            treatment.append(game[int(rng.integers(0, len(game)))])
        controls.append(control)
        treatments.append(treatment)
    return controls, treatments


def train_degradation_forks(
    run: Path,
    output: Path,
    *,
    round_index: int = 48,
    training_seeds: tuple[int, ...] = (0, 1, 2),
    train_steps: int = 8,
    batch_size: int = 32,
    shared_fraction: float = 0.1,
    variants: tuple[Variant, ...] = VARIANTS,
    learning_rate: float | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    run_manifest = json.loads((run / "manifest.json").read_text())
    ratio = float(run_manifest["ratios"][0])
    if len(run_manifest["ratios"]) != 1:
        raise ValueError("degradation gate requires a single objective ratio")
    round_dir = run / "rounds" / f"{round_index:06d}"
    if not round_dir.is_dir():
        raise FileNotFoundError(round_dir)
    protocol = {
        "schema": "distillation-degradation-v1",
        "run": str(run.resolve()),
        "run_protocol_sha256": run_manifest["protocol_sha256"],
        "round_index": round_index,
        "training_seeds": list(training_seeds),
        "train_steps": train_steps,
        "batch_size": batch_size,
        "shared_fraction": shared_fraction,
        "ratio": ratio,
        "variants": list(variants),
        "learning_rate": learning_rate,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "manifest.json", protocol)

    reports: list[dict[str, Any]] = []
    for training_seed in training_seeds:
        for variant in variants:
            saved = load_round_state(round_dir, map_location=device)
            for scientist_index, source in enumerate(run_manifest["checkpoints"]):
                name = source["name"]
                scientist = load_scientist(
                    name,
                    Path(source["path"]),
                    seed=training_seed + scientist_index * 10_000,
                    device=device,
                    require_factorized=True,
                    objective_budget_channel=bool(run_manifest.get("objective_budget", False)),
                )
                _restore_scientist(scientist, saved["scientists"][name])
                if learning_rate is not None:
                    for parameter_group in scientist.optimizer.param_groups:
                        parameter_group["lr"] = learning_rate
                games = scientist.replay.games
                dose = 1 if variant.startswith("d1-") else 10
                selected = (
                    select_shared_games(games, dose, ratio) if variant.startswith("d") else []
                )
                control, treatment = matched_batches(
                    games,
                    selected,
                    train_steps=train_steps,
                    batch_size=batch_size,
                    shared_fraction=shared_fraction,
                    seed=training_seed * 1_000_000 + scientist_index * 10_000 + 71,
                )
                losses = []
                if variant != "pre":
                    scientist.network.shared_auxiliary_only = variant.endswith("-aux")
                    batches = treatment if selected else control
                    replay = FixedBatchReplay(batches)
                    for _ in range(train_steps):
                        losses.append(
                            train_alphazero_step(
                                scientist.network,
                                scientist.optimizer,
                                replay,  # type: ignore[arg-type]
                                batch_size,
                                torch.device(device),
                            )
                        )
                target = output / "checkpoints" / f"seed-{training_seed}" / variant
                target.mkdir(parents=True, exist_ok=True)
                checkpoint = target / f"{name}.pt"
                torch.save(
                    {
                        "network": scientist.network.state_dict(),
                        "degradation": {
                            "variant": variant,
                            "training_seed": training_seed,
                            "round_index": round_index,
                            "selected_shared_episode_seeds": [
                                int(getattr(game[0], "episode_seed", 0)) for game in selected
                            ],
                        },
                    },
                    checkpoint,
                )
                reports.append(
                    {
                        "training_seed": training_seed,
                        "variant": variant,
                        "scientist": name,
                        "checkpoint": str(checkpoint.resolve()),
                        "native_games": sum(not _is_shared(game) for game in games),
                        "available_shared_games": sum(_is_shared(game) for game in games),
                        "selected_shared_games": len(selected),
                        "selected_shared_objectives": [
                            _objective(game, ratio) for game in selected
                        ],
                        "last_loss": losses[-1] if losses else None,
                    }
                )
    report = {**protocol, "forks": reports}
    _atomic_json(output / "training-report.json", report)
    return report


def _summarize(
    rows: list[dict[str, Any]], scientist_names: list[str], ratio: float, move_budget: int
) -> dict[str, Any]:
    def one(name: str | None) -> dict[str, Any]:
        solved_items: list[str] = []
        best_by_item: dict[str, dict[str, Any]] = {}
        capped = 0.0
        for row in rows:
            attempts = [
                attempt
                for attempt in row["attempts"]
                if attempt["solved"]
                and attempt["ratio"] == ratio
                and (name is None or attempt["scientist"] == name)
            ]
            if attempts:
                best = min(
                    attempts,
                    key=lambda attempt: (
                        attempt["objective"],
                        attempt["crossing_changes"],
                        attempt["moves"],
                    ),
                )
                solved_items.append(row["item"])
                capped += best["objective"]
                best_by_item[row["item"]] = {
                    key: best[key]
                    for key in ("scientist", "crossing_changes", "moves", "objective")
                }
            else:
                capped += ratio * 20 + move_budget
        return {
            "portfolio_solved": len(solved_items),
            "representations": len(rows),
            "capped_objective_sum": capped,
            "solved_items": sorted(solved_items),
            "best_by_item": best_by_item,
        }

    return {
        "portfolio": one(None),
        "scientists": {name: one(name) for name in scientist_names},
    }


def evaluate_degradation_fork(
    experiment: Path,
    output: Path,
    *,
    training_seed: int,
    variant: Variant,
    split: Literal["base", "new70"],
    simulations: int = 16,
    limit: int = 0,
    evaluation_seed: int = 0,
    device: str = "cpu",
    resume: bool = False,
    bank: Path | None = None,
) -> dict[str, Any]:
    experiment_manifest = json.loads((experiment / "manifest.json").read_text())
    run = Path(experiment_manifest["run"])
    run_manifest = json.loads((run / "manifest.json").read_text())
    source = bank if bank is not None else run / ("base.json" if split == "base" else "new-70.json")
    items = _bank_from_payload(json.loads(source.read_text()))
    if limit:
        items = items[:limit]
    ratio = float(experiment_manifest["ratio"])
    protocol = {
        "schema": "distillation-degradation-evaluation-v1",
        "experiment_protocol_sha256": experiment_manifest["protocol_sha256"],
        "training_seed": training_seed,
        "variant": variant,
        "split": split,
        "split_sha256": _json_hash(json.loads(source.read_text())),
        "simulations": simulations,
        "limit": limit,
        "evaluation_seed": evaluation_seed,
        "external_bank": str(bank.resolve()) if bank is not None else None,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; pass resume=True")
        if json.loads(manifest_path.read_text())["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("evaluation resume protocol differs")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest_path, protocol)

    scientists = []
    for scientist_index, original in enumerate(run_manifest["checkpoints"]):
        scientist = load_scientist(
            original["name"],
            Path(original["path"]),
            seed=evaluation_seed + scientist_index * 10_000,
            device=device,
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=bool(run_manifest.get("objective_budget", False)),
        )
        checkpoint = (
            experiment / "checkpoints" / f"seed-{training_seed}" / variant / f"{scientist.name}.pt"
        )
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        scientist.network.load_state_dict(payload["network"])
        scientists.append(scientist)

    item_dir = output / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    for item_index, item in enumerate(items):
        path = item_dir / f"{item_index:04d}.json"
        if path.exists():
            continue
        attempts = []
        for scientist_index, scientist in enumerate(scientists):
            seed = evaluation_seed + item_index * 1_000_000 + scientist_index * 10_000
            verified, compute = _evaluation_record(scientist, item.knot, ratio, simulations, seed)
            attempt = {
                "scientist": scientist.name,
                "ratio": ratio,
                "solved": verified is not None,
                "compute": compute,
            }
            if verified is not None:
                attempt.update(
                    crossing_changes=verified[0],
                    moves=verified[1],
                    objective=ratio * verified[0] + verified[1],
                )
            attempts.append(attempt)
        _atomic_json(path, {"item": item.id, "attempts": attempts})
    rows = [json.loads(path.read_text()) for path in sorted(item_dir.glob("*.json"))]
    summary = _summarize(
        rows,
        [scientist.name for scientist in scientists],
        ratio,
        int(scientists[0].config.game.simplify_budget),
    )
    report = {**protocol, "completed_items": len(rows), "summary": summary}
    _atomic_json(output / "report.json", report)
    return report


def analyze_degradation_experiment(evaluations: Path, output: Path | None = None) -> dict[str, Any]:
    """Aggregate paired treatment effects and apply the conservative safety gate."""
    seed_dirs = sorted(evaluations.glob("seed-[0-9]*"))
    seeds = sorted(
        int(path.name.removeprefix("seed-")) for path in seed_dirs if (path / "rl0").is_dir()
    )
    if not seeds:
        raise ValueError("no matched RL0 evaluation directories")
    variants = sorted(
        path.name
        for path in (evaluations / f"seed-{seeds[0]}").iterdir()
        if path.is_dir() and path.name not in {"pre", "rl0"}
    )
    pre = {}
    for split in ("base50", "new70"):
        pre_path = evaluations / "seed-0" / "pre" / split / "report.json"
        pre[split] = json.loads(pre_path.read_text())["summary"]

    treatments: dict[str, Any] = {}
    for variant in variants:
        treatment_splits = {}
        accepted = True
        for split in ("base50", "new70"):
            paired = []
            for seed in seeds:
                treatment = json.loads(
                    (evaluations / f"seed-{seed}" / variant / split / "report.json").read_text()
                )["summary"]
                control = json.loads(
                    (evaluations / f"seed-{seed}" / "rl0" / split / "report.json").read_text()
                )["summary"]
                treatment_portfolio = treatment["portfolio"]
                control_portfolio = control["portfolio"]
                treatment_solved = set(treatment_portfolio["solved_items"])
                control_solved = set(control_portfolio["solved_items"])
                row = {
                    "seed": seed,
                    "solve_delta": (
                        treatment_portfolio["portfolio_solved"]
                        - control_portfolio["portfolio_solved"]
                    ),
                    "capped_loss_delta": (
                        treatment_portfolio["capped_objective_sum"]
                        - control_portfolio["capped_objective_sum"]
                    ),
                    "treatment_only": sorted(treatment_solved - control_solved),
                    "control_only": sorted(control_solved - treatment_solved),
                    "scientists": {},
                }
                for name in treatment["scientists"]:
                    scientist_treatment = treatment["scientists"][name]
                    scientist_control = control["scientists"][name]
                    row["scientists"][name] = {
                        "solve_delta": (
                            scientist_treatment["portfolio_solved"]
                            - scientist_control["portfolio_solved"]
                        ),
                        "capped_loss_delta": (
                            scientist_treatment["capped_objective_sum"]
                            - scientist_control["capped_objective_sum"]
                        ),
                    }
                paired.append(row)
            split_accepted = all(
                row["solve_delta"] >= 0 and row["capped_loss_delta"] <= 0 for row in paired
            )
            accepted &= split_accepted
            treatment_splits[split] = {
                "paired": paired,
                "median_solve_delta": median(row["solve_delta"] for row in paired),
                "median_capped_loss_delta": median(row["capped_loss_delta"] for row in paired),
                "passes_all_seeds": split_accepted,
            }
        treatments[variant] = {
            "splits": treatment_splits,
            "accepted": accepted,
        }

    rl0_vs_pre = {}
    for split in ("base50", "new70"):
        rows = []
        pre_portfolio = pre[split]["portfolio"]
        pre_solved = set(pre_portfolio["solved_items"])
        for seed in seeds:
            control = json.loads(
                (evaluations / f"seed-{seed}" / "rl0" / split / "report.json").read_text()
            )["summary"]["portfolio"]
            solved = set(control["solved_items"])
            rows.append(
                {
                    "seed": seed,
                    "solve_delta": control["portfolio_solved"] - pre_portfolio["portfolio_solved"],
                    "capped_loss_delta": control["capped_objective_sum"]
                    - pre_portfolio["capped_objective_sum"],
                    "rl0_only": sorted(solved - pre_solved),
                    "pre_only": sorted(pre_solved - solved),
                }
            )
        rl0_vs_pre[split] = rows

    accepted = [variant for variant, row in treatments.items() if row["accepted"]]
    report = {
        "schema": "distillation-degradation-decision-v1",
        "training_seeds": seeds,
        "gate": (
            "For both BASE50 and NEW70, every seed must have solve_delta >= 0 "
            "and capped_loss_delta <= 0 versus its matched RL0 control."
        ),
        "rl0_vs_pre": rl0_vs_pre,
        "treatments_vs_rl0": treatments,
        "accepted_treatments": accepted,
        "decision": (
            "stop_current_training_and_distillation" if not accepted else f"admit_{accepted[0]}"
        ),
        "next_gate": (
            "Test a rollback-guarded update with one auxiliary-only witness, one "
            "shared slot per batch, success-balanced native rehearsal, and a lower "
            "learning rate. Keep the pre-update checkpoint unless a frozen BASE "
            "canary is non-inferior; do not consult NEW70 for rollback decisions."
        ),
    }
    if output is not None:
        _atomic_json(output, report)
    return report
