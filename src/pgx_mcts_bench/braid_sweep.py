"""Screen several braid-training approaches against one shared anchor set.

The point is to answer a question that a single run cannot: **does training help
at all, and which configuration helps most?** The control variant (`no-training`)
is not decoration -- an untrained network with 24 simulations already solves a
majority of K=3 anchors by search alone, so any claim about learning has to beat
that number, not zero.

Every variant is scored on the *same* frozen anchors with the *same* exact
optimal lengths, so the comparison is like-for-like. Variants that change `K`
generate different anchors and are reported separately.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pgx_mcts_bench.braid_progress import BraidProgress, evaluate_scrambler_difficulty
from pgx_mcts_bench.config import (
    BraidGameConfig,
    ExperimentConfig,
    ModelConfig,
    SearchConfig,
    TrainConfig,
)
from pgx_mcts_bench.networks import BraidAlphaZeroNet
from pgx_mcts_bench.training import TrainedAgent, evaluate_against_random, train_agent


@dataclass(frozen=True)
class Variant:
    name: str
    rationale: str
    exploration: str = "u1"
    simulations: int = 32
    channels: int = 32
    scramble_budget: int = 3
    iterations: int = 8
    selfplay_games: int = 8
    train_steps: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    temperature_moves: int = 12
    first_role_exploration_only: bool = False
    role_balanced_batches: bool = False
    max_children: int = 0
    dirichlet_scale: float = 0.0
    speed_bonus: float = 0.0
    value_head: str = "masked"
    train: bool = True


def default_variants(iterations: int, scramble_budget: int) -> list[Variant]:
    """Ten approaches worth separating before committing compute to any of them."""
    common = {"iterations": iterations, "scramble_budget": scramble_budget}
    return [
        Variant(
            "no-training",
            "control: search only, weights never updated. Everything else must beat this.",
            train=False,
            **common,
        ),
        Variant("u1-puct", "AlphaZero PUCT, the default", exploration="u1", **common),
        Variant(
            "u3-uct",
            "prior-free UCT: how much is the learned prior worth?",
            exploration="u3",
            **common,
        ),
        Variant("u4", "prior-weighted UCT; best mean and tightest spread so far",
                exploration="u4", **common),
        Variant("u5-muzero", "MuZero pb_c schedule", exploration="u5", **common),
        Variant(
            "search-heavy",
            "128 simulations: is this a search problem or a learning problem?",
            simulations=128,
            **common,
        ),
        Variant(
            "search-light",
            "8 simulations: forces the network to carry the policy",
            simulations=8,
            **common,
        ),
        Variant("wide-net", "96 channels: is the network underfitting?",
                channels=96, train_steps=128, **common),
        Variant("long-train", "4x the optimizer steps per iteration", train_steps=256, **common),
        Variant(
            "role-exploration",
            "FIX: explore only while scrambling; play the Simplifier sharply",
            first_role_exploration_only=True,
            **common,
        ),
        Variant(
            "balanced-batches",
            "FIX: equal gradient per role; the buffer is otherwise ~85% Simplifier",
            role_balanced_batches=True,
            **common,
        ),
        Variant(
            "topk-16",
            "FIX: expand only the 16 highest-prior children; 74% of actions are insertions",
            max_children=16,
            **common,
        ),
        Variant(
            "speed-bonus",
            "FIX: reward shorter solutions; solution length IS the unknotting bound",
            speed_bonus=0.3,
            **common,
        ),
        Variant(
            "value-flat",
            "A/B: reads every position; accurate but tied to one word capacity",
            value_head="flat",
            **common,
        ),
        Variant(
            "value-pooled",
            "A/B: pools over all L slots; length-agnostic but dilutes by ~6x",
            value_head="pooled",
            **common,
        ),
        Variant(
            "value-masked",
            "A/B: pools over occupied positions only; no dilution, still agnostic",
            value_head="masked",
            **common,
        ),
        Variant(
            "all-fixes",
            "the fixes that worked, bundled: role exploration + balanced batches",
            exploration="u4",
            first_role_exploration_only=True,
            role_balanced_batches=True,
            dirichlet_scale=10.0,
            **common,
        ),
    ]


@dataclass
class VariantResult:
    name: str
    rationale: str
    solve_rate: float
    mean_excess: float | None
    scrambler_win_rate: float
    simplifier_win_rate: float
    scrambler_depth: float
    scrambler_beyond_cutoff: float
    seconds: float
    final_loss: float | None
    history: list[dict[str, float]] = field(default_factory=list)
    per_iteration: list[dict[str, Any]] = field(default_factory=list)


def _experiment(
    variant: Variant, game: BraidGameConfig, seed: int, device: str
) -> ExperimentConfig:
    return ExperimentConfig(
        game=game,
        search=SearchConfig(
            simulations=variant.simulations,
            exploration=variant.exploration,  # type: ignore[arg-type]
            max_children=variant.max_children,
            root_dirichlet_scale=variant.dirichlet_scale,
        ),
        model=ModelConfig(
            channels=variant.channels,
            latent_channels=variant.channels,
            braid_value_head=variant.value_head,
        ),
        train=TrainConfig(
            iterations=variant.iterations,
            selfplay_games=variant.selfplay_games,
            train_steps=variant.train_steps,
            batch_size=variant.batch_size,
            learning_rate=variant.learning_rate,
            temperature_moves=variant.temperature_moves,
            first_role_exploration_only=variant.first_role_exploration_only,
            role_balanced_batches=variant.role_balanced_batches,
            seed=seed,
            device=device,
        ),
    )


def run_variant(
    variant: Variant,
    tier: BraidGameConfig,
    out: Path,
    *,
    anchors: int,
    baseline_games: int,
    seed: int,
    device: str,
    label: str | None = None,
) -> VariantResult:
    from dataclasses import replace

    label = label or variant.name
    game = replace(
        tier,
        scramble_budget=variant.scramble_budget,
        simplifier_speed_bonus=variant.speed_bonus,
    )
    config = _experiment(variant, game, seed, device)
    # Anchors are pinned to a fixed seed so that every variant AND every seed is
    # scored on the same instances.
    progress = BraidProgress(config, out / label, anchors=anchors, seed=10_000)
    started = time.perf_counter()

    if variant.train:
        per_iteration: list[dict[str, Any]] = []

        def hook(iteration: int, network) -> str:
            report = progress.evaluate(iteration, network)
            per_iteration.append(
                {
                    "iteration": iteration,
                    "solve_rate": report.solve_rate,
                    "mean_excess": report.mean_excess,
                }
            )
            return progress.summary_line(report)

        agent = train_agent("alphazero", config, iteration_hook=hook)
    else:
        # Seed before construction. An untrained policy prior still steers MCTS,
        # so the control's score depends strongly on its random init -- at K=4,
        # 32 simulations, two inits gave 0.00 and 0.50 on the same anchors. A
        # single-seed control is not a control.
        torch.manual_seed(config.train.seed)
        network = BraidAlphaZeroNet(game, config.model)
        report = progress.evaluate(0, network)
        per_iteration = [
            {"iteration": 0, "solve_rate": report.solve_rate, "mean_excess": report.mean_excess}
        ]
        agent = TrainedAgent("alphazero", network, [], config)

    final = progress.reports[-1]
    baseline = evaluate_against_random(agent, baseline_games, seed=seed + 500_000)
    difficulty = evaluate_scrambler_difficulty(agent, baseline_games, seed=seed + 700_000)
    return VariantResult(
        name=label,
        rationale=variant.rationale,
        solve_rate=final.solve_rate,
        mean_excess=final.mean_excess,
        scrambler_win_rate=float(baseline["first_role_win_rate"]),
        simplifier_win_rate=float(baseline["second_role_win_rate"]),
        scrambler_depth=float(difficulty["mean_optimal_depth"]),
        scrambler_beyond_cutoff=float(difficulty["beyond_cutoff"]),
        seconds=time.perf_counter() - started,
        final_loss=float(agent.history[-1]["loss"]) if agent.history else None,
        history=agent.history,
        per_iteration=per_iteration,
    )


def _worker_init() -> None:
    """Pin each worker to one thread.

    The nets are tiny (40k parameters, batch 8), so BLAS threading buys nothing
    and 8 workers x 4 threads on 8 cores would thrash. One thread per worker,
    parallelism across runs instead of inside them.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)


def _run_job(payload: tuple) -> VariantResult:
    variant, tier, out, anchors, baseline_games, seed, device, label = payload
    return run_variant(
        variant,
        tier,
        out,
        anchors=anchors,
        baseline_games=baseline_games,
        seed=seed,
        device=device,
        label=label,
    )


def render_summary(results: list[VariantResult], control: VariantResult | None) -> str:
    lines = [
        "# Braid approach screen",
        "",
        "All variants scored on the **same frozen anchor set** with exact optimal",
        "solution lengths from breadth-first search.",
        "",
        "* `solve rate` — fraction of anchors untied within the Simplifier's budget.",
        "* `excess` — moves used beyond a shortest solution, averaged over solved anchors.",
        "  Lower is better; the reward is win/lose, so nothing directly optimises this.",
        "* `Simp vs rnd` — Simplifier win rate against a uniform-random Scrambler.",
        "* `Scr depth` — exact BFS-optimal depth of the instances this Scrambler makes.",
        "  A uniform-random Scrambler at K=4 scores 3.16; that is the number to beat.",
        "  `>cutoff` is the fraction too hard for BFS to solve — the interesting ones.",
        "",
    ]
    if control is not None:
        lines += [
            f"**Control (`{control.name}`): solve rate {control.solve_rate:.2f}, "
            f"excess {control.mean_excess}.** A variant that does not beat this has "
            "learned nothing that search was not already doing.",
            "",
        ]
    lines += [
        "| variant | solve rate | excess | Simp vs rnd | Scr depth | Scr >cutoff "
        "| final loss | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda r: (-r.solve_rate, r.mean_excess or 1e9)):
        excess = "-" if result.mean_excess is None else f"{result.mean_excess:+.2f}"
        loss = "-" if result.final_loss is None else f"{result.final_loss:.3f}"
        lines.append(
            f"| `{result.name}` | {result.solve_rate:.3f} | {excess} "
            f"| {result.simplifier_win_rate:.2f} | {result.scrambler_depth:.2f} "
            f"| {result.scrambler_beyond_cutoff:.2f} | {loss} | {result.seconds:.0f} |"
        )
    lines += ["", "## What each variant was testing", ""]
    for result in results:
        lines.append(f"* `{result.name}` — {result.rationale}")
    lines += ["", "## Solve rate by iteration", ""]
    for result in results:
        trace = " ".join(f"{row['solve_rate']:.2f}" for row in result.per_iteration)
        lines.append(f"* `{result.name}`: {trace}")
    return "\n".join(lines) + "\n"


def run_sweep(
    variants: list[Variant],
    tier: BraidGameConfig,
    out: Path,
    *,
    anchors: int = 12,
    baseline_games: int = 10,
    seed: int = 0,
    seeds: int = 1,
    device: str = "cpu",
    workers: int = 1,
    log=print,
) -> list[VariantResult]:
    out.mkdir(parents=True, exist_ok=True)
    results: list[VariantResult] = []
    jobs = [
        (
            variant,
            tier,
            out,
            anchors,
            baseline_games,
            seed + offset,
            device,
            variant.name if seeds == 1 else f"{variant.name}#s{seed + offset}",
        )
        for variant in variants
        for offset in range(seeds)
    ]

    def record(result: VariantResult) -> None:
        results.append(result)
        excess = "-" if result.mean_excess is None else f"{result.mean_excess:+.2f}"
        log(
            f"[{len(results)}/{len(jobs)}] {result.name}: solve {result.solve_rate:.3f}  "
            f"excess {excess}  Scr-depth {result.scrambler_depth:.2f}  {result.seconds:.0f}s"
        )
        control = next((r for r in results if r.name.startswith("no-training")), None)
        (out / "summary.json").write_text(
            json.dumps([vars(r) for r in results], indent=2, default=float) + "\n"
        )
        (out / "summary.md").write_text(render_summary(results, control))

    if workers <= 1:
        for payload in jobs:
            record(_run_job(payload))
        return results

    # The runs share nothing: separate output directories, separate seeds, and
    # results that depend only on (variant, seed). So parallel execution gives
    # byte-identical results, just sooner.
    log(f"running {len(jobs)} jobs across {workers} workers")
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_run_job, payload): payload[-1] for payload in jobs}
        for future in as_completed(futures):
            record(future.result())
    return results


def best_variant(results: list[VariantResult]) -> VariantResult:
    """Highest solve rate, ties broken by fewer wasted moves."""
    return min(results, key=lambda r: (-r.solve_rate, r.mean_excess if r.mean_excess else 0.0))


__all__ = [
    "Variant",
    "VariantResult",
    "best_variant",
    "default_variants",
    "render_summary",
    "run_sweep",
    "run_variant",
]
