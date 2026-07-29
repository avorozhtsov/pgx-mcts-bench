from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from pgx_mcts_bench.braid_progress import BraidProgress
from pgx_mcts_bench.braid_sweep import default_variants, run_sweep
from pgx_mcts_bench.config import (
    BraidGameConfig,
    ExperimentConfig,
    GameConfig,
    ModelConfig,
    SearchConfig,
    TrainConfig,
    artifact_dir,
)
from pgx_mcts_bench.exploration import describe_rules
from pgx_mcts_bench.training import (
    compare_agents,
    compare_pair,
    evaluate_against_random,
    evaluate_learning_curve,
    load_agent,
    save_braid_experiment,
    save_experiment,
    train_agent,
)

app = typer.Typer(no_args_is_help=True)


def _config(
    exploration: str,
    simulations: int,
    iterations: int,
    selfplay_games: int,
    selfplay_positions: int,
    train_steps: int,
    batch_size: int,
    channels: int,
    seed: int,
    device: str,
    checkpoint_iterations: tuple[int, ...] = (),
    learning_curve_games: int = 0,
    exact_position_budget: bool = True,
) -> ExperimentConfig:
    if exploration not in describe_rules():
        raise typer.BadParameter(f"exploration must be one of {', '.join(describe_rules())}")
    return ExperimentConfig(
        game=GameConfig(),
        search=SearchConfig(simulations=simulations, exploration=exploration),  # type: ignore[arg-type]
        model=ModelConfig(channels=channels, latent_channels=channels),
        train=TrainConfig(
            iterations=iterations,
            selfplay_games=selfplay_games,
            selfplay_positions_per_iteration=selfplay_positions,
            train_steps=train_steps,
            batch_size=batch_size,
            seed=seed,
            device=device,
            exact_position_budget=exact_position_budget,
            checkpoint_iterations=checkpoint_iterations,
            learning_curve_games=learning_curve_games,
        ),
    )


def _iteration_list(value: str, final_iteration: int) -> tuple[int, ...]:
    try:
        values = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as error:
        raise typer.BadParameter(
            "checkpoint iterations must be comma-separated integers"
        ) from error
    if any(iteration < 1 for iteration in values):
        raise typer.BadParameter("checkpoint iterations must be positive")
    values.add(final_iteration)
    return tuple(sorted(iteration for iteration in values if iteration <= final_iteration))


@app.command()
def rules() -> None:
    """Print the U1-U5 exploration rules."""
    for name, formula in describe_rules().items():
        typer.echo(f"{name}: {formula}")


@app.command()
def compare(
    exploration: Annotated[str, typer.Option(help="One of u1, u2, u3, u4, u5")] = "u1",
    simulations: Annotated[int, typer.Option(min=1)] = 16,
    iterations: Annotated[int, typer.Option(min=1)] = 2,
    selfplay_games: Annotated[int, typer.Option(min=1)] = 1,
    selfplay_positions: Annotated[
        int,
        typer.Option(min=0, help="Minimum positions per iteration; 0 means one game batch"),
    ] = 0,
    train_steps: Annotated[int, typer.Option(min=1)] = 4,
    batch_size: Annotated[int, typer.Option(min=1)] = 8,
    arena_games: Annotated[int, typer.Option(min=2)] = 4,
    channels: Annotated[int, typer.Option(min=4)] = 16,
    seed: int = 0,
    device: str = "cpu",
    output: Path | None = None,
    resume: Annotated[
        bool,
        typer.Option(help="Resume each agent from its latest checkpoint in the output directory"),
    ] = False,
    checkpoint_iterations: Annotated[
        str,
        typer.Option(help="Comma-separated iterations to checkpoint"),
    ] = "1,3,10,20,40",
    curve_games: Annotated[
        int,
        typer.Option(min=0, help="Arena games per saved checkpoint; 0 disables curves"),
    ] = 0,
    exact_positions: Annotated[
        bool,
        typer.Option(
            "--exact-positions/--minimum-positions",
            help="Keep exactly the requested positions or retain whole generated games",
        ),
    ] = True,
) -> None:
    """Train compact AlphaZero and MuZero agents, then play a color-balanced arena."""
    checkpoints = _iteration_list(checkpoint_iterations, iterations)
    config = _config(
        exploration,
        simulations,
        iterations,
        selfplay_games,
        selfplay_positions,
        train_steps,
        batch_size,
        channels,
        seed,
        device,
        checkpoints,
        curve_games,
        exact_positions,
    )
    typer.echo(
        f"6x6 Go, komi={config.game.komi}, max_moves={config.game.max_moves}, "
        f"{simulations} simulations, rule={exploration}"
    )
    if resume and output is None:
        raise typer.BadParameter("--resume requires --output")
    label = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = output or artifact_dir(Path.cwd(), label)
    checkpoint_dir = out / "checkpoints"
    alphazero = train_agent(
        "alphazero",
        config,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )
    muzero = train_agent(
        "muzero",
        config,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )
    arena = compare_agents(alphazero, muzero, config, arena_games)
    save_experiment(out, config, alphazero, muzero, arena)
    learning_curve = evaluate_learning_curve(out, config, curve_games)
    save_experiment(out, config, alphazero, muzero, arena, learning_curve)
    typer.echo(f"Arena: {arena}")
    typer.echo(f"Saved: {out / 'results.json'}")


@app.command()
def smoke(output: Path | None = None) -> None:
    """Fast end-to-end check; its match result is not statistically meaningful."""
    compare(
        exploration="u1",
        simulations=2,
        iterations=1,
        selfplay_games=1,
        selfplay_positions=0,
        train_steps=1,
        batch_size=2,
        arena_games=2,
        channels=4,
        seed=0,
        device="cpu",
        output=output,
        resume=False,
        checkpoint_iterations="1",
        curve_games=0,
        exact_positions=True,
    )


BRAID_TIERS: dict[str, BraidGameConfig] = {
    "tier0": BraidGameConfig(
        max_len=32, max_strands=5, scramble_budget=6, simplify_budget=24
    ),
    "tier1": BraidGameConfig(
        max_len=64, max_strands=8, scramble_budget=12, simplify_budget=48
    ),
}


@app.command()
def braid(
    tier: Annotated[str, typer.Option(help="tier0 (small) or tier1")] = "tier0",
    scramble_budget: Annotated[
        int, typer.Option(min=1, help="K, the difficulty dial; 0 keeps the tier default")
    ] = 0,
    exploration: Annotated[str, typer.Option(help="One of u1, u2, u3, u4, u5")] = "u1",
    simulations: Annotated[int, typer.Option(min=1)] = 32,
    iterations: Annotated[int, typer.Option(min=1)] = 10,
    selfplay_games: Annotated[int, typer.Option(min=1)] = 8,
    selfplay_positions: Annotated[int, typer.Option(min=0)] = 0,
    train_steps: Annotated[int, typer.Option(min=1)] = 32,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    channels: Annotated[int, typer.Option(min=4)] = 32,
    baseline_games: Annotated[
        int, typer.Option(min=1, help="Games per role against a uniform-random opponent")
    ] = 20,
    anchors: Annotated[
        int,
        typer.Option(min=0, help="Frozen instances evaluated after each iteration; 0 disables"),
    ] = 16,
    seed: Annotated[int, typer.Option()] = 0,
    device: Annotated[str, typer.Option()] = "cpu",
    output: Annotated[Path | None, typer.Option()] = None,
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train AlphaZero on Scrambler vs. Simplifier over braid words.

    Reports each role's win rate against a uniform-random opponent, which is the
    only measurement with a known baseline: an agent that learns nothing scores
    about 0.016 as Simplifier at tier-0 K=6.
    """
    if tier not in BRAID_TIERS:
        raise typer.BadParameter(f"tier must be one of {', '.join(BRAID_TIERS)}")
    if exploration not in describe_rules():
        raise typer.BadParameter(f"exploration must be one of {', '.join(describe_rules())}")
    game_config = BRAID_TIERS[tier]
    if scramble_budget:
        game_config = replace(game_config, scramble_budget=scramble_budget)

    config = ExperimentConfig(
        game=game_config,
        search=SearchConfig(simulations=simulations, exploration=exploration),  # type: ignore[arg-type]
        model=ModelConfig(channels=channels, latent_channels=channels),
        train=TrainConfig(
            iterations=iterations,
            selfplay_games=selfplay_games,
            selfplay_positions_per_iteration=selfplay_positions,
            train_steps=train_steps,
            batch_size=batch_size,
            seed=seed,
            device=device,
            checkpoint_iterations=(iterations,),
        ),
    )
    typer.echo(
        f"braid {tier}: L={game_config.max_len}, N={game_config.max_strands}, "
        f"K={game_config.scramble_budget}, M={game_config.simplify_budget}, "
        f"{game_config.action_size} actions, {simulations} simulations, rule={exploration}"
    )
    if resume and output is None:
        raise typer.BadParameter("--resume requires --output")
    label = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = output or artifact_dir(Path.cwd(), f"braid-{label}")
    progress = BraidProgress(config, out, anchors=anchors, seed=seed + 10_000) if anchors else None

    def hook(iteration: int, network) -> str | None:
        if progress is None:
            return None
        return progress.summary_line(progress.evaluate(iteration, network))

    agent = train_agent(
        "alphazero",
        config,
        checkpoint_dir=out / "checkpoints",
        resume=resume,
        iteration_hook=hook,
    )
    baseline = evaluate_against_random(agent, baseline_games, seed=seed + 500_000)
    save_braid_experiment(out, config, agent, baseline)
    typer.echo(
        f"vs random -- as Scrambler: {baseline['first_role_win_rate']:.3f}, "
        f"as Simplifier: {baseline['second_role_win_rate']:.3f}"
    )
    typer.echo(f"Saved: {out / 'results.json'}")
    if progress is not None:
        typer.echo(f"Progress report: {out / 'progress.md'}")


@app.command()
def braid_smoke(output: Path | None = None) -> None:
    """Fast end-to-end braid check; its numbers are not statistically meaningful."""
    braid(
        tier="tier0",
        scramble_budget=3,
        exploration="u1",
        simulations=2,
        iterations=1,
        selfplay_games=2,
        selfplay_positions=0,
        train_steps=1,
        batch_size=2,
        channels=4,
        baseline_games=2,
        anchors=3,
        seed=0,
        device="cpu",
        output=output,
        resume=False,
    )


@app.command()
def braid_ladder(
    candidates_only: Annotated[str, typer.Option("--only", help="Comma-separated names")] = "",
    seed: Annotated[int, typer.Option()] = 0,
    max_iterations: Annotated[int, typer.Option(min=1)] = 25,
    eval_games: Annotated[int, typer.Option(min=4)] = 16,
    promote_at: Annotated[float, typer.Option()] = 0.8,
    workers: Annotated[int, typer.Option(min=1)] = 1,
    device: Annotated[str, typer.Option()] = "cpu",
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Climb the complexity ladder; score is the highest stage cleared."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from pgx_mcts_bench.braid_sweep import _worker_init, enable_jax_compilation_cache
    from pgx_mcts_bench.ladder import STAGES, _silent, candidates, run_ladder, save

    enable_jax_compilation_cache()
    chosen = candidates()
    if candidates_only:
        wanted = {n.strip() for n in candidates_only.split(",") if n.strip()}
        chosen = [c for c in chosen if c.name in wanted]
    out = output or artifact_dir(Path.cwd(), "ladder")
    typer.echo(f"{len(chosen)} candidates over {len(STAGES)} stages, {workers} workers")
    for index, stage in enumerate(STAGES):
        typer.echo(f"  stage {index}: {stage[0]} + {stage[1]} scramble moves")

    results = []
    if workers <= 1:
        for candidate in chosen:
            results.append(
                run_ladder(candidate, seed=seed, device=device,
                           checkpoint_dir=out / "checkpoints",
                           max_iterations_per_stage=max_iterations,
                           eval_games=eval_games, promote_at=promote_at, log=typer.echo)
            )
            save(results, out)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            futures = {
                pool.submit(run_ladder, c, seed=seed, device=device,
                            checkpoint_dir=out / "checkpoints",
                            max_iterations_per_stage=max_iterations,
                            eval_games=eval_games, promote_at=promote_at, log=_silent): c
                for c in chosen
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                typer.echo(
                    f"  [{len(results)}/{len(chosen)}] {result.name}: "
                    f"highest stage {result.highest_stage}  {result.seconds:.0f}s"
                )
                save(results, out)
    typer.echo(f"Saved: {out / 'ladder.md'}")


@app.command()
def braid_multi(
    tier: str = "tier0",
    max_crossings: Annotated[int, typer.Option(min=0)] = 5,
    max_scramble: Annotated[int, typer.Option(min=0)] = 3,
    simulations: Annotated[int, typer.Option(min=1)] = 48,
    iterations: Annotated[int, typer.Option(min=1)] = 12,
    selfplay_games: Annotated[int, typer.Option(min=1)] = 8,
    train_steps: Annotated[int, typer.Option(min=1)] = 64,
    eval_games: Annotated[int, typer.Option(min=1)] = 12,
    seed: Annotated[int, typer.Option()] = 0,
    device: Annotated[str, typer.Option()] = "cpu",
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Train on `A*crossing_changes + B*total_moves` and score against theorems.

    Instances come from the graded generator, so every source knot has a *proved*
    unknotting number -- u(T(p,q)) = (p-1)(q-1)/2. The question is whether the
    agent reaches it, and whether the crossing-change/move trade-off actually
    moves with log(A/B) rather than collapsing to one compromise policy.
    """
    import json as _json
    from dataclasses import replace as _replace

    import numpy as np

    from pgx_mcts_bench.game import BraidUnknotGame
    from pgx_mcts_bench.search import NeuralMCTS

    base = BRAID_TIERS[tier]
    game_cfg = _replace(
        base,
        max_len=48,
        simplify_budget=32,
        allow_crossing_change=True,
        multi_objective=True,
        log_ratio_range=(-3.0, 3.0),
        generator_max_crossings=max_crossings,
        generator_max_scramble=max_scramble,
    )
    config = ExperimentConfig(
        game=game_cfg,
        search=SearchConfig(simulations=simulations),
        model=ModelConfig(channels=32, latent_channels=32),
        train=TrainConfig(
            iterations=iterations,
            selfplay_games=selfplay_games,
            train_steps=train_steps,
            batch_size=32,
            seed=seed,
            device=device,
        ),
    )
    game = BraidUnknotGame(game_cfg)
    typer.echo(
        "sources: "
        + ", ".join(f"{s.name}(u={s.unknotting_number})" for s in game.generator.sources)
    )
    agent = train_agent("alphazero", config)

    # Score against the theorem, and sweep log(A/B) to see whether the trade-off
    # actually moves.
    search = NeuralMCTS(game, agent.network, config.search, device)
    rows = []
    for source in game.generator.sources:
        for log_ratio in (-3.0, 0.0, 3.0):
            solved = crossings = moves = 0
            for index in range(eval_games):
                rng = np.random.default_rng(seed + 7000 * index)
                instance = game.generator.generate(source, max_scramble, rng)
                state = game.env.init_from_word(
                    list(instance.word), instance.strands, log_ratio=log_ratio
                )
                t = game._view(state, reward=0.0)
                while not t.terminated:
                    action = search.run(
                        t.state, t.observation, t.legal_actions, rng,
                        temperature=0.0, add_root_noise=False,
                    ).action
                    t = game.step(t.state, action)
                final = game.unwrap(t.state)
                won = bool((np.asarray(final._word) == 0).all()) and int(final._n) == 1
                solved += won
                if won:
                    crossings += int(np.asarray(final._crossing_changes))
                    moves += int(game_cfg.simplify_budget - int(np.asarray(final._budget)))
            row = {
                "source": source.name,
                "u": source.unknotting_number,
                "log_ratio": log_ratio,
                "solved": solved / eval_games,
                "crossings": crossings / solved if solved else float("nan"),
                "moves": moves / solved if solved else float("nan"),
            }
            rows.append(row)
            typer.echo(
                f"  {source.name:<8} u={source.unknotting_number}"
                f"  log(A/B)={log_ratio:+.0f}  solved {row['solved']:.2f}"
                f"  crossings {row['crossings']:.2f}  moves {row['moves']:.1f}"
            )
    out = output or artifact_dir(Path.cwd(), f"multi-{seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pareto.json").write_text(_json.dumps(rows, indent=2) + "\n")
    typer.echo(f"Saved: {out / 'pareto.json'}")


@app.command()
def braid_screen(
    tier: Annotated[str, typer.Option(help="tier0 (small) or tier1")] = "tier0",
    scramble_budget: Annotated[int, typer.Option(min=1, help="K, the difficulty dial")] = 3,
    iterations: Annotated[int, typer.Option(min=1)] = 8,
    anchors: Annotated[int, typer.Option(min=1)] = 12,
    baseline_games: Annotated[int, typer.Option(min=1)] = 10,
    seed: Annotated[int, typer.Option()] = 0,
    seeds: Annotated[int, typer.Option(min=1, help="Independent runs per variant")] = 1,
    workers: Annotated[
        int, typer.Option(min=1, help="Parallel runs; the sweep is embarrassingly parallel")
    ] = 1,
    only: Annotated[str, typer.Option(help="Comma-separated variant names to run")] = "",
    device: Annotated[str, typer.Option()] = "cpu",
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Screen ~10 approaches on small instances and rank them on a shared anchor set.

    Includes a `no-training` control, because search alone already solves a
    majority of small anchors -- any learning claim has to beat that, not zero.
    """
    if tier not in BRAID_TIERS:
        raise typer.BadParameter(f"tier must be one of {', '.join(BRAID_TIERS)}")
    label = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = output or artifact_dir(Path.cwd(), f"braid-screen-{label}")
    variants = default_variants(iterations, scramble_budget)
    if only:
        wanted = {name.strip() for name in only.split(",") if name.strip()}
        unknown = wanted - {v.name for v in variants}
        if unknown:
            raise typer.BadParameter(f"unknown variants: {', '.join(sorted(unknown))}")
        variants = [v for v in variants if v.name in wanted]
    typer.echo(
        f"screening {len(variants)} variants x {seeds} seed(s), "
        f"K={scramble_budget}, {anchors} anchors"
    )
    results = run_sweep(
        variants,
        BRAID_TIERS[tier],
        out,
        anchors=anchors,
        baseline_games=baseline_games,
        seed=seed,
        seeds=seeds,
        device=device,
        workers=workers,
        log=typer.echo,
    )
    typer.echo(f"Summary: {out / 'summary.md'}")
    del results


@app.command()
def sweep(
    simulations: Annotated[int, typer.Option(min=1)] = 8,
    iterations: Annotated[int, typer.Option(min=1)] = 1,
    selfplay_games: Annotated[int, typer.Option(min=1)] = 1,
    selfplay_positions: Annotated[
        int,
        typer.Option(min=0, help="Minimum positions per iteration; 0 means one game batch"),
    ] = 0,
    train_steps: Annotated[int, typer.Option(min=1)] = 2,
    batch_size: Annotated[int, typer.Option(min=1)] = 4,
    arena_games: Annotated[int, typer.Option(min=2)] = 2,
    channels: Annotated[int, typer.Option(min=4)] = 8,
    seed: int = 0,
    device: str = "cpu",
    output: Path | None = None,
    resume: bool = False,
    checkpoint_iterations: str = "1,3,10,20,40",
    curve_games: Annotated[int, typer.Option(min=0)] = 0,
    exact_positions: Annotated[
        bool,
        typer.Option("--exact-positions/--minimum-positions"),
    ] = True,
) -> None:
    """Run the same AlphaZero-versus-MuZero comparison for U1 through U5."""
    label = datetime.now().strftime("sweep-%Y%m%d-%H%M%S")
    out = output or artifact_dir(Path.cwd(), label)
    summary: dict[str, dict[str, float | int]] = {}
    checkpoints = _iteration_list(checkpoint_iterations, iterations)
    for rule in describe_rules():
        typer.echo(f"=== {rule}: {describe_rules()[rule]} ===")
        config = _config(
            rule,
            simulations,
            iterations,
            selfplay_games,
            selfplay_positions,
            train_steps,
            batch_size,
            channels,
            seed,
            device,
            checkpoints,
            curve_games,
            exact_positions,
        )
        rule_out = out / rule
        alphazero = train_agent(
            "alphazero",
            config,
            checkpoint_dir=rule_out / "checkpoints",
            resume=resume,
        )
        muzero = train_agent(
            "muzero",
            config,
            checkpoint_dir=rule_out / "checkpoints",
            resume=resume,
        )
        arena = compare_agents(alphazero, muzero, config, arena_games)
        save_experiment(rule_out, config, alphazero, muzero, arena)
        learning_curve = evaluate_learning_curve(rule_out, config, curve_games)
        save_experiment(rule_out, config, alphazero, muzero, arena, learning_curve)
        summary[rule] = arena
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    typer.echo(f"Saved sweep: {out / 'summary.json'}")


@app.command()
def crossplay(
    first: Annotated[Path, typer.Option(help="First experiment artifact directory")],
    second: Annotated[Path, typer.Option(help="Second experiment artifact directory")],
    kind: Annotated[
        str,
        typer.Option(help="Agent family for both sides unless overridden"),
    ] = "alphazero",
    first_kind: Annotated[
        str | None,
        typer.Option(help="Override the first agent family"),
    ] = None,
    second_kind: Annotated[
        str | None,
        typer.Option(help="Override the second agent family"),
    ] = None,
    games: Annotated[int, typer.Option(min=2)] = 40,
    seed: int = 200_000,
    device: str = "cpu",
    output: Path | None = None,
) -> None:
    """Play two trained agents, preserving each artifact's exploration rule."""
    first_family = first_kind or kind
    second_family = second_kind or kind
    valid_families = {"alphazero", "muzero"}
    if first_family not in valid_families or second_family not in valid_families:
        raise typer.BadParameter("agent families must be alphazero or muzero")
    first_agent = load_agent(first, first_family, device=device)
    second_agent = load_agent(second, second_family, device=device)
    result = compare_pair(first_agent, second_agent, games, seed=seed)
    payload = {
        "kind": kind if first_family == second_family else None,
        "first_kind": first_family,
        "second_kind": second_family,
        "first": str(first),
        "second": str(second),
        "first_rule": first_agent.config.search.exploration,
        "second_rule": second_agent.config.search.exploration,
        "arena": result,
    }
    rendered = json.dumps(payload, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
        typer.echo(f"Saved: {output}")
    typer.echo(rendered)


if __name__ == "__main__":
    app()
