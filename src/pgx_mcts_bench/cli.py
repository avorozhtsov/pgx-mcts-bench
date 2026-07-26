from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from pgx_mcts_bench.config import (
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
    evaluate_learning_curve,
    load_agent,
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
