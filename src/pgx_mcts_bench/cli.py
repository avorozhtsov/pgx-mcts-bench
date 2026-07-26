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
from pgx_mcts_bench.training import compare_agents, save_experiment, train_agent

app = typer.Typer(no_args_is_help=True)


def _config(
    exploration: str,
    simulations: int,
    iterations: int,
    selfplay_games: int,
    train_steps: int,
    batch_size: int,
    channels: int,
    seed: int,
    device: str,
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
            train_steps=train_steps,
            batch_size=batch_size,
            seed=seed,
            device=device,
        ),
    )


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
    train_steps: Annotated[int, typer.Option(min=1)] = 4,
    batch_size: Annotated[int, typer.Option(min=1)] = 8,
    arena_games: Annotated[int, typer.Option(min=2)] = 4,
    channels: Annotated[int, typer.Option(min=4)] = 16,
    seed: int = 0,
    device: str = "cpu",
    output: Path | None = None,
) -> None:
    """Train compact AlphaZero and MuZero agents, then play a color-balanced arena."""
    config = _config(
        exploration,
        simulations,
        iterations,
        selfplay_games,
        train_steps,
        batch_size,
        channels,
        seed,
        device,
    )
    typer.echo(
        f"6x6 Go, komi={config.game.komi}, max_moves={config.game.max_moves}, "
        f"{simulations} simulations, rule={exploration}"
    )
    alphazero = train_agent("alphazero", config)
    muzero = train_agent("muzero", config)
    arena = compare_agents(alphazero, muzero, config, arena_games)
    label = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = output or artifact_dir(Path.cwd(), label)
    save_experiment(out, config, alphazero, muzero, arena)
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
        train_steps=1,
        batch_size=2,
        arena_games=2,
        channels=4,
        seed=0,
        device="cpu",
        output=output,
    )


@app.command()
def sweep(
    simulations: Annotated[int, typer.Option(min=1)] = 8,
    iterations: Annotated[int, typer.Option(min=1)] = 1,
    selfplay_games: Annotated[int, typer.Option(min=1)] = 1,
    train_steps: Annotated[int, typer.Option(min=1)] = 2,
    batch_size: Annotated[int, typer.Option(min=1)] = 4,
    arena_games: Annotated[int, typer.Option(min=2)] = 2,
    channels: Annotated[int, typer.Option(min=4)] = 8,
    seed: int = 0,
    device: str = "cpu",
    output: Path | None = None,
) -> None:
    """Run the same AlphaZero-versus-MuZero comparison for U1 through U5."""
    label = datetime.now().strftime("sweep-%Y%m%d-%H%M%S")
    out = output or artifact_dir(Path.cwd(), label)
    summary: dict[str, dict[str, float | int]] = {}
    for rule in describe_rules():
        typer.echo(f"=== {rule}: {describe_rules()[rule]} ===")
        config = _config(
            rule,
            simulations,
            iterations,
            selfplay_games,
            train_steps,
            batch_size,
            channels,
            seed,
            device,
        )
        alphazero = train_agent("alphazero", config)
        muzero = train_agent("muzero", config)
        arena = compare_agents(alphazero, muzero, config, arena_games)
        save_experiment(out / rule, config, alphazero, muzero, arena)
        summary[rule] = arena
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    typer.echo(f"Saved sweep: {out / 'summary.json'}")


if __name__ == "__main__":
    app()
