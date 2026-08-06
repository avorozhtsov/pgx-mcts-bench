from __future__ import annotations

import json
from dataclasses import asdict, replace
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


@app.command("braid-budget-search-savings")
def braid_budget_search_savings(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    start_index: Annotated[int, typer.Option(min=0)] = 25,
    items: Annotated[int, typer.Option(min=1)] = 20,
    games_per_item: Annotated[int, typer.Option(min=1)] = 4,
    ratio: float = 10.0,
    multiplier: Annotated[float, typer.Option(min=0.1)] = 2.0,
    strategy: str = "restart",
    solve_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.04,
    simulations: Annotated[int, typer.Option(min=1)] = 32,
    minimum_savings: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.20,
    seed: int = 20261240,
    device: str = "cpu",
) -> None:
    """Compare paired full-budget and cap-and-restart search."""
    from pgx_mcts_bench.budget_savings import run_budget_savings

    report = run_budget_savings(
        checkpoint,
        output,
        start_index=start_index,
        items=items,
        games_per_item=games_per_item,
        ratio=ratio,
        multiplier=multiplier,
        strategy=strategy,
        solve_threshold=solve_threshold,
        simulations=simulations,
        minimum_savings=minimum_savings,
        seed=seed,
        device=device,
    )
    typer.echo(json.dumps({"summary": report["summary"], "decision": report["decision"]}, indent=2))


@app.command("braid-budget-heldout-gate")
def braid_budget_heldout_gate(
    baseline_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    trained_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    scientist: str = "s-window-128",
    training_items: Annotated[int, typer.Option(min=1)] = 5,
    heldout_start: Annotated[int, typer.Option(min=1)] = 5,
    heldout_items: Annotated[int, typer.Option(min=1)] = 10,
    games_per_cap: Annotated[int, typer.Option(min=1)] = 4,
    simulations: Annotated[int, typer.Option(min=1)] = 32,
    rung_eval_games: Annotated[int, typer.Option(min=1)] = 12,
    rung_simulations: Annotated[int, typer.Option(min=1)] = 128,
    seed: int = 20261140,
    device: str = "cpu",
    training_bank: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    heldout_bank: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Run the source-disjoint budget-calibration and retention gate."""
    from pgx_mcts_bench.budget_gate import run_budget_gate

    report = run_budget_gate(
        baseline_checkpoint,
        trained_checkpoint,
        output,
        scientist_name=scientist,
        training_items=training_items,
        heldout_start=heldout_start,
        heldout_items=heldout_items,
        games_per_cap=games_per_cap,
        simulations=simulations,
        rung_eval_games=rung_eval_games,
        rung_simulations=rung_simulations,
        seed=seed,
        device=device,
        training_bank=training_bank,
        heldout_bank=heldout_bank,
    )
    typer.echo(json.dumps({"summary": report["summary"], "decision": report["decision"]}, indent=2))


@app.command("braid-budget-critic-curriculum")
def braid_budget_critic_curriculum(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    scientist: str = "s-window-128",
    items: Annotated[int, typer.Option(min=1)] = 5,
    ratio: float = 10.0,
    cap_fractions: str = "0.005,0.01,0.02,0.05,1.0",
    games_per_cap: Annotated[int, typer.Option(min=1)] = 2,
    train_steps_per_item: Annotated[int, typer.Option(min=1)] = 32,
    simulations: Annotated[int, typer.Option(min=1)] = 32,
    rung_eval_games: Annotated[int, typer.Option(min=1)] = 4,
    rehearsal_games: Annotated[int, typer.Option(min=0)] = 8,
    seed: int = 20261040,
    device: str = "cpu",
    bank: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Train one budget-conditioned roster critic on the simplest knots."""
    from pgx_mcts_bench.budget_curriculum import train_budget_curriculum

    report = train_budget_curriculum(
        checkpoint,
        output,
        scientist_name=scientist,
        items=items,
        ratio=ratio,
        cap_fractions=tuple(
            float(value) for value in cap_fractions.split(",") if value.strip()
        ),
        games_per_cap=games_per_cap,
        train_steps_per_item=train_steps_per_item,
        simulations=simulations,
        rung_eval_games=rung_eval_games,
        rehearsal_games=rehearsal_games,
        seed=seed,
        device=device,
        bank=bank,
    )
    typer.echo(json.dumps(report["decision"], indent=2))


@app.command("braid-fit-solve-calibration")
def braid_fit_solve_calibration(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    validation_report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_checkpoint: Annotated[Path, typer.Option(dir_okay=False)],
    output_report: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    """Fit a monotone held-out p(solve) calibrator without changing weights."""
    from pgx_mcts_bench.solve_calibration import fit_solve_calibration

    report = fit_solve_calibration(
        checkpoint, validation_report, output_checkpoint, output_report
    )
    typer.echo(
        json.dumps(
            {"calibration": report["calibration"], "fitted": report["fitted"]},
            indent=2,
        )
    )


@app.command("braid-budget-head-audit")
def braid_budget_head_audit(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    outcomes: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    training_metrics: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    caps: str = "16,32,64,128,256,512,704",
    device: str = "cpu",
) -> None:
    """Audit budget sensitivity, monotonicity, calibration, and label balance."""
    from pgx_mcts_bench.budget_head_audit import audit_budget_head

    report = audit_budget_head(
        checkpoint,
        bank,
        output,
        outcomes=outcomes,
        training_metrics=training_metrics,
        scientist=scientist,
        ratio=ratio,
        caps=tuple(float(value) for value in caps.split(",") if value.strip()),
        device=device,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-rapid-paired-gate")
def braid_rapid_paired_gate(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    old_bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    simulations: Annotated[int, typer.Option(min=1)] = 128,
    selfplay_games: Annotated[int, typer.Option(min=1)] = 8,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    seeds: str = "20260980,20260981,20260982",
    workers: Annotated[int, typer.Option(min=1)] = 7,
    device: str = "cpu",
    resume: bool = False,
    gate_games: Annotated[int, typer.Option(min=1)] = 12,
    gate_min_solve_rate: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
) -> None:
    """Run the three-seed paired causal gate before any 200-item expansion."""
    from pgx_mcts_bench.rapid_gate import run_paired_rapid_gate

    report = run_paired_rapid_gate(
        checkpoint,
        bank,
        old_bank,
        output,
        scientist=scientist,
        ratio=ratio,
        simulations=simulations,
        selfplay_games=selfplay_games,
        batch_size=batch_size,
        seeds=tuple(int(value) for value in seeds.split(",") if value.strip()),
        workers=workers,
        device=device,
        resume=resume,
        gate_games=gate_games,
        gate_min_solve_rate=gate_min_solve_rate,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-rapid-adaptation")
def braid_rapid_adaptation(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    target_bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    old_bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    initial_f: int = 5,
    f_old: Annotated[int, typer.Option(min=0)] = 1,
    threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
    simulations: Annotated[int, typer.Option(min=1)] = 128,
    selfplay_games: Annotated[int, typer.Option(min=1)] = 8,
    train_steps: Annotated[int, typer.Option(min=0)] = 96,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    limit: Annotated[int, typer.Option(min=1)] = 200,
    workers: Annotated[int, typer.Option(min=1)] = 6,
    seed: int = 20260840,
    device: str = "cpu",
    resume: bool = False,
    gate_games: Annotated[int, typer.Option(min=1)] = 12,
    gate_min_solve_rate: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
) -> None:
    """Run disposable task-local adaptation with blockwise F escalation."""
    from pgx_mcts_bench.rapid_adaptation import run_rapid_adaptation

    report = run_rapid_adaptation(
        checkpoint,
        target_bank,
        old_bank,
        output,
        scientist=scientist,
        ratio=ratio,
        initial_f=initial_f,
        f_old=f_old,
        threshold=threshold,
        simulations=simulations,
        selfplay_games=selfplay_games,
        train_steps=train_steps,
        batch_size=batch_size,
        limit=limit,
        workers=workers,
        seed=seed,
        device=device,
        resume=resume,
        gate_games=gate_games,
        gate_min_solve_rate=gate_min_solve_rate,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-policy-update-diagnostic")
def braid_policy_update_diagnostic(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    old_bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    target: str = "11a_33",
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    simulations: Annotated[int, typer.Option(min=1)] = 64,
    selfplay_games: Annotated[int, typer.Option(min=1)] = 8,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    evaluation_games: Annotated[int, typer.Option(min=1)] = 8,
    seeds: str = "20261361,20261362,20261363",
    workers: Annotated[int, typer.Option(min=1)] = 7,
    device: str = "cpu",
    resume: bool = False,
) -> None:
    """Diagnose destructive task-local policy updates with paired pre/post probes."""
    from pgx_mcts_bench.policy_update_diagnostic import run_policy_update_diagnostic

    report = run_policy_update_diagnostic(
        checkpoint,
        bank,
        old_bank,
        output,
        target=target,
        scientist=scientist,
        ratio=ratio,
        simulations=simulations,
        selfplay_games=selfplay_games,
        batch_size=batch_size,
        evaluation_games=evaluation_games,
        seeds=tuple(int(value) for value in seeds.split(",") if value.strip()),
        workers=workers,
        device=device,
        resume=resume,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-split-loss-gate")
def braid_split_loss_gate(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    old_bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    simulations: Annotated[int, typer.Option(min=1)] = 64,
    selfplay_games: Annotated[int, typer.Option(min=1)] = 8,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    seeds: str = "20261420,20261421,20261422",
    workers: Annotated[int, typer.Option(min=1)] = 7,
    device: str = "cpu",
    resume: bool = False,
) -> None:
    """Run the small paired admission gate for split-loss task learning."""
    from pgx_mcts_bench.split_loss_gate import run_split_loss_gate

    report = run_split_loss_gate(
        checkpoint,
        bank,
        old_bank,
        output,
        scientist=scientist,
        ratio=ratio,
        simulations=simulations,
        selfplay_games=selfplay_games,
        batch_size=batch_size,
        seeds=tuple(int(value) for value in seeds.split(",") if value.strip()),
        workers=workers,
        device=device,
        resume=resume,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-native-learning-gate")
def braid_native_learning_gate(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    evaluation_simulations: Annotated[int, typer.Option(min=1)] = 64,
    train_steps: Annotated[int, typer.Option(min=1)] = 24,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    seeds: str = "20261520,20261521,20261522",
    device: str = "cpu",
    resume: bool = False,
) -> None:
    """Run native discovery with transactional solved-set retention."""
    from pgx_mcts_bench.native_learning_gate import run_native_learning_gate

    report = run_native_learning_gate(
        checkpoint,
        bank,
        output,
        scientist=scientist,
        ratio=ratio,
        evaluation_simulations=evaluation_simulations,
        train_steps=train_steps,
        batch_size=batch_size,
        seeds=tuple(int(value) for value in seeds.split(",") if value.strip()),
        device=device,
        resume=resume,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-replay-integrity-gate")
def braid_replay_integrity_gate(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    simulations: Annotated[int, typer.Option(min=1)] = 32,
    games_per_item: Annotated[int, typer.Option(min=1)] = 4,
    capped_games_per_item: Annotated[int, typer.Option(min=1)] = 2,
    capped_objective: Annotated[float, typer.Option(min=1.0)] = 12.0,
    sample_size: Annotated[int, typer.Option(min=16)] = 512,
    seed: int = 20260850,
    device: str = "cpu",
) -> None:
    """Generate real attempts and audit replay-v3 sampling and resume state."""
    from pgx_mcts_bench.replay_gate import run_replay_integrity_gate

    report = run_replay_integrity_gate(
        checkpoint,
        bank,
        output,
        scientist=scientist,
        ratio=ratio,
        simulations=simulations,
        games_per_item=games_per_item,
        capped_games_per_item=capped_games_per_item,
        capped_objective=capped_objective,
        sample_size=sample_size,
        seed=seed,
        device=device,
    )
    typer.echo(json.dumps(report["decision"], indent=2))


@app.command("braid-replay-learning-gate")
def braid_replay_learning_gate(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    old_bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    scientist: str = "s-window-128",
    ratio: float = 10.0,
    simulations: Annotated[int, typer.Option(min=1)] = 64,
    selfplay_games: Annotated[int, typer.Option(min=1)] = 8,
    train_steps: Annotated[int, typer.Option(min=1)] = 24,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    diagnostic_games: Annotated[int, typer.Option(min=1)] = 8,
    seeds: str = "20260851,20260852",
    workers: Annotated[int, typer.Option(min=1)] = 7,
    device: str = "cpu",
    resume: bool = False,
) -> None:
    """Compare old success-balanced replay with replay-v3 on paired tasks."""
    from pgx_mcts_bench.replay_gate import run_replay_learning_gate

    report = run_replay_learning_gate(
        checkpoint,
        bank,
        old_bank,
        output,
        scientist=scientist,
        ratio=ratio,
        simulations=simulations,
        selfplay_games=selfplay_games,
        train_steps=train_steps,
        batch_size=batch_size,
        diagnostic_games=diagnostic_games,
        seeds=tuple(int(value) for value in seeds.split(",") if value.strip()),
        workers=workers,
        device=device,
        resume=resume,
    )
    typer.echo(json.dumps(report["decision"], indent=2))


@app.command("braid-critic-readiness")
def braid_critic_readiness(
    validation_report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    calibration_report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    """Apply current critic readiness rules to unrebalanced held-out evidence."""
    from pgx_mcts_bench.critic_readiness import build_critic_readiness_report

    report = build_critic_readiness_report(
        validation_report, calibration_report, output
    )
    typer.echo(json.dumps(report["decision"], indent=2))


@app.command("braid-sharing-admission-gate")
def braid_sharing_admission_gate(
    donor_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    donor_replay: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    receiver: Annotated[list[str] | None, typer.Option()] = None,
    donor_name: str = "s-window-128",
    ratio: float = 10.0,
    simulations: Annotated[int, typer.Option(min=1)] = 64,
    evaluation_games: Annotated[int, typer.Option(min=1)] = 4,
    train_steps: Annotated[int, typer.Option(min=1)] = 24,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    seed: int = 20260853,
    device: str = "cpu",
) -> None:
    """Test real witness translation, option learning, and receiver retention."""
    from pgx_mcts_bench.sharing_gate import run_sharing_admission_gate

    receivers: dict[str, Path] = {}
    for assignment in receiver or []:
        if "=" not in assignment:
            raise typer.BadParameter("receiver must be NAME=CHECKPOINT")
        name, path = assignment.split("=", 1)
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise typer.BadParameter(f"missing receiver checkpoint: {checkpoint}")
        receivers[name] = checkpoint
    if not receivers:
        raise typer.BadParameter("at least one --receiver NAME=CHECKPOINT is required")
    report = run_sharing_admission_gate(
        donor_checkpoint,
        donor_replay,
        bank,
        receivers,
        output,
        donor_name=donor_name,
        ratio=ratio,
        simulations=simulations,
        evaluation_games=evaluation_games,
        train_steps=train_steps,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    typer.echo(json.dumps(report["decision"], indent=2))


@app.command("braid-interleaved-sharing-gate")
def braid_interleaved_sharing_gate(
    donor_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    donor_replay: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    receiver: Annotated[list[str] | None, typer.Option()] = None,
    donor_name: str = "s-window-128",
    ratio: float = 10.0,
    simulations: Annotated[int, typer.Option(min=1)] = 64,
    evaluation_games: Annotated[int, typer.Option(min=1)] = 4,
    update_cycles: Annotated[int, typer.Option(min=1)] = 8,
    batch_size: Annotated[int, typer.Option(min=1)] = 16,
    option_learning_rate_scale: Annotated[float, typer.Option(min=0.001)] = 1.0,
    option_target_reduction: Annotated[float, typer.Option(min=0.001, max=0.999)] = 0.1,
    max_option_steps: Annotated[int, typer.Option(min=1)] = 16,
    witness_bank: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    item: Annotated[list[str] | None, typer.Option()] = None,
    target_item: Annotated[list[str] | None, typer.Option()] = None,
    native_refresh_games: Annotated[int, typer.Option(min=0)] = 0,
    evaluation_workers: Annotated[int, typer.Option(min=1)] = 1,
    gated_adapter: bool = False,
    route_gate_weight: Annotated[float, typer.Option(min=0.0)] = 0.1,
    off_route_kl_weight: Annotated[float, typer.Option(min=0.0)] = 1.0,
    off_route_gate_weight: Annotated[float, typer.Option(min=0.0)] = 0.1,
    off_route_batch_size: Annotated[int, typer.Option(min=1)] = 32,
    seed: int = 20260854,
    device: str = "cpu",
) -> None:
    """Compare native-plus-option sharing with matched native-only updates."""
    from pgx_mcts_bench.sharing_gate import run_interleaved_sharing_gate

    receivers: dict[str, Path] = {}
    for assignment in receiver or []:
        if "=" not in assignment:
            raise typer.BadParameter("receiver must be NAME=CHECKPOINT")
        name, path = assignment.split("=", 1)
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise typer.BadParameter(f"missing receiver checkpoint: {checkpoint}")
        receivers[name] = checkpoint
    if not receivers:
        raise typer.BadParameter("at least one --receiver NAME=CHECKPOINT is required")
    report = run_interleaved_sharing_gate(
        donor_checkpoint,
        donor_replay,
        bank,
        receivers,
        output,
        donor_name=donor_name,
        ratio=ratio,
        simulations=simulations,
        evaluation_games=evaluation_games,
        update_cycles=update_cycles,
        batch_size=batch_size,
        option_learning_rate_scale=option_learning_rate_scale,
        option_target_reduction=option_target_reduction,
        max_option_steps=max_option_steps,
        witness_bank=witness_bank,
        item_ids=tuple(item or ()),
        target_item_ids=tuple(target_item or ()),
        native_refresh_games=native_refresh_games,
        evaluation_workers=evaluation_workers,
        gated_adapter=gated_adapter,
        route_gate_weight=route_gate_weight,
        off_route_kl_weight=off_route_kl_weight,
        off_route_gate_weight=off_route_gate_weight,
        off_route_batch_size=off_route_batch_size,
        seed=seed,
        device=device,
    )
    typer.echo(json.dumps(report["decision"], indent=2))


@app.command("braid-multi-witness-screen")
def braid_multi_witness_screen(
    source_run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    receiver_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    receiver_name: str = "s-w11-128",
    candidate: Annotated[list[str] | None, typer.Option()] = None,
    ratio: float = 10.0,
    simulations: Annotated[int, typer.Option(min=1)] = 128,
    games: Annotated[int, typer.Option(min=1)] = 16,
    seed_blocks: Annotated[int, typer.Option(min=1)] = 1,
    panel_size: Annotated[int, typer.Option(min=1)] = 8,
    retention_size: Annotated[int, typer.Option(min=0)] = 8,
    workers: Annotated[int, typer.Option(min=1)] = 1,
    seed: int = 20260900,
    device: str = "cpu",
) -> None:
    """Freeze a certified panel the receiver never solves in strong screening."""
    from pgx_mcts_bench.multi_witness import run_multi_witness_screen

    report = run_multi_witness_screen(
        source_run,
        receiver_name,
        receiver_checkpoint,
        output,
        candidate_ids=tuple(candidate or ()),
        ratio=ratio,
        simulations=simulations,
        games=games,
        seed_blocks=seed_blocks,
        panel_size=panel_size,
        retention_size=retention_size,
        workers=workers,
        seed=seed,
        device=device,
    )
    typer.echo(json.dumps(report["decision"], indent=2))


@app.command("braid-adapter-counterfactual")
def braid_adapter_counterfactual(
    source_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    trained_checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    scientist_name: str = "s-tape4-h5",
    item: Annotated[list[str] | None, typer.Option()] = None,
    ratio: float = 10.0,
    simulations: Annotated[int, typer.Option(min=1)] = 128,
    evaluation_games: Annotated[int, typer.Option(min=1)] = 8,
    evaluation_workers: Annotated[int, typer.Option(min=1)] = 4,
    evaluation_seed: int = 820260950,
    device: str = "cpu",
) -> None:
    """Compare a trained sharing scientist with its adapter bypassed."""
    from pgx_mcts_bench.sharing_gate import run_adapter_counterfactual

    if not item:
        raise typer.BadParameter("at least one --item is required")
    report = run_adapter_counterfactual(
        source_checkpoint,
        trained_checkpoint,
        bank,
        output,
        scientist_name=scientist_name,
        item_ids=tuple(item),
        ratio=ratio,
        simulations=simulations,
        evaluation_games=evaluation_games,
        evaluation_workers=evaluation_workers,
        evaluation_seed=evaluation_seed,
        device=device,
    )
    typer.echo(
        json.dumps(
            {
                "enabled_only": report["enabled_only"],
                "disabled_only": report["disabled_only"],
                "enabled_capped_loss": report["enabled_capped_loss"],
                "disabled_capped_loss": report["disabled_capped_loss"],
            },
            indent=2,
        )
    )


@app.command("braid-rung-parity-gate")
def braid_rung_parity_gate(
    output: Annotated[Path, typer.Option(file_okay=False)],
    seed: int = 0,
    stop_after: Annotated[int, typer.Option(min=0, max=9)] = 9,
    max_iterations: Annotated[int, typer.Option(min=1)] = 100,
    eval_games: Annotated[int, typer.Option(min=4)] = 12,
    retro_games: Annotated[int, typer.Option(min=0)] = 4,
    workers: Annotated[int, typer.Option(min=1, max=2)] = 2,
    device: str = "cpu",
) -> None:
    """Reproduce the successful rung-18 recipe on the first ladder rungs."""
    from pgx_mcts_bench.readiness_gates import run_rung_parity_gate

    report = run_rung_parity_gate(
        output,
        seed=seed,
        stop_after=stop_after,
        max_iterations=max_iterations,
        eval_games=eval_games,
        retro_games=retro_games,
        workers=workers,
        device=device,
    )
    typer.echo(json.dumps(report["decision"], indent=2))


@app.command("braid-distillation-degradation-train")
def braid_distillation_degradation_train(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    round_index: Annotated[int, typer.Option(min=0)] = 48,
    training_seeds: str = "0,1,2",
    train_steps: Annotated[int, typer.Option(min=1)] = 8,
    batch_size: Annotated[int, typer.Option(min=2)] = 32,
    shared_fraction: Annotated[float, typer.Option(min=0.0, max=0.5)] = 0.1,
    variants: str = "pre,rl0,d1-full,d10-full,d1-aux,d10-aux",
    learning_rate: Annotated[float | None, typer.Option(min=0.0)] = None,
    device: str = "cpu",
) -> None:
    """Create matched RL-only and one/ten-witness checkpoint forks."""
    from pgx_mcts_bench.degradation_experiment import VARIANTS, train_degradation_forks

    seeds = tuple(int(value) for value in training_seeds.split(",") if value.strip())
    selected_variants = tuple(value.strip() for value in variants.split(",") if value.strip())
    unknown = set(selected_variants) - set(VARIANTS)
    if unknown:
        raise typer.BadParameter(f"unknown variants: {sorted(unknown)}")
    report = train_degradation_forks(
        run,
        output,
        round_index=round_index,
        training_seeds=seeds,
        train_steps=train_steps,
        batch_size=batch_size,
        shared_fraction=shared_fraction,
        variants=selected_variants,  # type: ignore[arg-type]
        learning_rate=learning_rate,
        device=device,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-distillation-degradation-evaluate")
def braid_distillation_degradation_evaluate(
    experiment: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    training_seed: Annotated[int, typer.Option()],
    variant: Annotated[str, typer.Option()],
    split: Annotated[str, typer.Option()],
    simulations: Annotated[int, typer.Option(min=1)] = 16,
    limit: Annotated[int, typer.Option(min=0)] = 0,
    evaluation_seed: int = 0,
    device: str = "cpu",
    resume: bool = False,
    bank: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Evaluate one matched degradation fork on a frozen split."""
    from pgx_mcts_bench.degradation_experiment import evaluate_degradation_fork

    report = evaluate_degradation_fork(
        experiment,
        output,
        training_seed=training_seed,
        variant=variant,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
        simulations=simulations,
        limit=limit,
        evaluation_seed=evaluation_seed,
        device=device,
        resume=resume,
        bank=bank,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-distillation-degradation-analyze")
def braid_distillation_degradation_analyze(
    evaluations: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
) -> None:
    """Apply the paired retention and transfer safety gate."""
    from pgx_mcts_bench.degradation_experiment import analyze_degradation_experiment

    report = analyze_degradation_experiment(evaluations, output)
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-adaptive-scientists")
def braid_adaptive_scientists(
    output: Annotated[Path, typer.Option(file_okay=False)],
    scientist: Annotated[
        list[str] | None,
        typer.Option(help="Repeat NAME=RUNG23_CHECKPOINT; defaults to the deep-ladder trio"),
    ] = None,
    rounds: Annotated[int, typer.Option(min=1)] = 20,
    pool_size: Annotated[int, typer.Option(min=1)] = 200,
    alpha: Annotated[float, typer.Option(min=0.0)] = 1.0,
    proposal_temperature: Annotated[float, typer.Option(min=0.0)] = 1.0,
    group_temperature: Annotated[float, typer.Option(min=0.0)] = 1.0,
    starvation_rounds: Annotated[
        int, typer.Option(min=0, help="0 uses the fairness guarantee 2*N")
    ] = 0,
    selfplay_games: Annotated[int, typer.Option(min=1)] = 2,
    train_steps: Annotated[int, typer.Option(min=0)] = 16,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    simulations: Annotated[
        int, typer.Option(min=0, help="0 preserves each rung-23 candidate setting")
    ] = 0,
    seed: int = 0,
    device: str = "cpu",
    require_factorized: Annotated[
        bool,
        typer.Option("--require-factorized/--allow-legacy-proxy"),
    ] = False,
) -> None:
    """Grow a shared curriculum proposed by diverse rung-23 scientists."""
    from pgx_mcts_bench.adaptive_scientists import (
        default_rung23_checkpoints,
        run_adaptive_scientists,
    )

    if scientist:
        checkpoints: dict[str, Path] = {}
        for value in scientist:
            if "=" not in value:
                raise typer.BadParameter("--scientist must be NAME=CHECKPOINT")
            name, raw_path = value.split("=", 1)
            path = Path(raw_path)
            if not path.is_file():
                raise typer.BadParameter(f"checkpoint does not exist: {path}")
            checkpoints[name] = path
    else:
        checkpoints = default_rung23_checkpoints(Path.cwd())
        missing = [path for path in checkpoints.values() if not path.is_file()]
        if missing:
            raise typer.BadParameter(f"default checkpoint does not exist: {missing[0]}")
    report = run_adaptive_scientists(
        checkpoints,
        output,
        rounds=rounds,
        pool_size=pool_size,
        alpha=alpha,
        proposal_temperature=proposal_temperature,
        group_temperature=group_temperature,
        starvation_rounds=starvation_rounds,
        selfplay_games=selfplay_games,
        train_steps=train_steps,
        batch_size=batch_size,
        simulations=simulations,
        seed=seed,
        device=device,
        require_factorized=require_factorized,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-collaborative-scientists")
def braid_collaborative_scientists(
    output: Annotated[Path, typer.Option(file_okay=False)],
    scientist: Annotated[
        list[str],
        typer.Option(help="Repeat NAME=CHECKPOINT for heterogeneous serial scientists"),
    ],
    arm: Annotated[
        str,
        typer.Option(
            help="adaptive-sharing, adaptive-sharing-aux-only, "
            "adaptive-no-sharing, static-sharing, static-no-sharing, "
            "or solo-compute-matched"
        ),
    ] = "adaptive-sharing",
    rounds: Annotated[int, typer.Option(min=1)] = 200,
    pool_size: Annotated[int, typer.Option(min=1)] = 200,
    anchor_size: Annotated[int, typer.Option(min=0)] = 70,
    frontier: Annotated[int, typer.Option(min=1)] = 100,
    ratios: str = "10,1000",
    qualification_simulations: Annotated[int, typer.Option(min=1)] = 16,
    simulations: Annotated[int, typer.Option(min=1)] = 128,
    train_every: Annotated[int, typer.Option(min=1)] = 10,
    train_steps: Annotated[int, typer.Option(min=0)] = 32,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    attempt_workers: Annotated[int, typer.Option(min=1)] = 1,
    objective_budget: bool = False,
    objective_budget_audit_every: Annotated[int, typer.Option(min=0)] = 10,
    bank_seed: int = 0,
    seed: int = 0,
    device: str = "cpu",
    resume: bool = False,
) -> None:
    """Run one resumable heterogeneous collaboration pilot arm."""
    from pgx_mcts_bench.collaborative_scientists import run_collaborative_scientists

    checkpoints: dict[str, Path] = {}
    for value in scientist:
        if "=" not in value:
            raise typer.BadParameter("--scientist must be NAME=CHECKPOINT")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not path.is_file():
            raise typer.BadParameter(f"checkpoint does not exist: {path}")
        checkpoints[name] = path
    if not checkpoints:
        raise typer.BadParameter("repeat --scientist NAME=CHECKPOINT at least once")
    parsed_ratios = tuple(float(value) for value in ratios.split(","))
    report = run_collaborative_scientists(
        checkpoints,
        output,
        arm=arm,  # type: ignore[arg-type]
        rounds=rounds,
        pool_size=pool_size,
        anchor_size=anchor_size,
        frontier=frontier,
        ratios=parsed_ratios,
        qualification_simulations=qualification_simulations,
        simulations=simulations,
        train_every=train_every,
        train_steps=train_steps,
        batch_size=batch_size,
        attempt_workers=attempt_workers,
        objective_budget=objective_budget,
        objective_budget_audit_every=objective_budget_audit_every,
        bank_seed=bank_seed,
        seed=seed,
        device=device,
        resume=resume,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-triad-build")
def braid_triad_build(
    window: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    scan: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    tape: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    device: str = "cpu",
) -> None:
    """Assemble s-triad-wst from the fixed r18/r10/r8 parent snapshots."""
    from pgx_mcts_bench.triad import build_triad_checkpoint

    report = build_triad_checkpoint(window, scan, tape, output, device=device)
    typer.echo(json.dumps(asdict(report), indent=2))


@app.command("braid-cyclic-memory-build")
def braid_cyclic_memory_build(
    window: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    seed: int = 0,
    device: str = "cpu",
) -> None:
    """Initialize s-cyclic-tape8-192 from an s-window-128 checkpoint."""
    from pgx_mcts_bench.cyclic_memory import build_cyclic_memory_checkpoint

    report = build_cyclic_memory_checkpoint(window, output, seed=seed, device=device)
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-collaboration-export")
def braid_collaboration_export(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    scientist: str,
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    """Export one scientist from the latest committed collaboration round."""
    from pgx_mcts_bench.collaboration_eval import export_collaboration_scientist

    report = export_collaboration_scientist(run, scientist, output)
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-cyclic-invariant-pretrain")
def braid_cyclic_invariant_pretrain(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    identities: Annotated[int, typer.Option(min=2)] = 400,
    calibration_identities: Annotated[int, typer.Option(min=2)] = 50,
    views_per_identity: Annotated[int, typer.Option(min=2)] = 4,
    steps: Annotated[int, typer.Option(min=1)] = 1_000,
    batch_size: Annotated[int, typer.Option(min=2)] = 32,
    learning_rate: float = 3e-4,
    temperature: float = 0.1,
    bank_seed: int = 20260802,
    seed: int = 0,
    device: str = "cpu",
) -> None:
    """Pretrain the cyclic tower to retrieve equivalent knot representations."""
    from pgx_mcts_bench.invariant_pretrain import pretrain_cyclic_invariants

    report = pretrain_cyclic_invariants(
        checkpoint,
        output,
        identities=identities,
        calibration_identities=calibration_identities,
        views_per_identity=views_per_identity,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        bank_seed=bank_seed,
        seed=seed,
        device=device,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-collaboration-evaluate")
def braid_collaboration_evaluate(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    state: str = "final",
    split: str = "new70",
    simulations: Annotated[int, typer.Option(min=1)] = 128,
    limit: Annotated[int, typer.Option(min=0)] = 0,
    seed: int = 0,
    device: str = "cpu",
    resume: bool = False,
    bank: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Evaluate an initial or final scientist portfolio on a frozen split."""
    from pgx_mcts_bench.collaboration_eval import evaluate_collaboration

    report = evaluate_collaboration(
        run,
        output,
        state=state,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
        simulations=simulations,
        limit=limit,
        seed=seed,
        device=device,
        resume=resume,
        bank=bank,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-collaboration-rebuild-anchor")
def braid_collaboration_rebuild_anchor(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_bank: Annotated[Path, typer.Option(dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option(dir_okay=False)],
    max_stage: Annotated[int, typer.Option(min=0)] = 22,
    seed: int = 20260802,
) -> None:
    """Replace held-out identities seen as explicit ladder sources."""
    from pgx_mcts_bench.bank_audit import rebuild_anchor_without_ladder_identities

    report = rebuild_anchor_without_ladder_identities(
        run,
        output_bank,
        output_manifest,
        max_stage=max_stage,
        seed=seed,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-build-development-bank")
def braid_build_development_bank(
    source_bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_bank: Annotated[Path, typer.Option(dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option(dir_okay=False)],
    size: Annotated[int, typer.Option(min=4)] = 20,
    seed: int = 20260970,
    max_stage: Annotated[int, typer.Option(min=0)] = 22,
) -> None:
    """Build an outcome-blind, ladder-source-disjoint development bank."""
    from pgx_mcts_bench.bank_audit import build_development_bank

    report = build_development_bank(
        source_bank,
        output_bank,
        output_manifest,
        size=size,
        seed=seed,
        max_stage=max_stage,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-build-critic-banks")
def braid_build_critic_banks(
    protected_bank: Annotated[list[Path], typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    train_size: Annotated[int, typer.Option(min=4)] = 60,
    validation_size: Annotated[int, typer.Option(min=4)] = 20,
    decision_size: Annotated[int, typer.Option(min=4)] = 20,
    seed: int = 20261310,
    max_stage: Annotated[int, typer.Option(min=0)] = 22,
) -> None:
    """Build critic splits disjoint from endpoints and ladder identities."""
    from pgx_mcts_bench.bank_audit import build_critic_calibration_banks

    report = build_critic_calibration_banks(
        protected_bank,
        output,
        train_size=train_size,
        validation_size=validation_size,
        decision_size=decision_size,
        seed=seed,
        max_stage=max_stage,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-collaboration-compare")
def braid_collaboration_compare(
    treatment: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    control: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
) -> None:
    """Compare two evaluations by exact solved representation identity."""
    from pgx_mcts_bench.collaboration_eval import compare_collaboration_evaluations

    report = compare_collaboration_evaluations(treatment, control, output)
    typer.echo(json.dumps(report, indent=2))


@app.command("braid-objective-budget-regression")
def braid_objective_budget_regression(
    bank: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    scientist: Annotated[list[str], typer.Option(help="Repeat NAME=CHECKPOINT for each scientist")],
    ratio: float = 10.0,
    simulations: Annotated[int, typer.Option(min=1)] = 16,
    limit: Annotated[int, typer.Option(min=1)] = 10,
    items: str = "",
    seed: int = 0,
    device: str = "cpu",
) -> None:
    """Compare the fixed move clock with predicted caps plus restarts."""
    from pgx_mcts_bench.collaboration_eval import benchmark_objective_budget

    checkpoints = {}
    for value in scientist:
        if "=" not in value:
            raise typer.BadParameter("--scientist must be NAME=CHECKPOINT")
        name, raw_path = value.split("=", 1)
        checkpoints[name] = Path(raw_path)
    report = benchmark_objective_budget(
        checkpoints,
        bank,
        output,
        ratio=ratio,
        simulations=simulations,
        limit=limit,
        item_ids=tuple(value for value in items.split(",") if value),
        seed=seed,
        device=device,
    )
    typer.echo(json.dumps(report["summary"], indent=2))


@app.command("braid-triad-frontier")
def braid_triad_frontier(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    eval_games: Annotated[int, typer.Option(min=1)] = 12,
    seed: int = 0,
    device: str = "cpu",
    start_rung: Annotated[int, typer.Option(min=0)] = 0,
    stop_rung: int = -1,
    stop_at_first_failure: Annotated[
        bool,
        typer.Option("--stop-at-first-failure/--all-rungs"),
    ] = True,
) -> None:
    """Find the first rung where the frozen triad should begin training."""
    from pgx_mcts_bench.triad import evaluate_frozen_triad

    report = evaluate_frozen_triad(
        checkpoint,
        output,
        eval_games=eval_games,
        seed=seed,
        device=device,
        stop_at_first_failure=stop_at_first_failure,
        start_rung=start_rung,
        stop_rung=stop_rung,
        log=typer.echo,
    )
    typer.echo(f"Recommended first training rung: {report['recommended_training_rung']}")


@app.command("braid-distill-u1")
def braid_distill_u1(
    teacher: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    episodes: Annotated[int, typer.Option(min=1)] = 128,
    train_steps: Annotated[int, typer.Option(min=1)] = 2_000,
    seed: int = 0,
    device: str = "cpu",
    internal_horizon: Annotated[int, typer.Option(min=1)] = 5,
    option_beam_width: Annotated[int, typer.Option(min=1)] = 8,
    option_batch_size: Annotated[int, typer.Option(min=1)] = 4,
) -> None:
    """Distill the latest parallel u1 policy and factorized values into serial students."""
    from pgx_mcts_bench.distill import run_distillation

    report = run_distillation(
        teacher,
        output,
        episodes=episodes,
        train_steps=train_steps,
        seed=seed,
        device=device,
        internal_horizon=internal_horizon,
        option_beam_width=option_beam_width,
        option_batch_size=option_batch_size,
    )
    typer.echo(json.dumps(asdict(report), indent=2))


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
    "tier0": BraidGameConfig(max_len=32, max_strands=5, scramble_budget=6, simplify_budget=24),
    "tier1": BraidGameConfig(max_len=64, max_strands=8, scramble_budget=12, simplify_budget=48),
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
    selfplay_games: Annotated[
        int,
        typer.Option(
            min=1,
            help="Concurrent self-play roots; increase to 32-128 when benchmarking CUDA",
        ),
    ] = 8,
    checkpoint_every: Annotated[
        int,
        typer.Option(
            min=0,
            help="Save current-rung replay/optimizer progress every N iterations; 0 disables",
        ),
    ] = 1,
    eval_games: Annotated[int, typer.Option(min=4)] = 16,
    promote_at: Annotated[float, typer.Option()] = 0.8,
    mix_decay: Annotated[
        float, typer.Option(help="Training mixture decay back from the frontier; 0 = frontier only")
    ] = 0.5,
    crossing_tolerance: Annotated[
        float, typer.Option(help="Promote cleanly when crossings <= u(K) + this")
    ] = 0.25,
    plateau_window: Annotated[int, typer.Option(min=1)] = 3,
    collapse_floor: Annotated[
        float, typer.Option(help="A single ratio below this is a collapse, not noise")
    ] = 0.5,
    max_consecutive_caps: Annotated[
        int, typer.Option(min=1, help="Capped rungs in a row before a candidate stops")
    ] = 3,
    stop_after: Annotated[
        int, typer.Option(help="Stop after this rung index; -1 climbs the whole ladder")
    ] = -1,
    min_iterations_per_rung: Annotated[
        float,
        typer.Option(help="Keep training a rung until iterations / rungs reaches this"),
    ] = 0.0,
    min_iterations_from: Annotated[
        int, typer.Option(help="First rung the training floor governs")
    ] = 0,
    bounds: Annotated[
        Path | None,
        typer.Option(help="Append best-known unknotting bounds to this log"),
    ] = None,
    retro_games: Annotated[int, typer.Option(min=0)] = 6,
    workers: Annotated[int, typer.Option(min=1)] = 1,
    device: Annotated[str, typer.Option()] = "cpu",
    use_auxiliary_value: Annotated[
        bool,
        typer.Option(
            "--use-auxiliary-value",
            help="Use the factorized ensemble value in MCTS; default is shadow-only",
        ),
    ] = False,
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
        known = {candidate.name for candidate in chosen}
        unknown = sorted(wanted - known)
        if unknown:
            raise typer.BadParameter(f"unknown candidate(s): {', '.join(unknown)}")
        chosen = [c for c in chosen if c.name in wanted]
    if use_auxiliary_value:
        chosen = [replace(candidate, use_auxiliary_value=True) for candidate in chosen]
    out = output or artifact_dir(Path.cwd(), "ladder")
    typer.echo(f"{len(chosen)} candidates over {len(STAGES)} stages, {workers} workers")
    for index, stage in enumerate(STAGES):
        typer.echo(f"  stage {index}: {stage[0]} + {stage[1]} scramble moves")

    results = []
    if workers <= 1:
        # A one-candidate container does not enter ProcessPoolExecutor, so its
        # initializer would otherwise never run.  That left Torch/BLAS free to
        # create one thread per vCPU for every queued candidate, defeating the
        # host-level CPU queue and producing severe oversubscription.
        _worker_init()
        for candidate in chosen:
            results.append(
                run_ladder(
                    candidate,
                    seed=seed,
                    device=device,
                    checkpoint_dir=out / "checkpoints",
                    max_iterations_per_stage=max_iterations,
                    selfplay_games=selfplay_games,
                    checkpoint_every=checkpoint_every,
                    eval_games=eval_games,
                    promote_at=promote_at,
                    mix_decay=mix_decay,
                    crossing_tolerance=crossing_tolerance,
                    plateau_window=plateau_window,
                    retro_games=retro_games,
                    collapse_floor=collapse_floor,
                    max_consecutive_caps=max_consecutive_caps,
                    stop_after=stop_after,
                    min_iterations_per_rung=min_iterations_per_rung,
                    min_iterations_from=min_iterations_from,
                    bounds_path=bounds,
                    log=typer.echo,
                )
            )
            save(results, out)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            futures = {
                pool.submit(
                    run_ladder,
                    c,
                    seed=seed,
                    device=device,
                    checkpoint_dir=out / "checkpoints",
                    max_iterations_per_stage=max_iterations,
                    selfplay_games=selfplay_games,
                    checkpoint_every=checkpoint_every,
                    eval_games=eval_games,
                    promote_at=promote_at,
                    mix_decay=mix_decay,
                    crossing_tolerance=crossing_tolerance,
                    plateau_window=plateau_window,
                    retro_games=retro_games,
                    collapse_floor=collapse_floor,
                    max_consecutive_caps=max_consecutive_caps,
                    stop_after=stop_after,
                    min_iterations_per_rung=min_iterations_per_rung,
                    min_iterations_from=min_iterations_from,
                    bounds_path=bounds,
                    log=_silent,
                ): c
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
def braid_device_benchmark(
    candidates_only: Annotated[
        str,
        typer.Option(
            "--only",
            help="Comma-separated representatives; default covers parallel, serial, tape and GRU",
        ),
    ] = "u1-puct,s-w11-128,s-tape4,s-scan-gru",
    devices: Annotated[str, typer.Option(help="Comma-separated: cpu,cuda,mps")] = "cpu,cuda",
    actor_batches: Annotated[
        str,
        typer.Option(help="Comma-separated concurrent self-play root counts"),
    ] = "8,32,64",
    stage: Annotated[int, typer.Option(min=0)] = 8,
    eval_games: Annotated[int, typer.Option(min=1)] = 4,
    measured_train_steps: Annotated[int, typer.Option(min=1)] = 8,
    torch_threads: Annotated[
        int,
        typer.Option(min=1, help="PyTorch CPU threads in this one-worker benchmark"),
    ] = 1,
    cpu_hourly: Annotated[
        float,
        typer.Option(help="Hourly cost of one CPU worker; default is Nebius cpu-d3/4"),
    ] = 0.0248,
    gpu_hourly: Annotated[
        float,
        typer.Option(help="Hourly GPU VM cost; default is Nebius L40S Intel 8/32"),
    ] = 1.5484,
    simulations: Annotated[
        int | None,
        typer.Option(min=1, help="Override simulations for a quick smoke test only"),
    ] = None,
    seed: int = 0,
    output: Path | None = None,
) -> None:
    """Measure an end-to-end CPU/CUDA ladder iteration and apply the 3x GPU gate."""
    from pgx_mcts_bench.device_benchmark import run_device_benchmark
    from pgx_mcts_bench.ladder import STAGES, candidates

    if stage >= len(STAGES):
        raise typer.BadParameter(f"stage must be below {len(STAGES)}")
    names = [name.strip() for name in candidates_only.split(",") if name.strip()]
    by_name = {candidate.name: candidate for candidate in candidates()}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise typer.BadParameter(f"unknown candidates: {', '.join(unknown)}")
    selected_devices = [name.strip() for name in devices.split(",") if name.strip()]
    unknown_devices = set(selected_devices) - {"cpu", "cuda", "mps"}
    if unknown_devices:
        raise typer.BadParameter(f"unknown devices: {', '.join(sorted(unknown_devices))}")
    try:
        batches = [int(value.strip()) for value in actor_batches.split(",") if value.strip()]
    except ValueError as error:
        raise typer.BadParameter("actor batches must be comma-separated integers") from error
    if not batches or any(value < 1 for value in batches):
        raise typer.BadParameter("actor batches must be positive")
    out = output or artifact_dir(Path.cwd(), "braid-device-benchmark")
    run_device_benchmark(
        [by_name[name] for name in names],
        devices=selected_devices,
        actor_batches=batches,
        stage_index=stage,
        eval_games=eval_games,
        measured_train_steps=measured_train_steps,
        seed=seed,
        output=out,
        simulations=simulations,
        torch_threads=torch_threads,
        cpu_hourly=cpu_hourly,
        gpu_hourly=gpu_hourly,
        log=typer.echo,
    )
    typer.echo(f"Saved: {out / 'device-benchmark.md'}")


@app.command()
def braid_ladder_merge(
    root: Annotated[Path, typer.Argument(help="Directory holding per-candidate runs")],
) -> None:
    """Combine per-candidate ladder outputs into one report."""
    from pgx_mcts_bench.ladder import merge

    results = merge(root)
    if not results:
        typer.echo(f"No ladder.json found under {root}")
        raise typer.Exit(1)
    for r in sorted(results, key=lambda x: -x.highest_stage):
        typer.echo(f"  {r.name:18s} stage {r.highest_stage:2d}  {r.seconds:.0f}s")
    typer.echo(f"Saved: {root / 'ladder.md'}")


@app.command()
def braid_ladder_leaderboard(
    roots: Annotated[
        list[Path] | None,
        typer.Option(
            "--root",
            help=(
                "Checkpoint root or .pt file; repeat for multiple local or server-snapshot roots"
            ),
        ),
    ] = None,
) -> None:
    """Print fresh standings from live ladder checkpoints."""
    from pgx_mcts_bench.leaderboard import DEFAULT_ROOTS, leaderboard, render

    selected = roots or list(DEFAULT_ROOTS)
    rows, warnings = leaderboard(selected)
    for warning in warnings:
        typer.echo(f"warning: {warning}", err=True)
    if not rows:
        typer.echo("No promoted ladder checkpoints found", err=True)
        raise typer.Exit(1)
    typer.echo(render(rows), nl=False)


@app.command()
def braid_ladder_values(
    roots: Annotated[
        list[Path] | None,
        typer.Option("--root", help="Checkpoint root; repeat for multiple roots"),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help="JSON output; a Markdown report is written beside it"),
    ] = Path("artifacts/value-eval-r31-r40.json"),
    seed: Annotated[int, typer.Option(help="Held-out instance seed")] = 1_337_000,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Evaluate raw value heads on the ten held-out mixed-sign rungs."""
    from pgx_mcts_bench.leaderboard import DEFAULT_ROOTS
    from pgx_mcts_bench.value_eval import evaluate_value_heads, save

    result, warnings = evaluate_value_heads(list(roots or DEFAULT_ROOTS), seed=seed, device=device)
    for warning in warnings:
        typer.echo(f"warning: {warning}", err=True)
    save(result, output)
    typer.echo(f"Evaluated {len(result['candidates'])} critics on {len(result['instances'])} rungs")
    typer.echo(f"Saved: {output}")
    typer.echo(f"Saved: {output.with_suffix('.md')}")


@app.command()
def braid_ladder_rescore(
    root: Annotated[Path, typer.Argument(help="Directory holding a ladder run")],
    games: Annotated[int, typer.Option(min=1)] = 12,
    simulations: Annotated[
        int, typer.Option(min=0, help="Override the search budget; 0 keeps the candidate's")
    ] = 0,
) -> None:
    """Re-measure every cleared rung with each candidate's final weights."""
    from pgx_mcts_bench.ladder import rescore

    out = rescore(root, games=games, simulations=simulations, log=typer.echo)
    if not out:
        typer.echo(f"No checkpoints found under {root}")
        raise typer.Exit(1)
    typer.echo(f"Saved: {root / 'rescore.json'}")


@app.command()
def braid_serial_screen(
    arms_only: Annotated[str, typer.Option("--only", help="Comma-separated arm names")] = "",
    seed: Annotated[int, typer.Option()] = 0,
    max_iterations: Annotated[int, typer.Option(min=1)] = 40,
    eval_games: Annotated[int, typer.Option(min=4)] = 12,
    promote_at: Annotated[float, typer.Option()] = 0.8,
    workers: Annotated[int, typer.Option(min=1)] = 4,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Screen serial-formulation settings on the ladder's own stages."""
    from pgx_mcts_bench.serial_screen import run_screen

    only = [n.strip() for n in arms_only.split(",") if n.strip()]
    out = output or artifact_dir(Path.cwd(), "serial-screen")
    run_screen(
        out,
        max_iterations=max_iterations,
        eval_games=eval_games,
        promote_at=promote_at,
        workers=workers,
        seed=seed,
        only=only,
        log=typer.echo,
    )
    typer.echo(f"Saved: {out / 'screen.md'}")


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
                        t.state,
                        t.observation,
                        t.legal_actions,
                        rng,
                        temperature=0.0,
                        add_root_noise=False,
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
