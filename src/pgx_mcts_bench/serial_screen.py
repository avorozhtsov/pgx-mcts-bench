"""Hyperparameter screen for the serial (moving-window) candidates.

The first ladder run put every serial candidate at stage 0. This runs the *same*
ladder, on the same stages and the same promotion rule, over a small grid of
serial-specific settings, so the numbers are directly comparable to
`artifacts/ladder-run/ladder.md` rather than to each other only.

It is a screen, not an experiment: one seed per arm, chosen to find a setting
that moves at all. Whatever wins here gets re-run with seeds before it is
believed -- three seeds gave a false positive on the proposer and it survived
two rounds of reporting.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from pgx_mcts_bench.ladder import run_ladder, serial_arms

# The grid itself lives in `ladder.py` as `serial_arms()`, so every bounded
# serial screen uses the same stage list and settings.
arms = serial_arms


def _run(args) -> dict:
    candidate, output, max_iterations, eval_games, promote_at, seed = args
    log_path = Path(output) / f"{candidate.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(*parts) -> None:
        # Written and flushed per line, never buffered through a pipe: a silent
        # background run is indistinguishable from a hung one.
        with log_path.open("a") as handle:
            handle.write(" ".join(str(p) for p in parts) + "\n")

    started = time.perf_counter()
    result = run_ladder(
        candidate,
        seed=seed,
        device="cpu",
        checkpoint_dir=Path(output) / "checkpoints",
        max_iterations_per_stage=max_iterations,
        eval_every=2,
        eval_games=eval_games,
        promote_at=promote_at,
        log=log,
    )
    row = {
        "name": candidate.name,
        "rationale": candidate.rationale,
        "candidate": asdict(candidate),
        "highest_stage": result.highest_stage,
        "seconds": time.perf_counter() - started,
        "stages": [asdict(s) for s in result.stages],
    }
    (Path(output) / f"{candidate.name}.json").write_text(json.dumps(row, indent=2))
    return row


def run_screen(
    output: Path,
    *,
    max_iterations: int = 40,
    eval_games: int = 12,
    promote_at: float = 0.8,
    workers: int = 4,
    seed: int = 0,
    only: list[str] | None = None,
    log=print,
) -> list[dict]:
    from pgx_mcts_bench.worker_runtime import enable_jax_compilation_cache, worker_init

    enable_jax_compilation_cache()
    output.mkdir(parents=True, exist_ok=True)
    grid = [a for a in arms() if not only or a.name in only]
    log(f"serial screen: {len(grid)} arms x up to {max_iterations} iterations/stage, "
        f"{workers} workers")
    jobs = [(a, str(output), max_iterations, eval_games, promote_at, seed) for a in grid]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for row in pool.map(_run, jobs):
            stage = row["highest_stage"]
            top = next(
                (s for s in row["stages"] if s["stage"] == stage and s.get("promoted")),
                None,
            )
            reached = f"{top['source']}+{top['scramble']}" if top else "nothing"
            log(f"  {row['name']:18s} stage {stage:2d} ({reached})  "
                f"{row['seconds']:.0f}s")
            rows.append(row)
            (output / "screen.json").write_text(json.dumps(rows, indent=2))
    rows.sort(key=lambda r: (-r["highest_stage"], r["seconds"]))
    (output / "screen.json").write_text(json.dumps(rows, indent=2))
    (output / "screen.md").write_text(_report(rows))
    return rows


def _report(rows: list[dict]) -> str:
    lines = [
        "# Serial screen",
        "",
        "Ten stages, the list in force when this screen ran. The current ladder has",
        "17 rungs and a different promotion rule, so compare by **rung name** against",
        "`artifacts/ladder-run` -- which used the same ten -- and not by stage number",
        "against anything newer. Every serial candidate scored 0 in that run.",
        "",
        "| arm | what it varies | highest stage | reached | seconds |",
        "|---|---|---:|---|---:|",
    ]
    for row in rows:
        stage = row["highest_stage"]
        # From the row's own record, never from the module-level STAGES: the stage
        # list has changed since this screen ran, and indexing today's list with
        # yesterday's index relabels an old run with rungs it never saw.
        cleared = {s["stage"]: s for s in row["stages"] if s.get("promoted")}
        top = cleared.get(stage)
        reached = f"`{top['source']}+{top['scramble']}`" if top else "--"
        lines.append(
            f"| `{row['name']}` | {row['rationale']} | {stage} | {reached} | "
            f"{row['seconds']:.0f} |"
        )
    lines += ["", "## Per stage", ""]
    for row in rows:
        lines += [f"### `{row['name']}`", "",
                  "| stage | instance | it | solved | crossings | optimal |",
                  "|---:|---|---:|---:|---:|---:|"]
        for s in row["stages"]:
            lines.append(
                f"| {s['stage']} | {s['source']}+{s['scramble']} | {s['iterations']} | "
                f"{s['solve_rate']:.2f} | {s['crossings']:.2f} | "
                f"{s['optimal_crossings']} |"
            )
        lines.append("")
    return "\n".join(lines)
