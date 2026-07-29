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

from pgx_mcts_bench.ladder import STAGES, Candidate, run_ladder


def arms() -> list[Candidate]:
    """The grid.

    Four factors, screened one at a time against a common base rather than
    crossed -- a full cross is 48 arms and this machine has five free cores.

    * `act_width`  -- head-only against acting anywhere visible. The first run
      could not separate these because neither could see *where*; with a
      positional policy head the comparison means something.
    * `simulations` -- the serial formulation spends plies on repositioning, so
      it needs depth the parallel one does not. `search-heavy` (128) was the
      strongest parallel candidate by a wide margin.
    * `budget`     -- 64 plies against 96, since head motion is charged.
    * `strides`    -- the new power-of-two set against the original single
      stride, which is the ablation for the reachability fix.
    """
    base = dict(exploration="u1", simulations=128, channels=32, train_steps=96)
    grid: list[Candidate] = [
        Candidate("s-head-128", "head-only, 128 sims", serial_window=7,
                  serial_act_width=1, **base),
        Candidate("s-window-128", "act anywhere in a 7-window, 128 sims",
                  serial_window=7, serial_act_width=7, **base),
        Candidate("s-w11-128", "11-window, act anywhere, 128 sims",
                  serial_window=11, serial_act_width=11, **base),
        Candidate("s-head-256", "head-only, 256 sims: is depth still the wall?",
                  serial_window=7, serial_act_width=1,
                  exploration="u1", simulations=256, channels=32, train_steps=96),
        Candidate("s-head-budget96", "head-only, 96 plies to pay for head motion",
                  serial_window=7, serial_act_width=1, simplify_budget=96, **base),
        Candidate("s-head-1stride", "ABLATION: head-only, the original single stride",
                  serial_window=7, serial_act_width=1, serial_shift_strides=(3,), **base),
    ]
    return grid


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
    from pgx_mcts_bench.braid_sweep import _worker_init, enable_jax_compilation_cache

    enable_jax_compilation_cache()
    output.mkdir(parents=True, exist_ok=True)
    grid = [a for a in arms() if not only or a.name in only]
    log(f"serial screen: {len(grid)} arms x up to {max_iterations} iterations/stage, "
        f"{workers} workers")
    jobs = [(a, str(output), max_iterations, eval_games, promote_at, seed) for a in grid]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        for row in pool.map(_run, jobs):
            stage = row["highest_stage"]
            reached = (
                f"{STAGES[stage][0]}+{STAGES[stage][1]}" if stage >= 0 else "nothing"
            )
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
        "Same stages and the same promotion rule as `artifacts/ladder-run`, so the",
        "`highest stage` column is comparable with that table. Every serial",
        "candidate scored 0 there.",
        "",
        "| arm | what it varies | highest stage | reached | seconds |",
        "|---|---|---:|---|---:|",
    ]
    for row in rows:
        stage = row["highest_stage"]
        reached = f"`{STAGES[stage][0]}+{STAGES[stage][1]}`" if stage >= 0 else "--"
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
