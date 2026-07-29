"""Staged curriculum over the generator's complexity grade.

Ten stages, monotone in both the source knot's unknotting number and the number
of scramble moves on top of it. A candidate trains at one stage until it can
solve a held-out sample of that stage, then moves up; if it cannot, it moves up
anyway after a cap so a stuck candidate spends its budget elsewhere rather than
grinding. **The score is the highest stage reached**, which lets candidates
spend different amounts of time per stage.

Evaluation instances come from a seed stream disjoint from training, so the
promotion signal is held out rather than measured on what was just trained on.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from pgx_mcts_bench.config import (
    BraidGameConfig,
    ExperimentConfig,
    ModelConfig,
    SearchConfig,
    TrainConfig,
)
from pgx_mcts_bench.data import ReplayBuffer
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.networks import BraidAlphaZeroNet, SerialBraidNet
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games, train_alphazero_step

# (source knot, scramble moves). Monotone in u(K) and in depth.
STAGES: list[tuple[str, int]] = [
    ("unknot", 2),
    ("unknot", 6),
    ("T(2,3)", 0),
    ("T(2,3)", 4),
    ("T(2,5)", 0),
    ("T(2,5)", 4),
    ("T(2,7)", 0),
    ("T(2,7)", 4),
    ("T(3,4)", 0),
    ("T(3,4)", 4),
]

# The three cost ratios the network must serve simultaneously, as requested:
# A:B = 1000:1 (crossing changes dominate -- unknotting-number minimisation),
# 10:1 (crossing changes preferred but moves matter), 1:10 (moves dominate).
# An episode samples one; the network is conditioned on log(A/B) through FiLM.
RATIOS: tuple[float, ...] = (1000.0, 10.0, 0.1)


@dataclass(frozen=True)
class Candidate:
    name: str
    rationale: str
    exploration: str = "u1"
    simulations: int = 64
    channels: int = 32
    train_steps: int = 96
    serial_window: int = 0
    serial_act_width: int = 1
    # 0 takes the default rule: 32 plies for parallel candidates, 64 for serial
    # ones, which pay plies to move the head.
    simplify_budget: int = 0
    # () takes the default power-of-two stride set. `(w // 2,)` is the original
    # single-stride tape, kept screenable so the change can be ablated.
    serial_shift_strides: tuple[int, ...] = ()
    train: bool = True


def candidates() -> list[Candidate]:
    return [
        Candidate("no-training", "control: search only, weights never updated", train=False),
        Candidate("u1-puct", "AlphaZero PUCT, parallel head", exploration="u1"),
        Candidate("u3-uct", "prior-free UCT; never collapsed in earlier screens", exploration="u3"),
        Candidate("search-heavy", "128 simulations: is depth the constraint?", simulations=128),
        Candidate("search-light", "16 simulations: the network must carry the policy",
                  simulations=16),
        Candidate("wide-net", "96 channels: is capacity the constraint?", channels=96,
                  train_steps=160),
        Candidate("serial-w7-head", "moving window, acting only at the head",
                  serial_window=7, serial_act_width=1),
        Candidate("serial-w7-window", "moving window, acting anywhere it can see",
                  serial_window=7, serial_act_width=7),
        Candidate("serial-w11", "window at the parallel net's receptive field (11)",
                  serial_window=11, serial_act_width=11),
    ]


@dataclass
class StageResult:
    stage: int
    source: str
    scramble: int
    iterations: int
    by_ratio: dict
    solve_rate: float
    crossings: float
    optimal_crossings: int
    promoted: bool
    seconds: float


@dataclass
class LadderResult:
    name: str
    rationale: str
    highest_stage: int
    seconds: float
    stages: list[StageResult] = field(default_factory=list)


def _silent(*args, **kwargs) -> None:
    """Module-level so it survives pickling into a worker process."""


def _config(candidate: Candidate, stage: tuple[str, int], seed: int, device: str):
    game = BraidGameConfig(
        max_len=48,
        max_strands=5,
        scramble_budget=1,
        simplify_budget=(
            candidate.simplify_budget
            or (32 if not candidate.serial_window else 64)
        ),
        allow_crossing_change=True,
        multi_objective=True,
        log_ratio_range=(float(np.log(min(RATIOS))), float(np.log(max(RATIOS)))),
        generator_max_crossings=10,
        generator_max_scramble=6,
        stage_source=stage[0],
        stage_scramble=stage[1],
        serial_window=candidate.serial_window,
        serial_act_width=candidate.serial_act_width,
        serial_shift_strides=candidate.serial_shift_strides,
    )
    return ExperimentConfig(
        game=game,
        search=SearchConfig(simulations=candidate.simulations, exploration=candidate.exploration),  # type: ignore[arg-type]
        model=ModelConfig(channels=candidate.channels, latent_channels=candidate.channels),
        train=TrainConfig(
            iterations=1,
            selfplay_games=8,
            train_steps=candidate.train_steps,
            batch_size=32,
            seed=seed,
            device=device,
        ),
    )


def evaluate_stage(game, network, config, games: int, seed: int) -> dict[float, dict]:
    """Per-ratio solve rate, crossing changes and moves on held-out instances.

    Evaluated at each of A:B = 1000:1, 10:1, 1:10 separately, so it is visible
    whether one network is really serving all three or collapsing to one policy.
    """
    search = NeuralMCTS(game, network, config.search, config.train.device)
    out: dict[float, dict] = {}
    for ratio in RATIOS:
        log_ratio = float(np.log(ratio))
        solved = crossings = moves = 0
        for index in range(games):
            rng = np.random.default_rng(seed + 100_003 * (index + 1))
            source, scramble = config.game.stage_source, config.game.stage_scramble
            src = next(s for s in game.generator.sources if s.name == source)
            instance = game.generator.generate(src, scramble, rng)
            transition = game.from_word(
                list(instance.word), instance.strands, log_ratio
            )
            while not transition.terminated:
                action = search.run(
                    transition.state, transition.observation, transition.legal_actions,
                    rng, temperature=0.0, add_root_noise=False,
                ).action
                transition = game.step(transition.state, action)
            final = game.unwrap(transition.state)
            if bool((np.asarray(final._word) == 0).all()) and int(final._n) == 1:
                solved += 1
                crossings += int(np.asarray(final._crossing_changes))
                moves += config.game.simplify_budget - int(np.asarray(final._budget))
        out[ratio] = {
            "solved": solved / games,
            "crossings": crossings / solved if solved else float("nan"),
            "moves": moves / solved if solved else float("nan"),
        }
    return out


def run_ladder(
    candidate: Candidate,
    *,
    seed: int = 0,
    device: str = "cpu",
    checkpoint_dir: Path | None = None,
    max_iterations_per_stage: int = 25,
    eval_every: int = 2,
    eval_games: int = 16,
    promote_at: float = 0.8,
    log=print,
) -> LadderResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    first = _config(candidate, STAGES[0], seed, device)
    network = (
        SerialBraidNet(first.game, first.model)
        if candidate.serial_window
        else BraidAlphaZeroNet(first.game, first.model)
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
    replay = ReplayBuffer(20_000, rng)
    result = LadderResult(candidate.name, candidate.rationale, -1, 0.0)

    # Resume: pick up at the stage after the last one cleared, with the weights
    # that cleared it. A ladder run is long enough that losing it to a laptop
    # closing would be silly.
    start_stage = 0
    path = checkpoint_dir / f"{candidate.name}.pt" if checkpoint_dir else None
    if path is not None and path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        network.load_state_dict(saved["network"])
        optimizer.load_state_dict(saved["optimizer"])
        result.stages = [StageResult(**row) for row in saved["stages"]]
        result.highest_stage = saved["highest_stage"]
        start_stage = saved["highest_stage"] + 1
        log(f"    [{candidate.name}] resumed at stage {start_stage}")

    for index, stage in enumerate(STAGES):
        if index < start_stage:
            continue
        config = _config(candidate, stage, seed, device)
        game = make_game(config.game)
        source = next(s for s in game.generator.sources if s.name == stage[0])
        search = NeuralMCTS(game, network, config.search, device)
        stage_started = time.perf_counter()
        solve_rate = crossings = float("nan")
        by_ratio: dict = {}
        promoted = False
        iterations = 0

        for iteration in range(max_iterations_per_stage):
            if candidate.train:
                seeds = [seed + index * 10_000 + iteration * 100 + g for g in range(8)]
                records = play_selfplay_games(
                    game, search, [np.random.default_rng(s + 7) for s in seeds], seeds, 12
                )
                for record in records:
                    replay.add(record)
                for _ in range(candidate.train_steps):
                    train_alphazero_step(network, optimizer, replay, 32, torch.device(device))
                iterations += 1
            if not candidate.train or (iteration + 1) % eval_every == 0:
                by_ratio = evaluate_stage(
                    game, network, config, eval_games, seed + 500_000 + index * 997
                )
                # Promotion needs the hardest setting solved, not the easiest.
                solve_rate = min(v["solved"] for v in by_ratio.values())
                crossings = by_ratio[max(RATIOS)]["crossings"]
                if solve_rate >= promote_at:
                    promoted = True
                    break
            if not candidate.train:
                break

        result.stages.append(
            StageResult(
                stage=index,
                by_ratio={str(k): v for k, v in by_ratio.items()},
                source=stage[0],
                scramble=stage[1],
                iterations=iterations,
                solve_rate=solve_rate,
                crossings=crossings,
                optimal_crossings=source.unknotting_number,
                promoted=promoted,
                seconds=time.perf_counter() - stage_started,
            )
        )
        log(
            f"    [{candidate.name}] stage {index} {stage[0]}+{stage[1]} "
            f"(u={source.unknotting_number}): solved {solve_rate:.2f} "
            f"crossings {crossings:.2f} after {iterations} it "
            f"({'promoted' if promoted else 'capped'})"
        )
        if promoted:
            result.highest_stage = index
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "network": network.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "stages": [asdict(s) for s in result.stages],
                        "highest_stage": result.highest_stage,
                        "candidate": candidate.name,
                    },
                    path,
                )
        else:
            break

    result.seconds = time.perf_counter() - started
    return result


def render(results: list[LadderResult]) -> str:
    lines = [
        "# Ladder: how far up the complexity grade does each candidate get?",
        "",
        "Ten stages, monotone in the source knot's unknotting number and in scramble",
        "depth. A candidate is promoted when it solves >= 80% of 16 held-out instances",
        "at its current stage, and capped at 25 iterations if it cannot. Score is the",
        "highest stage cleared.",
        "",
        "| candidate | highest stage | reached | total iterations | seconds |",
        "|---|---:|---|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: -x.highest_stage):
        last = r.stages[-1] if r.stages else None
        reached = f"{last.source}+{last.scramble}" if last else "-"
        lines.append(
            f"| `{r.name}` | {r.highest_stage} | {reached} "
            f"| {sum(s.iterations for s in r.stages)} | {r.seconds:.0f} |"
        )
    lines += ["", "## Per stage", ""]
    for r in results:
        lines.append(f"### `{r.name}` — {r.rationale}")
        lines.append("")
        lines.append(
            "| stage | instance | u(K) | it | A:B=1000:1 cc/moves "
            "| 10:1 cc/moves | 1:10 cc/moves |"
        )
        lines.append("|---:|---|---:|---:|---|---|---|")
        for st in r.stages:
            cells = []
            for ratio in ("1000.0", "10.0", "0.1"):
                v = st.by_ratio.get(ratio)
                cells.append(
                    "-" if not v else
                    f"{v['crossings']:.2f} / {v['moves']:.1f} ({v['solved']:.0%})"
                )
            lines.append(
                f"| {st.stage} | {st.source}+{st.scramble} | {st.optimal_crossings} "
                f"| {st.iterations} | " + " | ".join(cells) + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def save(results: list[LadderResult], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "ladder.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=float) + "\n"
    )
    (out / "ladder.md").write_text(render(results))
