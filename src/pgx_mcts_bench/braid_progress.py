"""Per-iteration progress tracking for the braid game.

The scalar that matters is not self-play win rate -- that is a ratio between two
things that are both moving. It is: **of a frozen set of scrambled instances, how
many can the Simplifier actually untie, and how close to optimal are its
solutions?** This module evaluates exactly that after every training iteration
and renders the results as braid diagrams, so the training curve can be looked at
rather than trusted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.config import BraidGameConfig, ExperimentConfig
from pgx_mcts_bench.game import BraidUnknotGame
from pgx_mcts_bench.search import NeuralMCTS


@dataclass
class AnchorAttempt:
    index: int
    word: list[int]
    strands: int
    length: int
    solved: bool
    moves: int
    move_names: list[str] = field(default_factory=list)
    optimal: int | None = None

    @property
    def excess(self) -> int | None:
        """Moves used beyond a shortest solution, when the optimum is known."""
        if not self.solved or self.optimal is None:
            return None
        return self.moves - self.optimal


@dataclass
class IterationReport:
    iteration: int
    solve_rate: float
    mean_moves_when_solved: float | None
    mean_excess: float | None
    attempts: list[AnchorAttempt]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["attempts"] = [asdict(attempt) for attempt in self.attempts]
        return payload


class BraidProgress:
    """Frozen anchor set plus per-iteration evaluation and rendering."""

    def __init__(
        self,
        config: ExperimentConfig,
        out: Path,
        *,
        anchors: int = 16,
        seed: int = 10_000,
        bfs_depth: int = 5,
        showcase: int = 6,
    ):
        if not isinstance(config.game, BraidGameConfig):
            raise TypeError("BraidProgress only applies to the braid game")
        from rf_knots.rollout import anchor_instances

        self.config = config
        self.game_config = config.game
        self.out = out
        self.showcase = showcase
        self.game = BraidUnknotGame(config.game)
        self.instances = anchor_instances(config.game.to_braid_config(), anchors, seed=seed)
        self.reports: list[IterationReport] = []
        self.optimal = self._solve_anchors_exactly(bfs_depth)

    def _solve_anchors_exactly(self, bfs_depth: int) -> list[int | None]:
        """Shortest solutions where breadth-first search can still reach them.

        `None` means "deeper than the cutoff", not "unsolvable" -- every anchor is
        an unknot by construction. Knowing the optimum turns "solved" into
        "solved in k moves more than necessary", which is a far sharper signal.
        """
        from rf_knots.reference import bfs_unknot

        spec = self.game.env.spec
        found: list[int | None] = []
        for word, strands in self.instances:
            path = bfs_unknot(
                spec,
                word,
                strands,
                max_depth=bfs_depth,
                max_growth=self.game_config.scramble_budget,
            )
            found.append(None if path is None else len(path))
        return found

    def evaluate(self, iteration: int, network: torch.nn.Module) -> IterationReport:
        search = NeuralMCTS(
            self.game, network, self.config.search, self.config.train.device
        )
        spec = self.game.env.spec
        attempts: list[AnchorAttempt] = []
        for index, (word, strands) in enumerate(self.instances):
            rng = np.random.default_rng(index)
            transition = self.game.from_word(list(word), strands)
            names: list[str] = []
            while not transition.terminated:
                result = search.run(
                    transition.state,
                    transition.observation,
                    transition.legal_actions,
                    rng,
                    temperature=0.0,
                    add_root_noise=False,
                )
                names.append(spec.describe(result.action))
                transition = self.game.step(transition.state, result.action)
            final_word = np.asarray(transition.state._word)
            solved = bool((final_word == 0).all()) and int(np.asarray(transition.state._n)) == 1
            attempts.append(
                AnchorAttempt(
                    index=index,
                    word=list(word),
                    strands=strands,
                    length=len(word),
                    solved=solved,
                    moves=len(names),
                    move_names=names if solved else [],
                    optimal=self.optimal[index],
                )
            )

        solved_attempts = [attempt for attempt in attempts if attempt.solved]
        excesses = [
            attempt.excess for attempt in solved_attempts if attempt.excess is not None
        ]
        report = IterationReport(
            iteration=iteration,
            solve_rate=len(solved_attempts) / len(attempts) if attempts else 0.0,
            mean_moves_when_solved=(
                float(np.mean([a.moves for a in solved_attempts])) if solved_attempts else None
            ),
            mean_excess=float(np.mean(excesses)) if excesses else None,
            attempts=attempts,
        )
        self.reports.append(report)
        self.write()
        return report

    # -- output ---------------------------------------------------------------

    def write(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "progress.json").write_text(
            json.dumps([report.to_dict() for report in self.reports], indent=2) + "\n"
        )
        (self.out / "progress.md").write_text(self.render_markdown())

    def summary_line(self, report: IterationReport) -> str:
        excess = "n/a" if report.mean_excess is None else f"{report.mean_excess:+.1f}"
        return (
            f"anchors solved {report.solve_rate:.2f} "
            f"({sum(a.solved for a in report.attempts)}/{len(report.attempts)}), "
            f"mean excess over optimal {excess}"
        )

    def render_markdown(self) -> str:
        from rf_knots.render import braid_svg, word_label

        game = self.game_config
        lines = [
            "# Braid unknotting progress",
            "",
            f"Anchor set: **{len(self.instances)} frozen instances**, scrambled with "
            f"K={game.scramble_budget} from the unknot on a braid of at most "
            f"{game.max_strands} strands. Every one of them *is* the unknot by "
            "construction, so a failure is always the agent's, never the label's.",
            "",
            "Optimal lengths come from exhaustive breadth-first search; `-` means the "
            "optimum is deeper than the search cutoff, not that the instance is unsolvable.",
            "",
            "## Solve rate by iteration",
            "",
            "| iteration | solved | solve rate | mean moves | mean excess over optimal |",
            "|---:|---:|---:|---:|---:|",
        ]
        for report in self.reports:
            solved = sum(attempt.solved for attempt in report.attempts)
            moves = (
                "-"
                if report.mean_moves_when_solved is None
                else f"{report.mean_moves_when_solved:.1f}"
            )
            excess = "-" if report.mean_excess is None else f"{report.mean_excess:+.2f}"
            lines.append(
                f"| {report.iteration} | {solved}/{len(report.attempts)} "
                f"| {report.solve_rate:.3f} | {moves} | {excess} |"
            )

        if not self.reports:
            return "\n".join(lines) + "\n"

        latest = self.reports[-1]
        lines += ["", f"## Anchor instances at iteration {latest.iteration}", ""]
        for attempt in latest.attempts[: self.showcase]:
            word = tuple(attempt.word)
            status = "solved" if attempt.solved else "**not solved**"
            optimal = "-" if attempt.optimal is None else str(attempt.optimal)
            lines += [
                f"### Anchor {attempt.index} — {status}",
                "",
                f"`{word_label(word, attempt.strands)}` "
                f"(length {attempt.length}, optimal {optimal}, "
                f"agent used {attempt.moves})",
                "",
                braid_svg(word, attempt.strands, title=word_label(word, attempt.strands)),
                "",
            ]
            if attempt.solved and attempt.move_names:
                lines += ["<details><summary>solution</summary>", ""]
                lines += [f"{step + 1}. `{name}`" for step, name in enumerate(attempt.move_names)]
                lines += ["", "</details>", ""]
        return "\n".join(lines) + "\n"


def evaluate_scrambler_difficulty(
    agent,
    games: int,
    *,
    seed: int,
    bfs_depth: int = 6,
) -> dict[str, float]:
    """How hard are the instances this Scrambler actually produces?

    Replaces "win rate against a random Simplifier", which saturates at 1.00 for
    every agent including an untrained one -- a random Simplifier essentially
    never solves anything, so that number cannot distinguish two Scramblers.

    What matters instead is the *exact optimal solution depth* of the instances
    the Scrambler generates, measured by breadth-first search. A uniform-random
    Scrambler buys about 0.7 moves of difficulty per move spent (K=3,4,5 give
    mean optimal depth 2.56, 3.16, 3.96). A trained Scrambler that is worth
    anything has to beat that at the same K.

    Instances are sampled at temperature 1.0: the start state is deterministic,
    so without sampling every game would produce the same word.
    """
    from rf_knots.reference import bfs_unknot

    game = BraidUnknotGame(agent.config.game)
    spec = game.env.spec
    search = NeuralMCTS(game, agent.network, agent.config.search, agent.config.train.device)
    budget = agent.config.game.scramble_budget

    depths: list[int] = []
    lengths: list[int] = []
    beyond_cutoff = 0
    for index in range(games):
        rng = np.random.default_rng(seed + index)
        transition = game.reset(seed + index)
        while not transition.terminated and int(np.asarray(transition.state._phase)) == 0:
            result = search.run(
                transition.state,
                transition.observation,
                transition.legal_actions,
                rng,
                temperature=1.0,
                add_root_noise=True,
            )
            transition = game.step(transition.state, result.action)
        word = tuple(int(x) for x in np.asarray(transition.state._word) if int(x) != 0)
        strands = int(np.asarray(transition.state._n))
        lengths.append(len(word))
        path = bfs_unknot(spec, word, strands, max_depth=bfs_depth, max_growth=budget)
        if path is None:
            beyond_cutoff += 1  # harder than the search cutoff: the interesting case
        else:
            depths.append(len(path))
    return {
        "games": games,
        "mean_optimal_depth": float(np.mean(depths)) if depths else float("nan"),
        "max_optimal_depth": float(max(depths)) if depths else float("nan"),
        "beyond_cutoff": beyond_cutoff / games,
        "mean_word_length": float(np.mean(lengths)),
        "difficulty_per_move": (float(np.mean(depths)) / budget) if depths else float("nan"),
    }
