"""Staged curriculum over the generator's complexity grade.

`STAGES` rungs, monotone in both the source knot's unknotting number and the
number of scramble moves on top of it -- the count is deliberately not written
down in prose, because it has changed twice and the prose did not.

A candidate trains at one rung until it either reaches the proved unknotting
number there or stops improving on it, then moves up; if it cannot solve the rung
at all it moves up anyway after a cap, so a stuck candidate spends its budget
elsewhere rather than grinding. **The score is the highest stage reached**, which
lets candidates spend different amounts of time per rung.

Training draws from a mixture over the rungs already cleared; evaluation stays
pinned to the frontier. Every promotion re-measures the rungs below, because
climbing is supposed to improve them and where it does not, that is a result.

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
from pgx_mcts_bench.networks import make_braid_network
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games, train_alphazero_step

# (source knot, scramble moves). Monotone in u(K) and in depth.
#
# Stages 10-12 extend the original ten, which `s-window-128` cleared outright and
# four other arms reached the top of. They keep the same shape -- a new source
# knot unscrambled, then scrambled, then a structurally harder knot of the *same*
# unknotting number, which is what the T(2,7)/T(3,4) pair at u=3 already does.
# T(2,9) is u=4 on two strands with 9 crossings; T(3,5) is u=4 on three with 10,
# so stage 12 separates "larger u" from "harder diagram at the same u".
# Graded finer in scramble depth and coarser in u, because the measurement says
# that is where the difficulty actually lives. Every `+0` stage in the first
# ladder promoted in 2 iterations at exactly the proved unknotting number; every
# `+4` stage overshot and none converged. So the source knot is nearly free and
# the scramble is the whole problem. T(2,7) is dropped: it is u=3, the same as
# T(3,4), which is strictly harder (3 strands, 8 letters against 7) -- one knot
# per unknotting number is enough. The plies freed pay for +2 and +8 rungs.
STAGES: list[tuple[str, int]] = [
    ("unknot", 2),
    ("unknot", 6),
    ("T(2,3)", 0),
    ("T(2,3)", 2),
    ("T(2,3)", 4),
    ("T(2,5)", 0),
    ("T(2,5)", 2),
    ("T(2,5)", 4),
    ("T(2,5)", 8),
    ("T(3,4)", 0),
    ("T(3,4)", 2),
    ("T(3,4)", 4),
    ("T(3,4)", 8),
    ("T(2,9)", 0),
    ("T(2,9)", 4),
    ("T(3,5)", 0),
    ("T(3,5)", 4),
    # Beyond u = 4 the rungs are random positive braids rather than torus knots.
    # `u = (c - s + 1) / 2` is exact for any positive braid whose closure is a
    # knot -- torus knots are the special case, and the two formulas are checked
    # against each other on the overlap. That buys many *different diagrams* at
    # each unknotting number, which is the axis the ladder ran out of first: four
    # arms cleared all seventeen torus rungs. Two knots per u, following the
    # established shape of "same u, harder diagram".
    # Rungs 0-16 above are the **calibration set**: every u is a theorem, so the
    # gap to truth is measurable. From here the knots are random mixed-sign words
    # with no label at all.
    #
    # The labelled families are the ones with structure, and that is exactly the
    # problem: every torus knot and every positive braid is fibred, chiral,
    # positive-signature, and satisfies u = g3 = g4. An agent can learn "reduce
    # monotonically, crossing changes always pay", be right on all twenty-five of
    # the old rungs, and have learned nothing that transfers. Ten of those rungs
    # were not even a second family -- on two strands a positive word is
    # sigma_1^c, so `P(2,11)` *is* `T(2,11)`.
    #
    # These have no theorem to reach. Their reference is the ratcheting
    # best-known bound in `bounds.py`: the fewest crossing changes any agent has
    # ever used, improving whenever anyone beats it. Promotion here can only end
    # on plateau or on the cap, never on `objective`.
    ("R(3,10)#0", 0),
    ("R(3,10)#0", 4),
    ("R(5,10)#0", 0),
    ("R(3,12)#0", 0),
    ("R(3,12)#0", 4),
    ("R(5,12)#0", 0),
    ("R(3,14)#0", 0),
    ("R(5,14)#0", 0),
    ("R(3,16)#0", 0),
    ("R(5,16)#0", 0),
    ("R(3,18)#0", 0),
    ("R(5,18)#0", 0),
    ("R(3,20)#0", 0),
    ("R(5,20)#0", 0),
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
    # Binary registers in the head, one TOGGLE action each: the finite control
    # state a memoryless scanning head is missing.
    serial_registers: int = 0
    # Colours the agent may paint onto strands, transported through crossings by
    # the environment. Three actions regardless of the palette size.
    serial_colours: int = 0
    serial_encoder: str = ""
    serial_encoder_states: int = 0
    serial_encoder_prime: int = 5
    train: bool = True


def serial_arms() -> list[Candidate]:
    """The serial grid, as screened in `artifacts/serial-screen`.

    Four factors varied one at a time against a common base rather than crossed.
    All six clear stage 1, which the pre-fix serial candidates could not, so the
    grid measures speed and objective quality rather than whether it works:

    * `act_width`  -- head-only against acting anywhere visible.
    * `simulations` -- 128 against 256; the serial formulation spends plies on
      repositioning, so it may need depth the parallel one does not.
    * `budget`     -- 64 plies against 96, since head motion is charged.
    * `strides`    -- the power-of-two set against the original single stride,
      the ablation for the reachability change.
    """
    base = dict(exploration="u1", simulations=128, channels=32, train_steps=96)
    return [
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


def memory_arms() -> list[Candidate]:
    """Head registers: does a finite control state help a scanning head?

    Matched to `s-head-128` in every other respect, so the comparison is the
    register and nothing else. Registers are *written by the agent*, so there is no
    gradient through the memory and no BPTT -- a TOGGLE is an action and gets its
    credit from MCTS like any other. That is what makes this arm cheap enough to
    run before the learned-embedding version, and it is a fair arm under the
    zero-human-knowledge constraint: a mechanism, not a feature.
    """
    base = dict(exploration="u1", simulations=128, channels=32, train_steps=96,
                serial_window=7, serial_act_width=1)
    return [
        Candidate("s-reg4", "head-only + 4 written registers (16 control states)",
                  serial_registers=4, **base),
        Candidate("s-reg8", "head-only + 8 written registers (256 control states)",
                  serial_registers=8, **base),
    ]


def colour_arms() -> list[Candidate]:
    """Painted strands: memory attached to a thread rather than to a slot.

    The register arms failed because a TOGGLE never changes the word, so the extra
    actions were branches MCTS could not make progress on -- and the cost grew with
    the count. Colours answer both halves of that. There are three actions however
    large the palette, and the environment *transports* a colour through every
    crossing the head passes, so a painted strand stays painted as the diagram is
    rewritten. That is a mechanism the agent can actually read back.

    Matched to `s-head-128` in every other respect.
    """
    base = dict(exploration="u1", simulations=128, channels=32, train_steps=96,
                serial_window=7, serial_act_width=1)
    return [
        Candidate("s-paint2", "head-only + 2 strand colours", serial_colours=2, **base),
        Candidate("s-paint4", "head-only + 4 strand colours", serial_colours=4, **base),
    ]


def invariant_learning_arms() -> list[Candidate]:
    """Automatic whole-tape accumulators under the same serial controller."""
    base = dict(
        exploration="u1",
        simulations=128,
        channels=32,
        train_steps=96,
        serial_window=7,
        serial_act_width=1,
    )
    return [
        Candidate(
            "s-gru128",
            "automatic full-tape scan with unconstrained GRU-128",
            serial_encoder="gru",
            serial_encoder_states=128,
            **base,
        ),
        Candidate(
            "s-fsa32",
            "automatic scan with a learned 32-state soft finite automaton",
            serial_encoder="fsa",
            serial_encoder_states=32,
            **base,
        ),
        Candidate(
            "s-ff4-p5",
            "automatic scan with learned 4x4 matrices over F_5",
            serial_encoder="finite-field",
            serial_encoder_states=4,
            serial_encoder_prime=5,
            **base,
        ),
        Candidate(
            "s-burau-oracle",
            "ORACLE: fixed Burau matrices at t=-1 and t=1/2",
            serial_encoder="burau",
            serial_encoder_states=0,
            **base,
        ),
    ]


def central_benchmark_arms() -> list[Candidate]:
    by_name = {c.name: c for c in serial_arms() + memory_arms() + invariant_learning_arms()}
    return [
        by_name[name]
        for name in (
            "s-head-128",
            "s-reg4",
            "s-reg8",
            "s-gru128",
            "s-fsa32",
            "s-ff4-p5",
            "s-burau-oracle",
        )
    ]


def parallel_arms() -> list[Candidate]:
    return [
        Candidate("no-training", "control: search only, weights never updated", train=False),
        Candidate("u1-puct", "AlphaZero PUCT, parallel head", exploration="u1"),
        Candidate("u3-uct", "prior-free UCT; never collapsed in earlier screens", exploration="u3"),
        Candidate("search-heavy", "128 simulations: is depth the constraint?", simulations=128),
        Candidate("search-light", "16 simulations: the network must carry the policy",
                  simulations=16),
        Candidate("wide-net", "96 channels: is capacity the constraint?", channels=96,
                  train_steps=160),
    ]


def candidates() -> list[Candidate]:
    return (
        parallel_arms()
        + serial_arms()
        + memory_arms()
        + colour_arms()
        + invariant_learning_arms()
    )


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
    # Why the stage ended: cleanly at the objective, on a plateau, or capped.
    reason: str = "capped"
    # Solve rate and crossing changes on every *already cleared* stage, measured
    # with the weights that just cleared this one. Without this the ladder never
    # checks whether climbing costs the stages below, which is the question the
    # promote-on-solve-rate rule cannot answer. Keyed by stage index, measured at
    # the crossing-dominant end of the front only -- that is the number a theorem
    # can be compared against, and evaluating one ratio instead of three keeps it
    # affordable at every promotion.
    retrospective: dict = field(default_factory=dict)


@dataclass
class LadderResult:
    name: str
    rationale: str
    highest_stage: int
    seconds: float
    stages: list[StageResult] = field(default_factory=list)


def _silent(*args, **kwargs) -> None:
    """Module-level so it survives pickling into a worker process."""


def stage_mixture(frontier: int, decay: float) -> tuple[tuple[str, int, float], ...]:
    """Training mixture over stages 0..frontier, geometric back from the frontier.

    `decay = 0` is the original rule -- train only at the frontier -- kept so the
    change is ablatable rather than assumed. `decay = 0.5` gives the frontier half
    the mass, the stage below a quarter, and so on, which keeps the tail cheap
    while never dropping a cleared stage out of the distribution entirely.

    Weight is per *stage*, so the deeper the ladder the thinner each old rung
    gets. That is the intended shape: the point is to stop the forgetting-shaped
    residual, not to re-train the whole curriculum every iteration.
    """
    if decay <= 0.0:
        return ()
    weights = [decay ** (frontier - i) for i in range(frontier + 1)]
    total = sum(weights)
    return tuple(
        (STAGES[i][0], STAGES[i][1], w / total) for i, w in enumerate(weights)
    )


def resume_point(saved_stages: list[dict]) -> tuple[int, set, list[str]]:
    """`(start_stage, cleared identities, gaps below the start)`.

    Keyed on `(source, scramble)` rather than on the stage index. The stage list is
    edited as measurements come in -- rungs inserted, a redundant knot dropped --
    and an index-keyed resume silently lands on a different rung after every such
    edit, with weights that never saw it. Identity matching also fills gaps: a
    candidate that cleared the old ladder starts at the first *newly inserted*
    stage it has not actually done, which is the honest place to restart.
    """
    cleared = {
        (row["source"], row["scramble"]) for row in saved_stages if row.get("promoted")
    }
    start = next((i for i, s in enumerate(STAGES) if s not in cleared), len(STAGES))
    # Gaps are the uncleared rungs *below the highest cleared one* -- newly
    # inserted stages that a candidate skipped past on the previous ladder.
    # Everything below `start` is cleared by construction, so measuring gaps
    # against `start` would always be empty.
    highest = max((i for i, s in enumerate(STAGES) if s in cleared), default=-1)
    gaps = [f"{s[0]}+{s[1]}" for s in STAGES[: highest + 1] if s not in cleared]
    return start, cleared, gaps


def promotion_reason(
    solve_rate: float,
    crossings: float,
    history: list[float],
    optimal: int,
    *,
    promote_at: float,
    tolerance: float,
    window: int,
    worst_ratio: float = 1.0,
    collapse_floor: float = 0.5,
) -> str | None:
    """Why this stage should end now, or `None` to keep training.

    Solve rate is a *feasibility* signal and the objective is crossing changes, so
    the original rule advanced on the wrong quantity: it fired as soon as the
    stage could be solved, long before it was solved well. Measured consequence --
    `s-window-128` promoted `T(2,3)+4` at 4.18 crossing changes against an optimum
    of 1.

    Gating on the objective alone would be the opposite mistake. The same
    measurement showed `T(2,5)+4` improving faster from *later* stages than it was
    improving in place, so grinding until optimal would spend budget worse than
    moving up. Hence two exits: reaching the objective, or plateauing on it.
    """
    # `solve_rate` is pooled across every evaluation episode, not the minimum of
    # the three per-ratio rates. The minimum was a conjunction of three noisy
    # twelve-game tests, and at promote_at = 0.8 that makes 10/12 the first
    # passing value and 9/12 the first failing one -- so promotion turned on a
    # single episode, three times over. On `unknot+6` it eliminated six of
    # seventeen arms, nearly every survivor landing exactly on 0.83 = 10/12.
    #
    # The minimum still matters, but as a *collapse* check rather than the gate:
    # a network serving one end of the Pareto front and abandoning another is a
    # real failure, and `collapse_floor` catches it without letting sampling
    # noise decide the ladder.
    if solve_rate < promote_at or worst_ratio < collapse_floor:
        return None
    # `optimal < 0` means u is not known -- a random knot rather than a torus
    # knot or positive braid. There is no theorem to reach, so the objective exit
    # cannot fire and the rung ends on plateau or on the cap. Falling through to
    # the tolerance test would compare against a sentinel and promote instantly.
    if optimal < 0:
        if len(history) >= 2 * window and min(history[-window:]) > min(history[:-window]) - 0.01:
            return "plateau"
        return None
    if crossings == crossings and crossings <= optimal + tolerance:
        return "objective"
    # Needs two windows of history: one to establish a best, one to fail to beat
    # it. Otherwise the first two flat evaluations of a stage read as a plateau.
    # The comparison is recent-window against everything *before* it -- comparing
    # against the whole history includes the recent window, so a monotonically
    # improving run has its global best inside the window and reads as flat.
    if len(history) >= 2 * window:
        if min(history[-window:]) > min(history[:-window]) - 0.01:
            return "plateau"
    return None


def _config(
    candidate: Candidate,
    stage: tuple[str, int],
    seed: int,
    device: str,
    *,
    frontier: int = -1,
    mix_decay: float = 0.0,
):
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
        generator_max_crossings=22,
        generator_max_scramble=6,
        generator_positive_braids=3,
        generator_positive_seed=0,
        generator_random_crossings=(10, 12, 14, 16, 18, 20),
        generator_random_per_grade=1,
        generator_random_seed=0,
        stage_source=stage[0],
        stage_scramble=stage[1],
        stage_mix=stage_mixture(frontier, mix_decay) if frontier >= 0 else (),
        serial_window=candidate.serial_window,
        serial_act_width=candidate.serial_act_width,
        serial_shift_strides=candidate.serial_shift_strides,
        serial_registers=candidate.serial_registers,
        serial_colours=candidate.serial_colours,
        serial_encoder=candidate.serial_encoder,
        serial_encoder_states=candidate.serial_encoder_states,
        serial_encoder_prime=candidate.serial_encoder_prime,
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


def evaluate_stage(
    game,
    network,
    config,
    games: int,
    seed: int,
    ratios: tuple[float, ...] = RATIOS,
    bounds_path: Path | None = None,
    agent: str = "",
) -> dict[float, dict]:
    """Per-ratio solve rate, crossing changes and moves on held-out instances.

    Evaluated at each of A:B = 1000:1, 10:1, 1:10 separately, so it is visible
    whether one network is really serving all three or collapsing to one policy.
    `ratios` narrows that: the retrospective pass measures only the
    crossing-dominant end, since that is the number a theorem compares against.
    """
    search = NeuralMCTS(game, network, config.search, config.train.device)
    out: dict[float, dict] = {}
    best_claim: tuple[int, int] | None = None
    for ratio in ratios:
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
                used = int(np.asarray(final._crossing_changes))
                spent = config.game.simplify_budget - int(np.asarray(final._budget))
                crossings += used
                moves += spent
                # Every solve is a witness for u(K) <= used, whatever rung or
                # ratio produced it. The best across the whole evaluation is
                # claimed once, rather than one write per episode.
                if best_claim is None or (used, spent) < best_claim:
                    best_claim = (used, spent)
        # `crossings` and `moves` are conditional on solving, which makes them
        # anti-correlated with `solved`: an arm that gives up on the hard
        # instances drops them out of the average, so failing more can make its
        # cost look *better*. `expected_*` divides by the solve rate, giving the
        # cost of *obtaining* a solution rather than the cost of the ones that
        # happened to land. At 1.00 solved the two agree; at 0.50 solved with 5.00
        # crossing changes the conditional figure is 5.00 and the expected one is
        # 10.00, which is the ordering a leaderboard wants.
        rate = solved / games
        out[ratio] = {
            "solved": rate,
            "crossings": crossings / solved if solved else float("nan"),
            "moves": moves / solved if solved else float("nan"),
            "expected_crossings": (crossings / solved) / rate if solved else float("nan"),
            "expected_moves": (moves / solved) / rate if solved else float("nan"),
        }
    if bounds_path is not None and best_claim is not None:
        from pgx_mcts_bench import bounds

        source = next(
            s for s in game.generator.sources if s.name == config.game.stage_source
        )
        bounds.claim(
            bounds_path,
            bounds.Bound(
                knot=bounds.knot_id(source.word, source.strands),
                crossings=best_claim[0],
                moves=best_claim[1],
                agent=agent,
                witness=list(source.word),
                strands=source.strands,
            ),
        )
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
    mix_decay: float = 0.5,
    crossing_tolerance: float = 0.25,
    plateau_window: int = 3,
    collapse_floor: float = 0.5,
    max_consecutive_caps: int = 3,
    stop_after: int = -1,
    min_iterations_per_rung: float = 0.0,
    min_iterations_from: int = 0,
    bounds_path: Path | None = None,
    retro_games: int = 6,
    log=print,
) -> LadderResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    first = _config(candidate, STAGES[0], seed, device)
    network = make_braid_network(first.game, first.model)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
    replay = ReplayBuffer(20_000, rng)
    result = LadderResult(candidate.name, candidate.rationale, -1, 0.0)

    # Resume: pick up at the first stage this candidate has *not* cleared, with
    # the weights that cleared the last one. Matching is by stage **identity**
    # (source, scramble), not by index -- the stage list is edited as the
    # measurements come in, and an index-keyed resume silently means a different
    # rung after every such edit. A ladder run is long enough that losing it to a
    # laptop closing would be silly, and long enough that resuming onto the wrong
    # rung would be worse.
    start_stage = 0
    path = checkpoint_dir / f"{candidate.name}.pt" if checkpoint_dir else None
    if path is not None and path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        network.load_state_dict(saved["network"])
        optimizer.load_state_dict(saved["optimizer"])
        start_stage, cleared, gaps = resume_point(saved["stages"])
        result.stages = [
            StageResult(**row)
            for row in saved["stages"]
            if (row["source"], row["scramble"]) in cleared
        ]
        result.highest_stage = start_stage - 1
        log(f"    [{candidate.name}] resumed at stage {start_stage} "
            f"({len(cleared)} cleared" + (f", filling {gaps}" if gaps else "") + ")")

    def snapshot(index: int, when: str, stage_result: StageResult | None = None) -> None:
        """Weights either side of a stage, kept rather than overwritten.

        The resume pointer (`<name>.pt`) only ever holds the last *promoted*
        state, which makes the trajectory through a stage invisible -- and the
        trajectory is what a promote-on-solve-rate rule hides. `crossings` at
        promotion measures wherever the network happened to be when it crossed
        the threshold, so telling improvement from threshold-crossing needs the
        before and after weights side by side. Optimizer state is deliberately
        omitted: it doubles the size and nothing downstream replays training
        from a snapshot.
        """
        if checkpoint_dir is None:
            return
        directory = checkpoint_dir / candidate.name
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": network.state_dict(),
                "candidate": candidate.name,
                "stage": index,
                "source": STAGES[index][0],
                "scramble": STAGES[index][1],
                "when": when,
                "stage_result": asdict(stage_result) if stage_result else None,
            },
            directory / f"stage{index:02d}-{when}.pt",
        )

    consecutive_caps = 0
    for index, stage in enumerate(STAGES):
        if index < start_stage:
            continue
        # Replication runs climb only the prefix that every arm has in common.
        # A second seed exists to put error bars on a comparison, and rungs no
        # other arm has reached contribute nothing to that while costing the most.
        if 0 <= stop_after < index:
            log(f"    [{candidate.name}] stopping after rung {stop_after}")
            break
        snapshot(index, "before")
        # Training draws from a mixture over stages 0..index; evaluation stays
        # pinned to `index`, because `evaluate_stage` builds its instances from
        # `stage_source`/`stage_scramble` directly rather than through the
        # generator's sampler. One game object therefore serves both.
        config = _config(
            candidate, stage, seed, device, frontier=index, mix_decay=mix_decay
        )
        game = make_game(config.game)
        source = next(s for s in game.generator.sources if s.name == stage[0])
        search = NeuralMCTS(game, network, config.search, device)
        stage_started = time.perf_counter()
        solve_rate = crossings = float("nan")
        by_ratio: dict = {}
        promoted = False
        reason = "capped"
        iterations = 0
        history: list[float] = []

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
                    game, network, config, eval_games, seed + 500_000 + index * 997,
                    bounds_path=bounds_path, agent=candidate.name,
                )
                rates = [v["solved"] for v in by_ratio.values()]
                # Pooled over every episode rather than the worst ratio: see
                # `promotion_reason`. The worst ratio is kept as a collapse check.
                solve_rate = sum(rates) / len(rates)
                worst_ratio = min(rates)
                crossings = by_ratio[max(RATIOS)]["crossings"]
                # Unsolved sorts as worst rather than as missing, so a stage that
                # stops solving reads as "not improving" instead of dropping out
                # of the plateau test entirely.
                history.append(crossings if crossings == crossings else float("inf"))
                verdict = promotion_reason(
                    solve_rate, crossings, history, source.unknotting_number,
                    promote_at=promote_at, tolerance=crossing_tolerance,
                    window=plateau_window, worst_ratio=worst_ratio,
                    collapse_floor=collapse_floor,
                )
                if verdict is not None:
                    # A training-density floor, applied to the *average* over
                    # rungs rather than per rung: an arm that spent forty
                    # iterations on one hard rung has earned the right to fly
                    # through an easy one, and forcing seven on `unknot+2` buys
                    # nothing. This equalises training effort so comparisons stop
                    # being confounded -- `total_it` currently varies 7x across
                    # the field. It is a measurement fix, not a performance one:
                    # the arm with the *lowest* average is also the best, and the
                    # ones training 4-5x more sit mid-table.
                    # The average is taken over the rungs the floor *governs*,
                    # not over the whole climb. Averaging from rung 0 would make
                    # an arm that cleared the cheap early rungs in two iterations
                    # each pay that debt back on the hard ones -- `search-heavy`
                    # would owe 65 extra iterations at rung 10 for having been
                    # efficient at rung 3. "From rung N, average M" is the rule
                    # that means what it says.
                    governed = [s for s in result.stages if s.stage >= min_iterations_from]
                    if index >= min_iterations_from:
                        spent = sum(s.iterations for s in governed) + iterations
                        density = spent / (len(governed) + 1)
                    else:
                        density = float("inf")  # below the floor's remit
                    if density >= min_iterations_per_rung:
                        promoted, reason = True, verdict
                        break
            if not candidate.train:
                break

        # Look back: are the stages already cleared still solved, and at what cost?
        # Nothing measured this before, so "climbing costs the rungs below" was
        # neither confirmed nor ruled out. Only on promotion -- there is nothing to
        # look back on from a stage that was never cleared.
        retrospective: dict = {}
        if promoted and index > 0:
            cc_edge = max(RATIOS)
            for earlier in range(index):
                back = _config(candidate, STAGES[earlier], seed, device)
                back_game = make_game(back.game)
                rows = evaluate_stage(
                    back_game, network, back, retro_games,
                    seed + 700_000 + earlier * 997, ratios=(cc_edge,),
                )[cc_edge]
                back_source = next(
                    s for s in back_game.generator.sources if s.name == STAGES[earlier][0]
                )
                retrospective[str(earlier)] = {
                    "source": STAGES[earlier][0],
                    "scramble": STAGES[earlier][1],
                    "solved": rows["solved"],
                    "crossings": rows["crossings"],
                    "optimal_crossings": back_source.unknotting_number,
                }

        stage_result = StageResult(
            stage=index,
            by_ratio={str(k): v for k, v in by_ratio.items()},
            source=stage[0],
            scramble=stage[1],
            iterations=iterations,
            solve_rate=solve_rate,
            crossings=crossings,
            optimal_crossings=source.unknotting_number,
            promoted=promoted,
            reason=reason,
            retrospective=retrospective,
            seconds=time.perf_counter() - stage_started,
        )
        result.stages.append(stage_result)
        snapshot(index, "after", stage_result)
        regressed = [
            f"{v['source']}+{v['scramble']} {v['crossings']:.2f}/{v['optimal_crossings']}"
            for v in retrospective.values()
            if v["solved"] < promote_at
        ]
        log(
            f"    [{candidate.name}] stage {index} {stage[0]}+{stage[1]} "
            f"(u={source.unknotting_number}): solved {solve_rate:.2f} "
            f"crossings {crossings:.2f} after {iterations} it ({reason})"
            + (f"  REGRESSED: {', '.join(regressed)}" if regressed else "")
        )
        if promoted:
            consecutive_caps = 0
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
            # A capped rung no longer ends the candidate. The docstring has always
            # said a stuck candidate "moves up anyway after a cap, so it spends its
            # budget elsewhere rather than grinding"; the code broke instead, so a
            # single bad rung retired an arm permanently. On the 25-rung ladder
            # `unknot+6` -- rung 1 of 25, and an *unknot* -- ended six of seventeen
            # arms before the ladder had measured anything about them.
            #
            # Bounded, because an arm that cannot clear three rungs in a row is not
            # going to clear the fourth, and its cores are better spent elsewhere.
            consecutive_caps += 1
            if consecutive_caps >= max_consecutive_caps:
                log(
                    f"    [{candidate.name}] stopping: {consecutive_caps} rungs "
                    f"capped in a row"
                )
                break

    result.seconds = time.perf_counter() - started
    return result


def render(results: list[LadderResult]) -> str:
    """The run's own report.

    Everything describing the ladder is derived from **what these results
    recorded**, never from the module-level `STAGES`. The header used to hardcode
    "Ten stages ... capped at 25 iterations", which survived two changes to the
    stage list and one to the promotion rule and so described a run that had not
    happened. Reading the current `STAGES` instead would be the same bug one level
    down: re-rendering an old run would relabel it with today's rungs.
    """
    recorded: dict[int, tuple[str, int]] = {}
    for r in results:
        for st in r.stages:
            recorded[st.stage] = (st.source, st.scramble)
    rungs = ", ".join(f"{recorded[i][0]}+{recorded[i][1]}" for i in sorted(recorded))
    matches_current = all(
        index < len(STAGES) and STAGES[index] == rung for index, rung in recorded.items()
    )
    lines = [
        "# Ladder: how far up the complexity grade does each candidate get?",
        "",
        f"{len(recorded)} stages seen in this run, monotone in the source knot's",
        "unknotting number and in scramble depth. Score is the highest stage cleared.",
        "",
        f"Rungs: {rungs}.",
        "",
    ]
    if not matches_current:
        lines += [
            "> **Historical.** These rungs are not the current stage list, so the",
            "> stage indices here do not line up with a run made today. Compare by",
            "> rung name, not by number.",
            "",
        ]
    lines += [
        "A stage ends for one of three reasons, and which one it was is recorded:",
        "",
        "* `objective` — solved, at or within tolerance of the proved unknotting",
        "  number. This is the only exit that means the stage was solved *well*.",
        "* `plateau` — solved, but crossing changes stopped improving. Moving up is",
        "  worth more than grinding, because training at a higher stage measurably",
        "  improves the lower ones.",
        "* `capped` — hit the iteration limit without clearing the promotion bar.",
        "",
        "| candidate | highest stage | reached | total iterations | seconds |",
        "|---|---:|---|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: -x.highest_stage):
        cleared = [s for s in r.stages if s.promoted]
        last = cleared[-1] if cleared else None
        reached = f"`{last.source}+{last.scramble}`" if last else "—"
        lines.append(
            f"| `{r.name}` | {r.highest_stage} | {reached} "
            f"| {sum(s.iterations for s in r.stages)} | {r.seconds:.0f} |"
        )

    regressions = [
        (r.name, st, key, row)
        for r in results
        for st in r.stages
        for key, row in sorted((st.retrospective or {}).items(), key=lambda kv: int(kv[0]))
        if row.get("solved", 1.0) < 0.8
    ]
    if regressions:
        lines += [
            "",
            "## Regressions",
            "",
            "Rungs already cleared that the weights no longer solve, measured at the",
            "crossing-dominant end after each promotion. Climbing is supposed to",
            "improve the rungs below; where it does not, this is the evidence.",
            "",
            "| candidate | after clearing | regressed rung | solved | cc | u(K) |",
            "|---|---|---|---:|---:|---:|",
        ]
        for name, st, _key, row in regressions:
            cc = row.get("crossings", float("nan"))
            cc = "—" if cc != cc else f"{cc:.2f}"
            lines.append(
                f"| `{name}` | {st.source}+{st.scramble} "
                f"| {row['source']}+{row['scramble']} | {row['solved']:.2f} | {cc} "
                f"| {row['optimal_crossings']} |"
            )

    lines += ["", "## Per stage", ""]
    for r in sorted(results, key=lambda x: -x.highest_stage):
        lines.append(f"### `{r.name}` — {r.rationale}")
        lines.append("")
        lines.append(
            "| stage | instance | u(K) | it | why | A:B=1000:1 cc/moves "
            "| 10:1 cc/moves | 1:10 cc/moves |"
        )
        lines.append("|---:|---|---:|---:|---|---|---|---|")
        for st in r.stages:
            cells = []
            for ratio in ("1000.0", "10.0", "0.1"):
                v = st.by_ratio.get(ratio)
                # cc and moves are conditional on solving; the solve rate is
                # printed alongside because without it they cannot be read.
                cells.append(
                    "—" if not v else
                    f"{v['crossings']:.2f} / {v['moves']:.1f} ({v['solved']:.0%})"
                )
            lines.append(
                f"| {st.stage} | {st.source}+{st.scramble} | {st.optimal_crossings} "
                f"| {st.iterations} | {st.reason} | " + " | ".join(cells) + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def save(results: list[LadderResult], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "ladder.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=float) + "\n"
    )
    (out / "ladder.md").write_text(render(results))


def load(path: Path) -> list[LadderResult]:
    """Results from a `ladder.json`, whatever stage list wrote it."""
    return [
        LadderResult(
            name=row["name"],
            rationale=row["rationale"],
            highest_stage=row["highest_stage"],
            seconds=row["seconds"],
            stages=[StageResult(**s) for s in row["stages"]],
        )
        for row in json.loads(path.read_text())
    ]


def rescore(
    root: Path,
    *,
    games: int = 12,
    simulations: int = 0,
    device: str = "cpu",
    log=print,
) -> dict:
    """Re-measure every cleared rung with each candidate's **final** weights.

    A rung's `cc` is recorded once, at the moment it was promoted, and never
    revisited -- so a leaderboard built from those numbers compares networks of
    very different maturity and presents it as one column. That is how `u1-puct`
    came to show 9.00 crossing changes on `T(3,4)+0` against an optimum of 3: the
    figure was true of the network that cleared the rung and false of the network
    that exists now, which scores exactly 3.00.

    `simulations` overrides the search budget. Worth using, because the same
    weights on `T(3,5)+0` score 14.00 at 64 simulations and 6.00 at 128 -- at the
    crossing-dominant end of the front the optimum is a long Reidemeister path,
    and finding it is a search problem. A single number there conflates what the
    network prefers with what its search can reach.
    """
    from dataclasses import replace as _replace

    by_name = {c.name: c for c in candidates()}
    out: dict = {}
    for path in sorted(root.glob("*/checkpoints/*.pt")) + sorted(root.glob("checkpoints/*.pt")):
        if path.is_dir() or path.stem not in by_name:
            continue
        candidate = by_name[path.stem]
        if simulations:
            candidate = _replace(candidate, simulations=simulations)
        saved = torch.load(path, map_location=device, weights_only=False)
        cleared = [s for s in saved.get("stages", []) if s.get("promoted")]
        if not cleared:
            continue
        first = _config(candidate, STAGES[0], 0, device)
        network = make_braid_network(first.game, first.model)
        network.load_state_dict(saved["network"])
        network.eval()

        rows = []
        for row in cleared:
            stage = (row["source"], row["scramble"])
            if stage not in STAGES:
                continue
            index = STAGES.index(stage)
            config = _config(candidate, stage, 0, device)
            game = make_game(config.game)
            source = next(s for s in game.generator.sources if s.name == stage[0])
            measured = evaluate_stage(
                game, network, config, games, 900_000 + index * 997
            )
            rows.append({
                "stage": index, "source": stage[0], "scramble": stage[1],
                "optimal_crossings": source.unknotting_number,
                "then": row.get("crossings", float("nan")),
                "now": measured[max(RATIOS)]["crossings"],
                "solved": min(v["solved"] for v in measured.values()),
                "by_ratio": {str(k): v for k, v in measured.items()},
            })
            log(f"    [{path.stem}] {stage[0]}+{stage[1]} u={source.unknotting_number} "
                f"then {rows[-1]['then']:.2f} -> now {rows[-1]['now']:.2f}")
        out[path.stem] = rows
    (root / "rescore.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def merge(root: Path) -> list[LadderResult]:
    """Combine per-candidate output directories into one report.

    Running one process per candidate is what keeps every core busy, but it leaves
    a `ladder.json` per directory and no combined view -- so the run has no report
    of itself, which is how a stale one survives.
    """
    results: list[LadderResult] = []
    for path in sorted(root.glob("*/ladder.json")):
        results.extend(load(path))
    if results:
        save(results, root)
    return results
