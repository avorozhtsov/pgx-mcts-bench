# What is actually being trained on the braid game

Written because the design notes describe the *intended* league and
`rf-knots/docs/representation.md` describes the *environment*, but what the code
trains today was written down nowhere.

## The objective

Not "unknot a knot in under K moves". It is a two-player zero-sum game:

```
phase 0   Scrambler   K plies   start from the empty 1-braid (= the unknot) and
                                apply K type-preserving moves. the closure is
                                STILL the unknot, by construction.
phase 1   Simplifier  M plies   sees only the resulting word, not the history.
                                wins iff it reaches the empty 1-braid.
reward                          +1 to the Simplifier if it reaches it within M,
                                else -1. Zero-sum.
```

`K` is the **Scrambler's move budget** and the difficulty dial. `M` is the
**Simplifier's move budget** — the number of moves it may spend, not a target to
beat. Nothing rewards a *shorter* solution unless `simplifier_speed_bonus > 0`,
in which case a win pays `1 - λ(1 - moves_left/M)` and a fast win beats a slow one.

Both roles are played by the **same network**, distinguished by a phase channel
in the observation.

## There are no "simple knots" — every instance is the unknot

This is the single most important thing to be clear about. The Simplifier never
sees a trefoil, or a figure-eight, or anything from a knot table. **Every
instance is the unknot**, presented as a scrambled braid word. The difficulty
lives entirely in the *diagram*, never in the knot type.

That is deliberate and it is what makes the ground truth exact and free at every
difficulty: instances are generated *from* the answer, so a failure is always the
agent's and never the label's. No majority vote, no verifier drift.

Knot tables enter only later, at roadmap M3, as held-out evaluation.

## What the instances look like

Actual scrambles at each curriculum step, with their exact BFS-optimal solution
lengths:

| K | example instance | optimal |
|---|---|---|
| 1 | `B2: s1` | 1 |
| 2 | `B3: s1 s2` | 2 |
| 3 | `B3: s1 s2^-1 s2 s2` | 3 |
| 4 | `B3: s1 s2^-1 s2 s2^-1 s2 s2` | 4 |
| 6 | `B3: s1 s2^-1 s2 s2^-1 s1^-1 s1 s2 s2 s2^-1 s2` | 6 |

The simplest possible instance is `K=1`: one stabilisation, giving `σ₁` on two
strands, solved by a single destabilisation. Note `K=2` can produce the empty
1-braid outright — a scramble that undoes itself — which the anchor set filters
out because it carries no signal.

## The schedule

Defaults as of this document; see `braid_sweep.Variant` and `config.TrainConfig`.

**Tiers** (`BRAID_TIERS` in `cli.py`):

| | `max_len` L | `max_strands` N | K | M | actions |
|---|---|---|---|---|---|
| tier0 | 32 | 5 | 6 | 24 | 388 |
| tier1 | 64 | 8 | 12 | 48 | 1156 |

**Per iteration:** 8 self-play games → replay buffer → 64 optimizer steps at
batch 32, AdamW lr 1e-3. 8 iterations by default. MCTS runs 32 simulations per
move; the network is ~40k parameters (32 channels, 2 residual blocks).

**Curriculum** (`curriculum_start_k`, off by default, on in the `curriculum`
variants):

```
train at K = 1
after each iteration:
    if the Simplifier's self-play win rate >= 0.5 and K < target:
        K += 1
```

So a run climbs 1 → 2 → … → 6 as fast as it keeps winning, and stalls whenever it
does not. A typical trace:

```
K:         2   3   4   5   6   6   6   6
win rate: .88 1.0 1.0 1.0 1.0 1.0 .88 .86
```

This exists because at a fixed K=6 the Simplifier won *no* self-play games in
4 of 42 runs, every training target then read "the Simplifier lost", and those
runs were unrecoverable from iteration 1. The curriculum removed that entirely:
0 collapses in 6 seeds, worst case 0.917 against 0.708 for fixed K.

## What is measured

**The anchor set is the benchmark.** 24 instances scrambled from fixed seed
10 000 at the *target* K, frozen for the whole project, with exact BFS-optimal
solution lengths cached on disk. The curriculum changes what the agent *trains*
on; it never changes what it is *scored* on.

| metric | meaning |
|---|---|
| `solve rate` | fraction of the 24 anchors untied within M |
| `excess` | moves used beyond a shortest solution, averaged over solved anchors |
| `Scr depth` | BFS-optimal depth of instances the **Scrambler** produces — the proposer-side metric. Random-Scrambler baselines: K=3 → 2.56, K=4 → 3.16, K=5 → 3.96, K=6 → 4.60 |

`no-training` — the same network never updated — is always run as a control,
because MCTS alone solves a large fraction of anchors and any learning claim has
to beat that number rather than zero.

## Current state

- Simplifier: `curriculum` reaches **0.951 ± 0.029** on the anchors, 0 collapses
  in 6 seeds, versus 0.375 for the untrained control.
- Scrambler: **unproven**. Trained Scramblers produce instances of BFS depth
  ~3.5–4.0 against a random baseline of 4.60 at K=6. Under curriculum the trend
  is upward (2.95 → 3.79 over 8 iterations) but it has not yet beaten random.

## Where the rest is written down

| file | contents |
|---|---|
| `rf-knots/docs/representation.md` | the encoding, the move set, the Reidemeister/Markov correspondence |
| `rf-knots/research/01-game-design.md` | the designed N-agent league, and the revision replacing "hard for others" with learning progress |
| `rf-knots/research/08-roadmap.md` | milestones |
| this file | what the code trains today |
