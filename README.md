# PGX MCTS Bench

Fixed-compute experiments with AlphaZero, MuZero, and alternative tree
exploration rules on 6×6 Go.

This repository is designed for small local experiments, not for reproducing
DeepMind-scale results. It makes the comparison inspectable:

- **AlphaZero** searches exact successor states produced by Pgx.
- **MuZero** searches a learned latent transition and reward model.
- Both use the same Pgx game, self-play schedule, simulation count, arena
  protocol, and compact convolutional capacity.
- `U1` through `U5` can be selected without changing the rest of the pipeline.

## Is 6×6 Go available?

Yes. Pgx exposes `pgx.go.Go(size=6)`. The action space has 36 board points plus
pass. This benchmark uses komi 3.5, four observation-history frames, and an
explicit 72-move cap. To prevent randomly initialized agents from collapsing
into trivial pass-pass games, pass is masked for the first 24 plies for both
agents. The cap is Pgx's default `2 * board_size**2`; both modifications are
part of the artificial benchmark definition and are not claims about a
standard 6×6 ruleset.

## Install

Python 3.11–3.13 is required. On macOS:

```bash
uv sync --extra dev --python 3.12
uv run pgx-mcts-bench rules
```

Fast end-to-end validation:

```bash
uv run pgx-mcts-bench smoke
```

A still-small comparison:

```bash
uv run pgx-mcts-bench compare \
  --exploration u1 \
  --simulations 32 \
  --iterations 10 \
  --selfplay-games 8 \
  --selfplay-positions 256 \
  --train-steps 32 \
  --batch-size 32 \
  --arena-games 40 \
  --channels 32
```

Checkpoints, configuration, training metrics, parameter counts, and arena
results are written under `artifacts/<timestamp>/`.

Run the same small comparison for all five exploration rules:

```bash
uv run pgx-mcts-bench sweep \
  --simulations 16 \
  --iterations 3 \
  --selfplay-games 4 \
  --selfplay-positions 256 \
  --train-steps 16 \
  --arena-games 20
```

Each rule gets a separate directory and the aggregate arena results are saved
to `summary.json`. Use multiple seeds for evidence; a one-seed sweep is only an
engineering check.

`--selfplay-games` is the concurrent search batch size.
`--selfplay-positions` sets the minimum new-position budget per iteration; the
trainer launches additional batches when games are short. Keeping a position
budget avoids rewarding an agent for prematurely ending its own games.

## Exploration rules

With parent count \(N=N(s)\), child count \(n=N(s,a)\), and prior \(P=P(s,a)\):

| Rule | Exploration bonus |
|---|---|
| U1 | \(cP\sqrt{N}/(1+n)\), AlphaZero PUCT |
| U2 | \(cP\sqrt{N}/\sqrt{1+n}\), slow child-count decay |
| U3 | \(c\sqrt{\log(N+1)/(1+n)}\), prior-free UCT control |
| U4 | \(c\sqrt{P\log(N+1)/(1+n)}\), prior-weighted UCT |
| U5 | \(cP\sqrt{N}(c_1+\log((N+c_2+1)/c_2))/(1+n)\), MuZero `pb_c` |

The default constants are `c=1.5`, `c1=1.25`, and `c2=19652`.

## What “MuZero” means here

The MuZero agent has:

1. a representation network mapping the real observation to a latent state;
2. a learned dynamics network mapping latent state and action to a new latent
   state and reward;
3. a prediction network producing policy and value.

Because the purpose is to compare tree search rather than whether a small model
can rediscover Go's rules, the default **rules-aware MuZero** search uses Pgx
for exact legality, termination, and rewards at imagined nodes. Policy, value,
representation, and latent dynamics remain learned. The model also trains
auxiliary legality and terminal heads, allowing a later pure learned-rules
ablation via `SearchConfig.muzero_exact_rules=False`.

This is a compact research implementation. It deliberately omits distributed
self-play, reanalysis, prioritized replay, support-based value transforms,
symmetry augmentation, resignation, and Gumbel Sequential Halving. Those should
be added one at a time rather than hidden inside the initial comparison.

## A statistically meaningful protocol

Do not interpret the `smoke` match as evidence. For each exploration rule:

1. run at least 5 independent seeds;
2. keep simulations, generated positions, optimizer steps, channels, and arena
   games fixed;
3. alternate colors and report score with a bootstrap confidence interval;
4. report wall-clock time and parameter count alongside playing strength;
5. first compare exploration rules within AlphaZero, then repeat within
   MuZero; only afterward compare the two agent families.

MuZero solves a harder problem because it must learn dynamics. Equal wall-clock,
equal parameter, equal generated-position, and equal inference-budget
comparisons answer different questions; a serious report should show more than
one of them.

Arena games use paired random six-ply openings. Each opening is played twice,
with the agents swapping colors, and results include color-conditioned wins.

## Final exact-budget results

The benchmark retains exactly 256 positions per iteration, uses three seeds,
saves learning checkpoints at iterations 1, 3, and 10, and evaluates
same-family and cross-family paired-color matches.

The direct AlphaZero scores against MuZero were 71.7% for U1, 59.2% for U2,
68.3% for U3, 73.3% for U4, and 62.5% for U5 over 120 games per rule. These
percentages do not rank the rules. Cross-play provided the more informative
result:

| Family | U1 vs U2 | U1 vs U3 | U1 vs U4 | U1 vs U5 |
|---|---:|---:|---:|---:|
| AlphaZero | 51–9 | 20–40 | 36–24 | 38–22 |
| MuZero | 36–24 | 18–42 | 26–34 | 18–42 |

U3 beat U1 in every seed for both agent families. The checkpoint arenas did
not show consistent evidence that MuZero overtakes AlphaZero under U1–U4. U5
did show a possible stayer trajectory: AlphaZero's checkpoint score fell from
27/30 at iteration 1 to 17/30 at iteration 10, and MuZero-U5 beat MuZero-U1
42–18. See
[`docs/exact_budget_u1_u5_study.md`](docs/exact_budget_u1_u5_study.md) for the
full configuration, learning curves, color split, compute measurements, device
benchmark, caveats, and next experiment.

The provisional top-two cross-family tournament selected MuZero-U3/U5 and
AlphaZero-U3/U1. AlphaZero-U3 was the clear winner: it beat AlphaZero-U1
40–20, MuZero-U3 40–20, and MuZero-U5 46–14. AlphaZero-U1 placed second by
beating both MuZero finalists. No winner cycle exists within this four-agent
set, although the complete U1–U5 rule round robin is still unfinished.

The Apple M2 MPS path now runs correctly, but the representative workload was
1.6× slower for AlphaZero and 2.8× slower for MuZero than CPU. Because the
predeclared GPU gate failed, no Nebius GPU VM was created.

## Resumable experiments and cross-play

`compare` and `sweep` save atomic checkpoints at requested iterations. An
interrupted run can continue from the latest compatible checkpoint:

```bash
uv run pgx-mcts-bench compare \
  --exploration u3 \
  --simulations 32 \
  --iterations 10 \
  --selfplay-games 8 \
  --selfplay-positions 256 \
  --train-steps 32 \
  --batch-size 32 \
  --arena-games 40 \
  --channels 32 \
  --checkpoint-iterations 1,3,10 \
  --curve-games 10 \
  --exact-positions \
  --output artifacts/u3-seed-0 \
  --resume
```

Compare two trained agents from the same family while preserving each
artifact's own exploration rule:

```bash
uv run pgx-mcts-bench crossplay \
  --first artifacts/u1-seed-0 \
  --second artifacts/u3-seed-0 \
  --kind alphazero \
  --games 40 \
  --output artifacts/crossplay/alphazero-u1-vs-u3.json
```
