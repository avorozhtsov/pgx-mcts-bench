# The ladder: plan, state, and how to continue

Long-run comparison of candidate architectures on a staged curriculum of knots,
scored by **how far up the complexity grade each one gets**.

## The task

Instances come from `rf_knots.generator.GradedGenerator`: a torus knot plus `N`
uniform-random scramble moves. Crossing changes are enabled, so every instance is
solvable, and the source family carries **proved unknotting numbers** —
`u(T(p,q)) = (p-1)(q-1)/2`, Kronheimer–Mrowka. The agent is therefore scored
against a theorem, not against a search that may not have looked hard enough.

Objective: minimise `A · crossing_changes + B · total_moves`. Three ratios are
sampled per episode and evaluated separately:

| A : B | log(A/B) | meaning |
|---|---:|---|
| 1000 : 1 | +6.9 | crossing changes dominate — unknotting-number minimisation |
| 10 : 1 | +2.3 | crossing changes preferred, moves still matter |
| 1 : 10 | −2.3 | moves dominate — shortest path, crossing changes cheap |

The network is conditioned on `log(A/B)` through **FiLM**, so one set of weights
is meant to serve all three rather than collapsing to a compromise.

## The stages

| # | instance | u(K) |
|---:|---|---:|
| 0 | unknot + 2 | 0 |
| 1 | unknot + 6 | 0 |
| 2 | T(2,3) + 0 | 1 |
| 3 | T(2,3) + 4 | 1 |
| 4 | T(2,5) + 0 | 2 |
| 5 | T(2,5) + 4 | 2 |
| 6 | T(2,7) + 0 | 3 |
| 7 | T(2,7) + 4 | 3 |
| 8 | T(3,4) + 0 | 3 |
| 9 | T(3,4) + 4 | 3 |

**Promotion:** solve ≥ 80% of 12 held-out instances *at every ratio* (the minimum
across the three, so the hardest setting governs), evaluated every 2 iterations,
capped at 25 iterations per stage. Held-out means a seed stream disjoint from
training. Candidates therefore spend different amounts of time per stage, which
is intended.

**Score:** highest stage cleared.

## Candidates

| name | what it tests |
|---|---|
| `s-head-128` | moving window, acting **only at the head** |
| `s-window-128` | moving window, acting **anywhere it can see** |
| `s-w11-128` | wider local window |
| `s-tape4` | aligned writable four-symbol memory tape |
| `s-scan-gru` | compulsory scan followed by recurrent full-word memory |

All supported candidates use the bounded serial formulation, whose action space
is independent of `L`. Every candidate must make a semantic braid edit after at
most five consecutive internal controller actions.

## Continuing tomorrow

State is checkpointed **after every cleared stage** to
`artifacts/ladder-run/checkpoints/<candidate>.pt`, holding the network weights,
the optimizer state, every stage result so far, and the highest stage cleared.
Re-running the same command resumes each candidate at the stage after its last
cleared one, with the weights that cleared it:

```bash
uv run pgx-mcts-bench braid-ladder --max-iterations 25 --eval-games 12 \
    --promote-at 0.8 --workers 5 --output artifacts/ladder-run
```

To push a specific candidate further, give it a larger cap:

```bash
uv run pgx-mcts-bench braid-ladder --only s-window-128 \
    --max-iterations 80 --output artifacts/ladder-run
```

Results land in `artifacts/ladder-run/ladder.md` (per-stage, per-ratio crossing
changes and moves) and `ladder.json`.

## Known limitations, so they are not rediscovered

**The ladder may be too easy.** A smoke run cleared all ten of an earlier, softer
stage list in 2 iterations per stage, with exactly optimal crossing changes
(3.00 against `u(T(2,7)) = 3`). The stages here are harder, but if several
candidates still reach stage 9 the ladder cannot discriminate and needs
extending — larger `p` in `T(p,q)`, or more scramble depth.

**The A/B axis is probably inert on these instances, and that is a property of
the environment, not a bug.** An exhaustive Pareto-front check found
`moves[k] = m₀ + k` on every instance tested: each extra crossing change costs
exactly one extra move and never *saves* any, because a crossing change *is* a
move and does not collapse a tangle by itself. So

```
λ·k + moves[k] = m₀ + k·(λ + 1)
```

is minimised at the smallest `k` for every `λ > 0`. The three ratios should
therefore produce the *same* policy here. Where a genuine trade-off should exist
is **hard unknot diagrams** — `u = 0` but requiring many moves and a temporary
increase in crossing number — where one crossing change might collapse a tangle
that would otherwise take twenty Reidemeister moves. Running the same front check
on those is the outstanding question.

**At A:B = 1000:1 the move tiebreak is below the value head's resolution.** One
extra move changes the reward by `6.2e-05`, against roughly `1e-4` that a tanh
head trained on noisy returns can resolve. That ratio is effectively "minimise
crossing changes, ignore moves". Correct in principle, and worth knowing before
reading a flat moves column as a finding.
