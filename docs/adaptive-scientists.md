# Adaptive scientists curriculum

This experiment replaces the fixed knot order with a curriculum proposed by a
group of independently trained rung-23 networks. The fixed ladder remains the
control experiment.

The pool is frozen before training: the 200 nontrivial knots with smallest
minimal crossing number in the `rf-knots` table, after filtering to the rung-23
network envelope (at most 5 strands and a braid word of at most 48 letters).
Every scientist evaluates every remaining knot and assigns

```text
simplicity = p(solve) * (20 - predicted_crossing_changes).
```

Each scientist samples one proposal from a softmax over its scores. The group
samples among those proposals using

```text
priority = alpha * simplicity + rounds_since_this_scientist_was_accepted.
```

If a scientist has been ignored for `2*N` rounds, sampling is overridden and
that scientist's highest-scoring remaining knot is accepted. Thus diversity is
not dependent on a lucky softmax draw.

Every scientist then searches the selected knot and trains on its own games. If
any scientist solves it, the best trajectory is replayed with the independent
`rf-knots` reference semantics. A verified trajectory is copied into every
peer's replay buffer. Its costs are trained as one-sided upper bounds: predicting
a better cost is allowed, while predicting a worse cost is penalized. A witness
is not treated as proof of optimality.

The historical rung-23 checkpoints do not contain trained factorized solve/cost
heads. By default, round zero uses an explicitly labelled compatibility proxy
from the legacy scalar value. Once a verified solved example has trained the new
heads, later rounds use those factorized predictions. Pass
`--require-factorized` to reject legacy snapshots instead.

Run the three default deep-ladder scientists with:

```bash
.venv/bin/pgx-mcts-bench braid-adaptive-scientists \
  --output artifacts/adaptive-scientists/run-001 \
  --rounds 200 \
  --pool-size 200
```

Use repeated `--scientist NAME=CHECKPOINT` options to select another group.
`pool.json` freezes the population, `schedule.jsonl` records every prediction,
proposal, fairness state, selection and shared witness, and `report.json`
summarizes the run. Per-scientist checkpoints are written after every round.
