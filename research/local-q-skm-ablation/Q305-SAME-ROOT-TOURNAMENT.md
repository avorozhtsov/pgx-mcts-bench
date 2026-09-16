# Q305 same-root trajectory tournament

Status: `PREPARED` as a training primitive; the Q305 bank and launch gate remain
separate prerequisites.

For every registered representation and objective ratio, generate exactly ten
independent equal-budget trajectories from the same root. Every trajectory must
replay exactly before it can affect the relative update. Missing or invalid
routes are ignored, never labelled negative.

The split is deliberately interpretable. Solved trajectories outrank unsolved
ones. Among solved routes, lower crossing-change count is primary and semantic
move count is secondary. Among unsolved routes, smaller best residual word
length is only a progress comparison, not evidence of impossibility. The split
uses the largest robust adjacent gap: one crossing change, at least four or ten
percent of the median semantic moves, or two residual letters. If no robust gap
exists but there is a unique best route, use best versus the other nine at
quarter weight. If the best is tied, skip the relative policy update.

For a positive group of size `P` and negative group of size `N`, each positive
trajectory receives advantage `+c/P` and each negative receives `-c/N`, where
`c` is the boundary confidence. Thus each root contributes zero total relative
mass. Sampling remains equal per episode; trajectory length does not multiply
its influence. The stable contrastive loss imitates the chosen action on a
positive trajectory and minimizes the chosen-action probability on a negative
trajectory. It never minimizes a signed log-probability, which would be
unbounded below. Negative trajectories are relative losers for this root and
budget only, not globally bad or unsolvable examples.

Promotion remains an equal-budget exact-common MCTS comparison against the
Q304 parent. Training loss alone is not evidence of progress.

## Fourth descendant: trimmed divergence-local credit

After the original and stable-loss pilots regressed, the next isolated fork
keeps ten equal-budget trajectories but uses only the best and worst thirds.
The ambiguous middle is ignored.  Mixed solved/unsolved outcomes retain that
categorical boundary; homogeneous outcomes require a robust crossing-change,
semantic-move, or residual-length margin.

Relative policy credit is assigned only at replay-identical states where the
selected positive and negative groups chose different group-exclusive actions.
Shared prefix actions and states reached only after divergence receive no
relative label.  A quarter of each optimizer batch is reserved for these sparse
relative targets, and their loss weight is 0.5.  The fork starts from the same
frozen Q304 parent with its own seed and must pass a Q305--Q314 exact-common
promotion gate before any Q315--Q354 continuation.
