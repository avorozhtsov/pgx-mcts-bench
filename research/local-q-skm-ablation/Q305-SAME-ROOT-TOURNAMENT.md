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
