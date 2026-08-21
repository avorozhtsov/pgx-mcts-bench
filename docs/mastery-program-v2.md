# Multi-knot mastery v2

V2 is an additive continuation protocol. A v1 scientist is stopped only long
enough to copy one durable `program-state.json` / `scientist-state.pt.gz` pair;
the v1 artifact remains immutable and the copy is migrated with
`scripts/migrate_mastery_program_v2.py`.

The operational solve label is `P(this solver succeeds | representation,
crossing-change budget, L1000 budget, simulation dose)`. Raw head probabilities
are retained. A separate beta-binomial table calibrates them for each simulation
dose, and paired same-representation/same-seed probes may promote the dose only
when the higher dose improves success or L1000 without reducing successes per
CPU-second.

Heap scheduling uses calibrated probability plus the declared upper-bound
bonus, uncertainty exploration, bounded age fairness, and a measured compute
penalty. The fair refresh cursor still visits every live item eventually. Every
subtask retains its original knot, target, parent, and complete recursive
crossing-change lineage.

Search outcomes are mutually exclusive:

- `strict_challenge_success`
- `relaxed_training_success`
- `supported_search_failure`
- `hard_timeout`
- `unsupported_capacity`
- `invalid_witness`

Only replay-verified witnesses can change an upper bound. Relaxed successes may
train the network but are reported separately. A negative enters replay only
after the configured independent-seed confirmation count.

Every v2 state pins the SHA-256 of one `best-solutions-pool.json` export. The
snapshot can improve a known incumbent at admission without exposing the run to
a changing database. A later snapshot is adopted only by an explicit new
migration or group-boundary protocol update.

The first genuine improvement produces a certificate containing the replayed
upper-bound witness and all available independent computed/tabulated lower-bound
claims. The status is `exact-certified` only when a lower bound reaches the new
upper bound; otherwise it remains `upper-bound-certified`.

Retention is measured at each inherited witness's historical target, recorded
by strand count and source. Rehearsal remains adaptive from 5% to 50%. Falling
below 0.80 at maximum rehearsal is a capacity alert, not permission to silently
change width or replace the frozen panel.

## Registered exhaustion stops

An operational futility stop must remain external to the immutable scientist
protocol. `scripts/guard_mastery_stop_gate.py` waits for an exact durable group
checkpoint, verifies the checkpoint manifest and both state hashes, writes an
atomic provenance record, and only then stops the named systemd unit. Any
durable strict success or upper-bound improvement cancels the stop. The guard
does not edit program state, heaps, training data, curricula, simulation dose,
or checkpoint contents.

Use an isolated low-weight service for a live guard and retain its stop record
with the experiment artifacts. A run that has already ended can be registered
with the same verified-checkpoint record without restarting it.
