# Experiment execution contract

This project treats agreement to run an experiment as authorization for both implementation and execution. A completed launcher is not the experimental outcome.

## Status vocabulary

- `PLANNED`: scope and protocol were discussed, but no runnable invocation exists.
- `PREPARED`: code, manifests, and launch commands validate, but no process is running.
- `QUEUED`: an active launcher is durably waiting for a registered prerequisite.
- `LAUNCHED`: the operating system accepted the experiment process.
- `VERIFIED ACTIVE`: the intended PID and child-worker count are alive, CPU time advances across multiple samples, and the expected manifest or fresh artifact exists.
- `COMPLETED`: the registered terminal report and durable states exist.
- `BLOCKED`: execution cannot safely continue; the exact blocker is reported immediately.

Only `VERIFIED ACTIVE` may be reported to the user as “running.”

## Default launch policy

After the user agrees to run, continue, or train:

1. Implement and validate the invocation.
2. Launch it without asking for a second confirmation.
3. Within the same turn, verify PID, expected child-worker count, advancing CPU over repeated samples, artifact root, manifest, disk headroom, and absence of immediate errors.
4. Attach or update a monitor that knows the expected workers and the next stage.
5. At every stage transition, verify that the next registered workers actually start. A waiting or exited launcher is not progress.

Do not auto-launch only when the implementation is not working, required inputs are absent, an immutable protocol would be violated, or the action adds material unapproved cost, external impact, or destructive risk. Report that exception immediately.

## Response handoff

End every substantive status or completion message with the recommended next step and ask whether to proceed. This question must not delay already-authorized execution: finish and verify the current scope first, then ask about the next useful scope beyond it. Prefer one evidence-backed recommendation over an unranked list.

## Q60 application

The current aligned Q population run is `VERIFIED ACTIVE` only while the
registered one-thread population children are alive, aggregate experiment CPU
stays within 4–6 cores, artifacts remain fresh, and no duplicate branch writes
to the same root.  The monitor must alert if a promised wave has no active child
within five minutes, and may safely resume the exact registered invocation only
after proving that no duplicate exists.

Compute dose is a run-level protocol, never an architecture or scientist
property.  Each lineage enters the aligned protocol exactly once with
`F_native=4` and `simulations=40`; after that, its adaptive controller state is
carried across every curriculum and stage boundary without reset.  Adaptive
levels are exactly `F_native={4,6,8,12,16}` and
`simulations={40,64,80,128,256}`.  Every iteration uses four self-play games
and 24 optimizer steps, and every objective uses two final evaluation attempts.
Qualification uses one attempt at the current run-level simulation dose.  All
lineages receive the same controller policy.  Realized adaptive doses may
diverge only after the common deficient-block rule fires, and reports must show
the resulting network-evaluation and optimizer-step cost.

An aligned boundary may be placed independently at each lineage's latest
atomic durable representation.  Preserve its network, optimizer, replay,
rehearsal, and adaptive-controller state.  Discard and repeat any uncommitted
representation.  Pre-boundary results remain historical capability evidence
but must not be mixed into a post-boundary compute-efficiency window.

All registered Q continuations use `scheduled-no-sharing`.  Bank-file row order
is the curriculum order.  A preserved-history continuation removes already
durable identities and keeps the remaining bank rows in their original order;
it never replays a completed identity merely to repair historical ordering.

A hard rehearsal timeout is a censored evaluation outcome, not evidence that
retention is deficient and not permission to discard completed optimizer work.
It must therefore hold `F_old` at its current value; only a complete trailing
retention evaluation may update that controller.  Every timeout-enabled
lineage owns exactly one atomic rehearsal checkpoint.  Its cursor covers
`retention_before`, `train`, and `retention_after`, including completed
retention cells, selected order, completed rehearsal iterations, network,
optimizer, replay, and rehearsal exposure.  Schema-v4 also persists an
in-progress iteration after each completed self-play game and after safe
optimizer-step boundaries, including the frozen game plan and separate
self-play/optimizer wall-time counters.  Overwrite it at the first safe
completed cell, game, optimizer step, or iteration boundary after ten minutes
and unconditionally at every phase transition.  On timeout or process recovery, restore the latest
checkpoint and report its phase plus completed cell and iteration counts.
Completed retention cells remain measured; missing cells remain hard-timeout
failures in the denominator.  Never serialize model state from an asynchronous
signal handler.  At most the completed work since the preceding ten-minute
checkpoint plus the currently incomplete retention cell, self-play game, or
optimizer step may be discarded.

Q154 remains strict `scheduled-no-sharing` in exact bank row order.  No scalar
budget, network weight, replay record, witness, trajectory, outcome, or
controller datum crosses lineages.  Every rehearsal iteration keeps four
self-play games and 24 optimizer steps: two games use L10 and two use L1000.
All four games use only that lineage's best native incumbent cap for the exact
representation and ratio.  A historical carry that predates exact
representation indexing may use only the same lineage's knot-identity
incumbent as an explicitly reported compatibility fallback.  Missing local
caps fall back to the global game budget and are reported as deficits.

Q154 block rehearsal uses a deterministic expanding round-robin panel of at
most 20 representations in exact historical bank order.  The durable absolute
cursor is part of state and checkpoint data.  Only that panel receives the
`retention_before -> train -> retention_after` transaction, for at most
`20 x 2 ratios x 2 phases = 80` retention cells per block.  Replay optimizer
batches target four equal strata: L10 positive, L10 negative/censored, L1000
positive, and L1000 negative/censored.  If a stratum is absent, use the
declared deterministic fallback order and record the exact deficit; do not
retry until a desired outcome appears.

The six fast Q/SKM lineages reached the original priority/exposure task order
through the exact-common Q134 boundary (30 Q154 rows).  Their cohort marker
binds all six 30-event prefixes and state hashes.  After the verified
fast-6/slow-4 split gate, those six may continue through Q135-Q154 without
waiting for the four slow lineages.  Each slow lineage independently uses the
original order through its own slow-4 exact-common Q134 barrier.  After each
cohort's verified Q134 transition gate, panel membership and the absolute cursor remain unchanged,
but task order is mixed reproducibly: `retention_before` and
`retention_after` use the same lineage/round/cursor-seeded permutation, while
training tasks are interleaved across complete L10/L1000 outcome signatures
inside each least-exposed tier.  The seed, retention order, selected training
order, present outcome signatures, and fallback deficits are persisted in the
schema-v3 checkpoint and completed rehearsal event.  A transition before the
applicable Q134 marker, a different order on resume, or a branch crossing Q134
under the new policy before its cohort barrier is `BLOCKED`.

Before Q154 native curriculum begins, each lineage repays all Q104 rehearsal
debt, defined as the sum over censored Q104 rehearsal blocks of
`max(0, F_old - completed_rehearsal_iterations)`.  Repair is written as a
separate `q104-rehearsal-repair-v1` carry and uses consecutive round-robin
panels in chunks of at most eight iterations.  It must not replay a native Q104
identity, commit a Q104 event, advance curriculum, or rewrite the historical
Q104 result.  Every repair chunk is atomic and resumable.

For the slow-4 Q154 lineages, the two-hour rehearsal deadline is a resumable
compute segment.  An expired segment kills only that scientist child and
resumes the same panel transaction from its schema-v4 checkpoint without
committing an event or advancing curriculum.  The cumulative cap is computed
from the bounded panel size, ratio count, current simulation dose, and current
rehearsal iteration dose, then rounded up to a whole segment.  The slow-cohort
training estimate is 7200 seconds per iteration at 80 simulations; this changes
only the cumulative deadline, never `F_old`, self-play games, optimizer steps,
or adaptive simulation dose.  Only cumulative cap exhaustion is a censored
hard timeout and holds `F_old`; slow native tasks use the same two-hour hard
deadline.  After durable Q154 completion, run exactly
one full-history, after-only retention audit over all 154 representations and
both ratios.  It performs no training and cannot update `F_old`.

## Fast-6 and slow-4 cohorts

The Q104-to-Q154 transition is cohort-scoped.  The historical
`PRIMARY_8_LINEAGES_Q104_COMPLETE.json` remains an immutable binding of the
eight completed Q/SKM Q104 reports, but Q154 execution is split by measured
runtime after the exact-common Q134 boundary.  The authoritative executable
gate is `FAST6_SLOW4_COHORT_SPLIT_V5_VERIFIED.json`; its compatibility rule
normalizes only protocol-neutral legacy default spellings and records resume
provenance.  Any compute-dose, order, bank, or sharing difference remains
`BLOCKED`.  V5 also isolates recovery status writes before gate verification;
an operational recovery must never overwrite the original launcher status.
Removing slow lineages from fast dispatch must not renumber random
seeds: every fast lineage retains its original primary-8 seed index.  V1--V4
gates are immutable historical records and do not authorize a new dispatch;
already-active V4 branch transactions remain durable and continue unchanged.

Fast-6 contains `q-grown-raster-axial-12`, `q-grown-strand-graph-12`,
`q-grown-cyclic-memory-12`, `skm-v2-high-cyclic-memory`,
`skm-v1-simple-raster-axial`, and `skm-v1-simple-strand-graph`.  They run as six
concurrent one-thread branches.  Their terminal marker is
`ALL_FAST_6_LINEAGES_Q154_COMPLETE`.
If the original post-Q134 launcher has already bound one active writer while
other branches rejected a pre-V4 seed mismatch, the registered recovery script
`scripts/run_local_q154_fast6_transition_recovery.py` may own only the five
writer-free roots.  It must prove the original writer chain and absence of a
writer in every recovery root before launch; it never restarts or duplicates
the active branch.

Slow-4 contains the two combined-dual lineages plus `cyclic-memory-deep-v3`
and `cyclic-graph-dual-v3`.  Exactly one one-thread slow worker runs at a time.
The combined-dual branches resume imported, hash-bound Q104 debt-repair carries;
the V3 branches first resume their deferred 33/44 and 24/44 Q104 boundaries
through `scripts/run_local_q104_v3_backfill.py`.  After
`SLOW_4_LINEAGES_Q104_READY.json`, the four continue in a separate Q154 root and
share no state.  Their terminal marker is `ALL_SLOW_4_LINEAGES_Q154_COMPLETE`.

The two launchers may overlap: six fast workers plus one slow worker, for an
explicit maximum of seven one-thread experiment workers.  Results may be
compared only on exact-common prefixes within a cohort; slow-4 joins a combined
leaderboard only when an equal prefix exists.  No partial prefix may be ranked
against a longer prefix, and no result may be ranked by iteration count alone.

## Focused successor after Q254/Q154

The historical six-lineage Q304 program is superseded. Its prepared bank and
audits remain immutable evidence, but its preparation, cohort launcher, and
branch launcher are not authorized to execute. They must fail closed against
`focused-successor-v1-policy.json`; the absence or modification of that policy
is not permission to fall back to the old program.

Finish the active Fast Q254 and Slow Q154 cohorts under their existing frozen
hash gates. The focused successor remains `QUEUED` until the Fast terminal
marker exists. Slow Q154 must also reach its terminal audit before the diverse
Slow parent is selected, but this does not prevent the four already specified
Fast-parent descendants from being prepared after Fast completion. Do not
automatically continue the former Slow-4 Q204 or Fast-6 Q304 cohorts.

The focused successor has exactly five registered lines: three descendants of
the terminal `q-grown-strand-graph-12` state, one terminal
`q-grown-raster-axial-12` continuation, and one diverse Slow parent selected
from the exact-common terminal audit. The three strand-graph children are an
RL-only seed control, a replay-validated set-valued NodeDB proof-distilled
child, and a proof-distilled child with whole-word embedding conditioning.
Every fork binds the same parent state hash and receives a distinct registered
random seed. Sharing is permitted only for this initial parent copy; after the
fork, network, optimizer, replay, controller, exposure, trajectory, and outcome
state are isolated.

NodeDB supervision treats replayed currently-best actions as a set of accepted
targets, worse completed routes only as comparisons, and absent graph actions
as unknown rather than negative. Every training row and every generated
witness must replay exactly. The embedding child remains `QUEUED` until an
external terminal selection manifest binds the checkpoint and source hashes.
Its initial bridge is zero-initialized; parent and embedding weights remain
frozen while only bridge and calibration parameters train.

Promotion uses a frozen exact-common set and reports L10 and L1000 solved
counts, replayed crossing changes, capped iteration means, complementary solved
identities, neural evaluations, optimizer steps, wall time, and historical
retention. Training loss, unequal prefixes, and unreplayed proposals are not
promotion evidence. A replayed path establishes an upper bound only; equality
with the unknotting number requires an independent lower bound.
