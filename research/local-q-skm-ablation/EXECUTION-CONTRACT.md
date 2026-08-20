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
optimizer, replay, and rehearsal exposure.  Overwrite it at the first safe
completed cell or iteration boundary after ten minutes and unconditionally at
every phase transition.  On timeout or process recovery, restore the latest
checkpoint and report its phase plus completed cell and iteration counts.
Completed retention cells remain measured; missing cells remain hard-timeout
failures in the denominator.  Never serialize model state from an asynchronous
signal handler.  At most the completed work since the preceding ten-minute
checkpoint plus the currently incomplete cell or rehearsal iteration may be
discarded.
