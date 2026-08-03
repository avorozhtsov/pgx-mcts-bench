# Resumable heterogeneous scientist collaboration

The legacy `braid-adaptive-scientists` command remains a regression fixture. It
shares raw replay records and therefore supports only scientists with the same
action space. The corrected experiment is
`braid-collaborative-scientists`.

## Frozen experiment

Each run writes:

- `manifest.json`: checkpoint paths and hashes, arm, budgets, ratios, fixed bank
  seed, and protocol hash;
- `base.json`: 200 knot-identity-distinct representations spanning four quartiles
  of a fixed cheap difficulty proxy;
- `new-70.json`: 70 identity-disjoint held-out representations;
- `rounds/NNNNNN/{event.json,state.pt.gz}`: one atomic, immutable, compressed
  round transaction (the reader also accepts the uncompressed pilot-v1 format);
- `schedule.jsonl`: a rebuildable projection of committed events; and
- `report.json`: progress, translation coverage, and compute accounting.

Changing any protocol field or checkpoint hash makes `--resume` fail. Replay
buffers, optimizer state, predictions, active tasks, and the reveal cursor are
checkpointed after every round.

Both `10:1` and `1000:1` objectives are attempted for every selected
representation. Adaptive and static arms spend the same number of qualification
and full-search attempts. Qualification games select a task but never enter
training replay, avoiding an unpaired adaptive-data advantage.

## Heterogeneous witness translation

A solved native trajectory is replayed exactly and reduced to global semantic
braid edits. For each receiver, every edit is routed through its own head and
memory actions. The translated receiver-native trajectory is replayed again and
accepted only if it reaches the unknot inside the receiver's move budget. It
enters replay only when its receiver-native objective strictly improves that
receiver's own full attempt, or rescues a failure. Its crossing changes and
receiver-native moves are stored as a one-sided upper-bound training record.
Translation success and replay admission are logged separately.

The 2026-08-02 heterogeneous smoke completed 13 rounds. Six selected tasks had a
verified winner and all 22 attempted sender/receiver translations succeeded. A
separate process-pool smoke committed two rounds with 8/8 translations, followed
by a successful resume test.

## Pilot arms

The first causal pilot runs the same three checkpoints in:

1. `adaptive-sharing`;
2. `adaptive-no-sharing`; and
3. `static-sharing`.

This isolates sharing at fixed adaptive scheduling and scheduling at fixed
sharing. The primary final score is capped portfolio loss on `new-70.json`; solve
rate and conditional costs are always reported beside it.

Run one local paired seed with:

```bash
scripts/run_collaboration_pilot.sh local artifacts/collaboration-pilot-200-local
```

If that low-search engineering run cannot maintain a productive frontier, run
the 75-round CPU-budget gate before renting a VM:

```bash
scripts/run_collaboration_pilot.sh local-high artifacts/collaboration-pilot-75-local-high
```

Run three paired CPU-32 seeds with a 70-hour workload limit with:

```bash
scripts/run_cpu32_budgeted.sh artifacts/collaboration-pilot-200-cpu32
```

That command defaults to the preregistered K=3 roster. Run the admitted K=4
sensitivity separately, under a different artifact root, with
`ROSTER=k4 CYCLIC_CHECKPOINT=...`; do not inject it into a partially completed K=3
run.

After all three arms complete, evaluate their final portfolios and their common
initial portfolio on the frozen 70-item anchor with:

```bash
scripts/evaluate_collaboration_pilot.sh ARTIFACT_ROOT SEED SIMULATIONS
```

The four resumable evaluations run concurrently and produce `comparison.json`.

## Current gate result

The 2026-08-02 low-search 200-item seed and the 75-round CPU-budget local gate did
not justify a CPU-32 launch. The latter found 24 adaptive-sharing, 26
adaptive-no-sharing, and 27 static-sharing winner rounds. On the frozen 70-item
anchor at 16 evaluation simulations, adaptive sharing did not beat both controls.

Strict receiver-improvement admission was then added. In a paired 75-round rerun,
102/102 translations replayed, 90 entered replay, and winner rounds rose to 27.
Held-out ratio-10 coverage improved from 5/70 initially to 7/70, but capped loss
17,316 remained worse than static sharing (17,248) and adaptive-no-sharing
(17,281). Ratio-1000 capped loss also remained worse than the initial portfolio.
Keep CPU-32 blocked until a local strict value/cost-only sharing arm beats both
controls; valid translated witnesses need not be good cross-architecture policy
targets.

The value/cost-only arm was subsequently implemented as
`adaptive-sharing-aux-only`. Shared records do not contribute policy or scalar
value equality targets. Its paired 75-round result improved ratio-10 capped loss
to 17,259, versus 17,281 for adaptive-no-sharing, but did not beat static sharing
at 17,248. Ratio-1000 performance regressed to 5/70 solved and capped loss
1,331,233. CPU-32 remains blocked. The next fair gate is a ratio-10-only rerun of
all controls; ratio 1000 should be separate until ratio-conditioned adapters or
stratified replay pass a multi-task interference test.

Set `SHUTDOWN_WHEN_DONE=1` only on a disposable rented VM where passwordless
`sudo shutdown` is configured. Without it, the workload stops but the VM must
still be stopped in the cloud console to stop compute billing.

At the 2026-08-02 listed Nebius rates, 32 vCPU plus 128 GiB RAM costs about
USD 0.7936/hour. Seventy hours is USD 55.55; a 200-GiB network SSD for that period
adds about USD 1.36, for USD 56.91 before tax. Disk billing continues until the
volume is deleted.

## Cyclic-memory capacity branch

`s-cyclic-tape8-192` combines an imported `s-window-128` controller with a
five-scale cyclic full-word residual encoder and an aligned eight-symbol tape.
The value representation is invariant to cyclic rotation by construction. New
policy/value/auxiliary paths are zero-initialized, and tape-write actions start
with negligible probability, so mapped parent actions and parent value outputs
are preserved at initialization.

The model has 306,214 parameters. On the frozen ten-representation challenge at
32 simulations, the imported model matched parent coverage. After a 40-task local
run it doubled challenge coverage from 1/10 to 2/10 at both objectives, but failed
the identity-disjoint gate: held-out ratio-10 coverage fell from 3/20 to 2/20 and
ratio-1000 capped loss worsened.

The accepted checkpoint adds self-supervised equivalence pretraining. It
uses multiple exact isotopy/Markov views of 400 identities while reserving the
pilot 200, `NEW_70`, and 50 calibration identities. Equivalent-view retrieval on
the 50 unseen identities improved from 60% to 92%. After the paired 40-task run,
coverage on the untouched 20-item local anchor increased from 3/20 to 4/20 at
both objectives. Capped loss fell from 4,653 to 4,462 for ratio 10 (4.1%) and
from 350,119 to 336,068 for ratio 1000 (4.0%). It therefore passes the local
admission gate, but enters only a K=4 sensitivity run; it does not alter the
preregistered K=3 comparison.
