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

## Distillation degradation gate

`braid-distillation-degradation-train` forks a saved collaboration transaction
without mutating the run. It creates matched no-update, native-RL, one-witness,
and ten-witness checkpoints. Full-policy and auxiliary-only witness treatments
use the same native minibatch prefix as the native-RL control, replace a fixed
number of batch slots, and sample shared trajectories episode-uniformly so long
translations are not overweighted.

The round-48 gate used three training seeds, eight optimizer steps, batch size 32,
and three shared slots per treatment batch. `braid-distillation-degradation-evaluate`
then evaluated the first 50 BASE representations and all NEW70 representations at
16 simulations. `braid-distillation-degradation-analyze` requires every seed to
avoid both solve loss and capped-loss regression on both splits.

No treatment passed. One auxiliary-only witness improved NEW70 capped loss by
11, 11, and 16 relative to matched native RL, with no portfolio coverage change,
but its BASE50 deltas were +157, -65, and +175 and it lost a retained solve in two
seeds. Ten auxiliary-only witnesses lost `12n_684` on NEW70 in two seeds. Native
RL itself was worse than the untouched checkpoint in every seed: BASE50 capped
loss increased by 66, 399, and 203, while NEW70 increased by 22, 55, and 22.

Therefore pause both current RL and current distillation. The next admissible gate
is a rollback-guarded update: one auxiliary-only witness, one shared slot per
batch, success-balanced native rehearsal, four steps at learning rate 0.00025,
and per-scientist retention checks on a frozen BASE canary. Keep the pre-update
checkpoint unless the candidate is non-inferior. NEW70 remains untouched until
the end and cannot be used to choose or roll back updates.

### Corrected 128-simulation gate

The follow-up fixed three protocol defects before rerunning anything: the budget
channel now uses an absolute global scale (rather than making every fresh cap
look like `1.0`), objective-cap exhaustion is censored out of policy/value
training, and collaboration replay is episode-uniform with balanced native
success rehearsal and a hard shared fraction. A source audit identified
`T(3,5)` as BASE item `10_124` and `T(3,4)` as NEW70 item `8_19`. The corrected
70-item endpoint removes `8_19` and replaces it with same-quartile `12n_683`.
One rung source, `R(5,12)#0`, is not identifiable by the bundled table, so the
claim is disjoint from every *identified* ladder source, not absolute proof of
identity disjointness.

At 128 simulations the old frozen portfolios gave:

| portfolio | BASE200 solved / capped loss | corrected NEW70 solved / capped loss |
|---|---:|---:|
| initial | 29 / 46,765 | 8 / 17,063 |
| static sharing final | 27 / 47,251 | 7 / 17,094 |
| static no-sharing final | 27 / 47,694 | 7 / 17,116 |

On NEW70 both final arms solved the exact same seven items; initial additionally
solved `11a_14`. On BASE, sharing and no-sharing intersected on 25 items, with
sharing-only `11n_27`, `11n_46` and no-sharing-only `11n_76`, `11n_9`.
BASE is a rehearsal/exposure diagnostic because it contains `10_124`; NEW70 is
the transfer endpoint. The deeper search therefore confirms, rather than
reverses, the old-training regression.

A corrected 50-round static-sharing run then supplied a reduced fork at round
48: `pre`, matched `RL0`, and one auxiliary-only witness, three minibatch seeds,
four optimizer steps, one shared slot of 32, and learning rate 0.00025. At 128
simulations D1-aux versus RL0 had identical portfolio coverage in every pair,
but capped-loss deltas were:

| split | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| BASE50 | +2 | +13 | -2 |
| corrected NEW70 | -11 | +22 | +11 |

No treatment passes. Native RL itself is search-budget dependent: versus `pre`,
it gained the same two BASE50 items in all seeds and reduced capped loss by
378--382, while NEW70 deltas were +22, -86 (one added solve), and +33. This is
not stable transfer and NEW70 cannot be used to select a seed.

The rung solve rates are not contradictory. One ladder iteration supplies eight
self-play episodes from one source family and 96 optimizer steps; a candidate
usually receives roughly ten or more iterations before promotion, then solve
rate pools 16 evaluation episodes per ratio. The portfolio endpoint instead
gives a frozen network one search attempt per heterogeneous representation and
no task-local update. The next local experiment should compare a disposable
task-local fork with `F=5` current-representation iterations and `F_old=1`
iteration on one distinct old BASE representation. Compare it against the
`F=5, F_old=0` ablation and frozen deeper MCTS matched by measured network
evaluations and wall time. Start with ten BASE development tasks after removing
identified ladder identities and `s-window-128`; expand to 20 tasks and the
three-scientist portfolio only if
`5+1` beats both controls. Discard each fork after its task. Keep CPU-32 blocked
until rapid adaptation beats deeper search and a rollback rule can be chosen
from BASE alone.

An attempted 200-item expansion on 2026-08-03 is invalid. It accidentally used
`s-window-128/stage22-after.pt`, an unpromoted snapshot with embedded solve rate
0.0 after 100 capped iterations, instead of the last promoted
`stage21-after.pt`. The rapid runner now rejects unpromoted checkpoints and
reruns the checkpoint's own promoted-rung held-out evaluation before creating
an output manifest. The corrected stage-21 snapshot reproduced 12/12 solves at
ratio 10 with 128 simulations. This gate verifies checkpoint capability only:
BASE200 is a heterogeneous table-knot transfer bank, not the generated rung
distribution, so a separate small paired BASE gate is still required before a
full run.

That paired gate used an outcome-blind, identified-source-disjoint BASE20 bank
with five 3-strand, five 4-strand, and ten 5-strand presentations. Across three
paired seeds the frozen `5+1` search solved 7/20, while both trained `5+0` and
trained `5+1` solved 6/20. Every trained solve occurred in the first target
iteration, before an optimizer update; there were zero post-training rescues.
The frozen arm alone found `11n_9` on a later iteration in every seed. Relative
to frozen search, trained `5+1` changed capped L10 by +215, +206, and +235. Its
rehearsal-retention counts were 1/20, 2/20, and 0/20 versus frozen 3/20, 2/20,
and 0/20. The gate rejects rapid adaptation and therefore blocks the adaptive
200 run.

The original remaining-objective-budget path fails its independent admission gate.
Migrating the promoted checkpoint correctly appends channel values from 0.023
to 1.0, but `p(solve)` is exactly constant across seven caps on all 20 tasks.
The associated factorized-head training artifact has 1,077 eligible positions,
all 1,077 positive and no negative solve labels. On BASE20, mean predicted
probability is 0.612 against observed coverage 0.35 (Brier 0.330, log loss
1.349). Do not use predicted-loss caps with that checkpoint.

The repaired `s-window-128` prototype now treats cap exhaustion as a negative
label only for the conditional solve event, while continuing to mask it from
policy, scalar-value, and conditional-cost losses. Solve loss reaches the shared
body and encoder; cost losses remain detached from them. The critic predicts
`cc` and `moves`, constructs `L=A*cc+B*moves` exactly, then conditions a residual
solve branch on shared features, remaining budget, `cc`, `moves`, and `L`.
Zero-initialized budget skips feed the shared body, scalar value, cost heads, and
solve branch, so migration preserves policy and every value output exactly.
Multi-cap replay retains failed restart attempts, balances cap strata, and adds
a paired monotonic loss on lower/higher budgets for the same state.

The first easy-five local pilot revealed that ordinary BatchNorm adaptation could
destroy the promoted policy. Freezing its running statistics and using learning
rate 0.00025 fixed that regression under the same seed: all five simple knots
became monotone and budget-sensitive, while the six-game promoted-rung check
remained 6/6 and improved conditional cost from 5.00/18.00 to 4.67/17.83. This is
a smoke result on the training knots, not admission evidence. Predicted caps stay
disabled until a source-disjoint held-out calibration gate passes.

That gate is now passed on a fresh decision split, with an important protocol
correction recorded separately. The first ten held-out knots improved Brier from
0.430 to 0.133 and AUC from 0.827 to 0.886, retained all 23 baseline successes,
added one, and retained 12/12 on the promoted rung. Its preregistered requirement
that 8/10 *all* knots be budget-sensitive nevertheless failed because six knots
were never solved even at the global cap. That run remains a formal failure and
was not reused for admission.

Before opening the next split, the gate was corrected to require sensitivity on
every empirically budget-informative knot and low probability on knots never
solved at any cap. On the untouched next ten knots (200 attempts), all ten curves
were monotone, all 7/7 informative knots were sensitive, and all 3/3 never-solved
knots had maximum `p(solve) <= 0.1`. Brier improved from 0.722 to 0.232, AUC from
0.672 to 0.824, and attempt coverage from 33/200 to 36/200. The paired sets were
30 shared, 6 trained-only, and 3 baseline-only attempts. Promoted-rung retention
remained 12/12; conditional cost improved from 4.75 crossings/19.83 moves to
3.58/16.83. This admits only a bounded search-savings ablation. It does not yet
authorize predicted caps in collaboration or the 200/2,700 experiments.

The bounded search-savings ablation tested two rules on further untouched
slices. Naive `L_max=2*L_predicted` with geometric restart preserved the exact
20/80 solved attempts and improved their aggregate L10 from 992 to 862, but used
604,032 versus 133,056 scheduled evaluations and 445 versus 96 wall seconds:
246 restarts made it 4.54 times more expensive. That rule is rejected.

The diagnostic split showed that restarts were spent almost entirely on tasks
whose global-budget `p(solve)` was already very low. Before opening another
20-knot slice, the rule was frozen as follows: when global `p(solve)<0.04`, make
one `2*L_predicted` probe and accept failure; otherwise run the global budget
directly. On 80 fresh paired attempts it retained the exact 11-attempt solved
set and identical aggregate L10=573 while reducing scheduled evaluations from
149,985 to 101,178 (32.5%) and wall time from 108.7 to 72.5 seconds. Sixty
attempts used the probe and twenty used the global budget. This passes the
bounded savings gate. The rule may be included as an optional arm once the
separate learning/scheduling gates unblock the 200 pilot; it does not by itself
unblock collaboration or the 2,700 run.

### K=3 architecture repair and fast-learning gate, 2026-08-04

The remaining-budget repair now applies to the preregistered K=3 roster:
`s-window-128`, `d-tape4-u1`, and `s-w11-128`. All three use cost-first
`cc`/moves prediction, exact `L=A*cc+B*moves`, a residual solve branch conditioned
on shared features and remaining budget, solve gradients into the shared encoder,
censored cap failures, and paired monotonic training. Narrow fine-tuning freezes
BatchNorm statistics and adds a frozen-teacher policy/value trust region plus
promoted-rung rehearsal. The tape student uses a slower controller rate and a
faster auxiliary rate because a single rate either erased its controller or left
its historical solve head saturated.

The launcher had two stale checkpoint defaults. `s-window-128` stage 22 and
`s-w11-128` stage 19 are both unpromoted capped snapshots. Defaults now use the
last promoted stages, 21 and 18 respectively, and launcher preflight rejects any
unpromoted or wrong-scientist checkpoint. Predicted objective budgets are opt-in,
not the launcher default.

Each scientist trained over 65 difficulty-ordered identities with two games at
five caps and only eight optimizer updates per identity (520 updates total), plus
12 promoted-rung rehearsal games. Every training curve was monotone; solve
sensitivity appeared on all 23 informative identities for window and tape and all
20 for wide-window. All three retained their promoted rung internally.

The final decision used fresh identities 75--84, 200 paired attempts per model,
and a separate 12-game, 128-simulation promoted-rung check:

| scientist | Brier baseline -> trained | AUC baseline -> trained | paired solves baseline -> trained | rung SR baseline -> trained | cap decision |
|---|---:|---:|---:|---:|---|
| `s-window-128` | 0.506 -> 0.043 | 0.826 -> 0.995 | 16 -> 15 | 1.000 -> 1.000 | reject: lost `10_123@704#1` |
| `d-tape4-u1` | 0.353 -> 0.0002 | 0.826 -> 1.000 | 16 -> 16, exact set | 1.000 -> 1.000 | admit |
| `s-w11-128` | 0.446 -> 0.049 | 0.805 -> 0.980 | 10 -> 11 | 0.917 -> 0.917 | reject: two never-solved identities remained overconfident |

Thus all three architectures can learn the repaired critic quickly without native-
rung collapse under the guarded curriculum, but only `d-tape4-u1` is admitted for
predicted-budget search. Window and wide-window remain eligible for controlled
full-budget arms starting from their promoted source checkpoints; their trained
critics may be observed in shadow mode, but may not control caps or early failure.
This is a fast **critic-learning** result, not evidence that all three solver
policies improve quickly: paired coverage was -1, 0, and +1 respectively. It does
not unblock the persistent-RL or sharing gates for the 200-representation
experiment.

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
