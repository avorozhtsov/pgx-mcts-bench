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

Every receiver action is a charged move.  The portable `UnknotWitness` omits
states that change only the serial head, tape, register, or colour memory; that
compaction is correct for proof exchange but not for the experimental objective.
The runner therefore verifies the compact witness while computing `moves` from
the complete receiver-native record.

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

## Corrected sharing implementation (2026-08-06)

An implementation audit found four defects in the path from the bounded-option
gate to the long runner.

First, verified cost used the portable witness length and therefore made serial
controller operations free.  Verification still rejected invalid solutions, so
historical solved identities remain meaningful, but historical receiver move
objectives, capped losses, and strict-improvement admissions from this path must
be recomputed before comparison with the repaired protocol.

Second, the native optimizer was created before the option adapter was attached.
Native updates used the adapter-augmented policy, failed to clear its gradients,
and clipped the base update using parameters the optimizer did not own.  Native
training now bypasses a separately optimized option controller, clears all
network gradients, and clips only optimizer-owned parameters.  Option training
also clears the entire network before updating only adapter and gate parameters.

Third, the long runner attached the residual adapter but not the conservative
state gate tested by the bounded gate.  Sharing arms now attach both modules,
start the gate at probability 0.1, train route applicability, preserve native
off-route policy by KL, and penalize off-route gate activation.  The adapter and
gate include the head-cell feature explicitly rather than relying only on
mean/max sequence pools.  Historical global-pool adapter checkpoints migrate
with identical outputs by zero-initializing the new head feature.

At each training event the corrected runner gives sharing
`ceil(train_steps / 4)` adapter/gate updates; no-sharing controls receive the
same number of extra native optimizer steps.  The manifest records this dose and
the isolation/retention losses, and its schema is now
`collaborative-scientists-v3`, preventing accidental resume from the old
protocol.

Fourth, policy distillation previously checked a donation only against the
receiver's latest stochastic attempt. A failed current attempt could therefore
admit a witness worse than a solution the receiver had found earlier. The long
runner then computed a filtered option set but failed to pass it to the trainer,
which silently resampled stale donations from the full replay. Replay now keeps
a persistent best-native objective for each `(representation, ratio)`. A shared
trajectory remains an active policy target only while its fully charged
objective is strictly lower than that incumbent; equality is rejected, only the
best active donation is selected, and eligibility is rechecked at every
training event. Stale records remain available only for their safe one-sided
critic upper bound. Frozen evaluation results seed the archive even when worker
processes do not retain their trajectories.

This repair does not make the current option target-conditioned.  It still
teacher-forces a deterministic shortest neutral head route followed by the
certified edit.  It does not yet learn arbitrary tape/register programs or
select among several donated solutions by an explicit target embedding.  That
is a separate research redesign, not part of this correctness patch.

The corrected unit, migration, gradient-isolation, stale-witness, and
transactional-resume tests pass.

The corrected v6 multi-witness gate has now run on the frozen 17-identity
`s-tape4-h5` panel for seeds 20260950--20260952.  Sharing solved 13, 17, and 15
identities versus 6, 6, and 4 for the compute-matched native control.  Mean
target transfer was 91.7% versus 4.2%, and sharing solved the held-out
receiver-unsolved `12a_850` in every seed while control never solved it.  Charged
capped L10 also favoured sharing in every seed: 2,200 versus 3,407; 1,634 versus
3,364; and 1,998 versus 3,757.

The gate still failed exact retention in two seeds.  Sharing lost frozen
canaries `11n_46` and `12a_1199` in seeds 20260950 and 20260952; `12a_1199` was
also control-only in seed 20260950.  Bypassing the final adapter did not recover
either canary, so the loss resides in the trained base scientist rather than in
direct adapter activation at evaluation.  Long sharing and CPU-32 remain closed
until native base updates inside the sharing learner use explicit canary
retention or transactional rollback and pass this same three-seed gate.

That last decision was the v6 rule. In v9 exact frozen-network retention is a
reported secondary criterion rather than a hard blocker: stochastic learned
policies need not preserve every old fixed-seed solve to be useful. The primary
per-receiver gate is now (1) charged aggregate capped loss no worse than the
compute-matched native control and (2) every active distillation event reaching
its registered canonical-route loss reduction before the hard optimizer-step
cap. Temporary route-loss increases inside an event are permitted because the
optimizer also protects off-route native behaviour. The overall gate additionally
requires at least one paired sharing-only solved identity. Reports still include
lost frozen solves, exact solved-set intersections and differences, objective
quality on common successes, and compute accounting.

### Expanded preflight and simulation dose

The expanded preflight used 17 representations: the eight transfer targets, a
held-out receiver-unsolved witness, five canaries, and three deliberately stale
donations. Native refresh correctly made `10_126` and `11n_119` stale as well.
All five stale donations performed zero adapter updates, so the strict
better-than-native filter passes its functional gate.

The learning/search budget materially changes the result. With 128 simulations
during learning, sharing and control each solved 11/17, but sharing lost capped
L10 3,280 to 2,512. With 64 simulations during learning, sharing solved 10/17
versus 4/17 and narrowly won capped L10 3,656 to 3,671. Re-evaluating the two
fixed checkpoint pairs at matched 32, 64, 128, and 256 simulation doses gave:

| learning simulations | evaluation simulations | sharing / control solved | capped L10 sharing / control |
|---:|---:|---:|---:|
| 64 | 32 | 12 / 4 | 3,377 / 3,794 |
| 64 | 64 | 10 / 4 | 3,656 / 3,671 |
| 64 | 128 | 14 / 6 | 3,263 / 3,412 |
| 64 | 256 | 17 / 7 | 2,767 / 3,261 |
| 128 | 32 | 9 / 10 | 3,762 / 2,771 |
| 128 | 64 | 10 / 10 | 3,452 / 2,716 |
| 128 | 128 | 11 / 11 | 3,280 / 2,512 |
| 128 | 256 | 11 / 13 | 3,178 / 2,164 |

More evaluation search therefore strengthens coverage for the checkpoint learned
at 64 simulations, but it does not rescue the checkpoint learned at 128. At
every evaluation dose, the 64-trained sharing checkpoint has worse total
objective on the representations both arms solve. Its capped-loss advantage
comes from additional coverage, not shorter common solutions. Training search
and final evaluation search are consequently separate manifest fields; the next
gate used 64 for learning and 128 for evaluation.

### v9 split-budget admission result

Seed 20260950 was rerun from the untouched rung-18 checkpoint under the v9
event-level rule, with 64 learning simulations, 128 evaluation simulations,
eight final attempts, sixteen cycles, and four evaluation workers. Sharing
solved 14/17 versus control 4/17 and won capped L10 2,392 to 3,721. The four
common successes were `10_100`, `10_124`, `10_152`, and `12a_1203`; their summed
objectives were 465 for sharing versus 289 for control. Sharing added ten
identities and control added none. It lost frozen canary `12a_1199`.

Strict stale filtering worked: the donated/native objectives were 68/66 for
`10_126` and 162/110, later 162/85, for `11n_119`, and none of those stale
events updated the adapter. Eleven of twelve active distillation events reached
the registered 10% route-loss reduction. The first `11a_15` event reduced route
loss only from 6.9719 to 6.8420 before the 16-step cap, rather than reaching
6.2747. The seed therefore fails the preregistered primary gate despite the
strong aggregate result.

Seeds 20260951 and 20260952 were not run after that failure. The 30--50-item
pilot, capacity-expansion branch, 200-item arms, and paid CPU-32 run remain
closed. The next bounded question is whether a compute-matched learning-rate and
step-cap search can make *every* active route event reach its target without
erasing coverage; the aggregate seed must not be used to waive that test.

### v10 block-balanced admission result

The v9 requirement that every individual witness reach a fixed route-loss
reduction was superseded before the next confirmatory run. Route loss is a
teacher-forced imitation diagnostic, not an external solver objective. Protocol
v10 waits for at least ten active strictly superior witnesses, samples an equal
number of canonical-route positions from each, applies one fixed 16-step adapter
block every ten cycles, and matches the control by state examples as well as
optimizer work. Per-witness route loss no longer vetoes a block.

The confirmatory panel contained 25 representations: 19 registered training
targets and six non-target retention canaries. Their start states were used by
the off-route preservation loss, so they are not a truly unseen held-out set.
Training used 64 simulations; paired final evaluation used 128 simulations and
eight attempts per representation. All three fresh seeds completed one real
sharing block.

| seed | sharing / control solved | capped `L10` sharing / control | `sharing - control` |
|---:|---:|---:|---:|
| 20261000 | 16 / 21 | 3,775 / 3,067 | +708 |
| 20261001 | 12 / 17 | 4,480 / 3,700 | +780 |
| 20261002 | 16 / 16 | 3,535 / 3,805 | -270 |

Every active witness improved its canonical-route loss in every seed, with mean
block reductions of 0.90%, 1.07%, and 0.84%. External performance nevertheless
favoured control in two seeds. Mean complete-panel delta was +406 and median
delta was +708. Training-target deltas were +277, +762, and -82; non-target
canary deltas were +431, +18, and -188. Thus the negative result is not explained
by one outlier route or by the canary subset alone. A generalization claim still
requires an identity-disjoint panel unused by donation, replay, preservation
losses, and model selection.

The sharing-only union was `10_152`, `10_159`, `11a_231`, `12a_1199`, and
`12a_1255`, so the adapter did transfer some distinct behaviours. Those gains
did not compensate for lost and control-only solves. The v10 multi-seed gate
fails, and long sharing or paid compute remains closed. The next experiment
should test selective route applicability or native/share gradient conflict,
while preserving the same paired external endpoint and compute accounting.

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

### Common structural objective-budget protocol, 2026-08-07

The optional solve-gated rule above remains a historical search-savings result,
but it is no longer the collaboration protocol. A local serial scientist sees
only its current window at the initial state, so neither `2*L_predicted` nor any
other fixed multiplier of that prediction is a defensible scientific cap. It
also gives different scientists different solving opportunities.

Schema `collaborative-scientists-v5-common-structural-budget` therefore never
uses a scientist prediction as an attempt cap. For observed braid-word length
`c`, objective ratio `A/B`, and common native action horizon `H`, every scientist
receives the same first tier

```text
min((A/B + 1) * H, (A/B) * ceil(c / 2) + H).
```

This is an economical structural probe, not a claim of solvability within that
tier. Every objective-censored failure is repeated with the same seed at the
common global cap `(A/B + 1) * H`. Both records enter replay at their encoded
budgets; only the final attempt decides task success. An ordinary action-horizon
failure is not repeated because the objective cap did not censor it. The old
10% predicted-cap audit option has been removed from the collaboration command
and launcher.

The first v5 equivalence audit used the independent K=3 rung checkpoints,
`L1000`, 100 distinct representations from the mastered scrambled prefix of the
ladder, two paired attempts per representation, and a separate 20-representation
simulation calibration split. The lowest registered dose, 32 simulations, gave
20/20 portfolio calibration coverage. On the audit panel, direct-global and
structural-first/global-restart both solved 92/100 representations with identical
capped `L1000 = 639,569`. All 600 paired final outcomes and all solved costs
matched; 598/600 native action sequences matched exactly. The two differing
sequences were failures with the same registered outcome.

Equivalence therefore passes, but economy does not: 36 restarts increased
scheduled network evaluations from 578,424 to 617,133, or 6.69%. Keep the
structural protocol opt-in and disabled in the main five-arm comparison. Its
artifact is
`artifacts/structural-budget-equivalence-k3-100-seed20261600-20260807/report.json`.

### Transactional positive-acquisition gate v3, 2026-08-07

The next native-learning gate moved the final objective to `L1000` and expanded
search tiers from simulations alone to `(simulations, native action horizon)`:
`(64,64)`, `(128,96)`, `(256,128)`, and `(512,128)`. Higher tiers require
residual progress, and the final two require a registered near-solve. The
starting `s-window-128` checkpoint still passed its promoted-rung regression at
12/12 under `L1000`.

The fixed 12-representation development panel started at 4/12 solved in every
seed. Transactional rollback retained every initial solved identity and improved
capped `L1000` from 179,560 to 149,459, 162,509, and 147,457. Final coverage was
only 6/12, 5/12, and 6/12, however. `12a_981` was the only declared discovery
identity rescued in at least two seeds; the gate required two identities and at
least 70% final panel coverage in every seed.

The failure separates search from learning. Search solved `10_149` in all 4/4
native attempts in every seed, but each 24-step consolidation candidate was
rolled back because it lost `11a_26` or worsened capped `L1000`. Conversely, the
hard frontier produced no certified positive: `11n_107` remained at residual
length 11 at both `(64,64)` and `(128,96)`, just outside the preregistered
near-solve threshold 10, while `10_71` and `10_137` made no residual progress.
No task qualified for the 128-action tiers.

Thus horizon escalation does not open the K=4 smoke. The next bounded repair
belongs in success consolidation—dose selection inside a transaction or an
explicit canary-preserving update—not in more search on `10_149`. The artifact
is `artifacts/native-learning-gate-swindow-v3-horizon-L1000-20260807/report.json`.

### Historical distilled-tape K=3 critic gate, 2026-08-04

This historical predicted-budget ablation used the temporary K=3 roster
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
unpromoted or wrong-scientist checkpoint. Objective-budget search remains opt-in,
not the launcher default, and now uses the common structural protocol above.

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
rung collapse under the guarded curriculum. The historical gate admitted only
`d-tape4-u1` for the now-superseded predicted-budget ablation. Under v5, no
scientist's critic controls caps or early failure; all three may be evaluated
under the same structural-first/global-restart protocol.
This is a fast **critic-learning** result, not evidence that all three solver
policies improve quickly: paired coverage was -1, 0, and +1 respectively. It does
not unblock the persistent-RL or sharing gates for the 200-representation
experiment.

The current collaboration roster instead uses `s-window-128`, independent
`s-tape4`, and `s-w11-128`. Its `s-tape4` stage-18 checkpoint was trained through
the native rung ladder and contains no distilled weights. `d-tape4-u1` remains a
separate reproducibility artifact whose `d` prefix and `-u1` suffix identify
distillation from the `u1-puct` teacher.

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

## Semantic-cost sharing v11

Sharing now uses a solver-independent cost contract. `final_moves` counts only
verified portable braid edits. Serial head shifts, tape/register operations, and
other controller-only actions are recorded as `final_native_plies` and
`final_internal_plies`; they consume receiver compute limits but do not enter
`L_A:B`. A translated record is rejected unless replay proves that its
`(crossing_changes, semantic_moves)` exactly matches the donor witness.

The v11 witness bank contains 25 certified solutions. The new remaining-semantic-
`L` input migrated the real rung-18 K=3 checkpoints with zero numerical output
difference and identical paired MCTS action sequences. A guarded `s-tape4`
curriculum on ladder rungs 0--9 made all 10 tested solve curves budget-sensitive
and monotone while promoted-rung solve rate changed from 7/8 to 8/8.

The 25-item local sharing preflight exposed and fixed two defects: an unroutable
witness was still being scheduled, and the fresh option adapter inherited the
native controller's `5e-5` rate. The scheduler now cycles only through translated
targets and reports unroutable identities; the isolated adapter uses `1e-3`.
This raised mean route-loss reduction under the same 16-step dose from 0.006% to
0.757% across 12 routable witnesses.

At 16 evaluation simulations, sharing and control both solved only `10_124`;
semantic capped `L10` was 6,387 versus 6,393. Neither solved a training target or
one of the six identities excluded from donation, replay, and preservation. A
post-hoc 64-simulation dose added sharing-only `10_100`, but a fresh
64-simulation/four-attempt preflight did not replicate the gain: both arms again
solved only `10_124`, tied at 6,387, and both lost the frozen `10_100` solve.

The v11 accounting and training plumbing pass, but useful sharing does not yet
pass a fresh preflight. Do not run the three-seed confirmation or long sharing
arms from these checkpoints. The next valid sharing test must preregister a
learning change such as repeated blocks or a target-conditioned option policy;
raising final search after observing a result is not sufficient.

## Continual-learning portfolio criterion

Exact per-representation retention is not the primary continual-learning
invariant. Training on a new representation may lose an old solve, and rehearsal
may lose a recent gain. The experiment therefore maintains two separate ledgers:

* the current network is evaluated on one fixed old-plus-seen portfolio; and
* the permanent solution bank keeps the best verified semantic solution ever
  found for every representation.

For `L10`, the failure penalty is one empirical cap frozen before learning: the
maximum verified `L10` on the registered calibration panel. Both paired arms use
the same cap and exactly the same task denominator. Every ten rounds, the current
network may retain the block when total solved count is nondecreasing and capped
portfolio `L10` is nonincreasing. At least one block must make a strict
improvement. A regressing block receives targeted recovery; if recovery still
fails, the network and optimizer return to the block-start state while the
permanent solution bank remains intact.

The preceding `L1000` engineering diagnostic demonstrated why the split matters:
the continual arm's permanent archive covered 20 representations while its final
network reproduced 18 of them. Its final new-panel solve rate was 13/20, below the
registered 70% floor, so it did not open a longer gate. The corrected `L10` smoke
uses 64 simulations, four attempts, `F_new=5`, `F_old=1`, and ten-round blocks;
its registered artifact is
`artifacts/portfolio-progress-smoke-swindow-seed20261720-20260807`.

The smoke completed in 70 minutes. Both treatment blocks improved their own
complete portfolios: block 1 kept 15/16 solves and reduced capped `L10` from 807
to 785; block 2 moved from 17/26 to 18/26 and from 1,579 to 1,537. Neither block
needed recovery. Replay was balanced at 45,713 positive and 46,080 negative
positions; 5.0% of treatment failures were budget-censored.

The longer gate nevertheless remains closed. Final treatment coverage was 12/20
on NEW, 6/6 on old rehearsal, 3/10 on held-out, and 0/4 on hard stress. It kept
the exact initial NEW solved set while improving NEW capped `L10` by 35. The
transactional diagnostic finished at 13/20 NEW and gained `11a_288`. On the
common 26-item current-network portfolio, treatment/control were 18/19 solves
and capped `L10` 1,537/1,523; their lifetime banks were 19/20 and 1,424/1,412.
Treatment did win held-out capped `L10` by 22, but both NEW solve rates missed the
70% floor. Do not start the 50-item or collaboration gates from this result.

## Joint-pretraining rewind audit

`braid-joint-pretrain` defines a budget-aware `s-window-128` with the historical
two residual blocks and width 32. Remaining semantic `L` is appended as a
function-preserving input; the H5 ablation also appends remaining internal
budget. Solve BCE may update the shared encoder, while cc/moves regression is
kept out of it. Cost heads still learn from shared features. Migration of the
real independent checkpoint changed every policy/value/cost output by exactly
zero.

The conservative arm reproduced rungs 0--9 at 100% in two iterations each, and
its internal budget gate kept 80/80 easy-prefix solves while making 20/20 train
and 20/20 held-out curves monotone. That apparent success failed the mandatory
source-disjoint gate. Restarting a rung-21 network at rung 0 reduced its original
promoted-rung solve rate from 12/12 to 2/12 before budget calibration and 3/12
afterward. The fixed 400-attempt `L10` panel fell from 16 solves to 3; `L1000`
fell from 14 to 6. The critic still had good rank AUC, demonstrating that critic
quality cannot substitute for solver retention.

Random initialization did learn the first five rungs, while H5 matched the warm
model's two-iteration progression. The channels and existing depth are therefore
not the immediate problem. The invalid operation is rewinding a promoted model
onto an easy-only mixture that omits the later mastered rungs. The command now
rejects this unless `--allow-warm-rewind-ablation` is explicit, and an internal
checkpoint is never labelled admitted before the untouched gate.

Use the already admitted independent calibrated checkpoint for the near-term
roster. A new from-scratch controller must climb the full ladder forward once,
with explicit old-rung rehearsal; do not widen or deepen it until that valid
curriculum reaches a measured capacity limit.

The independent checkpoint also passed the matching 400-attempt `L1000` gate:
18 solves versus 14 for the original rung-18 network, four calibrated-only
solves, no original-only solve, and exact 12/12 promoted-rung retention. Its
critic scored AUC 0.964, Brier 0.0323, Brier skill 0.248, and ECE 0.0273, with
20/20 monotone held-out budget curves.

`run_ladder` now supports `rehearsal_games_per_cleared_stage`. These episodes are
pinned to each cleared rung and are additional to the stochastic geometric
mixture, so `F_old=1` has an exact auditable meaning. The first valid scratch
pilot cleared rungs 0--4 but showed that this fixed dose is insufficient:
`unknot+6` fluctuated from 2/4 to 4/4 to 3/4 during later rungs. The final budget
block kept 34/40 solved attempts but worsened capped `L10 + L1000` from 205,540
to 208,571 and was rolled back. Future acceptance uses aggregate solved attempts
and capped objective; exact cell retention is secondary. Use targeted recovery
and block-start rollback before extending the scratch curriculum.
