# Handoff — read this first

Operational state of the training runs. **Dated 2026-08-01**; the "Running now"
section below has a half-life of days, so check it against `ps` before trusting it.

This file used to live in `rf-knots`, next to the mathematics rather than next to
the runs it describes. What stayed behind there:

| where | what |
|---|---|
| [rf-knots/README.md](../rf-knots/README.md) | what the project is, and its scope |
| [rf-knots/docs/rungs.md](../rf-knots/docs/rungs.md) | every ladder rung, its rationale, and which knot it actually is |
| [rf-knots/docs/lessons.md](../rf-knots/docs/lessons.md) | process notes — the things that cost time to learn |
| [rf-knots/research/13-directions.md](../rf-knots/research/13-directions.md) | the big next moves, unscheduled |

Both repos are public; `pgx-mcts-bench` depends on `rf-knots` by path
(`../rf-knots`), so they must sit side by side.

## The reference is now a ratchet, not a theorem

`artifacts/bounds.jsonl` — append-only claim log, `bounds.py` folds it on read.
A knot's `u` is **the fewest crossing changes anyone has ever used**, with the
holder recorded; it moves the moment someone beats it. Seeded from 263 claims
across every checkpoint, giving 23 knots a standing record. `artifacts/bounds.md`
is the rendered table.

Two things it already showed:

* **On eighteen labelled knots the record equals the theorem exactly.** The
  ratchet independently reproduced the Milnor conjecture.
* **`P(3,20)#0` sits at 11 against a theorem of 9.** Not a contradiction — an
  upper bound can be loose — but it means nobody found the optimal sequence there.
  That gap is invisible on unlabelled knots, which is the calibration set earning
  its keep.

**The ratchet can now be checked on the challenge half too.** Most of those knots
turn out to have a published or derivable `u` ([`docs/rungs.md`](docs/rungs.md)),
so a record can be compared against truth rather than only against other agents.
The first comparison is not flattering: the standing record on `R(3,18)#0` is 6
against `u = 2`.

Runs claim automatically via `--bounds artifacts/bounds.jsonl`. **The currently
running jobs were started without it**, so they are not claiming; add it on the
next restart.

## The finding worth chasing

**The ranking on structured knots and unstructured knots disagree.**

| arm | labelled rungs | unlabelled rungs |
|---|---|---|
| `search-heavy` | exactly optimal at every `T(3,4)`/`T(3,5)` rung | 4.00 on `R(3,14)#0` |
| `u1-puct` | +1.42, +3.92, +6.00 — worst in the field | **3.00** on `R(3,14)#0`, holds 3 of 6 records |

`u1-puct` is the worst arm where `u = g` and a greedy positive-braid strategy
works, and the best where it does not. Visible at rungs 23 and 27. If it survives
a second seed it says the labelled ladder was rewarding a heuristic that does not
generalise — which is the whole reason the challenge set exists.

**Nothing here has more than one seed.** That is the single biggest weakness of
every table in this project.

## Running now

There is **no active promotion queue**. Do not infer one from old queue files or
from the presence of ladder worker processes.

| where | what |
|---|---|
| local | Drain only. `s-paint2`, `s-fsa32`, `s-gru128`, and `s-head-128` are finishing the checkpoint that was already in progress. Their queue controller is gone, and the monitor stops each process group as soon as its target checkpoint appears. |
| local | Three tape/scan workers are finishing rung 0 for `s-tape2`, `s-tape4`, and `s-scan-gru-tape2`. Their scheduler is paused so it cannot dispatch the fourth candidate or another rung; the pool is stopped after all three checkpoint files appear. |
| local | `u1-puct` reached `stage26-after.pt` and has already stopped. `wide-net`, `search-heavy`, `s-head-256`, `s-reg4`, and `s-ff4-p5` are deliberately not running. |
| Nebius | One isolated preemptible L40S VM, `braid-gpu-gate-20260801`, is running `braid-device-benchmark`. It compares CPU and CUDA at actor batches 8/32/64 for `u1-puct`, `s-w11-128`, `s-tape4`, and `s-scan-gru`. This is a throughput/cost gate, **not ladder training**. |

The Nebius benchmark has a five-minute read-only heartbeat. It reports phase
changes, process failure, preemption, result completion, and spend thresholds.
It must not start or restart work. The authorized cap for the disposable gate VM
is $20; retrieve the reports and delete the VM when the gate finishes.

The production LocusCanvas host is no longer a benchmark execution target. Do
not schedule knot training there and do not carry its historical `--cpus=1.2`
cap into Nebius sizing. The old host log remains an artifact of the earlier run,
not current operational guidance.

The reproducible, account-neutral Nebius lifecycle is in
[`docs/nebius-device-gate.md`](docs/nebius-device-gate.md): dedicated
least-privilege service account, registry image, preemptible VM, device gate,
artifact retrieval, and teardown. It intentionally contains no project IDs,
addresses, SSH keys, or tokens.

### What happens next

1. Finish the device gate and copy its JSON and Markdown reports locally.
2. Choose CPU, GPU, or a hybrid configuration from equal-work time and cost.
3. Delete the benchmark VM and its temporary IAM resources.
4. Select the top nine conceptually different candidates plus five interesting
   candidates (`s-tape4`, not `s-tape2`, is the preferred tape representative).
5. Transfer their completed checkpoints and explicitly start promotion to rung
   18. Promotion has **not** started yet.

## Open, in rough order of value

1. **Finish the Nebius device gate and choose the rental shape.** Compare
   equal-work iteration time and cost, not forward-pass latency. The small models
   may leave an L40S under-filled; CPU or a hybrid split remains a live outcome.
2. **Promote the selected portfolio to rung 18.** Start only after the device
   decision, checkpoint transfer, and an explicit launch. The intended portfolio
   is nine top but conceptually different candidates plus five interesting ones.
3. **Store the unknotting sequence as the witness.** `bounds.py` currently records
   the knot's defining word, so a bound can be attributed but not re-verified.
   Until then these are trusted claims, not checkable ones.
4. **Add `--bounds` to the promotion jobs** so the ratchet accumulates instead of
   needing to be re-seeded by hand.
5. **Run `braid-ladder-rescore`.** Recorded `cc` is measured once, at promotion,
   with the weights of that moment. The rescore re-measures with current weights
   and has already shown drift larger than the gaps between adjacent rows —
   `s-gru128` moved 2.10 → 3.33 on one rung. It was killed by a restart and never
   completed.
6. **Run a second seed after portfolio selection.** Every current leaderboard
   number is one seed. Replicate the selected set rather than spending on every
   discarded arm; the old `jobs-seed1.jsonl` is input material, not an active
   queue.
7. **Certified lower bounds** (`|σ|/2`, `|s|/2`, `|τ|`) with branch-and-bound —
   what turns an upper bound into `u(K) = n`. Unaffected by the zero-knowledge
   constraint: bounds verify output, they are not features. **Partly done:**
   `rf_knots.invariants` computes `|σ|/2` per knot and names the knot against a
   table of 2870, which is where the exact `u` on 19 rungs came from. What is
   missing is `|s|/2`, `|τ|`, and any of it being wired into the search.

## Process notes

Moved to [rf-knots/docs/lessons.md](../rf-knots/docs/lessons.md), along with the
ones this repository's tooling has since added.
