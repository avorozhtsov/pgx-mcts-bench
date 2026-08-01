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

| where | what |
|---|---|
| local, 4 slots | **leaders**, open-ended on unlabelled rungs: `u1-puct`, `wide-net`, `search-heavy`, `s-head-256`, `s-reg4` |
| local, 4 slots | **climbers**, `--stop-after 16`: the other eleven, climbing to the top of the calibration set then exiting |
| server, 3 containers | `s-burau-oracle`, `s-head-1stride`, `s-reg8` |

Queue scripts live in the session scratchpad (`queue-lead.py`, `queue-climb.py`,
`jobs-*.jsonl`). One job per candidate, process-group isolation so SIGTERM kills
the tree, start rate-limited to one per 20s — a gate on `getloadavg` cannot see a
job launched five seconds ago and will happily burst-start eight onto a full
machine.

The split exists because **slowest-first only works when jobs terminate.** The
leaders are on rungs that can only end at the cap, so they held all eight slots
for nine hours and eight arms never started. Bounded targets for the climbers fix
that.

### The server

`locuscanvas.com` / `89.169.108.199`, user `artemvorozhtsov`, key `~/.ssh/id_ed25519`,
passwordless sudo. Full details in `pgx-mcts-bench/artifacts/oracle/locuscanvas_log.md`.

**It runs the user's production stack.** `locuscanvas-postgres` and
`locuscanvas-persona-backend` have been in restart loops since the machine last
booted — a permissions failure on `/var/run/postgresql`, unrelated to the training
containers and untouched. Training is capped at `--cpus=1.2` with
`OMP_NUM_THREADS=1` each so it cannot starve the web services.

## Open, in rough order of value

1. **A second seed.** Every number in every table is one seed. `--stop-after 16`
   makes a clean replication cheap: 15 arms over the calibration set only. The
   jobs file (`jobs-seed1.jsonl`) already exists and was never run.
2. **Add `--bounds` to the running jobs** so the ratchet accumulates instead of
   needing to be re-seeded by hand.
3. **Store the unknotting sequence as the witness.** `bounds.py` currently records
   the knot's defining word, so a bound can be attributed but not re-verified.
   Until then these are trusted claims, not checkable ones.
4. **Run `braid-ladder-rescore`.** Recorded `cc` is measured once, at promotion,
   with the weights of that moment. The rescore re-measures with current weights
   and has already shown drift larger than the gaps between adjacent rows —
   `s-gru128` moved 2.10 → 3.33 on one rung. It was killed by a restart and never
   completed.
5. **`s-burau-oracle` has not cleared rung 9 in nine hours.** Meanwhile
   `s-head-1stride` — plain window, worst stride set, no accumulator — went from
   rung 1 to 14 on the same box. If the oracle caps, that is evidence against the
   whole whole-tape-accumulator direction, and it applies with more force to
   `s-fsa32`, `s-gru128` and `s-ff4-p5`, which are *learning* what it is handed.
6. **Certified lower bounds** (`|σ|/2`, `|s|/2`, `|τ|`) with branch-and-bound —
   what turns an upper bound into `u(K) = n`. Unaffected by the zero-knowledge
   constraint: bounds verify output, they are not features. **Partly done:**
   `rf_knots.invariants` computes `|σ|/2` per knot and names the knot against a
   table of 2870, which is where the exact `u` on 19 rungs came from. What is
   missing is `|s|/2`, `|τ|`, and any of it being wired into the search.
7. **Batch the MCTS leaf evaluations.** 7.8× measured on this laptop, and the
   prerequisite for a GPU ever being worth renting.

## Process notes

Moved to [rf-knots/docs/lessons.md](../rf-knots/docs/lessons.md), along with the
ones this repository's tooling has since added.
