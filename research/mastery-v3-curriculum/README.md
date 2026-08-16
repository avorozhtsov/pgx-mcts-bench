# Controlled mastery-v3 curriculum

This directory registers the agreed three-track program and its isolated v3 implementation.
Migration, equivalence pretraining, proof-aware distillation, and paired 20-knot screening are authorized; full-240 remains
mechanically forbidden until the registered audit returns exactly one winner. No live v2, Q4000,
DKT72, or B* experiment is mutated.

`protocol-spec.json` pins the two architecture candidates, GPU execution contract, lineage split,
and fail-closed promotion gates. The generated curriculum is shared by both candidates. Training
and screening knot identities are disjoint, all DKT72 identities and exact representations are
excluded, evidence is frozen, and screening uses the same 20 representations and two seeds.

Q4000 has no native 8-strand row. Registration therefore derives 8-strand alternatives by one
deterministic Markov stabilization of holdout-filtered 7-strand rows. These remain representations
of their parent knot identities; every derived row records the parent representation, seed, sign,
and output representation hash.

Build the immutable registration with:

```bash
PYTHONPATH=src python scripts/register_mastery_v3_curriculum.py build \
  --q-root /path/to/pinned/q4000-v1 \
  --dkt-panel /path/to/dkt2026-table1-authors-pd-braids-v1.json \
  --evidence-snapshot /path/to/best-solutions-pool-20260815T2000Z.json \
  --output research/mastery-v3-curriculum/curriculum.json
```

Promotion reports are audited with the `audit` subcommand. It verifies exact paired screening
keys, migration tolerance, overall and strand-stratified retention, zero capacity exceptions,
strict evidence, evidence rate, and measured end-to-end GPU speedup. It emits either no winner or
exactly one winner; a relaxed training success alone never advances an arm.

## Implemented execution path

`pgx_mcts_bench.mastery_v3` contains both opt-in network encoders. Each child nests the exact
`cyclic-memory-12` controller; all new policy, value, solve, and conditional-cost routes are zero
initialized. The safe arm is a 128-channel, ten-block cyclic tower with the pinned
`1,2,4,8,16` dilation cycle and zero LayerScale. The ambitious arm additionally has bounded
physical-strand message passing, combined-invariant conditioning, a shared 11-row-pair scorer,
and separate invalid/capacity diagnostics.

Prepare either checkpoint fork locally with:

```bash
PYTHONPATH=src python scripts/prepare_mastery_v3.py \
  --source-checkpoint /path/to/exact-high-cyclic-memory-v2.pt \
  --candidate cyclic-memory-deep-v3 \
  --output /new/isolated/path/cyclic-memory-deep-v3.pt
```

The command accepts both plain ladder checkpoints and durable `.pt.gz` SKM scientist states. It
refuses to overwrite an output, hashes the checkpoint and controlling sources,
checks policy/value/factorized-head equivalence on registered real-state probes, removes the
parent optimizer, and records `launched: false`. Repeat with `cyclic-graph-dual-v3` for the paired
arm. A report above `1e-6` fails before an artifact is published.

Run equivalence pretraining independently for both forks:

```bash
PYTHONPATH=src python scripts/pretrain_mastery_v3.py \
  --checkpoint /isolated/cyclic-memory-deep-v3.pt \
  --curriculum research/mastery-v3-curriculum/curriculum.json \
  --candidate cyclic-memory-deep-v3 \
  --output /isolated/cyclic-memory-deep-v3-pretrained.pt
```

Then run proof-aware distillation from the same frozen dataset for both arms:

```bash
PYTHONPATH=src python scripts/distill_mastery_v3.py \
  --checkpoint /isolated/cyclic-memory-deep-v3-pretrained.pt \
  --curriculum research/mastery-v3-curriculum/curriculum.json \
  --evidence-snapshot /path/to/best-solutions-pool-20260815T2000Z.json \
  --candidate cyclic-memory-deep-v3 \
  --output /isolated/cyclic-memory-deep-v3-distilled.pt
```

The feasibility head receives negative labels only below `ratio * certified_lower_bound`.
Positive budget and policy labels require a replay-verified witness from the exact starting braid
and use its full `ratio * crossing_changes + moves` cost. The uncertified interval is masked.
Separate ordinal and lower/upper heads learn the mathematical bounds; the operational `p_solve`
head remains untouched. The v3-only budget feature combines the existing fixed-cap linear channel
with a bounded `log1p` normalization so small L10 and L1000 budgets retain usable scale.

`pgx_mcts_bench.gpu_inference.PersistentInferenceCoordinator` is the long-lived accelerator
worker for CPU MCTS actors. It uses FIFO dynamic batches, registered length/strand/dose buckets,
BF16 autocast on CUDA, per-request deadlines, and an MCTS-compatible blocking proxy. Its measured
end-to-end speedup still has to clear the preregistered `5x` gate on the chosen GPU. The screening
runner durably journals every registered item, runs the frozen parent control at historical upper
bounds for retention, measures an end-to-end CPU/GPU probe, and writes a fail-closed audit:

```bash
PYTHONPATH=src python scripts/run_mastery_v3_screening.py \
  --curriculum research/mastery-v3-curriculum/curriculum.json \
  --source-checkpoint /isolated/parent-scientist-state.pt.gz \
  --deep-checkpoint /isolated/cyclic-memory-deep-v3-distilled.pt \
  --graph-checkpoint /isolated/cyclic-graph-dual-v3-distilled.pt \
  --output /isolated/screening
```

This runner never contains a full-240 launch path.
