# Long-word horizontal stacking

Status: deferred research direction. This note does not modify or authorize any
change to the active cohort-scoped training protocol. The current Fast and Slow
lineages should continue under `EXECUTION-CONTRACT.md` until their frozen
benchmark is complete or genuinely blocked.

## Motivation

The current registered scientists have word capacity 48. There are two related
but distinct goals:

1. make an existing checkpoint technically accept words longer than 48;
2. make the resulting controller reason effectively about distant parts of a
   long braid word.

`simple-raster-axial` is easiest to lift in the first sense. It sees a local
seven-column window, and its learned parameter shapes do not depend on the full
word length. The same checkpoint can therefore be applied while the environment
uses a larger word capacity. This alone does not give the controller global
context, however, and navigation becomes increasingly expensive as the word
grows.

`strand-graph-12` is the better foundation in the second sense. Before each
decision, its adapter performs a deterministic full-word scan and compiles the
previous and next crossings along both physical strands. The network then passes
messages along the cyclic word and these physical-strand edges, retaining a
feature at each possible edit site. Its learned message-passing tensors are not
indexed by absolute word position.

## Proposed hybrid

Use a pretrained `simple-raster-axial` checkpoint as a shared local expert and a
lightweight strand-graph network as the global router.

For every possible head position, construct an overlapping seven-column local
window. Evaluate all windows as one batch with the same raster checkpoint; do
not create independent parameter copies. Retain the local embedding, action
logits, and site value for each position.

Pass those per-position embeddings through a sparse global router whose edges
are:

- adjacency along the cyclic Artin word;
- previous and next crossings along each physical strand;
- optionally a small pooled global summary.

The router chooses a promising position or navigation target. The local raster
expert chooses the semantic edit at that position. Recompute windows and graph
edges after every semantic edit because braid moves can alter block boundaries
and physical-strand incidence.

Conceptually:

```text
long braid word
  -> batched overlapping local windows
  -> shared pretrained raster expert
  -> per-position embeddings and local policies
  -> strand-graph router
  -> target position plus local semantic action
```

This is horizontal stacking by shared application over positions, not by
concatenating independently trained networks over disjoint chunks. Disjoint
chunking is unsuitable because the word is cyclic, physical strands cross chunk
boundaries, and a local rewrite immediately changes neighbouring chunks.

## Function-preserving migration

The safest child should initially reproduce a parent controller exactly:

- import and initially freeze the complete raster expert;
- initialize the router's contribution to policy and value to zero;
- preserve the existing legal-action mapping and objective conditioning;
- train only the router and calibration layers first;
- unfreeze only the last raster blocks later, at a substantially smaller
  learning rate;
- retain short-word rehearsal to prevent catastrophic forgetting.

The existing `strand-graph-local` design is a useful implementation precedent:
it combines a local tower with a global graph tower through initially suppressed
residual contributions. A raster-local variant could follow the same principle.

## Navigation at greater length

The current fixed head strides are `1, 2, 4, 8, 16`, chosen for capacity 48.
Merely enlarging the input capacity leaves the action count unchanged but makes
distant navigation require more search plies. For substantially longer words,
compare two protocol-compatible choices:

1. function-preservingly extend the policy head with strides `32`, `64`, and
   later powers of two, initializing new route logits conservatively;
2. introduce bounded navigation options that compose existing shifts while
   preserving exact replay and accounting.

Navigation cost must be reported separately from semantic progress. Otherwise a
model may appear unable to simplify long words when it is actually spending its
search horizon reaching the relevant site.

## Suggested training curriculum

1. Freeze the raster expert and train only the router on lengths at or below 48.
2. Increase maximum length progressively: 48, 64, 96, then 128.
3. Mix short and long examples throughout training.
4. Unfreeze the final raster blocks only after the router learns useful site
   selection, using a learning rate roughly 5--10 times smaller than the router's.
5. Consider full end-to-end fine-tuning only after demonstrating transfer on a
   representation-disjoint holdout.

Multiple existing scientists may also be retained as heterogeneous local
experts, with a small learned gate calibrating their logits. That can improve
robustness, but an ensemble without a position-aware global router does not by
itself solve the long-horizon problem.

## First controlled experiment

Run only after the current frozen benchmark permits a new scientific objective.
Use common identities, exact representations, seeds, search accounting, and
evaluation budgets for three arms:

1. unchanged `simple-raster-axial` at increased capacities;
2. unchanged `strand-graph-12` at increased capacities;
3. frozen raster expert plus zero-initialized trainable strand-graph router.

Evaluate at lengths 48, 64, and 96 before attempting 128 or larger. Report:

- solved count under both L10 and L1000;
- crossing changes and total moves for solved cases;
- navigation plies before the first useful semantic edit;
- semantic edits per solved trajectory;
- inference time and MCTS evaluations;
- retention on the original short-word exact-common set.

This comparison separates three possible bottlenecks: local perceptual quality,
global site selection, and navigation/search horizon. Promotion to larger words
should require both improved long-word performance and no material regression on
the frozen short-word benchmark.
