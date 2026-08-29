# Proof-context adapter: conservative first stage

The proof graph supplies replayed upper-bound paths, not unique optimal action
labels. The first proof-guidance stage therefore keeps the admitted base network
frozen and forbids ordinary next-action behavioural cloning.

## Three-way action contract

Every legal action considered at a root has one of three statuses:

- `accepted`: bounded search completed and independently replayed a continuation
  on the same observed objective frontier;
- `compared_nonfrontier`: the same bounded protocol completed and replayed a
  continuation with a worse observed result;
- `unknown`: the search did not establish a comparable continuation.

Unknown does not mean bad. It is excluded from the policy loss and has exactly
zero gradient. For the primary unknotting-number objective, every completed
action reaching the same minimum CC count is accepted even when it reaches a
different graph node, uses a different ordering of zero-CC moves, or has a
different semantic length. L10/L1000 move-sensitive tie-breaking is a separate,
explicit ratio-conditioned experiment.

The conservative set loss is

```text
logsumexp(logits[accepted union compared_nonfrontier])
  - logsumexp(logits[accepted])
```

It moves probability mass from adjudicated worse outcomes to the acceptable
set without selecting a canonical witness action. A row with no replayed worse
comparison abstains.

## Required producer protocol

The producer must evaluate alternatives through complete suffixes, not by
comparing a raw immediate successor to a stored edge:

1. apply the proposed primitive action exactly;
2. run the pinned preprocessing and normalization contract where applicable;
3. exploit an exact proof-graph hit as a verified suffix, or continue the same
   fixed-budget MCTS protocol;
4. independently replay the complete solution and recompute `(CC, moves)`;
5. group all equal-frontier actions into one accepted set;
6. leave timeouts, misses, and non-replayed proposals unknown.

In particular, a zero-CC action from a stored program is not a unique target.
Independent/commuting reorderings become accepted when their complete replayed
suffix reaches the same frontier.

## Gradient boundary

Initially graph structural losses update only the proof encoder, and the set
policy loss updates only a zero-initialized adapter and applicability gate. The
base network remains frozen. Promotion requires a paired base-only versus
graph-assisted MCTS gate on held-out knot identities, with zero replay failures
and no protected-corpus regression.

`pgx_mcts_bench.proof_guidance` implements the set loss and frontier grouping.
Its `adapter_only_set_objective` obtains detached base logits with the adapter
bypassed and exposes a differentiable path only through the adapter and gate,
even if the surrounding caller has not frozen the base parameter flags.
It does not reinterpret an observed upper-bound route as a lower-bound proof or
claim that an observed nonfrontier action can never improve at a larger budget.
