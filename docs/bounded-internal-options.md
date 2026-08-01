# Bounded internal options

Serial agents may need to move their head, update registers or colours, or write
a tape before performing the braid edit selected by a parallel teacher. These
operations are treated as a bounded option rather than prescribed as a single
hand-written route.

For a teacher edit `X`, training maximizes the probability of every retained
path

```text
internal action -> ... -> internal action -> X
```

of at most five internal actions. The loss is the negative log of the summed
path probability. Beam membership uses detached scores, while action
probabilities remain differentiable. Student-preferred internal actions are
retained, together with a shortest-navigation branch that guarantees a useful
signal for a randomly initialized policy.

Only paths ending in `X` receive credit. A repeated toggle, shift loop, or tape
write is therefore not reinforced merely because the current network selected
it. Repeated controller states are pruned.

The horizon is part of the environment state and observation. After five
consecutive internal operations, all internal actions are illegal and the sixth
action must manipulate the braid. Any external braid action resets the counter.
This makes training, MCTS, and evaluation obey the same rule.

Distillation preserves the student's head, registers, colours, and tape across
successive teacher edits. The parallel teacher still supplies the braid
trajectory, so all students train against the same requested external moves.

Example:

```bash
.venv/bin/pgx-mcts-bench braid-distill-u1 \
  --teacher artifacts/deep-ladder/u1-puct/checkpoints/u1-puct/stage23-after.pt \
  --output artifacts/distill-bounded-options \
  --internal-horizon 5 \
  --option-beam-width 8 \
  --option-batch-size 4
```

The distillation report records the horizon, beam width, and final option loss
for every student. The existing shortest-route policy targets remain as a
stabilizing control signal; the bounded option loss is the part that permits a
student to discover a different internal computation.
