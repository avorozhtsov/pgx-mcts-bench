# Bounded internal options

Serial agents may need to move their head, update registers or colours, or write
a tape before performing a donated semantic braid edit. These operations are
treated as a bounded option rather than prescribed as a single hand-written
route.

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

Collaboration preserves the receiver's head, registers, colours, and tape across
successive donated edits. `train_bounded_option_step` is the reusable training
primitive; `braid-collaborative-scientists` records its dose and route-loss
diagnostics. The shortest-route target remains a stabilizing signal, while the
bounded option loss permits a receiver to discover a different internal
computation.
