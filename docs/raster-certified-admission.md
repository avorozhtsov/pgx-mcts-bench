# Raster-certified candidate: first admission run

`raster-certified` is the research label for the current `raster-axial`
scientist.  The implementation name is more precise: it consumes the exact
paired braid raster and applies shared axial blocks along the braid-word and
strand axes.  “Certified” means the raster encoding round-trips to the Artin
word and its semantic actions remain verifier-checkable.  It does **not** mean
that an unknotted answer is accepted without the semantic verifier, nor that
the neural solver itself has been proved correct.

## Why it is interesting

Unlike `window-local`, the raster trunk exposes local strand adjacency directly
and reuses the same block weights across strand positions.  Its trunk parameter
shapes do not depend on the configured strand capacity.  The tested network is
also smaller than the matched baseline: 247,508 trainable parameters versus
414,164 for `window-local`.  This makes the candidate a plausible route toward
more than four strands without paying for a separate token alphabet at every
strand count.

## 2026-08-09 small admission run

The two arms started from random weights and used the same seed and compute
limits:

- seed 71;
- 64 MCTS simulations per move;
- at most six self-play/training iterations per stage;
- two self-play games per iteration and 128 optimizer steps;
- ten held-out evaluation attempts for each of `L10` and `L1000`;
- promotion threshold 80%;
- balanced replay, success-only policy/value training, and adaptive rehearsal;
- stages `unknot+2`, `T(2,3)`, `P(3,4)#0`, `P(4,5)#0`.

The intended reproducer is:

```bash
uv run pgx-mcts-bench strand-architecture-gate \
  artifacts/raster-admission \
  --only window-local,raster-axial --seeds 71 --workers 1 \
  --simulations 64 --max-iterations 6 --selfplay-games 2 \
  --eval-games 10 --eval-every 2 --promote-at 0.8 --stage-limit 4
```

The original sequential invocation completed `window-local` and then exposed a
non-idempotent PyTorch worker-initialization bug before starting `raster-axial`.
The raster arm was therefore run immediately afterward in a fresh process with
the same protocol.  Worker initialization is now idempotent and has a regression
test.

| scientist | unknot | `T(2,3)` | `P(3,4)#0` | `P(4,5)#0` | wall time |
|---|---:|---:|---:|---:|---:|
| `window-local` | 85%, 4 it | 100%, 6 it | 100%, 2 it | **0%, capped at 6 it** | 559 s |
| `raster-axial` | 80%, 2 it | 100%, 4 it | 100%, 2 it | **100%, 4 it** | 467 s |

At the decisive four-strand stage, `raster-axial` solved 10/10 held-out attempts
under each objective, with one crossing change and seven charged semantic moves.
`window-local` solved 0/10 under each objective.  Both arms solved the preceding
three-strand stage at 100%.

## Decision for the local baseline

**Promising as a baseline: retain `raster-axial` in the first architecture
gate.** It
passed the four-strand admission test where the matched local-window baseline
failed, and it did so with fewer parameters and less total wall time.

This is an admission result, not a claim of superiority: it uses one training
seed and easy positive braids.  The next confirmation should use three paired
seeds and add scrambled four-strand and five-strand representations.  Promotion
to a long run should require at least 70% paired held-out solve rate and report
both solve-set differences and capped semantic `L10`/`L1000` quality.

## Improved candidate: `raster-routed`

The long-run candidate is now a distinct architecture rather than a rename of
the successful local model. `raster-routed` adds:

- a full `max_len` raster for the critic and action policy;
- one shared scorer for every adjacent row pair and the dynamic active-strand
  torus seam;
- four recurrent word-axis dilation steps, `1, 2, 4, 8`;
- zero-initialized residual gates;
- separate active-workspace and real-word-content masks;
- per-column normalization so identity workspace does not enter a global
  normalization denominator; and
- zero-initialized layer-wise conditioning on `log(A/B)`, remaining semantic
  objective budget, and remaining internal-step budget.

It can replace `raster-axial` only after a matched learning gate. The local raster
remains the control that tells us whether global routing actually helped.

## 2026-08-09 matched routed gate

All three arms used seed 71, 64 simulations per move, two self-play roots per
iteration, ten held-out evaluation attempts per objective, a six-iteration cap,
and the neutral 1:1 L10/L1000 foundation mixture.

| scientist | last stage | decisive result | wall time through result |
|---|---|---|---:|
| `window-local` | `P(4,5)#0` | 0/10, capped | 815 s |
| `raster-axial` | `P(4,5)#0` | 10/10, optimal one crossing change | 711 s |
| `raster-routed` | `unknot+2` | 10/10 solved, but mean 0.3 unnecessary crossing changes | 932 s |

The routed model learned feasibility but failed the known zero-crossing optimum.
The declared adaptive continuation reused the exact final network, optimizer, and
replay state for one additional `F=8` block at 64 simulations. Solve rate stayed
10/10, but mean crossing changes worsened from 0.3 to 0.6. It is therefore **not
admitted as the fourth scientist**. The active roster retains `raster-axial`; a
further routed-architecture redesign must use a new name and a new seed panel.
