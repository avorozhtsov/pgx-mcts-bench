# First corrected U1 study

Date: 2026-07-26

## Why the original baseline was rejected

The first compact learned-rules MuZero agent lost 120/120 games, but its
self-play length collapsed from about 49 plies to 14. It did not reliably
model the two-pass terminal condition or terminal reward, and short games gave
it roughly half AlphaZero's training positions. A direct arena also reproduced
the same empty-board game and was dominated by color.

Those results diagnosed the harness; they were not evidence about AlphaZero
versus MuZero.

## Corrections

- Added previous-pass and normalized-move observation planes.
- Added terminal prediction and terminal-transition oversampling.
- Increased supervision weight for terminal rewards and termination.
- Batched root and leaf neural inference across concurrent games.
- Made the default MuZero rules-aware: Pgx supplies exact legality,
  termination, and reward at imagined nodes; representation, latent dynamics,
  policy, and value remain learned.
- Retained pure learned-rules search as
  `SearchConfig.muzero_exact_rules=False`.
- Masked pass for both agents through ply 23. This artificial benchmark rule
  prevents trivial pass-pass games and guarantees useful depth.
- Added a minimum generated-position budget per training iteration.
- Changed the arena to paired random six-ply openings, with every opening
  played under both color assignments.
- Added color-conditioned arena metrics.

On the local M2 CPU, batching eight roots reduced median 32-simulation search
time from 0.147s to 0.061s for AlphaZero (2.39x), and from 0.158s to 0.071s
for rules-aware MuZero (2.21x).

## Configuration

- Pgx Go 6x6, komi 3.5
- Pass available from ply 24
- Maximum 72 plies
- U1 AlphaZero PUCT
- 32 simulations per move
- Three independent training seeds
- Ten iterations
- Eight concurrently generated self-play games per batch
- At least 256 new positions per iteration
- 32 optimizer steps per iteration, batch size 32
- 32 network channels
- 40 paired-opening arena games per seed
- AlphaZero parameters: 44,369
- MuZero parameters: 92,887

## Results

| Seed | AlphaZero | MuZero | AZ as Black | AZ as White |
|---:|---:|---:|---:|---:|
| 0 | 29 | 11 | 18/20 | 11/20 |
| 1 | 36 | 4 | 17/20 | 19/20 |
| 2 | 32 | 8 | 17/20 | 15/20 |
| Total | 97 | 23 | 52/60 | 45/60 |

AlphaZero score: **80.8%**. Wilson 95% interval: **72.9%–86.9%**.

Mean self-play game length across iterations was 42.7 plies for AlphaZero and
36.2 for MuZero. Final replay sizes were 3,220–3,591 AlphaZero positions and
3,660–4,108 MuZero positions, so the result is not explained by MuZero
receiving less data.

## Interpretation

Under this small-compute, rules-aware setup, AlphaZero is clearly stronger
than the compact MuZero baseline. The comparison is still not parameter
matched: MuZero has about 2.1 times as many parameters and spends roughly
3.7 times as long in training/self-play per seed.

The next experiment should hold this harness fixed and compare U1 through U5
within each agent family over several seeds. Only after that should pure
learned-rules MuZero be revisited as a separate ablation.
