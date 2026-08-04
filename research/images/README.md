# Research images

This directory stores important, publication-ready vector figures and their provenance.

## Rung-18 pair-oracle Pareto analysis

- File: `rung18-all-pair-oracles-pareto.svg`
- Generated: 2026-08-02
- Source code revision: `a3367badf92353ed521e9e1c879536f0dc4e12a8` plus the uncommitted leaderboard-reference change to `s-window-128`
- Data roots:
  - `artifacts/nebius-rung18-20260801-current/runs`
  - `artifacts/local-rung18-backfill-20260802/runs`
- Scope: candidates reaching rung 18; rung records 19 and above excluded.
- Horizontal metric:
  ```
  mean_r[L_10:1(candidate, r) - L_10:1(s-window-128, r)]
    + 10 * (1 - mean_r solve_rate(candidate, r))
  ```
- Vertical metric: total training iterations divided by cleared rungs.
- Pair oracle: at each common measured rung, use the member with lower `L_10:1`; ties prefer the higher solve rate, then the faster member. Pair `it/r` is weighted by strict per-rung win counts.
- The `s-tape4` pairs have 18 common usable ratio-10 records; other displayed pairs have 19.
- The orange point is the normalized two-axis knee, `s-window-128 + s-head-budget96`.

## Rung-12 selected-pool oracle analysis

- File: `rung12-selected-pool-oracle-pareto.svg`
- Generated: 2026-08-02
- Scope: rung records 0 through 12; records 13 and above excluded.
- Pool: the prior seven-candidate oracle pool plus local `s-triad-wst` and `s-scan-gru-tape2`.
- Partial local coverage is shown explicitly: `s-tape4` and `s-triad-wst` have 12/13 usable ratio-10 records; `s-scan-gru-tape2` has 9/13.
- Pair and triple oracles are ranked by the horizontal loss/solve objective. Missing provisional-member rungs conservatively fall back to another ensemble member.
- The top three pairs and top three triples are embedded in the figure.

## Rung-12 s-scan-gru-tape2 pair analysis

- File: `rung12-scan-gru-tape2-pairs.svg`
- Generator: `scripts/plot_rung12_scan_gru_tape2_pairs.py`
- Generated: 2026-08-02
- Scope: completed rung records 0 through 12; records 13 and above excluded.
- Pool: `s-scan-gru-tape2` paired separately with every other member of the
  selected rung-12 pool.
- `s-scan-gru-tape2` currently has 12/13 completed records. On its missing rung,
  each pair oracle conservatively falls back to the other member.
- Pair `it/r` is the parents' `it/r` weighted by the number of rungs on which
  each parent supplies the selected solution. Pair links connect each oracle to
  both parents; blue diamonds mark the Pareto frontier among these eight pairs.
