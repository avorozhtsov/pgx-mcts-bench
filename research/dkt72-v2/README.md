# DKT72-PD-v2 gate

DKT72-PD-v2 is blocked until two genuinely trained 12-strand checkpoint
systems pass the machine-readable eligibility gate. Shape migration alone is
not training and cannot satisfy the gate.

The gate is fail-closed and outcome-blind. It requires training examples and
retention measurements at every strand count from 6 through 12, at least one
native success on a 6+-strand registered Q representation, 72/72 capacity in a
no-search dry run, and no exact-representation or knot-identity overlap with
the DKT72 panel.

The frozen Q4000 schedule contains two exact DKT72 braid representations and
20 rows belonging to 13 DKT72 knot identities. The first overlap appears in
`q200-3`. Consequently, checkpoints through the clean 500-row prefix
(`Q20`, both `Q40` groups, `Q200-1`, and `Q200-2`) remain candidates; later
checkpoints require a separately registered holdout-filtered continuation.

Use `scripts/register_dkt72_v2.py` in four stages:

1. `eligibility` creates one report per checkpoint from a pipeline-produced
   metadata JSON matching `eligibility-template.json`.
2. `select` chooses exactly two systems without reading DKT72 outcomes. If an
   eligible cyclic-band system exists, it is paired with the best ordinary
   Artin system so the extra action alphabet is explicit.
3. `freeze` pins the panel, both checkpoint hashes, seeds, common compute dose,
   horizon, all-72 denominators, and certification policy.
4. `preflight` re-hashes everything immediately before a separate evaluator is
   launched. A nonzero/blocked result must prevent launch.

Every strict current upper-bound improvement is passed to `certify`, which
replays the complete witness, adds it to the append-only evidence inventory
with solver and search metadata, and writes an independent lower-bound
certificate. It is labelled exact only if that lower bound meets the new upper
bound.
