# Pre-semantic-moves-v1 experiment archive

Snapshot created on 2026-08-09.

The payload under `artifacts/` contains historical reports and non-model run data
produced before the objective contract was fixed to charge semantic knot actions
only. It is intentionally ignored by Git because it is large. The tracked
`artifact-index.json` records every top-level tree moved into the snapshot.

Model histories were pruned after the move. `binary-retention.json` records the
4,924 deleted progress/interruption/stage/round binaries and the SHA-256 of 111
retained final or latest-per-arm model states. Duplicate checkpoint transfer
tarballs were removed as model history.

These checkpoints may be used for migration regression tests, but they are not
valid initial weights or comparable leaderboard entries for the new scientist
roster. The active roster is trained from scratch.
