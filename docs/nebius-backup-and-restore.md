# Nebius experiment backup and recovery

This protocol protects the live Semantic-v2 experiments against loss of the
current VM and makes continuation on a replacement VM possible.

## Schedule

| Layer | Frequency | Content | Destination |
| --- | ---: | --- | --- |
| Mac mirror | Every 2 hours | All non-model results plus restart states, final checkpoints, recently active pretraining checkpoints, code diffs, untracked code, host metadata, and systemd units | `artifacts/nebius-semantic-v2-live-backup/mirror` |
| Drive delta | Every 6 hours | Files changed since the preceding Drive capsule, with the complete current SHA-256 manifest and deletion list | `projects/rf-knots/artefacts/runs/nebius-semantic-v2-live-resume` |
| Drive baseline | Sundays at 04:30 Europe/London | A complete self-contained copy of the selected mirror | The same Drive folder |

These clocks are enforced by the existing 30-minute
`Nebius experiments and recovery monitor` heartbeat. On each heartbeat it
checks the verified local receipts and runs only actions that are due, so the
health monitor and backup transfers cannot create duplicate workers.

The Mac mirror is intentionally broader for result data than for model data.
It keeps every non-`*.pt` artifact. For model data it keeps all resumable
`state.pt.gz` states, non-progress selected/final checkpoints, and only
`progress.pt` files modified during the last two days. Smoke, invalidated, and
archived model binaries are excluded.

Google Drive folder:
<https://drive.google.com/drive/folders/1w436_Q8r85BLB8qQ4IBygH9V5fzMqnyR>

## Manual operation

From the `pgx-mcts-bench` checkout:

```bash
scripts/sync_nebius_live_resume.sh sync
scripts/sync_nebius_live_resume.sh delta
scripts/sync_nebius_live_resume.sh full
```

The script is read-only on Nebius. It refuses to run when the Mac has less than
8 GiB free and uses a lock directory to prevent overlapping snapshots.

## Recovery on another VM

1. Download the newest `full` capsule and every later `delta` capsule.
2. For split capsules, verify `*.parts.sha256`, concatenate the `*.part-*` files
   in lexical order, and verify `*.tar.gz.sha256`.
3. Extract capsules in timestamp order into a new, explicit recovery directory.
4. After each extraction, remove only non-empty, non-comment paths listed by
   that capsule's `deleted-paths.txt`, resolved relative to that recovery
   directory.
5. Verify the recovered tree against the newest `*.manifest.tsv`.
6. Recreate the code revision and dirty patches recorded under
   `provenance/repos`, install the environment, and recreate the captured
   systemd units.
7. Copy the required run directory from `results/` and `resume/` back to the
   same relative artifact path. Start the launcher with its existing resume
   option; do not create a new run manifest or seed.

Before resuming, check that the last committed event/rung in the event log
matches the state checkpoint. If they disagree, preserve both and resume from
the last atomic state rather than editing the log manually.
