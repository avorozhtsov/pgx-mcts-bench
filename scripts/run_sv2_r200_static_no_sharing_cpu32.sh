#!/usr/bin/env bash
set -euo pipefail

# Continue the completed local R24 static/no-sharing scientists through the
# frozen R200 group while yielding CPU priority to the three active R24 arms.
output=${1:-/srv/braid/artifacts/semantic-v2-r200/static-no-sharing}
bank=${BANK:-research/semantic-v2-r200.json}
prior_bank=${PRIOR_BANK:-research/semantic-v2-prefix24.json}
state_root=${STATE_ROOT:-/srv/braid/artifacts/semantic-v2-r200-initial-states}
cpus=${CPUS:-0-29}
nice_level=${NICE_LEVEL:-10}

exec taskset -c "$cpus" nice -n "$nice_level" \
  env OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 PYTHONUNBUFFERED=1 \
  .venv/bin/pgx-mcts-bench braid-sv2-coordinated \
  --output "$output" \
  --bank "$bank" \
  --prior-bank "$prior_bank" \
  --scientist strand-graph=artifacts/current/semantic-moves-v1/k3-budget-critic-repair-20260810/strand-graph/rollback-guarded.pt \
  --scientist raster-axial=artifacts/current/semantic-moves-v1/k3-budget-critic-repair-20260810/raster-axial/rollback-guarded.pt \
  --scientist cyclic-memory=artifacts/current/semantic-moves-v1/k3-budget-critic-repair-20260810/cyclic-memory/rollback-guarded.pt \
  --initial-state "strand-graph=$state_root/strand-graph/state.pt.gz" \
  --initial-state "raster-axial=$state_root/raster-axial/state.pt.gz" \
  --initial-state "cyclic-memory=$state_root/cyclic-memory/state.pt.gz" \
  --arm static-no-sharing \
  --ratios 10,1000 \
  --simulations 64 \
  --qualification-simulations 64 \
  --qualification-attempts 1 \
  --f-native 5 \
  --selfplay-games 8 \
  --train-steps 96 \
  --batch-size 64 \
  --evaluation-attempts 4 \
  --block-size 10 \
  --retention-target 0.80 \
  --action-horizon 128 \
  --rungs 200 \
  --seed 20262120 \
  --torch-threads 3 \
  --parallel-scientists \
  --adaptive-compute \
  --device cpu
