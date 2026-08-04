#!/usr/bin/env bash
set -euo pipefail

mode=${1:-local}
artifact_root=${2:-artifacts/collaboration-pilot-200}
bank_seed=${BANK_SEED:-20260802}
ratios=${RATIOS:-10}
# Predicted caps stay opt-in until every scientist used by the arm passes its
# own source-disjoint calibration and paired-retention gate.
objective_budget=${OBJECTIVE_BUDGET:-0}
objective_budget_audit_every=${OBJECTIVE_BUDGET_AUDIT_EVERY:-10}

window_checkpoint=${WINDOW_CHECKPOINT:-artifacts/nebius-rung18-20260801-current/runs/s-window-128/checkpoints/s-window-128/stage21-after.pt}
tape_checkpoint=${TAPE_CHECKPOINT:-artifacts/nebius-rung18-20260801-current/runs/d-tape4-u1/checkpoints/d-tape4-u1/stage21-after.pt}
w11_checkpoint=${W11_CHECKPOINT:-artifacts/nebius-rung18-20260801-current/runs/s-w11-128/checkpoints/s-w11-128/stage18-after.pt}
cyclic_checkpoint=${CYCLIC_CHECKPOINT:-}
roster=${ROSTER:-k3}

case "$mode" in
  local)
    seeds=(20260810)
    rounds=200
    threads=1
    attempt_workers=2
    qualification_simulations=4
    simulations=16
    train_steps=8
    ;;
  local-high)
    seeds=(20260830)
    rounds=75
    threads=1
    attempt_workers=2
    qualification_simulations=16
    simulations=128
    train_steps=32
    ;;
  cpu32)
    seeds=(20260810 20260811 20260812)
    rounds=200
    threads=1
    attempt_workers=2
    qualification_simulations=16
    simulations=128
    train_steps=32
    ;;
  *)
    echo "usage: $0 {local|local-high|cpu32} [artifact-root]" >&2
    exit 2
    ;;
esac

for checkpoint in "$window_checkpoint" "$tape_checkpoint" "$w11_checkpoint"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 2
  fi
done
scientist_names=(s-window-128 d-tape4-u1 s-w11-128)
scientist_checkpoints=("$window_checkpoint" "$tape_checkpoint" "$w11_checkpoint")
for index in "${!scientist_names[@]}"; do
  .venv/bin/python - "${scientist_names[$index]}" "${scientist_checkpoints[$index]}" <<'PY'
import sys
from pathlib import Path

from pgx_mcts_bench.rapid_adaptation import promoted_checkpoint_metadata

promoted_checkpoint_metadata(Path(sys.argv[2]), sys.argv[1])
PY
done
case "$roster" in
  k3) ;;
  k4)
    if [[ -z "$cyclic_checkpoint" || ! -f "$cyclic_checkpoint" ]]; then
      echo "ROSTER=k4 requires an existing CYCLIC_CHECKPOINT" >&2
      exit 2
    fi
    ;;
  *)
    echo "ROSTER must be k3 or k4" >&2
    exit 2
    ;;
esac

mkdir -p "$artifact_root/logs"
arms=(adaptive-sharing adaptive-no-sharing static-sharing static-no-sharing)
pids=()
labels=()

stop_children() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}
trap stop_children INT TERM

launch_one() {
  local seed=$1
  local arm=$2
  local output="$artifact_root/seed-$seed/$arm"
  local log="$artifact_root/logs/seed-$seed--$arm.log"
  local resume=()
  if [[ -f "$output/manifest.json" ]]; then
    resume=(--resume)
  fi
  local scientists=(
    --scientist "s-window-128=$window_checkpoint"
    --scientist "d-tape4-u1=$tape_checkpoint"
    --scientist "s-w11-128=$w11_checkpoint"
  )
  if [[ "$roster" == k4 ]]; then
    scientists+=(--scientist "s-cyclic-tape8-192=$cyclic_checkpoint")
  fi
  local budget_args=()
  if [[ "$objective_budget" == 1 ]]; then
    budget_args=(
      --objective-budget
      --objective-budget-audit-every "$objective_budget_audit_every"
    )
  fi
  OMP_NUM_THREADS=$threads \
  MKL_NUM_THREADS=$threads \
  OPENBLAS_NUM_THREADS=$threads \
  VECLIB_MAXIMUM_THREADS=$threads \
  NUMEXPR_NUM_THREADS=$threads \
    .venv/bin/pgx-mcts-bench braid-collaborative-scientists \
      --output "$output" \
      "${scientists[@]}" \
      --arm "$arm" \
      --rounds "$rounds" \
      --pool-size 200 \
      --anchor-size 70 \
      --frontier 100 \
      --ratios "$ratios" \
      --qualification-simulations "$qualification_simulations" \
      --simulations "$simulations" \
      --train-every 10 \
      --train-steps "$train_steps" \
      --batch-size 32 \
      --attempt-workers "$attempt_workers" \
      "${budget_args[@]}" \
      --bank-seed "$bank_seed" \
      --seed "$seed" \
      --device cpu \
      "${resume[@]}" >"$log" 2>&1 &
  local pid=$!
  local label="seed-$seed/$arm"
  pids+=("$pid")
  labels+=("$label")
  echo "started $label pid=$pid log=$log"
}

for seed in "${seeds[@]}"; do
  for arm in "${arms[@]}"; do
    launch_one "$seed" "$arm"
  done
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed ${labels[$index]}"
  else
    echo "failed ${labels[$index]} (see log)" >&2
    status=1
  fi
done
trap - INT TERM
exit "$status"
