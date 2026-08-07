#!/usr/bin/env bash
set -euo pipefail

mode=${1:-local}
artifact_root=${2:-artifacts/collaboration-pilot-200}
bank_seed=${BANK_SEED:-20260802}
ratios=${RATIOS:-10}
# The common structural first cap stays opt-in until the paired compute gate.
# It is representation-derived and never uses one scientist's predicted cost.
objective_budget=${OBJECTIVE_BUDGET:-0}

window_checkpoint=${WINDOW_CHECKPOINT:-artifacts/nebius-rung18-20260801-current/runs/s-window-128/checkpoints/s-window-128/stage21-after.pt}
tape_checkpoint=${TAPE_CHECKPOINT:-artifacts/local-rung18-backfill-20260802/runs/s-tape4/checkpoints/s-tape4/stage18-after.pt}
w11_checkpoint=${W11_CHECKPOINT:-artifacts/nebius-rung18-20260801-current/runs/s-w11-128/checkpoints/s-w11-128/stage18-after.pt}
cyclic_checkpoint=${CYCLIC_CHECKPOINT:-}
roster=${ROSTER:-k3}

case "$mode" in
  smoke)
    seeds=(20260805)
    rounds=3
    threads=1
    attempt_workers=2
    qualification_simulations=2
    simulations=4
    train_every=1
    train_steps=2
    ;;
  schedule-smoke)
    seeds=(20260960)
    rounds=20
    threads=1
    attempt_workers=2
    qualification_simulations=16
    simulations=64
    train_every=5
    train_steps=16
    ;;
  schedule-pilot)
    seeds=(20260961)
    rounds=50
    threads=1
    attempt_workers=2
    qualification_simulations=16
    simulations=64
    train_every=5
    train_steps=16
    ;;
  local)
    seeds=(20260810)
    rounds=200
    threads=1
    attempt_workers=2
    qualification_simulations=4
    simulations=16
    train_every=10
    train_steps=8
    ;;
  local-high)
    seeds=(20260830)
    rounds=75
    threads=1
    attempt_workers=2
    qualification_simulations=16
    simulations=128
    train_every=10
    train_steps=32
    ;;
  cpu32)
    seeds=(20260810 20260811 20260812)
    rounds=200
    threads=1
    attempt_workers=2
    qualification_simulations=16
    simulations=128
    train_every=10
    train_steps=32
    ;;
  *)
    echo "usage: $0 {smoke|schedule-smoke|schedule-pilot|local|local-high|cpu32} [artifact-root]" >&2
    exit 2
    ;;
esac

scientist_names=(s-window-128 s-tape4 s-w11-128)
scientist_checkpoints=("$window_checkpoint" "$tape_checkpoint" "$w11_checkpoint")
case "$roster" in
  k3) ;;
  k4)
    if [[ -z "$cyclic_checkpoint" || ! -f "$cyclic_checkpoint" ]]; then
      echo "ROSTER=k4 requires an existing CYCLIC_CHECKPOINT" >&2
      exit 2
    fi
    scientist_names+=(s-cyclic-tape8-192)
    scientist_checkpoints+=("$cyclic_checkpoint")
    ;;
  *)
    echo "ROSTER must be k3 or k4" >&2
    exit 2
    ;;
esac
for index in "${!scientist_names[@]}"; do
  checkpoint=${scientist_checkpoints[$index]}
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 2
  fi
  .venv/bin/python - "${scientist_names[$index]}" "$checkpoint" <<'PY'
import sys
from pathlib import Path

from pgx_mcts_bench.rapid_adaptation import promoted_checkpoint_metadata

promoted_checkpoint_metadata(Path(sys.argv[2]), sys.argv[1])
PY
done

mkdir -p "$artifact_root/logs"
default_arms="adaptive-sharing adaptive-no-sharing static-sharing static-no-sharing solo-compute-matched"
read -r -a arms <<< "${ARMS:-$default_arms}"
for arm in "${arms[@]}"; do
  case "$arm" in
    adaptive-sharing|adaptive-no-sharing|static-sharing|static-no-sharing|solo-compute-matched) ;;
    *)
      echo "unknown arm in ARMS: $arm" >&2
      exit 2
      ;;
  esac
done
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
    --scientist "s-tape4=$tape_checkpoint"
    --scientist "s-w11-128=$w11_checkpoint"
  )
  if [[ "$roster" == k4 ]]; then
    scientists+=(--scientist "s-cyclic-tape8-192=$cyclic_checkpoint")
  fi
  local arm_qualification_simulations=$qualification_simulations
  local arm_simulations=$simulations
  local arm_train_steps=$train_steps
  if [[ "$arm" == solo-compute-matched ]]; then
    local scientist_count=${#scientist_names[@]}
    if [[ "$roster" == k4 ]]; then
      scientist_count=4
    fi
    scientists=(--scientist "s-window-128=$window_checkpoint")
    arm_qualification_simulations=$((qualification_simulations * scientist_count))
    arm_simulations=$((simulations * scientist_count))
    arm_train_steps=$((train_steps * scientist_count))
  fi
  local budget_args=()
  if [[ "$objective_budget" == 1 ]]; then
    budget_args=(--objective-budget)
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
      --qualification-simulations "$arm_qualification_simulations" \
      --simulations "$arm_simulations" \
      --train-every "$train_every" \
      --train-steps "$arm_train_steps" \
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
