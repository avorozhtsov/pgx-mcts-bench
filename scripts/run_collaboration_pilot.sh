#!/usr/bin/env bash
set -euo pipefail

mode=${1:-smoke}
artifact_root=${2:-artifacts/current/semantic-moves-v1/collaboration-smoke-k3}
foundation_root=${FOUNDATION_ROOT:-artifacts/current/semantic-moves-v1/foundation-pretrain-semantic-v1-20260809}
assessor_gate=${ASSESSOR_GATE:-}
bank_seed=${BANK_SEED:-20260810}
ratios=${RATIOS:-10,1000}
objective_budget=${OBJECTIVE_BUDGET:-0}
remaining_budget_channel=${REMAINING_BUDGET_CHANNEL:-1}
input_bank=${INPUT_BANK:-}
input_anchor_bank=${INPUT_ANCHOR_BANK:-}

strand_checkpoint=${STRAND_CHECKPOINT:-$foundation_root/strand-graph/seed-71/checkpoint.pt}
raster_checkpoint=${RASTER_CHECKPOINT:-$foundation_root/raster-axial/seed-71/checkpoint.pt}
cyclic_checkpoint=${CYCLIC_CHECKPOINT:-$foundation_root/cyclic-memory/seed-73/checkpoint.pt}

case "$mode" in
  smoke)
    seeds=(20260810)
    rounds=${ROUNDS:-3}
    pool_size=${POOL_SIZE:-8}
    anchor_size=${ANCHOR_SIZE:-4}
    frontier=${FRONTIER:-4}
    threads=${THREADS:-1}
    attempt_workers=${ATTEMPT_WORKERS:-2}
    qualification_simulations=${QUALIFICATION_SIMULATIONS:-2}
    simulations=${SIMULATIONS:-4}
    train_every=${TRAIN_EVERY:-1}
    train_steps=${TRAIN_STEPS:-2}
    retention_attempts=${RETENTION_ATTEMPTS:-3}
    retention_simulations=${RETENTION_SIMULATIONS:-2}
    ;;
  pilot)
    seeds=(${SEEDS:-20260810})
    rounds=${ROUNDS:-200}
    pool_size=${POOL_SIZE:-200}
    anchor_size=${ANCHOR_SIZE:-70}
    frontier=${FRONTIER:-100}
    threads=${THREADS:-1}
    attempt_workers=${ATTEMPT_WORKERS:-2}
    qualification_simulations=${QUALIFICATION_SIMULATIONS:-32}
    simulations=${SIMULATIONS:-128}
    train_every=${TRAIN_EVERY:-10}
    train_steps=${TRAIN_STEPS:-32}
    retention_attempts=${RETENTION_ATTEMPTS:-24}
    retention_simulations=${RETENTION_SIMULATIONS:-64}
    ;;
  cpu32)
    seeds=(${SEEDS:-20260810 20260811 20260812})
    rounds=${ROUNDS:-200}
    pool_size=${POOL_SIZE:-200}
    anchor_size=${ANCHOR_SIZE:-70}
    frontier=${FRONTIER:-100}
    threads=${THREADS:-1}
    attempt_workers=${ATTEMPT_WORKERS:-2}
    qualification_simulations=${QUALIFICATION_SIMULATIONS:-32}
    simulations=${SIMULATIONS:-128}
    train_every=${TRAIN_EVERY:-10}
    train_steps=${TRAIN_STEPS:-32}
    retention_attempts=${RETENTION_ATTEMPTS:-24}
    retention_simulations=${RETENTION_SIMULATIONS:-64}
    ;;
  big)
    seeds=(${SEEDS:-20260820 20260821 20260822})
    rounds=${ROUNDS:-1000}
    pool_size=${POOL_SIZE:-1200}
    anchor_size=${ANCHOR_SIZE:-200}
    frontier=${FRONTIER:-100}
    threads=${THREADS:-1}
    attempt_workers=${ATTEMPT_WORKERS:-2}
    qualification_simulations=${QUALIFICATION_SIMULATIONS:-32}
    simulations=${SIMULATIONS:-128}
    train_every=${TRAIN_EVERY:-10}
    train_steps=${TRAIN_STEPS:-32}
    retention_attempts=${RETENTION_ATTEMPTS:-24}
    retention_simulations=${RETENTION_SIMULATIONS:-64}
    ;;
  *)
    echo "usage: $0 {smoke|pilot|cpu32|big} [artifact-root]" >&2
    exit 2
    ;;
esac

scientist_names=(strand-graph raster-axial cyclic-memory)
scientist_checkpoints=("$strand_checkpoint" "$raster_checkpoint" "$cyclic_checkpoint")

for index in "${!scientist_names[@]}"; do
  name=${scientist_names[$index]}
  checkpoint=${scientist_checkpoints[$index]}
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 2
  fi
  .venv/bin/python - "$name" "$checkpoint" <<'PY'
import sys
from pathlib import Path

import torch

from pgx_mcts_bench.adaptive_scientists import load_scientist

name, raw = sys.argv[1:]
checkpoint = Path(raw)
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
if payload.get("candidate") != name:
    raise SystemExit(f"checkpoint candidate mismatch: expected {name}, got {payload.get('candidate')}")
load_scientist(
    name,
    checkpoint,
    seed=0,
    device="cpu",
    simulations=1,
    require_factorized=True,
    objective_budget_channel=True,
)
PY
done

mkdir -p "$artifact_root/logs"
default_arms="adaptive-sharing adaptive-no-sharing static-sharing static-no-sharing solo-compute-matched"
read -r -a arms <<< "${ARMS:-$default_arms}"
for arm in "${arms[@]}"; do
  case "$arm" in
    adaptive-sharing|adaptive-no-sharing|static-sharing|static-no-sharing|solo-compute-matched) ;;
    *)
      echo "unknown paper arm in ARMS: $arm" >&2
      exit 2
      ;;
  esac
done
if printf '%s\n' "${arms[@]}" | grep -Eq '^(adaptive-|solo-compute-matched$)'; then
  if [[ -z "$assessor_gate" || ! -f "$assessor_gate" ]]; then
    echo "adaptive and solo scheduling require ASSESSOR_GATE" >&2
    exit 2
  fi
fi

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
    --scientist "strand-graph=$strand_checkpoint"
    --scientist "raster-axial=$raster_checkpoint"
    --scientist "cyclic-memory=$cyclic_checkpoint"
  )
  local arm_qualification_simulations=$qualification_simulations
  local arm_simulations=$simulations
  local arm_train_steps=$train_steps
  if [[ "$arm" == solo-compute-matched ]]; then
    scientists=(--scientist "strand-graph=$strand_checkpoint")
    arm_qualification_simulations=$((qualification_simulations * 3))
    arm_simulations=$((simulations * 3))
    arm_train_steps=$((train_steps * 3))
  fi
  local budget_args=()
  if [[ "$objective_budget" == 1 ]]; then
    budget_args+=(--objective-budget)
  fi
  if [[ "$remaining_budget_channel" == 1 ]]; then
    budget_args+=(--remaining-budget-channel)
  fi
  local assessor_args=()
  if [[ "$arm" == adaptive-* || "$arm" == solo-compute-matched ]]; then
    assessor_args=(--assessor-gate "$assessor_gate")
  fi
  local bank_args=()
  if [[ -n "$input_bank" || -n "$input_anchor_bank" ]]; then
    if [[ ! -f "$input_bank" || ! -f "$input_anchor_bank" ]]; then
      echo "INPUT_BANK and INPUT_ANCHOR_BANK must both name existing files" >&2
      exit 2
    fi
    bank_args=(--input-bank "$input_bank" --input-anchor-bank "$input_anchor_bank")
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
      --pool-size "$pool_size" \
      --anchor-size "$anchor_size" \
      --frontier "$frontier" \
      --ratios "$ratios" \
      --qualification-simulations "$arm_qualification_simulations" \
      --simulations "$arm_simulations" \
      --train-every "$train_every" \
      --train-steps "$arm_train_steps" \
      --batch-size 32 \
      --attempt-workers "$attempt_workers" \
      --adaptive-rehearsal \
      --rehearsal-games-per-block 8 \
      --max-rehearsal-games-per-block 32 \
      --retention-attempts "$retention_attempts" \
      --retention-simulations "$retention_simulations" \
      --direct-shared-fraction 0.05 \
      "${budget_args[@]}" \
      "${assessor_args[@]}" \
      "${bank_args[@]}" \
      --bank-seed "$bank_seed" \
      --seed "$seed" \
      --device cpu \
      "${resume[@]}" >"$log" 2>&1 &
  local pid=$!
  pids+=("$pid")
  labels+=("seed-$seed/$arm")
  echo "started seed-$seed/$arm pid=$pid log=$log"
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
