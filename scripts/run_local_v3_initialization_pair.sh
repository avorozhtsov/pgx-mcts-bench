#!/usr/bin/env bash
set -euo pipefail

repo=${LOCAL_ABLATION_REPO:-/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation}
run=${LOCAL_ABLATION_RUN:-/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817}
warm=${LOCAL_ABLATION_WARM:-/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/mastery-v3-l40s-20260816/artifacts/cyclic-memory-deep-v3-migrated.pt}
native=${LOCAL_ABLATION_NATIVE:-$run/checkpoints/native-deep-v3.pt}
bank=$run/inputs/train-bank.json
heldout=$run/inputs/heldout-bank.json

export VECLIB_MAXIMUM_THREADS=6
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

mkdir -p "$run/initialization-pair"
cd "$repo"

run_q() {
  local label=$1
  local checkpoint=$2
  local output=$run/initialization-pair/$label-q20
  if [[ ! -f "$output/report.json" ]]; then
    local resume_args=()
    if [[ -f "$output/manifest.json" ]]; then
      resume_args=(--resume)
    fi
    uv run pgx-mcts-bench braid-sv2-coordinated \
      --output "$output" \
      --bank "$bank" \
      --scientist "cyclic-memory-deep-v3=$checkpoint" \
      --arm static-no-sharing \
      --ratios 10,1000 \
      --simulations 32 \
      --qualification-simulations 32 \
      --qualification-attempts 1 \
      --f-native 2 \
      --selfplay-games 2 \
      --train-steps 24 \
      --batch-size 32 \
      --evaluation-attempts 2 \
      --no-evaluation-root-noise \
      --block-size 4 \
      --retention-target 0.8 \
      --action-horizon 128 \
      --rungs 20 \
      --seed 2026081802 \
      --torch-threads 6 \
      --no-parallel-scientists \
      --device cpu \
      "${resume_args[@]}"
  fi
  if [[ ! -f "$run/initialization-pair/$label-heldout-q20.json" ]]; then
    uv run python scripts/evaluate_local_q_skm_checkpoint.py \
      --state "$output/state.pt.gz" \
      --scientist cyclic-memory-deep-v3 \
      --bank "$heldout" \
      --output "$run/initialization-pair/$label-heldout-q20.json" \
      --simulations 32 \
      --threads 6 \
      --seed 2026081803
  fi
}

run_q warm "$warm"
run_q native "$native"
