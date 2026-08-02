#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"

checkpoint=${CHECKPOINT:-artifacts/paired-markov-s-window-128-rung21-20260802/s-window-128.pt}
output=${OUTPUT:-artifacts/factorized-critic-five-seed-r21-20260802}
mkdir -p "$output"

for seed in 20260802 20260803 20260804 20260805 20260806; do
  seed_dir="$output/seed-$seed"
  mkdir -p "$seed_dir"
  if [[ ! -f "$seed_dir/training-metrics.json" ]]; then
    uv run python scripts/run_factorized_critic_ablation.py \
      --checkpoint "$checkpoint" \
      --candidate s-window-128 \
      --stage 21 \
      --selfplay-games 64 \
      --train-steps 256 \
      --batch-size 32 \
      --seed "$seed" \
      --device cpu \
      --output "$seed_dir" \
      >"$seed_dir/training.log" 2>&1
  fi

  if [[ ! -f "$seed_dir/shaped-report.json" ]]; then
    uv run python scripts/evaluate_paired_markov_witnesses.py \
      --checkpoint "$seed_dir/shaped-scalar.pt" \
      --candidate s-window-128 \
      --source 'R(3,12)#0' \
      --scramble-depths 4 \
      --seeds-per-depth 20 \
      --ratios 1000,10,0.1 \
      --seed "$seed" \
      --device cpu \
      --potential-cost-shaping \
      --output "$seed_dir/shaped-report.json" \
      >"$seed_dir/shaped-eval.log" 2>&1
  fi

  if [[ ! -f "$seed_dir/factorized-report.json" ]]; then
    uv run python scripts/evaluate_paired_markov_witnesses.py \
      --checkpoint "$seed_dir/factorized.pt" \
      --candidate s-window-128 \
      --source 'R(3,12)#0' \
      --scramble-depths 4 \
      --seeds-per-depth 20 \
      --ratios 1000,10,0.1 \
      --seed "$seed" \
      --device cpu \
      --use-auxiliary-value \
      --output "$seed_dir/factorized-report.json" \
      >"$seed_dir/factorized-eval.log" 2>&1
  fi
done
