#!/usr/bin/env bash

set -uo pipefail

image=${IMAGE:?Set IMAGE to the published pgx-mcts-bench image}
output_root=${OUTPUT_ROOT:-/srv/braid/artifacts/nebius-cpu-gate}
stage=${STAGE:-8}
eval_games=${EVAL_GAMES:-4}
train_steps=${MEASURED_TRAIN_STEPS:-8}

candidates=(
  u1-puct
  u3-uct
  search-light
  s-head-128
  s-head-budget96
  s-head-1stride
  s-window-128
  s-w11-128
  s-reg8
  s-gru128
  s-fsa32
  s-burau-oracle
  s-tape4
  s-scan-gru
)

mkdir -p "$output_root"
started=$(date +%s)
printf 'candidate\tcore\texit_code\n' > "$output_root/summary.tsv"
printf 'started_at=%s\nimage=%s\nlogical_cpus=%s\ncandidates=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$image" \
  "$(nproc)" \
  "${#candidates[@]}" > "$output_root/run.env"

pids=()
for index in "${!candidates[@]}"; do
  candidate=${candidates[$index]}
  docker run --rm \
    --cpuset-cpus "$index" \
    --volume "$output_root:/work/pgx-mcts-bench/artifacts/nebius-cpu-gate" \
    "$image" \
    braid-device-benchmark \
    --devices cpu \
    --only "$candidate" \
    --actor-batches 8 \
    --stage "$stage" \
    --eval-games "$eval_games" \
    --measured-train-steps "$train_steps" \
    --torch-threads 1 \
    --cpu-hourly 0.0248 \
    --gpu-hourly 0.7492 \
    --output "artifacts/nebius-cpu-gate/$candidate" \
    > "$output_root/$candidate.log" 2>&1 &
  pids+=("$!")
done

overall=0
for index in "${!pids[@]}"; do
  candidate=${candidates[$index]}
  if wait "${pids[$index]}"; then
    code=0
  else
    code=$?
    overall=1
  fi
  printf '%s\t%s\t%s\n' "$candidate" "$index" "$code" >> "$output_root/summary.tsv"
done

finished=$(date +%s)
printf 'finished_at=%s\nwall_seconds=%s\nexit_code=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$((finished - started))" \
  "$overall" >> "$output_root/run.env"
exit "$overall"
