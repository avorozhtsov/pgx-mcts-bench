#!/usr/bin/env bash
set -euo pipefail

artifact_root=${1:-artifacts/collaboration-pilot-200}
seed=${2:-20260810}
simulations=${3:-16}
evaluation_seed=${EVALUATION_SEED:-20260820}
run_root="$artifact_root/seed-$seed"
output_root="$artifact_root/evaluations/seed-$seed-sims-$simulations"
mkdir -p "$output_root/logs"

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
  local label=$1
  local run=$2
  local state=$3
  local output="$output_root/$label"
  local resume=()
  if [[ -f "$output/manifest.json" ]]; then
    resume=(--resume)
  fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/pgx-mcts-bench braid-collaboration-evaluate \
      --run "$run" \
      --output "$output" \
      --state "$state" \
      --split new70 \
      --simulations "$simulations" \
      --limit 70 \
      --seed "$evaluation_seed" \
      --device cpu \
      "${resume[@]}" >"$output_root/logs/$label.log" 2>&1 &
  pids+=("$!")
  labels+=("$label")
}

launch_one initial "$run_root/adaptive-sharing" initial
for arm in adaptive-sharing adaptive-no-sharing static-sharing static-no-sharing; do
  launch_one "final-$arm" "$run_root/$arm" final
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

if [[ $status -eq 0 ]]; then
  temporary="$output_root/.comparison.json.tmp"
  jq -s '{
    schema: "collaboration-comparison-v1",
    initial: .[0],
    adaptive_sharing: .[1],
    adaptive_no_sharing: .[2],
    static_sharing: .[3],
    static_no_sharing: .[4]
  }' \
    "$output_root/initial/report.json" \
    "$output_root/final-adaptive-sharing/report.json" \
    "$output_root/final-adaptive-no-sharing/report.json" \
    "$output_root/final-static-sharing/report.json" \
    "$output_root/final-static-no-sharing/report.json" >"$temporary"
  mv "$temporary" "$output_root/comparison.json"
  .venv/bin/pgx-mcts-bench braid-collaboration-compare \
    --treatment "$output_root/final-adaptive-sharing" \
    --control "$output_root/final-adaptive-no-sharing" \
    --output "$output_root/paired-sharing-at-adaptive.json" >/dev/null
  .venv/bin/pgx-mcts-bench braid-collaboration-compare \
    --treatment "$output_root/final-adaptive-sharing" \
    --control "$output_root/final-static-sharing" \
    --output "$output_root/paired-ordering-with-sharing.json" >/dev/null
  .venv/bin/pgx-mcts-bench braid-collaboration-compare \
    --treatment "$output_root/final-adaptive-no-sharing" \
    --control "$output_root/final-static-no-sharing" \
    --output "$output_root/paired-ordering-without-sharing.json" >/dev/null
  .venv/bin/pgx-mcts-bench braid-collaboration-compare \
    --treatment "$output_root/final-static-sharing" \
    --control "$output_root/final-static-no-sharing" \
    --output "$output_root/paired-sharing-at-static.json" >/dev/null
  jq '{initial: .initial.summary, adaptive_sharing: .adaptive_sharing.summary,
    adaptive_no_sharing: .adaptive_no_sharing.summary,
    static_sharing: .static_sharing.summary,
    static_no_sharing: .static_no_sharing.summary}' "$output_root/comparison.json"
fi
exit "$status"
