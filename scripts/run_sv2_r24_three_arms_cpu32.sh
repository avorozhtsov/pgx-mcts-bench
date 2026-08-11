#!/usr/bin/env bash
set -Eeuo pipefail

# Run the three remaining Semantic-v2 R24 arms on one 32-logical-CPU host.
# Each arm owns ten CPUs; CPUs 30-31 remain available for the OS and monitoring.

output_root=${1:-/srv/braid/artifacts/semantic-v2-r24}
checkpoint_root=${CHECKPOINT_ROOT:-artifacts/current/semantic-moves-v1/k3-budget-critic-repair-20260810}
bank=${BANK:-research/semantic-v2-prefix24.json}
required_cpus=${REQUIRED_CPUS:-32}
torch_threads=${TORCH_THREADS_PER_SCIENTIST:-3}
seed=${SEED:-20262020}

if [[ "$(uname -s)" != Linux ]]; then
  echo "This launcher requires Linux taskset CPU isolation" >&2
  exit 2
fi
for command in nproc taskset sha256sum; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
if [[ "$(nproc)" != "$required_cpus" ]]; then
  echo "expected exactly $required_cpus logical CPUs; found $(nproc)" >&2
  exit 2
fi
if [[ ! -x .venv/bin/pgx-mcts-bench || ! -x .venv/bin/python ]]; then
  echo "run from a pgx-mcts-bench checkout with .venv installed" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" || -n "$(git -C ../rf-knots status --porcelain)" ]]; then
  echo "refusing a scientifically mutable launch from dirty source checkouts" >&2
  exit 2
fi

strand="$checkpoint_root/strand-graph/rollback-guarded.pt"
raster="$checkpoint_root/raster-axial/rollback-guarded.pt"
cyclic="$checkpoint_root/cyclic-memory/rollback-guarded.pt"
for path in "$bank" "$strand" "$raster" "$cyclic"; do
  [[ -f "$path" ]] || { echo "missing frozen input: $path" >&2; exit 2; }
done

mkdir -p "$output_root/logs"
{
  echo "schema=semantic-v2-r24-cpu32-launch-v1"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pgx_commit=$(git rev-parse HEAD)"
  echo "rf_knots_commit=$(git -C ../rf-knots rev-parse HEAD)"
  echo "logical_cpus=$(nproc)"
  echo "torch_threads_per_scientist=$torch_threads"
  echo "seed=$seed"
  sha256sum "$bank" "$strand" "$raster" "$cyclic"
} > "$output_root/launch.env"

.venv/bin/python - "$strand" "$raster" "$cyclic" <<'PY'
import sys
from pathlib import Path

from pgx_mcts_bench.adaptive_scientists import load_scientist

for name, raw in zip(("strand-graph", "raster-axial", "cyclic-memory"), sys.argv[1:]):
    load_scientist(
        name,
        Path(raw),
        seed=0,
        device="cpu",
        simulations=1,
        require_factorized=True,
        objective_budget_channel=True,
    )
PY

arms=(adaptive-no-sharing static-sharing adaptive-sharing)
cpusets=(0-9 10-19 20-29)
pids=()

stop_children() {
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait || true
}
trap stop_children INT TERM

for index in "${!arms[@]}"; do
  arm=${arms[$index]}
  output="$output_root/$arm"
  resume=()
  [[ ! -f "$output/manifest.json" ]] || resume=(--resume)
  taskset -c "${cpusets[$index]}" \
    env OMP_NUM_THREADS="$torch_threads" \
        MKL_NUM_THREADS="$torch_threads" \
        OPENBLAS_NUM_THREADS="$torch_threads" \
        NUMEXPR_NUM_THREADS="$torch_threads" \
        PYTHONUNBUFFERED=1 \
    .venv/bin/pgx-mcts-bench braid-sv2-coordinated \
      --output "$output" \
      --bank "$bank" \
      --scientist "strand-graph=$strand" \
      --scientist "raster-axial=$raster" \
      --scientist "cyclic-memory=$cyclic" \
      --arm "$arm" \
      --ratios 10,1000 \
      --simulations 64 \
      --qualification-simulations 64 \
      --qualification-attempts 1 \
      --f-native 10 \
      --selfplay-games 8 \
      --train-steps 96 \
      --batch-size 64 \
      --evaluation-attempts 4 \
      --block-size 10 \
      --retention-target 0.80 \
      --action-horizon 128 \
      --rungs 24 \
      --seed "$seed" \
      --torch-threads "$torch_threads" \
      --parallel-scientists \
      --device cpu \
      "${resume[@]}" > "$output_root/logs/$arm.log" 2>&1 &
  pids+=("$!")
  echo "started $arm pid=${pids[-1]} cpus=${cpusets[$index]}"
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    code=0
  else
    code=$?
    status=1
  fi
  printf '%s\t%s\t%s\n' "${arms[$index]}" "${cpusets[$index]}" "$code" \
    >> "$output_root/completion.tsv"
done

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_root/launch.env"
exit "$status"
