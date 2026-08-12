#!/usr/bin/env bash
set -Eeuo pipefail

# Low-priority, resumable three-seed foundation pretraining for the three
# raster-axial capacity variants. Run from its own immutable checkout: active
# paper arms may keep importing modules in newly spawned worker processes.

output=${1:-/srv/braid/artifacts/raster-axial-capacity-pretrain-20260811}
required_cpus=${REQUIRED_CPUS:-32}
workers=${WORKERS:-9}

if [[ "$(uname -s)" != Linux ]]; then
  echo "This launcher requires Linux taskset CPU isolation" >&2
  exit 2
fi
if [[ "$(nproc)" != "$required_cpus" ]]; then
  echo "expected $required_cpus logical CPUs; found $(nproc)" >&2
  exit 2
fi
if [[ ! -x .venv/bin/pgx-mcts-bench ]]; then
  echo "missing .venv/bin/pgx-mcts-bench" >&2
  exit 2
fi

mkdir -p "$output"
{
  echo "schema=raster-axial-capacity-pretrain-launch-v1"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "base_commit=$(git rev-parse HEAD)"
  echo "workers=$workers"
  echo "nice=15"
  echo "cpuset=0-29"
} > "$output/launch.env"

exec taskset -c 0-29 nice -n 15 \
  env OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      PYTHONUNBUFFERED=1 \
  .venv/bin/pgx-mcts-bench braid-foundation-pretrain "$output" \
    --only raster-axial-v2,raster-axial-v3,raster-axial-v4 \
    --seeds 71,72,73 \
    --workers "$workers" \
    --native-levels 5,8,12,16 \
    --simulation-levels 64,128,256,512 \
    --initial-old-cycles 1 \
    --max-old-cycles 8 \
    --evaluation-target 0.70 \
    --retention-target 0.80 \
    --selfplay-games 2 \
    --eval-games 10 \
    --eval-every 2 \
    --retro-games 24 \
    --promote-at 0.80 \
    --stage-limit 6 \
    --device cpu
