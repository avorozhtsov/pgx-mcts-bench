#!/usr/bin/env bash
set -Eeuo pipefail

# Resumable three-seed foundation pretraining for the five human-invariant
# ablations. Run only from a separate immutable checkout.
output=${1:-/srv/braid/artifacts/invariant-oracle-pretrain-20260812}
required_cpus=${REQUIRED_CPUS:-32}
workers=${WORKERS:-10}

[[ "$(uname -s)" == Linux && "$(nproc)" == "$required_cpus" ]] || {
  echo "this launcher requires the 32-vCPU Linux host" >&2
  exit 2
}
[[ -x .venv/bin/pgx-mcts-bench ]] || { echo "missing .venv" >&2; exit 2; }

mkdir -p "$output"
{
  echo "schema=invariant-oracle-pretrain-launch-v1"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "base_commit=$(git rev-parse HEAD)"
  echo "source_tree_sha256=$(find src scripts -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
  echo "workers=$workers"
  echo "cpuset=0-29"
} > "$output/launch.env"

exec taskset -c 0-29 nice -n 10 \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" \
  .venv/bin/pgx-mcts-bench braid-foundation-pretrain "$output" \
    --only raster-invariant-classical,raster-invariant-alexander,raster-invariant-jones,raster-invariant-combined-film,raster-invariant-combined-dual \
    --seeds 81,82,83 \
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
