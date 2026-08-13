#!/usr/bin/env bash
set -Eeuo pipefail

fork_root=${1:?usage: $0 FORK_ROOT OUTPUT [SEED]}
output=${2:?usage: $0 FORK_ROOT OUTPUT [SEED]}
seed=${3:-20262411}
prefix_root=${PREFIX_ROOT:-/srv/braid/artifacts/invariant-oracle-architecture-fork-r24-20260813}
checkpoint_root=${CHECKPOINT_ROOT:-/srv/braid/artifacts/invariant-oracle-pretrain-20260812}
bank=${BANK:-$prefix_root/remaining-suffix.json}
prior_bank=${PRIOR_BANK:-$prefix_root/prior-prefix.json}

[[ "$(uname -s)" == Linux && "$(nproc --all)" == 32 ]] || {
  echo "this launcher requires the 32-vCPU Linux host" >&2
  exit 2
}
[[ -d "$fork_root" && -f "$bank" && -f "$prior_bank" ]] || {
  echo "missing fork root or paired R11 banks" >&2
  exit 2
}

film_checkpoint=$checkpoint_root/raster-invariant-combined-film/seed-81/checkpoint.pt
dual_checkpoint=$checkpoint_root/raster-invariant-combined-dual/seed-81/checkpoint.pt
film_parent=$prefix_root/parents/raster-invariant-combined-film/state.pt.gz
dual_parent=$prefix_root/parents/raster-invariant-combined-dual/state.pt.gz

scientists=(
  "raster-invariant-combined-film=$film_checkpoint"
  "raster-invariant-combined-film-deep1=$film_checkpoint"
  "raster-invariant-combined-film-deep2=$film_checkpoint"
  "raster-invariant-combined-dual=$dual_checkpoint"
  "raster-invariant-combined-dual-deep1=$dual_checkpoint"
  "raster-invariant-combined-dual-deep=$dual_checkpoint"
  "raster-invariant-combined-dual-deep4=$dual_checkpoint"
)
states=(
  "raster-invariant-combined-film=$film_parent"
  "raster-invariant-combined-film-deep1=$fork_root/film/raster-invariant-combined-film-deep1/state.pt.gz"
  "raster-invariant-combined-film-deep2=$fork_root/film/raster-invariant-combined-film-deep2/state.pt.gz"
  "raster-invariant-combined-dual=$dual_parent"
  "raster-invariant-combined-dual-deep1=$fork_root/dual/raster-invariant-combined-dual-deep1/state.pt.gz"
  "raster-invariant-combined-dual-deep=$fork_root/dual/raster-invariant-combined-dual-deep/state.pt.gz"
  "raster-invariant-combined-dual-deep4=$fork_root/dual/raster-invariant-combined-dual-deep4/state.pt.gz"
)
for required in "$film_checkpoint" "$dual_checkpoint" "$film_parent" "$dual_parent"; do
  [[ -f "$required" ]] || { echo "missing $required" >&2; exit 2; }
done
for state in "${states[@]}"; do
  [[ -f "${state#*=}" ]] || { echo "missing ${state#*=}" >&2; exit 2; }
done

arguments=()
for scientist in "${scientists[@]}"; do arguments+=(--scientist "$scientist"); done
for state in "${states[@]}"; do arguments+=(--initial-state "$state"); done
resume=()
[[ ! -f "$output/manifest.json" ]] || resume=(--resume)

exec taskset -c 0-29 \
  env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      NUMEXPR_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
      PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" \
  .venv/bin/pgx-mcts-bench braid-sv2-coordinated \
    --output "$output" \
    --bank "$bank" \
    --prior-bank "$prior_bank" \
    "${arguments[@]}" \
    --arm static-no-sharing \
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
    --rungs 11 \
    --seed "$seed" \
    --torch-threads 2 \
    --parallel-scientists \
    --device cpu \
    "${resume[@]}"
