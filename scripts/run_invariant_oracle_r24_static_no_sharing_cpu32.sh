#!/usr/bin/env bash
set -Eeuo pipefail

selection=${1:?usage: $0 SELECTION.json [OUTPUT]}
output=${2:-/srv/braid/artifacts/invariant-oracle-r24-static-no-sharing-20260812}
bank=${BANK:-research/semantic-v2-prefix24.json}

[[ "$(uname -s)" == Linux && "$(nproc)" == 32 ]] || {
  echo "this launcher requires the 32-vCPU Linux host" >&2
  exit 2
}
[[ -f "$selection" && -f "$bank" ]] || {
  echo "missing selection or R24 bank" >&2
  exit 2
}

mapfile -t scientists < <(
  .venv/bin/python - "$selection" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["selected"]:
    print(f'{row["name"]}={row["checkpoint"]}')
PY
)
[[ "${#scientists[@]}" == 5 ]] || { echo "selection must contain five scientists" >&2; exit 2; }

arguments=()
for scientist in "${scientists[@]}"; do arguments+=(--scientist "$scientist"); done
resume=()
[[ ! -f "$output/manifest.json" ]] || resume=(--resume)

exec taskset -c 0-29 \
  env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      NUMEXPR_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
      PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" \
  .venv/bin/pgx-mcts-bench braid-sv2-coordinated \
    --output "$output" \
    --bank "$bank" \
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
    --rungs 24 \
    --seed 20262221 \
    --torch-threads 2 \
    --parallel-scientists \
    --device cpu \
    "${resume[@]}"
