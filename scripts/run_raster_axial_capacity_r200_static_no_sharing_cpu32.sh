#!/usr/bin/env bash
set -Eeuo pipefail

selection=${1:?usage: $0 SELECTION.json STATE_ROOT [OUTPUT]}
state_root=${2:?usage: $0 SELECTION.json STATE_ROOT [OUTPUT]}
output=${3:-/srv/braid/artifacts/raster-axial-capacity-r200-static-no-sharing-20260811}
bank=${BANK:-research/semantic-v2-r200.json}
prior_bank=${PRIOR_BANK:-research/semantic-v2-prefix24.json}

[[ "$(uname -s)" == Linux && "$(nproc)" == 32 ]] || {
  echo "this launcher requires the 32-vCPU Linux host" >&2
  exit 2
}
[[ -f "$selection" && -f "$bank" && -f "$prior_bank" ]] || {
  echo "missing selection, R200 bank, or R24 prior bank" >&2
  exit 2
}

mapfile -t scientists < <(
  .venv/bin/python - "$selection" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["selected"]:
    print(f'{row["name"]}={row["checkpoint"]}')
PY
)
arguments=()
for scientist in "${scientists[@]}"; do
  name=${scientist%%=*}
  state="$state_root/$name/state.pt.gz"
  [[ -f "$state" ]] || { echo "missing continuation state: $state" >&2; exit 2; }
  arguments+=(--scientist "$scientist" --initial-state "$name=$state")
done
resume=()
[[ ! -f "$output/manifest.json" ]] || resume=(--resume)

exec taskset -c 0-29 nice -n 10 \
  env OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 \
      NUMEXPR_NUM_THREADS=3 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
      PYTHONUNBUFFERED=1 \
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
    --f-native 5 \
    --selfplay-games 8 \
    --train-steps 96 \
    --batch-size 64 \
    --evaluation-attempts 4 \
    --block-size 10 \
    --retention-target 0.80 \
    --action-horizon 128 \
    --rungs 200 \
    --seed 20262320 \
    --torch-threads 3 \
    --parallel-scientists \
    --adaptive-compute \
    --device cpu \
    "${resume[@]}"
