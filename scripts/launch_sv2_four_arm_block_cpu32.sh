#!/usr/bin/env bash
set -Eeuo pipefail

block=${1:?usage: $0 BLOCK PREVIOUS_ROOT [OUTPUT_ROOT]}
previous_root=${2:?usage: $0 BLOCK PREVIOUS_ROOT [OUTPUT_ROOT]}
output_root=${3:-/srv/braid/artifacts/semantic-v2-large/$block}
bank_root=${BANK_ROOT:-/srv/braid/banks/semantic-v2-large}
state_root=${STATE_ROOT:-/srv/braid/artifacts/semantic-v2-continuation-states/$block}
selection=${SELECTION:-research/semantic-v1-k3-selection.json}

[[ "$(uname -s)" == Linux && "$(nproc)" == 32 ]] || {
  echo "this launcher requires the 32-vCPU Linux host" >&2
  exit 2
}
bank="$bank_root/$block.json"
prior="$bank_root/prior-$block.json"
[[ -f "$bank" && -f "$prior" ]] || { echo "missing frozen bank or prior bank" >&2; exit 2; }

case "$block" in
  r400-[1-6]) index=${block#r400-} ;;
  r100-tail) index=7 ;;
  *) echo "unknown continuation block: $block" >&2; exit 2 ;;
esac
seed=$((20262120 + 100 * index))
rungs=$(python - "$bank" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))["rows"]))
PY
)

arms=(static-no-sharing adaptive-no-sharing static-sharing adaptive-sharing)
for arm in "${arms[@]}"; do
  previous="$previous_root/$arm/state.pt.gz"
  [[ -f "$previous" ]] || { echo "missing completed state: $previous" >&2; exit 2; }
  [[ ! -e "$output_root/$arm" ]] || { echo "output already exists: $output_root/$arm" >&2; exit 2; }
  PYTHONPATH=src /srv/braid/work/pgx-mcts-bench/.venv/bin/python \
    scripts/export_sv2_scientist_states.py \
    "$previous" "$selection" "$state_root/$arm"
done

for arm in "${arms[@]}"; do
  output="$output_root/$arm"
  initial="$state_root/$arm"
  unit="sv2-${block}-${arm//-/_}"
  command="cd $PWD && exec taskset -c 0-29 nice -n 5 env OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 PYTHONUNBUFFERED=1 PYTHONPATH=src /srv/braid/work/pgx-mcts-bench/.venv/bin/pgx-mcts-bench braid-sv2-coordinated --output $output --bank $bank --prior-bank $prior --scientist strand-graph=/srv/braid/work/pgx-mcts-bench/artifacts/current/semantic-moves-v1/k3-budget-critic-repair-20260810/strand-graph/rollback-guarded.pt --scientist raster-axial=/srv/braid/work/pgx-mcts-bench/artifacts/current/semantic-moves-v1/k3-budget-critic-repair-20260810/raster-axial/rollback-guarded.pt --scientist cyclic-memory=/srv/braid/work/pgx-mcts-bench/artifacts/current/semantic-moves-v1/k3-budget-critic-repair-20260810/cyclic-memory/rollback-guarded.pt --initial-state strand-graph=$initial/strand-graph/state.pt.gz --initial-state raster-axial=$initial/raster-axial/state.pt.gz --initial-state cyclic-memory=$initial/cyclic-memory/state.pt.gz --arm $arm --ratios 10,1000 --simulations 64 --qualification-simulations 64 --qualification-attempts 1 --f-native 5 --selfplay-games 8 --train-steps 96 --batch-size 64 --evaluation-attempts 4 --block-size 10 --retention-target 0.80 --action-horizon 128 --rungs $rungs --seed $seed --torch-threads 3 --parallel-scientists --adaptive-compute --device cpu"
  sudo systemd-run \
    --unit="$unit" \
    --description="SV2 $block $arm" \
    --property=User=artemvorozhtsov \
    --property=Group=artemvorozhtsov \
    --property=WorkingDirectory="$PWD" \
    --property=Nice=5 \
    --property=CPUWeight=100 \
    --property=StandardOutput=journal \
    --property=StandardError=journal \
    /bin/bash -lc "$command"
done

echo "launched $block ($rungs representations) for all four arms with seed $seed"
