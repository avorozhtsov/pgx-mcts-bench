#!/usr/bin/env bash
set -euo pipefail

repo=${SKM_REPO:-/srv/braid/work-single-knot-mastery/pgx-mcts-bench}
catalogue=${SKM_CATALOGUE:?set SKM_CATALOGUE to the pinned gap catalogue}
checkpoint=${SKM_CHECKPOINT:?set SKM_CHECKPOINT to the pinned inherited checkpoint}
output=${SKM_OUTPUT:?set SKM_OUTPUT to a new or resumable artifact directory}
knot=${SKM_KNOT:?set SKM_KNOT}
scientist=${SKM_SCIENTIST:?set SKM_SCIENTIST}
distillation=${SKM_DISTILLATION:-}
rehearsal_panel=${SKM_REHEARSAL_PANEL:-}
source_hashes=${SKM_SOURCE_HASHES:-$output/source-hashes.sha256}
python=${SKM_PYTHON:-/srv/braid/work/pgx-mcts-bench/.venv/bin/python}
cores=${SKM_CORES:-16-31}
parallel=${SKM_PARALLEL_SEARCHES:-8}
torch_threads=${SKM_TORCH_THREADS:-8}
simulations=${SKM_SIMULATIONS:-256}
simulation_levels=${SKM_SIMULATION_LEVELS:-}
negative_confirmations=${SKM_NEGATIVE_CONFIRMATIONS:-3}
target_positive_fraction=${SKM_TARGET_POSITIVE_FRACTION:-0.5}
strict_search_fraction=${SKM_STRICT_SEARCH_FRACTION:-0.5}
action_horizon=${SKM_ACTION_HORIZON:-128}
steps=${SKM_STEPS:-100000}
seed=${SKM_SEED:-20260815}

test -s "$catalogue"
test -s "$checkpoint"
test -s "$source_hashes"
if [[ -n "$distillation" ]]; then
  test -s "$distillation"
fi
if [[ -n "$rehearsal_panel" ]]; then
  test -s "$rehearsal_panel"
fi
cd "$repo"
sha256sum -c "$source_hashes"

resume=()
if [[ -s "$output/state.json" && -s "$output/scientist-state.pt.gz" ]]; then
  resume=(--resume)
elif [[ -e "$output/state.json" || -e "$output/scientist-state.pt.gz" ]]; then
  echo "refusing partial mastery state in $output" >&2
  exit 1
fi

distillation_args=()
if [[ -n "$distillation" ]]; then
  distillation_args=(--distillation "$distillation")
fi

rehearsal_args=()
if [[ -n "$rehearsal_panel" ]]; then
  rehearsal_args=(--rehearsal-panel "$rehearsal_panel")
fi

simulation_level_args=()
if [[ -n "$simulation_levels" ]]; then
  simulation_level_args=(--simulation-levels "$simulation_levels")
fi

exec taskset -c "$cores" nice -n 19 env \
  OMP_WAIT_POLICY=PASSIVE \
  GOMP_SPINCOUNT=0 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=src \
  "$python" -m pgx_mcts_bench.single_knot_mastery \
  --catalogue "$catalogue" \
  --knot "$knot" \
  --scientist "$scientist" \
  --checkpoint "$checkpoint" \
  --output "$output" \
  "${distillation_args[@]}" \
  "${rehearsal_args[@]}" \
  --parallel-searches "$parallel" \
  --torch-threads "$torch_threads" \
  --simulations "$simulations" \
  "${simulation_level_args[@]}" \
  --negative-confirmations "$negative_confirmations" \
  --target-positive-fraction "$target_positive_fraction" \
  --strict-search-fraction "$strict_search_fraction" \
  --action-horizon "$action_horizon" \
  --steps "$steps" \
  --seed "$seed" \
  "${resume[@]}"
