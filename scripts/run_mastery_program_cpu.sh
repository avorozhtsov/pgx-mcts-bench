#!/usr/bin/env bash
set -euo pipefail

repo=${MP_REPO:-/srv/braid/work-multi-knot-mastery-v1/pgx-mcts-bench}
sequence=${MP_SEQUENCE:?set MP_SEQUENCE to a pinned 200+ challenge sequence}
checkpoint=${MP_CHECKPOINT:?set MP_CHECKPOINT to a pinned inherited checkpoint}
output=${MP_OUTPUT:?set MP_OUTPUT to a new or resumable scientist directory}
inventory=${MP_EVIDENCE_INVENTORY:?set MP_EVIDENCE_INVENTORY}
scientist=${MP_SCIENTIST:?set MP_SCIENTIST}
reservoir=${MP_CHALLENGE_RESERVOIR:-}
rehearsal_panel=${MP_REHEARSAL_PANEL:-}
source_hashes=${MP_SOURCE_HASHES:-$output/source-hashes.sha256}
python=${MP_PYTHON:-/srv/braid/work/pgx-mcts-bench/.venv/bin/python}
cores=${MP_CORES:-0-7}
parallel=${MP_PARALLEL_SEARCHES:-2}
torch_threads=${MP_TORCH_THREADS:-2}
simulations=${MP_SIMULATIONS:-128}
action_horizon=${MP_ACTION_HORIZON:-256}
heap_size=${MP_HEAP_SIZE:-200}
challenge_attempt_limit=${MP_CHALLENGE_ATTEMPT_LIMIT:-24}
challenge_seconds_limit=${MP_CHALLENGE_SECONDS_LIMIT:-900}
steps=${MP_STEPS:-100000}
seed=${MP_SEED:-20260815}

test -s "$sequence"
test -s "$checkpoint"
test -s "$source_hashes"
if [[ -n "$reservoir" ]]; then
  test -s "$reservoir"
fi
if [[ -n "$rehearsal_panel" ]]; then
  test -s "$rehearsal_panel"
fi

cd "$repo"
sha256sum -c "$source_hashes"

resume=()
if [[ -s "$output/program-state.json" && -s "$output/scientist-state.pt.gz" ]]; then
  resume=(--resume)
elif [[ -e "$output/program-state.json" || -e "$output/scientist-state.pt.gz" ]]; then
  echo "refusing partial mastery-program state in $output" >&2
  exit 1
fi

reservoir_args=()
if [[ -n "$reservoir" ]]; then
  reservoir_args=(--challenge-reservoir "$reservoir")
fi

rehearsal_args=()
if [[ -n "$rehearsal_panel" ]]; then
  rehearsal_args=(--rehearsal-panel "$rehearsal_panel")
fi

exec taskset -c "$cores" nice -n 19 env \
  OMP_WAIT_POLICY=PASSIVE \
  GOMP_SPINCOUNT=0 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=src \
  "$python" -m pgx_mcts_bench.mastery_program \
  --sequence "$sequence" \
  "${reservoir_args[@]}" \
  --scientist "$scientist" \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --evidence-inventory "$inventory" \
  "${rehearsal_args[@]}" \
  --parallel-searches "$parallel" \
  --torch-threads "$torch_threads" \
  --simulations "$simulations" \
  --action-horizon "$action_horizon" \
  --max-heap "$heap_size" \
  --challenge-attempt-limit "$challenge_attempt_limit" \
  --challenge-seconds-limit "$challenge_seconds_limit" \
  --steps "$steps" \
  --seed "$seed" \
  "${resume[@]}"
