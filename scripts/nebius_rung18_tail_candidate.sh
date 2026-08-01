#!/usr/bin/env bash

# Add one candidate to an already-running full rung-18 queue without disturbing
# its containers. Wait until the original controller has dispatched its complete
# list, then claim the first physical-core slot it leaves free.

set -Eeuo pipefail

image=${IMAGE:?Set IMAGE to an image containing the candidate}
candidate=${CANDIDATE:?Set CANDIDATE}
output_root=${OUTPUT_ROOT:-/srv/braid/rung18}
wait_for_logs=${WAIT_FOR_LOGS:-}
wait_for_running_or_done=${WAIT_FOR_RUNNING_OR_DONE:-}
poll_seconds=${POLL_SECONDS:-10}
load_limit=${LOAD_LIMIT:-28}
reserve_physical_cores=${RESERVE_PHYSICAL_CORES:-2}
stop_after=${STOP_AFTER:-18}
max_iterations=${MAX_ITERATIONS:-100}

checkpoint="$output_root/runs/$candidate/checkpoints/$candidate.pt"
if [[ ! -f "$checkpoint" ]]; then
  printf 'Missing initial checkpoint: %s\n' "$checkpoint" >&2
  exit 2
fi

IFS=, read -r -a prerequisite_logs <<< "$wait_for_logs"
for prerequisite in "${prerequisite_logs[@]}"; do
  [[ -z "$prerequisite" ]] && continue
  while [[ ! -e "$output_root/logs/$prerequisite.log" ]]; do
    sleep "$poll_seconds"
  done
done

IFS=, read -r -a prerequisite_candidates <<< "$wait_for_running_or_done"
for prerequisite in "${prerequisite_candidates[@]}"; do
  [[ -z "$prerequisite" ]] && continue
  while ! docker inspect "braid-r18-$prerequisite" >/dev/null 2>&1 \
      && [[ ! -e "$output_root/state/$prerequisite.done" ]] \
      && [[ ! -e "$output_root/state/$prerequisite.failed" ]] \
      && [[ ! -e "$output_root/state/$prerequisite.interrupted" ]]; do
    sleep "$poll_seconds"
  done
done

declare -A cpus_by_core=()
core_keys=()
while IFS=, read -r cpu core socket; do
  [[ "$cpu" == \#* || -z "$cpu" ]] && continue
  key="$socket:$core"
  if [[ -v "cpus_by_core[$key]" ]]; then
    cpus_by_core[$key]="${cpus_by_core[$key]},$cpu"
  else
    cpus_by_core[$key]="$cpu"
    core_keys+=("$key")
  fi
done < <(lscpu -p=CPU,CORE,SOCKET)

container="braid-r18-$candidate"
log_path="$output_root/logs/$candidate.log"
created=0

cleanup() {
  docker kill --signal TERM "$container" >/dev/null 2>&1 || true
}
terminate() {
  cleanup
  exit 143
}
trap terminate INT TERM

# The original queue does not use this lock, which is why tail candidates wait
# for its full list to be dispatched first. Tail candidates do share the lock:
# reserve a Docker container while holding it, so the next tail sees the cpuset
# even before the first container has started executing Python.
exec 9>/run/lock/braid-rung18-tail.lock
cpuset=""
while [[ -z "$cpuset" ]]; do
  flock -x 9
  declare -A occupied=()
  while IFS= read -r used; do
    [[ -n "$used" ]] && occupied[$used]=1
  done < <(
    docker ps -aq --filter name=braid-r18- \
      | xargs -r docker inspect -f '{{.HostConfig.CpusetCpus}}'
  )
  for ((slot = reserve_physical_cores; slot < ${#core_keys[@]}; slot++)); do
    offered=${cpus_by_core[${core_keys[$slot]}]}
    if [[ ! -v "occupied[$offered]" ]]; then
      cpuset=$offered
      break
    fi
  done
  current_load=$(cut -d ' ' -f 1 /proc/loadavg)
  if ! awk -v current="$current_load" -v limit="$load_limit" \
      'BEGIN {exit !(current < limit)}'; then
    cpuset=""
  fi
  if [[ -n "$cpuset" ]]; then
    docker create \
      --name "$container" \
      --cpuset-cpus "$cpuset" \
      --env OMP_NUM_THREADS=1 \
      --env MKL_NUM_THREADS=1 \
      --env OPENBLAS_NUM_THREADS=1 \
      --env NUMEXPR_NUM_THREADS=1 \
      --env PYTHONUNBUFFERED=1 \
      --volume "$output_root:/work/pgx-mcts-bench/artifacts/nebius-rung18" \
      "$image" \
      braid-ladder \
      --only "$candidate" \
      --workers 1 \
      --device cpu \
      --selfplay-games 8 \
      --max-iterations "$max_iterations" \
      --eval-games 12 \
      --retro-games 4 \
      --crossing-tolerance 0.5 \
      --collapse-floor 0.45 \
      --max-consecutive-caps 99 \
      --min-iterations-per-rung 9 \
      --min-iterations-from 10 \
      --checkpoint-every 1 \
      --stop-after "$stop_after" \
      --bounds artifacts/nebius-rung18/bounds.jsonl \
      --output "artifacts/nebius-rung18/runs/$candidate" \
      >/dev/null
    created=1
  fi
  flock -u 9
  [[ -n "$cpuset" ]] || sleep "$poll_seconds"
done

started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
started_epoch=$(date +%s)

set +e
docker start --attach "$container" > "$log_path" 2>&1
code=$?
set -e
if (( created )); then
  docker rm "$container" >/dev/null
fi

finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
wall_seconds=$(($(date +%s) - started_epoch))
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$candidate" "$cpuset" "$started" "$finished" "$wall_seconds" "$code" \
  >> "$output_root/summary.tsv"
if (( code == 0 )); then
  touch "$output_root/state/$candidate.done"
elif (( code == 143 )); then
  touch "$output_root/state/$candidate.interrupted"
else
  touch "$output_root/state/$candidate.failed"
fi
exit "$code"
