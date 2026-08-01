#!/usr/bin/env bash

# Overlap rung-27 promotion with the tail of rung 18.  Start this controller
# only after all but WAITING_RUNG18 candidates have completed rung 18.  It
# claims free physical cores under the same host lock as tail workers, never
# exceeds MAX_TOTAL_JOBS across rung-18 and rung-27 containers, and waits for
# the remaining rung-18 checkpoints to become eligible.

set -Eeuo pipefail

image=${IMAGE:?Set IMAGE to the published pgx-mcts-bench image}
candidates_file=${CANDIDATES_FILE:?Set CANDIDATES_FILE to one candidate per line}
output_root=${OUTPUT_ROOT:-/srv/braid/rung18}
required_logical_cpus=${REQUIRED_LOGICAL_CPUS:-32}
reserve_physical_cores=${RESERVE_PHYSICAL_CORES:-2}
max_total_jobs=${MAX_TOTAL_JOBS:-14}
waiting_rung18=${WAITING_RUNG18:-7}
load_limit=${LOAD_LIMIT:-28}
poll_seconds=${POLL_SECONDS:-5}
stop_after=${STOP_AFTER:-27}
max_iterations=${MAX_ITERATIONS:-100}
selfplay_games=${SELFPLAY_GAMES:-8}
eval_games=${EVAL_GAMES:-12}
retro_games=${RETRO_GAMES:-4}
crossing_tolerance=${CROSSING_TOLERANCE:-0.5}
collapse_floor=${COLLAPSE_FLOOR:-0.45}
max_consecutive_caps=${MAX_CONSECUTIVE_CAPS:-99}
min_iterations_per_rung=${MIN_ITERATIONS_PER_RUNG:-9}
min_iterations_from=${MIN_ITERATIONS_FROM:-10}
dry_run=${DRY_RUN:-0}

if [[ ! -r "$candidates_file" ]]; then
  printf 'Candidate file is not readable: %s\n' "$candidates_file" >&2
  exit 2
fi
if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  printf 'POLL_SECONDS must be a positive integer\n' >&2
  exit 2
fi
if (( stop_after != 27 )); then
  printf 'Refusing STOP_AFTER=%s: this controller is locked to rung 27\n' "$stop_after" >&2
  exit 2
fi

logical_cpus=$(nproc)
if (( logical_cpus != required_logical_cpus )); then
  printf 'Refusing to run: expected %s logical CPUs, found %s\n' \
    "$required_logical_cpus" "$logical_cpus" >&2
  exit 2
fi

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

physical_cores=${#core_keys[@]}
available_cores=$((physical_cores - reserve_physical_cores))
if (( reserve_physical_cores < 1 || available_cores < 1 )); then
  printf 'Invalid RESERVE_PHYSICAL_CORES=%s for %s physical cores\n' \
    "$reserve_physical_cores" "$physical_cores" >&2
  exit 2
fi
if (( max_total_jobs < 1 || max_total_jobs > available_cores )); then
  printf 'MAX_TOTAL_JOBS must be between 1 and %s\n' "$available_cores" >&2
  exit 2
fi

candidates=()
declare -A seen_candidates=()
while IFS= read -r line || [[ -n "$line" ]]; do
  candidate=${line%%#*}
  candidate=${candidate//[[:space:]]/}
  [[ -z "$candidate" ]] && continue
  if [[ ! "$candidate" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    printf 'Invalid candidate name: %s\n' "$candidate" >&2
    exit 2
  fi
  if [[ "$candidate" == search-light ]]; then
    printf 'search-light is permanently stopped and cannot enter rung 27\n' >&2
    exit 2
  fi
  if [[ -v "seen_candidates[$candidate]" ]]; then
    printf 'Duplicate candidate: %s\n' "$candidate" >&2
    exit 2
  fi
  seen_candidates[$candidate]=1
  candidates+=("$candidate")
done < "$candidates_file"
if (( ${#candidates[@]} == 0 )); then
  printf 'Candidate file contains no candidates\n' >&2
  exit 2
fi
if (( waiting_rung18 < 0 || waiting_rung18 >= ${#candidates[@]} )); then
  printf 'WAITING_RUNG18 must be between 0 and %s\n' \
    "$((${#candidates[@]} - 1))" >&2
  exit 2
fi

rung18_state="$output_root/state"
rung27_state="$output_root/state-r27"
rung27_logs="$output_root/logs-r27"
mkdir -p "$rung27_state" "$rung27_logs"

ready_count=0
for candidate in "${candidates[@]}"; do
  [[ -e "$rung18_state/$candidate.done" ]] && ready_count=$((ready_count + 1))
done
minimum_ready=$((${#candidates[@]} - waiting_rung18))
if (( ready_count < minimum_ready )); then
  printf 'Not ready: %s/%s candidates completed rung 18; need at least %s\n' \
    "$ready_count" "${#candidates[@]}" "$minimum_ready" >&2
  exit 2
fi

if [[ "$dry_run" == 1 ]]; then
  printf 'DRY RUN ready=%s/%s minimum_ready=%s max_total_jobs=%s\n' \
    "$ready_count" "${#candidates[@]}" "$minimum_ready" "$max_total_jobs"
  for candidate in "${candidates[@]}"; do
    if [[ -e "$rung18_state/$candidate.done" ]]; then
      checkpoint="$output_root/runs/$candidate/checkpoints/$candidate.pt"
      [[ -f "$checkpoint" ]] || {
        printf 'Eligible candidate lacks checkpoint: %s\n' "$checkpoint" >&2
        exit 2
      }
      printf 'READY %s %s\n' "$candidate" "$checkpoint"
    else
      printf 'WAIT  %s\n' "$candidate"
    fi
  done
  exit 0
fi

summary="$output_root/summary-r27.tsv"
run_env="$output_root/run-r27.env"
if [[ ! -e "$summary" ]]; then
  printf 'candidate\tcpuset\tstarted_at\tfinished_at\twall_seconds\texit_code\n' > "$summary"
fi
printf 'started_at=%s\nimage=%s\ncandidate_count=%s\nready_at_start=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$image" "${#candidates[@]}" "$ready_count" \
  > "$run_env"
printf 'waiting_rung18=%s\nmax_total_jobs=%s\nstop_after=%s\ndry_run=%s\n' \
  "$waiting_rung18" "$max_total_jobs" "$stop_after" "$dry_run" >> "$run_env"

declare -A candidate_by_pid=()
declare -A cpuset_by_pid=()
declare -A started_by_pid=()
declare -A started_epoch_by_pid=()
declare -A container_by_pid=()
declare -A attempted=()
active_pids=()
stop_requested=0

cleanup() {
  stop_requested=1
  for pid in "${active_pids[@]}"; do
    container=${container_by_pid[$pid]:-}
    [[ -n "$container" ]] || continue
    docker kill --signal TERM "$container" >/dev/null 2>&1 || true
  done
  deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    running=0
    for pid in "${active_pids[@]}"; do
      container=${container_by_pid[$pid]:-}
      [[ -n "$container" ]] || continue
      if [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" == true ]]; then
        running=1
      fi
    done
    (( running )) || return
    sleep 5
  done
  for pid in "${active_pids[@]}"; do
    container=${container_by_pid[$pid]:-}
    [[ -n "$container" ]] || continue
    docker kill --signal KILL "$container" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

load_is_safe() {
  local current
  current=$(cut -d ' ' -f 1 /proc/loadavg)
  awk -v current="$current" -v limit="$load_limit" \
    'BEGIN {exit !(current < limit)}'
}

# Print a free cpuset while holding the shared tail-worker lock. The caller
# keeps fd 9 locked until Docker has reserved the selected cpuset.
find_free_cpuset() {
  local status used slot offered worker_count=0
  declare -A occupied=()
  while IFS='|' read -r status used; do
    [[ "$status" == running || "$status" == created ]] || continue
    [[ -n "$used" ]] && occupied[$used]=1
  done < <(
    docker ps -aq \
      | xargs -r docker inspect -f '{{.State.Status}}|{{.HostConfig.CpusetCpus}}'
  )
  while IFS= read -r name; do
    [[ "$name" == braid-r18-* || "$name" == braid-r27-* ]] \
      && worker_count=$((worker_count + 1))
  done < <(docker ps -a --filter status=running --filter status=created --format '{{.Names}}')
  (( worker_count < max_total_jobs )) || return 1
  for ((slot = reserve_physical_cores; slot < physical_cores; slot++)); do
    offered=${cpus_by_core[${core_keys[$slot]}]}
    if [[ ! -v "occupied[$offered]" ]]; then
      printf '%s\n' "$offered"
      return 0
    fi
  done
  return 1
}

start_candidate() {
  local candidate=$1 cpuset container started started_epoch log_path
  container="braid-r27-${candidate//[^a-zA-Z0-9_.-]/-}"
  log_path="$rung27_logs/$candidate.log"

  exec 9>/run/lock/braid-rung18-tail.lock
  flock -x 9
  if ! cpuset=$(find_free_cpuset) || ! load_is_safe; then
    flock -u 9
    return 1
  fi
  rm -f "$rung27_state/$candidate.failed" "$rung27_state/$candidate.interrupted"
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
    --selfplay-games "$selfplay_games" \
    --max-iterations "$max_iterations" \
    --eval-games "$eval_games" \
    --retro-games "$retro_games" \
    --crossing-tolerance "$crossing_tolerance" \
    --collapse-floor "$collapse_floor" \
    --max-consecutive-caps "$max_consecutive_caps" \
    --min-iterations-per-rung "$min_iterations_per_rung" \
    --min-iterations-from "$min_iterations_from" \
    --checkpoint-every 1 \
    --stop-after 27 \
    --bounds artifacts/nebius-rung18/bounds.jsonl \
    --output "artifacts/nebius-rung18/runs/$candidate" \
    >/dev/null
  flock -u 9

  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  started_epoch=$(date +%s)
  (
    set +e
    docker start --attach "$container" > "$log_path" 2>&1
    exit $?
  ) &
  pid=$!
  attempted[$candidate]=1
  candidate_by_pid[$pid]=$candidate
  cpuset_by_pid[$pid]=$cpuset
  started_by_pid[$pid]=$started
  started_epoch_by_pid[$pid]=$started_epoch
  container_by_pid[$pid]=$container
  active_pids+=("$pid")
  printf 'started candidate=%s pid=%s cpuset=%s own_active=%s total_limit=%s\n' \
    "$candidate" "$pid" "$cpuset" "${#active_pids[@]}" "$max_total_jobs"
}

reap_finished() {
  local survivors=() pid candidate code finished wall_seconds container
  for pid in "${active_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      survivors+=("$pid")
      continue
    fi
    candidate=${candidate_by_pid[$pid]}
    container=${container_by_pid[$pid]}
    if wait "$pid"; then code=0; else code=$?; fi
    finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    wall_seconds=$(($(date +%s) - started_epoch_by_pid[$pid]))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$candidate" "${cpuset_by_pid[$pid]}" \
      "${started_by_pid[$pid]}" "$finished" "$wall_seconds" "$code" >> "$summary"
    docker rm "$container" >/dev/null 2>&1 || true
    if (( code == 0 )); then
      touch "$rung27_state/$candidate.done"
    elif (( code == 143 )); then
      touch "$rung27_state/$candidate.interrupted"
    else
      touch "$rung27_state/$candidate.failed"
    fi
    printf 'finished candidate=%s exit=%s own_active=%s\n' \
      "$candidate" "$code" "${#survivors[@]}"
    unset 'candidate_by_pid[$pid]' 'cpuset_by_pid[$pid]'
    unset 'started_by_pid[$pid]' 'started_epoch_by_pid[$pid]'
    unset 'container_by_pid[$pid]'
  done
  active_pids=("${survivors[@]}")
}

all_done() {
  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -e "$rung27_state/$candidate.done" ]] || return 1
  done
  return 0
}

all_terminal() {
  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -e "$rung27_state/$candidate.done" || -e "$rung27_state/$candidate.failed" ]] \
      || return 1
  done
  return 0
}

while ! all_terminal || (( ${#active_pids[@]} > 0 )); do
  reap_finished
  started_one=0
  if (( ! stop_requested )); then
    for candidate in "${candidates[@]}"; do
      [[ -e "$rung27_state/$candidate.done" ]] && continue
      [[ -v "attempted[$candidate]" ]] && continue
      [[ -e "$rung18_state/$candidate.done" ]] || continue
      checkpoint="$output_root/runs/$candidate/checkpoints/$candidate.pt"
      if [[ ! -f "$checkpoint" ]]; then
        printf 'eligible candidate lacks checkpoint: %s\n' "$checkpoint" >&2
        attempted[$candidate]=1
        touch "$rung27_state/$candidate.failed"
        continue
      fi
      if start_candidate "$candidate"; then
        started_one=1
      else
        break
      fi
    done
  fi
  if (( stop_requested && ${#active_pids[@]} == 0 )); then
    break
  fi
  if ! all_terminal || (( ${#active_pids[@]} > 0 )); then
    sleep "$poll_seconds"
  elif (( started_one == 0 )); then
    break
  fi
done

printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$run_env"
if (( stop_requested )); then
  printf 'Queue stopped; rerun the same command to resume incomplete candidates.\n' >&2
  exit 130
fi
if ! all_done; then
  printf 'One or more rung-27 candidates failed; inspect %s\n' "$rung27_state" >&2
  exit 1
fi
