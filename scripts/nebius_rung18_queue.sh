#!/usr/bin/env bash

# Run one resumable ladder process per physical CPU core.  This is deliberately
# a host queue around single-candidate containers: sharing one ladder output
# between concurrent workers races on ladder.json, while per-candidate roots can
# be merged safely after every worker has stopped.

set -Eeuo pipefail

image=${IMAGE:?Set IMAGE to the published pgx-mcts-bench image}
candidates_file=${CANDIDATES_FILE:?Set CANDIDATES_FILE to one candidate name per line}
output_root=${OUTPUT_ROOT:-/srv/braid/artifacts/nebius-rung18}
required_logical_cpus=${REQUIRED_LOGICAL_CPUS:-32}
reserve_physical_cores=${RESERVE_PHYSICAL_CORES:-2}
max_jobs=${MAX_JOBS:-0}
load_limit=${LOAD_LIMIT:-0}
poll_seconds=${POLL_SECONDS:-5}
stop_after=${STOP_AFTER:-18}
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

logical_cpus=$(nproc)
if (( logical_cpus != required_logical_cpus )); then
  printf 'Refusing to run: expected %s logical CPUs, found %s\n' \
    "$required_logical_cpus" "$logical_cpus" >&2
  exit 2
fi

# Build cpusets by physical core.  Each job owns both SMT siblings, while its
# numerical libraries remain single-threaded.  The sibling is available for
# runtime/helper threads without letting a candidate spill across the machine.
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
if (( reserve_physical_cores < 1 || reserve_physical_cores >= physical_cores )); then
  printf 'RESERVE_PHYSICAL_CORES must be between 1 and %s\n' \
    "$((physical_cores - 1))" >&2
  exit 2
fi
available_cores=$((physical_cores - reserve_physical_cores))
if (( max_jobs == 0 )); then
  max_jobs=$available_cores
fi
if (( max_jobs < 1 || max_jobs > available_cores )); then
  printf 'MAX_JOBS must be between 1 and %s\n' "$available_cores" >&2
  exit 2
fi
if (( load_limit == 0 )); then
  # Leave the reserved cores genuinely available to Docker, ssh and monitoring.
  load_limit=$((logical_cpus - 2 * reserve_physical_cores))
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

mkdir -p "$output_root/runs" "$output_root/logs" "$output_root/state"
for candidate in "${candidates[@]}"; do
  promoted="$output_root/runs/$candidate/checkpoints/$candidate.pt"
  progress="$output_root/runs/$candidate/checkpoints/$candidate/progress.pt"
  interrupt="$output_root/runs/$candidate/checkpoints/$candidate/interrupt.pt"
  if [[ ! -f "$promoted" && ! -f "$progress" && ! -f "$interrupt" ]]; then
    printf 'Refusing a fresh run: no checkpoint for %s under %s\n' \
      "$candidate" "$output_root/runs/$candidate/checkpoints" >&2
    exit 2
  fi
done
summary="$output_root/summary.tsv"
run_env="$output_root/run.env"
if [[ ! -e "$summary" ]]; then
  printf 'candidate\tcpuset\tstarted_at\tfinished_at\twall_seconds\texit_code\n' > "$summary"
fi
printf 'started_at=%s\nimage=%s\nlogical_cpus=%s\nphysical_cores=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$image" "$logical_cpus" "$physical_cores" \
  > "$run_env"
printf 'reserved_physical_cores=%s\nmax_jobs=%s\nload_limit=%s\nstop_after=%s\n' \
  "$reserve_physical_cores" "$max_jobs" "$load_limit" "$stop_after" >> "$run_env"
printf 'candidate_count=%s\ndry_run=%s\nmax_iterations=%s\n' \
  "${#candidates[@]}" "$dry_run" "$max_iterations" >> "$run_env"
printf 'max_consecutive_caps=%s\nmin_iterations_per_rung=%s\nmin_iterations_from=%s\n' \
  "$max_consecutive_caps" "$min_iterations_per_rung" "$min_iterations_from" >> "$run_env"

declare -A candidate_by_pid=()
declare -A cpuset_by_pid=()
declare -A slot_by_pid=()
declare -A started_by_pid=()
declare -A started_epoch_by_pid=()
declare -A container_by_pid=()
active_pids=()
next_candidate=0
overall=0
stop_requested=0

cleanup() {
  stop_requested=1
  for pid in "${active_pids[@]}"; do
    [[ -n "${container_by_pid[$pid]:-}" ]] || continue
    docker kill --signal TERM "${container_by_pid[$pid]}" >/dev/null 2>&1 || true
  done
  # All candidates receive TERM together. Give safe-boundary checkpoints time
  # to finish, then force only containers that ignored the graceful request.
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
    [[ -n "${container_by_pid[$pid]:-}" ]] || continue
    docker kill --signal KILL "${container_by_pid[$pid]}" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

load_is_safe() {
  local current
  current=$(cut -d ' ' -f 1 /proc/loadavg)
  awk -v current="$current" -v limit="$load_limit" 'BEGIN {exit !(current < limit)}'
}

next_free_slot() {
  local slot pid occupied
  for ((slot = 0; slot < max_jobs; slot++)); do
    occupied=0
    for pid in "${active_pids[@]}"; do
      if [[ "${slot_by_pid[$pid]:-}" == "$slot" ]]; then
        occupied=1
        break
      fi
    done
    if (( ! occupied )); then
      printf '%s\n' "$slot"
      return 0
    fi
  done
  return 1
}

start_candidate() {
  local candidate=$1 slot=$2 cpuset container started started_epoch log_path
  cpuset=${cpus_by_core[${core_keys[$((slot + reserve_physical_cores))]}]}
  container="braid-r18-${candidate//[^a-zA-Z0-9_.-]/-}"
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  started_epoch=$(date +%s)
  log_path="$output_root/logs/$candidate.log"

  if [[ "$dry_run" == 1 ]]; then
    printf 'DRY RUN candidate=%s cpuset=%s container=%s\n' "$candidate" "$cpuset" "$container"
    return
  fi

  (
    set +e
    docker run --rm \
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
      --stop-after "$stop_after" \
      --bounds artifacts/nebius-rung18/bounds.jsonl \
      --output "artifacts/nebius-rung18/runs/$candidate" \
      > "$log_path" 2>&1
    code=$?
    if (( code == 0 )); then
      touch "$output_root/state/$candidate.done"
    elif (( code == 143 )); then
      touch "$output_root/state/$candidate.interrupted"
    else
      touch "$output_root/state/$candidate.failed"
    fi
    exit "$code"
  ) &
  pid=$!
  candidate_by_pid[$pid]=$candidate
  cpuset_by_pid[$pid]=$cpuset
  slot_by_pid[$pid]=$slot
  started_by_pid[$pid]=$started
  started_epoch_by_pid[$pid]=$started_epoch
  container_by_pid[$pid]=$container
  active_pids+=("$pid")
  printf 'started candidate=%s pid=%s cpuset=%s active=%s/%s\n' \
    "$candidate" "$pid" "$cpuset" "${#active_pids[@]}" "$max_jobs"
}

reap_finished() {
  local survivors=() pid candidate code finished finished_epoch wall_seconds
  for pid in "${active_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      survivors+=("$pid")
      continue
    fi
    candidate=${candidate_by_pid[$pid]}
    if wait "$pid"; then code=0; else code=$?; overall=1; fi
    finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    finished_epoch=$(date +%s)
    wall_seconds=$((finished_epoch - started_epoch_by_pid[$pid]))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$candidate" "${cpuset_by_pid[$pid]}" \
      "${started_by_pid[$pid]}" "$finished" "$wall_seconds" "$code" >> "$summary"
    printf 'finished candidate=%s exit=%s active=%s/%s\n' \
      "$candidate" "$code" "${#survivors[@]}" "$max_jobs"
    unset 'candidate_by_pid[$pid]' 'cpuset_by_pid[$pid]' 'slot_by_pid[$pid]'
    unset 'started_by_pid[$pid]' 'started_epoch_by_pid[$pid]'
    unset 'container_by_pid[$pid]'
  done
  active_pids=("${survivors[@]}")
}

while (( next_candidate < ${#candidates[@]} || ${#active_pids[@]} > 0 )); do
  reap_finished
  while (( ! stop_requested && next_candidate < ${#candidates[@]} \
           && ${#active_pids[@]} < max_jobs )); do
    candidate=${candidates[$next_candidate]}
    next_candidate=$((next_candidate + 1))
    if [[ -e "$output_root/state/$candidate.done" ]]; then
      printf 'skipped completed candidate=%s\n' "$candidate"
      continue
    fi
    if ! load_is_safe; then
      next_candidate=$((next_candidate - 1))
      break
    fi
    rm -f "$output_root/state/$candidate.failed" "$output_root/state/$candidate.interrupted"
    slot=$(next_free_slot)
    start_candidate "$candidate" "$slot"
  done
  if (( stop_requested && ${#active_pids[@]} == 0 )); then
    break
  fi
  if (( ${#active_pids[@]} > 0 || next_candidate < ${#candidates[@]} )); then
    sleep "$poll_seconds"
  fi
done

printf 'finished_at=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$overall" \
  >> "$run_env"
if (( stop_requested )); then
  printf 'Queue stopped; rerun the same command to resume incomplete candidates.\n' >&2
  exit 130
fi
exit "$overall"
