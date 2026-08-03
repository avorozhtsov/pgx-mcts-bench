#!/usr/bin/env bash
set -euo pipefail

artifact_root=${1:-artifacts/collaboration-pilot-200-cpu32}
max_hours=${MAX_HOURS:-70}
hourly_usd=${HOURLY_USD:-0.7936}
min_free_gib=${MIN_FREE_GIB:-40}

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required on the CPU-32 host" >&2
  exit 2
fi

estimated=$(awk -v hours="$max_hours" -v hourly="$hourly_usd" \
  'BEGIN { printf "%.2f", hours * hourly }')
echo "workload timeout: ${max_hours}h; estimated VM compute: USD $estimated"

mkdir -p "$artifact_root"
available_kib=$(df -Pk "$artifact_root" | awk 'NR == 2 { print $4 }')
required_kib=$((min_free_gib * 1024 * 1024))
if (( available_kib < required_kib )); then
  echo "need at least ${min_free_gib} GiB free for compressed transactions" >&2
  df -h "$artifact_root" >&2
  exit 2
fi

status=0
timeout --signal=TERM --kill-after=10m "${max_hours}h" \
  scripts/run_collaboration_pilot.sh cpu32 "$artifact_root" || status=$?
if [[ $status -eq 124 ]]; then
  echo "workload stopped at the configured time limit"
elif [[ $status -ne 0 ]]; then
  echo "workload failed with status $status" >&2
fi

scripts/monitor_collaboration_pilot.sh "$artifact_root"

if [[ ${SHUTDOWN_WHEN_DONE:-0} == 1 ]]; then
  echo "SHUTDOWN_WHEN_DONE=1: halting the VM to stop compute billing"
  sudo shutdown -h now
fi
exit "$status"
