#!/usr/bin/env bash
set -euo pipefail

repo=${EVIDENCE_COLLECTOR_REPO:-/srv/braid/work-evidence-metadata-collector-v1/pgx-mcts-bench}
python=${EVIDENCE_COLLECTOR_PYTHON:-/srv/braid/work/pgx-mcts-bench/.venv/bin/python}
output=${EVIDENCE_COLLECTOR_OUTPUT:-/srv/braid/artifacts/evidence-catalog-v1}
database=${EVIDENCE_COLLECTOR_DATABASE:-$output/evidence.sqlite3}
lock=${EVIDENCE_COLLECTOR_LOCK:-$output/collector.lock}
metadata_limit=${EVIDENCE_COLLECTOR_METADATA_LIMIT:-8}
max_full_strands=${EVIDENCE_COLLECTOR_MAX_FULL_STRANDS:-8}

mkdir -p "$output"
exec 9>"$lock"
flock -n 9 || exit 0

experiment_args=()
IFS=: read -r -a experiment_roots <<< "${EVIDENCE_COLLECTOR_EXPERIMENT_ROOTS:-}"
for root in "${experiment_roots[@]}"; do
  [[ -n "$root" ]] && experiment_args+=(--experiment-root "$root")
done

mastery_args=()
IFS=: read -r -a mastery_inventories <<< "${EVIDENCE_COLLECTOR_MASTERY_INVENTORIES:-}"
for inventory in "${mastery_inventories[@]}"; do
  [[ -n "$inventory" ]] && mastery_args+=(--mastery-inventory "$inventory")
done

external_args=()
IFS=: read -r -a external_evidence <<< "${EVIDENCE_COLLECTOR_EXTERNAL_EVIDENCE:-}"
for package in "${external_evidence[@]}"; do
  [[ -n "$package" ]] && external_args+=(--external-evidence "$package")
done

cd "$repo"
exec nice -n 19 env PYTHONUNBUFFERED=1 PYTHONPATH=src \
  "$python" -m pgx_mcts_bench.evidence_catalog \
  --database "$database" \
  --output "$output" \
  --metadata-limit "$metadata_limit" \
  --max-full-invariant-strands "$max_full_strands" \
  "${experiment_args[@]}" \
  "${mastery_args[@]}" \
  "${external_args[@]}"
