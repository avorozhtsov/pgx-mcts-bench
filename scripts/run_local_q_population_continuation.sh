#!/usr/bin/env bash
set -euo pipefail

repo=${LOCAL_ABLATION_REPO:-/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation}
run=${LOCAL_ABLATION_RUN:-/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817}
archive=${LOCAL_Q4000_ARCHIVE:-/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/nebius-main32-final-20260817/artifacts/q4000-strand12-20260814}
qroot=$run/inputs/q4000-v1
current_q40=$run/continuation/q4000-v1-exploratory-q40-1-20260818/q40-1-static-no-sharing
output_root=$run/continuation/q4000-v1-population-20260818
recovery=$output_root/q20-recovery
tail=$output_root/q20-recovery-tail-static-no-sharing
q40=$output_root/q40-1-static-no-sharing
source_state=$archive/q20-static-no-sharing/state.pt.gz
source_selection=$repo/research/local-q-skm-ablation/q4000-source-four-selection.json
population_selection=$repo/research/local-q-skm-ablation/q4000-population-three-selection.json

export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

verify_hash() {
  local expected=$1
  local path=$2
  local observed
  observed=$(shasum -a 256 "$path" | awk '{print $1}')
  if [[ "$observed" != "$expected" ]]; then
    echo "hash mismatch for $path: expected $expected, observed $observed" >&2
    exit 2
  fi
}

audit_allow_capability_failures() {
  local group=$1
  local output=$2
  local status=0
  uv run python scripts/audit_local_q4000_group.py \
    --group "$group" --output "$output" || status=$?
  if (( status == 0 )); then
    return
  fi
  uv run python - "$output/local-group-audit.json" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1]))
allowed = (
    "retention ",
    "no native self-play success on a 6+ strand representation",
)
unexpected = [
    failure
    for failure in audit.get("failures", [])
    if not any(str(failure).startswith(prefix) for prefix in allowed)
]
if unexpected:
    raise SystemExit(f"unexpected structural audit failures: {unexpected}")
PY
}

scientist_args=(
  --scientist "raster-axial-12=$archive/migrated/raster-axial-12/checkpoint.pt"
  --scientist "raster-invariant-combined-dual-12=$archive/migrated/raster-invariant-combined-dual-12/checkpoint.pt"
  --scientist "strand-graph-12=$archive/migrated/strand-graph-12/checkpoint.pt"
)

cd "$repo"
mkdir -p "$output_root/protocol" "$recovery" "$output_root/source-exports" "$output_root/tail-exports"
cp research/local-q-skm-ablation/q4000-population-continuation-20260818.json "$output_root/protocol/"
echo "$(date -u +%FT%TZ) Q60 population queue started; existing v3 Q40 remains immutable"

verify_hash e9a25901ee49373d10ec07ea5b1cccddc7556947f269f4e304fe6f06f8eb89ae "$qroot/q20.json"
verify_hash 488340fbfd925f1e73c68cc8eb0e46c0c789315711bfd75af7623b49461ea879 "$qroot/q40-1.json"
verify_hash db526211dd3a737b4c36b67484e049c0b83982479fac6f125ae5c6e3b66ad42d "$qroot/prior-q40-1.json"
verify_hash 94b58cc4ffd165bb5a3a3e35f948296efc28f1445a8ddd609ac8690a93a561dd "$source_state"
verify_hash ab43f8b45fd019c27984a05dc82fea9de639b63c6bc247b5b6bbd8746b58f8a3 "$archive/migrated/raster-axial-12/checkpoint.pt"
verify_hash eddb4715abe700239148b5825a862d98947d1478bae0d2f6aef2f0a1f08ef6d7 "$archive/migrated/raster-invariant-combined-dual-12/checkpoint.pt"
verify_hash daeb70bf342602f50534a0435e1587344573449c608ae4ab7756beba9f27da4d "$archive/migrated/strand-graph-12/checkpoint.pt"

uv run python scripts/build_q20_recovery_tail.py \
  --source-bank "$qroot/q20.json" \
  --source-state "$source_state" \
  --tail-bank "$recovery/q20-tail.json" \
  --prior-bank "$recovery/prior-q20-tail.json" \
  --manifest "$recovery/manifest.json"

if [[ ! -f "$output_root/source-exports/raster-axial-12/state.pt.gz" ]]; then
  uv run python scripts/export_sv2_scientist_states.py \
    "$source_state" "$source_selection" "$output_root/source-exports"
fi

tail_initial=(
  --initial-state "raster-axial-12=$output_root/source-exports/raster-axial-12/state.pt.gz"
  --initial-state "raster-invariant-combined-dual-12=$output_root/source-exports/raster-invariant-combined-dual-12/state.pt.gz"
  --initial-state "strand-graph-12=$output_root/source-exports/strand-graph-12/state.pt.gz"
)

if [[ ! -f "$tail/report.json" ]]; then
  resume_args=()
  if [[ -f "$tail/manifest.json" ]]; then
    resume_args=(--resume)
  fi
  uv run pgx-mcts-bench braid-sv2-coordinated \
    --output "$tail" \
    --bank "$recovery/q20-tail.json" \
    --prior-bank "$recovery/prior-q20-tail.json" \
    "${scientist_args[@]}" \
    "${tail_initial[@]}" \
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
    --retention-target 0.8 \
    --action-horizon 128 \
    --rungs 0 \
    --seed 2026081831 \
    --torch-threads 1 \
    --parallel-scientists \
    --adaptive-compute \
    --device cpu \
    "${resume_args[@]}"
fi
audit_allow_capability_failures "$recovery/q20-tail.json" "$tail"

if [[ ! -f "$output_root/tail-exports/raster-axial-12/state.pt.gz" ]]; then
  uv run python scripts/export_sv2_scientist_states.py \
    "$tail/state.pt.gz" "$population_selection" "$output_root/tail-exports"
fi

q40_initial=(
  --initial-state "raster-axial-12=$output_root/tail-exports/raster-axial-12/state.pt.gz"
  --initial-state "raster-invariant-combined-dual-12=$output_root/tail-exports/raster-invariant-combined-dual-12/state.pt.gz"
  --initial-state "strand-graph-12=$output_root/tail-exports/strand-graph-12/state.pt.gz"
)
if [[ ! -f "$q40/report.json" ]]; then
  resume_args=()
  if [[ -f "$q40/manifest.json" ]]; then
    resume_args=(--resume)
  fi
  uv run pgx-mcts-bench braid-sv2-coordinated \
    --output "$q40" \
    --bank "$qroot/q40-1.json" \
    --prior-bank "$qroot/prior-q40-1.json" \
    "${scientist_args[@]}" \
    "${q40_initial[@]}" \
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
    --retention-target 0.8 \
    --action-horizon 128 \
    --rungs 0 \
    --seed 2026081832 \
    --torch-threads 1 \
    --parallel-scientists \
    --adaptive-compute \
    --device cpu \
    "${resume_args[@]}"
fi
audit_allow_capability_failures "$qroot/q40-1.json" "$q40"
touch "$output_root/Q_GROWN_THREE_Q60_COMPLETE"

run_full_branch_to_q60() {
  local lineage=$1
  local scientist=$2
  local checkpoint=$3
  local initial_state=$4
  local simulations=$5
  local seed=$6
  local selection=$7
  local branch=$output_root/branches/$lineage
  local q20_output=$branch/q20-static-no-sharing
  local q20_export=$branch/q20-export
  local q40_output=$branch/q40-1-static-no-sharing
  mkdir -p "$branch" "$q20_export"

  if [[ ! -f "$q20_output/report.json" ]]; then
    local resume_args=()
    if [[ -f "$q20_output/manifest.json" ]]; then
      resume_args=(--resume)
    fi
    uv run pgx-mcts-bench braid-sv2-coordinated \
      --output "$q20_output" \
      --bank "$qroot/q20.json" \
      --scientist "$scientist=$checkpoint" \
      --initial-state "$scientist=$initial_state" \
      --arm static-no-sharing \
      --ratios 10,1000 \
      --simulations "$simulations" \
      --qualification-simulations "$simulations" \
      --qualification-attempts 1 \
      --f-native 5 \
      --selfplay-games 8 \
      --train-steps 96 \
      --batch-size 64 \
      --evaluation-attempts 4 \
      --block-size 10 \
      --retention-target 0.8 \
      --action-horizon 128 \
      --rungs 0 \
      --seed "$seed" \
      --torch-threads 1 \
      --no-parallel-scientists \
      --adaptive-compute \
      --device cpu \
      "${resume_args[@]}"
  fi
  audit_allow_capability_failures "$qroot/q20.json" "$q20_output"

  if [[ ! -f "$q20_export/$scientist/state.pt.gz" ]]; then
    uv run python scripts/export_sv2_scientist_states.py \
      "$q20_output/state.pt.gz" "$selection" "$q20_export"
  fi
  if [[ ! -f "$q40_output/report.json" ]]; then
    local resume_args=()
    if [[ -f "$q40_output/manifest.json" ]]; then
      resume_args=(--resume)
    fi
    uv run pgx-mcts-bench braid-sv2-coordinated \
      --output "$q40_output" \
      --bank "$qroot/q40-1.json" \
      --prior-bank "$qroot/prior-q40-1.json" \
      --scientist "$scientist=$checkpoint" \
      --initial-state "$scientist=$q20_export/$scientist/state.pt.gz" \
      --arm static-no-sharing \
      --ratios 10,1000 \
      --simulations "$simulations" \
      --qualification-simulations "$simulations" \
      --qualification-attempts 1 \
      --f-native 5 \
      --selfplay-games 8 \
      --train-steps 96 \
      --batch-size 64 \
      --evaluation-attempts 4 \
      --block-size 10 \
      --retention-target 0.8 \
      --action-horizon 128 \
      --rungs 0 \
      --seed "$((seed + 1))" \
      --torch-threads 1 \
      --no-parallel-scientists \
      --adaptive-compute \
      --device cpu \
      "${resume_args[@]}"
  fi
  audit_allow_capability_failures "$qroot/q40-1.json" "$q40_output"
  touch "$branch/Q60_COMPLETE"
}

run_recovery_cyclic_to_q60() {
  local branch=$output_root/branches/q-grown-cyclic-memory-12
  local tail_output=$branch/q20-recovery-tail-static-no-sharing
  local tail_export=$branch/q20-export
  local q40_output=$branch/q40-1-static-no-sharing
  local scientist=cyclic-memory-12
  local checkpoint=$archive/migrated/cyclic-memory-12/checkpoint.pt
  local initial_state=$output_root/source-exports/cyclic-memory-12/state.pt.gz
  local selection=$repo/research/local-q-skm-ablation/single-cyclic-memory-12-selection.json
  mkdir -p "$branch" "$tail_export"

  if [[ ! -f "$tail_output/report.json" ]]; then
    local resume_args=()
    if [[ -f "$tail_output/manifest.json" ]]; then
      resume_args=(--resume)
    fi
    uv run pgx-mcts-bench braid-sv2-coordinated \
      --output "$tail_output" \
      --bank "$recovery/q20-tail.json" \
      --prior-bank "$recovery/prior-q20-tail.json" \
      --scientist "$scientist=$checkpoint" \
      --initial-state "$scientist=$initial_state" \
      --arm static-no-sharing \
      --ratios 10,1000 \
      --simulations 128 \
      --qualification-simulations 128 \
      --qualification-attempts 1 \
      --f-native 5 \
      --selfplay-games 8 \
      --train-steps 96 \
      --batch-size 64 \
      --evaluation-attempts 4 \
      --block-size 10 \
      --retention-target 0.8 \
      --action-horizon 128 \
      --rungs 0 \
      --seed 2026081841 \
      --torch-threads 1 \
      --no-parallel-scientists \
      --adaptive-compute \
      --device cpu \
      "${resume_args[@]}"
  fi
  audit_allow_capability_failures "$recovery/q20-tail.json" "$tail_output"
  if [[ ! -f "$tail_export/$scientist/state.pt.gz" ]]; then
    uv run python scripts/export_sv2_scientist_states.py \
      "$tail_output/state.pt.gz" "$selection" "$tail_export"
  fi
  if [[ ! -f "$q40_output/report.json" ]]; then
    local resume_args=()
    if [[ -f "$q40_output/manifest.json" ]]; then
      resume_args=(--resume)
    fi
    uv run pgx-mcts-bench braid-sv2-coordinated \
      --output "$q40_output" \
      --bank "$qroot/q40-1.json" \
      --prior-bank "$qroot/prior-q40-1.json" \
      --scientist "$scientist=$checkpoint" \
      --initial-state "$scientist=$tail_export/$scientist/state.pt.gz" \
      --arm static-no-sharing \
      --ratios 10,1000 \
      --simulations 128 \
      --qualification-simulations 128 \
      --qualification-attempts 1 \
      --f-native 5 \
      --selfplay-games 8 \
      --train-steps 96 \
      --batch-size 64 \
      --evaluation-attempts 4 \
      --block-size 10 \
      --retention-target 0.8 \
      --action-horizon 128 \
      --rungs 0 \
      --seed 2026081842 \
      --torch-threads 1 \
      --no-parallel-scientists \
      --adaptive-compute \
      --device cpu \
      "${resume_args[@]}"
  fi
  audit_allow_capability_failures "$qroot/q40-1.json" "$q40_output"
  touch "$branch/Q60_COMPLETE"
}

mastery_root=/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/nebius-main32-final-20260817/artifacts
bridge_root=$output_root/skm-bridges
mkdir -p "$bridge_root"
verify_hash 3c557d06f1975977e55126d7dcc36e19679e8f84a9d43e03e594c0ce3fa124bd "$mastery_root/multi-knot-mastery-v2-20260815/scientists/high-combined-dual/scientist-state.pt.gz"
verify_hash 352f5737149540c77c6705421dc882b915e8d52fbaaf4e7cdc1b9118288d31ac "$mastery_root/multi-knot-mastery-v2-20260815/scientists/high-cyclic-memory/scientist-state.pt.gz"
verify_hash 25dfc6be76cd38e76e5025e9b03ff83ba33ad9d9bc0127f003b68052e14a6aa1 "$mastery_root/multi-knot-mastery-v1-20260815/scientists/simple-raster-axial/scientist-state.pt.gz"
verify_hash 55c9f861e5dd73807bc01b0087b4ae2c0206454c5b068fa89bff26b8b705de0f "$mastery_root/multi-knot-mastery-v1-20260815/scientists/simple-strand-graph/scientist-state.pt.gz"

if [[ ! -f "$bridge_root/skm-v2-high-combined-dual.pt.gz" ]]; then
  uv run python scripts/bridge_mastery_state_to_sv2.py \
    "$mastery_root/multi-knot-mastery-v2-20260815/scientists/high-combined-dual/scientist-state.pt.gz" \
    "$bridge_root/skm-v2-high-combined-dual.pt.gz" \
    --scientist raster-invariant-combined-dual-12 --f-old 2 --f-native 5 \
    --simulations 32 --lineage skm-v2-high-combined-dual
fi
if [[ ! -f "$bridge_root/skm-v2-high-cyclic-memory.pt.gz" ]]; then
  uv run python scripts/bridge_mastery_state_to_sv2.py \
    "$mastery_root/multi-knot-mastery-v2-20260815/scientists/high-cyclic-memory/scientist-state.pt.gz" \
    "$bridge_root/skm-v2-high-cyclic-memory.pt.gz" \
    --scientist cyclic-memory-12 --f-old 2 --f-native 5 \
    --simulations 128 --lineage skm-v2-high-cyclic-memory
fi
if [[ ! -f "$bridge_root/skm-v1-simple-raster-axial.pt.gz" ]]; then
  uv run python scripts/bridge_mastery_state_to_sv2.py \
    "$mastery_root/multi-knot-mastery-v1-20260815/scientists/simple-raster-axial/scientist-state.pt.gz" \
    "$bridge_root/skm-v1-simple-raster-axial.pt.gz" \
    --scientist raster-axial-12 --f-old 2 --f-native 5 \
    --simulations 64 --lineage skm-v1-simple-raster-axial
fi
if [[ ! -f "$bridge_root/skm-v1-simple-strand-graph.pt.gz" ]]; then
  uv run python scripts/bridge_mastery_state_to_sv2.py \
    "$mastery_root/multi-knot-mastery-v1-20260815/scientists/simple-strand-graph/scientist-state.pt.gz" \
    "$bridge_root/skm-v1-simple-strand-graph.pt.gz" \
    --scientist strand-graph-12 --f-old 8 --f-native 5 \
    --simulations 64 --lineage skm-v1-simple-strand-graph
fi

echo "$(date -u +%FT%TZ) starting five-lineage Q60 wave"
wave_pids=()
wave_labels=()

run_recovery_cyclic_to_q60 &
wave_pids+=("$!")
wave_labels+=(q-grown-cyclic-memory-12)

# Place final SKM weights on the identical Q20 -> Q40-1 curriculum.  These are
# separate lineages even when the architecture name is shared.
run_full_branch_to_q60 \
  skm-v2-high-combined-dual raster-invariant-combined-dual-12 \
  "$archive/migrated/raster-invariant-combined-dual-12/checkpoint.pt" \
  "$bridge_root/skm-v2-high-combined-dual.pt.gz" 32 2026081851 \
  "$repo/research/local-q-skm-ablation/single-raster-invariant-combined-dual-12-selection.json" &
wave_pids+=("$!")
wave_labels+=(skm-v2-high-combined-dual)

run_full_branch_to_q60 \
  skm-v2-high-cyclic-memory cyclic-memory-12 \
  "$archive/migrated/cyclic-memory-12/checkpoint.pt" \
  "$bridge_root/skm-v2-high-cyclic-memory.pt.gz" 128 2026081853 \
  "$repo/research/local-q-skm-ablation/single-cyclic-memory-12-selection.json" &
wave_pids+=("$!")
wave_labels+=(skm-v2-high-cyclic-memory)

run_full_branch_to_q60 \
  skm-v1-simple-raster-axial raster-axial-12 \
  "$archive/migrated/raster-axial-12/checkpoint.pt" \
  "$bridge_root/skm-v1-simple-raster-axial.pt.gz" 64 2026081855 \
  "$repo/research/local-q-skm-ablation/single-raster-axial-12-selection.json" &
wave_pids+=("$!")
wave_labels+=(skm-v1-simple-raster-axial)

run_full_branch_to_q60 \
  skm-v1-simple-strand-graph strand-graph-12 \
  "$archive/migrated/strand-graph-12/checkpoint.pt" \
  "$bridge_root/skm-v1-simple-strand-graph.pt.gz" 64 2026081857 \
  "$repo/research/local-q-skm-ablation/single-strand-graph-12-selection.json" &
wave_pids+=("$!")
wave_labels+=(skm-v1-simple-strand-graph)

wave_failed=0
for index in "${!wave_pids[@]}"; do
  if ! wait "${wave_pids[$index]}"; then
    echo "$(date -u +%FT%TZ) lineage failed: ${wave_labels[$index]}" >&2
    wave_failed=1
  else
    echo "$(date -u +%FT%TZ) lineage complete: ${wave_labels[$index]}"
  fi
done
if (( wave_failed )); then
  exit 4
fi

touch "$output_root/ALL_PROMISING_LINEAGES_Q60_COMPLETE"
