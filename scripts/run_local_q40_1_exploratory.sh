#!/usr/bin/env bash
set -euo pipefail

repo=${LOCAL_ABLATION_REPO:-/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation}
run=${LOCAL_ABLATION_RUN:-/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817}
v3=${LOCAL_MASTERY_V3_ROOT:-/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/mastery-v3-l40s-20260816/artifacts}
qroot=$run/inputs/q4000-v1
canonical_q20=$run/continuation/q4000-v1/q20-static-no-sharing
output_root=$run/continuation/q4000-v1-exploratory-q40-1-20260818
graph_q20=$output_root/cyclic-graph-dual-v3-q20-static-no-sharing
q40=$output_root/q40-1-static-no-sharing
override=$repo/research/local-q-skm-ablation/q40-1-exploratory-override.json
deep_selection=$repo/research/local-q-skm-ablation/cyclic-memory-deep-v3-selection.json
graph_selection=$repo/research/local-q-skm-ablation/cyclic-graph-dual-v3-selection.json
deep_checkpoint=$v3/cyclic-memory-deep-v3-migrated.pt
graph_checkpoint=$v3/cyclic-graph-dual-v3-migrated.pt

export VECLIB_MAXIMUM_THREADS=6
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

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

audit_allow_registered_scientific_failures() {
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
    raise SystemExit(f"unexpected Q20 audit failures: {unexpected}")
PY
}

cd "$repo"
verify_hash 43b64ea3035f116a2f25391aaedd574126937d2d0b19248b5aa672578d0a8322 "$qroot/manifest.json"
verify_hash e9a25901ee49373d10ec07ea5b1cccddc7556947f269f4e304fe6f06f8eb89ae "$qroot/q20.json"
verify_hash 488340fbfd925f1e73c68cc8eb0e46c0c789315711bfd75af7623b49461ea879 "$qroot/q40-1.json"
verify_hash db526211dd3a737b4c36b67484e049c0b83982479fac6f125ae5c6e3b66ad42d "$qroot/prior-q40-1.json"
verify_hash 905ef1cb0b67a66b7b8a479ceb79685a4e0d51c135d02e0fc24ae0d21a1a4f05 "$deep_checkpoint"
verify_hash bf8889112808f60c2289396b15470d3ae46a057641d024ae09b1cbb85a18afa4 "$graph_checkpoint"
verify_hash 88a2a4c55bee11ba5597005a00954c60524b1f0655932790e089520a8f23431c "$canonical_q20/state.pt.gz"
verify_hash b3471f6cc1d36289efd07ff24f990c2d665eb3e68058237de8e8dec9d99847b3 "$canonical_q20/report.json"

mkdir -p "$output_root/protocol" "$output_root/exports"
cp "$override" "$output_root/protocol/"
audit_allow_registered_scientific_failures "$qroot/q20.json" "$canonical_q20"

if [[ ! -f "$graph_q20/report.json" ]]; then
  resume_args=()
  if [[ -f "$graph_q20/manifest.json" ]]; then
    resume_args=(--resume)
  fi
  uv run pgx-mcts-bench braid-sv2-coordinated \
    --output "$graph_q20" \
    --bank "$qroot/q20.json" \
    --scientist "cyclic-graph-dual-v3=$graph_checkpoint" \
    --arm static-no-sharing \
    --ratios 10,1000 \
    --simulations 32 \
    --qualification-simulations 32 \
    --qualification-attempts 1 \
    --f-native 2 \
    --selfplay-games 2 \
    --train-steps 24 \
    --batch-size 32 \
    --evaluation-attempts 2 \
    --no-evaluation-root-noise \
    --block-size 10 \
    --retention-target 0.8 \
    --action-horizon 128 \
    --rungs 0 \
    --seed 2026081822 \
    --torch-threads 6 \
    --no-parallel-scientists \
    --device cpu \
    "${resume_args[@]}"
fi
audit_allow_registered_scientific_failures "$qroot/q20.json" "$graph_q20"

deep_state=$output_root/exports/cyclic-memory-deep-v3/state.pt.gz
if [[ ! -f "$deep_state" ]]; then
  uv run python scripts/export_sv2_scientist_states.py \
    "$canonical_q20/state.pt.gz" "$deep_selection" "$output_root/exports" \
    --f-native 2 --simulations 32
fi
graph_state=$output_root/exports/cyclic-graph-dual-v3/state.pt.gz
if [[ ! -f "$graph_state" ]]; then
  uv run python scripts/export_sv2_scientist_states.py \
    "$graph_q20/state.pt.gz" "$graph_selection" "$output_root/exports" \
    --f-native 2 --simulations 32
fi

if [[ ! -f "$q40/report.json" ]]; then
  resume_args=()
  if [[ -f "$q40/manifest.json" ]]; then
    resume_args=(--resume)
  fi
  uv run pgx-mcts-bench braid-sv2-coordinated \
    --output "$q40" \
    --bank "$qroot/q40-1.json" \
    --prior-bank "$qroot/prior-q40-1.json" \
    --scientist "cyclic-memory-deep-v3=$deep_checkpoint" \
    --scientist "cyclic-graph-dual-v3=$graph_checkpoint" \
    --initial-state "cyclic-memory-deep-v3=$deep_state" \
    --initial-state "cyclic-graph-dual-v3=$graph_state" \
    --arm static-no-sharing \
    --ratios 10,1000 \
    --simulations 32 \
    --qualification-simulations 32 \
    --qualification-attempts 1 \
    --f-native 2 \
    --selfplay-games 2 \
    --train-steps 24 \
    --batch-size 32 \
    --evaluation-attempts 2 \
    --no-evaluation-root-noise \
    --block-size 10 \
    --retention-target 0.8 \
    --action-horizon 128 \
    --rungs 0 \
    --seed 2026081823 \
    --torch-threads 6 \
    --no-parallel-scientists \
    --device cpu \
    "${resume_args[@]}"
fi

uv run python scripts/audit_local_q4000_group.py \
  --group "$qroot/q40-1.json" --output "$q40"
