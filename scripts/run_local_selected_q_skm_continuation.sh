#!/usr/bin/env bash
set -euo pipefail

repo=${LOCAL_ABLATION_REPO:-/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation}
run=${LOCAL_ABLATION_RUN:-/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817}
warm=${LOCAL_ABLATION_WARM:-/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/mastery-v3-l40s-20260816/artifacts/cyclic-memory-deep-v3-migrated.pt}
native=${LOCAL_ABLATION_NATIVE:-$run/checkpoints/native-deep-v3.pt}
source_root=/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/nebius-semantic-v2-live-backup/mirror/results/multi-knot-mastery-v1-20260815
sequence=$source_root/inputs/sequence-high-strand-240.json
rehearsal=$source_root/inputs/rehearsal-panel-32.json
qroot=$run/inputs/q4000-v1

export VECLIB_MAXIMUM_THREADS=6
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

cd "$repo"
initialization=$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$run/initialization-selection.json")
process=$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$run/process-selection.json")
if [[ "$initialization" == warm ]]; then
  checkpoint=$warm
elif [[ "$initialization" == native ]]; then
  checkpoint=$native
else
  echo "invalid initialization selection: $initialization" >&2
  exit 2
fi

mkdir -p "$run/continuation"
if [[ "$process" == q ]]; then
  expected_manifest=43b64ea3035f116a2f25391aaedd574126937d2d0b19248b5aa672578d0a8322
  observed_manifest=$(shasum -a 256 "$qroot/manifest.json" | awk '{print $1}')
  if [[ "$observed_manifest" != "$expected_manifest" ]]; then
    echo "Q4000 manifest hash mismatch: $observed_manifest" >&2
    exit 2
  fi
  qrun=$run/continuation/q4000-v1
  mkdir -p "$qrun/exports"
  selection=$qrun/single-scientist-selection.json
  if [[ ! -f "$selection" ]]; then
    uv run python -c \
      'import json,sys; open(sys.argv[1],"w").write(json.dumps({"selected":[{"name":"cyclic-memory-deep-v3"}]},indent=2)+"\n")' \
      "$selection"
  fi
  previous_state=
  group_index=0
  while IFS=$'\t' read -r group_name bank_name prior_name group_size; do
    echo "starting or auditing $group_name ($group_size rows)"
    output=$qrun/$group_name-static-no-sharing
    initial_args=()
    if [[ -n "$previous_state" ]]; then
      initial_args=(--initial-state "cyclic-memory-deep-v3=$previous_state")
    fi
    resume_args=()
    if [[ -f "$output/manifest.json" && ! -f "$output/report.json" ]]; then
      resume_args=(--resume)
    fi
    if [[ ! -f "$output/report.json" ]]; then
      prior_count=$(uv run python -c \
        'import json,sys; payload=json.load(open(sys.argv[1])); print(len(payload.get("rows", [])))' \
        "$qroot/$prior_name")
      prior_args=()
      if (( prior_count > 0 )); then
        prior_args=(--prior-bank "$qroot/$prior_name")
      fi
      uv run pgx-mcts-bench braid-sv2-coordinated \
        --output "$output" \
        --bank "$qroot/$bank_name" \
        "${prior_args[@]}" \
        --scientist "cyclic-memory-deep-v3=$checkpoint" \
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
        --seed "$((2026081822 + group_index))" \
        --torch-threads 6 \
        --no-parallel-scientists \
        --device cpu \
        "${initial_args[@]}" \
        "${resume_args[@]}"
    fi
    uv run python scripts/audit_local_q4000_group.py \
      --group "$qroot/$bank_name" --output "$output"
    export_root=$qrun/exports/$group_name
    if [[ ! -f "$export_root/cyclic-memory-deep-v3/state.pt.gz" ]]; then
      uv run python scripts/export_sv2_scientist_states.py \
        "$output/state.pt.gz" "$selection" "$export_root" \
        --f-native 2 --simulations 32
    fi
    previous_state=$export_root/cyclic-memory-deep-v3/state.pt.gz
    group_index=$((group_index + 1))
  done < <(
    uv run python -c \
      'import json,sys; m=json.load(open(sys.argv[1])); [print(g["name"],g["bank"],g["prior"],g["size"],sep="\t") for g in m["groups"]]' \
      "$qroot/manifest.json"
  )
elif [[ "$process" == skm ]]; then
  inventory=$run/continuation/evidence-inventory
  if [[ ! -d "$inventory/witnesses" ]]; then
    mkdir -p "$run/continuation"
    cp -R "$source_root/evidence-inventory" "$inventory"
  fi
  skm_resume=()
  if [[ -f "$run/continuation/skm-high-strand-240/program-state.json" ]]; then
    skm_resume=(--resume)
  fi
  uv run python -m pgx_mcts_bench.mastery_program \
    --sequence "$sequence" \
    --scientist cyclic-memory-deep-v3 \
    --checkpoint "$checkpoint" \
    --output "$run/continuation/skm-high-strand-240" \
    --evidence-inventory "$inventory" \
    --rehearsal-panel "$rehearsal" \
    --device cpu \
    --steps 100000 \
    --parallel-searches 6 \
    --torch-threads 6 \
    --simulations 32 \
    --action-horizon 128 \
    --max-heap 200 \
    --challenge-attempt-limit 24 \
    --challenge-seconds-limit 900 \
    --seed 2026081822 \
    "${skm_resume[@]}"
else
  echo "invalid process selection: $process" >&2
  exit 2
fi
