#!/usr/bin/env bash
set -euo pipefail

repo=${LOCAL_ABLATION_REPO:-/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation}
run=${LOCAL_ABLATION_RUN:-/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817}
warm=${LOCAL_ABLATION_WARM:-/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/mastery-v3-l40s-20260816/artifacts/cyclic-memory-deep-v3-migrated.pt}
native=${LOCAL_ABLATION_NATIVE:-$run/checkpoints/native-deep-v3.pt}
bank=$run/inputs/train-bank.json
heldout=$run/inputs/heldout-bank.json
sequence=$run/inputs/train-sequence.json

export VECLIB_MAXIMUM_THREADS=6
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

cd "$repo"
mkdir -p "$run/process-pair"

if [[ ! -f "$run/initialization-pair/warm-heldout-q20.json" ]] || \
   [[ ! -f "$run/initialization-pair/native-heldout-q20.json" ]]; then
  bash scripts/run_local_v3_initialization_pair.sh
fi

if [[ ! -f "$run/initialization-selection.json" ]]; then
  if ! uv run python scripts/select_local_q_skm_arm.py \
    --kind initialization \
    --report "warm=$run/initialization-pair/warm-heldout-q20.json" \
    --report "native=$run/initialization-pair/native-heldout-q20.json" \
    --output "$run/initialization-selection.json"; then
    for label in warm native; do
      if [[ ! -f "$run/initialization-pair/$label-heldout-sim128.json" ]]; then
        uv run python scripts/evaluate_local_q_skm_checkpoint.py \
          --state "$run/initialization-pair/$label-q20/state.pt.gz" \
          --scientist cyclic-memory-deep-v3 \
          --bank "$heldout" \
          --output "$run/initialization-pair/$label-heldout-sim128.json" \
          --simulations 128 --threads 6 --seed 2026081803
      fi
    done
    uv run python scripts/select_local_q_skm_arm.py \
      --kind initialization \
      --report "warm=$run/initialization-pair/warm-heldout-sim128.json" \
      --report "native=$run/initialization-pair/native-heldout-sim128.json" \
      --output "$run/initialization-selection.json"
  fi
fi
selected=$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$run/initialization-selection.json")
if [[ "$selected" == warm ]]; then
  checkpoint=$warm
elif [[ "$selected" == native ]]; then
  checkpoint=$native
else
  echo "invalid initialization selection: $selected" >&2
  exit 2
fi

q_output=$run/process-pair/q
if [[ ! -f "$q_output/report.json" ]]; then
  resume_args=()
  if [[ -f "$q_output/manifest.json" ]]; then
    resume_args=(--resume)
  fi
  uv run pgx-mcts-bench braid-sv2-coordinated \
    --output "$q_output" \
    --bank "$bank" \
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
    --block-size 4 \
    --retention-target 0.8 \
    --action-horizon 128 \
    --rungs 20 \
    --seed 2026081812 \
    --torch-threads 6 \
    --no-parallel-scientists \
    --device cpu \
    "${resume_args[@]}"
fi
if [[ ! -f "$run/process-pair/q-heldout.json" ]]; then
  q_completed=$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["completed_rungs"])' "$q_output/report.json")
  if [[ "$q_completed" != 20 ]]; then
    echo "Q process arm completed $q_completed of 20 registered training rows" >&2
    exit 2
  fi
  uv run python scripts/evaluate_local_q_skm_checkpoint.py \
    --state "$q_output/state.pt.gz" \
    --scientist cyclic-memory-deep-v3 \
    --bank "$heldout" \
    --output "$run/process-pair/q-heldout.json" \
    --simulations 32 --threads 6 --seed 2026081813
fi

skm_output=$run/process-pair/skm
if [[ ! -f "$skm_output/program-state.json" ]] || [[ ! -f "$skm_output/process-pair-complete" ]]; then
  skm_resume=()
  if [[ -f "$skm_output/program-state.json" ]]; then
    skm_resume=(--resume)
  fi
  uv run python -m pgx_mcts_bench.mastery_program \
    --short-ablation \
    --sequence "$sequence" \
    --scientist cyclic-memory-deep-v3 \
    --checkpoint "$checkpoint" \
    --output "$skm_output" \
    --evidence-inventory "$run/process-pair/skm-evidence" \
    --device cpu \
    --steps 20 \
    --parallel-searches 6 \
    --torch-threads 6 \
    --simulations 32 \
    --action-horizon 128 \
    --max-heap 40 \
    --challenge-attempt-limit 3 \
    --challenge-seconds-limit 120 \
    --seed 2026081812 \
    "${skm_resume[@]}"
  touch "$skm_output/process-pair-complete"
fi
if [[ ! -f "$run/process-pair/skm-heldout.json" ]]; then
  skm_introduced=$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["introduced_count"])' "$skm_output/program-state.json")
  if [[ "$skm_introduced" != 20 ]]; then
    echo "SKM process arm introduced $skm_introduced of 20 registered training rows" >&2
    exit 2
  fi
  uv run python scripts/evaluate_local_q_skm_checkpoint.py \
    --state "$skm_output/scientist-state.pt.gz" \
    --scientist cyclic-memory-deep-v3 \
    --bank "$heldout" \
    --output "$run/process-pair/skm-heldout.json" \
    --simulations 32 --threads 6 --seed 2026081813
fi

if [[ ! -f "$run/process-selection.json" ]]; then
  if ! uv run python scripts/select_local_q_skm_arm.py \
    --kind process \
    --report "q=$run/process-pair/q-heldout.json" \
    --report "skm=$run/process-pair/skm-heldout.json" \
    --output "$run/process-selection.json"; then
    for label in q skm; do
      if [[ ! -f "$run/process-pair/$label-heldout-sim128.json" ]]; then
        if [[ "$label" == q ]]; then
          state=$q_output/state.pt.gz
        else
          state=$skm_output/scientist-state.pt.gz
        fi
        uv run python scripts/evaluate_local_q_skm_checkpoint.py \
          --state "$state" \
          --scientist cyclic-memory-deep-v3 \
          --bank "$heldout" \
          --output "$run/process-pair/$label-heldout-sim128.json" \
          --simulations 128 --threads 6 --seed 2026081813
      fi
    done
    uv run python scripts/select_local_q_skm_arm.py \
      --kind process \
      --report "q=$run/process-pair/q-heldout-sim128.json" \
      --report "skm=$run/process-pair/skm-heldout-sim128.json" \
      --output "$run/process-selection.json"
  fi
fi
