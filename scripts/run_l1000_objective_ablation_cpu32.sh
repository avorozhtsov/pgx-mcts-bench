#!/usr/bin/env bash
set -Eeuo pipefail

output_root=${1:-/srv/braid/artifacts/l1000-objective-ablation-20260813}
r24_bank=${R24_BANK:-research/semantic-v2-prefix24.json}
r200_bank=${R200_BANK:-research/semantic-v2-r200.json}
r24_seed=${R24_SEED:-20262313}
r200_seed=${R200_SEED:-20262413}

[[ "$(uname -s)" == Linux && "$(nproc)" == 32 ]] || {
  echo "this launcher requires the 32-vCPU Linux host" >&2
  exit 2
}
[[ -f "$r24_bank" && -f "$r200_bank" ]] || {
  echo "missing frozen R24 or R200 bank" >&2
  exit 2
}

declare -A source_checkpoints=(
  [raster-invariant-combined-dual]="/srv/braid/artifacts/invariant-oracle-pretrain-20260812/raster-invariant-combined-dual/seed-81/checkpoint.pt"
  [raster-invariant-combined-film]="/srv/braid/artifacts/invariant-oracle-pretrain-20260812/raster-invariant-combined-film/seed-81/checkpoint.pt"
  [strand-graph]="/srv/braid/work/pgx-mcts-bench/artifacts/current/semantic-moves-v1/k3-budget-critic-repair-20260810/strand-graph/rollback-guarded.pt"
  [raster-axial]="/srv/braid/work/pgx-mcts-bench/artifacts/current/semantic-moves-v1/k3-budget-critic-repair-20260810/raster-axial/rollback-guarded.pt"
)
declare -A expected_hashes=(
  [raster-invariant-combined-dual]="1de838ad0222c907a4d742609b132d9e6b7e911b1c0b8158a9ed02c18b988195"
  [raster-invariant-combined-film]="e2255bcc5896dcefd6efc7dda2176c312de77cef98e7012a714b960c51800c4d"
  [strand-graph]="1d818bfd1ace9e7893c454301e03eacd0953dd2486b7af86cfe4d3a2d6f1e0f2"
  [raster-axial]="fa9b25a2c3d32133c8fd5a0208879369ef4b160353496f1edea68bf13716edef"
)
scientist_names=(
  raster-invariant-combined-dual
  raster-invariant-combined-film
  strand-graph
  raster-axial
)
curricula=(l1000-only mixed-fixed mixed-matched)
declare -A training_ratios=(
  [l1000-only]="1000"
  [mixed-fixed]="10,1000"
  [mixed-matched]="10,1000"
)
declare -A selfplay_games=(
  [l1000-only]=8
  [mixed-fixed]=8
  [mixed-matched]=16
)
declare -A cpu_sets=(
  [l1000-only]="0-9"
  [mixed-fixed]="10-19"
  [mixed-matched]="20-29"
)

mkdir -p "$output_root/frozen-checkpoints" "$output_root/logs"
selection="$output_root/selection.json"
source_patch="$output_root/source.patch"
launcher_snapshot="$output_root/launcher.sh"
temporary_patch="$output_root/source.patch.tmp"
git diff --binary >"$temporary_patch"
if [[ -f "$source_patch" ]]; then
  cmp --silent "$temporary_patch" "$source_patch" || {
    echo "immutable source patch differs: $source_patch" >&2
    exit 2
  }
  rm "$temporary_patch"
else
  mv "$temporary_patch" "$source_patch"
fi
if [[ -f "$launcher_snapshot" ]]; then
  cmp --silent "$0" "$launcher_snapshot" || {
    echo "immutable launcher snapshot differs: $launcher_snapshot" >&2
    exit 2
  }
else
  cp "$0" "$launcher_snapshot"
fi
selection_args=()
for name in "${scientist_names[@]}"; do
  source=${source_checkpoints[$name]}
  expected=${expected_hashes[$name]}
  [[ -f "$source" ]] || { echo "missing checkpoint: $source" >&2; exit 2; }
  actual=$(sha256sum "$source" | cut -d' ' -f1)
  [[ "$actual" == "$expected" ]] || {
    echo "checkpoint hash mismatch for $name: $actual != $expected" >&2
    exit 2
  }
  destination="$output_root/frozen-checkpoints/$name.pt"
  if [[ ! -f "$destination" ]]; then
    cp --preserve=timestamps "$source" "$destination"
  fi
  frozen_hash=$(sha256sum "$destination" | cut -d' ' -f1)
  [[ "$frozen_hash" == "$expected" ]] || {
    echo "frozen checkpoint hash mismatch for $name" >&2
    exit 2
  }
  selection_args+=("$name" "$destination" "$source" "$expected")
done

PYTHONPATH="$PWD/src" .venv/bin/python - \
  "$selection" "$output_root/experiment-manifest.json" \
  "$r24_bank" "$r200_bank" "$source_patch" "$launcher_snapshot" \
  "$r24_seed" "$r200_seed" \
  "${selection_args[@]}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

selection_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
r24_bank = Path(sys.argv[3]).resolve()
r200_bank = Path(sys.argv[4]).resolve()
source_patch = Path(sys.argv[5]).resolve()
launcher = Path(sys.argv[6]).resolve()
r24_seed = int(sys.argv[7])
r200_seed = int(sys.argv[8])
raw = sys.argv[9:]
rows = []
for offset in range(0, len(raw), 4):
    name, checkpoint, source, digest = raw[offset : offset + 4]
    rows.append(
        {
            "name": name,
            "checkpoint": str(Path(checkpoint).resolve()),
            "source_checkpoint": str(Path(source).resolve()),
            "sha256": digest,
        }
    )
selection = {"schema": "l1000-objective-ablation-selection-v1", "selected": rows}

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = {
    "schema": "l1000-objective-ablation-v1",
    "question": "Does mixed L10/L1000 training help or distract final L1000 learning?",
    "architectures": [row["name"] for row in rows],
    "selection": selection,
    "curricula": {
        "l1000-only": {"training_ratios": [1000.0], "games_per_iteration": 8},
        "mixed-fixed": {
            "training_ratios": [10.0, 1000.0],
            "games_per_iteration": 8,
            "games_per_ratio": {"10": 4, "1000": 4},
        },
        "mixed-matched": {
            "training_ratios": [10.0, 1000.0],
            "games_per_iteration": 16,
            "games_per_ratio": {"10": 8, "1000": 8},
        },
    },
    "evaluation_ratios": [1000.0],
    "ordering": "static ACS",
    "sharing": False,
    "phases": [
        {"name": "R24", "bank": str(r24_bank), "sha256": file_hash(r24_bank)},
        {"name": "R200", "bank": str(r200_bank), "sha256": file_hash(r200_bank)},
    ],
    "group_boundary_controller_reset": {"F_native": 5, "simulations": 64},
    "paired_seeds": {"R24": r24_seed, "R200": r200_seed},
    "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    "code_patch": {"path": str(source_patch), "sha256": file_hash(source_patch)},
    "launcher": {"path": str(launcher), "sha256": file_hash(launcher)},
}

for path, payload in ((selection_path, selection), (manifest_path, manifest)):
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != payload:
            raise SystemExit(f"immutable experiment metadata differs: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
PY

scientist_arguments=()
for name in "${scientist_names[@]}"; do
  scientist_arguments+=(--scientist "$name=$output_root/frozen-checkpoints/$name.pt")
done

children=()
cleanup_children() {
  for pid in "${children[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup_children INT TERM

wait_for_phase() {
  local phase=$1
  local failed=0
  for pid in "${children[@]}"; do
    if ! wait "$pid"; then
      failed=1
      break
    fi
  done
  if (( failed )); then
    cleanup_children
    wait || true
    echo "$phase failed; resumable artifacts are preserved" >&2
    exit 1
  fi
  children=()
}

run_r24() {
  local curriculum=$1
  local output="$output_root/r24/$curriculum"
  local resume=()
  mkdir -p "$output"
  [[ ! -f "$output/manifest.json" ]] || resume=(--resume)
  exec taskset -c "${cpu_sets[$curriculum]}" \
    nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
      PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" \
    .venv/bin/pgx-mcts-bench braid-sv2-coordinated \
      --output "$output" \
      --bank "$r24_bank" \
      "${scientist_arguments[@]}" \
      --arm static-no-sharing \
      --ratios 1000 \
      --training-ratios "${training_ratios[$curriculum]}" \
      --simulations 64 \
      --qualification-simulations 64 \
      --qualification-attempts 1 \
      --f-native 10 \
      --selfplay-games "${selfplay_games[$curriculum]}" \
      --train-steps 96 \
      --batch-size 64 \
      --evaluation-attempts 4 \
      --block-size 10 \
      --retention-target 0.80 \
      --action-horizon 128 \
      --rungs 24 \
      --seed "$r24_seed" \
      --torch-threads 1 \
      --parallel-scientists \
      --device cpu \
      "${resume[@]}"
}

for curriculum in "${curricula[@]}"; do
  mkdir -p "$output_root/logs"
  run_r24 "$curriculum" >"$output_root/logs/r24-$curriculum.log" 2>&1 &
  children+=("$!")
done
wait_for_phase R24

for curriculum in "${curricula[@]}"; do
  export_root="$output_root/r24-exports/$curriculum"
  PYTHONPATH="$PWD/src" .venv/bin/python scripts/export_sv2_scientist_states.py \
    "$output_root/r24/$curriculum/state.pt.gz" \
    "$selection" \
    "$export_root" \
    --f-native 5 \
    --simulations 64 \
    >"$output_root/logs/export-$curriculum.log" 2>&1
done

run_r200() {
  local curriculum=$1
  local output="$output_root/r200/$curriculum"
  local resume=()
  local initial_arguments=()
  for name in "${scientist_names[@]}"; do
    initial_arguments+=(
      --initial-state "$name=$output_root/r24-exports/$curriculum/$name/state.pt.gz"
    )
  done
  mkdir -p "$output"
  [[ ! -f "$output/manifest.json" ]] || resume=(--resume)
  exec taskset -c "${cpu_sets[$curriculum]}" \
    nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
      PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" \
    .venv/bin/pgx-mcts-bench braid-sv2-coordinated \
      --output "$output" \
      --bank "$r200_bank" \
      --prior-bank "$r24_bank" \
      "${scientist_arguments[@]}" \
      "${initial_arguments[@]}" \
      --arm static-no-sharing \
      --ratios 1000 \
      --training-ratios "${training_ratios[$curriculum]}" \
      --simulations 64 \
      --qualification-simulations 64 \
      --qualification-attempts 1 \
      --f-native 5 \
      --selfplay-games "${selfplay_games[$curriculum]}" \
      --train-steps 96 \
      --batch-size 64 \
      --evaluation-attempts 4 \
      --block-size 10 \
      --retention-target 0.80 \
      --action-horizon 128 \
      --rungs 200 \
      --seed "$r200_seed" \
      --torch-threads 1 \
      --parallel-scientists \
      --adaptive-compute \
      --device cpu \
      "${resume[@]}"
}

for curriculum in "${curricula[@]}"; do
  run_r200 "$curriculum" >"$output_root/logs/r200-$curriculum.log" 2>&1 &
  children+=("$!")
done
wait_for_phase R200

echo "L1000 objective ablation completed: $output_root"
