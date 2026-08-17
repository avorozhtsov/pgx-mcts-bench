#!/usr/bin/env bash
set -euo pipefail

repo=${LOCAL_ABLATION_REPO:-/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation}
cd "$repo"
bash scripts/run_local_v3_initialization_pair.sh
bash scripts/run_local_q_skm_process_pair.sh
bash scripts/run_local_selected_q_skm_continuation.sh
