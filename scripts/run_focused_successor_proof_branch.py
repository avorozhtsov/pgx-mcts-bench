#!/usr/bin/env python3
"""Run the hash-bound proof-distilled focused successor branch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    if gate.get("schema") != "semantic-v2-q254-first-block-seeded-order-v1" or not gate.get("passed"):
        raise RuntimeError("proof-distilled continuation gate did not pass")
    if gate.get("cohort") != "focused-successor-v1-proof-distilled" or gate.get("sharing") != "strict-none":
        raise RuntimeError("proof-distilled focused cohort differs")
    binding = gate["focused_lines"]["strand-graph-12-proof-distilled"]
    state, bank, prior = Path(binding["initial_state"]), Path(gate["bank"]), Path(gate["prior_bank"])
    if sha256(state) != binding["initial_state_sha256"]:
        raise RuntimeError("proof-distilled carry hash changed")
    if sha256(bank) != gate["bank_byte_sha256"] or sha256(prior) != gate["prior_bank_byte_sha256"]:
        raise RuntimeError("proof-distilled bank hash changed")
    runtime_path = Path(gate["runtime"])
    if sha256(runtime_path) != gate["runtime_sha256"]:
        raise RuntimeError("proof-distilled runtime hash changed")
    sys.path.insert(0, str(runtime_path.parents[1]))
    import pgx_mcts_bench
    package_dir = str(runtime_path.parent)
    if package_dir not in pgx_mcts_bench.__path__:
        pgx_mcts_bench.__path__.insert(0, package_dir)
    name = "pgx_mcts_bench.q254_sv2_curriculum_runtime"
    spec = importlib.util.spec_from_file_location(name, runtime_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load proof-distilled runtime")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[name] = runtime
    spec.loader.exec_module(runtime)
    report = runtime.run_coordinated_arm(
        {binding["scientist"]: args.checkpoint}, bank, args.output,
        arm="scheduled-no-sharing", prior_bank=prior,
        initial_states={binding["scientist"]: state}, ratios=(10.0, 1000.0),
        simulations=40, qualification_simulations=40, qualification_attempts=1,
        f_native=4, selfplay_games=4, train_steps=24, batch_size=64,
        evaluation_attempts=2, block_size=10, retention_target=0.8,
        action_horizon=128, rungs=0, seed=int(binding["seed"]), torch_threads=1,
        parallel_scientists=True, rehearsal_panel_size=20,
        strict_own_budget_rehearsal=True, terminal_full_retention_audit=True,
        rehearsal_task_order_transition=args.gate, adaptive_compute=True,
        f_native_levels=(4, 6, 8, 12, 16), simulation_levels=(40, 64, 80, 128, 256),
        device="cpu", resume=(args.output / "manifest.json").is_file(),
    )
    print(json.dumps({"completed_rungs": report["completed_rungs"]}, indent=2))


if __name__ == "__main__":
    main()
