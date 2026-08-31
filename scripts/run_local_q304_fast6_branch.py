#!/usr/bin/env python3
"""Run one fast Q304 lineage with its isolated seeded boundary-0 runtime."""

from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
sys.path.insert(0, str(REPO / "scripts"))
from focused_successor_policy import assert_legacy_q304_launch_authorized  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def load_runtime(path: Path, expected: str) -> Any:
    if sha256(path) != expected: raise RuntimeError("isolated Q304 runtime hash changed")
    sys.path.insert(0, str(path.parents[1])); import pgx_mcts_bench
    if str(path.parent) not in pgx_mcts_bench.__path__: pgx_mcts_bench.__path__.insert(0, str(path.parent))
    name = "pgx_mcts_bench.q304_sv2_curriculum_runtime"; spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError("could not load isolated Q304 runtime")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


def main() -> None:
    assert_legacy_q304_launch_authorized()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "bank", "prior-bank", "checkpoint", "initial-state", "gate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scientist", required=True); parser.add_argument("--seed", type=int, required=True); args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    if gate.get("schema") != "semantic-v2-q304-first-block-seeded-order-v1" or not gate.get("passed") or gate.get("boundary_completed_rungs") != 0 or not gate.get("first_rehearsal_block_seeded") or gate.get("sharing") != "strict-none":
        raise RuntimeError("Q304 first-block rehearsal gate did not pass")
    if sha256(args.bank) != gate["bank_byte_sha256"] or sha256(args.prior_bank) != gate["prior_bank_byte_sha256"]:
        raise RuntimeError("Q304 protocol bank hash changed")
    runtime = load_runtime(Path(gate["runtime"]), gate["runtime_sha256"])
    report = runtime.run_coordinated_arm({args.scientist: args.checkpoint}, args.bank, args.output,
        arm="scheduled-no-sharing", prior_bank=args.prior_bank, initial_states={args.scientist: args.initial_state},
        ratios=(10.0, 1000.0), simulations=40, qualification_simulations=40, qualification_attempts=1,
        f_native=4, selfplay_games=4, train_steps=24, batch_size=64, evaluation_attempts=2, block_size=10,
        retention_target=0.8, action_horizon=128, rungs=0, seed=args.seed, torch_threads=1,
        parallel_scientists=True, rehearsal_panel_size=20, strict_own_budget_rehearsal=True,
        terminal_full_retention_audit=True, rehearsal_task_order_transition=args.gate, adaptive_compute=True,
        f_native_levels=(4, 6, 8, 12, 16), simulation_levels=(40, 64, 80, 128, 256), device="cpu",
        resume=(args.output / "manifest.json").is_file())
    print(json.dumps({"completed_rungs": report["completed_rungs"]}, indent=2))


if __name__ == "__main__": main()
