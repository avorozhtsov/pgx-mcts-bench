#!/usr/bin/env python3
"""Evaluate the prepared four-example DKT shadow gate without learning."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_runtime(path: Path, expected_hash: str) -> Any:
    if sha256(path) != expected_hash:
        raise RuntimeError("bound shadow evaluation runtime hash changed")
    runtime_root = path.parents[1]
    sys.path.insert(0, str(runtime_root))
    import pgx_mcts_bench

    package_dir = str(path.parent)
    if package_dir not in pgx_mcts_bench.__path__:
        pgx_mcts_bench.__path__.insert(0, package_dir)
    name = "pgx_mcts_bench.dkt_shadow4_q204_runtime"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load bound shadow evaluation runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def network_digest(network: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def summarize(rows: list[dict[str, Any]], *, network_unchanged: bool) -> dict[str, Any]:
    attempts = sum(len(row["evaluation"]["10.0"]["attempts"]) for row in rows)
    repeat_or_improve = [
        row
        for row in rows
        if row["best_crossing_changes"] is not None
        and int(row["best_crossing_changes"]) <= int(row["registered_upper_bound"])
    ]
    complete = len(rows) == 4 and attempts == 16
    passed = complete and network_unchanged and bool(repeat_or_improve)
    return {
        "status": "COMPLETED",
        "passed": passed,
        "examples": len(rows),
        "attempts": attempts,
        "solved_examples": sum(row["best_crossing_changes"] is not None for row in rows),
        "repeat_or_improve_examples": len(repeat_or_improve),
        "repeat_or_improve_ids": [row["id"] for row in repeat_or_improve],
        "network_unchanged": network_unchanged,
        "blocking_reasons": [
            reason
            for reason, active in (
                ("incomplete_4x4_evaluation", not complete),
                ("network_changed_during_evaluation", not network_unchanged),
                ("no_replay_verified_repeat_or_improvement", not repeat_or_improve),
            )
            if active
        ],
    }


def run(
    gate_path: Path, output: Path, *, evaluation_confirmed: bool = False
) -> dict[str, Any]:
    if not evaluation_confirmed:
        raise RuntimeError("shadow evaluation requires explicit confirmation")
    if output.exists():
        raise FileExistsError(f"shadow result already exists: {output}")
    gate = json.loads(gate_path.read_text())
    if (
        gate.get("schema") != "dkt-disjoint-shadow4-raster-axial-prepared-v1"
        or gate.get("status") != "PREPARED"
        or not gate.get("prepared")
    ):
        raise RuntimeError("shadow gate is not PREPARED")
    protocol = gate["protocol"]
    expected = {
        "learning": False,
        "sharing": False,
        "objective_ratio": 10.0,
        "examples": 4,
        "attempts_per_example": 4,
        "total_attempts": 16,
        "simulations": 256,
        "action_horizon": 128,
        "root_noise": True,
        "temperature": 0.0,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"shadow protocol differs at {key}")
    for raw_path, expected_hash in gate.get("source_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"gate-bound shadow source changed: {path}")
    bound_paths = {
        "bank": Path(gate["bank"]),
        "selection_audit": Path(gate["selection_audit"]),
        "q204_gate": Path(gate["q204_gate"]),
        "runtime": Path(gate["runtime"]),
        "checkpoint": Path(gate["checkpoint"]),
        "state": Path(gate["terminal"]["state"]),
        "report": Path(gate["terminal"]["report"]),
        "terminal_audit": Path(gate["terminal"]["terminal_audit"]),
    }
    expected_hashes = {
        "bank": gate["bank_byte_sha256"],
        "selection_audit": gate["selection_audit_sha256"],
        "q204_gate": gate["q204_gate_sha256"],
        "runtime": gate["runtime_sha256"],
        "checkpoint": gate["checkpoint_sha256"],
        "state": gate["terminal"]["state_sha256"],
        "report": gate["terminal"]["report_sha256"],
        "terminal_audit": gate["terminal"]["terminal_audit_sha256"],
    }
    for key, path in bound_paths.items():
        if not path.is_file() or sha256(path) != expected_hashes[key]:
            raise RuntimeError(f"bound shadow input changed: {key}")

    runtime = load_runtime(bound_paths["runtime"], expected_hashes["runtime"])
    bank = json.loads(bound_paths["bank"].read_text())
    if bank.get("size") != 4 or len(bank.get("rows", [])) != 4:
        raise RuntimeError("shadow bank is not exactly four rows")
    items = runtime._bank_from_payload(bank["rows"])
    state = runtime._load_state(bound_paths["state"])
    scientist_name = str(gate["scientist"])
    if set(state.get("scientists", {})) != {scientist_name}:
        raise RuntimeError("terminal state does not contain exactly the bound scientist")
    scientist = runtime._load_roster(
        {scientist_name: bound_paths["checkpoint"]},
        seed=int(protocol["base_seed"]),
        device="cpu",
        simulations=256,
        action_horizon=128,
    )[0]
    runtime._restore_scientist(scientist, state["scientists"][scientist_name])
    scientist.network.eval()
    before = network_digest(scientist.network)
    torch.set_num_threads(1)
    rows = []
    source_rows = {str(row["id"]): row for row in bank["rows"]}
    for index, item in enumerate(items):
        evaluation = runtime._evaluate(
            scientist,
            item.knot,
            ratios=(10.0,),
            attempts=4,
            simulations=256,
            seed=int(protocol["base_seed"]) + index * 100_000,
            add_root_noise=True,
        )
        best = evaluation["10.0"]["best_witness"]
        source = source_rows[item.id]
        rows.append(
            {
                "id": item.id,
                "name": str(source["name"]),
                "representation_id": str(source_rows[item.id].get("representation_id", "")),
                "registered_upper_bound": int(
                    source["certified_unknotting_upper_bound"]
                ),
                "best_crossing_changes": (
                    int(best["crossing_changes"]) if best is not None else None
                ),
                "best_semantic_moves": (
                    int(best["semantic_moves"]) if best is not None else None
                ),
                "evaluation": evaluation,
            }
        )
    after = network_digest(scientist.network)
    summary = summarize(rows, network_unchanged=before == after)
    report = {
        "schema": "dkt-disjoint-shadow4-raster-axial-result-v1",
        "gate": str(gate_path.resolve()),
        "gate_sha256": sha256(gate_path),
        "scientist": scientist_name,
        "lineage": gate["lineage"],
        "protocol": protocol,
        "training_performed": False,
        "network_sha256_before": before,
        "network_sha256_after": after,
        "results": rows,
        "summary": summary,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-evaluate",
        action="store_true",
        help="explicitly confirm the 16 frozen 256-simulation attempts",
    )
    args = parser.parse_args()
    report = run(
        args.gate,
        args.output,
        evaluation_confirmed=args.confirm_evaluate,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
