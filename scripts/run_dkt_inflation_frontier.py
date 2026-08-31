#!/usr/bin/env python3
"""Run the prepared two-stage DKT inflation-frontier solution miner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
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
        raise RuntimeError("bound Q204 runtime hash changed")
    runtime_root = path.parents[1]
    sys.path.insert(0, str(runtime_root))
    import pgx_mcts_bench

    package_dir = str(path.parent)
    if package_dir not in pgx_mcts_bench.__path__:
        pgx_mcts_bench.__path__.insert(0, package_dir)
    name = "pgx_mcts_bench.dkt_inflation_frontier_q204_runtime"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load bound Q204 runtime")
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


def _attempt(
    runtime: Any,
    scientist: Any,
    item: Any,
    *,
    ratio: float,
    simulations: int,
    seed: int,
    objective_cap: float,
    root_noise: bool,
) -> dict[str, Any]:
    verified, measured = runtime._evaluation_records(
        scientist,
        item.knot,
        ratio,
        simulations,
        [seed],
        objective_cap=objective_cap,
        add_root_noise=root_noise,
    )[0]
    return {
        "seed": seed,
        "simulations": simulations,
        "solved": verified is not None,
        "crossing_changes": int(verified[0]) if verified is not None else None,
        "semantic_moves": int(verified[1]) if verified is not None else None,
        "semantic_actions": (
            [int(action) for action in verified[2]] if verified is not None else None
        ),
        **measured,
    }


def certify(
    verifier: Any,
    original_word: tuple[int, ...],
    original_strands: int,
    prefix: list[int],
    attempt: dict[str, Any],
    *,
    published_upper_bound: int,
    total_semantic_move_cap: int,
) -> dict[str, Any] | None:
    if not attempt["solved"]:
        return None
    actions = [*prefix, *attempt["semantic_actions"]]
    witness = verifier.verify_actions(original_word, original_strands, actions)
    if witness.crossing_changes > published_upper_bound:
        return None
    if witness.semantic_moves > total_semantic_move_cap:
        return None
    return {
        "crossing_changes": witness.crossing_changes,
        "semantic_moves": witness.semantic_moves,
        "semantic_actions": actions,
        "prefix_semantic_moves": len(prefix),
        "suffix_semantic_moves": attempt["semantic_moves"],
        "peak_strands": max(state.strands for state in witness.states),
        "peak_word_length": max(len(state.word) for state in witness.states),
        "terminal_word": list(witness.states[-1].word),
        "terminal_strands": witness.states[-1].strands,
        "replay_verified": True,
    }


def summarize(rows: list[dict[str, Any]], *, network_unchanged: bool) -> dict[str, Any]:
    certificates = [
        certificate
        for row in rows
        for certificate in row.get("certificates", [])
        if certificate is not None
    ]
    best = min(
        certificates,
        key=lambda row: (row["crossing_changes"], row["semantic_moves"]),
        default=None,
    )
    return {
        "status": "COMPLETED",
        "passed": bool(certificates) and network_unchanged,
        "frontier_states": len(rows),
        "screening_attempts": sum(len(row["screening_attempts"]) for row in rows),
        "promotion_attempts": sum(len(row["promotion_attempts"]) for row in rows),
        "replay_verified_certificates": len(certificates),
        "best_crossing_changes": best["crossing_changes"] if best else None,
        "best_semantic_moves": best["semantic_moves"] if best else None,
        "network_unchanged": network_unchanged,
        "blocking_reasons": [
            reason
            for reason, active in (
                ("network_changed_during_evaluation", not network_unchanged),
                ("no_replay_verified_path_at_published_upper_bound", not certificates),
            )
            if active
        ],
    }


def run(
    policy_path: Path,
    bank_path: Path,
    output: Path,
    *,
    evaluation_confirmed: bool = False,
) -> dict[str, Any]:
    if not evaluation_confirmed:
        raise RuntimeError("DKT inflation-frontier evaluation requires explicit confirmation")
    if output.exists():
        raise FileExistsError(f"DKT inflation-frontier result already exists: {output}")
    policy = json.loads(policy_path.read_text())
    bank = json.loads(bank_path.read_text())
    if policy.get("status") != "PREPARED" or policy.get("evaluation", {}).get("learning"):
        raise RuntimeError("DKT inflation-frontier policy is not PREPARED evaluation-only")
    if bank.get("schema") != "dkt-proof-carrying-inflation-frontier-bank-v1":
        raise RuntimeError("unknown DKT inflation-frontier bank")
    status_path = output.with_suffix(output.suffix + ".status.json")
    atomic_json(
        status_path,
        {
            "schema": "dkt-proof-carrying-inflation-frontier-status-v1",
            "status": "LAUNCHED",
            "pid": os.getpid(),
            "started_unix": time.time(),
            "policy": str(policy_path.resolve()),
            "policy_sha256": sha256(policy_path),
            "bank": str(bank_path.resolve()),
            "bank_sha256": sha256(bank_path),
            "learning": False,
        },
    )
    for key, source in policy["sources"].items():
        path = Path(source["path"])
        if not path.is_file() or sha256(path) != source["byte_sha256"]:
            raise RuntimeError(f"gate-bound DKT source changed: {key}")

    representative_gate = json.loads(
        Path(policy["sources"]["representative_gate"]["path"]).read_text()
    )
    for raw_path, expected_hash in representative_gate.get("source_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"representative gate-bound source changed: {path}")
    bound = {
        "runtime": Path(representative_gate["runtime"]),
        "checkpoint": Path(representative_gate["checkpoint"]),
        "state": Path(representative_gate["terminal"]["state"]),
        "report": Path(representative_gate["terminal"]["report"]),
        "terminal_audit": Path(representative_gate["terminal"]["terminal_audit"]),
    }
    hashes = {
        "runtime": representative_gate["runtime_sha256"],
        "checkpoint": representative_gate["checkpoint_sha256"],
        "state": representative_gate["terminal"]["state_sha256"],
        "report": representative_gate["terminal"]["report_sha256"],
        "terminal_audit": representative_gate["terminal"]["terminal_audit_sha256"],
    }
    for key, path in bound.items():
        if not path.is_file() or sha256(path) != hashes[key]:
            raise RuntimeError(f"representative input changed: {key}")

    runtime = load_runtime(bound["runtime"], hashes["runtime"])
    items = runtime._bank_from_payload(bank["rows"])
    source_rows = {str(row["id"]): row for row in bank["rows"]}
    state = runtime._load_state(bound["state"])
    scientist_name = str(representative_gate["scientist"])
    evaluation = policy["evaluation"]
    scientist = runtime._load_roster(
        {scientist_name: bound["checkpoint"]},
        seed=int(evaluation["base_seed"]),
        device="cpu",
        simulations=int(evaluation["promotion"]["simulations"]),
        action_horizon=int(evaluation["suffix_action_horizon"]),
    )[0]
    runtime._restore_scientist(scientist, state["scientists"][scientist_name])
    scientist.network.eval()
    torch.set_num_threads(1)
    before = network_digest(scientist.network)

    from pgx_mcts_bench.collaborative_scientists import prediction_details
    from pgx_mcts_bench.semantic_verifier import SemanticBraidVerifier

    ratio = float(evaluation["objective_ratio"])
    ranked = []
    for item in items:
        prediction = prediction_details(scientist, item, (ratio,))[0]
        ranked.append((float(prediction["p_solve"]), item.id, prediction))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    promoted = {
        item_id for _score, item_id, _prediction
        in ranked[: int(evaluation["promotion"]["count"])]
    }
    predictions = {item_id: prediction for _score, item_id, prediction in ranked}

    verifier = SemanticBraidVerifier.from_config(scientist.game.config)
    original_word = tuple(int(value) for value in bank["original"]["word"])
    original_strands = int(bank["original"]["strands"])
    upper = int(evaluation["published_crossing_change_cap"])
    total_moves = int(evaluation["total_semantic_move_cap"])
    rows = []
    for index, item in enumerate(items):
        source = source_rows[item.id]
        prefix = [int(action) for action in source["prefix_actions"]]
        objective_cap = ratio * upper + (total_moves - len(prefix))
        screening = [
            _attempt(
                runtime,
                scientist,
                item,
                ratio=ratio,
                simulations=int(evaluation["screening"]["simulations"]),
                seed=int(evaluation["base_seed"]) + index * 100_000,
                objective_cap=objective_cap,
                root_noise=bool(evaluation["screening"]["root_noise"]),
            )
            for _ in range(int(evaluation["screening"]["attempts_per_frontier"]))
        ]
        promotion = []
        if item.id in promoted:
            promotion = [
                _attempt(
                    runtime,
                    scientist,
                    item,
                    ratio=ratio,
                    simulations=int(evaluation["promotion"]["simulations"]),
                    seed=int(evaluation["base_seed"]) + index * 100_000 + 10_000 + attempt,
                    objective_cap=objective_cap,
                    root_noise=bool(evaluation["promotion"]["root_noise"]),
                )
                for attempt in range(int(evaluation["promotion"]["attempts_per_frontier"]))
            ]
        attempts = [*screening, *promotion]
        certificates = [
            certificate
            for attempt in attempts
            if (
                certificate := certify(
                    verifier,
                    original_word,
                    original_strands,
                    prefix,
                    attempt,
                    published_upper_bound=upper,
                    total_semantic_move_cap=total_moves,
                )
            ) is not None
        ]
        rows.append(
            {
                "id": item.id,
                "prefix_depth": len(prefix),
                "frontier_strands": int(source["strands"]),
                "frontier_word_length": len(source["word"]),
                "prediction": predictions[item.id],
                "promoted": item.id in promoted,
                "screening_attempts": screening,
                "promotion_attempts": promotion,
                "certificates": certificates,
            }
        )

    after = network_digest(scientist.network)
    summary = summarize(rows, network_unchanged=before == after)
    report = {
        "schema": "dkt-proof-carrying-inflation-frontier-result-v1",
        "status": "COMPLETED",
        "policy": str(policy_path.resolve()),
        "policy_sha256": sha256(policy_path),
        "bank": str(bank_path.resolve()),
        "bank_sha256": sha256(bank_path),
        "representative_gate": policy["sources"]["representative_gate"],
        "scientist": scientist_name,
        "training_performed": False,
        "network_sha256_before": before,
        "network_sha256_after": after,
        "results": rows,
        "summary": summary,
    }
    atomic_json(output, report)
    atomic_json(
        status_path,
        {
            "schema": "dkt-proof-carrying-inflation-frontier-status-v1",
            "status": "COMPLETED",
            "pid": os.getpid(),
            "completed_unix": time.time(),
            "result": str(output.resolve()),
            "result_sha256": sha256(output),
            "summary": summary,
            "learning": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-evaluate", action="store_true")
    args = parser.parse_args()
    report = run(
        args.policy,
        args.bank,
        args.output,
        evaluation_confirmed=args.confirm_evaluate,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
