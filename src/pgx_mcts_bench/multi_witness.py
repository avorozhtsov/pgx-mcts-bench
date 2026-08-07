"""Mine certified portfolio solutions and freeze receiver-unsolved witness panels."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

from pgx_mcts_bench.adaptive_scientists import load_scientist
from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _json_hash,
    _sha256,
    load_round_state,
    verified_record_cost,
)
from pgx_mcts_bench.sharing_gate import _evaluate


def _screen_item_worker(
    receiver_name: str,
    receiver_checkpoint: str,
    item: Any,
    ratio: float,
    simulations: int,
    games: int,
    seed_blocks: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    receiver = load_scientist(
        receiver_name,
        Path(receiver_checkpoint),
        seed=seed,
        device=device,
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=False,
    )
    attempts = []
    for block in range(seed_blocks):
        row = _evaluate(
            receiver,
            [item],
            ratio=ratio,
            simulations=simulations,
            games=games,
            seed=seed + block * 100_000_000,
        )["rows"][0]
        attempts.extend(row["attempts"])
    solved = [attempt for attempt in attempts if attempt["solved"]]
    return {
        "item": item.id,
        "solved": bool(solved),
        "solved_attempts": len(solved),
        "best_objective": (
            min(float(attempt["objective"]) for attempt in solved) if solved else None
        ),
        "attempts": attempts,
    }


def _source_round(episode_seed: int, *, seed: int, scientist_index: int) -> int | None:
    translated = episode_seed - seed - 900_000_000
    if translated >= 0 and translated % 10_000 < 100:
        return translated // 10_000
    native = episode_seed - seed - 500_000_000 - scientist_index * 10_000
    if native >= 0:
        return native // 1_000_000
    return None


def mine_certified_witnesses(
    source_run: Path,
    *,
    ratio: float = 10.0,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover exact solution provenance from a transactional collaboration run."""
    manifest = json.loads((source_run / "manifest.json").read_text())
    bank_payload = json.loads((source_run / "base.json").read_text())
    by_id = {item.id: item for item in _bank_from_payload(bank_payload)}
    round_dirs = sorted((source_run / "rounds").iterdir())
    if not round_dirs:
        raise ValueError(f"no committed rounds in {source_run}")
    saved = load_round_state(round_dirs[-1], map_location="cpu")
    witnesses: dict[str, dict[str, Any]] = {}
    for scientist_index, checkpoint in enumerate(manifest["checkpoints"]):
        scientist = load_scientist(
            checkpoint["name"],
            Path(checkpoint["path"]),
            seed=int(manifest["seed"]),
            device=device,
            simulations=1,
            require_factorized=True,
            objective_budget_channel=bool(manifest.get("objective_budget", False)),
        )
        replay = saved["scientists"][checkpoint["name"]]["replay"]
        for record in replay.games:
            if not record or record[0].solved < 0.5:
                continue
            round_index = _source_round(
                int(record[0].episode_seed),
                seed=int(manifest["seed"]),
                scientist_index=scientist_index,
            )
            if round_index is None:
                continue
            event_path = source_run / "rounds" / f"{round_index:06d}" / "event.json"
            if not event_path.exists():
                continue
            identity = json.loads(event_path.read_text())["selected"]
            verified = verified_record_cost(
                scientist.game, by_id[identity].knot, ratio, record
            )
            if verified is None:
                continue
            crossings, moves, actions = verified
            objective = ratio * crossings + moves
            incumbent = witnesses.get(identity)
            if incumbent is not None and float(incumbent["objective"]) <= objective:
                continue
            witnesses[identity] = {
                "crossing_changes": crossings,
                "moves": moves,
                "objective": objective,
                "source_native_plies": len(record),
                "source_internal_plies": max(len(record) - moves, 0),
                "semantic_actions": actions,
                "author": checkpoint["name"],
                "episode_seed": int(record[0].episode_seed),
                "source_round": round_index,
            }
    provenance = {
        "schema": "certified-semantic-collaboration-witness-bank-v2",
        "move_metric": "verified portable semantic witness steps",
        "source_run": str(source_run.resolve()),
        "source_manifest_sha256": _json_hash(manifest),
        "source_final_state": str(round_dirs[-1].resolve()),
        "bank": str((source_run / "base.json").resolve()),
        "bank_sha256": _json_hash(bank_payload),
        "ratio": ratio,
        "witnesses": dict(sorted(witnesses.items())),
    }
    return provenance, by_id


def write_certified_witness_bank(
    source_run: Path,
    output: Path,
    *,
    ratio: float = 10.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Materialize a v2 bank after replaying native records semantically."""
    provenance, _ = mine_certified_witnesses(
        source_run, ratio=ratio, device=device
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "witness-bank.json", provenance)
    report = {
        "schema": "certified-semantic-collaboration-witness-bank-report-v1",
        "source_run": str(source_run.resolve()),
        "ratio": ratio,
        "witness_count": len(provenance["witnesses"]),
        "witness_bank": str((output / "witness-bank.json").resolve()),
        "witness_bank_sha256": _json_hash(provenance),
    }
    _atomic_json(output / "report.json", report)
    return report


def _stratified(items: list[Any], size: int) -> list[Any]:
    buckets: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for item in items:
        buckets[(len(item.knot.word) // 4, item.knot.strands)].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.id)
    selected = []
    keys = sorted(buckets)
    while len(selected) < min(size, len(items)):
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < size:
                selected.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def run_multi_witness_screen(
    source_run: Path,
    receiver_name: str,
    receiver_checkpoint: Path,
    output: Path,
    *,
    candidate_ids: tuple[str, ...] = (),
    ratio: float = 10.0,
    simulations: int = 128,
    games: int = 16,
    seed_blocks: int = 1,
    panel_size: int = 8,
    retention_size: int = 8,
    workers: int = 1,
    seed: int = 20260900,
    device: str = "cpu",
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if seed_blocks < 1:
        raise ValueError("seed_blocks must be positive")
    witness_bank, by_id = mine_certified_witnesses(
        source_run, ratio=ratio, device=device
    )
    available = set(witness_bank["witnesses"])
    selected_ids = list(candidate_ids) if candidate_ids else sorted(available)
    missing = sorted(set(selected_ids) - available)
    if missing:
        raise ValueError(f"candidate identities lack certified witnesses: {missing}")
    protocol = {
        "schema": "receiver-unsolved-witness-screen-v1",
        "source_run": str(source_run.resolve()),
        "source_manifest_sha256": witness_bank["source_manifest_sha256"],
        "receiver": receiver_name,
        "receiver_checkpoint": str(receiver_checkpoint.resolve()),
        "receiver_checkpoint_sha256": _sha256(receiver_checkpoint),
        "candidate_ids": selected_ids,
        "ratio": ratio,
        "simulations": simulations,
        "games": games,
        "seed_blocks": seed_blocks,
        "total_attempts_per_item": games * seed_blocks,
        "panel_size": panel_size,
        "retention_size": retention_size,
        "workers": workers,
        "seed": seed,
        "device": device,
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("screen protocol differs from frozen manifest")
    else:
        _atomic_json(manifest_path, protocol)
        _atomic_json(output / "witness-bank.json", witness_bank)
    row_dir = output / "rows"
    row_dir.mkdir(exist_ok=True)
    pending = []
    for item_index, identity in enumerate(selected_ids):
        row_path = row_dir / f"{identity}.json"
        if not row_path.exists():
            pending.append((item_index, identity, row_path))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _screen_item_worker,
                receiver_name,
                str(receiver_checkpoint),
                by_id[identity],
                ratio,
                simulations,
                games,
                seed_blocks,
                seed + item_index * 100_000,
                device,
            ): row_path
            for item_index, identity, row_path in pending
        }
        for future in as_completed(futures):
            _atomic_json(futures[future], future.result())
    rows = [json.loads((row_dir / f"{identity}.json").read_text()) for identity in selected_ids]
    unsolved = [by_id[row["item"]] for row in rows if not row["solved"]]
    solved = [by_id[row["item"]] for row in rows if row["solved"]]
    panel = _stratified(unsolved, panel_size)
    heldout = [item for item in unsolved if item.id not in {x.id for x in panel}]
    retention = _stratified(solved, retention_size)
    report = {
        **protocol,
        "rows": rows,
        "certified_witnesses": len(witness_bank["witnesses"]),
        "panel": [item.id for item in panel],
        "heldout": [item.id for item in heldout],
        "retention": [item.id for item in retention],
        "decision": {
            "passed": len(panel) >= panel_size,
            "receiver_unsolved": len(unsolved),
            "next_step": (
                "run three-arm multi-witness gate"
                if len(panel) >= panel_size
                else "mine more certified witnesses"
            ),
        },
    }
    _atomic_json(output / "report.json", report)
    return report
