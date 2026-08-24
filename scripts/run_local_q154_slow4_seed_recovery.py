#!/usr/bin/env python3
"""Resume slow-4 Q154 after binding the two deferred V3 seed indices."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
POPULATION = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818"
)
ROOT = POPULATION / "q154-slow4-20260822"
GATE = ROOT / "SLOW4_Q154_SEED_INDEX_RECOVERY_VERIFIED.json"
SLOW_SOURCE = REPO / "scripts/run_local_q_slow4_continuation.py"
Q154_SOURCE = REPO / "scripts/run_local_q154_updated_continuation.py"
RECOVERY_SEED_INDICES = {
    "cyclic-memory-deep-v3": 8,
    "cyclic-graph-dual-v3": 9,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _effective_seed_indices(launcher: ModuleType) -> dict[str, int]:
    indices = {str(key): int(value) for key, value in launcher.PRIMARY_8_SEED_INDEX.items()}
    overlap = set(indices) & set(RECOVERY_SEED_INDICES)
    if overlap:
        raise RuntimeError(f"V3 seed indices unexpectedly already registered: {sorted(overlap)}")
    indices.update(RECOVERY_SEED_INDICES)
    return indices


def _verify_gate() -> dict[str, Any]:
    payload = json.loads(GATE.read_text())
    if payload.get("schema") != "slow4-q154-seed-index-recovery-gate-v1":
        raise RuntimeError("unexpected slow-4 seed recovery gate schema")
    if payload.get("v3_seed_indices") != RECOVERY_SEED_INDICES:
        raise RuntimeError("slow-4 seed recovery indices differ from verified gate")
    for raw_path, expected in payload.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"slow-4 seed recovery source hash mismatch: {path}")
    return payload


def main() -> None:
    _verify_gate()
    slow = _load_module("q154_slow4_recovery_wrapper", SLOW_SOURCE)
    if (ROOT / "ALL_SLOW_4_LINEAGES_Q154_COMPLETE").is_file():
        return
    if not slow.SLOW_Q104_MARKER.is_file():
        raise RuntimeError("slow-4 Q104 completion marker is absent")

    lock_handle = slow._acquire_lock()
    try:
        launcher = slow._q154_launcher_module()
        launcher.PRIMARY_8_SEED_INDEX = _effective_seed_indices(launcher)
        slow._q154_launcher_module = lambda: launcher
        slow._status("LAUNCHED", "slow-4 Q154 seed-index recovery")
        slow._run_q154()
        slow._status("COMPLETED", "slow-4 Q154 seed-index recovery")
    except Exception as error:
        slow._status("BLOCKED", "slow-4 Q154 seed-index recovery", repr(error))
        raise
    finally:
        del lock_handle


if __name__ == "__main__":
    main()
