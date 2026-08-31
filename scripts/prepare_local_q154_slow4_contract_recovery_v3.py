#!/usr/bin/env python3
"""Build exact gates for the contract-only Slow Q154 resume repair."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
POPULATION = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818"
)
ROOT = POPULATION / "q154-slow4-20260822"
PRIMARY = POPULATION / "q154-updated-20260819"
OLD_SPLIT_GATE = PRIMARY / "FAST6_SLOW4_COHORT_SPLIT_V5_VERIFIED.json"
NEW_SPLIT_GATE = ROOT / "FAST6_SLOW4_COHORT_SPLIT_V5_CONTRACT_TRANSITION_VERIFIED.json"
OLD_RECOVERY_GATE = ROOT / "SLOW4_Q154_SEED_INDEX_RECOVERY_V2_VERIFIED.json"
NEW_RECOVERY_GATE = ROOT / "SLOW4_Q154_CONTRACT_TRANSITION_RECOVERY_V3_VERIFIED.json"
CONTRACT = REPO / "research/local-q-skm-ablation/EXECUTION-CONTRACT.md"
LAUNCHER = REPO / "scripts/run_local_q154_slow4_contract_recovery_v3.py"
TEST = REPO / "tests/test_q154_slow4_contract_recovery_v3.py"
LAUNCHD_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    old = json.loads(OLD_SPLIT_GATE.read_text())
    contract_key = str(CONTRACT)
    old_contract_hash = old["source_sha256"][contract_key]
    new_contract_hash = sha256(CONTRACT)
    if old_contract_hash == new_contract_hash:
        raise RuntimeError("execution contract has not changed")
    replacement = dict(old)
    replacement["source_sha256"] = dict(old["source_sha256"])
    replacement["source_sha256"][contract_key] = new_contract_hash
    replacement["contract_transition"] = {
        "authorized_program_change_date": "2026-08-29",
        "old_contract_sha256": old_contract_hash,
        "new_contract_sha256": new_contract_hash,
        "protocol_source_changes": "none",
        "active_transaction_policy": "finish Slow Q154 exactly as previously registered",
    }
    replacement["verified_at"] = datetime.now(UTC).isoformat()
    atomic_json(NEW_SPLIT_GATE, replacement)

    source_hashes = {
        str(path): sha256(path)
        for path in (LAUNCHER, TEST, CONTRACT, OLD_SPLIT_GATE, OLD_RECOVERY_GATE)
    }
    recovery_gate = {
        "schema": "slow4-q154-contract-transition-recovery-gate-v3",
        "passed": True,
        "sharing": "strict-none",
        "launchd_path": LAUNCHD_PATH,
        "failure_repaired": "authorized execution-contract-only hash transition at Q134 resume",
        "old_split_gate_sha256": sha256(OLD_SPLIT_GATE),
        "old_recovery_gate_sha256": sha256(OLD_RECOVERY_GATE),
        "replacement_split_gate": str(NEW_SPLIT_GATE),
        "replacement_split_gate_sha256": sha256(NEW_SPLIT_GATE),
        "source_hashes": source_hashes,
        "verified_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(NEW_RECOVERY_GATE, recovery_gate)
    print(sha256(NEW_SPLIT_GATE), NEW_SPLIT_GATE)
    print(sha256(NEW_RECOVERY_GATE), NEW_RECOVERY_GATE)


if __name__ == "__main__":
    main()
