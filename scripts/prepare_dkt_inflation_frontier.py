#!/usr/bin/env python3
"""Prepare an exact hash gate for the DKT 11a_14 inflation-frontier pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dkt_inflation_frontier as builder  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def prepare(
    policy_path: Path, protocol_dir: Path, *, verified_tests: int
) -> dict[str, Any]:
    if verified_tests < 1:
        raise ValueError("verified test count must be positive")
    audit = builder.build(policy_path, protocol_dir)
    bank_path = Path(audit["bank"])
    audit_path = protocol_dir / "dkt-11a14-inflation-frontier-audit.json"
    sources = [
        policy_path,
        ROOT / "scripts/build_dkt_inflation_frontier.py",
        ROOT / "scripts/prepare_dkt_inflation_frontier.py",
        ROOT / "scripts/run_dkt_inflation_frontier.py",
        ROOT / "tests/test_dkt_inflation_frontier.py",
    ]
    gate = {
        "schema": "dkt-proof-carrying-inflation-frontier-prepared-v1",
        "status": "PREPARED",
        "prepared": True,
        "scientific_role": "evaluation-only DKT solution mining; no learning",
        "policy": str(policy_path.resolve()),
        "policy_sha256": builder.sha256(policy_path),
        "bank": str(bank_path.resolve()),
        "bank_sha256": builder.sha256(bank_path),
        "audit": str(audit_path.resolve()),
        "audit_sha256": builder.sha256(audit_path),
        "verified_tests": verified_tests,
        "source_sha256": {
            str(path.resolve()): builder.sha256(path) for path in sources
        },
        "launch": {
            "python": str((ROOT / ".venv/bin/python").resolve()),
            "script": str((ROOT / "scripts/run_dkt_inflation_frontier.py").resolve()),
            "arguments": [
                "--policy",
                str(policy_path.resolve()),
                "--bank",
                str(bank_path.resolve()),
                "--output",
                str((protocol_dir.parent / "result.json").resolve()),
                "--confirm-evaluate",
            ],
        },
    }
    gate_path = protocol_dir.parent / "DKT_11A14_INFLATION_FRONTIER_PREPARED.json"
    builder.atomic_json(gate_path, gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--verified-tests", type=int, required=True)
    args = parser.parse_args()
    gate = prepare(
        args.policy, args.protocol_dir, verified_tests=args.verified_tests
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
