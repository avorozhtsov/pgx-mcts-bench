#!/usr/bin/env python3
"""Bind multiprocessing-import recovery v3 for focused Fast controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
ROOT = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818/focused-successor-v1"
)
V2_GATE = ROOT / "FOCUSED_FAST_CONTROLS_RECOVERY_V2_VERIFIED.json"
V3_GATE = ROOT / "FOCUSED_FAST_CONTROLS_RECOVERY_V3_VERIFIED.json"

sys.path.insert(0, str(REPO / "scripts"))
import prepare_focused_successor_fast_controls as base  # noqa: E402
import prepare_focused_successor_fast_controls_recovery_v2 as v2  # noqa: E402

SOURCES = v2.V2_SOURCES + (
    REPO / "scripts/prepare_focused_successor_fast_controls_recovery_v3.py",
    REPO / "scripts/run_focused_successor_fast_controls_recovery_v3.py",
)


def main() -> None:
    gate = json.loads(V2_GATE.read_text())
    gate.update(
        verified_at=base.datetime.now(base.UTC).isoformat(),
        recovery="multiprocessing-import-name-v3",
        predecessor_gate=str(V2_GATE),
        predecessor_gate_sha256=v2.sha256(V2_GATE),
        source_sha256={str(path): v2.sha256(path) for path in SOURCES},
    )
    base.atomic_json(V3_GATE, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
