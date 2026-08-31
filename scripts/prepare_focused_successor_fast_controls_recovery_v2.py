#!/usr/bin/env python3
"""Repair focused Fast inputs by extracting one-scientist Q254 carries."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
ROOT = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818/focused-successor-v1"
)
V1_GATE = ROOT / "FOCUSED_FAST_CONTROLS_VERIFIED.json"
V2_GATE = ROOT / "FOCUSED_FAST_CONTROLS_RECOVERY_V2_VERIFIED.json"

sys.path.insert(0, str(REPO / "scripts"))
import prepare_focused_successor_fast_controls as base  # noqa: E402

V2_SOURCES = base.SOURCES + (
    REPO / "scripts/prepare_focused_successor_fast_controls_recovery_v2.py",
    REPO / "scripts/run_focused_successor_fast_controls_recovery_v2.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    with gzip.open(path, "rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=False)


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        torch.save(value, handle)
    os.replace(temporary, path)


def carry(source: Path, scientist: str, destination: Path) -> None:
    state = load(source)
    save(
        destination,
        {
            "schema": "focused-successor-v1-q254-carry-v2",
            "source_q254_state_sha256": sha256(source),
            "scientist": state["scientists"][scientist],
            "f_old": int(state["f_old"][scientist]),
            "rehearsal_exposure": state["rehearsal_exposure"][scientist],
            "rehearsal_panel_cursor": int(state["rehearsal_panel_cursor"][scientist]),
            "f_native": int(state["f_native"][scientist]),
            "simulations": int(state["simulations"][scientist]),
            "donation_dose": int(state.get("donation_dose", 1)),
            "donation_healthy_streak": int(state.get("donation_healthy_streak", 0)),
            "controller_reset": None,
        },
    )


def main() -> None:
    gate = json.loads(V1_GATE.read_text())
    for line, binding in gate["focused_lines"].items():
        source = Path(binding["initial_state"])
        destination = source.parent / "initial-carry-v2.pt.gz"
        carry(source, binding["scientist"], destination)
        binding["parent_state"] = str(source)
        binding["parent_state_sha256"] = binding["initial_state_sha256"]
        binding["initial_state"] = str(destination)
        binding["initial_state_sha256"] = sha256(destination)
    gate.update(
        verified_at=base.datetime.now(base.UTC).isoformat(),
        recovery="one-scientist-carry-v2",
        predecessor_gate=str(V1_GATE),
        predecessor_gate_sha256=sha256(V1_GATE),
        source_sha256={str(path): sha256(path) for path in V2_SOURCES},
    )
    base.atomic_json(V2_GATE, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
