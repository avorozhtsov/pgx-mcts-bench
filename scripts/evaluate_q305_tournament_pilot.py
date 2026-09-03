#!/usr/bin/env python3
"""Equal-budget exact-common evaluation of Q304 parent and Q314 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.collaborative_scientists import _restore_scientist
from pgx_mcts_bench.sv2_curriculum import (
    _bank_from_payload,
    _bank_rows,
    _evaluate,
    _load_roster,
    _load_state,
    _sha256,
)


def capped(cell: dict, ratio: float, horizon: int) -> float:
    value = cell.get("best_objective")
    return float(value) if value is not None else ratio * 20.0 + horizon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-state", type=Path, required=True)
    parser.add_argument("--pilot-state", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, rows = _bank_rows(args.bank)
    items = _bank_from_payload(rows)[:10]
    ratios = (10.0, 1000.0)
    results = {}
    for label, state_path in (("q304-parent", args.parent_state), ("q314-pilot", args.pilot_state)):
        scientist = _load_roster(
            {"strand-graph-12": args.checkpoint},
            seed=202609031000,
            device="cpu",
            simulations=40,
            action_horizon=128,
        )[0]
        state = _load_state(state_path)
        restored = state.get("scientists", {}).get("strand-graph-12", state.get("scientist"))
        if restored is None:
            raise RuntimeError(f"missing scientist state in {state_path}")
        _restore_scientist(scientist, restored)
        cells = {}
        for index, item in enumerate(items):
            cells[item.id] = _evaluate(
                scientist,
                item.knot,
                ratios=ratios,
                attempts=2,
                simulations=40,
                seed=202609031100 + index * 100_000,
                add_root_noise=True,
            )
        results[label] = {
            "state": str(state_path),
            "state_sha256": _sha256(state_path),
            "cells": cells,
        }
    summary = {}
    for label, row in results.items():
        summary[label] = {}
        for ratio in ratios:
            key = str(ratio)
            values = [cell[key] for cell in row["cells"].values()]
            summary[label][key] = {
                "solved": sum(cell.get("best_objective") is not None for cell in values),
                "capped_objective": sum(capped(cell, ratio, 128) for cell in values),
            }
    pilot = summary["q314-pilot"]
    parent = summary["q304-parent"]
    passed = all(
        pilot[str(ratio)]["solved"] >= parent[str(ratio)]["solved"]
        and pilot[str(ratio)]["capped_objective"] <= parent[str(ratio)]["capped_objective"]
        for ratio in ratios
    )
    payload = {
        "schema": "q305-same-root-tournament-exact-common-gate-v1",
        "status": "PREPARED" if passed else "BLOCKED",
        "passed": passed,
        "protocol": {
            "representations": [item.id for item in items],
            "ratios": list(ratios),
            "attempts": 2,
            "simulations": 40,
            "paired_seeds": True,
            "root_noise": True,
        },
        "inputs": {
            "checkpoint_sha256": _sha256(args.checkpoint),
            "bank_sha256": _sha256(args.bank),
        },
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"passed": passed, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
