#!/usr/bin/env python3
"""Fork one scientist from a coordinated SV2 state into architecture children."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pgx_mcts_bench.adaptive_scientists import load_scientist
from pgx_mcts_bench.architecture_fork import migrate_optimizer_state_by_name
from pgx_mcts_bench.collaborative_scientists import _restore_scientist, _scientist_state
from pgx_mcts_bench.networks import load_policy_value_state_dict
from pgx_mcts_bench.sv2_curriculum import _load_state, _save_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coordinated_state", type=Path)
    parser.add_argument("parent")
    parser.add_argument("parent_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("children", nargs="+")
    parser.add_argument("--seed", type=int, default=20262301)
    args = parser.parse_args()

    coordinated = _load_state(args.coordinated_state)
    exported = coordinated.get("schema") == "semantic-v2-exported-scientist-state-v1"
    parent_saved = coordinated["scientist"] if exported else coordinated["scientists"][args.parent]
    simulations = int(
        coordinated.get("simulations", 64)
        if exported
        else coordinated.get("simulations", {}).get(args.parent, 64)
    )
    parent = load_scientist(
        args.parent,
        args.parent_checkpoint,
        seed=args.seed,
        device="cpu",
        simulations=simulations,
        require_factorized=True,
        objective_budget_channel=True,
    )
    _restore_scientist(parent, parent_saved)

    manifest = {
        "schema": "semantic-v2-architecture-fork-v1",
        "source_state": str(args.coordinated_state.resolve()),
        "parent": args.parent,
        "children": [],
    }
    for offset, name in enumerate(args.children):
        child = load_scientist(
            name,
            args.parent_checkpoint,
            seed=args.seed + (offset + 1) * 100_000_000,
            device="cpu",
            simulations=simulations,
            require_factorized=True,
            objective_budget_channel=True,
        )
        load_policy_value_state_dict(child.network, parent.network.state_dict())
        migration = migrate_optimizer_state_by_name(
            parent.network, parent.optimizer, child.network, child.optimizer
        )
        child.replay = copy.deepcopy(parent.replay)
        child.prediction_source = parent.prediction_source
        destination = args.output / name / "state.pt.gz"
        _save_state(
            destination,
            {
                "schema": "semantic-v2-exported-scientist-state-v1",
                "scientist": _scientist_state(child),
                "f_old": int(
                    coordinated.get("f_old", 1)
                    if exported
                    else coordinated["f_old"][args.parent]
                ),
                "rehearsal_exposure": (
                    coordinated.get("rehearsal_exposure", {})
                    if exported
                    else coordinated["rehearsal_exposure"][args.parent]
                ),
                "f_native": int(
                    coordinated.get("f_native", 10)
                    if exported
                    else coordinated.get("f_native", {}).get(args.parent, 10)
                ),
                "simulations": simulations,
                "donation_dose": int(coordinated.get("donation_dose", 1)),
                "donation_healthy_streak": int(coordinated.get("donation_healthy_streak", 0)),
            },
        )
        manifest["children"].append(
            {
                "name": name,
                "state": str(destination.resolve()),
                "optimizer_parameters_copied": len(migration["copied"]),
                "fresh_parameters": migration["fresh"],
            }
        )
        print(f"{name}={destination.resolve()}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
