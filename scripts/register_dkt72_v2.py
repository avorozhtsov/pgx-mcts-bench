#!/usr/bin/env python3
"""Register DKT72-PD-v2 eligibility, selection, protocol, and certificates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.dkt72_v2 import (
    audit_training_leakage,
    certify_improvement,
    freeze_protocol,
    preflight,
    register_checkpoint_eligibility,
    select_two_checkpoint_systems,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("eligibility")
    register.add_argument("--metadata", type=Path, required=True)
    register.add_argument("--panel", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)

    leakage = sub.add_parser("leakage")
    leakage.add_argument("--panel", type=Path, required=True)
    leakage.add_argument("--training-bank", type=Path, action="append", required=True)
    leakage.add_argument("--output", type=Path, required=True)

    select = sub.add_parser("select")
    select.add_argument("--report", type=Path, action="append", required=True)
    select.add_argument("--output", type=Path, required=True)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--panel", type=Path, required=True)
    freeze.add_argument("--selection", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--simulations", type=int, required=True)
    freeze.add_argument("--attempts", type=int, required=True)
    freeze.add_argument("--action-horizon", type=int, required=True)
    freeze.add_argument("--seed", type=int, required=True)

    check = sub.add_parser("preflight")
    check.add_argument("--protocol", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)

    certify = sub.add_parser("certify")
    certify.add_argument("--witness", type=Path, required=True)
    certify.add_argument("--inventory", type=Path, required=True)
    certify.add_argument("--certificates", type=Path, required=True)
    certify.add_argument("--knot", required=True)
    certify.add_argument("--representation", required=True)
    certify.add_argument("--scientist", required=True)
    certify.add_argument("--previous-upper", type=int, required=True)
    certify.add_argument("--declared-lower", type=int, required=True)
    certify.add_argument("--solver-metadata", type=Path, required=True)
    certify.add_argument("--search-parameters", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "eligibility":
        result = register_checkpoint_eligibility(
            json.loads(args.metadata.read_text()), args.panel, args.output
        )
    elif args.command == "leakage":
        result = audit_training_leakage(args.panel, args.training_bank, args.output)
    elif args.command == "select":
        result = select_two_checkpoint_systems(args.report, args.output)
    elif args.command == "freeze":
        result = freeze_protocol(
            panel=args.panel,
            selection=args.selection,
            output=args.output,
            simulations=args.simulations,
            attempts=args.attempts,
            action_horizon=args.action_horizon,
            seed=args.seed,
        )
    elif args.command == "preflight":
        result = preflight(args.protocol, args.output)
    else:
        result = certify_improvement(
            witness_path=args.witness,
            inventory_root=args.inventory,
            certificate_root=args.certificates,
            knot_name=args.knot,
            representation_id=args.representation,
            scientist=args.scientist,
            previous_upper_bound=args.previous_upper,
            declared_lower_bound=args.declared_lower,
            solver_metadata=json.loads(args.solver_metadata.read_text()),
            search_parameters=json.loads(args.search_parameters.read_text()),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
