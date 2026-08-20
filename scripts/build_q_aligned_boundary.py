#!/usr/bin/env python3
"""Close one durable Q segment and build its exact unprocessed continuation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pgx_mcts_bench.sv2_curriculum import _bank_from_payload, _load_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _rows(payload: Any, label: str) -> list[dict[str, Any]]:
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SystemExit(f"{label} must contain a rows list")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--initial-prior-bank", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--remaining-bank", type=Path, required=True)
    parser.add_argument("--prior-bank", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--order",
        choices=("scheduled", "static"),
        default="scheduled",
        help="scheduled preserves bank-file row order; static sorts by cheap score",
    )
    args = parser.parse_args()

    source_payload = json.loads(args.source_bank.read_text())
    source_rows = _rows(source_payload, "source bank")
    if args.order == "scheduled":
        ordered_rows = list(source_rows)
    else:
        ordered_rows = [
            source_rows[index]
            for index in sorted(
                range(len(source_rows)),
                key=lambda index: (
                    _bank_from_payload([source_rows[index]])[0].cheap_score,
                    str(source_rows[index]["id"]),
                ),
            )
        ]
    ordered_ids = [str(row["id"]) for row in ordered_rows]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise SystemExit("source bank contains duplicate representation IDs")

    state = _load_state(args.state)
    processed = [str(value) for value in state.get("processed", [])]
    if len(set(processed)) != len(processed):
        raise SystemExit("state processed list contains duplicate representation IDs")
    if not set(processed) <= set(ordered_ids):
        raise SystemExit("durable state contains identities outside the source bank")

    initial_payload: Any = {"rows": []}
    initial_rows: list[dict[str, Any]] = []
    if args.initial_prior_bank is not None:
        initial_payload = json.loads(args.initial_prior_bank.read_text())
        initial_rows = _rows(initial_payload, "initial prior bank")
    initial_ids = {str(row["id"]) for row in initial_rows}
    if initial_ids & set(ordered_ids):
        raise SystemExit("initial prior and source banks overlap")

    by_id = {str(row["id"]): row for row in ordered_rows}
    completed_rows = [by_id[item_id] for item_id in processed]
    processed_set = set(processed)
    remaining_rows = [row for row in ordered_rows if str(row["id"]) not in processed_set]
    prior_rows = [*initial_rows, *completed_rows]

    prior = {
        "schema": "q-aligned-boundary-prior-v1",
        "name": f"aligned-prior-{len(prior_rows)}",
        "size": len(prior_rows),
        "rows": prior_rows,
    }
    remaining = copy.deepcopy(source_payload) if isinstance(source_payload, dict) else {}
    remaining.update(
        {
            "schema": "q-aligned-boundary-remaining-v1",
            "name": f"aligned-remaining-{len(remaining_rows)}",
            "size": len(remaining_rows),
            "cumulative_representations": len(prior_rows) + len(remaining_rows),
            "rows": remaining_rows,
            "skip_policy": {
                "maximum_skips": 0,
                "fraction": 0.0,
                "allowed_reasons": [],
                "retained_in_denominators": True,
            },
        }
    )
    manifest = {
        "schema": "q-aligned-compute-boundary-v1",
        "source_bank": str(args.source_bank.resolve()),
        "source_bank_sha256": _sha256(args.source_bank),
        "initial_prior_bank": (
            str(args.initial_prior_bank.resolve()) if args.initial_prior_bank else None
        ),
        "initial_prior_bank_sha256": (
            _sha256(args.initial_prior_bank) if args.initial_prior_bank else None
        ),
        "state": str(args.state.resolve()),
        "state_sha256": _sha256(args.state),
        "scientists": sorted(state["scientists"]),
        "completed_ids": processed,
        "order": args.order,
        "arm": "scheduled-no-sharing" if args.order == "scheduled" else "static-no-sharing",
        "durable_order_is_scheduled_prefix": processed == ordered_ids[: len(processed)],
        "remaining_ids": [str(row["id"]) for row in remaining_rows],
        "completed_in_source": len(processed),
        "source_size": len(source_rows),
        "prior_size": len(prior_rows),
        "remaining_size": len(remaining_rows),
        "selection_uses_outcomes": False,
        "boundary_protocol": {
            "controller_transition": "carry-forward-without-reset",
            "initial_aligned_F_native": 4,
            "F_native_levels": [4, 6, 8, 12, 16],
            "initial_aligned_simulations": 40,
            "simulation_levels": [40, 64, 80, 128, 256],
            "selfplay_games_per_iteration": 4,
            "optimizer_steps_per_iteration": 24,
            "evaluation_attempts_per_objective": 2,
            "adaptive_compute": True,
        },
    }
    _atomic_json(args.prior_bank, prior)
    _atomic_json(args.remaining_bank, remaining)
    _atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
