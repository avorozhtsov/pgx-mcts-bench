#!/usr/bin/env python3
"""Wrap a final SKM scientist state as an SV2 group-boundary initial state."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pgx_mcts_bench.sv2_curriculum import _load_state, _save_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scientist", required=True)
    parser.add_argument("--f-old", type=int, required=True)
    parser.add_argument("--f-native", type=int, default=5)
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--lineage", required=True)
    args = parser.parse_args()

    source = _load_state(args.source)
    source_scientist = str(source.get("scientist"))
    if source_scientist != args.scientist:
        raise SystemExit(
            f"source scientist {source_scientist!r} does not match {args.scientist!r}"
        )
    required = {"network", "optimizer", "replay"}
    missing = sorted(required - set(source))
    if missing:
        raise SystemExit(f"SKM state is missing required fields: {missing}")

    _save_state(
        args.output,
        {
            "schema": "semantic-v2-exported-scientist-state-v1",
            "scientist": {
                "network": source["network"],
                "optimizer": source["optimizer"],
                "replay": source["replay"],
                "prediction_source": f"skm-lineage:{args.lineage}",
            },
            "f_old": args.f_old,
            "rehearsal_exposure": {},
            "f_native": args.f_native,
            "simulations": args.simulations,
            "controller_reset": {
                "f_native": args.f_native,
                "simulations": args.simulations,
                "reason": "registered SKM-to-Q60 lineage bridge",
            },
            "donation_dose": 1,
            "donation_healthy_streak": 0,
            "lineage": {
                "name": args.lineage,
                "source": str(args.source.resolve()),
                "source_sha256": _sha256(args.source),
                "source_schema": source.get("schema"),
                "source_rehearsal_ids": list(source.get("rehearsal_ids", [])),
            },
        },
    )
    print(f"{args.lineage}={args.output.resolve()}")


if __name__ == "__main__":
    main()
