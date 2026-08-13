#!/usr/bin/env python3
"""Split an ordered SV2 bank at a coordinated state's completed prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.sv2_curriculum import _load_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("suffix", type=Path)
    args = parser.parse_args()

    bank = json.loads(args.bank.read_text())
    state = _load_state(args.state)
    processed = [str(value) for value in state["processed"]]
    ids = [str(row["id"]) for row in bank["rows"]]
    if ids[: len(processed)] != processed:
        raise SystemExit("coordinated state is not an exact prefix of the ordered bank")

    for path, rows, role in (
        (args.prefix, bank["rows"][: len(processed)], "processed-prefix"),
        (args.suffix, bank["rows"][len(processed) :], "unprocessed-suffix"),
    ):
        payload = {
            **bank,
            "rows": rows,
            "parent_bank": str(args.bank.resolve()),
            "split_after": len(processed),
            "split_role": role,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"{role}={path.resolve()} rows={len(rows)}")


if __name__ == "__main__":
    main()
