#!/usr/bin/env python3
"""Convert a frozen SKM sequence into the equivalent static Q bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from register_local_q_skm_ablation import q_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace {args.output}")
    source = json.loads(args.sequence.read_text())
    challenges = list(source["challenges"])
    payload = {
        "schema": "q-from-mastery-sequence-v1",
        "name": f"q-static-{source['name']}",
        "size": len(challenges),
        "source_sequence": str(args.sequence.resolve()),
        "source_sequence_sha256": hashlib.sha256(args.sequence.read_bytes()).hexdigest(),
        "skip_policy": {
            "maximum_skips": len(challenges) // 20,
            "failures_remain_in_denominator": True
        },
        "rows": [q_row(row, index) for index, row in enumerate(challenges)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
