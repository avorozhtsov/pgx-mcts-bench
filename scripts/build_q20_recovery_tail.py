#!/usr/bin/env python3
"""Build a provenance-bearing Q20 recovery tail from a saved coordinated state."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pgx_mcts_bench.sv2_curriculum import _load_state


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


def _strand_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    strands = [int(row["strands"]) for row in rows]
    return {
        "strands_ge_6": sum(value >= 6 for value in strands),
        "strands_ge_9": sum(value >= 9 for value in strands),
        "strands_ge_10": sum(value >= 10 for value in strands),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--tail-bank", type=Path, required=True)
    parser.add_argument("--prior-bank", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_bank.read_text())
    rows = source.get("rows") if isinstance(source, dict) else source
    if not isinstance(rows, list) or len(rows) != 20:
        raise SystemExit("the recovery source must be the frozen 20-row Q20 bank")

    state = _load_state(args.source_state)
    processed = [str(value) for value in state.get("processed", [])]
    processed_set = set(processed)
    source_ids = [str(row["id"]) for row in rows]
    if len(processed) != 10 or len(processed_set) != 10:
        raise SystemExit(f"expected exactly 10 completed Q20 identities, got {processed}")
    if not processed_set <= set(source_ids):
        raise SystemExit("saved state contains identities outside the frozen Q20 bank")

    prior_rows = [row for row in rows if str(row["id"]) in processed_set]
    tail_rows = [row for row in rows if str(row["id"]) not in processed_set]
    if len(prior_rows) != 10 or len(tail_rows) != 10:
        raise SystemExit("Q20 recovery split is not exactly 10 completed plus 10 remaining")

    prior = {
        "schema": "q4000-recovery-prior-v1",
        "name": "q20-completed-prefix-recovery",
        "size": len(prior_rows),
        "rows": prior_rows,
    }
    tail = copy.deepcopy(source)
    tail.update(
        {
            "schema": "q4000-recovery-tail-v1",
            "name": "q20-recovery-tail",
            "size": len(tail_rows),
            "cumulative_representations": 20,
            "rows": tail_rows,
            "strand_counts": _strand_counts(tail_rows),
            "skip_policy": {
                "maximum_skips": 0,
                "reason": "10-row recovery tail has a floor(5%) skip allowance of zero",
            },
        }
    )
    tail["strand_quotas"] = {
        "strands_ge_6_required": tail["strand_counts"]["strands_ge_6"],
        "strands_ge_9_required": tail["strand_counts"]["strands_ge_9"],
        "strands_ge_10_required": tail["strand_counts"]["strands_ge_10"],
    }

    manifest = {
        "schema": "q4000-q20-recovery-split-v1",
        "source_bank": str(args.source_bank.resolve()),
        "source_bank_sha256": _sha256(args.source_bank),
        "source_state": str(args.source_state.resolve()),
        "source_state_sha256": _sha256(args.source_state),
        "selection_uses_outcomes": False,
        "selection_rule": "remaining identities are exactly source rows absent from state.processed",
        "completed_ids_in_source_order": [str(row["id"]) for row in prior_rows],
        "remaining_ids_in_source_order": [str(row["id"]) for row in tail_rows],
        "scientists_in_source_state": sorted(state["scientists"]),
    }

    _atomic_json(args.prior_bank, prior)
    _atomic_json(args.tail_bank, tail)
    _atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
