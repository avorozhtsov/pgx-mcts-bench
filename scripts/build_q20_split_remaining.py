#!/usr/bin/env python3
"""Split the durable coordinated Q20 tail into prior-16 and remaining-4 banks."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--initial-prior-bank", type=Path, required=True)
    parser.add_argument("--tail-bank", type=Path, required=True)
    parser.add_argument("--tail-state", type=Path, required=True)
    parser.add_argument("--remaining-bank", type=Path, required=True)
    parser.add_argument("--prior-bank", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_bank.read_text())
    source_rows = source.get("rows") if isinstance(source, dict) else source
    initial_prior = json.loads(args.initial_prior_bank.read_text())
    initial_rows = initial_prior.get("rows")
    tail = json.loads(args.tail_bank.read_text())
    tail_rows = tail.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != 20:
        raise SystemExit("source bank must be the frozen 20-row Q20 bank")
    if not isinstance(initial_rows, list) or len(initial_rows) != 10:
        raise SystemExit("initial recovery prior must contain exactly 10 rows")
    if not isinstance(tail_rows, list) or len(tail_rows) != 10:
        raise SystemExit("recovery tail must contain exactly 10 rows")

    state = _load_state(args.tail_state)
    processed = [str(value) for value in state.get("processed", [])]
    if len(processed) != 6 or len(set(processed)) != 6:
        raise SystemExit(f"expected durable 6/10 recovery state, got {processed}")
    tail_ids = [str(row["id"]) for row in tail_rows]
    if not set(processed) <= set(tail_ids):
        raise SystemExit("tail state contains an identity outside the recovery tail")

    initial_ids = {str(row["id"]) for row in initial_rows}
    if initial_ids & set(processed):
        raise SystemExit("initial Q10 and completed recovery identities overlap")
    completed_ids = initial_ids | set(processed)
    prior_rows = [row for row in source_rows if str(row["id"]) in completed_ids]
    remaining_rows = [row for row in tail_rows if str(row["id"]) not in set(processed)]
    if len(prior_rows) != 16 or len(remaining_rows) != 4:
        raise SystemExit("split is not exactly prior-16 plus remaining-4")

    original_static_indices = [
        tail_ids.index(str(row["id"])) for row in remaining_rows
    ]
    expected = list(
        range(original_static_indices[0], original_static_indices[0] + len(remaining_rows))
    )
    if original_static_indices != expected:
        raise SystemExit(
            "remaining tail indices are not contiguous; one seed offset cannot preserve seeds"
        )
    seed_offset = original_static_indices[0] * 10_000_000

    prior = {
        "schema": "q4000-split-prior-v1",
        "name": "q20-durable-prior-16",
        "size": 16,
        "rows": prior_rows,
    }
    remaining = copy.deepcopy(tail)
    remaining.update(
        {
            "schema": "q4000-split-remaining-v1",
            "name": "q20-recovery-remaining-4",
            "size": 4,
            "cumulative_representations": 20,
            "rows": remaining_rows,
            "skip_policy": {"maximum_skips": 0},
        }
    )
    manifest = {
        "schema": "q4000-q20-durable-split-v1",
        "source_bank": str(args.source_bank.resolve()),
        "source_bank_sha256": _sha256(args.source_bank),
        "initial_prior_bank": str(args.initial_prior_bank.resolve()),
        "initial_prior_bank_sha256": _sha256(args.initial_prior_bank),
        "tail_bank": str(args.tail_bank.resolve()),
        "tail_bank_sha256": _sha256(args.tail_bank),
        "tail_state": str(args.tail_state.resolve()),
        "tail_state_sha256": _sha256(args.tail_state),
        "completed_recovery_ids": processed,
        "remaining_ids": [str(row["id"]) for row in remaining_rows],
        "remaining_original_static_indices": original_static_indices,
        "single_scientist_seed_offset": seed_offset,
        "selection_uses_outcomes": False,
        "scientists_in_source_state": sorted(state["scientists"]),
    }
    _atomic_json(args.prior_bank, prior)
    _atomic_json(args.remaining_bank, remaining)
    _atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
