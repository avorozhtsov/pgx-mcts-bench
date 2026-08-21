#!/usr/bin/env python3
"""Copy one stopped v1 mastery scientist into a pinned, resumable v2 state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--evidence-snapshot", type=Path, required=True)
    parser.add_argument("--simulation-levels", required=True)
    parser.add_argument("--attempt-wall-seconds-limit", type=float, default=900.0)
    args = parser.parse_args()

    state = args.source / "program-state.json"
    scientist = args.source / "scientist-state.pt.gz"
    if not state.is_file() or not scientist.is_file():
        raise ValueError("source must contain a durable program/scientist state pair")
    if args.destination.exists():
        raise ValueError("destination already exists; refusing to overwrite")
    if not args.evidence_snapshot.is_file():
        raise ValueError("evidence snapshot is missing")
    levels = tuple(sorted(set(int(value) for value in args.simulation_levels.split(","))))
    if not levels or min(levels) < 1:
        raise ValueError("simulation levels must be positive")

    source_hashes = {
        "program_state_sha256": sha256(state),
        "scientist_state_sha256": sha256(scientist),
    }
    shutil.copytree(args.source, args.destination)
    destination_state = args.destination / "program-state.json"
    payload = json.loads(destination_state.read_text())
    if payload.get("schema") not in {
        "multi-knot-mastery-program-v1",
        "multi-knot-mastery-program-v2",
    }:
        raise ValueError("unsupported source state schema")
    config = dict(payload["config"])
    config.update(
        {
            "protocol_version": 2,
            "attempt_wall_seconds_limit": float(args.attempt_wall_seconds_limit),
            "simulation_levels": list(levels),
            "simulation_probe_interval": 20,
            "simulation_probe_lanes": 2,
            "simulation_probe_min_pairs": 12,
            "simulation_success_margin": 0.05,
            "simulation_l1000_tolerance": 5.0,
            "heap_uncertainty_bonus": 0.10,
            "heap_age_bonus": 0.02,
            "heap_cost_penalty": 0.01,
        }
    )
    payload["schema"] = "multi-knot-mastery-program-v2"
    payload["config"] = config
    payload["evidence_snapshot_sha256"] = sha256(args.evidence_snapshot)
    payload.setdefault("outcome_totals", {})
    payload.setdefault("snapshot_distilled_train_steps", 0)
    payload.setdefault("consumed_snapshot_evidence", [])
    payload.setdefault("dose_controller", {"current": levels[0], "observations": {}})
    payload.setdefault(
        "dose_calibration",
        {"bins": 10, "prior_strength": 4.0, "observations": {}, "wall_seconds": {}},
    )
    atomic_json(destination_state, payload)
    manifest = {
        "schema": "mastery-program-v1-to-v2-migration-v1",
        "source": str(args.source),
        "destination": str(args.destination),
        "source_hashes": source_hashes,
        "destination_program_state_sha256": sha256(destination_state),
        "destination_scientist_state_sha256": sha256(args.destination / "scientist-state.pt.gz"),
        "evidence_snapshot": {
            "path": str(args.evidence_snapshot),
            "sha256": sha256(args.evidence_snapshot),
        },
        "simulation_levels": list(levels),
        "attempt_wall_seconds_limit": args.attempt_wall_seconds_limit,
    }
    atomic_json(args.destination / "v2-migration.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
