#!/usr/bin/env python3
"""Select one fully promoted seed per invariant-oracle architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CANDIDATES = (
    "raster-invariant-classical",
    "raster-invariant-alexander",
    "raster-invariant-jones",
    "raster-invariant-combined-film",
    "raster-invariant-combined-dual",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    report_path = root / "report.json"
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    required_stage = int(manifest["stage_limit"]) - 1
    selected = []
    for candidate in CANDIDATES:
        eligible = sorted(
            (
                row
                for row in report["runs"]
                if row["candidate"] == candidate
                and int(row["highest_stage"]) >= required_stage
                and row["checkpoint_basis"] == "latest-promoted"
                and {
                    int(attempt["stage_index"])
                    for attempt in row["attempts"]
                    if bool(attempt["promoted"])
                }
                == set(range(required_stage + 1))
            ),
            key=lambda row: int(row["seed"]),
        )
        if not eligible:
            raise SystemExit(f"{candidate} has no fully promoted seed")
        row = eligible[0]
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_file():
            raise SystemExit(f"missing checkpoint: {checkpoint}")
        selected.append(
            {
                "name": candidate,
                "seed": int(row["seed"]),
                "highest_stage": int(row["highest_stage"]),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
    return {
        "schema": "invariant-oracle-selection-v1",
        "rule": "lowest seed with objective promotion and retention at every foundation stage",
        "foundation_manifest": str(manifest_path.resolve()),
        "foundation_manifest_sha256": sha256(manifest_path),
        "foundation_report_sha256": sha256(report_path),
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("foundation_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = select(args.foundation_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
