#!/usr/bin/env python3
"""Select reproducible foundation checkpoints for the raster capacity family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CANDIDATES = ("raster-axial-v2", "raster-axial-v3", "raster-axial-v4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_runs(root: Path, required_stage: int, source: str) -> list[dict]:
    report_path = root / "report.json"
    report = json.loads(report_path.read_text())
    rows = []
    for row in report["runs"]:
        if (
            int(row["highest_stage"]) >= required_stage
            and row["checkpoint_basis"] == "latest-promoted"
            and {
                int(attempt["stage_index"])
                for attempt in row["attempts"]
                if bool(attempt["promoted"])
            }
            == set(range(required_stage + 1))
        ):
            rows.append({**row, "selection_source": source})
    return rows


def select(root: Path, supplementary_root: Path | None = None) -> dict:
    manifest_path = root / "manifest.json"
    report_path = root / "report.json"
    manifest = json.loads(manifest_path.read_text())
    required_stage = int(manifest["stage_limit"]) - 1
    pools = eligible_runs(root, required_stage, "primary")
    supplementary_report_path = None
    if supplementary_root is not None:
        supplementary_manifest = json.loads(
            (supplementary_root / "manifest.json").read_text()
        )
        if int(supplementary_manifest["stage_limit"]) - 1 != required_stage:
            raise SystemExit("primary and supplementary stage limits differ")
        supplementary_report_path = supplementary_root / "report.json"
        pools.extend(
            eligible_runs(supplementary_root, required_stage, "supplementary")
        )
    selected = []
    for candidate in CANDIDATES:
        eligible = sorted(
            (row for row in pools if row["candidate"] == candidate),
            key=lambda row: (
                0 if row["selection_source"] == "primary" else 1,
                int(row["seed"]),
            ),
        )
        if not eligible:
            raise SystemExit(
                f"{candidate} has no seed with objective promotion at every stage"
            )
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
                "selection_source": row["selection_source"],
            }
        )
    return {
        "schema": "raster-axial-capacity-selection-v1",
        "rule": (
            "lowest primary seed with objective promotion at every foundation "
            "stage; otherwise lowest supplementary seed"
        ),
        "analysis_status": (
            "primary" if all(row["selection_source"] == "primary" for row in selected)
            else "exploratory-supplementary-substitution"
        ),
        "foundation_manifest": str(manifest_path.resolve()),
        "foundation_manifest_sha256": sha256(manifest_path),
        "foundation_report_sha256": sha256(report_path),
        "supplementary_report": (
            str(supplementary_report_path.resolve())
            if supplementary_report_path is not None
            else None
        ),
        "supplementary_report_sha256": (
            sha256(supplementary_report_path)
            if supplementary_report_path is not None
            else None
        ),
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("foundation_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--supplementary-root", type=Path)
    args = parser.parse_args()
    payload = select(args.foundation_root, args.supplementary_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
