#!/usr/bin/env python3
"""Build an offline, fail-closed external evidence package for Mastery targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.external_evidence import (
    build_collection,
    render_coverage_tsv,
    render_summary,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", action="append", type=Path, required=True)
    parser.add_argument("--upperbounds-repo", type=Path, required=True)
    parser.add_argument("--unknotter-repo", type=Path, required=True)
    parser.add_argument("--untangling-number-repo", type=Path, required=True)
    parser.add_argument("--lowerbounds-repo", type=Path, required=True)
    parser.add_argument("--knotinfo-xlsx", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection = build_collection(
        sequences=args.sequence,
        upperbounds_repo=args.upperbounds_repo,
        assessed_repositories=[
            (
                args.unknotter_repo,
                "generated RL/result data inspected; no named complete native action paths",
            ),
            (
                args.untangling_number_repo,
                "notebooks/models inspected; no named complete native action paths",
            ),
            (
                args.lowerbounds_repo,
                "lower-bound computation only; excluded from upper-bound evidence",
            ),
        ],
        knotinfo_xlsx=args.knotinfo_xlsx,
        knotinfo_source_repo=args.unknotter_repo if args.knotinfo_xlsx else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collection_path = args.output_dir / "collection.json"
    summary_path = args.output_dir / "SUMMARY.md"
    coverage_path = args.output_dir / "TARGET_COVERAGE.tsv"
    collection_path.write_text(json.dumps(collection, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(render_summary(collection))
    coverage_path.write_text(render_coverage_tsv(collection))
    manifest = {
        "schema": "external-upper-bound-package-manifest-v1",
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (collection_path, coverage_path, summary_path)
        ],
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(render_summary(collection), end="")


if __name__ == "__main__":
    main()
