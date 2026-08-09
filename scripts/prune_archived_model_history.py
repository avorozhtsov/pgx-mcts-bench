#!/usr/bin/env python3
"""Keep one latest final binary per archived model identity; delete progress history."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

BINARY_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
DATA_STEMS = {"dataset", "replay"}
PROGRESS_NAMES = {"progress.pt", "distill-progress.pt", "interrupt.pt", "state.pt"}
PROGRESS_PATTERNS = (
    re.compile(r"stage\d+-(?:before|after)\.pt$"),
    re.compile(r"round\d+\.pt$"),
    re.compile(r"(?:alphazero|muzero)-iteration-\d+\.pt$"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_progress(path: Path) -> bool:
    return path.name in PROGRESS_NAMES or any(
        pattern.fullmatch(path.name) for pattern in PROGRESS_PATTERNS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    root = (repo / "archive" / "pre-semantic-moves-v1" / "artifacts").resolve()
    if root == repo or repo not in root.parents or not root.is_dir():
        raise SystemExit(f"refusing unexpected archive root: {root}")

    binaries = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix in BINARY_SUFFIXES
    )
    compressed_states = sorted(root.rglob("state.pt.gz"))
    packed_snapshots = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (".tar.gz" in path.name or ".tar.gz.part-" in path.name)
    )
    keep: set[Path] = {path for path in binaries if path.stem.lower() in DATA_STEMS}
    delete: set[Path] = {path for path in binaries if _is_progress(path)}

    finalists: dict[str, list[Path]] = defaultdict(list)
    for path in binaries:
        if path in keep or path in delete:
            continue
        # The filename is the stable model identity used throughout the old
        # artifacts. Keep the most recently modified final for each identity.
        finalists[path.stem].append(path)
    for paths in finalists.values():
        latest = max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        keep.add(latest)
        delete.update(path for path in paths if path != latest)

    # Collaboration rounds store a compressed model/optimizer bundle per round.
    # Preserve only the latest bundle within each experiment/seed/arm `rounds`
    # directory; all earlier rounds are progress history.
    compressed_by_run: dict[Path, list[Path]] = defaultdict(list)
    for path in compressed_states:
        rounds_root = next(
            (parent for parent in path.parents if parent.name == "rounds"),
            path.parent,
        )
        compressed_by_run[rounds_root].append(path)
    for paths in compressed_by_run.values():
        latest = max(paths, key=lambda path: (path.parent.name, path.stat().st_mtime_ns))
        keep.add(latest)
        delete.update(path for path in paths if path != latest)

    # These transfer archives duplicate checkpoint trees already represented by
    # the retained final binaries and hashes.
    delete.update(packed_snapshots)

    classified = set(binaries) | set(compressed_states) | set(packed_snapshots)
    if keep & delete or keep | delete != classified:
        raise SystemExit("binary classification is incomplete or overlapping")

    kept_rows = [
        {
            "model_identity": path.stem,
            "path": str(path.relative_to(repo)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(keep)
    ]
    deleted_rows = [
        {"path": str(path.relative_to(repo)), "bytes": path.stat().st_size}
        for path in sorted(delete)
    ]
    report_path = repo / "archive" / "pre-semantic-moves-v1" / "binary-retention.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    previous_deleted = previous.get("deleted", [])
    previously_deleted_paths = {row["path"] for row in previous_deleted}
    cumulative_deleted = [
        *previous_deleted,
        *(row for row in deleted_rows if row["path"] not in previously_deleted_paths),
    ]
    report = {
        "schema": "archived-model-binary-retention-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "applied": bool(args.apply),
        "policy": (
            "delete progress/interruption/round/stage binaries; retain datasets and the "
            "newest final binary for each filename-based model identity"
        ),
        "kept_count": len(kept_rows),
        "kept_bytes": sum(row["bytes"] for row in kept_rows),
        "deleted_count": len(cumulative_deleted),
        "deleted_bytes": sum(row["bytes"] for row in cumulative_deleted),
        "kept": kept_rows,
        "deleted": cumulative_deleted,
    }
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("applied", "kept_count", "kept_bytes", "deleted_count", "deleted_bytes")
            }
        )
    )
    if not args.apply:
        return

    for path in delete:
        path.unlink()
    for directory in sorted(root.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
