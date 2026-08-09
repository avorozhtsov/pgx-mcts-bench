#!/usr/bin/env python3
"""Move top-level legacy artifact trees into a provenance-preserving archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


def _bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    source = repo / "artifacts"
    destination = repo / "archive" / "pre-semantic-moves-v1" / "artifacts"
    index_path = repo / "archive" / "pre-semantic-moves-v1" / "artifact-index.json"
    excluded = set(args.exclude)
    existing: dict[str, dict] = {}
    if index_path.exists():
        prior = json.loads(index_path.read_text())
        existing = {entry["name"]: entry for entry in prior.get("entries", [])}

    entries = dict(existing)
    planned: list[tuple[Path, Path, dict]] = []
    for path in sorted(source.iterdir() if source.exists() else []):
        if path.name in excluded:
            continue
        target = destination / path.name
        if target.exists():
            raise SystemExit(f"refusing collision: {target}")
        entry = {
            "name": path.name,
            "original": str(path.relative_to(repo)),
            "archived": str(target.relative_to(repo)),
            "kind": "directory" if path.is_dir() else "file",
            "bytes": _bytes(path),
            "sha256": _sha256(path),
        }
        entries[path.name] = entry
        planned.append((path, target, entry))

    print(json.dumps({"move_count": len(planned), "bytes": sum(x[2]["bytes"] for x in planned)}))
    if args.dry_run:
        return

    destination.mkdir(parents=True, exist_ok=True)
    for path, target, _ in planned:
        shutil.move(str(path), str(target))

    payload = {
        "schema": "pgx-mcts-bench-legacy-artifact-index-v1",
        "updated_at": datetime.now(UTC).isoformat(),
        "entries": [entries[name] for name in sorted(entries)],
    }
    index_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
