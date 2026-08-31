#!/usr/bin/env python3
"""Prune redundant legacy mastery group checkpoints with a durable audit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACTS = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts")
MAIN = ARTIFACTS / "nebius-main32-final-20260817/artifacts"
MIRROR = ARTIFACTS / "nebius-semantic-v2-live-backup/mirror"
EXPERIMENTS = ("multi-knot-mastery-v1-20260815", "multi-knot-mastery-v2-20260815")
SEARCH_ROOTS = tuple(
    root / experiment / "scientists"
    for root in (MAIN, MIRROR / "results", MIRROR / "resume")
    for experiment in EXPERIMENTS
)
UNkNOTDB_MODEL = Path(
    "/Users/artemvorozhtsov/projects/unknotdb/models/"
    "q-grown-raster-axial-12-q254-frozen-20260824-v0"
)
DEFAULT_MANIFEST = (
    ARTIFACTS
    / "cleanup-manifests/legacy-mastery-checkpoint-prune-20260829.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def directory_inventory(path: Path) -> dict[str, Any]:
    files = []
    links = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            links.append({"path": str(item), "target": os.readlink(item)})
        elif item.is_file():
            files.append(
                {"path": str(item), "bytes": item.stat().st_size, "sha256": sha256(item)}
            )
    return {
        "path": str(path),
        "bytes": sum(item["bytes"] for item in files),
        "files": files,
        "symlinks": links,
    }


def discover() -> tuple[list[Path], list[Path]]:
    delete: list[Path] = []
    retain: list[Path] = []
    for scientists in SEARCH_ROOTS:
        if not scientists.is_dir():
            continue
        for checkpoints in sorted(scientists.glob("*/checkpoints")):
            groups: list[tuple[int, Path]] = []
            for path in checkpoints.glob("group-[0-9][0-9][0-9]"):
                if path.is_dir():
                    groups.append((int(path.name.removeprefix("group-")), path))
            groups.sort()
            if not groups:
                continue
            numbers = [number for number, _ in groups]
            keep = {numbers[0], *numbers[-2:]}
            keep.update(number for number in numbers if number % 100 == 0)
            for number, path in groups:
                (retain if number in keep else delete).append(path)
    return sorted(delete), sorted(retain)


def verify_unknotdb_model() -> dict[str, Any]:
    manifest_path = UNkNOTDB_MODEL / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = UNkNOTDB_MODEL / manifest["checkpoint"]
    actual = sha256(checkpoint)
    if actual != manifest["checkpoint_sha256"]:
        raise RuntimeError("UnknotDB frozen preprocessing checkpoint hash differs")
    source = Path(manifest["source_q_state"])
    if not source.is_file():
        raise RuntimeError("UnknotDB source Q state is missing; refuse cleanup")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual,
        "source_q_state": str(source),
        "source_q_state_current_sha256": sha256(source),
    }


def verify_no_live_reference(candidates: list[Path]) -> None:
    output = subprocess.run(
        ["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True
    ).stdout
    for path in candidates:
        if str(path) in output:
            raise RuntimeError(f"live process references checkpoint group: {path}")


def build_manifest() -> dict[str, Any]:
    delete, retain = discover()
    protected = verify_unknotdb_model()
    verify_no_live_reference(delete)
    source = Path(protected["source_q_state"])
    for path in delete:
        if path == source or path in source.parents or source in path.parents:
            raise RuntimeError(f"candidate intersects protected UnknotDB source: {path}")
        if not any(root in path.parents for root in SEARCH_ROOTS):
            raise RuntimeError(f"candidate escaped exact legacy roots: {path}")
    deletion_inventory = [directory_inventory(path) for path in delete]
    return {
        "schema": "legacy-mastery-checkpoint-prune-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PREPARED",
        "policy": {
            "scope": list(EXPERIMENTS),
            "retain_first_group": True,
            "retain_every_100_group_boundary": True,
            "retain_latest_groups": 2,
            "active_q254_q154_and_focused_roots_excluded": True,
            "unknotdb_model_and_source_excluded": True,
        },
        "protected_unknotdb": protected,
        "delete": deletion_inventory,
        "delete_directories": len(deletion_inventory),
        "delete_bytes": sum(item["bytes"] for item in deletion_inventory),
        "retain": [directory_inventory(path) for path in retain],
        "retain_directories": len(retain),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    payload = build_manifest()
    atomic_json(args.manifest, payload)
    if args.apply:
        for item in payload["delete"]:
            path = Path(item["path"])
            if not path.is_dir() or not any(root in path.parents for root in SEARCH_ROOTS):
                raise RuntimeError(f"candidate changed before deletion: {path}")
            shutil.rmtree(path)
        payload["status"] = "COMPLETED"
        payload["applied_at"] = datetime.now(UTC).isoformat()
        payload["deleted_directories"] = sum(
            not Path(item["path"]).exists() for item in payload["delete"]
        )
        payload["retained_directories_present"] = sum(
            Path(item["path"]).is_dir() for item in payload["retain"]
        )
        atomic_json(args.manifest, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "delete_directories": payload["delete_directories"],
                "delete_bytes": payload["delete_bytes"],
                "retain_directories": payload["retain_directories"],
                "manifest": str(args.manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
