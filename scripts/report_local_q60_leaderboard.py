#!/usr/bin/env python3
"""Build a task-identity-paired leaderboard for the local Q60 population.

The reporter never compares raw prefixes with different task identities.  It
collects every registered lineage, intersects their durably completed
representation IDs, and computes objective coverage, capped loss, innovation,
and first-contact compute only on that exact intersection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

CAPS = {"10.0": 328.0, "1000.0": 20_128.0}


def _events(root: Path) -> list[dict[str, Any]]:
    directory = root / "events"
    if not directory.is_dir():
        return []
    return [
        json.loads(path.read_text())
        for path in sorted(directory.glob("*.json"), key=lambda value: int(value.stem))
    ]


def _state_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with gzip.open(path, "rb") as handle:
        payload = torch.load(handle, map_location="cpu", weights_only=False)
    return list(payload.get("events", []))


def _native_evaluations(row: dict[str, Any]) -> int:
    training = sum(
        int(item.get("scheduled_network_evaluations", 0))
        for item in row.get("iterations", [])
    )
    evaluation = sum(
        int(attempt.get("scheduled_network_evaluations", 0))
        for cell in row.get("evaluation", {}).values()
        for attempt in cell.get("attempts", [])
    )
    return training + evaluation


def _train_steps(row: dict[str, Any]) -> int:
    return sum(int(item.get("train_steps", 0)) for item in row.get("iterations", []))


def _add_events(
    observations: dict[str, dict[str, dict[str, Any]]],
    lineage: str,
    scientist: str,
    events: list[dict[str, Any]],
    *,
    offset: int,
) -> None:
    for event in events:
        representation = event.get("selected")
        row = event.get("scientists", {}).get(scientist)
        if representation is None or row is None:
            continue
        if representation in observations[lineage]:
            raise RuntimeError(f"duplicate representation for {lineage}: {representation}")
        observations[lineage][representation] = {
            "evaluation": row.get("evaluation", {}),
            "native_network_evaluations": _native_evaluations(row),
            "train_steps": _train_steps(row),
            "skip": row.get("curriculum_skip"),
            "curriculum_index": offset + int(event.get("round", 0)),
        }


def _block_paths(root: Path) -> list[Path]:
    directory = root / "blocks"
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def _load_metadata(input_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in ("q20.json", "q40-1.json"):
        payload = json.loads((input_root / name).read_text())
        for row in payload["rows"]:
            result[str(row["id"])] = row
    return result


def _boundary_completed(aligned: Path, label: str) -> int:
    path = aligned / "protocol" / label / "boundary-manifest.json"
    if not path.is_file():
        return 0
    return int(json.loads(path.read_text())["completed_in_source"])


def collect(run_root: Path) -> tuple[
    dict[str, dict[str, dict[str, Any]]], dict[str, list[Path]], dict[str, Any]
]:
    continuation = run_root / "continuation"
    exploratory = continuation / "q4000-v1-exploratory-q40-1-20260818"
    population = continuation / "q4000-v1-population-20260818"
    split = population / "split-20260819"
    aligned = population / "aligned-20260819"
    archive = (
        run_root.parent
        / "nebius-main32-final-20260817"
        / "artifacts/q4000-strand12-20260814/q20-static-no-sharing/state.pt.gz"
    )
    source_events = _state_events(archive)
    observations: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    block_roots: dict[str, list[Path]] = defaultdict(list)

    v3_q40 = exploratory / "q40-1-static-no-sharing"
    v3 = {
        "cyclic-memory-deep-v3": (
            "cyclic-memory-deep-v3",
            continuation / "q4000-v1/q20-static-no-sharing",
        ),
        "cyclic-graph-dual-v3": (
            "cyclic-graph-dual-v3",
            exploratory / "cyclic-graph-dual-v3-q20-static-no-sharing",
        ),
    }
    for lineage, (scientist, q20_root) in v3.items():
        _add_events(observations, lineage, scientist, _events(q20_root), offset=0)
        _add_events(observations, lineage, scientist, _events(v3_q40), offset=20)
        block_roots[lineage].extend((q20_root, v3_q40))

    q_grown = {
        "q-grown:raster-axial-12": "raster-axial-12",
        "q-grown:raster-invariant-combined-dual-12": (
            "raster-invariant-combined-dual-12"
        ),
        "q-grown:strand-graph-12": "strand-graph-12",
    }
    shared_tail = population / "q20-recovery-tail-static-no-sharing"
    for lineage, scientist in q_grown.items():
        slug = lineage.replace("q-grown:", "q-grown-")
        branch = split / "branches" / slug
        _add_events(observations, lineage, scientist, source_events, offset=0)
        _add_events(observations, lineage, scientist, _events(shared_tail), offset=10)
        _add_events(
            observations,
            lineage,
            scientist,
            _events(branch / "q20-remaining-static-no-sharing"),
            offset=16,
        )
        _add_events(
            observations,
            lineage,
            scientist,
            _events(branch / "q40-1-static-no-sharing"),
            offset=20,
        )
        aligned_q40 = aligned / "branches" / slug / "q40-1-aligned-remaining"
        _add_events(
            observations,
            lineage,
            scientist,
            _events(aligned_q40),
            offset=20 + _boundary_completed(aligned, slug),
        )
        block_roots[lineage].extend(
            (
                shared_tail,
                branch / "q20-remaining-static-no-sharing",
                branch / "q40-1-static-no-sharing",
                aligned_q40,
            )
        )

    cyclic_lineage = "q-grown:cyclic-memory-12"
    cyclic_scientist = "cyclic-memory-12"
    cyclic_root = split / "branches/q-grown-cyclic-memory-12"
    _add_events(observations, cyclic_lineage, cyclic_scientist, source_events, offset=0)
    _add_events(
        observations,
        cyclic_lineage,
        cyclic_scientist,
        _events(cyclic_root / "q20-recovery-tail-static-no-sharing"),
        offset=10,
    )
    aligned_cyclic = aligned / "branches/q-grown-cyclic-memory-12"
    _add_events(
        observations,
        cyclic_lineage,
        cyclic_scientist,
        _events(aligned_cyclic / "q20-aligned-remaining"),
        offset=10 + _boundary_completed(aligned, "q-grown-cyclic-memory-12"),
    )
    _add_events(
        observations,
        cyclic_lineage,
        cyclic_scientist,
        _events(cyclic_root / "q40-1-static-no-sharing"),
        offset=20,
    )
    _add_events(
        observations,
        cyclic_lineage,
        cyclic_scientist,
        _events(aligned_cyclic / "q40-1-aligned"),
        offset=20,
    )
    block_roots[cyclic_lineage].extend(
        (
            cyclic_root / "q20-recovery-tail-static-no-sharing",
            cyclic_root / "q40-1-static-no-sharing",
            aligned_cyclic / "q20-aligned-remaining",
            aligned_cyclic / "q40-1-aligned",
        )
    )

    skm = {
        "skm-v2:high-combined-dual": (
            "skm-v2-high-combined-dual",
            "raster-invariant-combined-dual-12",
        ),
        "skm-v2:high-cyclic-memory": (
            "skm-v2-high-cyclic-memory",
            "cyclic-memory-12",
        ),
        "skm-v1:simple-raster-axial": (
            "skm-v1-simple-raster-axial",
            "raster-axial-12",
        ),
        "skm-v1:simple-strand-graph": (
            "skm-v1-simple-strand-graph",
            "strand-graph-12",
        ),
    }
    for lineage, (slug, scientist) in skm.items():
        branch = split / "branches" / slug
        _add_events(
            observations,
            lineage,
            scientist,
            _events(branch / "q20-static-no-sharing"),
            offset=0,
        )
        _add_events(
            observations,
            lineage,
            scientist,
            _events(branch / "q40-1-static-no-sharing"),
            offset=20,
        )
        aligned_branch = aligned / "branches" / slug
        if _boundary_completed(aligned, slug):
            aligned_q20 = None
            aligned_q40 = aligned_branch / "q40-1-aligned-remaining"
            aligned_q40_offset = 20 + _boundary_completed(aligned, slug)
        else:
            aligned_q20 = aligned_branch / "q20-aligned"
            aligned_q40 = aligned_branch / "q40-1-aligned"
            aligned_q40_offset = 20
            _add_events(
                observations,
                lineage,
                scientist,
                _events(aligned_q20),
                offset=0,
            )
        _add_events(
            observations,
            lineage,
            scientist,
            _events(aligned_q40),
            offset=aligned_q40_offset,
        )
        block_roots[lineage].extend(
            (
                branch / "q20-static-no-sharing",
                branch / "q40-1-static-no-sharing",
                *([aligned_q20] if aligned_q20 is not None else []),
                aligned_q40,
            )
        )

    expected = list(v3) + list(q_grown) + [cyclic_lineage] + list(skm)
    return dict(observations), dict(block_roots), {"expected": expected}


def rank(
    observations: dict[str, dict[str, dict[str, Any]]],
    metadata: dict[str, dict[str, Any]],
    expected: list[str],
) -> dict[str, Any]:
    admitted = sorted(lineage for lineage, rows in observations.items() if len(rows) >= 10)
    common_set = (
        set.intersection(*(set(observations[lineage]) for lineage in admitted))
        if admitted
        else set()
    )
    curriculum_indices: dict[str, int] = {}
    for representation in common_set:
        indices = {
            int(observations[lineage][representation]["curriculum_index"])
            for lineage in admitted
        }
        if len(indices) != 1:
            raise RuntimeError(
                f"curriculum index differs on the common panel: {representation} {indices}"
            )
        curriculum_indices[representation] = indices.pop()
    common = sorted(common_set, key=lambda value: curriculum_indices[value])
    latest_block = common[-10:]
    rows: dict[str, dict[str, Any]] = {}
    for lineage in admitted:
        stats: dict[str, Any] = {
            "lineage": lineage,
            "completed_representations": len(observations[lineage]),
            "common_representations": len(common),
            "native_network_evaluations": 0,
            "train_steps": 0,
            "skips": 0,
            "objectives": {},
            "latest_common_block": {
                "representations": len(latest_block),
                "native_network_evaluations": 0,
                "train_steps": 0,
                "objectives": {},
            },
        }
        for ratio in CAPS:
            stats["objectives"][ratio] = {
                "solved": 0,
                "capped_cost": 0.0,
                "capped_average": None,
                "high_strand_solved": 0,
                "win_credit": 0.0,
                "sole_solves": 0,
            }
            stats["latest_common_block"]["objectives"][ratio] = {
                "solved": 0,
                "capped_cost": 0.0,
                "capped_average": None,
                "high_strand_solved": 0,
            }
        for representation in common:
            observation = observations[lineage][representation]
            stats["native_network_evaluations"] += observation[
                "native_network_evaluations"
            ]
            stats["train_steps"] += observation["train_steps"]
            stats["skips"] += int(observation["skip"] is not None)
            for ratio, cap in CAPS.items():
                best = observation["evaluation"].get(ratio, {}).get("best_objective")
                solved = best is not None
                objective = stats["objectives"][ratio]
                objective["solved"] += int(solved)
                objective["capped_cost"] += min(cap, float(best)) if solved else cap
                objective["high_strand_solved"] += int(
                    solved and int(metadata.get(representation, {}).get("strands", 0)) >= 6
                )
        for representation in latest_block:
            observation = observations[lineage][representation]
            latest = stats["latest_common_block"]
            latest["native_network_evaluations"] += observation[
                "native_network_evaluations"
            ]
            latest["train_steps"] += observation["train_steps"]
            for ratio, cap in CAPS.items():
                best = observation["evaluation"].get(ratio, {}).get("best_objective")
                solved = best is not None
                objective = latest["objectives"][ratio]
                objective["solved"] += int(solved)
                objective["capped_cost"] += min(cap, float(best)) if solved else cap
                objective["high_strand_solved"] += int(
                    solved and int(metadata.get(representation, {}).get("strands", 0)) >= 6
                )
        for ratio in CAPS:
            stats["objectives"][ratio]["capped_average"] = (
                stats["objectives"][ratio]["capped_cost"] / len(common)
                if common
                else None
            )
            stats["latest_common_block"]["objectives"][ratio]["capped_average"] = (
                stats["latest_common_block"]["objectives"][ratio]["capped_cost"]
                / len(latest_block)
                if latest_block
                else None
            )
        rows[lineage] = stats

    for ratio in CAPS:
        for representation in common:
            solved = {
                lineage: observations[lineage][representation]["evaluation"]
                .get(ratio, {})
                .get("best_objective")
                for lineage in admitted
            }
            solved = {lineage: value for lineage, value in solved.items() if value is not None}
            if not solved:
                continue
            best = min(float(value) for value in solved.values())
            winners = [lineage for lineage, value in solved.items() if float(value) == best]
            for lineage in winners:
                rows[lineage]["objectives"][ratio]["win_credit"] += 1.0 / len(winners)
            if len(solved) == 1:
                only = next(iter(solved))
                rows[only]["objectives"][ratio]["sole_solves"] += 1

    ordered = sorted(
        rows.values(),
        key=lambda row: (
            -row["objectives"]["10.0"]["solved"],
            -row["objectives"]["1000.0"]["solved"],
            row["objectives"]["10.0"]["capped_average"],
            row["native_network_evaluations"],
            row["lineage"],
        ),
    )
    return {
        "schema": "local-q60-common-identity-leaderboard-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "measurement": (
            "post-native-learning, pre-sharing; exact completed representation intersection"
        ),
        "common_representations": common,
        "common_curriculum_indices": [curriculum_indices[value] for value in common],
        "common_count": len(common),
        "latest_common_block": {
            "representations": latest_block,
            "curriculum_indices": [curriculum_indices[value] for value in latest_block],
        },
        "high_strand_common_count": sum(
            int(metadata.get(representation, {}).get("strands", 0)) >= 6
            for representation in common
        ),
        "admitted_lineages": admitted,
        "waiting_lineages": [lineage for lineage in expected if lineage not in admitted],
        "ranking": ordered,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local Q60 common-identity leaderboard",
        "",
        f"Generated: {report['generated_at']}",
        "",
        (
            f"Common completed panel: **{report['common_count']}** representations; "
            f"6+ strand representations: **{report['high_strand_common_count']}**."
        ),
        "",
        "| Rank | Lineage | Completed | L10 solved | L10 capped avg | "
        "L1000 solved | L1000 capped avg | L10/L1000 wins | Sole solves | "
        "6+ solves | Network evals | Train steps |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(report["ranking"], 1):
        l10 = row["objectives"]["10.0"]
        l1000 = row["objectives"]["1000.0"]
        lines.append(
            f"| {index} | {row['lineage']} | {row['completed_representations']} | "
            f"{l10['solved']}/{report['common_count']} | {l10['capped_average']:.1f} | "
            f"{l1000['solved']}/{report['common_count']} | "
            f"{l1000['capped_average']:.1f} | "
            f"{l10['win_credit']:.1f}/{l1000['win_credit']:.1f} | "
            f"{l10['sole_solves']}/{l1000['sole_solves']} | "
            f"{l10['high_strand_solved']}/{l1000['high_strand_solved']} | "
            f"{row['native_network_evaluations']} | {row['train_steps']} |"
        )
    latest_indices = report["latest_common_block"]["curriculum_indices"]
    latest_label = (
        f"Q{latest_indices[0] + 1}–Q{latest_indices[-1] + 1}"
        if latest_indices
        else "n/a"
    )
    lines.extend(
        (
            "",
            f"## Incremental latest common block: {latest_label}",
            "",
            "| Lineage | New L10 solves | New L1000 solves | Block capped L10 | "
            "Block capped L1000 | Additional net evals | Additional train steps |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in report["ranking"]:
        latest = row["latest_common_block"]
        l10 = latest["objectives"]["10.0"]
        l1000 = latest["objectives"]["1000.0"]
        lines.append(
            f"| {row['lineage']} | {l10['solved']}/{latest['representations']} | "
            f"{l1000['solved']}/{latest['representations']} | "
            f"{l10['capped_average']:.1f} | {l1000['capped_average']:.1f} | "
            f"{latest['native_network_evaluations']} | {latest['train_steps']} |"
        )
    lines.extend(("", "Waiting for a comparable 10-task prefix:"))
    lines.extend(f"- {lineage}" for lineage in report["waiting_lineages"])
    lines.extend(
        (
            "",
            "Unsolved caps: L10 = 328; L1000 = 20128. Raw unequal-prefix "
            "averages are intentionally omitted.",
            "",
        )
    )
    return "\n".join(lines)


def block_signature(block_roots: dict[str, list[Path]]) -> str:
    entries = []
    for lineage, roots in sorted(block_roots.items()):
        for root in roots:
            for path in _block_paths(root):
                entries.append(
                    {
                        "lineage": lineage,
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
            "local-q-skm-ablation-20260817"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    observations, roots, registry = collect(args.run_root)
    metadata = _load_metadata(args.run_root / "inputs/q4000-v1")
    report = rank(observations, metadata, registry["expected"])
    signature = block_signature(roots)
    output = args.run_root / "continuation/q4000-v1-population-20260818/leaderboards"
    output.mkdir(parents=True, exist_ok=True)
    signature_path = output / "block-signature.txt"
    changed = (
        args.force
        or not signature_path.is_file()
        or signature_path.read_text().strip() != signature
    )
    (output / "latest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "latest.md").write_text(markdown(report))
    if changed:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base = output / f"block-{stamp}-common-{report['common_count']:03d}"
        base.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        base.with_suffix(".md").write_text(markdown(report))
        signature_path.write_text(signature + "\n")
        print(f"NEW_BLOCK_LEADERBOARD {base} common={report['common_count']}")
    else:
        print(f"UNCHANGED_BLOCK_SIGNATURE common={report['common_count']}")


if __name__ == "__main__":
    main()
