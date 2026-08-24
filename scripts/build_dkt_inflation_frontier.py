#!/usr/bin/env python3
"""Build a deterministic proof-carrying no-CC inflation frontier for one DKT knot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rf_knots.actions import ActionSpec

from pgx_mcts_bench.semantic_verifier import SemanticBraidVerifier


@dataclass(frozen=True)
class FrontierNode:
    word: tuple[int, ...]
    strands: int
    actions: tuple[int, ...]


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


def _stable_key(*values: Any) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _kind(description: str) -> str:
    return description.split("(", 1)[0]


def _sample_actions(
    verifier: SemanticBraidVerifier,
    node: FrontierNode,
    *,
    seed: int,
    inserts: int,
    shuffles: int,
) -> list[int]:
    state = verifier.state(node.word, node.strands)
    grouped: dict[str, list[int]] = {}
    for action in verifier.legal_actions(state, allow_crossing_change=False):
        kind = _kind(str(verifier.spec.describe(action)))
        if kind in {"STABILIZE_POS", "STABILIZE_NEG", "INSERT", "BRAID", "COMMUTE"}:
            grouped.setdefault(kind, []).append(action)

    def choose(actions: list[int], limit: int) -> list[int]:
        return sorted(
            actions,
            key=lambda action: _stable_key(
                seed, node.word, node.strands, node.actions, action
            ),
        )[:limit]

    selected = [
        *grouped.get("STABILIZE_POS", []),
        *grouped.get("STABILIZE_NEG", []),
        *choose(grouped.get("INSERT", []), inserts),
        *choose(
            [*grouped.get("BRAID", []), *grouped.get("COMMUTE", [])],
            shuffles,
        ),
    ]
    return sorted(set(selected))


def _diverse_beam(nodes: list[FrontierNode], width: int, seed: int) -> list[FrontierNode]:
    unique: dict[tuple[tuple[int, ...], int], FrontierNode] = {}
    for node in nodes:
        key = (node.word, node.strands)
        incumbent = unique.get(key)
        if incumbent is None or node.actions < incumbent.actions:
            unique[key] = node
    buckets: dict[tuple[int, int], list[FrontierNode]] = {}
    for node in unique.values():
        buckets.setdefault((node.strands, len(node.word)), []).append(node)
    for values in buckets.values():
        values.sort(key=lambda node: _stable_key(seed, node.word, node.strands, node.actions))
    bucket_order = sorted(buckets, key=lambda key: (-key[0], -key[1], key))
    result: list[FrontierNode] = []
    while len(result) < width and bucket_order:
        remaining = []
        for key in bucket_order:
            if len(result) >= width:
                break
            result.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        bucket_order = remaining
    return result


def generate(
    word: tuple[int, ...],
    strands: int,
    *,
    spec: ActionSpec,
    depths: tuple[int, ...],
    beam_width: int,
    insert_actions_per_parent: int,
    shuffle_actions_per_parent: int,
    seed: int,
) -> list[FrontierNode]:
    verifier = SemanticBraidVerifier(spec)
    beam = [FrontierNode(word, strands, ())]
    exported: list[FrontierNode] = []
    for depth in range(1, max(depths) + 1):
        successors = []
        for node in beam:
            state = verifier.state(node.word, node.strands)
            for action in _sample_actions(
                verifier,
                node,
                seed=seed + depth,
                inserts=insert_actions_per_parent,
                shuffles=shuffle_actions_per_parent,
            ):
                next_state = verifier.apply(
                    state, action, allow_crossing_change=False
                )
                successors.append(
                    FrontierNode(
                        next_state.word,
                        next_state.strands,
                        (*node.actions, action),
                    )
                )
        beam = _diverse_beam(successors, beam_width, seed + depth)
        if depth in depths:
            exported.extend(
                node
                for node in beam
                if node.strands > strands or len(node.word) > len(word)
            )
    return exported


def build(policy_path: Path, output_dir: Path) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text())
    if policy.get("schema") != "dkt-proof-carrying-inflation-frontier-policy-v1":
        raise ValueError("unknown DKT inflation frontier policy schema")
    sources = policy["sources"]
    loaded = {}
    for key in ("dkt_panel", "dkt_upper_bounds", "representative_gate"):
        path = Path(sources[key]["path"])
        if sha256(path) != sources[key]["byte_sha256"]:
            raise ValueError(f"frozen source hash mismatch: {key}")
        loaded[key] = json.loads(path.read_text())

    identity = str(policy["target"]["identity"])
    instance = next(
        row for row in loaded["dkt_panel"]["instances"]
        if str(row["source_id"]) == identity
    )
    upper = next(
        row for row in loaded["dkt_upper_bounds"]["targets"]
        if str(row["canonical_name"]) == identity
    )
    if str(instance["instance_id"]) != policy["target"]["representation_id"]:
        raise ValueError("target representation differs")
    published = int(upper["paper_workbook_interval"][1])
    if published != int(policy["target"]["published_upper_bound"]):
        raise ValueError("target upper bound differs")
    if bool(upper["paper_reports_exact_value"]) != bool(
        policy["target"]["paper_reports_exact_value"]
    ):
        raise ValueError("target exactness differs")

    frontier = policy["frontier"]
    spec = ActionSpec(
        int(frontier["max_len"]),
        int(frontier["max_strands"]),
        bool(frontier["cyclic_band_generators"]),
    )
    original_word = tuple(int(value) for value in instance["payload"]["word"])
    original_strands = int(instance["payload"]["strands"])
    nodes = generate(
        original_word,
        original_strands,
        spec=spec,
        depths=tuple(int(value) for value in frontier["depths"]),
        beam_width=int(frontier["beam_width"]),
        insert_actions_per_parent=int(frontier["insert_actions_per_parent"]),
        shuffle_actions_per_parent=int(frontier["shuffle_actions_per_parent"]),
        seed=int(frontier["seed"]),
    )
    verifier = SemanticBraidVerifier(spec)
    rows = []
    for index, node in enumerate(nodes):
        final = verifier.state(original_word, original_strands)
        for action in node.actions:
            if _kind(str(spec.describe(action))) == "CROSSING_CHANGE":
                raise RuntimeError("inflation prefix contains a crossing change")
            final = verifier.apply(final, action, allow_crossing_change=False)
        if (final.word, final.strands) != (node.word, node.strands):
            raise RuntimeError("frontier prefix replay differs")
        rows.append(
            {
                "id": f"{identity}-inflation-{index:03d}",
                "name": identity,
                "crossings": int(identity.split("a", 1)[0].split("n", 1)[0]),
                "word": list(node.word),
                "strands": node.strands,
                "cheap_score": float(len(node.word) + 8 * (node.strands - 2)),
                "difficulty_quartile": 3,
                "known_unknotting_number": published,
                "certified_unknotting_lower_bound": int(
                    upper["paper_workbook_interval"][0]
                ),
                "prefix_actions": list(node.actions),
                "prefix_action_descriptions": [
                    str(spec.describe(action)) for action in node.actions
                ],
                "prefix_depth": len(node.actions),
                "original_word_length": len(original_word),
                "frontier_word_length": len(node.word),
                "original_strands": original_strands,
                "frontier_strands": node.strands,
                "delta_word_length": len(node.word) - len(original_word),
                "delta_strands": node.strands - original_strands,
            }
        )

    bank = {
        "schema": "dkt-proof-carrying-inflation-frontier-bank-v1",
        "status": "PREPARED",
        "role": "evaluation-only solution mining; no learning",
        "target": identity,
        "published_upper_bound": published,
        "paper_reports_exact_value": bool(upper["paper_reports_exact_value"]),
        "original": {
            "representation_id": instance["instance_id"],
            "word": list(original_word),
            "strands": original_strands,
        },
        "size": len(rows),
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / "dkt-11a14-inflation-frontier.json"
    atomic_json(bank_path, bank)
    audit = {
        "schema": "dkt-proof-carrying-inflation-frontier-audit-v1",
        "status": "passed",
        "policy": str(policy_path.resolve()),
        "policy_sha256": sha256(policy_path),
        "source_hashes": {
            key: sha256(Path(value["path"])) for key, value in sources.items()
        },
        "bank": str(bank_path.resolve()),
        "bank_byte_sha256": sha256(bank_path),
        "frontier_rows": len(rows),
        "depth_counts": {
            str(depth): sum(row["prefix_depth"] == depth for row in rows)
            for depth in frontier["depths"]
        },
        "delta_strands_range": [
            min(row["delta_strands"] for row in rows),
            max(row["delta_strands"] for row in rows),
        ],
        "delta_word_length_range": [
            min(row["delta_word_length"] for row in rows),
            max(row["delta_word_length"] for row in rows),
        ],
        "checks": {
            "nonempty": bool(rows),
            "all_prefixes_replay": True,
            "all_prefixes_have_zero_crossing_changes": True,
            "all_frontiers_inflate": all(
                row["delta_strands"] > 0 or row["delta_word_length"] > 0
                for row in rows
            ),
            "evaluation_only": policy["evaluation"]["learning"] is False,
            "no_pre_clean_evaluation_training": policy["acceptance"][
                "may_enter_training_before_clean_dkt_evaluation"
            ] is False,
        },
    }
    if not all(audit["checks"].values()):
        raise RuntimeError("DKT inflation frontier audit failed")
    atomic_json(output_dir / "dkt-11a14-inflation-frontier-audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.policy, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
