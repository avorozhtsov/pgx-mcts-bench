"""Training-source identity audits and minimally repaired held-out banks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pgx_mcts_bench.collaborative_scientists import (
    _atomic_json,
    _bank_from_payload,
    _bank_item,
    _bank_payload,
    _cheap_score,
    _compatible_table,
    _json_hash,
)
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import STAGES, _config, candidates


def ladder_source_identity_audit(max_stage: int = 22) -> dict[str, Any]:
    """Identify every explicit ladder source reachable through ``max_stage``."""
    from rf_knots.invariants import invariants

    candidate = next(item for item in candidates() if item.name == "s-window-128")
    config = _config(
        candidate,
        STAGES[max_stage],
        0,
        "cpu",
        frontier=max_stage,
        selfplay_games=1,
    )
    game = make_game(config.game)
    rows = []
    excluded: set[str] = set()
    for source_name in dict.fromkeys(source for source, _ in STAGES[: max_stage + 1]):
        source = next(item for item in game.generator.sources if item.name == source_name)
        result = invariants(source.word, source.strands, identify_knot=True)
        names = sorted(name.strip() for name in (result.name or "").split(" or ") if name.strip())
        excluded.update(names)
        rows.append(
            {
                "source": source_name,
                "word": list(source.word),
                "strands": source.strands,
                "identified_names": names,
                "notes": list(result.notes),
            }
        )
    return {
        "max_stage": max_stage,
        "sources": rows,
        "excluded_knot_ids": sorted(excluded),
    }


def rebuild_anchor_without_ladder_identities(
    run: Path,
    output_bank: Path,
    output_manifest: Path,
    *,
    max_stage: int = 22,
    seed: int = 20260802,
) -> dict[str, Any]:
    """Retain safe NEW70 rows and replace overlaps within their quartiles."""
    base_payload = json.loads((run / "base.json").read_text())
    old_payload = json.loads((run / "new-70.json").read_text())
    base_ids = {str(row["id"]) for row in base_payload}
    audit = ladder_source_identity_audit(max_stage)
    excluded = set(audit["excluded_knot_ids"])
    retained_payload = [row for row in old_payload if str(row["id"]) not in excluded]
    removed = [str(row["id"]) for row in old_payload if str(row["id"]) in excluded]

    compatible = sorted(_compatible_table(), key=lambda knot: (_cheap_score(knot), knot.name))
    quartile_by_id: dict[str, int] = {}
    for quartile, part in enumerate(np.array_split(np.asarray(compatible, dtype=object), 4)):
        quartile_by_id.update((knot.name, quartile) for knot in part)
    old_by_id = {str(row["id"]): row for row in old_payload}
    occupied = base_ids | {str(row["id"]) for row in retained_payload} | excluded
    replacements = []
    for removed_id in removed:
        quartile = int(old_by_id[removed_id]["difficulty_quartile"])
        eligible = [
            knot
            for knot in compatible
            if quartile_by_id[knot.name] == quartile and knot.name not in occupied
        ]
        eligible.sort(
            key=lambda knot: hashlib.sha256(
                f"{seed}:replacement:{removed_id}:{knot.name}".encode()
            ).digest()
        )
        if not eligible:
            raise ValueError(f"no replacement available for {removed_id}")
        replacement = eligible[0]
        occupied.add(replacement.name)
        replacements.append(_bank_item(replacement, quartile))

    retained = _bank_from_payload(retained_payload)
    repaired = sorted(
        retained + replacements,
        key=lambda item: (item.difficulty_quartile, item.id),
    )
    if len(repaired) != len(old_payload):
        raise AssertionError("repaired anchor changed size")
    repaired_ids = {item.id for item in repaired}
    if repaired_ids & base_ids or repaired_ids & excluded:
        raise AssertionError("repaired anchor is not identity-disjoint")
    payload = _bank_payload(repaired)
    _atomic_json(output_bank, payload)
    report = {
        "schema": "ladder-disjoint-anchor-v1",
        "source_run": str(run.resolve()),
        "source_base_sha256": _json_hash(base_payload),
        "source_anchor_sha256": _json_hash(old_payload),
        "output_anchor_sha256": _json_hash(payload),
        "seed": seed,
        "ladder_audit": audit,
        "removed": removed,
        "replacements": [item.id for item in replacements],
        "retained": len(retained),
        "representations": len(repaired),
    }
    _atomic_json(output_manifest, report)
    return report


def build_development_bank(
    source_bank: Path,
    output_bank: Path,
    output_manifest: Path,
    *,
    size: int = 20,
    seed: int = 20260970,
    max_stage: int = 22,
) -> dict[str, Any]:
    """Select an outcome-blind, source-disjoint bank spanning strands and score."""
    if size < 4:
        raise ValueError("development bank needs at least four representations")
    source_payload = json.loads(source_bank.read_text())
    audit = ladder_source_identity_audit(max_stage)
    excluded = set(audit["excluded_knot_ids"])
    eligible = [row for row in source_payload if str(row["id"]) not in excluded]

    # Reserve one quarter each for 3- and 4-strand presentations and use the
    # remainder for 5-strand presentations. Within each group, divide the cheap
    # score order into equal bins and choose by a pinned hash. No solve outcome,
    # prediction, or adaptation result participates in selection.
    quotas = {3: size // 4, 4: size // 4}
    quotas[5] = size - quotas[3] - quotas[4]
    selected: list[dict[str, Any]] = []
    for strands, quota in quotas.items():
        group = sorted(
            (row for row in eligible if int(row["strands"]) == strands),
            key=lambda row: (float(row["cheap_score"]), str(row["id"])),
        )
        if len(group) < quota:
            raise ValueError(f"only {len(group)} eligible {strands}-strand rows; need {quota}")
        for bin_index, part in enumerate(np.array_split(np.asarray(group, dtype=object), quota)):
            candidates_in_bin = list(part)
            candidates_in_bin.sort(
                key=lambda row: hashlib.sha256(
                    f"{seed}:{strands}:{bin_index}:{row['id']}".encode()
                ).digest()
            )
            selected.append(candidates_in_bin[0])

    selected.sort(key=lambda row: (float(row["cheap_score"]), str(row["id"])))
    selected_ids = [str(row["id"]) for row in selected]
    if len(selected_ids) != size or len(set(selected_ids)) != size:
        raise AssertionError("development-bank selection is not unique and complete")
    _atomic_json(output_bank, selected)
    report = {
        "schema": "development-bank-v1",
        "source_bank": str(source_bank.resolve()),
        "source_bank_sha256": _json_hash(source_payload),
        "output_bank_sha256": _json_hash(selected),
        "size": size,
        "seed": seed,
        "max_stage": max_stage,
        "selection": "strand quotas 3:1/4, 4:1/4, 5:remainder; cheap-score bins; pinned hash",
        "uses_outcomes": False,
        "strand_quotas": {str(key): value for key, value in quotas.items()},
        "excluded_knot_ids": sorted(excluded),
        "selected_ids": selected_ids,
        "cheap_score_range": [
            min(float(row["cheap_score"]) for row in selected),
            max(float(row["cheap_score"]) for row in selected),
        ],
    }
    _atomic_json(output_manifest, report)
    return report


def build_critic_calibration_banks(
    protected_banks: list[Path],
    output: Path,
    *,
    train_size: int = 60,
    validation_size: int = 20,
    decision_size: int = 20,
    seed: int = 20261310,
    max_stage: int = 22,
) -> dict[str, Any]:
    """Build identity-disjoint critic splits outside every protected endpoint.

    Selection uses only representation metadata. Each split receives one quarter
    3-strand, one quarter 4-strand, and the remainder 5-strand knots, sampled
    across the full cheap-score range. Outcomes and network predictions never
    participate.
    """
    sizes = {
        "train": train_size,
        "validation": validation_size,
        "decision": decision_size,
    }
    if any(size < 4 for size in sizes.values()):
        raise ValueError("every critic split needs at least four identities")
    protected_payloads = [json.loads(path.read_text()) for path in protected_banks]
    protected_ids = {
        str(row["id"])
        for payload in protected_payloads
        for row in payload
    }
    audit = ladder_source_identity_audit(max_stage)
    excluded = protected_ids | set(audit["excluded_knot_ids"])
    compatible = [knot for knot in _compatible_table() if knot.name not in excluded]

    quotas: dict[str, dict[int, int]] = {}
    for split, size in sizes.items():
        quotas[split] = {3: size // 4, 4: size // 4}
        quotas[split][5] = size - quotas[split][3] - quotas[split][4]

    selected: dict[str, list[Any]] = {split: [] for split in sizes}
    for strands in (3, 4, 5):
        total = sum(quotas[split][strands] for split in sizes)
        group = sorted(
            (knot for knot in compatible if knot.strands == strands),
            key=lambda knot: (_cheap_score(knot), knot.name),
        )
        if len(group) < total:
            raise ValueError(f"only {len(group)} eligible {strands}-strand knots; need {total}")
        chosen = []
        for bin_index, part in enumerate(np.array_split(np.asarray(group, dtype=object), total)):
            ranked = sorted(
                list(part),
                key=lambda knot: hashlib.sha256(
                    f"{seed}:critic:{strands}:{bin_index}:{knot.name}".encode()
                ).digest(),
            )
            chosen.append(ranked[0])
        chosen.sort(
            key=lambda knot: hashlib.sha256(
                f"{seed}:critic:assign:{strands}:{knot.name}".encode()
            ).digest()
        )
        offset = 0
        for split in sizes:
            count = quotas[split][strands]
            selected[split].extend(chosen[offset : offset + count])
            offset += count

    output.mkdir(parents=True, exist_ok=True)
    split_payloads = {}
    for split, knots in selected.items():
        items = [_bank_item(knot, 0) for knot in knots]
        items.sort(key=lambda item: (item.cheap_score, item.id))
        payload = _bank_payload(items)
        split_payloads[split] = payload
        _atomic_json(output / f"{split}.json", payload)

    split_ids = {
        split: {str(row["id"]) for row in payload}
        for split, payload in split_payloads.items()
    }
    if any(split_ids[left] & split_ids[right] for left in sizes for right in sizes if left < right):
        raise AssertionError("critic splits overlap")
    if set().union(*split_ids.values()) & excluded:
        raise AssertionError("critic split overlaps a protected identity")
    report = {
        "schema": "critic-calibration-banks-v1",
        "seed": seed,
        "max_stage": max_stage,
        "uses_outcomes": False,
        "protected_banks": [str(path.resolve()) for path in protected_banks],
        "protected_ids": len(protected_ids),
        "excluded_ladder_ids": sorted(audit["excluded_knot_ids"]),
        "sizes": sizes,
        "quotas": {
            split: {str(strands): count for strands, count in by_strand.items()}
            for split, by_strand in quotas.items()
        },
        "split_sha256": {
            split: _json_hash(payload) for split, payload in split_payloads.items()
        },
        "selected_ids": {
            split: [str(row["id"]) for row in payload]
            for split, payload in split_payloads.items()
        },
    }
    _atomic_json(output / "manifest.json", report)
    return report
