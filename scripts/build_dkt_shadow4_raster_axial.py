#!/usr/bin/env python3
"""Build four frozen DKT-disjoint minimum-ACS10 shadow examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_q50_1_updated as q50  # noqa: E402


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


def _load_bound(path: Path, expected_hash: str) -> dict[str, Any]:
    if sha256(path) != expected_hash:
        raise ValueError(f"frozen source hash mismatch: {path}")
    return json.loads(path.read_text())


def _dkt_profile(
    panel_payload: dict[str, Any], upper_payload: dict[str, Any]
) -> dict[str, Any]:
    instances = panel_payload.get("instances", [])
    targets = upper_payload.get("targets", [])
    if len(instances) != 72 or len(targets) != 72:
        raise ValueError("the frozen DKT profile must contain exactly 72 rows")
    upper = {
        str(row["canonical_name"]): int(row["paper_workbook_interval"][1])
        for row in targets
    }
    rows = []
    for row in instances:
        payload = row["payload"]
        identity = str(row["source_id"])
        registered_upper = upper[identity]
        rows.append(
            {
                "identity": identity,
                "representation_id": str(row["instance_id"]),
                "strands": int(payload["strands"]),
                "word_length": len(payload["word"]),
                "upper_bound": registered_upper,
                "acs10": (
                    10 * int(payload["strands"])
                    + 5 * registered_upper
                    + len(payload["word"])
                ),
            }
        )
    return {
        "rows": rows,
        "minimum_acs10": min(row["acs10"] for row in rows),
        "strand_support": sorted({row["strands"] for row in rows}),
        "upper_bound_support": sorted({row["upper_bound"] for row in rows}),
        "word_length_min": min(row["word_length"] for row in rows),
        "word_length_max": max(row["word_length"] for row in rows),
        "identities": {row["identity"] for row in rows},
        "representations": {row["representation_id"] for row in rows},
    }


def build(policy_path: Path, output_dir: Path) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text())
    if policy.get("schema") != "dkt-disjoint-shadow4-raster-axial-policy-v1":
        raise ValueError("unknown DKT shadow policy schema")
    sources = policy["sources"]
    loaded = {
        key: _load_bound(Path(value["path"]), str(value["byte_sha256"]))
        for key, value in sources.items()
    }
    profile = _dkt_profile(loaded["dkt_panel"], loaded["dkt_upper_bounds"])

    predecessor_rows = [
        *loaded["q254_prior"]["rows"],
        *loaded["q254_bank"]["rows"],
    ]
    used_ids = {str(row["id"]) for row in predecessor_rows}
    used_names = {str(row["name"]) for row in predecessor_rows}
    used_representations = {q50.representation_id(row) for row in predecessor_rows}
    selection = policy["selection"]
    required_crossings = {
        int(value) for value in selection["required_source_minimal_crossings"]
    }

    candidates = []
    for source in loaded["shadow_source_pool"]["rows"]:
        representation = q50.representation_id(source)
        if (
            str(source["id"]) in used_ids
            or str(source["name"]) in used_names
            or representation in used_representations
            or str(source["name"]) in profile["identities"]
            or representation in profile["representations"]
        ):
            continue
        upper = int(source["certified_unknotting_upper_bound"])
        strands = int(source["strands"])
        word_length = int(source["presentation_crossings"])
        if int(source["source_minimal_crossings"]) not in required_crossings:
            continue
        if strands not in profile["strand_support"]:
            continue
        if upper not in profile["upper_bound_support"]:
            continue
        if not profile["word_length_min"] <= word_length <= profile["word_length_max"]:
            continue
        if float(source["acs10"]) != float(profile["minimum_acs10"]):
            continue
        candidates.append(deepcopy(source))

    candidates.sort(
        key=lambda row: (
            float(row["acs10"]),
            int(row["strands"]),
            int(row["presentation_crossings"]),
            str(row["id"]),
        )
    )
    count = int(selection["count"])
    selected = candidates[:count]
    for row in selected:
        row["representation_id"] = q50.representation_id(row)
    selected_ids = {str(row["id"]) for row in selected}
    selected_names = {str(row["name"]) for row in selected}
    selected_representations = {q50.representation_id(row) for row in selected}
    if (
        len(selected) != count
        or len(selected_ids) != count
        or len(selected_names) != count
        or len(selected_representations) != count
    ):
        raise ValueError("could not select four unique DKT-disjoint shadow rows")

    bank = {
        "schema": "dkt-disjoint-shadow4-bank-v1",
        "name": "dkt-disjoint-shadow4-minimum-acs10",
        "role": "evaluation-only; reserved from all training and selection",
        "size": count,
        "dkt_minimum_acs10": profile["minimum_acs10"],
        "selection_order": ["acs10", "strands", "presentation_crossings", "id"],
        "rows": selected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / "dkt-disjoint-shadow4.json"
    atomic_json(bank_path, bank)
    audit = {
        "schema": "dkt-disjoint-shadow4-selection-audit-v1",
        "status": "passed",
        "policy": str(policy_path.resolve()),
        "policy_sha256": sha256(policy_path),
        "source_hashes": {
            key: sha256(Path(value["path"])) for key, value in sources.items()
        },
        "dkt_profile": {
            "rows": len(profile["rows"]),
            "minimum_acs10": profile["minimum_acs10"],
            "strand_support": profile["strand_support"],
            "upper_bound_support": profile["upper_bound_support"],
            "word_length_range": [
                profile["word_length_min"],
                profile["word_length_max"],
            ],
        },
        "training_rows_excluded": len(predecessor_rows),
        "eligible_minimum_l10_candidates": len(candidates),
        "selected_ids": [str(row["id"]) for row in selected],
        "selected_names": [str(row["name"]) for row in selected],
        "selected_representation_ids": [q50.representation_id(row) for row in selected],
        "selected_acs10": [float(row["acs10"]) for row in selected],
        "checks": {
            "exactly_four_rows": len(selected) == 4,
            "all_match_dkt_minimum_acs10": all(
                float(row["acs10"]) == float(profile["minimum_acs10"])
                for row in selected
            ),
            "no_training_identity_overlap_through_q254": not (
                selected_names & used_names
            ),
            "no_training_representation_overlap_through_q254": not (
                selected_representations & used_representations
            ),
            "no_dkt_identity_overlap": not (selected_names & profile["identities"]),
            "no_dkt_representation_overlap": not (
                selected_representations & profile["representations"]
            ),
            "evaluation_only": True,
            "reserved_from_future_training": True,
        },
        "bank": str(bank_path.resolve()),
        "bank_byte_sha256": sha256(bank_path),
    }
    if not all(audit["checks"].values()):
        raise RuntimeError("DKT shadow selection audit failed")
    atomic_json(output_dir / "dkt-disjoint-shadow4-audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.policy, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
