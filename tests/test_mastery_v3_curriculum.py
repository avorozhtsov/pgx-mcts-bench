from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pgx_mcts_bench.mastery_v3_curriculum import (
    REQUIRED_CAPACITY_STRANDS,
    audit_promotion,
    build_curriculum,
)


def _write_json(path: Path, value: object) -> str:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _row(name: str, strands: int, offset: int) -> dict[str, object]:
    generator = max(1, strands - 1)
    word = [1 if index % 2 == 0 else -generator for index in range(6 + offset % 9)]
    return {"id": name, "name": name, "strands": strands, "word": word}


def _sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    q_root = tmp_path / "q"
    q_root.mkdir()
    clean_rows = [_row(f"simple-{index}", 3 + index % 2, index) for index in range(25)]
    later_rows = []
    for strands, count in ((6, 7), (7, 12), (9, 7), (10, 7), (11, 7), (12, 7)):
        later_rows.extend(
            _row(f"heavy-{strands}-{index}", strands, 100 * strands + index)
            for index in range(count)
        )
    later_rows.extend(_row(f"screen-extra-{index}", 5, 800 + index) for index in range(30))
    groups = []
    for name, rows in (("q20", clean_rows), ("q200-3", later_rows)):
        bank = {"name": name, "rows": rows}
        bank_name = f"{name}.json"
        groups.append(
            {
                "name": name,
                "bank": bank_name,
                "bank_sha256": _write_json(q_root / bank_name, bank),
                "size": len(rows),
            }
        )
    _write_json(q_root / "manifest.json", {"groups": groups})
    dkt = tmp_path / "dkt.json"
    _write_json(
        dkt,
        {
            "instances": [
                {
                    "source_id": "heavy-6-0",
                    "payload": {"strands": 6, "word": [1, -5, 1]},
                }
            ]
        },
    )
    evidence = tmp_path / "evidence.json"
    _write_json(evidence, {"schema": "test-evidence", "items": []})
    return q_root, dkt, evidence


def test_build_curriculum_is_disjoint_and_fills_strand_eight(tmp_path: Path) -> None:
    q_root, dkt, evidence = _sources(tmp_path)
    result = build_curriculum(q_root, dkt, evidence)
    stages = result["stages"]
    assert stages["simple_adaptation"]["row_count"] == 10
    assert stages["heavy_capacity"]["row_count"] == 28
    assert stages["screening"]["row_count"] == 20
    assert stages["heavy_capacity"]["strand_counts"] == {
        str(strands): 4 for strands in REQUIRED_CAPACITY_STRANDS
    }
    eight = [row for row in stages["heavy_capacity"]["rows"] if row["strands"] == 8]
    assert len(eight) == 4
    assert all(row["transformation"]["kind"] == "markov-stabilization" for row in eight)
    assert all(row["transformation"]["preserves_knot_identity"] for row in eight)
    assert result["overlap_audit"] == {
        "training_screening_identity_intersection": [],
        "dkt_training_identity_intersection": [],
        "dkt_screening_identity_intersection": [],
    }
    assert result["sources"]["dkt72_exclusion_panel"]["q4000_identity_overlap_excluded"] == [
        "heavy-6-0"
    ]
    assert result["implementation"]["launch_authorized"]["stages"] == [
        "migration",
        "pretraining",
        "proof-distillation",
        "paired-screening-20",
    ]
    assert result["proof_distillation"]["operational_p_solve_is_separate"] is True


def _passing_arm(curriculum: dict[str, object], *, strict: int, capped: float) -> dict[str, object]:
    return {
        "migration_max_abs_error": 1e-7,
        "migration_tolerance": 1e-6,
        "screening_items": [
            {"representation_id": row["representation_id"], "seed": seed}
            for row in curriculum["stages"]["screening"]["rows"]
            for seed in curriculum["paired_screening_seeds"]
        ],
        "screening_retention": 0.95,
        "retention_by_strand": {str(strands): 0.85 for strands in REQUIRED_CAPACITY_STRANDS},
        "capacity_exceptions": 0,
        "gpu_end_to_end_speedup": 5.2,
        "strict_successes": strict,
        "genuine_improvements": 0,
        "capped_l1000": capped,
        "strict_evidence_per_accelerator_second": 0.01,
    }


def test_promotion_advances_exactly_one_candidate(tmp_path: Path) -> None:
    curriculum = build_curriculum(*_sources(tmp_path))
    report = {
        "arms": {
            "cyclic-memory-deep-v3": _passing_arm(curriculum, strict=2, capped=1400.0),
            "cyclic-graph-dual-v3": _passing_arm(curriculum, strict=3, capped=1500.0),
        }
    }
    audit = audit_promotion(curriculum, report)
    assert audit["status"] == "passed"
    assert audit["winner"] == "cyclic-graph-dual-v3"
    assert audit["advance_to_full_240"] == ["cyclic-graph-dual-v3"]


def test_promotion_fails_closed_on_retention_and_incomplete_pairs(tmp_path: Path) -> None:
    curriculum = build_curriculum(*_sources(tmp_path))
    arm = _passing_arm(curriculum, strict=1, capped=1000.0)
    arm["screening_retention"] = 0.89
    arm["screening_items"] = arm["screening_items"][:-1]
    audit = audit_promotion(
        curriculum,
        {"arms": {candidate: arm for candidate in curriculum["candidates"]}},
    )
    assert audit["status"] == "failed"
    assert audit["advance_to_full_240"] == []
    assert set(audit["candidate_audits"]["cyclic-memory-deep-v3"]["failures"]) >= {
        "screening_key_set",
        "screening_retention",
    }
