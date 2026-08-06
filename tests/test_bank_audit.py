import json
from pathlib import Path

from pgx_mcts_bench.bank_audit import (
    build_critic_calibration_banks,
    build_development_bank,
    ladder_source_identity_audit,
)


def test_ladder_source_identity_audit_finds_known_bank_overlaps() -> None:
    audit = ladder_source_identity_audit(22)

    assert "10_124" in audit["excluded_knot_ids"]
    assert "8_19" in audit["excluded_knot_ids"]
    assert any(row["source"] == "T(3,5)" for row in audit["sources"])


def test_development_bank_is_outcome_blind_and_spans_strands(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    rows = []
    for strands, count in ((3, 8), (4, 8), (5, 16)):
        for index in range(count):
            rows.append(
                {
                    "id": f"test-{strands}-{index}",
                    "name": f"test-{strands}-{index}",
                    "strands": strands,
                    "crossings": index + 1,
                    "word": [1] * (index + 1),
                    "cheap_score": float(index),
                    "difficulty_quartile": index // 4,
                    "solved": index % 2 == 0,
                }
            )
    source.write_text(json.dumps(rows))
    output = tmp_path / "development.json"
    manifest = tmp_path / "manifest.json"

    report = build_development_bank(source, output, manifest, size=20, seed=7)
    selected = json.loads(output.read_text())

    assert report["uses_outcomes"] is False
    assert {
        strands: sum(row["strands"] == strands for row in selected) for strands in (3, 4, 5)
    } == {
        3: 5,
        4: 5,
        5: 10,
    }
    assert len({row["id"] for row in selected}) == 20


def test_critic_banks_are_mutually_and_endpoint_disjoint(tmp_path: Path) -> None:
    protected = tmp_path / "protected.json"
    protected.write_text(
        json.dumps(
            [
                {
                    "id": "3_1",
                    "name": "3_1",
                    "strands": 2,
                    "crossings": 3,
                    "word": [1, 1, 1],
                    "cheap_score": 6.0,
                    "difficulty_quartile": 0,
                }
            ]
        )
    )
    output = tmp_path / "critic"
    report = build_critic_calibration_banks(
        [protected], output, train_size=4, validation_size=4, decision_size=4, seed=9
    )
    payloads = {
        split: json.loads((output / f"{split}.json").read_text())
        for split in ("train", "validation", "decision")
    }
    ids = {split: {row["id"] for row in rows} for split, rows in payloads.items()}

    assert report["uses_outcomes"] is False
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["decision"]
    assert not ids["validation"] & ids["decision"]
    assert "3_1" not in set().union(*ids.values())
    for rows in payloads.values():
        assert {strand: sum(row["strands"] == strand for row in rows) for strand in (3, 4, 5)} == {
            3: 1,
            4: 1,
            5: 2,
        }
