from __future__ import annotations

import json
import sqlite3

from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, REDUCE, ActionSpec
from rf_knots.evidence import UnknotWitness

from pgx_mcts_bench.evidence_catalog import EvidenceCatalog, collect


def _trefoil_witness(
    *, cyclic_band_generators: bool = False
) -> tuple[UnknotWitness, list[int]]:
    spec = ActionSpec(
        max_len=32,
        max_strands=4,
        cyclic_band_generators=cyclic_band_generators,
    )
    actions = [
        spec.encode(CROSSING_CHANGE, position=0),
        spec.encode(REDUCE, position=0),
        spec.encode(DESTABILIZE),
    ]
    return UnknotWitness.from_actions((1, 1, 1), 2, spec, actions), actions


def test_collects_qr_and_mastery_witnesses_idempotently(tmp_path):
    witness, actions = _trefoil_witness()
    run = tmp_path / "q-run"
    (run / "events").mkdir(parents=True)
    (run / "bank.json").write_text(
        json.dumps(
            {
                "schema": "test-bank-v1",
                "rows": [
                    {
                        "id": "3_1",
                        "name": "3_1",
                        "dataset_origin": "unit-test",
                        "word": [1, 1, 1],
                        "strands": 2,
                    }
                ],
            }
        )
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "name": "q-test",
                "protocol_sha256": "abc",
                "simulations": 64,
                "action_horizon": 128,
                "checkpoints": {
                    "q-scientist": {
                        "path": "/checkpoints/q-scientist.pt",
                        "sha256": "checkpoint-sha",
                    }
                },
            }
        )
    )
    witness_row = {
        "crossing_changes": 1,
        "semantic_moves": 3,
        "semantic_actions": actions,
        "objective": 1003,
    }
    (run / "events" / "000.json").write_text(
        json.dumps(
            {
                "schema": "test-event-v1",
                "selected": "3_1",
                "scientists": {
                    "q-scientist": {
                        "evaluation": {"1000.0": {"best_witness": witness_row}},
                        "native_best": {},
                    }
                },
            }
        )
    )
    inventory = tmp_path / "inventory" / "witnesses"
    inventory.mkdir(parents=True)
    (inventory / "mastery.json").write_text(
        json.dumps(
            {
                "schema": "mastery-evidence-inventory-row-v1",
                "scientist": "mastery-scientist",
                "sequence_name": "test-sequence",
                "challenge_id": "test-3_1",
                "representation_id": "3_1",
                "knot_name": "3_1",
                "previous_upper_bound": 2,
                "witness": witness.to_dict(),
            }
        )
    )
    output = tmp_path / "catalog"
    database = output / "evidence.sqlite3"

    first = collect(
        database=database,
        output=output,
        experiment_roots=[run],
        mastery_inventories=[inventory.parent],
        metadata_limit=8,
        max_full_invariant_strands=8,
    )
    second = collect(
        database=database,
        output=output,
        experiment_roots=[run],
        mastery_inventories=[inventory.parent],
        metadata_limit=8,
        max_full_invariant_strands=8,
    )

    assert first["counts"]["representations"] == 1
    assert first["counts"]["evidence"] == 2
    assert first["counts"]["eligible_knot_evidence"] == 2
    assert first["counts"]["ordinary_action_evidence"] == 2
    assert first["counts"]["cyclic_band_evidence"] == 0
    assert second["counts"] == first["counts"]
    assert second["activity"]["experiment"][0]["sources"] == 0
    best = json.loads((output / "best-knot-upper-bounds.json").read_text())
    assert best["knots"]["3_1"]["crossing_changes"] == 1
    with sqlite3.connect(database) as connection:
        statuses = connection.execute(
            "SELECT DISTINCT mapping_status FROM evidence"
        ).fetchall()
        contexts = connection.execute(
            """SELECT s.solver_name, s.checkpoint_sha256,
                      p.simulations, p.action_horizon
               FROM evidence_context c
               JOIN solver_versions s USING (solver_version_id)
               JOIN search_protocols p USING (search_protocol_id)
               ORDER BY s.solver_name"""
        ).fetchall()
    assert statuses == [("verified-unique",)]
    assert contexts[0] == ("mastery-scientist", None, None, None)
    assert contexts[1] == ("q-scientist", "checkpoint-sha", 64, 128)
    pool = json.loads((output / "best-solutions-pool.json").read_text())
    solver_pool = pool["verified"]["best_by_representation_and_solver_version"]
    assert len(solver_pool) == 2
    assert next(row for row in solver_pool if row["solver"] == "q-scientist")[
        "simulations"
    ] == 64
    assert all(row["cyclic_band_generators"] is False for row in solver_pool)


def test_same_representation_accepts_ordinary_and_cyclic_band_evidence(tmp_path):
    ordinary, _ = _trefoil_witness()
    cyclic, _ = _trefoil_witness(cyclic_band_generators=True)
    inventory = tmp_path / "inventory" / "witnesses"
    inventory.mkdir(parents=True)
    for scientist, witness in (("ordinary", ordinary), ("bstar", cyclic)):
        (inventory / f"{scientist}.json").write_text(
            json.dumps(
                {
                    "schema": "mastery-evidence-inventory-row-v1",
                    "scientist": scientist,
                    "sequence_name": "test-sequence",
                    "challenge_id": f"test-{scientist}",
                    "representation_id": "3_1",
                    "knot_name": "3_1",
                    "previous_upper_bound": 2,
                    "witness": witness.to_dict(),
                }
            )
        )
    output = tmp_path / "catalog"
    result = collect(
        database=output / "evidence.sqlite3",
        output=output,
        experiment_roots=[],
        mastery_inventories=[inventory.parent],
        metadata_limit=8,
        max_full_invariant_strands=8,
    )

    assert result["counts"]["representations"] == 1
    assert result["counts"]["ordinary_action_evidence"] == 1
    assert result["counts"]["cyclic_band_evidence"] == 1
    assert result["counts"]["representations_with_cyclic_band_evidence"] == 1
    pool = json.loads((output / "best-solutions-pool.json").read_text())
    rows = pool["verified"]["best_by_representation_and_solver_version"]
    assert {row["cyclic_band_generators"] for row in rows} == {False, True}


def test_existing_catalogue_adds_evidence_action_alphabet_column(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE representations (
                instance_id TEXT PRIMARY KEY,
                word_json TEXT NOT NULL,
                strands INTEGER NOT NULL,
                cyclic_band_generators INTEGER NOT NULL,
                first_seen_at TEXT NOT NULL
            );
            CREATE TABLE evidence (
                evidence_id TEXT PRIMARY KEY,
                instance_id TEXT NOT NULL REFERENCES representations(instance_id),
                solver TEXT NOT NULL,
                experiment TEXT NOT NULL,
                objective_ratio TEXT,
                crossing_changes INTEGER NOT NULL,
                moves INTEGER NOT NULL,
                l10 INTEGER NOT NULL,
                l1000 INTEGER NOT NULL,
                witness_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                replay_verified INTEGER NOT NULL CHECK (replay_verified = 1),
                mapping_status TEXT NOT NULL DEFAULT 'pending',
                mapped_knot TEXT,
                eligible_knot_upper_bound INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            """
        )
    catalog = EvidenceCatalog(database)
    catalog.close()
    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(evidence)")
        }
    assert "cyclic_band_generators" in columns


def test_quarantines_unreplayable_event_without_blocking_metadata(tmp_path):
    run = tmp_path / "r-run"
    (run / "events").mkdir(parents=True)
    (run / "bank.json").write_text(
        json.dumps(
            {"rows": [{"id": "3_1", "name": "3_1", "word": [1, 1, 1], "strands": 2}]}
        )
    )
    (run / "events" / "000.json").write_text(
        json.dumps(
            {
                "selected": "3_1",
                "scientists": {
                    "broken": {
                        "evaluation": {
                            "1000.0": {
                                "best_witness": {
                                    "crossing_changes": 1,
                                    "semantic_moves": 1,
                                    "semantic_actions": [999999],
                                }
                            }
                        }
                    }
                },
            }
        )
    )
    output = tmp_path / "catalog"
    result = collect(
        database=output / "evidence.sqlite3",
        output=output,
        experiment_roots=[run],
        mastery_inventories=[],
        metadata_limit=8,
        max_full_invariant_strands=8,
    )
    assert result["counts"]["evidence"] == 0
    assert result["counts"]["quarantine"] == 1
    assert "witness replay failed" in (output / "quarantine.jsonl").read_text()


def test_external_claim_stays_separate_and_is_not_distillation_evidence(tmp_path):
    package = tmp_path / "external.json"
    package.write_text(
        json.dumps(
            {
                "schema": "external-test-v1",
                "records": [
                    {
                        "evidence_id": "external-claim-1",
                        "classification": "external-compositional-pd-upper-bound-claim",
                        "canonical_name": "12n_570",
                        "claimed_new_upper": 2,
                        "old_bound": [1, 3],
                        "replay_verified": False,
                        "distillation_eligible": True,
                        "source": {
                            "repository_url": "https://example.invalid/repository",
                            "commit": "abc123",
                        },
                    }
                ],
            }
        )
    )
    output = tmp_path / "catalog"
    result = collect(
        database=output / "evidence.sqlite3",
        output=output,
        experiment_roots=[],
        mastery_inventories=[],
        external_evidence_paths=[package],
        metadata_limit=0,
        max_full_invariant_strands=8,
    )

    assert result["counts"]["evidence"] == 0
    assert result["counts"]["external_records"] == 1
    assert result["counts"]["external_claim_only"] == 1
    assert result["counts"]["external_distillation_eligible"] == 0
    pool = json.loads((output / "best-solutions-pool.json").read_text())
    assert pool["verified"]["best_by_knot"] == {}
    assert pool["external"][0]["verification_tier"] == "claim-only"
    assert pool["external"][0]["distillation_eligible"] is False
