"""Incremental, replay-verified evidence catalogue for Q, R, and mastery runs.

The catalogue keeps exact braid representations separate from claims about the
knot type they close to.  Every witness is replayed before admission.  Knot
names are promoted to a usable upper-bound index only when computed invariants
are compatible with the source claim; ambiguous or conflicting mappings remain
inspectable without silently becoming knot-level facts.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rf_knots.actions import ActionSpec
from rf_knots.evidence import UnknotWitness
from rf_knots.invariants import (
    alexander_polynomial,
    invariants,
    signature,
    to_pairs,
)
from rf_knots.knot_table import canonical_name

SCHEMA = "q-r-skm-evidence-catalog-v2"
METADATA_ALGORITHM = "rf-knots-invariants-v1"
DEFAULT_MAX_FULL_INVARIANT_STRANDS = 8


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def _candidate_names(name: str | None) -> tuple[str, ...]:
    if not name:
        return ()
    names = []
    for value in name.split(" or "):
        canonical = canonical_name(value.strip())
        if canonical is not None:
            names.append(canonical)
    return tuple(sorted(set(names)))


def _replay_flat_witness(
    bank_row: dict[str, Any],
    witness_row: dict[str, Any],
    *,
    cyclic_band_generators: bool = False,
) -> tuple[UnknotWitness, ActionSpec]:
    word = tuple(int(value) for value in bank_row["word"])
    strands = int(bank_row["strands"])
    actions = tuple(int(value) for value in witness_row["semantic_actions"])
    expected = (
        int(witness_row["crossing_changes"]),
        int(witness_row["semantic_moves"]),
    )
    matches: list[tuple[UnknotWitness, ActionSpec]] = []
    for max_len in (32, 48, 64, 96, 128, 192, 256):
        if max_len < len(word):
            continue
        for max_strands in range(max(strands, 2), 13):
            spec = ActionSpec(
                max_len=max_len,
                max_strands=max_strands,
                cyclic_band_generators=cyclic_band_generators,
            )
            if actions and max(actions) >= spec.num_actions:
                continue
            try:
                replayed = UnknotWitness.from_actions(word, strands, spec, actions)
            except (IndexError, ValueError):
                continue
            if (replayed.crossing_changes, replayed.moves) == expected:
                matches.append((replayed, spec))
    if not matches:
        raise ValueError("no supported ActionSpec replays the reported witness")
    return min(matches, key=lambda item: (item[1].max_len, item[1].max_strands))


class EvidenceCatalog:
    """Single-writer SQLite catalogue with append-only evidence rows."""

    def __init__(self, database: Path) -> None:
        self.database = database
        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS catalogue_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS source_scans (
                source_path TEXT NOT NULL,
                scan_digest TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                scanned_at TEXT NOT NULL,
                admitted INTEGER NOT NULL,
                quarantined INTEGER NOT NULL,
                PRIMARY KEY (source_path, scan_digest)
            );
            CREATE TABLE IF NOT EXISTS representations (
                instance_id TEXT PRIMARY KEY,
                word_json TEXT NOT NULL,
                strands INTEGER NOT NULL,
                cyclic_band_generators INTEGER NOT NULL,
                first_seen_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS identity_claims (
                claim_id TEXT PRIMARY KEY,
                instance_id TEXT NOT NULL REFERENCES representations(instance_id),
                declared_name TEXT,
                canonical_name TEXT,
                dataset_origin TEXT,
                source_kind TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence (
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
                cyclic_band_generators INTEGER NOT NULL DEFAULT 0,
                replay_verified INTEGER NOT NULL CHECK (replay_verified = 1),
                mapping_status TEXT NOT NULL DEFAULT 'pending',
                mapped_knot TEXT,
                eligible_knot_upper_bound INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence_sources (
                evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                source_channel TEXT NOT NULL,
                PRIMARY KEY (evidence_id, source_path, source_channel)
            );
            CREATE TABLE IF NOT EXISTS representation_metadata (
                instance_id TEXT PRIMARY KEY REFERENCES representations(instance_id),
                algorithm TEXT NOT NULL,
                status TEXT NOT NULL,
                invariants_json TEXT,
                identified_candidates_json TEXT NOT NULL,
                computed_at TEXT NOT NULL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS quarantine (
                item_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_sha256 TEXT,
                reason TEXT NOT NULL,
                context_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS solver_versions (
                solver_version_id TEXT PRIMARY KEY,
                solver_name TEXT NOT NULL,
                checkpoint_path TEXT,
                checkpoint_sha256 TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS search_protocols (
                search_protocol_id TEXT PRIMARY KEY,
                simulations INTEGER,
                action_horizon INTEGER,
                objective_ratio TEXT,
                protocol_sha256 TEXT,
                parameters_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence_context (
                evidence_id TEXT PRIMARY KEY REFERENCES evidence(evidence_id),
                solver_version_id TEXT NOT NULL REFERENCES solver_versions(solver_version_id),
                search_protocol_id TEXT NOT NULL REFERENCES search_protocols(search_protocol_id),
                source_family TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS science_collections (
                collection_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                collection_kind TEXT NOT NULL,
                description TEXT NOT NULL,
                policy_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS science_collection_evidence (
                collection_id TEXT NOT NULL REFERENCES science_collections(collection_id),
                evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
                reason TEXT NOT NULL,
                PRIMARY KEY (collection_id, evidence_id)
            );
            CREATE TABLE IF NOT EXISTS external_records (
                external_record_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                record_type TEXT NOT NULL,
                canonical_name TEXT,
                claimed_upper INTEGER,
                bound_interval_json TEXT,
                verification_tier TEXT NOT NULL,
                replay_verified INTEGER NOT NULL,
                distillation_eligible INTEGER NOT NULL,
                l10 INTEGER,
                l1000 INTEGER,
                source_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS science_collection_external_records (
                collection_id TEXT NOT NULL REFERENCES science_collections(collection_id),
                external_record_id TEXT NOT NULL REFERENCES external_records(external_record_id),
                reason TEXT NOT NULL,
                PRIMARY KEY (collection_id, external_record_id)
            );
            CREATE INDEX IF NOT EXISTS evidence_instance_idx ON evidence(instance_id);
            CREATE INDEX IF NOT EXISTS claims_instance_idx ON identity_claims(instance_id);
            CREATE INDEX IF NOT EXISTS context_solver_idx
                ON evidence_context(solver_version_id);
            CREATE INDEX IF NOT EXISTS context_protocol_idx
                ON evidence_context(search_protocol_id);
            """
        )
        evidence_columns = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(evidence)")
        }
        if "cyclic_band_generators" not in evidence_columns:
            self.connection.execute(
                "ALTER TABLE evidence ADD COLUMN cyclic_band_generators "
                "INTEGER NOT NULL DEFAULT 0"
            )
            self.connection.execute(
                """UPDATE evidence SET cyclic_band_generators=(
                       SELECT r.cyclic_band_generators FROM representations r
                       WHERE r.instance_id=evidence.instance_id
                   )"""
            )
        self.connection.execute(
            "INSERT OR REPLACE INTO catalogue_meta(key, value) VALUES('schema', ?)",
            (SCHEMA,),
        )
        self.connection.commit()

    def _register_collection(
        self,
        collection_id: str,
        *,
        name: str,
        collection_kind: str,
        description: str,
        policy: dict[str, Any],
    ) -> str:
        self.connection.execute(
            """INSERT OR IGNORE INTO science_collections
               (collection_id, name, collection_kind, description, policy_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                collection_id,
                name,
                collection_kind,
                description,
                _canonical_json(policy),
                _utc_now(),
            ),
        )
        return collection_id

    def _dynamic_collection(
        self, kind: str, label: str, description: str, policy: dict[str, Any]
    ) -> str:
        collection_id = f"{kind}:{_digest(label)[:16]}"
        return self._register_collection(
            collection_id,
            name=label,
            collection_kind=kind,
            description=description,
            policy=policy,
        )

    def _record_context(
        self,
        *,
        evidence_id: str,
        solver: str,
        experiment: str,
        objective_ratio: str | None,
        source_family: str,
        solver_metadata: dict[str, Any],
        search_parameters: dict[str, Any],
    ) -> None:
        solver_payload = {"solver_name": solver, **solver_metadata}
        solver_version_id = _digest(solver_payload)
        checkpoint = solver_payload.get("checkpoint")
        checkpoint_path = checkpoint.get("path") if isinstance(checkpoint, dict) else checkpoint
        checkpoint_sha = (
            checkpoint.get("sha256") if isinstance(checkpoint, dict)
            else solver_payload.get("checkpoint_sha256")
        )
        self.connection.execute(
            """INSERT OR IGNORE INTO solver_versions
               (solver_version_id, solver_name, checkpoint_path, checkpoint_sha256,
                metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                solver_version_id,
                solver,
                checkpoint_path,
                checkpoint_sha,
                _canonical_json(solver_payload),
                _utc_now(),
            ),
        )
        protocol_payload = {
            "objective_ratio": objective_ratio,
            **search_parameters,
        }
        search_protocol_id = _digest(protocol_payload)
        self.connection.execute(
            """INSERT OR IGNORE INTO search_protocols
               (search_protocol_id, simulations, action_horizon, objective_ratio,
                protocol_sha256, parameters_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                search_protocol_id,
                protocol_payload.get("simulations"),
                protocol_payload.get("action_horizon"),
                objective_ratio,
                protocol_payload.get("protocol_sha256"),
                _canonical_json(protocol_payload),
                _utc_now(),
            ),
        )
        self.connection.execute(
            """INSERT OR IGNORE INTO evidence_context
               (evidence_id, solver_version_id, search_protocol_id, source_family, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                evidence_id,
                solver_version_id,
                search_protocol_id,
                source_family,
                _utc_now(),
            ),
        )
        memberships = [
            (
                self._register_collection(
                    "verified-all",
                    name="All replay-verified solutions",
                    collection_kind="verification-tier",
                    description="Every native witness that replays to the unknot",
                    policy={"replay_verified": True},
                ),
                "replay-verified",
            ),
            (
                self._dynamic_collection(
                    "source-family",
                    source_family,
                    f"Replay-verified evidence from {source_family}",
                    {"source_family": source_family},
                ),
                source_family,
            ),
            (
                self._dynamic_collection(
                    "experiment",
                    experiment,
                    f"Replay-verified evidence from experiment {experiment}",
                    {"experiment": experiment},
                ),
                experiment,
            ),
            (
                self._dynamic_collection(
                    "solver-version",
                    solver_version_id,
                    f"Replay-verified evidence from solver version {solver_version_id}",
                    {"solver_version_id": solver_version_id},
                ),
                solver_version_id,
            ),
        ]
        for collection_id, reason in memberships:
            self.connection.execute(
                """INSERT OR IGNORE INTO science_collection_evidence
                   (collection_id, evidence_id, reason) VALUES (?, ?, ?)""",
                (collection_id, evidence_id, reason),
            )

    def _scan_seen(self, path: Path, scan_digest: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM source_scans WHERE source_path=? AND scan_digest=?",
            (str(path), scan_digest),
        ).fetchone()
        return row is not None

    def _record_scan(
        self, path: Path, scan_digest: str, source_kind: str, admitted: int, quarantined: int
    ) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO source_scans
               (source_path, scan_digest, source_kind, scanned_at, admitted, quarantined)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(path), scan_digest, source_kind, _utc_now(), admitted, quarantined),
        )
        self.connection.commit()

    def _record_representation(self, witness: UnknotWitness) -> str:
        instance_id = witness.instance_id
        word_json = _canonical_json(list(witness.start.word))
        existing = self.connection.execute(
            "SELECT word_json, strands, cyclic_band_generators FROM representations "
            "WHERE instance_id=?",
            (instance_id,),
        ).fetchone()
        representation = (word_json, witness.start.strands)
        if existing is not None and tuple(existing)[:2] != representation:
            raise ValueError(f"instance id collision for {instance_id}")
        row = (*representation, int(witness.start.cyclic_band_generators))
        self.connection.execute(
            """INSERT OR IGNORE INTO representations
               (instance_id, word_json, strands, cyclic_band_generators, first_seen_at)
               VALUES (?, ?, ?, ?, ?)""",
            (instance_id, *row, _utc_now()),
        )
        if witness.start.cyclic_band_generators:
            self.connection.execute(
                "UPDATE representations SET cyclic_band_generators=1 WHERE instance_id=?",
                (instance_id,),
            )
        return instance_id

    def _record_claim(
        self,
        *,
        instance_id: str,
        declared_name: str | None,
        dataset_origin: str | None,
        source_kind: str,
        source_path: Path,
        source_sha256: str,
    ) -> None:
        canonical = canonical_name(declared_name) if declared_name else None
        body = {
            "instance_id": instance_id,
            "declared_name": declared_name,
            "canonical_name": canonical,
            "dataset_origin": dataset_origin,
            "source_kind": source_kind,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
        }
        self.connection.execute(
            """INSERT OR IGNORE INTO identity_claims
               (claim_id, instance_id, declared_name, canonical_name, dataset_origin,
                source_kind, source_path, source_sha256, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _digest(body),
                instance_id,
                declared_name,
                canonical,
                dataset_origin,
                source_kind,
                str(source_path),
                source_sha256,
                _utc_now(),
            ),
        )

    def _record_evidence(
        self,
        *,
        witness: UnknotWitness,
        solver: str,
        experiment: str,
        objective_ratio: str | None,
        provenance: dict[str, Any],
        source_path: Path,
        source_sha256: str,
        source_channel: str,
        declared_name: str | None,
        dataset_origin: str | None,
        claim_source_path: Path,
        claim_source_sha256: str,
        claim_source_kind: str,
        source_family: str,
        solver_metadata: dict[str, Any],
        search_parameters: dict[str, Any],
    ) -> bool:
        witness.verify()
        instance_id = self._record_representation(witness)
        self._record_claim(
            instance_id=instance_id,
            declared_name=declared_name,
            dataset_origin=dataset_origin,
            source_kind=claim_source_kind,
            source_path=claim_source_path,
            source_sha256=claim_source_sha256,
        )
        witness_dict = witness.to_dict()
        identity = {
            "witness": witness_dict,
            "solver": solver,
            "experiment": experiment,
            "objective_ratio": objective_ratio,
        }
        evidence_id = _digest(identity)
        inserted = self.connection.execute(
            """INSERT OR IGNORE INTO evidence
               (evidence_id, instance_id, solver, experiment, objective_ratio,
                crossing_changes, moves, l10, l1000, witness_json, provenance_json,
                cyclic_band_generators, replay_verified, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
            (
                evidence_id,
                instance_id,
                solver,
                experiment,
                objective_ratio,
                witness.crossing_changes,
                witness.moves,
                10 * witness.crossing_changes + witness.moves,
                1000 * witness.crossing_changes + witness.moves,
                _canonical_json(witness_dict),
                _canonical_json(provenance),
                int(witness.start.cyclic_band_generators),
                _utc_now(),
            ),
        ).rowcount
        self.connection.execute(
            """INSERT OR IGNORE INTO evidence_sources
               (evidence_id, source_path, source_sha256, source_channel)
               VALUES (?, ?, ?, ?)""",
            (evidence_id, str(source_path), source_sha256, source_channel),
        )
        self._record_context(
            evidence_id=evidence_id,
            solver=solver,
            experiment=experiment,
            objective_ratio=objective_ratio,
            source_family=source_family,
            solver_metadata=solver_metadata,
            search_parameters=search_parameters,
        )
        return bool(inserted)

    def quarantine(
        self, path: Path, reason: str, context: dict[str, Any], source_sha256: str | None = None
    ) -> None:
        body = {
            "source_path": str(path),
            "source_sha256": source_sha256,
            "reason": reason,
            "context": context,
        }
        self.connection.execute(
            """INSERT OR IGNORE INTO quarantine
               (item_id, source_path, source_sha256, reason, context_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                _digest(body),
                str(path),
                source_sha256,
                reason,
                _canonical_json(context),
                _utc_now(),
            ),
        )

    def scan_experiment(self, root: Path) -> dict[str, int]:
        bank_path = root / "bank.json"
        manifest_path = root / "manifest.json"
        if not bank_path.is_file():
            return {"sources": 0, "admitted": 0, "quarantined": 0}
        bank_sha = _sha256(bank_path)
        manifest_sha = _sha256(manifest_path) if manifest_path.is_file() else None
        bank = json.loads(bank_path.read_text())
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        protocol_keys = (
            "simulations",
            "qualification_simulations",
            "evaluation_attempts_per_objective",
            "evaluation_root_noise",
            "action_horizon",
            "F_native",
            "selfplay_games_per_iteration",
            "optimizer_steps_per_iteration",
            "batch_size",
            "ratios",
            "seed",
            "protocol_sha256",
        )
        rows = {
            str(row.get("id", row.get("name"))): row
            for row in bank.get("rows", [])
            if row.get("id", row.get("name")) is not None
        }
        totals = {"sources": 0, "admitted": 0, "quarantined": 0}
        event_paths = sorted(
            {
                *(root / "native-events").glob("*.json"),
                *(root / "events").glob("*.json"),
            }
        )
        for event_path in event_paths:
            event_sha = _sha256(event_path)
            scan_digest = _digest(
                {"event": event_sha, "bank": bank_sha, "manifest": manifest_sha}
            )
            if self._scan_seen(event_path, scan_digest):
                continue
            totals["sources"] += 1
            admitted = quarantined = 0
            try:
                event = json.loads(event_path.read_text())
                selected = str(event["selected"])
                bank_row = rows[selected]
                scientists = event["scientists"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self.quarantine(event_path, f"invalid event join: {error}", {}, event_sha)
                self.connection.commit()
                self._record_scan(event_path, scan_digest, "experiment-event", 0, 1)
                totals["quarantined"] += 1
                continue
            for scientist, scientist_row in sorted(scientists.items()):
                channels: list[tuple[str, str, dict[str, Any]]] = []
                for ratio, evaluation in sorted((scientist_row.get("evaluation") or {}).items()):
                    witness_row = evaluation.get("best_witness")
                    if witness_row and witness_row.get("semantic_actions"):
                        channels.append(("evaluation-best", str(ratio), witness_row))
                for ratio, witness_row in sorted((scientist_row.get("native_best") or {}).items()):
                    if witness_row and witness_row.get("semantic_actions"):
                        channels.append(("native-best", str(ratio), witness_row))
                for channel, ratio, witness_row in channels:
                    try:
                        witness, spec = _replay_flat_witness(
                            bank_row,
                            witness_row,
                            cyclic_band_generators=bool(
                                manifest.get("cyclic_band_generators", False)
                            ),
                        )
                        provenance = {
                            "schema": "q-r-skm-evidence-provenance-v1",
                            "source_family": "q-or-r-run",
                            "run_root": str(root),
                            "event": str(event_path),
                            "event_sha256": event_sha,
                            "event_schema": event.get("schema"),
                            "bank": str(bank_path),
                            "bank_sha256": bank_sha,
                            "bank_schema": bank.get("schema"),
                            "manifest": str(manifest_path) if manifest_path.is_file() else None,
                            "manifest_sha256": manifest_sha,
                            "protocol_sha256": manifest.get("protocol_sha256"),
                            "selected": selected,
                            "source_channel": channel,
                            "action_spec_inferred": dataclasses.asdict(spec),
                        }
                        if self._record_evidence(
                            witness=witness,
                            solver=str(scientist),
                            experiment=str(manifest.get("name", root.name)),
                            objective_ratio=ratio,
                            provenance=provenance,
                            source_path=event_path,
                            source_sha256=event_sha,
                            source_channel=channel,
                            declared_name=bank_row.get("name") or selected,
                            dataset_origin=bank_row.get("dataset_origin"),
                            claim_source_path=bank_path,
                            claim_source_sha256=bank_sha,
                            claim_source_kind="experiment-bank",
                            source_family="q-or-r-run",
                            solver_metadata={
                                "architecture": str(scientist),
                                "checkpoint": (manifest.get("checkpoints") or {}).get(
                                    str(scientist)
                                ),
                                "manifest": str(manifest_path)
                                if manifest_path.is_file()
                                else None,
                                "manifest_sha256": manifest_sha,
                                "protocol_sha256": manifest.get("protocol_sha256"),
                            },
                            search_parameters={
                                key: manifest.get(key)
                                for key in protocol_keys
                                if key in manifest
                            }
                            | {
                                "source_channel": channel,
                                "action_spec": dataclasses.asdict(spec),
                            },
                        ):
                            admitted += 1
                    except (KeyError, TypeError, ValueError) as error:
                        quarantined += 1
                        self.quarantine(
                            event_path,
                            f"witness replay failed: {error}",
                            {"scientist": scientist, "ratio": ratio, "channel": channel},
                            event_sha,
                        )
            self.connection.commit()
            self._record_scan(
                event_path, scan_digest, "experiment-event", admitted, quarantined
            )
            totals["admitted"] += admitted
            totals["quarantined"] += quarantined
        return totals

    def scan_mastery_inventory(self, inventory: Path) -> dict[str, int]:
        witness_root = inventory / "witnesses" if (inventory / "witnesses").is_dir() else inventory
        program_root = witness_root.parent.parent
        mastery_runs: dict[str, dict[str, Any]] = {}
        for manifest_path in sorted((program_root / "scientists").glob("*/run-manifest.json")):
            try:
                run_manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            live_path = manifest_path.with_name("live-protocol.json")
            live_protocol = (
                json.loads(live_path.read_text()) if live_path.is_file() else {}
            )
            scientist = str(run_manifest.get("scientist", ""))
            if scientist:
                mastery_runs[scientist] = {
                    "manifest_path": manifest_path,
                    "manifest_sha256": _sha256(manifest_path),
                    "manifest": run_manifest,
                    "live_path": live_path if live_path.is_file() else None,
                    "live_sha256": _sha256(live_path) if live_path.is_file() else None,
                    "live": live_protocol,
                }
        totals = {"sources": 0, "admitted": 0, "quarantined": 0}
        for path in sorted(witness_root.glob("*.json")):
            source_sha = _sha256(path)
            if self._scan_seen(path, source_sha):
                continue
            totals["sources"] += 1
            try:
                row = json.loads(path.read_text())
                witness = UnknotWitness.from_dict(row["witness"])
                witness.verify()
                run = mastery_runs.get(str(row["scientist"]), {})
                run_manifest = run.get("manifest") or {}
                live_protocol = run.get("live") or {}
                provenance = {
                    "schema": "q-r-skm-evidence-provenance-v1",
                    "source_family": "single-knot-mastery",
                    "inventory_row": str(path),
                    "inventory_row_sha256": source_sha,
                    "inventory_schema": row.get("schema"),
                    "challenge_id": row.get("challenge_id"),
                    "representation_id": row.get("representation_id"),
                    "sequence_name": row.get("sequence_name"),
                    "previous_upper_bound": row.get("previous_upper_bound"),
                    "run_manifest": (
                        str(run["manifest_path"]) if run.get("manifest_path") else None
                    ),
                    "run_manifest_sha256": run.get("manifest_sha256"),
                    "live_protocol": (
                        str(run["live_path"]) if run.get("live_path") else None
                    ),
                    "live_protocol_sha256": run.get("live_sha256"),
                }
                inserted = self._record_evidence(
                    witness=witness,
                    solver=str(row["scientist"]),
                    experiment=str(row.get("sequence_name", "single-knot-mastery")),
                    objective_ratio="1000",
                    provenance=provenance,
                    source_path=path,
                    source_sha256=source_sha,
                    source_channel="mastery-verified-solution",
                    declared_name=row.get("knot_name"),
                    dataset_origin="mastery-sequence",
                    claim_source_path=path,
                    claim_source_sha256=source_sha,
                    claim_source_kind="mastery-inventory",
                    source_family="single-knot-mastery",
                    solver_metadata={
                        "architecture": str(row["scientist"]),
                        "checkpoint": {
                            "path": run_manifest.get("checkpoint"),
                            "sha256": run_manifest.get("checkpoint_sha256"),
                        },
                        "run_manifest": (
                            str(run["manifest_path"]) if run.get("manifest_path") else None
                        ),
                        "run_manifest_sha256": run.get("manifest_sha256"),
                        "live_protocol_sha256": run.get("live_sha256"),
                        "source_hash_manifest_sha256": live_protocol.get(
                            "source_hash_manifest_sha256"
                        ),
                    },
                    search_parameters={
                        key: live_protocol.get(key, run_manifest.get(key))
                        for key in (
                            "simulations",
                            "action_horizon",
                            "parallel_searches",
                            "challenge_attempt_limit",
                            "challenge_seconds_limit",
                            "negative_confirmations",
                            "heap_capacity",
                            "seed",
                            "rehearsal",
                        )
                        if key in live_protocol or key in run_manifest
                    }
                    | {
                        "objective_budget": run_manifest.get("objective_move_budget"),
                        "live_protocol_sha256": run.get("live_sha256"),
                    },
                )
                self.connection.commit()
                self._record_scan(path, source_sha, "mastery-witness", int(inserted), 0)
                totals["admitted"] += int(inserted)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self.quarantine(path, f"invalid mastery witness: {error}", {}, source_sha)
                self.connection.commit()
                self._record_scan(path, source_sha, "mastery-witness", 0, 1)
                totals["quarantined"] += 1
        return totals

    def scan_external_evidence(self, path: Path) -> dict[str, int]:
        """Import external records without promoting claim-only rows to witnesses."""
        source_sha = _sha256(path)
        if self._scan_seen(path, source_sha):
            return {"sources": 0, "admitted": 0, "replayable": 0, "quarantined": 0}
        totals = {"sources": 1, "admitted": 0, "replayable": 0, "quarantined": 0}
        try:
            payload = json.loads(path.read_text())
            records = payload["records"]
            if not isinstance(records, list):
                raise ValueError("external records must be a list")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.quarantine(path, f"invalid external evidence package: {error}", {}, source_sha)
            self.connection.commit()
            self._record_scan(path, source_sha, "external-evidence", 0, 1)
            totals["quarantined"] = 1
            return totals

        external_collection = self._register_collection(
            "external-all",
            name="All external evidence records",
            collection_kind="external",
            description="External claims, references, attempts, and replayable paths",
            policy={
                "claim_only_is_not_a_native_witness": True,
                "distillation_requires_replay": True,
            },
        )
        for index, record in enumerate(records):
            try:
                if not isinstance(record, dict):
                    raise ValueError("record is not an object")
                external_id = str(record.get("evidence_id") or _digest(
                    {"source_sha256": source_sha, "index": index, "record": record}
                ))
                raw_name = record.get("canonical_name") or record.get("knot_name")
                mapped_name = canonical_name(str(raw_name)) if raw_name else None
                claimed_upper = record.get("claimed_new_upper")
                interval = (
                    record.get("bound_interval")
                    or record.get("catalogue_bound_interval")
                    or record.get("old_bound")
                    or record.get("old_bound_interval")
                )
                replay_verified = bool(record.get("replay_verified"))
                witness_row = record.get("witness")
                if replay_verified and witness_row:
                    verification_tier = "replay-verified"
                elif claimed_upper is not None:
                    verification_tier = "claim-only"
                elif record.get("record_type") == "external_upper_bound_reference":
                    verification_tier = "reference-only"
                elif record.get("record_type") == "external_search_attempt":
                    verification_tier = "search-attempt"
                else:
                    verification_tier = "representation-or-metadata-only"
                inserted = self.connection.execute(
                    """INSERT OR IGNORE INTO external_records
                       (external_record_id, source_path, source_sha256, record_type,
                        canonical_name, claimed_upper, bound_interval_json,
                        verification_tier, replay_verified, distillation_eligible,
                        l10, l1000, source_json, payload_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        external_id,
                        str(path),
                        source_sha,
                        str(record.get("record_type") or record.get("classification") or "unknown"),
                        mapped_name,
                        int(claimed_upper) if claimed_upper is not None else None,
                        _canonical_json(interval) if interval is not None else None,
                        verification_tier,
                        int(replay_verified),
                        int(bool(record.get("distillation_eligible")) and replay_verified),
                        record.get("l10"),
                        record.get("l1000"),
                        _canonical_json(record.get("source") or {}),
                        _canonical_json(record),
                        _utc_now(),
                    ),
                ).rowcount
                self.connection.execute(
                    """INSERT OR IGNORE INTO science_collection_external_records
                       (collection_id, external_record_id, reason) VALUES (?, ?, ?)""",
                    (external_collection, external_id, verification_tier),
                )
                totals["admitted"] += int(inserted)
                if replay_verified and witness_row:
                    witness = UnknotWitness.from_dict(witness_row)
                    witness.verify()
                    source = record.get("source") or {}
                    solver = str(
                        record.get("solver")
                        or source.get("repository")
                        or source.get("repository_url")
                        or "external"
                    )
                    if self._record_evidence(
                        witness=witness,
                        solver=solver,
                        experiment=f"external:{payload.get('schema', path.stem)}",
                        objective_ratio=None,
                        provenance={
                            "schema": "q-r-skm-evidence-provenance-v1",
                            "source_family": "external-replayable",
                            "package": str(path),
                            "package_sha256": source_sha,
                            "external_record_id": external_id,
                        },
                        source_path=path,
                        source_sha256=source_sha,
                        source_channel="external-replayable",
                        declared_name=mapped_name,
                        dataset_origin="external",
                        claim_source_path=path,
                        claim_source_sha256=source_sha,
                        claim_source_kind="external-package",
                        source_family="external-replayable",
                        solver_metadata={
                            "source": source,
                            "external_record_id": external_id,
                        },
                        search_parameters=dict(record.get("search_parameters") or {}),
                    ):
                        totals["replayable"] += 1
            except (KeyError, TypeError, ValueError) as error:
                totals["quarantined"] += 1
                self.quarantine(
                    path,
                    f"invalid external record: {error}",
                    {"index": index},
                    source_sha,
                )
        self.connection.commit()
        self._record_scan(
            path,
            source_sha,
            "external-evidence",
            totals["admitted"],
            totals["quarantined"],
        )
        return totals

    def backfill_context(self) -> int:
        """Normalize solver/protocol metadata for rows admitted by catalogue v1."""
        rows = self.connection.execute(
            """SELECT e.* FROM evidence e LEFT JOIN evidence_context c
               ON c.evidence_id=e.evidence_id WHERE c.evidence_id IS NULL
               ORDER BY e.created_at, e.evidence_id"""
        ).fetchall()
        protocol_keys = (
            "simulations",
            "qualification_simulations",
            "evaluation_attempts_per_objective",
            "evaluation_root_noise",
            "action_horizon",
            "F_native",
            "selfplay_games_per_iteration",
            "optimizer_steps_per_iteration",
            "batch_size",
            "ratios",
            "seed",
            "protocol_sha256",
        )
        for row in rows:
            provenance = json.loads(row["provenance_json"])
            family = str(provenance.get("source_family") or "unknown")
            solver_metadata: dict[str, Any] = {"architecture": row["solver"]}
            search_parameters: dict[str, Any] = {}
            if family == "q-or-r-run":
                manifest_path = Path(str(provenance.get("manifest") or ""))
                manifest = (
                    json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
                )
                solver_metadata.update(
                    {
                        "checkpoint": (manifest.get("checkpoints") or {}).get(row["solver"]),
                        "manifest": str(manifest_path) if manifest_path.is_file() else None,
                        "manifest_sha256": provenance.get("manifest_sha256"),
                        "protocol_sha256": provenance.get("protocol_sha256"),
                    }
                )
                search_parameters = {
                    key: manifest.get(key) for key in protocol_keys if key in manifest
                }
                search_parameters.update(
                    {
                        "source_channel": provenance.get("source_channel"),
                        "action_spec": provenance.get("action_spec_inferred"),
                    }
                )
            elif family == "single-knot-mastery":
                inventory_row = Path(str(provenance.get("inventory_row") or ""))
                program_root = inventory_row.parent.parent.parent
                for manifest_path in sorted(
                    (program_root / "scientists").glob("*/run-manifest.json")
                ):
                    manifest = json.loads(manifest_path.read_text())
                    if manifest.get("scientist") != row["solver"]:
                        continue
                    live_path = manifest_path.with_name("live-protocol.json")
                    live = json.loads(live_path.read_text()) if live_path.is_file() else {}
                    solver_metadata.update(
                        {
                            "checkpoint": {
                                "path": manifest.get("checkpoint"),
                                "sha256": manifest.get("checkpoint_sha256"),
                            },
                            "run_manifest": str(manifest_path),
                            "run_manifest_sha256": _sha256(manifest_path),
                            "live_protocol_sha256": (
                                _sha256(live_path) if live_path.is_file() else None
                            ),
                            "source_hash_manifest_sha256": live.get(
                                "source_hash_manifest_sha256"
                            ),
                        }
                    )
                    search_parameters = {
                        key: live.get(key, manifest.get(key))
                        for key in (
                            "simulations",
                            "action_horizon",
                            "parallel_searches",
                            "challenge_attempt_limit",
                            "challenge_seconds_limit",
                            "negative_confirmations",
                            "heap_capacity",
                            "seed",
                            "rehearsal",
                        )
                        if key in live or key in manifest
                    }
                    search_parameters["objective_budget"] = manifest.get(
                        "objective_move_budget"
                    )
                    break
            self._record_context(
                evidence_id=row["evidence_id"],
                solver=row["solver"],
                experiment=row["experiment"],
                objective_ratio=row["objective_ratio"],
                source_family=family,
                solver_metadata=solver_metadata,
                search_parameters=search_parameters,
            )
        self.connection.commit()
        return len(rows)

    def compute_metadata(self, *, limit: int, max_full_invariant_strands: int) -> int:
        pending = self.connection.execute(
            """SELECT r.* FROM representations r
               LEFT JOIN representation_metadata m ON m.instance_id=r.instance_id
               WHERE m.instance_id IS NULL OR m.algorithm != ? OR m.status='error'
               ORDER BY CASE WHEN m.status='error' THEN 1 ELSE 0 END,
                        r.first_seen_at, r.instance_id LIMIT ?""",
            (METADATA_ALGORITHM, limit),
        ).fetchall()
        completed = 0
        for row in pending:
            instance_id = str(row["instance_id"])
            word = tuple(int(value) for value in json.loads(row["word_json"]))
            strands = int(row["strands"])
            try:
                alexander = alexander_polynomial(word, strands)
                sigma = signature(word, strands)
                common = {
                    "word_length": len(word),
                    "strands": strands,
                    "writhe": sum(1 if value > 0 else -1 for value in word),
                    "alexander": to_pairs(alexander),
                    "determinant": abs(
                        sum(coefficient * (-1) ** (exponent % 2)
                            for exponent, coefficient in alexander.items())
                    ),
                    "signature": sigma,
                    "genus_lower": max(alexander) // 2 if alexander else 0,
                    "genus_upper": max(0, len(word) - strands + 1) // 2,
                    "unknotting_lower": None if sigma is None else abs(sigma) // 2,
                }
                if strands <= max_full_invariant_strands:
                    computed = invariants(word, strands, decompose=False)
                    payload = dataclasses.asdict(computed)
                    status = "complete"
                    candidates = _candidate_names(computed.name)
                else:
                    payload = {
                        **common,
                        "jones": None,
                        "name": None,
                        "notes": [
                            "Jones and table identification deferred: strand capacity exceeds "
                            f"the configured full-invariant limit {max_full_invariant_strands}"
                        ],
                    }
                    status = "partial-capacity"
                    candidates = ()
                self.connection.execute(
                    """INSERT OR REPLACE INTO representation_metadata
                       (instance_id, algorithm, status, invariants_json,
                        identified_candidates_json, computed_at, error)
                       VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                    (
                        instance_id,
                        METADATA_ALGORITHM,
                        status,
                        _canonical_json(payload),
                        _canonical_json(candidates),
                        _utc_now(),
                    ),
                )
                self._refresh_mapping(instance_id, candidates)
                self.connection.commit()
                completed += 1
            except Exception as error:  # metadata failure must not stop ingestion
                self.connection.execute(
                    """INSERT OR REPLACE INTO representation_metadata
                       (instance_id, algorithm, status, invariants_json,
                        identified_candidates_json, computed_at, error)
                       VALUES (?, ?, 'error', NULL, '[]', ?, ?)""",
                    (instance_id, METADATA_ALGORITHM, _utc_now(), repr(error)),
                )
                self.connection.commit()
        return completed

    def _refresh_mapping(self, instance_id: str, candidates: tuple[str, ...]) -> None:
        claims = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT DISTINCT canonical_name FROM identity_claims "
                "WHERE instance_id=? AND canonical_name IS NOT NULL",
                (instance_id,),
            )
        }
        mapped: str | None = None
        eligible = False
        if len(claims) > 1:
            status = "conflicting-source-claims"
        elif len(candidates) == 1:
            identified = candidates[0]
            if not claims:
                status, mapped, eligible = "computed-unique", identified, True
            elif identified in claims:
                status, mapped, eligible = "verified-unique", identified, True
            else:
                status = "invariant-source-conflict"
        elif len(candidates) > 1:
            if len(claims) == 1 and next(iter(claims)) in candidates:
                status = "source-disambiguated-compatible"
                mapped, eligible = next(iter(claims)), True
            elif claims:
                status = "invariant-source-conflict"
            else:
                status = "computed-ambiguous"
        elif len(claims) == 1:
            status, mapped = "declared-unverified", next(iter(claims))
        else:
            status = "unidentified"
        self.connection.execute(
            """UPDATE evidence SET mapping_status=?, mapped_knot=?,
               eligible_knot_upper_bound=? WHERE instance_id=?""",
            (status, mapped, int(eligible), instance_id),
        )

    def export(self, output: Path) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        counts = {
            "representations": self.connection.execute(
                "SELECT COUNT(*) FROM representations"
            ).fetchone()[0],
            "evidence": self.connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0],
            "eligible_knot_evidence": self.connection.execute(
                "SELECT COUNT(*) FROM evidence WHERE eligible_knot_upper_bound=1"
            ).fetchone()[0],
            "metadata_complete": self.connection.execute(
                "SELECT COUNT(*) FROM representation_metadata WHERE status='complete'"
            ).fetchone()[0],
            "metadata_partial": self.connection.execute(
                "SELECT COUNT(*) FROM representation_metadata WHERE status='partial-capacity'"
            ).fetchone()[0],
            "metadata_errors": self.connection.execute(
                "SELECT COUNT(*) FROM representation_metadata WHERE status='error'"
            ).fetchone()[0],
            "quarantine": self.connection.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0],
            "solver_versions": self.connection.execute(
                "SELECT COUNT(*) FROM solver_versions"
            ).fetchone()[0],
            "search_protocols": self.connection.execute(
                "SELECT COUNT(*) FROM search_protocols"
            ).fetchone()[0],
            "science_collections": self.connection.execute(
                "SELECT COUNT(*) FROM science_collections"
            ).fetchone()[0],
            "external_records": self.connection.execute(
                "SELECT COUNT(*) FROM external_records"
            ).fetchone()[0],
            "external_claim_only": self.connection.execute(
                "SELECT COUNT(*) FROM external_records WHERE verification_tier='claim-only'"
            ).fetchone()[0],
            "external_replay_verified": self.connection.execute(
                "SELECT COUNT(*) FROM external_records WHERE replay_verified=1"
            ).fetchone()[0],
            "external_distillation_eligible": self.connection.execute(
                "SELECT COUNT(*) FROM external_records WHERE distillation_eligible=1"
            ).fetchone()[0],
            "ordinary_action_evidence": self.connection.execute(
                "SELECT COUNT(*) FROM evidence WHERE cyclic_band_generators=0"
            ).fetchone()[0],
            "cyclic_band_evidence": self.connection.execute(
                "SELECT COUNT(*) FROM evidence WHERE cyclic_band_generators=1"
            ).fetchone()[0],
            "representations_with_cyclic_band_evidence": self.connection.execute(
                "SELECT COUNT(DISTINCT instance_id) FROM evidence "
                "WHERE cyclic_band_generators=1"
            ).fetchone()[0],
        }
        best_rows = self.connection.execute(
            """SELECT mapped_knot, evidence_id, instance_id, solver, experiment,
                      crossing_changes, moves, l10, l1000, mapping_status
               FROM evidence WHERE eligible_knot_upper_bound=1
               ORDER BY mapped_knot, crossing_changes, moves, evidence_id"""
        ).fetchall()
        best: dict[str, dict[str, Any]] = {}
        for row in best_rows:
            knot = str(row["mapped_knot"])
            best.setdefault(knot, dict(row))

        solution_rows = self.connection.execute(
            """SELECT e.evidence_id, e.instance_id, e.solver, e.experiment,
                      e.objective_ratio, e.crossing_changes, e.moves, e.l10, e.l1000,
                      e.mapping_status, e.mapped_knot, e.eligible_knot_upper_bound,
                      e.witness_json, e.provenance_json, e.cyclic_band_generators,
                      c.source_family, c.solver_version_id, c.search_protocol_id,
                      s.checkpoint_path, s.checkpoint_sha256,
                      p.simulations, p.action_horizon, p.protocol_sha256,
                      p.parameters_json
               FROM evidence e
               JOIN evidence_context c ON c.evidence_id=e.evidence_id
               JOIN solver_versions s ON s.solver_version_id=c.solver_version_id
               JOIN search_protocols p ON p.search_protocol_id=c.search_protocol_id
               ORDER BY e.instance_id, e.l1000, e.l10, e.evidence_id"""
        ).fetchall()

        def solution_item(row: sqlite3.Row) -> dict[str, Any]:
            return {
                "evidence_id": row["evidence_id"],
                "instance_id": row["instance_id"],
                "solver": row["solver"],
                "experiment": row["experiment"],
                "source_family": row["source_family"],
                "solver_version_id": row["solver_version_id"],
                "checkpoint": {
                    "path": row["checkpoint_path"],
                    "sha256": row["checkpoint_sha256"],
                },
                "search_protocol_id": row["search_protocol_id"],
                "search_parameters": json.loads(row["parameters_json"]),
                "protocol_sha256": row["protocol_sha256"],
                "simulations": row["simulations"],
                "action_horizon": row["action_horizon"],
                "objective_ratio": row["objective_ratio"],
                "crossing_changes": row["crossing_changes"],
                "moves": row["moves"],
                "l10": row["l10"],
                "l1000": row["l1000"],
                "mapping_status": row["mapping_status"],
                "mapped_knot": row["mapped_knot"],
                "eligible_knot_upper_bound": bool(row["eligible_knot_upper_bound"]),
                "cyclic_band_generators": bool(row["cyclic_band_generators"]),
                "witness": json.loads(row["witness_json"]),
                "provenance": json.loads(row["provenance_json"]),
            }

        global_by_representation: dict[str, dict[str, Any]] = {}
        global_by_knot: dict[str, dict[str, Any]] = {}
        by_solver_version: dict[tuple[str, str], dict[str, Any]] = {}
        for row in solution_rows:
            item = solution_item(row)
            global_by_representation.setdefault(str(row["instance_id"]), item)
            if bool(row["eligible_knot_upper_bound"]) and row["mapped_knot"]:
                global_by_knot.setdefault(str(row["mapped_knot"]), item)
            by_solver_version.setdefault(
                (str(row["instance_id"]), str(row["solver_version_id"])), item
            )

        external_pool = []
        for row in self.connection.execute(
            """SELECT external_record_id, source_path, source_sha256, record_type,
                      canonical_name, claimed_upper, bound_interval_json,
                      verification_tier, replay_verified, distillation_eligible,
                      l10, l1000, source_json
               FROM external_records
               ORDER BY CASE WHEN l10 IS NULL THEN 1 ELSE 0 END, l10,
                        canonical_name, external_record_id"""
        ):
            item = dict(row)
            item["bound_interval"] = (
                None
                if item.pop("bound_interval_json") is None
                else json.loads(row["bound_interval_json"])
            )
            item["source"] = json.loads(item.pop("source_json"))
            item["replay_verified"] = bool(item["replay_verified"])
            item["distillation_eligible"] = bool(item["distillation_eligible"])
            external_pool.append(item)

        solver_versions = []
        for row in self.connection.execute(
            """SELECT s.*, COUNT(c.evidence_id) AS evidence_count
               FROM solver_versions s LEFT JOIN evidence_context c
               ON c.solver_version_id=s.solver_version_id
               GROUP BY s.solver_version_id ORDER BY s.solver_name, s.solver_version_id"""
        ):
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            solver_versions.append(item)

        search_protocols = []
        for row in self.connection.execute(
            """SELECT p.*, COUNT(c.evidence_id) AS evidence_count
               FROM search_protocols p LEFT JOIN evidence_context c
               ON c.search_protocol_id=p.search_protocol_id
               GROUP BY p.search_protocol_id ORDER BY p.search_protocol_id"""
        ):
            item = dict(row)
            item["parameters"] = json.loads(item.pop("parameters_json"))
            search_protocols.append(item)

        collections = []
        for row in self.connection.execute(
            """SELECT c.*,
                      COUNT(DISTINCT ce.evidence_id) AS replay_verified_count,
                      COUNT(DISTINCT cx.external_record_id) AS external_record_count
               FROM science_collections c
               LEFT JOIN science_collection_evidence ce
                 ON ce.collection_id=c.collection_id
               LEFT JOIN science_collection_external_records cx
                 ON cx.collection_id=c.collection_id
               GROUP BY c.collection_id ORDER BY c.collection_kind, c.name"""
        ):
            item = dict(row)
            item["policy"] = json.loads(item.pop("policy_json"))
            collections.append(item)
        metadata_rows = self.connection.execute(
            """SELECT r.instance_id, r.word_json, r.strands, r.cyclic_band_generators,
                      m.status, m.invariants_json, m.identified_candidates_json,
                      m.computed_at, m.error
               FROM representations r LEFT JOIN representation_metadata m
               ON m.instance_id=r.instance_id ORDER BY r.instance_id"""
        ).fetchall()
        metadata_export = []
        for row in metadata_rows:
            item = dict(row)
            item["word"] = json.loads(item.pop("word_json"))
            item["cyclic_band_generators"] = bool(item["cyclic_band_generators"])
            item["invariants"] = (
                None if item.pop("invariants_json") is None
                else json.loads(row["invariants_json"])
            )
            item["identified_candidates"] = (
                [] if item.pop("identified_candidates_json") is None
                else json.loads(row["identified_candidates_json"])
            )
            metadata_export.append(item)
        quarantined = [
            {**dict(row), "context": json.loads(row["context_json"])}
            for row in self.connection.execute(
                "SELECT * FROM quarantine ORDER BY created_at, item_id"
            )
        ]
        for row in quarantined:
            row.pop("context_json", None)
        summary = {
            "schema": SCHEMA,
            "generated_at": _utc_now(),
            "database": str(self.database),
            "counts": counts,
        }
        _atomic_write(output / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _atomic_write(
            output / "best-knot-upper-bounds.json",
            json.dumps({"schema": SCHEMA, "knots": best}, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write(
            output / "best-solutions-pool.json",
            json.dumps(
                {
                    "schema": SCHEMA,
                    "policy": {
                        "verified_pool_requires_full_replay": True,
                        "external_claims_are_separate": True,
                        "claim_only_distillation_eligible": False,
                        "ranking": "minimum L1000, then L10, then evidence_id",
                    },
                    "verified": {
                        "best_by_representation": global_by_representation,
                        "best_by_knot": global_by_knot,
                        "best_by_representation_and_solver_version": list(
                            by_solver_version.values()
                        ),
                    },
                    "external": external_pool,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        _atomic_write(
            output / "solver-versions.json",
            json.dumps(
                {"schema": SCHEMA, "solver_versions": solver_versions},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        _atomic_write(
            output / "search-protocols.json",
            json.dumps(
                {"schema": SCHEMA, "search_protocols": search_protocols},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        _atomic_write(
            output / "science-collections.json",
            json.dumps(
                {"schema": SCHEMA, "collections": collections},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        _atomic_write(
            output / "representation-metadata.jsonl",
            "".join(_canonical_json(row) + "\n" for row in metadata_export),
        )
        _atomic_write(
            output / "quarantine.jsonl",
            "".join(_canonical_json(row) + "\n" for row in quarantined),
        )
        return summary


def _experiment_dirs(paths: Iterable[Path]) -> tuple[Path, ...]:
    found: set[Path] = set()
    for path in paths:
        if (path / "bank.json").is_file():
            found.add(path)
        if path.is_dir():
            for bank in path.rglob("bank.json"):
                root = bank.parent
                if (root / "events").is_dir() or (root / "native-events").is_dir():
                    found.add(root)
    return tuple(sorted(found))


def collect(
    *,
    database: Path,
    output: Path,
    experiment_roots: Iterable[Path],
    mastery_inventories: Iterable[Path],
    metadata_limit: int,
    max_full_invariant_strands: int,
    external_evidence_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    catalog = EvidenceCatalog(database)
    try:
        activity = {"experiment": [], "mastery": [], "external": []}
        for root in _experiment_dirs(experiment_roots):
            activity["experiment"].append({"root": str(root), **catalog.scan_experiment(root)})
        for inventory in mastery_inventories:
            activity["mastery"].append(
                {"inventory": str(inventory), **catalog.scan_mastery_inventory(inventory)}
            )
        for path in external_evidence_paths:
            activity["external"].append(
                {"path": str(path), **catalog.scan_external_evidence(path)}
            )
        context_backfilled = catalog.backfill_context()
        metadata_completed = catalog.compute_metadata(
            limit=metadata_limit,
            max_full_invariant_strands=max_full_invariant_strands,
        )
        summary = catalog.export(output)
        summary["activity"] = activity
        summary["metadata_computed_this_run"] = metadata_completed
        summary["contexts_backfilled_this_run"] = context_backfilled
        _atomic_write(output / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary
    finally:
        catalog.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, action="append", default=[])
    parser.add_argument("--mastery-inventory", type=Path, action="append", default=[])
    parser.add_argument("--external-evidence", type=Path, action="append", default=[])
    parser.add_argument("--metadata-limit", type=int, default=8)
    parser.add_argument(
        "--max-full-invariant-strands",
        type=int,
        default=DEFAULT_MAX_FULL_INVARIANT_STRANDS,
    )
    args = parser.parse_args()
    if args.metadata_limit < 0:
        parser.error("--metadata-limit must be non-negative")
    if args.max_full_invariant_strands < 1:
        parser.error("--max-full-invariant-strands must be positive")
    summary = collect(
        database=args.database,
        output=args.output,
        experiment_roots=args.experiment_root,
        mastery_inventories=args.mastery_inventory,
        metadata_limit=args.metadata_limit,
        max_full_invariant_strands=args.max_full_invariant_strands,
        external_evidence_paths=args.external_evidence,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
