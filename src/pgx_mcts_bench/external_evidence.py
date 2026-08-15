"""Collect provenance-bearing external unknotting upper-bound records.

The collector deliberately separates a bound claim, a search attempt, and a
replayable witness.  In particular, an unsuccessful external search is never a
negative example, and a PD diagram plus crossing indices is not promoted to a
native action trajectory.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

XLSX_NAMESPACE = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
PROCESSING_RE = re.compile(
    r"^\[(?P<ordinal>\d+)/(?:\d+)\] Processing:\s*(?P<knot>\S+)\s+"
    r"current range:\s*\[(?P<lower>\d+)\s*,\s*(?P<upper>\d+)\]$"
)
IMPROVED_RE = re.compile(
    r"Improved via variant (?P<variant>\d+), flip (?P<flip>\d+): "
    r"upper (?P<old_upper>\d+) -> (?P<new_upper>\d+)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_targets(sequence_paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for path in sequence_paths:
        payload = json.loads(path.read_text())
        sequence_name = str(payload.get("name") or path.stem)
        for index, challenge in enumerate(payload["challenges"]):
            name = str(challenge["canonical_name"])
            if name in targets:
                raise ValueError(f"duplicate target knot across curricula: {name}")
            targets[name] = {
                "canonical_name": name,
                "sequence": sequence_name,
                "sequence_path": str(path.resolve()),
                "sequence_sha256": sha256(path),
                "sequence_index": index,
                "challenge_id": str(challenge["challenge_id"]),
                "target_u": int(challenge["target_u"]),
                "bound_interval": [int(value) for value in challenge["bound_interval"]],
            }
    return targets


def git_snapshot(path: Path, *, assessment: str) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    return {
        "path": str(path.resolve()),
        "url": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "assessment": assessment,
    }


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str | None:
    if cell.attrib.get("t") == "inlineStr":
        parts = [node.text or "" for node in cell.iter(f"{XLSX_NAMESPACE}t")]
        return "".join(parts) or None
    value = cell.find(f"{XLSX_NAMESPACE}v")
    if value is None or value.text is None:
        return None
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value.text)]
    return value.text


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    strings = []
    with archive.open("xl/sharedStrings.xml") as source:
        for _event, node in ET.iterparse(source, events=("end",)):
            if node.tag != f"{XLSX_NAMESPACE}si":
                continue
            strings.append("".join(part.text or "" for part in node.iter(f"{XLSX_NAMESPACE}t")))
            node.clear()
    return strings


def iter_xlsx_rows(path: Path) -> Iterator[tuple[int, dict[str, str | None]]]:
    """Stream the first worksheet of the simple upstream workbook.

    The upstream file uses inline strings and numeric cells, so a full
    spreadsheet dependency is unnecessary.  This keeps the evidence collector
    runnable in the benchmark's base environment.
    """

    with zipfile.ZipFile(path) as archive:
        shared_strings = _shared_strings(archive)
        with archive.open("xl/worksheets/sheet1.xml") as source:
            headers: list[str] | None = None
            for _event, row in ET.iterparse(source, events=("end",)):
                if row.tag != f"{XLSX_NAMESPACE}row":
                    continue
                values: dict[int, str | None] = {}
                for cell in row.findall(f"{XLSX_NAMESPACE}c"):
                    reference = cell.attrib["r"]
                    letters = "".join(ch for ch in reference if ch.isalpha())
                    column = 0
                    for letter in letters:
                        column = column * 26 + ord(letter.upper()) - ord("A") + 1
                    values[column - 1] = _cell_value(cell, shared_strings)
                width = max(values, default=-1) + 1
                ordered = [values.get(index) for index in range(width)]
                row_number = int(row.attrib["r"])
                if headers is None:
                    headers = [str(value) for value in ordered]
                else:
                    yield (
                        row_number,
                        {
                            header: ordered[index] if index < len(ordered) else None
                            for index, header in enumerate(headers)
                        },
                    )
                row.clear()


def parse_bound(value: str | None) -> list[int] | None:
    if value is None:
        return None
    numbers = [int(number) for number in re.findall(r"\d+", str(value))]
    if not numbers:
        return None
    return [numbers[0], numbers[-1]]


def collect_catalogue_rows(
    workbook: Path, targets: dict[str, dict[str, Any]], repository: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    for row_number, row in iter_xlsx_rows(workbook):
        name = row.get("knot_id")
        if name not in targets:
            continue
        pd_raw = row.get("pd_presentation")
        pd = json.loads(pd_raw) if pd_raw else None
        records.append(
            {
                "record_type": "external_starting_representation",
                "canonical_name": name,
                "target": targets[name],
                "catalogue_bound_interval": parse_bound(row.get("unknotting_number")),
                "pd_presentation": pd,
                "jones_vector": json.loads(row["jones_vector"])
                if row.get("jones_vector", "").startswith("[")
                else row.get("jones_vector"),
                "source": {
                    "repository_url": repository["url"],
                    "repository_commit": repository["commit"],
                    "file": "data/unknotting.xlsx",
                    "file_sha256": sha256(workbook),
                    "row": row_number,
                },
                "replay_verified": False,
                "distillation_eligible": False,
                "training_negative_eligible": False,
                "l10": None,
                "exclusion_reason": "starting PD and bound only; no complete action path",
            }
        )
    return records


def collect_knotinfo_reference_rows(
    workbook: Path,
    targets: dict[str, dict[str, Any]],
    repository: dict[str, Any],
    source_archive: Path,
) -> list[dict[str, Any]]:
    records = []
    for row_number, row in iter_xlsx_rows(workbook):
        name = row.get("name")
        reference = row.get("unknotting_number_anon")
        if name not in targets or not reference:
            continue
        records.append(
            {
                "record_type": "external_upper_bound_reference",
                "canonical_name": name,
                "target": targets[name],
                "catalogue_bound_interval": parse_bound(row.get("unknotting_number")),
                "reference_url": f"https://knotinfo.org/{reference.lstrip('/')}",
                "source": {
                    "repository_url": repository["url"],
                    "repository_commit": repository["commit"],
                    "file": str(source_archive.relative_to(Path(repository["path"]))),
                    "file_sha256": sha256(source_archive),
                    "archive_member": "knotinfo_data_complete.xls",
                    "derived_xlsx_sha256": sha256(workbook),
                    "derived_row": row_number,
                },
                "replay_verified": False,
                "distillation_eligible": False,
                "training_negative_eligible": False,
                "l10": None,
                "exclusion_reason": "reference pointer and bound only; no complete action path",
            }
        )
    return records


def collect_text_search_attempts(
    log_path: Path, targets: dict[str, dict[str, Any]], repository: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    lines = log_path.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        match = PROCESSING_RE.match(line.strip())
        if match is None or match.group("knot") not in targets:
            continue
        end = next(
            (
                cursor
                for cursor in range(index + 1, len(lines))
                if PROCESSING_RE.match(lines[cursor].strip())
            ),
            len(lines),
        )
        block = "\n".join(lines[index:end])
        improvement = IMPROVED_RE.search(block)
        result = "upper_bound_improved" if improvement else "no_improvement_found"
        record = {
            "record_type": "external_search_attempt",
            "canonical_name": match.group("knot"),
            "target": targets[match.group("knot")],
            "searched_bound_interval": [int(match.group("lower")), int(match.group("upper"))],
            "result": result,
            "source": {
                "repository_url": repository["url"],
                "repository_commit": repository["commit"],
                "file": "output.txt",
                "file_sha256": sha256(log_path),
                "line": index + 1,
                "run_ordinal": int(match.group("ordinal")),
            },
            "replay_verified": False,
            "distillation_eligible": False,
            "training_negative_eligible": False,
            "l10": None,
            "exclusion_reason": (
                "an unsuccessful search is not proof of nonexistence and is not a training negative"
                if improvement is None
                else "external claim lacks a complete native semantic action path"
            ),
        }
        if improvement:
            record["claimed_new_upper"] = int(improvement.group("new_upper"))
            record["external_crossing_change"] = {
                "variant_index": int(improvement.group("variant")),
                "flip_index": int(improvement.group("flip")),
            }
        records.append(record)
    return records


def collect_structured_results(
    outputs: Path, targets: dict[str, dict[str, Any]], repository: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    for path in sorted(outputs.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row.get("knot"))
            if name not in targets:
                continue
            improved = bool(row.get("improved"))
            evidence = row.get("evidence")
            records.append(
                {
                    "record_type": "external_upper_bound_claim"
                    if improved
                    else "external_search_attempt",
                    "canonical_name": name,
                    "target": targets[name],
                    "old_bound_interval": [row.get("old_lower"), row.get("old_upper")],
                    "claimed_new_upper": row.get("new_upper") if improved else None,
                    "result": "upper_bound_improved" if improved else "no_improvement_found",
                    "external_evidence": evidence,
                    "source": {
                        "repository_url": repository["url"],
                        "repository_commit": repository["commit"],
                        "file": str(path.relative_to(outputs.parent)),
                        "file_sha256": sha256(path),
                        "line": line_number,
                    },
                    "replay_verified": False,
                    "distillation_eligible": False,
                    "training_negative_eligible": False,
                    "l10": None,
                    "exclusion_reason": (
                        "external compositional claim lacks a complete native semantic action path"
                        if improved
                        else (
                            "an unsuccessful search is not proof of nonexistence and is not a "
                            "training negative"
                        )
                    ),
                }
            )
    return records


def build_collection(
    *,
    sequences: list[Path],
    upperbounds_repo: Path,
    assessed_repositories: list[tuple[Path, str]],
    knotinfo_xlsx: Path | None = None,
    knotinfo_source_repo: Path | None = None,
) -> dict[str, Any]:
    targets = load_targets(sequences)
    upperbounds = git_snapshot(
        upperbounds_repo,
        assessment="upper-bound workbook, structured result rows, and search log",
    )
    repositories = [upperbounds]
    repositories.extend(
        git_snapshot(path, assessment=assessment) for path, assessment in assessed_repositories
    )
    workbook = upperbounds_repo / "data" / "unknotting.xlsx"
    records = collect_catalogue_rows(workbook, targets, upperbounds)
    if knotinfo_xlsx is not None:
        if knotinfo_source_repo is None:
            raise ValueError("knotinfo_source_repo is required with knotinfo_xlsx")
        resolved_source = knotinfo_source_repo.resolve()
        knotinfo_repository = next(
            repository for repository in repositories if Path(repository["path"]) == resolved_source
        )
        records.extend(
            collect_knotinfo_reference_rows(
                knotinfo_xlsx,
                targets,
                knotinfo_repository,
                resolved_source / "notebooks" / "data" / "knotinfo_data_complete.zip",
            )
        )
    records.extend(
        collect_text_search_attempts(upperbounds_repo / "output.txt", targets, upperbounds)
    )
    records.extend(collect_structured_results(upperbounds_repo / "outputs", targets, upperbounds))
    counts: dict[str, int] = {}
    for record in records:
        kind = str(record["record_type"])
        counts[kind] = counts.get(kind, 0) + 1
    exact_targets = {record["canonical_name"] for record in records}
    positive_claims = [
        record for record in records if record["record_type"] == "external_upper_bound_claim"
    ]
    replayable = [record for record in records if record["replay_verified"]]
    return {
        "schema": "external-upper-bound-collection-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy": {
            "exact_canonical_name_join_only": True,
            "unsuccessful_external_search_is_training_negative": False,
            "distillation_requires_complete_replay_verified_native_path": True,
            "l10_requires_complete_path": True,
            "live_mastery_inputs_modified": False,
        },
        "target_count": len(targets),
        "target_names_with_any_record": len(exact_targets),
        "positive_upper_bound_claims_for_targets": len(positive_claims),
        "replayable_witnesses_for_targets": len(replayable),
        "counts_by_record_type": counts,
        "repositories": repositories,
        "records": sorted(
            records,
            key=lambda record: (
                record["canonical_name"],
                record["record_type"],
                record["source"]["file"],
                int(record["source"].get("line", record["source"].get("row", 0))),
            ),
        ),
    }


def render_summary(collection: dict[str, Any]) -> str:
    counts = collection["counts_by_record_type"]
    lines = [
        "# External upper-bound evidence collection",
        "",
        f"- Target knots: {collection['target_count']}",
        f"- Targets with an exact external record: {collection['target_names_with_any_record']}",
        "- Positive upper-bound claims for current targets: "
        f"{collection['positive_upper_bound_claims_for_targets']}",
        "- Replayable witnesses for current targets: "
        f"{collection['replayable_witnesses_for_targets']}",
        f"- External starting PD/bound rows: {counts.get('external_starting_representation', 0)}",
        f"- External upper-bound reference pointers: "
        f"{counts.get('external_upper_bound_reference', 0)}",
        f"- External search-attempt rows: {counts.get('external_search_attempt', 0)}",
        "",
        "No external record is eligible for distillation unless it includes a complete path that "
        "has replayed successfully in the native semantic action environment. An unsuccessful "
        "search is provenance only, never a negative label.",
        "",
        "## Repository snapshots",
        "",
    ]
    for repository in collection["repositories"]:
        lines.append(
            f"- `{repository['url']}` at `{repository['commit']}`: {repository['assessment']}"
        )
    return "\n".join(lines) + "\n"


def render_coverage_tsv(collection: dict[str, Any]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in collection["records"]:
        grouped.setdefault(record["canonical_name"], []).append(record)
    output = io.StringIO()
    fields = [
        "canonical_name",
        "sequence",
        "sequence_index",
        "challenge_id",
        "target_u",
        "current_lower",
        "current_upper",
        "external_starting_pd",
        "upper_bound_reference",
        "external_search_attempt_records",
        "positive_upper_bound_claim_records",
        "replayable_witness_records",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, dialect="excel-tab", lineterminator="\n")
    writer.writeheader()
    for name, records in sorted(grouped.items()):
        target = records[0]["target"]
        writer.writerow(
            {
                "canonical_name": name,
                "sequence": target["sequence"],
                "sequence_index": target["sequence_index"],
                "challenge_id": target["challenge_id"],
                "target_u": target["target_u"],
                "current_lower": target["bound_interval"][0],
                "current_upper": target["bound_interval"][1],
                "external_starting_pd": int(
                    any(row["record_type"] == "external_starting_representation" for row in records)
                ),
                "upper_bound_reference": int(
                    any(row["record_type"] == "external_upper_bound_reference" for row in records)
                ),
                "external_search_attempt_records": sum(
                    row["record_type"] == "external_search_attempt" for row in records
                ),
                "positive_upper_bound_claim_records": sum(
                    row["record_type"] == "external_upper_bound_claim" for row in records
                ),
                "replayable_witness_records": sum(row["replay_verified"] for row in records),
            }
        )
    return output.getvalue()
