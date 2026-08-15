from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

from pgx_mcts_bench.external_evidence import (
    build_collection,
    collect_text_search_attempts,
    iter_xlsx_rows,
)


def make_sequence(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "test-sequence",
                "challenges": [
                    {
                        "challenge_id": "challenge-a",
                        "canonical_name": "12n_1",
                        "target_u": 1,
                        "bound_interval": [1, 2],
                    }
                ],
            }
        )
    )


def make_xlsx(path: Path) -> None:
    sheet = """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>
<row r="1">
<c r="A1" t="inlineStr"><is><t>knot_id</t></is></c>
<c r="B1" t="inlineStr"><is><t>jones_vector</t></is></c>
<c r="C1" t="inlineStr"><is><t>pd_presentation</t></is></c>
<c r="D1" t="inlineStr"><is><t>unknotting_number</t></is></c>
</row>
<row r="2">
<c r="A2" t="inlineStr"><is><t>12n_1</t></is></c>
<c r="B2" t="inlineStr"><is><t>[1, 2]</t></is></c>
<c r="C2" t="inlineStr"><is><t>[[0,1,2,3]]</t></is></c>
<c r="D2" t="inlineStr"><is><t>[1,2]</t></is></c>
</row>
</sheetData></worksheet>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def init_repo(path: Path, files: dict[str, str | bytes]) -> None:
    path.mkdir(exist_ok=True)
    for name, contents in files.items():
        destination = path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents if isinstance(contents, bytes) else contents.encode())
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "remote", "add", "origin", "https://example.test/repo"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )


def test_iter_xlsx_rows_reads_inline_and_numeric_cells(tmp_path: Path):
    path = tmp_path / "test.xlsx"
    make_xlsx(path)
    assert list(iter_xlsx_rows(path)) == [
        (
            2,
            {
                "knot_id": "12n_1",
                "jones_vector": "[1, 2]",
                "pd_presentation": "[[0,1,2,3]]",
                "unknotting_number": "[1,2]",
            },
        )
    ]


def test_unsuccessful_search_is_never_a_training_negative(tmp_path: Path):
    log = tmp_path / "output.txt"
    log.write_text("[1/2] Processing: 12n_1  current range: [1, 2]\n  No improvement found.\n")
    repository = {"url": "https://example.test/repo", "commit": "abc"}
    targets = {"12n_1": {"canonical_name": "12n_1"}}
    records = collect_text_search_attempts(log, targets, repository)
    assert len(records) == 1
    assert records[0]["result"] == "no_improvement_found"
    assert records[0]["training_negative_eligible"] is False
    assert records[0]["distillation_eligible"] is False


def test_collection_fails_closed_without_complete_paths(tmp_path: Path):
    sequence = tmp_path / "sequence.json"
    make_sequence(sequence)
    upperbounds = tmp_path / "upperbounds"
    upperbounds.mkdir()
    make_xlsx(upperbounds / "unknotting.xlsx")
    (upperbounds / "data").mkdir()
    (upperbounds / "unknotting.xlsx").replace(upperbounds / "data" / "unknotting.xlsx")
    (upperbounds / "outputs").mkdir()
    (upperbounds / "outputs" / "results.jsonl").write_text(
        json.dumps(
            {
                "knot": "12n_1",
                "old_lower": 1,
                "old_upper": 2,
                "new_upper": 1,
                "improved": True,
                "evidence": {"flips": [3]},
            }
        )
        + "\n"
    )
    (upperbounds / "output.txt").write_text(
        "[1/1] Processing: 12n_1  current range: [1, 2]\n"
        "Improved via variant 1, flip 3: upper 2 -> 1\n"
    )
    auxiliary = []
    for name in ("unknotter", "untangling", "lowerbounds"):
        repo = tmp_path / name
        init_repo(repo, {"README.md": name})
        auxiliary.append((repo, name))
    init_repo(
        upperbounds,
        {
            "placeholder": "already populated before git init",
        },
    )
    collection = build_collection(
        sequences=[sequence],
        upperbounds_repo=upperbounds,
        assessed_repositories=auxiliary,
    )
    assert collection["positive_upper_bound_claims_for_targets"] == 1
    assert collection["replayable_witnesses_for_targets"] == 0
    assert all(not record["distillation_eligible"] for record in collection["records"])
    assert all(not record["training_negative_eligible"] for record in collection["records"])
