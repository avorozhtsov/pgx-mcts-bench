#!/usr/bin/env python3
"""Freeze the 2,700-representation SV2 continuation into 400-item blocks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

XMLNS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workbook_bounds(path: Path, wanted: set[str]) -> dict[str, tuple[int, int]]:
    """Read columns A/D from the pinned inline-string XLSX without pandas."""
    result: dict[str, tuple[int, int]] = {}
    cell_tag = f"{{{XMLNS}}}c"
    row_tag = f"{{{XMLNS}}}row"
    text_tag = f"{{{XMLNS}}}t"
    value_tag = f"{{{XMLNS}}}v"
    with zipfile.ZipFile(path) as archive, archive.open("xl/worksheets/sheet1.xml") as worksheet:
        for _, row in ElementTree.iterparse(worksheet, events=("end",)):
            if row.tag != row_tag:
                continue
            values: dict[str, str] = {}
            for cell in row.findall(cell_tag):
                reference = cell.attrib.get("r", "")
                match = re.match(r"[A-Z]+", reference)
                if match is None or match.group() not in {"A", "D"}:
                    continue
                column = match.group()
                texts = [node.text or "" for node in cell.iter(text_tag)]
                value = cell.find(value_tag)
                values[column] = (
                    "".join(texts)
                    if texts
                    else (value.text if value is not None and value.text is not None else "")
                )
            name = values.get("A")
            if name in wanted:
                raw = values.get("D", "")
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple)):
                    lower, upper = int(parsed[0]), int(parsed[-1])
                else:
                    lower = upper = int(float(parsed))
                if not 0 <= lower <= upper:
                    raise ValueError(f"invalid unknotting range for {name}: {raw}")
                result[name] = (lower, upper)
            row.clear()
    missing = wanted - set(result)
    if missing:
        raise ValueError(f"workbook lacks {len(missing)} knots: {sorted(missing)[:5]}")
    return result


def acs(*, strands: int, upper_bound: int, word: tuple[int, ...]) -> float:
    return float(10 * strands + 5 * upper_bound + len(word))


def row_for(
    name: str,
    source: dict,
    word: tuple[int, ...],
    strands: int,
    bounds: tuple[int, int],
    *,
    identity: str,
    variant: str,
) -> dict:
    lower, upper = bounds
    score = acs(strands=strands, upper_bound=upper, word=word)
    row = {
        "id": identity,
        "name": name,
        "word": list(word),
        "strands": strands,
        "crossings": len(word),
        "presentation_crossings": len(word),
        "source_minimal_crossings": int(source["crossings"]),
        "certified_unknotting_lower_bound": lower,
        "certified_unknotting_upper_bound": upper,
        "acs": score,
        "cheap_score": score,
        "representation_variant": variant,
    }
    if lower == upper:
        row["known_unknotting_number"] = upper
    return row


def make_variant(
    name: str,
    source: dict,
    *,
    seed: int,
    occupied: set[tuple[tuple[int, ...], int]],
) -> tuple[tuple[int, ...], int] | None:
    from rf_knots.actions import PASS, ActionSpec
    from rf_knots.reference import successors

    original = (tuple(int(value) for value in source["braid"]), int(source["strands"]))
    spec = ActionSpec(max_len=48, max_strands=5)
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    for attempt in range(16):
        word, strands = original
        depth = 2 + (attempt % 5)
        for _ in range(depth):
            options = [
                (next_word, next_strands)
                for action, next_word, next_strands in successors(
                    spec, word, strands, allow_crossing=False
                )
                if spec.decode(action)[0] != PASS
                and len(next_word) <= 48
                and next_strands <= 5
                and (next_word, next_strands) != (word, strands)
            ]
            if not options:
                break
            word, strands = options[int(rng.integers(len(options)))]
        candidate = (tuple(word), int(strands))
        if candidate != original and candidate not in occupied:
            return candidate
    return None


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build(
    knot_table: Path,
    workbook: Path,
    r24: Path,
    r200: Path,
    output: Path,
    *,
    seed: int,
    total: int = 2700,
    block_size: int = 400,
) -> dict:
    table_payload = json.loads(knot_table.read_text())
    table = {
        name: row
        for name, row in table_payload["knots"].items()
        if int(row["strands"]) <= 5 and len(row["braid"]) <= 48
    }
    if total < len(table):
        raise ValueError("total is smaller than the compatible canonical table")
    bounds = workbook_bounds(workbook, set(table))
    canonical = [
        row_for(
            name,
            source,
            tuple(int(value) for value in source["braid"]),
            int(source["strands"]),
            bounds[name],
            identity=name,
            variant="canonical",
        )
        for name, source in table.items()
    ]
    occupied = {(tuple(row["word"]), int(row["strands"])) for row in canonical}
    variant_count = total - len(canonical)
    variants = []
    order = sorted(
        table,
        key=lambda name: (hashlib.sha256(f"{seed}:{name}".encode()).digest(), name),
    )
    for name in order:
        generated = make_variant(name, table[name], seed=seed, occupied=occupied)
        if generated is None:
            continue
        word, strands = generated
        occupied.add(generated)
        variants.append(
            row_for(
                name,
                table[name],
                word,
                strands,
                bounds[name],
                identity=f"{name}::markov-v1",
                variant="deterministic-markov-v1",
            )
        )
        if len(variants) == variant_count:
            break
    if len(variants) != variant_count:
        raise ValueError(f"generated {len(variants)} of {variant_count} required variants")

    r200_payload = json.loads(r200.read_text())
    r200_rows = r200_payload["rows"]
    r200_ids = {str(row["id"]) for row in r200_rows}
    all_rows = canonical + variants
    if len(all_rows) != total or len({row["id"] for row in all_rows}) != total:
        raise AssertionError("master representation bank is not exactly unique")
    if not r200_ids <= {row["id"] for row in canonical}:
        raise ValueError("R200 is not a canonical subset of the master bank")
    remaining = [row for row in all_rows if row["id"] not in r200_ids]
    remaining.sort(
        key=lambda row: (
            row["acs"],
            row["strands"],
            row["certified_unknotting_upper_bound"],
            row["presentation_crossings"],
            row["id"],
        )
    )
    r24_rows = json.loads(r24.read_text())["rows"]
    prior_rows = list(r24_rows) + list(r200_rows)
    blocks = []
    for index, start in enumerate(range(0, len(remaining), block_size), start=1):
        rows = remaining[start : start + block_size]
        suffix = f"r400-{index}" if len(rows) == block_size else f"r{len(rows)}-tail"
        prior_path = output / f"prior-{suffix}.json"
        bank_path = output / f"{suffix}.json"
        write_json(
            prior_path,
            {
                "schema": "semantic-v2-cumulative-prior-v1",
                "rows": prior_rows,
            },
        )
        write_json(
            bank_path,
            {
                "schema": "semantic-v2-continuation-block-v1",
                "block": index,
                "ordering": "ACS=10*strands+5*certified_u_upper_bound+presentation_crossings",
                "rows": rows,
            },
        )
        blocks.append(
            {
                "index": index,
                "name": suffix,
                "rows": len(rows),
                "bank": bank_path.name,
                "bank_sha256": sha256(bank_path),
                "prior": prior_path.name,
                "prior_sha256": sha256(prior_path),
            }
        )
        prior_rows.extend(rows)
    manifest = {
        "schema": "semantic-v2-2700-block-manifest-v1",
        "seed": seed,
        "total_representations": total,
        "canonical_representations": len(canonical),
        "markov_variants": len(variants),
        "already_frozen_r200": len(r200_rows),
        "continuation_representations": len(remaining),
        "knot_table": "rf-knots:src/rf_knots/data/knot_table.json",
        "knot_table_sha256": sha256(knot_table),
        "upperbounds_workbook": "dtubbenhauer/upperbounds:data/unknotting.xlsx",
        "upperbounds_workbook_sha256": sha256(workbook),
        "upperbounds_commit": "de66f29045e804931edd6d1c9735247f81ad68c1",
        "r24": "research/semantic-v2-prefix24.json",
        "r24_sha256": sha256(r24),
        "r200": "research/semantic-v2-r200.json",
        "r200_sha256": sha256(r200),
        "blocks": blocks,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("knot_table", type=Path)
    parser.add_argument("upperbounds_workbook", type=Path)
    parser.add_argument("r24", type=Path)
    parser.add_argument("r200", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20262270)
    args = parser.parse_args()
    manifest = build(
        args.knot_table,
        args.upperbounds_workbook,
        args.r24,
        args.r200,
        args.output,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
