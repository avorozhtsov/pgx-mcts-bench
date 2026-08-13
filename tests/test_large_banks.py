from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "build_sv2_large_banks.py"
    spec = importlib.util.spec_from_file_location("build_sv2_large_banks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workbook_bounds_reads_exact_and_range_cells(tmp_path: Path) -> None:
    module = _module()
    worksheet = """<?xml version="1.0" encoding="UTF-8"?>
    <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <sheetData>
        <row r="1"><c r="A1" t="inlineStr"><is><t>knot_id</t></is></c></row>
        <row r="2"><c r="A2" t="inlineStr"><is><t>3_1</t></is></c><c r="D2"><v>1</v></c></row>
        <row r="3"><c r="A3" t="inlineStr"><is><t>12n_1</t></is></c>
          <c r="D3" t="inlineStr"><is><t>[2, 4]</t></is></c></row>
      </sheetData>
    </worksheet>"""
    workbook = tmp_path / "bounds.xlsx"
    with zipfile.ZipFile(workbook, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
    assert module.workbook_bounds(workbook, {"3_1", "12n_1"}) == {
        "3_1": (1, 1),
        "12n_1": (2, 4),
    }


def test_markov_variant_is_distinct_and_preserves_basic_invariants() -> None:
    module = _module()
    source = {"braid": [1, 1, 1], "strands": 2, "crossings": 3}
    original = (tuple(source["braid"]), source["strands"])
    variant = module.make_variant("3_1", source, seed=17, occupied={original})
    assert variant is not None and variant != original

    from rf_knots.invariants import alexander_polynomial, jones_polynomial
    from rf_knots.reference import num_components

    word, strands = variant
    assert num_components(word, strands) == 1
    assert alexander_polynomial(word, strands) == alexander_polynomial(*original)
    assert jones_polynomial(word, strands) == jones_polynomial(*original)
