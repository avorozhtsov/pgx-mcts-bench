from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "build_q44_2_updated.py"
    spec = importlib.util.spec_from_file_location("build_q44_2_updated", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_markov_bridge_is_simple_and_replayable() -> None:
    module = _module()
    source = {
        "id": "3_1",
        "name": "3_1",
        "word": [1, 1, 1],
        "strands": 2,
        "crossings": 3,
        "presentation_crossings": 3,
        "source_minimal_crossings": 3,
        "certified_unknotting_lower_bound": 1,
        "certified_unknotting_upper_bound": 1,
        "known_unknotting_number": 1,
    }
    bridge = module.markov_stabilize(source, 6, seed=7, ordinal=1)
    assert bridge["strands"] == 6
    assert len(bridge["markov_stabilization_letters"]) == 4
    assert [abs(x) for x in bridge["markov_stabilization_letters"]] == [2, 3, 4, 5]
    assert bridge["word"][:3] == source["word"]
    assert bridge["verification"]["preserves_knot_identity"] is True
    module._audit_bridge(source, bridge)


def test_registered_target_mix_is_q44() -> None:
    module = _module()
    assert len(module.TARGET_STRANDS) == 14
    counts = {strand: module.TARGET_STRANDS.count(strand) for strand in set(module.TARGET_STRANDS)}
    assert counts == {
        5: 5,
        6: 4,
        7: 4,
        8: 1,
    }
