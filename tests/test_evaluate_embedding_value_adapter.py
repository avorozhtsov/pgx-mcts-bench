from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_embedding_value_adapter.py"
SPEC = importlib.util.spec_from_file_location("evaluate_embedding_value_adapter", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(base_solved, adapter_solved, base_cost=(None, None), adapter_cost=(None, None)):
    return {
        "baseline_solved": base_solved,
        "adapter_solved": adapter_solved,
        "baseline_crossings": base_cost[0],
        "baseline_moves": base_cost[1],
        "adapter_crossings": adapter_cost[0],
        "adapter_moves": adapter_cost[1],
    }


def test_paired_cost_outcome_prioritizes_solve_then_crossings_then_moves() -> None:
    assert MODULE.paired_cost_outcome(row(False, True)) == "adapter_better"
    assert MODULE.paired_cost_outcome(row(False, False)) == "both_failed"
    assert MODULE.paired_cost_outcome(row(True, True, (8, 1), (3, 99))) == "adapter_better"
    assert MODULE.paired_cost_outcome(row(True, True, (3, 8), (3, 9))) == "baseline_better"
    assert MODULE.paired_cost_outcome(row(True, True, (3, 8), (3, 8))) == "equal_cost"
