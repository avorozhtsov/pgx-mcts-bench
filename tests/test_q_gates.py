from __future__ import annotations

import json
from pathlib import Path

from pgx_mcts_bench.q_gates import audit_q20_gate


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _fixture(root: Path, *, solved_high: bool = True, retention: float = 1.0) -> None:
    scientists = ("a", "b")
    _write(
        root / "manifest.json",
        {"requested_rungs": 20, "checkpoints": {name: {} for name in scientists}},
    )
    rows = [{"id": f"k{index}", "strands": 6 if index == 19 else 3} for index in range(20)]
    _write(root / "bank.json", {"rows": rows})
    for index, row in enumerate(rows):
        scientist_rows = {}
        for scientist in scientists:
            high = solved_high and index == 19
            scientist_rows[scientist] = {
                "curriculum_skip": None,
                "native_best": {"1000.0": {"best_objective": 1001.0} if high else None},
                "evaluation": {
                    "10.0": {"best_objective": 20.0},
                    "1000.0": {"best_objective": 1001.0},
                },
            }
        _write(
            root / "native-events" / f"{index:03d}.json",
            {"round": index, "selected": row["id"], "scientists": scientist_rows},
        )
    attempts = 20
    _write(
        root / "blocks" / "020.json",
        {
            "scientists": {
                name: {
                    "retention_after": {"solved": int(retention * attempts), "attempts": attempts}
                }
                for name in scientists
            }
        },
    )


def test_q20_gate_requires_complete_durable_success_and_retention(tmp_path: Path) -> None:
    _fixture(tmp_path)
    report = audit_q20_gate(tmp_path, tmp_path / "gate.json")
    assert report["status"] == "passed"
    assert report["durable_scientist_items"] == 40
    assert report["advance_q40_1"] is True
    assert report["scientists"]["a"]["metrics_including_all_rows"]["1000.0"]["denominator"] == 20


def test_q20_gate_fails_without_high_strand_success(tmp_path: Path) -> None:
    _fixture(tmp_path, solved_high=False)
    report = audit_q20_gate(tmp_path)
    assert report["status"] == "failed"
    assert report["failures"] == ["no_native_success_on_6plus_strands"]


def test_q20_gate_stays_pending_until_all_native_events_are_committed(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "native-events" / "019.json").unlink()
    report = audit_q20_gate(tmp_path)
    assert report["status"] == "pending"
    assert report["advance_q40_1"] is False
