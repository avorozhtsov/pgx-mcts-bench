"""Read-only, denominator-safe gates for Q-program groups."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _contains_capacity_exception(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.lower().replace("-", "_")
        return "capacity" in normalized or "unsupported" in normalized
    if isinstance(value, dict):
        return any(_contains_capacity_exception(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_capacity_exception(item) for item in value)
    return False


def _solved(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("solved") is not None:
        return bool(row["solved"])
    return any(
        row.get(key) is not None for key in ("best_objective", "objective", "semantic_actions")
    )


def audit_q20_gate(
    root: Path,
    output: Path | None = None,
    *,
    retention_target: float = 0.80,
    failure_caps: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Audit Q20 without mutating or resuming its coordinated workers."""

    manifest = json.loads((root / "manifest.json").read_text())
    bank = json.loads((root / "bank.json").read_text())
    bank_rows = {str(row.get("id", row.get("name"))): row for row in bank.get("rows", [])}
    scientists = sorted((manifest.get("checkpoints") or {}).keys())
    expected_rungs = int(manifest.get("requested_rungs", 20))
    events = []
    malformed = []
    for path in sorted((root / "native-events").glob("*.json")):
        try:
            event = json.loads(path.read_text())
            if sorted(event.get("scientists", {})) != scientists:
                malformed.append(f"scientist_set:{path.name}")
            events.append(event)
        except (OSError, json.JSONDecodeError) as error:
            malformed.append(f"invalid_json:{path.name}:{error}")
    rounds = sorted(int(event.get("round", -1)) for event in events)
    exact_rounds = rounds == list(range(expected_rungs))
    block_path = root / "blocks" / f"{expected_rungs:03d}.json"
    final_block = json.loads(block_path.read_text()) if block_path.is_file() else None
    caps = failure_caps or {"10.0": 328.0, "1000.0": 20_128.0}
    rows_by_scientist: dict[str, dict[str, Any]] = {}
    any_high_strand_native_success = False
    capacity_exception_events = []
    for scientist in scientists:
        skips = 0
        ratio_values = {ratio: [] for ratio in caps}
        high_success_ids = []
        for event in events:
            scientist_row = (event.get("scientists") or {}).get(scientist, {})
            if scientist_row.get("curriculum_skip") is not None:
                skips += 1
            selected = str(event.get("selected"))
            bank_row = bank_rows.get(selected, {})
            if int(bank_row.get("strands", 0)) >= 6 and any(
                _solved(value) for value in (scientist_row.get("native_best") or {}).values()
            ):
                high_success_ids.append(selected)
                any_high_strand_native_success = True
            for ratio, cap in caps.items():
                evaluation = (scientist_row.get("evaluation") or {}).get(ratio) or {}
                objective = evaluation.get("best_objective")
                ratio_values[ratio].append(float(cap if objective is None else objective))
            if _contains_capacity_exception(scientist_row):
                capacity_exception_events.append(
                    {"scientist": scientist, "round": event.get("round"), "selected": selected}
                )
        retention = None
        if final_block is not None:
            retained = ((final_block.get("scientists") or {}).get(scientist) or {}).get(
                "retention_after", {}
            )
            attempts = int(retained.get("attempts", 0))
            retention = float(retained.get("solved", 0)) / attempts if attempts else None
        rows_by_scientist[scientist] = {
            "native_events": len(events),
            "curriculum_skips": skips,
            "skip_limit": 1,
            "retention": retention,
            "native_success_ids_strands_ge_6": sorted(set(high_success_ids)),
            "metrics_including_all_rows": {
                ratio: {
                    "coverage": sum(value < caps[ratio] for value in values),
                    "denominator": expected_rungs,
                    "capped_objective_mean": sum(values) / expected_rungs
                    if len(values) == expected_rungs
                    else None,
                    "failure_cap": caps[ratio],
                }
                for ratio, values in ratio_values.items()
            },
        }
    complete = (
        len(events) == expected_rungs and exact_rounds and not malformed and final_block is not None
    )
    failures = []
    if complete:
        for scientist, row in rows_by_scientist.items():
            if row["curriculum_skips"] > row["skip_limit"]:
                failures.append(f"skip_limit_exceeded:{scientist}")
            if row["retention"] is None or row["retention"] < retention_target:
                failures.append(f"retention_below_target:{scientist}")
        if capacity_exception_events:
            failures.append("capacity_exception")
        if not any_high_strand_native_success:
            failures.append("no_native_success_on_6plus_strands")
    report = {
        "schema": "q20-capacity-gate-audit-v1",
        "root": str(root.resolve()),
        "expected_rungs": expected_rungs,
        "durable_rungs": len(events),
        "durable_scientist_items": len(events) * len(scientists),
        "expected_scientist_items": expected_rungs * len(scientists),
        "rounds_exact": exact_rounds,
        "final_block_present": final_block is not None,
        "malformed": malformed,
        "capacity_exception_events": capacity_exception_events,
        "any_native_success_on_6plus_strands": any_high_strand_native_success,
        "scientists": rows_by_scientist,
        "status": "pending" if not complete else ("failed" if failures else "passed"),
        "failures": failures,
        "advance_q40_1": complete and not failures,
    }
    if output is not None:
        _atomic_json(output, report)
    return report
