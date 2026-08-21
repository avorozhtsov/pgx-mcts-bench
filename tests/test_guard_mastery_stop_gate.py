from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "guard_mastery_stop_gate.py"
SPEC = importlib.util.spec_from_file_location("guard_mastery_stop_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
scientific_success_reasons = MODULE.scientific_success_reasons
verify_checkpoint = MODULE.verify_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scientific_success_reasons_detects_durable_improvement() -> None:
    state = {
        "challenges": {
            "k": {
                "challenge_id": "k",
                "initial_upper_bound": 3,
                "current_upper_bound": 2,
                "status": "active",
            }
        },
        "recent_events": [],
    }
    assert scientific_success_reasons(state) == ["upper-bound-improvement:k"]


def test_verify_checkpoint_checks_exact_target_and_hashes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "group-100"
    checkpoint.mkdir()
    program = checkpoint / "program-state.json"
    scientist = checkpoint / "scientist-state.pt.gz"
    program.write_text(json.dumps({"introduced_count": 100, "step_index": 321}))
    scientist.write_bytes(b"checkpoint")
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "introduced": 100,
                "program_state_sha256": _sha256(program),
                "scientist_state_sha256": _sha256(scientist),
            }
        )
    )

    verified = verify_checkpoint(checkpoint, 100)

    assert verified["introduced"] == 100
    assert verified["step"] == 321


def test_verify_checkpoint_rejects_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "group-160"
    checkpoint.mkdir()
    (checkpoint / "program-state.json").write_text(json.dumps({"introduced_count": 160}))
    (checkpoint / "scientist-state.pt.gz").write_bytes(b"checkpoint")
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "introduced": 160,
                "program_state_sha256": "bad",
                "scientist_state_sha256": "bad",
            }
        )
    )

    with pytest.raises(ValueError, match="program-state hash mismatch"):
        verify_checkpoint(checkpoint, 160)
