from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _module():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "run_local_q154_fast6_post_completion_recovery.py"
    spec = importlib.util.spec_from_file_location("q154_post_completion_recovery_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_post_completion_recovery_accepts_frozen_unfinished_boundary(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "branch"
    events = output / "events"
    events.mkdir(parents=True)
    (output / "manifest.json").write_text("{}\n")
    (output / "state.pt.gz").write_bytes(b"state")
    event = events / "003.json"
    event.write_text("{}\n")
    binding = {
        "terminal": False,
        "manifest_sha256": _sha256(output / "manifest.json"),
        "event_count": 1,
        "last_event": event.name,
        "last_event_sha256": _sha256(event),
        "state_sha256": _sha256(output / "state.pt.gz"),
    }
    original_output = module.recovery._output
    module.recovery._output = lambda _label: output
    try:
        module._verify_branch_binding("lineage", binding)
        (events / "004.json").write_text(json.dumps({"advanced": True}) + "\n")
        (output / "state.pt.gz").write_bytes(b"advanced-state")
        module._verify_branch_binding("lineage", binding)
    finally:
        module.recovery._output = original_output


def test_post_completion_recovery_keeps_original_out_of_recovery_jobs() -> None:
    module = _module()
    assert module.recovery.ACTIVE_ORIGINAL_LABEL == "q-grown-raster-axial-12"
    labels = [row[0] for row in module.recovery.RECOVERY_BRANCHES]
    assert module.recovery.ACTIVE_ORIGINAL_LABEL not in labels
    assert len(labels) == 5
    assert module.RESUME_TRANSACTION_GATE.name == (
        "FAST6_SLOW4_COHORT_SPLIT_V4_VERIFIED.json"
    )
