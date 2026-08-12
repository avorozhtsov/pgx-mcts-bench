import json

import pytest

from pgx_mcts_bench.foundation_pretraining import (
    next_dose,
    run_foundation_pretraining,
    source_provenance,
)


def test_nonoptimal_high_solve_rate_raises_training_only() -> None:
    assert next_dose(
        promoted=False,
        solve_rate=0.9,
        native_index=0,
        simulation_index=0,
        native_levels=(5, 8, 12, 16),
        simulation_levels=(64, 128, 256),
        evaluation_target=0.7,
    ) == (1, 0, False)


def test_low_solve_rate_raises_training_and_search() -> None:
    assert next_dose(
        promoted=False,
        solve_rate=0.6,
        native_index=0,
        simulation_index=0,
        native_levels=(5, 8),
        simulation_levels=(64, 128),
        evaluation_target=0.7,
    ) == (1, 1, False)


def test_exhausted_dose_is_reported() -> None:
    assert next_dose(
        promoted=False,
        solve_rate=0.2,
        native_index=1,
        simulation_index=1,
        native_levels=(5, 8),
        simulation_levels=(64, 128),
        evaluation_target=0.7,
    ) == (1, 1, True)


def test_foundation_rejects_nonincreasing_levels(tmp_path) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        run_foundation_pretraining(
            tmp_path,
            candidate_names=["window-local"],
            seeds=[1],
            workers=1,
            native_levels=(5, 5),
        )


def test_source_provenance_hashes_executable_files(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "solver.py").write_text("answer = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )
    first = source_provenance(tmp_path)
    (tmp_path / "src" / "solver.py").write_text("answer = 2\n")
    second = source_provenance(tmp_path)
    assert not first["dirty"]
    assert second["dirty"]
    assert first["executable_source_sha256"] != second["executable_source_sha256"]


def test_manifest_embeds_complete_candidate_specs_and_provenance(monkeypatch, tmp_path) -> None:
    from pgx_mcts_bench import foundation_pretraining as module

    monkeypatch.setattr(module, "_run_one", lambda payload: {"job": payload})
    monkeypatch.setattr(
        module,
        "source_provenance",
        lambda: {"base_commit": "abc", "executable_source_sha256": "def"},
    )
    run_foundation_pretraining(
        tmp_path,
        candidate_names=["window-local"],
        seeds=[71],
        workers=1,
        stage_limit=1,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema"] == "semantic-v1-foundation-pretrain-protocol-v2"
    assert manifest["candidate_specs"][0]["name"] == "window-local"
    assert manifest["candidate_specs"][0]["objective_ratio_weights"] == [1.0, 1.0]
    assert manifest["source_provenance"]["base_commit"] == "abc"

    # A second invocation must accept the JSON-decoded tuple fields written by
    # the first invocation rather than rejecting its own resumable protocol.
    run_foundation_pretraining(
        tmp_path,
        candidate_names=["window-local"],
        seeds=[71],
        workers=1,
        stage_limit=1,
    )


def test_parallel_foundation_workers_use_spawn(monkeypatch, tmp_path) -> None:
    from pgx_mcts_bench import foundation_pretraining as module

    seen = {}

    class Pool:
        def __init__(self, *, max_workers, mp_context):
            seen["workers"] = max_workers
            seen["start_method"] = mp_context.get_start_method()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, jobs):
            return [function(job) for job in jobs]

    monkeypatch.setattr(module, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(module, "_run_one", lambda payload: {"job": payload})
    monkeypatch.setattr(module, "source_provenance", lambda: {"base_commit": "abc"})
    run_foundation_pretraining(
        tmp_path,
        candidate_names=["window-local"],
        seeds=[71, 72],
        workers=2,
        stage_limit=1,
    )
    assert seen == {"workers": 2, "start_method": "spawn"}
