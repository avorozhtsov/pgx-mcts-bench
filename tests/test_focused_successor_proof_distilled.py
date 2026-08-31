from pathlib import Path


def test_proof_distillation_is_conservative_and_continues_rl() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts/run_focused_successor_proof_distilled.py").read_text()
    preparation = (Path(__file__).resolve().parents[1] / "scripts/prepare_focused_proof_distillation_gate.py").read_text()
    assert '"absent_actions": "unknown-not-negative"' in preparation
    assert 'adapter_manifest["dataset"]["sha256"]' in source
    assert "for key, value in old_network.items()" in source
    assert "torch.equal(value, new_network[key])" in source
    assert '"strand-graph-12-proof-distilled"' in source
    assert "another proof-distilled launcher holds the lock" in source
    assert "proof distillation launch gate did not pass" in source


def test_recovery_v2_has_dedicated_proof_runner() -> None:
    root = Path(__file__).resolve().parents[1]
    branch = (root / "scripts/run_focused_successor_proof_branch.py").read_text()
    recovery = (root / "scripts/run_focused_successor_proof_distilled_recovery_v2.py").read_text()
    assert 'gate.get("cohort") != "focused-successor-v1-proof-distilled"' in branch
    assert 'name = "pgx_mcts_bench.q254_sv2_curriculum_runtime"' in branch
    assert 'q50-4-updated-scheduled-no-sharing-bounded-proof-recovery-v2' in recovery
