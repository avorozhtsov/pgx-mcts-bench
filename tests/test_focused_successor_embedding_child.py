from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_embedding_child_is_main_checkout_hash_gated() -> None:
    preparation = (ROOT / "scripts/prepare_focused_successor_embedding_child.py").read_text()
    launcher = (ROOT / "scripts/run_focused_successor_embedding_child.py").read_text()
    assert 'Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench")' in preparation
    assert 'branch != "main"' in preparation
    assert 'git_value("rev-parse", "HEAD") != git_value("rev-parse", "origin/main")' in preparation
    assert '"source_sha256": tournament["source_sha256"]' in preparation
    assert 'gate.get("commit") != head' in launcher
    assert 'raise RuntimeError("duplicate embedding-child launcher")' in launcher


def test_embedding_child_freezes_inputs_and_uses_fresh_equal_budget_data() -> None:
    preparation = (ROOT / "scripts/prepare_focused_successor_embedding_child.py").read_text()
    launcher = (ROOT / "scripts/run_focused_successor_embedding_child.py").read_text()
    assert '"parent_frozen": True' in preparation
    assert '"embedding_frozen": True' in preparation
    assert '"bridge_initialization": "exact-zero-residual"' in preparation
    assert '"--train-games"' in launcher and '"10"' in launcher
    assert '"--simulations"' in launcher and '"32"' in launcher
    assert '"--context-mode"' in launcher and '"full"' in launcher
    assert '"--stage"' in launcher and '"37"' in launcher
