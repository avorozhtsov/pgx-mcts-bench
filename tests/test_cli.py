from typer.testing import CliRunner

from pgx_mcts_bench.cli import app


def test_braid_ladder_rejects_unknown_candidate_before_starting_work() -> None:
    result = CliRunner().invoke(app, ["braid-ladder", "--only", "not-a-candidate"])

    assert result.exit_code == 2
    assert "unknown candidate(s): not-a-candidate" in result.output
