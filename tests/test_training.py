from dataclasses import replace

import pytest

from pgx_mcts_bench.config import (
    ExperimentConfig,
    ModelConfig,
    SearchConfig,
    TrainConfig,
)
from pgx_mcts_bench.training import _limit_records, train_agent


def _tiny_config(iterations: int) -> ExperimentConfig:
    return ExperimentConfig(
        search=SearchConfig(simulations=1),
        model=ModelConfig(channels=4, residual_blocks=1, latent_channels=4),
        train=TrainConfig(
            iterations=iterations,
            selfplay_games=1,
            selfplay_positions_per_iteration=5,
            train_steps=1,
            batch_size=2,
            checkpoint_iterations=(1, iterations),
        ),
    )


def test_limit_records_keeps_exact_position_budget() -> None:
    records = [[object() for _ in range(4)], [object() for _ in range(4)]]
    limited = _limit_records(records, 6)
    assert [len(record) for record in limited] == [4, 2]


def test_training_checkpoint_resumes_at_next_iteration(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    first = train_agent(
        "alphazero",
        _tiny_config(1),
        checkpoint_dir=checkpoint_dir,
    )
    assert first.history[0]["positions_generated"] == 5
    assert first.history[0]["positions_simulated"] >= 5

    resumed_config = replace(
        _tiny_config(1),
        train=replace(
            _tiny_config(1).train,
            iterations=2,
            checkpoint_iterations=(1, 2),
        ),
    )
    resumed = train_agent(
        "alphazero",
        resumed_config,
        checkpoint_dir=checkpoint_dir,
        resume=True,
    )
    assert [row["iteration"] for row in resumed.history] == [1.0, 2.0]
    assert (checkpoint_dir / "alphazero-iteration-0002.pt").exists()


def test_resume_rejects_changed_experiment_configuration(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    config = _tiny_config(1)
    train_agent("alphazero", config, checkpoint_dir=checkpoint_dir)
    changed = replace(config, search=replace(config.search, exploration="u2"))
    with pytest.raises(ValueError, match="does not match"):
        train_agent(
            "alphazero",
            changed,
            checkpoint_dir=checkpoint_dir,
            resume=True,
        )
