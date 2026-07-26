import math

import numpy as np

from pgx_mcts_bench.config import SearchConfig
from pgx_mcts_bench.exploration import exploration_bonus


def test_u1_is_alphazero_puct() -> None:
    config = SearchConfig(cpuct=2.0)
    actual = exploration_bonus(
        "u1",
        np.array([0.25, 0.75]),
        parent_visits=16,
        child_visits=np.array([3, 0]),
        config=config,
    )
    np.testing.assert_allclose(actual, np.array([0.5, 6.0]))


def test_u2_has_slower_child_count_decay_than_u1() -> None:
    config = SearchConfig()
    prior = np.array([1.0])
    visits = np.array([15])
    u1 = exploration_bonus("u1", prior, 20, visits, config)
    u2 = exploration_bonus("u2", prior, 20, visits, config)
    assert u2.item() > u1.item()


def test_u3_does_not_depend_on_prior() -> None:
    config = SearchConfig(cpuct=1.0)
    actual = exploration_bonus(
        "u3", np.array([0.01, 0.99]), 10, np.array([2, 2]), config
    )
    expected = math.sqrt(math.log(11) / 3)
    np.testing.assert_allclose(actual, [expected, expected])
