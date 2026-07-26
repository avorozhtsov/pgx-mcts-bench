from __future__ import annotations

import math

import numpy as np

from pgx_mcts_bench.config import ExplorationKind, SearchConfig


def exploration_bonus(
    kind: ExplorationKind,
    prior: np.ndarray,
    parent_visits: int,
    child_visits: np.ndarray,
    config: SearchConfig,
) -> np.ndarray:
    """Return U(s, a) for the five benchmarked count-based rules."""
    p = np.asarray(prior, dtype=np.float64)
    n = np.asarray(child_visits, dtype=np.float64)
    parent_n = float(parent_visits)

    if kind == "u1":
        # AlphaZero PUCT.
        factor = math.sqrt(parent_n) / (1.0 + n)
        return config.cpuct * p * factor
    if kind == "u2":
        # Slower child-count decay proposed for this experiment.
        factor = math.sqrt(parent_n) / np.sqrt(1.0 + n)
        return config.cpuct * p * factor
    if kind == "u3":
        # Classical UCT, included as a prior-free control.
        return config.cpuct * np.sqrt(math.log(parent_n + 1.0) / (1.0 + n))
    if kind == "u4":
        # Prior-weighted UCT.
        return config.cpuct * np.sqrt(p * math.log(parent_n + 1.0) / (1.0 + n))
    if kind == "u5":
        # MuZero's slowly increasing pb_c multiplier.
        pb_c = config.c1 + math.log((parent_n + config.c2 + 1.0) / config.c2)
        return config.cpuct * p * math.sqrt(parent_n) * pb_c / (1.0 + n)
    raise ValueError(f"Unknown exploration rule: {kind}")


def describe_rules() -> dict[str, str]:
    return {
        "u1": "c P sqrt(N)/(1+n) — AlphaZero PUCT",
        "u2": "c P sqrt(N)/sqrt(1+n) — slow child-count decay",
        "u3": "c sqrt(log(N+1)/(1+n)) — UCT control",
        "u4": "c sqrt(P log(N+1)/(1+n)) — prior-weighted UCT",
        "u5": "c P sqrt(N)/(1+n) [c1+log((N+c2+1)/c2)] — MuZero pb_c",
    }
