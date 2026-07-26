from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from pgx.go import Go

from pgx_mcts_bench.config import GameConfig


@dataclass(frozen=True)
class Transition:
    state: Any
    observation: np.ndarray
    legal_actions: np.ndarray
    reward: float
    terminated: bool
    player: int


class Go6x6:
    """Thin non-jitted Pgx adapter with observations in the side-to-move frame."""

    def __init__(self, config: GameConfig):
        if config.board_size != 6:
            raise ValueError("This benchmark is intentionally pinned to 6x6 Go")
        self.config = config
        self.env = Go(
            size=config.board_size,
            komi=config.komi,
            history_length=config.history_length,
            max_terminal_steps=config.max_moves,
        )
        # AlphaZero invokes the exact transition inside every simulation. Without
        # JIT this Python-loop benchmark spends nearly all its time dispatching
        # the many small JAX operations that implement one Go move.
        self._init = jax.jit(self.env.init)
        self._step = jax.jit(self.env.step)

    def reset(self, seed: int) -> Transition:
        state = self._init(jax.random.PRNGKey(seed))
        return self._view(state, reward=0.0)

    def step(self, state: Any, action: int) -> Transition:
        if not bool(self._legal_actions(state)[action]):
            raise ValueError(f"Illegal action {action}")
        actor = int(np.asarray(state.current_player))
        next_state = self._step(state, np.int32(action))
        rewards = np.asarray(next_state.rewards, dtype=np.float32)
        return self._view(next_state, reward=float(rewards[actor]))

    def _view(self, state: Any, reward: float) -> Transition:
        observation = np.asarray(state.observation, dtype=np.float32)
        size = self.config.board_size
        previous_pass = float(np.asarray(state._x.consecutive_pass_count) > 0)
        move_fraction = min(
            float(np.asarray(state._x.step_count)) / self.config.max_moves,
            1.0,
        )
        metadata = np.stack(
            [
                np.full((size, size), previous_pass, dtype=np.float32),
                np.full((size, size), move_fraction, dtype=np.float32),
            ],
            axis=-1,
        )
        return Transition(
            state=state,
            observation=np.concatenate([observation, metadata], axis=-1),
            legal_actions=self._legal_actions(state),
            reward=reward,
            terminated=bool(np.asarray(state.terminated)),
            player=int(np.asarray(state.current_player)),
        )

    def final_rewards(self, state: Any) -> np.ndarray:
        return np.asarray(state.rewards, dtype=np.float32)

    def _legal_actions(self, state: Any) -> np.ndarray:
        legal = np.asarray(state.legal_action_mask, dtype=bool).copy()
        move_count = int(np.asarray(state._x.step_count))
        if move_count < self.config.min_moves_before_pass:
            legal[-1] = False
        return legal
