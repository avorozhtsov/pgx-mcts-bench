from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Position:
    observation: np.ndarray
    legal_actions: np.ndarray
    policy: np.ndarray
    action: int
    player: int
    reward: float = 0.0
    next_terminated: bool = False
    outcome: float = 0.0


GameRecord = list[Position]


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.games: list[GameRecord] = []
        self.position_count = 0

    def add(self, game: GameRecord) -> None:
        self.games.append(game)
        self.position_count += len(game)
        while self.games and self.position_count > self.capacity:
            removed = self.games.pop(0)
            self.position_count -= len(removed)

    def sample_positions(self, batch_size: int) -> list[Position]:
        positions = [position for game in self.games for position in game]
        if not positions:
            raise RuntimeError("Cannot sample an empty replay buffer")
        indexes = self.rng.integers(0, len(positions), size=batch_size)
        return [positions[int(index)] for index in indexes]

    def sample_sequences(
        self,
        batch_size: int,
        unroll_steps: int,
        terminal_fraction: float = 0.5,
    ) -> list[GameRecord]:
        """Sample unrolls, deliberately including terminal transitions.

        Uniform position sampling makes the only non-zero reward and terminal
        target in a Go game extremely rare. A fixed fraction of sequences is
        therefore aligned so their final transition ends the game.
        """
        if not self.games:
            raise RuntimeError("Cannot sample an empty replay buffer")
        sequences: list[GameRecord] = []
        terminal_samples = round(batch_size * terminal_fraction)
        for index in range(batch_size):
            game = self.games[int(self.rng.integers(0, len(self.games)))]
            if index < terminal_samples:
                start = max(0, len(game) - unroll_steps - 1)
            else:
                start = int(self.rng.integers(0, len(game)))
            sequences.append(game[start : start + unroll_steps + 1])
        return sequences
