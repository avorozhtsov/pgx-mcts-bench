from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Position:
    observation: np.ndarray
    legal_actions: np.ndarray
    policy: np.ndarray
    action: int
    player: int
    # 0 if the mover held the game's first role (Black / Scrambler), else 1.
    role: int = 0
    reward: float = 0.0
    next_terminated: bool = False
    outcome: float = 0.0
    # Shadow factorized-value targets. ``solved < 0`` means this position came
    # from an old checkpoint or a game without braid cost labels.
    solved: float = -1.0
    final_crossing_changes: float = float("nan")
    final_moves: float = float("nan")
    # All positions from one episode share this seed, giving the four auxiliary
    # members a deterministic per-episode bootstrap mask.
    episode_seed: int = 0
    # A solved trajectory discovered by another scientist. Its realized costs
    # are certified upper bounds, not equality labels for this network.
    shared_witness: bool = False
    # Distillation-only exact serial state and the parallel teacher edit. They
    # define a bounded option target; ordinary self-play leaves both unset.
    option_state: Any = None
    target_external_action: int = -1


GameRecord = list[Position]


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.games: list[GameRecord] = []
        self.position_count = 0

    def add(self, game: GameRecord) -> None:
        if not game:
            return
        self.games.append(game)
        self.position_count += len(game)
        while self.games and self.position_count > self.capacity:
            removed = self.games.pop(0)
            self.position_count -= len(removed)

    def sample_positions(self, batch_size: int, balanced: bool = False) -> list[Position]:
        positions = [position for game in self.games for position in game]
        if not positions:
            raise RuntimeError("Cannot sample an empty replay buffer")
        if not balanced:
            indexes = self.rng.integers(0, len(positions), size=batch_size)
            return [positions[int(index)] for index in indexes]
        by_role: dict[int, list[Position]] = {}
        for position in positions:
            by_role.setdefault(position.role, []).append(position)
        roles = sorted(by_role)
        if len(roles) < 2:
            indexes = self.rng.integers(0, len(positions), size=batch_size)
            return [positions[int(index)] for index in indexes]
        share = batch_size // len(roles)
        batch: list[Position] = []
        for index, role in enumerate(roles):
            pool = by_role[role]
            count = share if index else batch_size - share * (len(roles) - 1)
            picks = self.rng.integers(0, len(pool), size=count)
            batch.extend(pool[int(pick)] for pick in picks)
        return batch

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
        terminal_games = [game for game in self.games if game and game[-1].next_terminated]
        terminal_samples = round(batch_size * terminal_fraction)
        for index in range(batch_size):
            use_terminal = index < terminal_samples and terminal_games
            source = terminal_games if use_terminal else self.games
            game = source[int(self.rng.integers(0, len(source)))]
            if use_terminal:
                start = max(0, len(game) - unroll_steps - 1)
            else:
                start = int(self.rng.integers(0, len(game)))
            sequences.append(game[start : start + unroll_steps + 1])
        return sequences
