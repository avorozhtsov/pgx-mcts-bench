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
    # Player-perspective potential at this position. With potential shaping the
    # value target is ``terminal_outcome - value_potential``. Keeping the field
    # on every record also permits paired target-only ablations on identical
    # replay data.
    value_potential: float = 0.0
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
    # Search stopped at a learned objective cap. This is a censored attempt, not
    # evidence that the state is unsolvable under the environment's full budget.
    objective_censored: bool = False
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

    def sample_collaboration_positions(
        self, batch_size: int, shared_fraction: float = 0.1
    ) -> list[Position]:
        """Success-balanced native replay with a hard cap on shared positions.

        Objective-censored attempts are negative examples for the conditional
        question "can this state be solved within the encoded remaining budget?"
        They share the native-failure stratum but remain masked from policy,
        scalar-value, and conditional-cost targets. Sampling is episode-uniform
        within each stratum before choosing a position, preventing long
        trajectories from dominating.
        """
        native_success = [
            game
            for game in self.games
            if game
            and not bool(getattr(game[0], "shared_witness", False))
            and not bool(getattr(game[0], "objective_censored", False))
            and float(getattr(game[0], "solved", -1.0)) > 0.5
        ]
        native_failure = [
            game
            for game in self.games
            if game
            and not bool(getattr(game[0], "shared_witness", False))
            and not bool(getattr(game[0], "objective_censored", False))
            and float(getattr(game[0], "solved", -1.0)) <= 0.5
        ]
        capped_failure = [
            game
            for game in self.games
            if game
            and not bool(getattr(game[0], "shared_witness", False))
            and bool(getattr(game[0], "objective_censored", False))
        ]
        shared = [
            game
            for game in self.games
            if game
            and bool(getattr(game[0], "shared_witness", False))
            and not bool(getattr(game[0], "objective_censored", False))
        ]
        if not native_success and not native_failure and not capped_failure and not shared:
            raise RuntimeError("Cannot sample an empty collaboration replay")

        shared_count = min(
            batch_size,
            max(0, round(batch_size * shared_fraction)) if shared else 0,
        )
        native_count = batch_size - shared_count
        has_negative = bool(native_failure or capped_failure)
        success_count = native_count // 2 if native_success and has_negative else native_count
        negative_count = native_count - success_count
        if not native_success:
            negative_count, success_count = native_count, 0
        if not has_negative:
            success_count, negative_count = native_count, 0

        capped_count = negative_count // 2 if native_failure and capped_failure else negative_count
        failure_count = negative_count - capped_count
        if not native_failure:
            capped_count, failure_count = negative_count, 0
        if not capped_failure:
            failure_count, capped_count = negative_count, 0

        def draw(games: list[GameRecord], count: int) -> list[Position]:
            result = []
            for _ in range(count):
                game = games[int(self.rng.integers(0, len(games)))]
                result.append(game[int(self.rng.integers(0, len(game)))])
            return result

        def draw_capped(count: int) -> list[Position]:
            # The remaining-budget feature is the final observation channel.
            # Choose its initial-value bucket first so a frequent cap does not
            # erase rarer low/high-cap failures from the conditional critic.
            buckets: dict[float, list[GameRecord]] = {}
            for game in capped_failure:
                initial_budget = float(np.asarray(game[0].observation)[..., -1].mean())
                buckets.setdefault(round(initial_budget, 2), []).append(game)
            keys = sorted(buckets)
            result = []
            for _ in range(count):
                key = keys[int(self.rng.integers(0, len(keys)))]
                game = buckets[key][int(self.rng.integers(0, len(buckets[key])))]
                result.append(game[int(self.rng.integers(0, len(game)))])
            return result

        return (
            draw(native_success, success_count)
            + draw(native_failure, failure_count)
            + draw_capped(capped_count)
            + draw(shared, shared_count)
        )

    def has_trainable_collaboration_positions(self) -> bool:
        return any(bool(game) for game in self.games)

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
