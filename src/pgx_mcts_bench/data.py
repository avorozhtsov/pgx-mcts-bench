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
    # Replay provenance. These defaults keep old checkpoints loadable; new
    # collaboration records fill them at generation or admission time.
    representation_id: str = ""
    termination_reason: str = ""
    objective_cap: float = float("nan")
    action_horizon: int = 0
    residual_word_length: int = -1
    best_residual_word_length: int = -1
    mcts_root_value: float = float("nan")
    mcts_visit_count: int = 0
    episode_position_index: int = -1
    # Exposure counters are deliberately stored with the replay payload so a
    # resumed run continues the same anti-oversampling schedule.
    replay_episode_uses: int = 0
    replay_position_uses: int = 0


GameRecord = list[Position]


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        rng: np.random.Generator,
        *,
        representation_capacity: int = 0,
    ):
        self.capacity = capacity
        self.rng = rng
        self.representation_capacity = representation_capacity
        self.games: list[GameRecord] = []
        self.position_count = 0
        self.representation_order: list[str] = []
        self.representation_embeddings: dict[str, np.ndarray] = {}
        self.last_collaboration_sample_trace: list[dict[str, Any]] = []

    def _ensure_replay_state(self) -> None:
        """Populate fields missing from pre-replay-v2 pickles."""
        if not hasattr(self, "representation_capacity"):
            self.representation_capacity = 0
        if not hasattr(self, "representation_order"):
            self.representation_order = []
        if not hasattr(self, "representation_embeddings"):
            self.representation_embeddings = {}
        if not hasattr(self, "last_collaboration_sample_trace"):
            self.last_collaboration_sample_trace = []

    @staticmethod
    def _representation(game: GameRecord) -> str:
        return str(getattr(game[0], "representation_id", "")) if game else ""

    def set_representation_embedding(
        self, representation_id: str, embedding: np.ndarray
    ) -> None:
        self._ensure_replay_state()
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not representation_id or not vector.size or not np.isfinite(vector).all():
            raise ValueError("representation embedding must be finite and non-empty")
        self.representation_embeddings[representation_id] = vector / max(norm, 1e-12)

    def add(self, game: GameRecord, *, representation_id: str | None = None) -> None:
        if not game:
            return
        self._ensure_replay_state()
        identity = representation_id or self._representation(game)
        for index, position in enumerate(game):
            if identity:
                position.representation_id = identity
            if int(getattr(position, "episode_position_index", -1)) < 0:
                position.episode_position_index = index
        self.games.append(game)
        self.position_count += len(game)
        if identity:
            if identity in self.representation_order:
                self.representation_order.remove(identity)
            self.representation_order.append(identity)
        while self.games and self.position_count > self.capacity:
            self._remove_game(0)
        self._evict_old_representations()

    def _remove_game(self, index: int) -> None:
        removed = self.games.pop(index)
        self.position_count -= len(removed)

    def _evict_old_representations(self) -> None:
        if self.representation_capacity <= 0:
            return
        present = {self._representation(game) for game in self.games}
        self.representation_order = [
            identity for identity in self.representation_order if identity in present
        ]
        while len(self.representation_order) > self.representation_capacity:
            oldest = self.representation_order.pop(0)
            for index in range(len(self.games) - 1, -1, -1):
                if self._representation(self.games[index]) == oldest:
                    self._remove_game(index)
            self.representation_embeddings.pop(oldest, None)

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
        self,
        batch_size: int,
        shared_fraction: float = 0.1,
        *,
        current_representation: str = "",
        current_fraction: float = 0.0,
        similar_fraction: float = 0.0,
        similar_representation_count: int = 8,
        positions_per_episode: int = 1,
        max_position_uses: int = 0,
    ) -> list[Position]:
        """Exposure-aware replay balanced by outcome and representation.

        Objective-censored attempts are negative examples for the conditional
        question "can this state be solved within the encoded remaining budget?"
        They share the native-failure stratum but remain masked from policy,
        scalar-value, and conditional-cost targets. A selected episode supplies
        several deliberately spread positions, while inverse-exposure sampling
        prevents a short or frequently reused attempt from dominating.
        """
        if batch_size < 1 or positions_per_episode < 1:
            raise ValueError("batch and positions_per_episode must be positive")
        if max_position_uses < 0:
            raise ValueError("max_position_uses must be non-negative")
        if current_fraction < 0.0 or similar_fraction < 0.0:
            raise ValueError("representation fractions must be non-negative")
        if current_fraction + similar_fraction > 1.0:
            raise ValueError("current and similar fractions must sum to at most one")
        self._ensure_replay_state()
        def eligible_game(game: GameRecord) -> bool:
            return bool(game) and (
                max_position_uses == 0
                or any(
                    int(getattr(position, "replay_position_uses", 0)) < max_position_uses
                    for position in game
                )
            )

        native_success = [
            game
            for game in self.games
            if eligible_game(game)
            and not bool(getattr(game[0], "shared_witness", False))
            and not bool(getattr(game[0], "objective_censored", False))
            and float(getattr(game[0], "solved", -1.0)) > 0.5
        ]
        native_failure = [
            game
            for game in self.games
            if eligible_game(game)
            and not bool(getattr(game[0], "shared_witness", False))
            and not bool(getattr(game[0], "objective_censored", False))
            and float(getattr(game[0], "solved", -1.0)) <= 0.5
        ]
        capped_failure = [
            game
            for game in self.games
            if eligible_game(game)
            and not bool(getattr(game[0], "shared_witness", False))
            and bool(getattr(game[0], "objective_censored", False))
        ]
        shared = [
            game
            for game in self.games
            if eligible_game(game)
            and bool(getattr(game[0], "shared_witness", False))
            and not bool(getattr(game[0], "objective_censored", False))
        ]
        if not native_success and not native_failure and not capped_failure and not shared:
            if self.games and max_position_uses:
                raise RuntimeError("All replay positions reached max_position_uses")
            raise RuntimeError("Cannot sample an empty collaboration replay")
        eligible_games = [game for game in self.games if eligible_game(game)]

        episode_slots = max(1, int(np.ceil(batch_size / positions_per_episode)))
        shared_count = min(
            episode_slots,
            max(0, round(episode_slots * shared_fraction)) if shared else 0,
        )
        native_count = episode_slots - shared_count
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

        strata = (
            ["success"] * success_count
            + ["failure"] * failure_count
            + ["capped"] * capped_count
            + ["shared"] * shared_count
        )
        self.rng.shuffle(strata)
        pools = {
            "success": native_success,
            "failure": native_failure,
            "capped": capped_failure,
            "shared": shared,
        }
        representation_groups = self._representation_slot_groups(
            episode_slots,
            current_representation=current_representation,
            current_fraction=current_fraction,
            similar_fraction=similar_fraction,
            similar_representation_count=similar_representation_count,
        )
        selected: list[GameRecord] = []
        trace: list[dict[str, Any]] = []
        for stratum, representation_group in zip(
            strata, representation_groups, strict=True
        ):
            candidates = self._filter_representations(
                pools[stratum], representation_group
            )
            fallback = "none"
            if not candidates:
                candidates = pools[stratum]
                fallback = "representation"
            if not candidates:
                candidates = self._filter_representations(
                    eligible_games, representation_group
                )
                fallback = "stratum"
            if not candidates:
                candidates = eligible_games
                fallback = "representation-and-stratum"
            game = self._draw_exposure_balanced_episode(candidates)
            selected.append(game)
            trace.append(
                {
                    "requested_representation_group": representation_group[0],
                    "requested_representations": sorted(representation_group[1]),
                    "requested_stratum": stratum,
                    "actual_representation": self._representation(game),
                    "fallback": fallback,
                    "positions": 0,
                }
            )

        batch: list[Position] = []
        for game, episode_trace in zip(selected, trace, strict=True):
            remaining = batch_size - len(batch)
            if remaining <= 0:
                break
            eligible_positions = [
                position
                for position in game
                if max_position_uses == 0
                or int(getattr(position, "replay_position_uses", 0)) < max_position_uses
            ]
            positions = self._spread_positions(
                eligible_positions,
                min(positions_per_episode, remaining, len(eligible_positions)),
            )
            batch.extend(positions)
            episode_trace["positions"] = len(positions)
        self.last_collaboration_sample_trace = trace
        return batch

    def _similar_representations(
        self, current: str, count: int
    ) -> set[str]:
        current_embedding = self.representation_embeddings.get(current)
        if current_embedding is None or count <= 0:
            return set()
        scored = []
        present = {self._representation(game) for game in self.games}
        for identity, embedding in self.representation_embeddings.items():
            if identity == current or identity not in present:
                continue
            if embedding.shape != current_embedding.shape:
                continue
            scored.append((float(np.dot(current_embedding, embedding)), identity))
        scored.sort(reverse=True)
        return {identity for _, identity in scored[:count]}

    def _representation_slot_groups(
        self,
        slots: int,
        *,
        current_representation: str,
        current_fraction: float,
        similar_fraction: float,
        similar_representation_count: int,
    ) -> list[tuple[str, set[str]]]:
        similar = self._similar_representations(
            current_representation, similar_representation_count
        )
        current_count = (
            round(slots * current_fraction) if current_representation else 0
        )
        similar_count = round(slots * similar_fraction) if similar else 0
        if current_count + similar_count > slots:
            similar_count = max(0, slots - current_count)
        groups = (
            [("current", {current_representation})] * current_count
            + [("similar", similar)] * similar_count
            + [("global", set())] * (slots - current_count - similar_count)
        )
        self.rng.shuffle(groups)
        return groups

    def _filter_representations(
        self, games: list[GameRecord], group: tuple[str, set[str]]
    ) -> list[GameRecord]:
        label, identities = group
        if label == "global" or not identities:
            return games
        return [game for game in games if self._representation(game) in identities]

    def _draw_exposure_balanced_episode(
        self, games: list[GameRecord]
    ) -> GameRecord:
        by_representation: dict[str, list[GameRecord]] = {}
        for game in games:
            by_representation.setdefault(self._representation(game), []).append(game)
        identities = sorted(by_representation)
        identity = identities[int(self.rng.integers(0, len(identities)))]
        candidates = by_representation[identity]
        weights = np.asarray(
            [
                1.0
                / (1.0 + int(getattr(game[0], "replay_episode_uses", 0)))
                for game in candidates
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()
        game = candidates[int(self.rng.choice(len(candidates), p=weights))]
        game[0].replay_episode_uses = int(
            getattr(game[0], "replay_episode_uses", 0)
        ) + 1
        return game

    def _spread_positions(self, game: GameRecord, count: int) -> list[Position]:
        if count <= 0:
            return []
        indexes: list[int] = [0]
        if count > 1 and len(game) > 1:
            indexes.append(len(game) - 1)
        if count > 2 and len(game) > 2:
            entropies = []
            for index, position in enumerate(game[1:-1], start=1):
                probability = np.asarray(position.policy, dtype=np.float64)
                positive = probability[probability > 0.0]
                entropy = float(-(positive * np.log(positive)).sum())
                entropies.append((entropy, index))
            if entropies:
                indexes.append(max(entropies)[1])
        while len(indexes) < count:
            weights = np.asarray(
                [
                    1.0 / (1.0 + int(getattr(position, "replay_position_uses", 0)))
                    for position in game
                ],
                dtype=np.float64,
            )
            if len(set(indexes)) < len(game):
                weights[indexes] = 0.0
            if not float(weights.sum()):
                weights.fill(1.0)
            weights /= weights.sum()
            indexes.append(int(self.rng.choice(len(game), p=weights)))
        result = [game[index] for index in indexes[:count]]
        for position in result:
            position.replay_position_uses = int(
                getattr(position, "replay_position_uses", 0)
            ) + 1
        return result

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
