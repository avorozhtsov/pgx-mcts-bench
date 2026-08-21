from __future__ import annotations

import copy
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
    # v11 cost contract: ``final_moves`` means portable semantic witness steps.
    # Controller/search effort is recorded separately and never enters L_A:B.
    final_moves: float = float("nan")
    final_native_plies: float = float("nan")
    final_internal_plies: float = float("nan")
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
    # Objective coefficient used for this attempt. Keeping it explicit prevents
    # a witness that is best for L10 from being treated as best for L1000.
    objective_ratio: float = float("nan")
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
        # Persistent incumbents survive replay eviction. Shared witnesses never
        # update the native archive used to decide whether policy imitation is
        # still beneficial.
        self.best_native_solution_objectives: dict[tuple[str, float], float] = {}
        self.best_shared_solution_objectives: dict[tuple[str, float], float] = {}
        # Unlike ordinary replay, this bank is not capacity-evicted.  Continual
        # learning can therefore rehearse the best certified native trajectory
        # for an old representation even after its original episodes age out.
        self.best_native_solution_records: dict[tuple[str, float], GameRecord] = {}

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
        if not hasattr(self, "best_native_solution_objectives"):
            self.best_native_solution_objectives = {}
        if not hasattr(self, "best_shared_solution_objectives"):
            self.best_shared_solution_objectives = {}
        if not hasattr(self, "best_native_solution_records"):
            self.best_native_solution_records = {}

    @staticmethod
    def _representation(game: GameRecord) -> str:
        return str(getattr(game[0], "representation_id", "")) if game else ""

    def set_representation_embedding(self, representation_id: str, embedding: np.ndarray) -> None:
        self._ensure_replay_state()
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not representation_id or not vector.size or not np.isfinite(vector).all():
            raise ValueError("representation embedding must be finite and non-empty")
        self.representation_embeddings[representation_id] = vector / max(norm, 1e-12)

    @staticmethod
    def _solution_objective(
        game: GameRecord,
    ) -> tuple[tuple[str, float], float, bool] | None:
        if not game or float(getattr(game[0], "solved", -1.0)) <= 0.5:
            return None
        identity = str(getattr(game[0], "representation_id", ""))
        ratio = float(getattr(game[0], "objective_ratio", float("nan")))
        crossings = float(getattr(game[0], "final_crossing_changes", float("nan")))
        moves = float(getattr(game[0], "final_moves", float("nan")))
        if not identity or not np.isfinite([ratio, crossings, moves]).all():
            return None
        return (
            (identity, ratio),
            ratio * crossings + moves,
            bool(getattr(game[0], "shared_witness", False)),
        )

    def _update_solution_archive(self, game: GameRecord) -> None:
        solution = self._solution_objective(game)
        if solution is None:
            return
        key, objective, shared = solution
        archive = (
            self.best_shared_solution_objectives if shared else self.best_native_solution_objectives
        )
        incumbent = archive.get(key, float("inf"))
        archive[key] = min(objective, incumbent)
        if not shared and objective < incumbent:
            self.best_native_solution_records[key] = copy.deepcopy(game)

    def best_native_objective(self, representation_id: str, ratio: float) -> float | None:
        self._ensure_replay_state()
        return self.best_native_solution_objectives.get((str(representation_id), float(ratio)))

    def record_native_objective(
        self, representation_id: str, ratio: float, objective: float
    ) -> None:
        """Remember a verified native incumbent even when its replay was not retained."""
        self._ensure_replay_state()
        key = (str(representation_id), float(ratio))
        value = float(objective)
        if not key[0] or not np.isfinite([key[1], value]).all():
            raise ValueError("native solution identity, ratio, and objective must be finite")
        self.best_native_solution_objectives[key] = min(
            value, self.best_native_solution_objectives.get(key, float("inf"))
        )

    def best_native_solution_record(
        self, representation_id: str, ratio: float
    ) -> GameRecord | None:
        """Return a fresh copy of the permanent best native solution."""
        self._ensure_replay_state()
        record = self.best_native_solution_records.get((str(representation_id), float(ratio)))
        return copy.deepcopy(record) if record is not None else None

    def active_distillation_records(self) -> list[GameRecord]:
        """Return only the best donations that still beat native incumbents."""
        self._ensure_replay_state()
        active: dict[tuple[str, float], tuple[float, GameRecord]] = {}
        for game in self.games:
            solution = self._solution_objective(game)
            if solution is None:
                continue
            key, shared_objective, shared = solution
            if not shared:
                continue
            native_objective = self.best_native_solution_objectives.get(key)
            if native_objective is not None and shared_objective >= native_objective:
                continue
            incumbent = active.get(key)
            if incumbent is None or shared_objective < incumbent[0]:
                active[key] = (shared_objective, game)
        return [value[1] for _, value in sorted(active.items())]

    def add(
        self,
        game: GameRecord,
        *,
        representation_id: str | None = None,
        objective_ratio: float | None = None,
    ) -> None:
        if not game:
            return
        self._ensure_replay_state()
        identity = representation_id or self._representation(game)
        for index, position in enumerate(game):
            if identity:
                position.representation_id = identity
            if objective_ratio is not None:
                position.objective_ratio = float(objective_ratio)
            if int(getattr(position, "episode_position_index", -1)) < 0:
                position.episode_position_index = index
        self._update_solution_archive(game)
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
            max(1, round(episode_slots * shared_fraction))
            if shared and shared_fraction > 0.0
            else 0,
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
        for stratum, representation_group in zip(strata, representation_groups, strict=True):
            candidates = self._filter_representations(pools[stratum], representation_group)
            fallback = "none"
            if not candidates:
                candidates = pools[stratum]
                fallback = "representation"
            if not candidates:
                candidates = self._filter_representations(eligible_games, representation_group)
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

    def sample_continual_positions(
        self,
        batch_size: int,
        *,
        current_representation: str,
        rehearsal_representations: set[str],
        rehearsal_fraction: float = 0.25,
        positions_per_episode: int = 4,
        max_position_uses: int = 0,
    ) -> list[Position]:
        """Sample exact outcome and new/rehearsal strata when available.

        With an even number of episode slots, native successes and failures get
        equal slots. ``rehearsal_fraction`` is an absolute fraction of the
        complete batch, capped by the available successful half. Thus 0.25
        gives the historical current/rehearsal split and 0.50 reserves every
        positive slot for inherited rehearsal.
        Episodes are uniform by representation and inverse-weighted by prior
        exposure; positions are deliberately spread within an episode.
        """
        if batch_size < 1 or positions_per_episode < 1:
            raise ValueError("batch and positions_per_episode must be positive")
        if not 0.0 <= rehearsal_fraction <= 0.5:
            raise ValueError("rehearsal_fraction must be in [0, 0.5]")
        if max_position_uses < 0:
            raise ValueError("max_position_uses must be non-negative")
        self._ensure_replay_state()

        def eligible(game: GameRecord) -> bool:
            return bool(game) and (
                max_position_uses == 0
                or any(
                    int(getattr(position, "replay_position_uses", 0)) < max_position_uses
                    for position in game
                )
            )

        # Frozen rehearsal witnesses can carry ``shared_witness`` provenance,
        # but they are intentionally part of continual replay.  The explicit
        # representation-id strata below keep them separate from current
        # native outcomes; filtering the flag here would erase the entire
        # rehearsal pool before that stratification can happen.
        available = [game for game in self.games if eligible(game)]
        successes = [
            game
            for game in available
            if not bool(getattr(game[0], "objective_censored", False))
            and float(getattr(game[0], "solved", -1.0)) > 0.5
        ]
        ordinary_failures = [
            game
            for game in available
            if not bool(getattr(game[0], "objective_censored", False))
            and float(getattr(game[0], "solved", -1.0)) <= 0.5
        ]
        capped_failures = [
            game
            for game in available
            if bool(getattr(game[0], "objective_censored", False))
        ]
        failures = ordinary_failures + capped_failures
        if not successes and not failures:
            if self.games and max_position_uses:
                raise RuntimeError("All replay positions reached max_position_uses")
            raise RuntimeError("Cannot sample an empty continual replay")

        slots = max(1, int(np.ceil(batch_size / positions_per_episode)))
        if successes and failures:
            success_slots = slots // 2
            failure_slots = slots - success_slots
        elif successes:
            success_slots, failure_slots = slots, 0
        else:
            success_slots, failure_slots = 0, slots

        current_successes = [
            game for game in successes if self._representation(game) == current_representation
        ]
        rehearsal_successes = [
            game for game in successes if self._representation(game) in rehearsal_representations
        ]
        success_requests: list[tuple[str, list[GameRecord]]] = []
        if current_successes and rehearsal_successes:
            rehearsal_slots = min(
                success_slots,
                max(0, round(slots * rehearsal_fraction)),
            )
            current_slots = success_slots - rehearsal_slots
            success_requests.extend([("current-success", current_successes)] * current_slots)
            success_requests.extend([("rehearsal-success", rehearsal_successes)] * rehearsal_slots)
        else:
            pool = current_successes or rehearsal_successes or successes
            label = (
                "current-success"
                if current_successes
                else "rehearsal-success"
                if rehearsal_successes
                else "global-success"
            )
            success_requests.extend([(label, pool)] * success_slots)

        failure_requests: list[tuple[str, list[GameRecord]]] = []
        if ordinary_failures and capped_failures:
            capped_slots = max(1, min(failure_slots // 3, failure_slots - 1))
            failure_requests.extend(
                [("ordinary-failure", ordinary_failures)] * (failure_slots - capped_slots)
            )
            failure_requests.extend([("budget-censored-failure", capped_failures)] * capped_slots)
        elif failures:
            label = "ordinary-failure" if ordinary_failures else "budget-censored-failure"
            failure_requests.extend([(label, failures)] * failure_slots)

        requests = success_requests + failure_requests
        self.rng.shuffle(requests)
        batch: list[Position] = []
        trace: list[dict[str, Any]] = []
        for label, pool in requests:
            if len(batch) >= batch_size:
                break
            game = self._draw_exposure_balanced_episode(pool)
            eligible_positions = [
                position
                for position in game
                if max_position_uses == 0
                or int(getattr(position, "replay_position_uses", 0)) < max_position_uses
            ]
            positions = self._spread_positions(
                eligible_positions,
                min(positions_per_episode, batch_size - len(batch), len(eligible_positions)),
            )
            batch.extend(positions)
            trace.append(
                {
                    "requested_stratum": label,
                    "actual_representation": self._representation(game),
                    "positions": len(positions),
                }
            )
        self.last_collaboration_sample_trace = trace
        return batch

    def sample_ratio_outcome_balanced_positions(
        self,
        batch_size: int,
        *,
        ratios: tuple[float, float] = (10.0, 1000.0),
        positions_per_episode: int = 4,
        max_position_uses: int = 0,
    ) -> list[Position]:
        """Sample fixed L10/L1000 x positive/negative rehearsal strata.

        Each of the four strata receives the same number of episode slots when
        possible.  A missing stratum falls back to the same ratio, then to the
        same outcome, then to the complete eligible replay.  Every fallback is
        recorded, so a nominal 50:50 mix can never silently become biased.
        """
        if len(ratios) != 2 or any(not np.isfinite(ratio) or ratio <= 0 for ratio in ratios):
            raise ValueError("balanced rehearsal requires two positive finite ratios")
        if batch_size < 1 or positions_per_episode < 1:
            raise ValueError("batch and positions_per_episode must be positive")
        if max_position_uses < 0:
            raise ValueError("max_position_uses must be non-negative")
        self._ensure_replay_state()

        def eligible(game: GameRecord) -> bool:
            return bool(game) and (
                max_position_uses == 0
                or any(
                    int(getattr(position, "replay_position_uses", 0)) < max_position_uses
                    for position in game
                )
            )

        def outcome(game: GameRecord) -> str:
            head = game[0]
            return (
                "positive"
                if not bool(getattr(head, "objective_censored", False))
                and float(getattr(head, "solved", -1.0)) > 0.5
                else "negative"
            )

        available = [
            game
            for game in self.games
            if eligible(game) and not bool(getattr(game[0], "shared_witness", False))
        ]
        if not available:
            if self.games and max_position_uses:
                raise RuntimeError("All replay positions reached max_position_uses")
            raise RuntimeError("Cannot sample an empty balanced rehearsal replay")
        pools: dict[tuple[float, str], list[GameRecord]] = {}
        for ratio in ratios:
            for label in ("positive", "negative"):
                pools[(float(ratio), label)] = [
                    game
                    for game in available
                    if np.isclose(float(getattr(game[0], "objective_ratio", np.nan)), ratio)
                    and outcome(game) == label
                ]

        slots = max(1, int(np.ceil(batch_size / positions_per_episode)))
        strata = [
            (float(ratio), label)
            for ratio in ratios
            for label in ("positive", "negative")
        ]
        requests = [strata[index % len(strata)] for index in range(slots)]
        # The deterministic quota is fixed; shuffling only removes optimizer
        # ordering effects and uses the durable replay RNG.
        self.rng.shuffle(requests)
        batch: list[Position] = []
        trace: list[dict[str, Any]] = []
        for ratio, label in requests:
            if len(batch) >= batch_size:
                break
            candidates = pools[(ratio, label)]
            fallback = "none"
            if not candidates:
                candidates = [
                    game
                    for game in available
                    if np.isclose(float(getattr(game[0], "objective_ratio", np.nan)), ratio)
                ]
                fallback = "outcome-deficit"
            if not candidates:
                candidates = [game for game in available if outcome(game) == label]
                fallback = "ratio-deficit"
            if not candidates:
                candidates = available
                fallback = "ratio-and-outcome-deficit"
            game = self._draw_exposure_balanced_episode(candidates)
            eligible_positions = [
                position
                for position in game
                if max_position_uses == 0
                or int(getattr(position, "replay_position_uses", 0)) < max_position_uses
            ]
            positions = self._spread_positions(
                eligible_positions,
                min(positions_per_episode, batch_size - len(batch), len(eligible_positions)),
            )
            batch.extend(positions)
            trace.append(
                {
                    "requested_ratio": ratio,
                    "requested_outcome": label,
                    "actual_ratio": float(getattr(game[0], "objective_ratio", np.nan)),
                    "actual_outcome": outcome(game),
                    "actual_representation": self._representation(game),
                    "fallback": fallback,
                    "positions": len(positions),
                }
            )
        self.last_collaboration_sample_trace = trace
        return batch

    def _similar_representations(self, current: str, count: int) -> set[str]:
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
        current_count = round(slots * current_fraction) if current_representation else 0
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

    def _draw_exposure_balanced_episode(self, games: list[GameRecord]) -> GameRecord:
        by_representation: dict[str, list[GameRecord]] = {}
        for game in games:
            by_representation.setdefault(self._representation(game), []).append(game)
        identities = sorted(by_representation)
        identity = identities[int(self.rng.integers(0, len(identities)))]
        candidates = by_representation[identity]
        weights = np.asarray(
            [1.0 / (1.0 + int(getattr(game[0], "replay_episode_uses", 0))) for game in candidates],
            dtype=np.float64,
        )
        weights /= weights.sum()
        game = candidates[int(self.rng.choice(len(candidates), p=weights))]
        game[0].replay_episode_uses = int(getattr(game[0], "replay_episode_uses", 0)) + 1
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
            position.replay_position_uses = int(getattr(position, "replay_position_uses", 0)) + 1
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
