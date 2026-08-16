"""Adaptive multi-scientist curriculum over a frozen knot-table pool."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.config import ExperimentConfig
from pgx_mcts_bench.data import ReplayBuffer
from pgx_mcts_bench.game import GameAdapter, Transition, make_game
from pgx_mcts_bench.ladder import Candidate, _config, candidates
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network

# The preregistered K=3 collaboration roster.  Budget-aware training is admitted
# per architecture, but keeping the list here prevents the launcher, loader, and
# admission tools from silently disagreeing about which scientists are repaired.
COLLABORATION_K3 = ("s-window-128", "s-tape4", "s-w11-128")
BUDGET_PROTOTYPES = (
    *COLLABORATION_K3,
    "s-cyclic-tape8-192",
    "strand-graph",
    "raster-axial",
    "cyclic-memory",
    "raster-axial-v2",
    "raster-axial-v3",
    "raster-axial-v4",
    "raster-axial-12",
    "cyclic-memory-12",
    "strand-graph-12",
    "raster-invariant-combined-dual-12",
    "raster-invariant-jones-12",
    "cyclic-memory-deep-v3",
    "cyclic-graph-dual-v3",
)
BUDGET_LEARNING_RATES = {
    "s-window-128": 2.5e-4,
    "s-tape4": 5.0e-5,
    "s-w11-128": 2.5e-4,
    "s-cyclic-tape8-192": 1.0e-4,
    "strand-graph": 1.0e-4,
    "raster-axial": 1.0e-4,
    "cyclic-memory": 1.0e-4,
    "raster-axial-v2": 1.0e-4,
    "raster-axial-v3": 1.0e-4,
    "raster-axial-v4": 1.0e-4,
    "raster-axial-12": 1.0e-4,
    "cyclic-memory-12": 1.0e-4,
    "strand-graph-12": 1.0e-4,
    "raster-invariant-combined-dual-12": 1.0e-4,
    "raster-invariant-jones-12": 1.0e-4,
    "cyclic-memory-deep-v3": 2.5e-4,
    "cyclic-graph-dual-v3": 2.5e-4,
}
BUDGET_AUXILIARY_LEARNING_RATES = {
    "s-window-128": 2.5e-4,
    "s-tape4": 1.0e-3,
    "s-w11-128": 2.5e-4,
    "s-cyclic-tape8-192": 5.0e-4,
    "strand-graph": 5.0e-4,
    "raster-axial": 5.0e-4,
    "cyclic-memory": 5.0e-4,
    "raster-axial-v2": 5.0e-4,
    "raster-axial-v3": 5.0e-4,
    "raster-axial-v4": 5.0e-4,
    "raster-axial-12": 5.0e-4,
    "cyclic-memory-12": 5.0e-4,
    "strand-graph-12": 5.0e-4,
    "raster-invariant-combined-dual-12": 5.0e-4,
    "raster-invariant-jones-12": 5.0e-4,
    "cyclic-memory-deep-v3": 5.0e-4,
    "cyclic-graph-dual-v3": 5.0e-4,
}
BUDGET_PRESERVATION_WEIGHTS = {
    "s-window-128": 1.0,
    "s-tape4": 20.0,
    "s-w11-128": 5.0,
    "s-cyclic-tape8-192": 5.0,
    "strand-graph": 5.0,
    "raster-axial": 5.0,
    "cyclic-memory": 5.0,
    "raster-axial-v2": 5.0,
    "raster-axial-v3": 5.0,
    "raster-axial-v4": 5.0,
    "raster-axial-12": 5.0,
    "cyclic-memory-12": 5.0,
    "strand-graph-12": 5.0,
    "raster-invariant-combined-dual-12": 5.0,
    "raster-invariant-jones-12": 5.0,
    "cyclic-memory-deep-v3": 5.0,
    "cyclic-graph-dual-v3": 5.0,
}
BUDGET_MONOTONIC_WEIGHTS = {
    "s-window-128": 0.25,
    "s-tape4": 1.0,
    "s-w11-128": 1.0,
    "s-cyclic-tape8-192": 0.25,
    "strand-graph": 1.0,
    "raster-axial": 1.0,
    "cyclic-memory": 1.0,
    "raster-axial-v2": 1.0,
    "raster-axial-v3": 1.0,
    "raster-axial-v4": 1.0,
    "raster-axial-12": 1.0,
    "cyclic-memory-12": 1.0,
    "strand-graph-12": 1.0,
    "raster-invariant-combined-dual-12": 1.0,
    "raster-invariant-jones-12": 1.0,
    "cyclic-memory-deep-v3": 1.0,
    "cyclic-graph-dual-v3": 1.0,
}


@dataclass(frozen=True)
class KnotItem:
    name: str
    crossings: int
    word: tuple[int, ...]
    strands: int


@dataclass
class Scientist:
    name: str
    candidate: Candidate
    checkpoint: Path
    config: ExperimentConfig
    game: GameAdapter
    network: Any
    optimizer: torch.optim.Optimizer
    replay: ReplayBuffer
    prediction_source: str
    solve_calibration_scale: float = 1.0
    solve_calibration_bias: float = 0.0
    last_accepted_round: int = -1
    ignored_rounds: int = 0


def calibrated_solve_probability(scientist: Scientist, solve_logits: torch.Tensor) -> torch.Tensor:
    """Apply the checkpoint's monotone held-out probability calibration."""
    raw = solve_logits.sigmoid().mean(dim=1).clamp(1e-7, 1.0 - 1e-7)
    return torch.sigmoid(
        scientist.solve_calibration_scale * torch.logit(raw)
        + scientist.solve_calibration_bias
    )


@dataclass(frozen=True)
class ObjectiveBudgetState:
    base_state: Any
    objective_cap: float
    cap_type: str
    exhausted: bool = False


class FixedWordGame:
    """Present one table knot at every reset while preserving the game API."""

    def __init__(
        self,
        base: GameAdapter,
        knot: KnotItem,
        ratio: float = 1000.0,
        *,
        objective_cap: float | None = None,
        cap_type: str = "global",
    ):
        self.base = base
        self.config = base.config
        self.knot = knot
        self.ratio = ratio
        self.objective_cap = objective_cap
        self.cap_type = cap_type
        if objective_cap is not None and not self.config.objective_budget_channel:
            raise ValueError("objective_cap requires objective_budget_channel")

    def _global_cap(self) -> float:
        return (self.ratio + 1.0) * self.config.simplify_budget

    def _raw(self, state: Any) -> Any:
        if isinstance(state, ObjectiveBudgetState):
            state = state.base_state
        return self.base.unwrap(state)

    def _spent(self, state: Any) -> float:
        raw = self._raw(state)
        return (
            self.ratio * int(np.asarray(raw._crossing_changes))
            + self.semantic_move_count(state)
        )

    def _solved(self, state: Any) -> bool:
        raw = self._raw(state)
        return int(np.asarray(raw._n)) == 1

    def _budgeted(self, transition: Transition, cap: float) -> Transition:
        solved = self._solved(transition.state)
        remaining = cap - self._spent(transition.state)
        # A checkpoint may consume the remaining-L feature even when the
        # experiment deliberately disables objective-capped termination.  In
        # that case ``objective_cap`` is None: expose the soft global remainder
        # but let the ordinary native-action clock decide failure.
        exhausted = (
            self.objective_cap is not None and remaining <= 0.0 and not solved
        )
        terminated = transition.terminated or exhausted
        observation = transition.observation.copy()
        # Encode absolute remaining objective against one fixed scale. Dividing
        # by the attempt's own cap made every fresh attempt equal to 1.0, so the
        # network could not distinguish a cap of 20 from a cap of 200 despite
        # receiving different terminal labels for them.
        observation[..., -1] = np.clip(remaining / max(self._global_cap(), 1.0), -1.0, 1.0)
        state = ObjectiveBudgetState(
            transition.state,
            objective_cap=cap,
            cap_type=self.cap_type,
            exhausted=exhausted,
        )
        reason = (
            "solved"
            if solved
            else "objective_budget_exhausted"
            if exhausted
            else transition.termination_reason
        )
        return replace(
            transition,
            state=state,
            observation=observation,
            reward=-1.0 if exhausted else transition.reward,
            terminated=terminated,
            termination_reason=reason,
        )

    def reset(self, seed: int) -> Transition:
        del seed
        transition = self.base.from_word(  # type: ignore[attr-defined]
            list(self.knot.word), self.knot.strands, log_ratio=math.log(self.ratio)
        )
        if not self.config.objective_budget_channel:
            return transition
        return self._budgeted(
            transition,
            self.objective_cap if self.objective_cap is not None else self._global_cap(),
        )

    def step(self, state: Any, action: int) -> Transition:
        if not isinstance(state, ObjectiveBudgetState):
            return self.base.step(state, action)
        if state.exhausted:
            raise ValueError("cannot step an exhausted objective-budget state")
        transition = self.base.step(state.base_state, action)
        return self._budgeted(transition, state.objective_cap)

    def final_rewards(self, state: Any) -> np.ndarray:
        if isinstance(state, ObjectiveBudgetState):
            if state.exhausted:
                raw = self._raw(state)
                simplifier = 1 - int(np.asarray(raw._scrambler))
                rewards = np.zeros(2, dtype=np.float32)
                rewards[simplifier] = -1.0
                rewards[1 - simplifier] = 1.0
                return rewards
            state = state.base_state
        return self.base.final_rewards(state)

    def value_potential(self, state: Any, player: int) -> float:
        if isinstance(state, ObjectiveBudgetState):
            state = state.base_state
        return self.base.value_potential(state, player)

    def state_info(self, state: Any) -> dict[str, int]:
        if isinstance(state, ObjectiveBudgetState):
            state = state.base_state
        return self.base.state_info(state)

    def first_role_player(self, state: Any) -> int:
        if isinstance(state, ObjectiveBudgetState):
            state = state.base_state
        return self.base.first_role_player(state)

    def unwrap(self, state: Any) -> Any:
        if isinstance(state, ObjectiveBudgetState):
            state = state.base_state
        return self.base.unwrap(state)

    def semantic_move_count(self, state: Any) -> int:
        if isinstance(state, ObjectiveBudgetState):
            state = state.base_state
        return self.base.semantic_move_count(state)

    def native_ply_count(self, state: Any) -> int:
        if isinstance(state, ObjectiveBudgetState):
            state = state.base_state
        return self.base.native_ply_count(state)

    def internal_ply_count(self, state: Any) -> int:
        if isinstance(state, ObjectiveBudgetState):
            state = state.base_state
        return self.base.internal_ply_count(state)


def smallest_crossing_pool(size: int = 200) -> list[KnotItem]:
    """Frozen table slice compatible with the bounded serial scientist envelope."""
    from rf_knots.knot_table import load_table

    items = []
    for name, row in load_table()["knots"].items():
        crossings = int(row["crossings"])
        word = tuple(int(letter) for letter in row["braid"])
        strands = int(row["strands"])
        if crossings <= 0 or strands > 5 or len(word) > 48:
            continue
        items.append(KnotItem(name, crossings, word, strands))
    items.sort(key=lambda item: (item.crossings, item.name))
    if len(items) < size:
        raise ValueError(f"only {len(items)} compatible nontrivial knots; requested {size}")
    return items[:size]


def simplicity(probability: np.ndarray, predicted_crossings: np.ndarray) -> np.ndarray:
    """The proposed raw curriculum score p(solve) * (20 - predicted cc)."""
    return probability * (20.0 - predicted_crossings)


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        out = np.zeros_like(values, dtype=np.float64)
        out[int(np.argmax(values))] = 1.0
        return out
    shifted = (np.asarray(values, dtype=np.float64) - np.max(values)) / temperature
    weights = np.exp(np.clip(shifted, -700.0, 0.0))
    return weights / weights.sum()


def choose_proposals(
    score_rows: list[np.ndarray], rng: np.random.Generator, temperature: float
) -> list[int]:
    return [int(rng.choice(len(row), p=softmax(row, temperature))) for row in score_rows]


def choose_group_proposal(
    score_rows: list[np.ndarray],
    proposals: list[int],
    ignored_rounds: list[int],
    rng: np.random.Generator,
    *,
    alpha: float,
    temperature: float,
    starvation_rounds: int,
) -> tuple[int, int, str]:
    """Choose (scientist, knot index, reason), with a hard fairness guarantee."""
    starved = [i for i, age in enumerate(ignored_rounds) if age >= starvation_rounds]
    if starved:
        scientist = max(starved, key=lambda i: (ignored_rounds[i], -i))
        return scientist, int(np.argmax(score_rows[scientist])), "starvation"
    priorities = np.asarray(
        [
            alpha * score_rows[i][proposal] + ignored_rounds[i]
            for i, proposal in enumerate(proposals)
        ]
    )
    scientist = int(rng.choice(len(proposals), p=softmax(priorities, temperature)))
    return scientist, proposals[scientist], "softmax-priority"


def _observation_tensor(observations: list[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack(observations)).permute(0, 3, 1, 2).float().to(device)


@torch.inference_mode()
def score_pool(scientist: Scientist, pool: list[KnotItem]) -> tuple[np.ndarray, np.ndarray]:
    observations = [FixedWordGame(scientist.game, knot).reset(0).observation for knot in pool]
    tensor = _observation_tensor(observations, torch.device(scientist.config.train.device))
    scientist.network.eval()
    _, legacy, auxiliary = scientist.network.forward_with_auxiliary(tensor)
    if scientist.prediction_source.startswith("factorized"):
        assert auxiliary is not None
        solve_logits, crossings, _ = auxiliary
        return calibrated_solve_probability(scientist, solve_logits).cpu().numpy(), (
            crossings.mean(dim=1).cpu().numpy()
        )
    probability = ((legacy + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy()
    return probability, 20.0 * (1.0 - probability)


def _candidate(name: str) -> Candidate:
    by_name = {candidate.name: candidate for candidate in candidates()}
    if name not in by_name:
        raise ValueError(f"{name!r} is not a ladder candidate")
    return by_name[name]


def load_scientist(
    name: str,
    checkpoint: Path,
    *,
    seed: int,
    device: str,
    simulations: int = 0,
    require_factorized: bool = False,
    objective_budget_channel: bool = False,
) -> Scientist:
    candidate = _candidate(name)
    if simulations:
        candidate = Candidate(**{**asdict(candidate), "simulations": simulations})
    config = _config(candidate, ("R(3,12)#0", 0), seed, device, selfplay_games=1)
    if objective_budget_channel:
        budget_prototype = name in BUDGET_PROTOTYPES
        config = replace(
            config,
            game=replace(config.game, objective_budget_channel=True),
            train=replace(
                config.train,
                learning_rate=(
                    BUDGET_LEARNING_RATES[name]
                    if budget_prototype
                    else config.train.learning_rate
                ),
            ),
            model=replace(
                config.model,
                auxiliary_solve_backprop_to_encoder=budget_prototype,
                auxiliary_budget_monotonic_weight=(
                    BUDGET_MONOTONIC_WEIGHTS[name] if budget_prototype else 0.0
                ),
                auxiliary_budget_conditioning=budget_prototype,
                freeze_batchnorm_stats=budget_prototype,
                policy_value_preservation_weight=(
                    BUDGET_PRESERVATION_WEIGHTS[name] if budget_prototype else 0.0
                ),
            ),
        )
    game = make_game(config.game)
    network = make_braid_network(config.game, config.model).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("network", payload)
    factorized = any(
        key.startswith("auxiliary.")
        or key.startswith("window.auxiliary.")
        or ".auxiliary." in key
        for key in state
    )
    if require_factorized and not factorized:
        raise ValueError(f"{checkpoint} has no trained factorized value heads")
    load_policy_value_state_dict(network, state)
    solve_calibration = payload.get("solve_calibration", {})
    if objective_budget_channel and budget_prototype:
        auxiliary_prefixes = (
            "auxiliary.",
            "window.auxiliary.",
            "scan.auxiliary.",
            "tape.auxiliary.",
            "solve_residual.",
            "cost_residual.",
        )
        auxiliary_parameters = [
            parameter
            for parameter_name, parameter in network.named_parameters()
            if parameter_name.startswith(auxiliary_prefixes)
        ]
        if not auxiliary_parameters:
            raise ValueError(f"{name} exposes no trainable factorized value parameters")
        auxiliary_ids = {id(parameter) for parameter in auxiliary_parameters}
        controller_parameters = [
            parameter for parameter in network.parameters() if id(parameter) not in auxiliary_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": controller_parameters, "lr": config.train.learning_rate},
                {
                    "params": auxiliary_parameters,
                    "lr": BUDGET_AUXILIARY_LEARNING_RATES[name],
                },
            ],
            weight_decay=config.train.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
    return Scientist(
        name=name,
        candidate=candidate,
        checkpoint=checkpoint,
        config=config,
        game=game,
        network=network,
        optimizer=optimizer,
        replay=ReplayBuffer(
            config.train.replay_capacity,
            np.random.default_rng(seed + 17),
            representation_capacity=100,
        ),
        prediction_source="factorized" if factorized else "legacy_proxy",
        solve_calibration_scale=float(solve_calibration.get("scale", 1.0)),
        solve_calibration_bias=float(solve_calibration.get("bias", 0.0)),
    )
