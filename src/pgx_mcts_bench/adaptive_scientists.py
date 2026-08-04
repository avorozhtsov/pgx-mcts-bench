"""Adaptive multi-scientist curriculum over a frozen knot-table pool."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pgx_mcts_bench.config import ExperimentConfig
from pgx_mcts_bench.data import GameRecord, ReplayBuffer
from pgx_mcts_bench.game import GameAdapter, Transition, make_game
from pgx_mcts_bench.ladder import Candidate, _config, candidates
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.training import play_selfplay_games, train_alphazero_step

# The preregistered K=3 collaboration roster.  Budget-aware training is admitted
# per architecture, but keeping the list here prevents the launcher, loader, and
# admission tools from silently disagreeing about which scientists are repaired.
COLLABORATION_K3 = ("s-window-128", "d-tape4-u1", "s-w11-128")
BUDGET_LEARNING_RATES = {
    "s-window-128": 2.5e-4,
    # The distilled tape controller is substantially more sensitive to a narrow
    # easy-task replay distribution than the two window controllers.
    "d-tape4-u1": 5.0e-5,
    "s-w11-128": 2.5e-4,
}
BUDGET_AUXILIARY_LEARNING_RATES = {
    "s-window-128": 2.5e-4,
    "d-tape4-u1": 1.0e-3,
    "s-w11-128": 2.5e-4,
}
BUDGET_PRESERVATION_WEIGHTS = {
    "s-window-128": 1.0,
    "d-tape4-u1": 20.0,
    "s-w11-128": 5.0,
}
BUDGET_MONOTONIC_WEIGHTS = {
    "s-window-128": 0.25,
    "d-tape4-u1": 1.0,
    "s-w11-128": 1.0,
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
    last_accepted_round: int = -1
    ignored_rounds: int = 0


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
        moves = max(self.config.simplify_budget - int(np.asarray(raw._budget)), 0)
        return self.ratio * int(np.asarray(raw._crossing_changes)) + moves

    def _solved(self, state: Any) -> bool:
        raw = self._raw(state)
        return int(np.asarray(raw._n)) == 1

    def _budgeted(self, transition: Transition, cap: float) -> Transition:
        solved = self._solved(transition.state)
        remaining = cap - self._spent(transition.state)
        exhausted = remaining <= 0.0 and not solved
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


def smallest_crossing_pool(size: int = 200) -> list[KnotItem]:
    """Frozen table slice compatible with the parallel rung-23 networks."""
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
        return (
            solve_logits.sigmoid().mean(dim=1).cpu().numpy(),
            crossings.mean(dim=1).cpu().numpy(),
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
        budget_prototype = name in COLLABORATION_K3
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
        key.startswith("auxiliary.") or key.startswith("window.auxiliary.") for key in state
    )
    if require_factorized and not factorized:
        raise ValueError(f"{checkpoint} has no trained factorized value heads")
    load_policy_value_state_dict(network, state)
    if objective_budget_channel and budget_prototype:
        auxiliary_parameters = list(network.auxiliary.parameters())
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
        replay=ReplayBuffer(config.train.replay_capacity, np.random.default_rng(seed + 17)),
        prediction_source="factorized" if factorized else "legacy_proxy",
    )


def _verified_witness(knot: KnotItem, record: GameRecord, config: ExperimentConfig):
    from rf_knots.evidence import UnknotWitness

    if not record or record[0].solved < 0.5:
        return None
    witness = UnknotWitness.from_actions(
        knot.word, knot.strands, config.game._spec, (position.action for position in record)
    )
    witness.verify()
    return witness


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_adaptive_scientists(
    checkpoints: dict[str, Path],
    output: Path,
    *,
    rounds: int = 20,
    pool_size: int = 200,
    alpha: float = 1.0,
    proposal_temperature: float = 1.0,
    group_temperature: float = 1.0,
    starvation_rounds: int = 0,
    selfplay_games: int = 2,
    train_steps: int = 16,
    batch_size: int = 32,
    simulations: int = 0,
    seed: int = 0,
    device: str = "cpu",
    require_factorized: bool = False,
) -> dict[str, Any]:
    if not checkpoints:
        raise ValueError("at least one scientist checkpoint is required")
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / "schedule.jsonl"
    if schedule_path.exists():
        raise FileExistsError(f"{schedule_path} already exists; choose a fresh output directory")
    pool = smallest_crossing_pool(pool_size)
    scientists = [
        load_scientist(
            name,
            path,
            seed=seed + i * 10_000,
            device=device,
            simulations=simulations,
            require_factorized=require_factorized,
        )
        for i, (name, path) in enumerate(checkpoints.items())
    ]
    starvation = starvation_rounds or 2 * len(scientists)
    rng = np.random.default_rng(seed)
    remaining = list(pool)
    pool_payload = [asdict(knot) for knot in pool]
    pool_json = json.dumps(pool_payload, sort_keys=True)
    (output / "pool.json").write_text(pool_json + "\n", encoding="utf-8")
    selected: list[str] = []
    for round_index in range(min(rounds, len(remaining))):
        remaining_names = [knot.name for knot in remaining]
        round_sources = [scientist.prediction_source for scientist in scientists]
        predictions = []
        score_rows = []
        for scientist in scientists:
            probability, crossings = score_pool(scientist, remaining)
            scores = simplicity(probability, crossings)
            predictions.append((probability, crossings))
            score_rows.append(scores)
        proposals = choose_proposals(score_rows, rng, proposal_temperature)
        proposal_names = [remaining[index].name for index in proposals]
        selected_scientist, selected_index, reason = choose_group_proposal(
            score_rows,
            proposals,
            [s.ignored_rounds for s in scientists],
            rng,
            alpha=alpha,
            temperature=group_temperature,
            starvation_rounds=starvation,
        )
        knot = remaining.pop(selected_index)
        matching = {i for i, proposal in enumerate(proposals) if proposal == selected_index}
        matching.add(selected_scientist)
        for i, scientist in enumerate(scientists):
            if i in matching:
                scientist.last_accepted_round = round_index
                scientist.ignored_rounds = 0
            else:
                scientist.ignored_rounds += 1

        solved_records: list[tuple[int, GameRecord, Any]] = []
        losses: dict[str, list[dict[str, float]]] = {}
        for scientist_index, scientist in enumerate(scientists):
            fixed = FixedWordGame(scientist.game, knot)
            search = NeuralMCTS(fixed, scientist.network, scientist.config.search, device)
            game_seeds = [
                seed + round_index * 100_000 + scientist_index * 1_000 + i
                for i in range(selfplay_games)
            ]
            records = play_selfplay_games(
                fixed,
                search,
                [np.random.default_rng(game_seed) for game_seed in game_seeds],
                game_seeds,
                scientist.config.train.temperature_moves,
            )
            for record in records:
                scientist.replay.add(record)
                witness = _verified_witness(knot, record, scientist.config)
                if witness is not None:
                    solved_records.append((scientist_index, record, witness))

        shared = None
        if solved_records:
            source, best_record, witness = min(
                solved_records,
                key=lambda item: (item[2].crossing_changes, item[2].moves, item[0]),
            )
            for scientist_index, scientist in enumerate(scientists):
                if scientist_index == source:
                    continue
                shared_record = copy.deepcopy(best_record)
                for position in shared_record:
                    position.shared_witness = True
                scientist.replay.add(shared_record)
            shared = {
                "source": scientists[source].name,
                "crossing_changes": witness.crossing_changes,
                "moves": witness.moves,
                "instance_id": witness.instance_id,
            }

        for scientist in scientists:
            losses[scientist.name] = []
            if scientist.replay.position_count:
                for _ in range(train_steps):
                    losses[scientist.name].append(
                        train_alphazero_step(
                            scientist.network,
                            scientist.optimizer,
                            scientist.replay,
                            batch_size,
                            torch.device(device),
                        )
                    )
            checkpoint_dir = output / "checkpoints" / scientist.name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "network": scientist.network.state_dict(),
                    "optimizer": scientist.optimizer.state_dict(),
                    "adaptive_round": round_index,
                    "selected": knot.name,
                },
                checkpoint_dir / f"round{round_index:03d}.pt",
            )

        # Legacy rung-23 snapshots need one real adaptive batch before their
        # newly introduced factorized towers are allowed to schedule knots.
        if train_steps and solved_records:
            for scientist in scientists:
                if scientist.prediction_source == "legacy_proxy":
                    scientist.prediction_source = "factorized_adaptive"

        row = {
            "round": round_index,
            "knot": asdict(knot),
            "selected_scientist": scientists[selected_scientist].name,
            "reason": reason,
            "proposals": [
                {
                    "scientist": scientist.name,
                    "knot": proposal_names[i],
                    "proposal_score": float(score_rows[i][proposal]),
                    "prediction_source": round_sources[i],
                    "ignored_after": scientist.ignored_rounds,
                }
                for i, (scientist, proposal) in enumerate(zip(scientists, proposals, strict=True))
            ],
            "selected_predictions": {
                scientist.name: {
                    "p_solve": float(predictions[i][0][selected_index]),
                    "predicted_crossings": float(predictions[i][1][selected_index]),
                    "simplicity": float(score_rows[i][selected_index]),
                }
                for i, scientist in enumerate(scientists)
            },
            "all_predictions": {
                scientist.name: [
                    {
                        "knot": remaining_names[knot_index],
                        "p_solve": float(predictions[i][0][knot_index]),
                        "predicted_crossings": float(predictions[i][1][knot_index]),
                        "simplicity": float(score_rows[i][knot_index]),
                    }
                    for knot_index in range(len(remaining_names))
                ]
                for i, scientist in enumerate(scientists)
            },
            "shared_witness": shared,
            "last_losses": {
                name: values[-1] if values else None for name, values in losses.items()
            },
        }
        _write_jsonl(schedule_path, row)
        selected.append(knot.name)

    report = {
        "scientists": [
            {
                "name": s.name,
                "checkpoint": str(s.checkpoint),
                "prediction_source": s.prediction_source,
            }
            for s in scientists
        ],
        "pool_size": len(pool),
        "pool_sha256": hashlib.sha256(pool_json.encode()).hexdigest(),
        "rounds": len(selected),
        "selected": selected,
        "starvation_rounds": starvation,
        "score": "p_solve * (20 - predicted_crossing_changes)",
        "shared_cost_semantics": "verified upper bound (one-sided hinge)",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def default_rung23_checkpoints(root: Path) -> dict[str, Path]:
    names = ("u1-puct", "search-heavy", "wide-net")
    return {
        name: root / "artifacts" / "deep-ladder" / name / "checkpoints" / name / "stage23-after.pt"
        for name in names
    }
