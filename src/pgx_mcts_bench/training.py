from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pgx_mcts_bench.config import ExperimentConfig
from pgx_mcts_bench.data import GameRecord, Position, ReplayBuffer
from pgx_mcts_bench.game import Go6x6
from pgx_mcts_bench.networks import AlphaZeroNet, MuZeroNet
from pgx_mcts_bench.search import NeuralMCTS


def _observations(items: list[Position], device: torch.device) -> Tensor:
    array = np.stack([item.observation for item in items])
    return torch.from_numpy(array).permute(0, 3, 1, 2).float().to(device)


def _policies(items: list[Position], device: torch.device) -> Tensor:
    return torch.from_numpy(np.stack([item.policy for item in items])).float().to(device)


def _outcomes(items: list[Position], device: torch.device) -> Tensor:
    return torch.tensor([item.outcome for item in items], dtype=torch.float32, device=device)


def _legal(items: list[Position], device: torch.device) -> Tensor:
    return torch.from_numpy(np.stack([item.legal_actions for item in items])).float().to(device)


def policy_loss(logits: Tensor, target: Tensor) -> Tensor:
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def play_selfplay_game(
    game: Go6x6,
    search: NeuralMCTS,
    rng: np.random.Generator,
    seed: int,
    temperature_moves: int,
) -> GameRecord:
    return play_selfplay_games(
        game,
        search,
        [rng],
        [seed],
        temperature_moves,
    )[0]


def play_selfplay_games(
    game: Go6x6,
    search: NeuralMCTS,
    rngs: list[np.random.Generator],
    seeds: list[int],
    temperature_moves: int,
) -> list[GameRecord]:
    """Play independent games while batching every neural search evaluation."""
    if len(rngs) != len(seeds):
        raise ValueError("One RNG is required per self-play game")
    transitions = [game.reset(seed) for seed in seeds]
    records: list[GameRecord] = [[] for _ in seeds]
    moves = [0 for _ in seeds]

    while True:
        active = [
            index for index, transition in enumerate(transitions) if not transition.terminated
        ]
        if not active:
            break
        results = search.run_batch(
            states=[transitions[index].state for index in active],
            observations=[transitions[index].observation for index in active],
            legal_actions=[transitions[index].legal_actions for index in active],
            rngs=[rngs[index] for index in active],
            temperatures=[
                1.0 if moves[index] < temperature_moves else 0.0 for index in active
            ],
            add_root_noise=True,
        )
        for index, result in zip(active, results, strict=True):
            transition = transitions[index]
            position = Position(
                observation=transition.observation,
                legal_actions=transition.legal_actions,
                policy=result.policy.astype(np.float32),
                action=result.action,
                player=transition.player,
            )
            next_transition = game.step(transition.state, result.action)
            position.reward = next_transition.reward
            position.next_terminated = next_transition.terminated
            records[index].append(position)
            transitions[index] = next_transition
            moves[index] += 1

    for record, transition in zip(records, transitions, strict=True):
        rewards = game.final_rewards(transition.state)
        for position in record:
            position.outcome = float(rewards[position.player])
    return records


def train_alphazero_step(
    network: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    network.train()
    batch = replay.sample_positions(batch_size)
    logits, values = network(_observations(batch, device))
    p_loss = policy_loss(logits, _policies(batch, device))
    v_loss = F.mse_loss(values, _outcomes(batch, device))
    loss = p_loss + v_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 5.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "policy": float(p_loss.item()),
        "value": float(v_loss.item()),
    }


def train_muzero_step(
    network: MuZeroNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    unroll_steps: int,
    device: torch.device,
) -> dict[str, float]:
    network.train()
    sequences = replay.sample_sequences(batch_size, unroll_steps)
    total = torch.zeros((), device=device)
    p_total = torch.zeros((), device=device)
    v_total = torch.zeros((), device=device)
    r_total = torch.zeros((), device=device)
    l_total = torch.zeros((), device=device)
    t_total = torch.zeros((), device=device)
    predictions = 0
    transitions = 0

    for sequence in sequences:
        first = sequence[0]
        hidden, logits, value, legal_logits, terminal_logits = network.initial_inference(
            _observations([first], device)
        )
        for index, position in enumerate(sequence):
            if index > 0:
                logits, value, legal_logits, terminal_logits = network.prediction(hidden)
                assert legal_logits is not None and terminal_logits is not None
            p_loss = policy_loss(logits, _policies([position], device))
            v_loss = F.mse_loss(value, _outcomes([position], device))
            legal_loss = F.binary_cross_entropy_with_logits(
                legal_logits, _legal([position], device)
            )
            state_terminal_loss = F.binary_cross_entropy_with_logits(
                terminal_logits,
                torch.zeros_like(terminal_logits),
            )
            p_total = p_total + p_loss
            v_total = v_total + v_loss
            l_total = l_total + legal_loss
            t_total = t_total + state_terminal_loss
            total = total + p_loss + v_loss + 0.25 * legal_loss + 0.25 * state_terminal_loss
            predictions += 1
            action = torch.tensor([position.action], dtype=torch.long, device=device)
            hidden, predicted_reward = network.dynamics(hidden, action)
            _, _, _, next_terminal_logits = network.prediction(hidden)
            assert next_terminal_logits is not None
            reward_target = torch.tensor(
                [position.reward], dtype=torch.float32, device=device
            )
            terminal_target = torch.tensor(
                [position.next_terminated],
                dtype=torch.float32,
                device=device,
            )
            reward_loss = F.mse_loss(predicted_reward, reward_target)
            terminal_weight = torch.where(
                terminal_target > 0,
                torch.full_like(terminal_target, 20.0),
                torch.ones_like(terminal_target),
            )
            terminal_loss = F.binary_cross_entropy_with_logits(
                next_terminal_logits,
                terminal_target,
                weight=terminal_weight,
            )
            reward_weight = 10.0 if position.next_terminated else 1.0
            r_total = r_total + reward_weight * reward_loss
            t_total = t_total + terminal_loss
            total = total + reward_weight * reward_loss + terminal_loss
            transitions += 1
            if index + 1 < len(sequence):
                # Prevent gradients through arbitrarily long sampled histories.
                hidden.register_hook(lambda gradient: gradient * 0.5)

    loss = total / max(predictions, 1)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 5.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "policy": float((p_total / predictions).item()),
        "value": float((v_total / predictions).item()),
        "reward": float((r_total / max(transitions, 1)).item()),
        "legal": float((l_total / predictions).item()),
        "terminal": float((t_total / (predictions + transitions)).item()),
    }


@dataclass
class TrainedAgent:
    name: str
    network: AlphaZeroNet | MuZeroNet
    history: list[dict[str, float]]


def train_agent(kind: str, config: ExperimentConfig) -> TrainedAgent:
    torch.manual_seed(config.train.seed)
    np_rng = np.random.default_rng(config.train.seed)
    device = torch.device(config.train.device)
    game = Go6x6(config.game)
    if kind == "alphazero":
        network: AlphaZeroNet | MuZeroNet = AlphaZeroNet(config.game, config.model)
    elif kind == "muzero":
        network = MuZeroNet(config.game, config.model)
    else:
        raise ValueError(f"Unknown agent: {kind}")
    network.to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    replay = ReplayBuffer(config.train.replay_capacity, np_rng)
    search = NeuralMCTS(game, network, config.search, config.train.device)
    history: list[dict[str, float]] = []

    for iteration in range(config.train.iterations):
        started = time.perf_counter()
        records: list[GameRecord] = []
        generated_positions = 0
        round_index = 0
        target_positions = config.train.selfplay_positions_per_iteration
        while round_index == 0 or (
            target_positions > 0 and generated_positions < target_positions
        ):
            base_seed = (
                config.train.seed
                + iteration * 1_000_000
                + round_index * config.train.selfplay_games
            )
            seeds = [
                base_seed + game_index
                for game_index in range(config.train.selfplay_games)
            ]
            game_rngs = [
                np.random.default_rng(seed + 1_000_003)
                for seed in seeds
            ]
            batch_records = play_selfplay_games(
                game,
                search,
                game_rngs,
                seeds,
                config.train.temperature_moves,
            )
            records.extend(batch_records)
            generated_positions += sum(len(record) for record in batch_records)
            round_index += 1
        lengths = []
        for record in records:
            replay.add(record)
            lengths.append(len(record))
        metrics: dict[str, float] = {}
        for _ in range(config.train.train_steps):
            if kind == "alphazero":
                metrics = train_alphazero_step(
                    network, optimizer, replay, config.train.batch_size, device
                )
            else:
                assert isinstance(network, MuZeroNet)
                metrics = train_muzero_step(
                    network,
                    optimizer,
                    replay,
                    config.train.batch_size,
                    config.train.unroll_steps,
                    device,
                )
        row = {
            "iteration": float(iteration + 1),
            "positions": float(replay.position_count),
            "positions_generated": float(generated_positions),
            "games_generated": float(len(records)),
            "mean_game_length": float(np.mean(lengths)),
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        history.append(row)
        print(f"{kind} iteration {iteration + 1}/{config.train.iterations}: {row}")
    return TrainedAgent(kind, network, history)


def play_arena_game(
    first: TrainedAgent,
    second: TrainedAgent,
    config: ExperimentConfig,
    seed: int,
    *,
    first_is_black: bool | None = None,
    opening_moves: int = 6,
) -> float:
    """Return +1 first-agent win, -1 second-agent win, or 0 draw."""
    game = Go6x6(config.game)
    rng = np.random.default_rng(seed)
    transition = game.reset(seed)
    for _ in range(opening_moves):
        if transition.terminated:
            break
        actions = np.flatnonzero(transition.legal_actions)
        action = int(rng.choice(actions))
        transition = game.step(transition.state, action)
    # Pgx randomizes the external player-id order. current_player at reset is
    # Black, so use the preserved mapping rather than assuming player 0 is Black.
    black_player = int(np.asarray(transition.state._player_order[0]))
    if first_is_black is None:
        first_is_black = seed % 2 == 0
    first_player = black_player if first_is_black else 1 - black_player
    searches = {
        first_player: NeuralMCTS(game, first.network, config.search, config.train.device),
        1 - first_player: NeuralMCTS(game, second.network, config.search, config.train.device),
    }
    while not transition.terminated:
        result = searches[transition.player].run(
            transition.state,
            transition.observation,
            transition.legal_actions,
            rng,
            temperature=0.0,
            add_root_noise=False,
        )
        transition = game.step(transition.state, result.action)
    rewards = game.final_rewards(transition.state)
    return float(rewards[first_player])


def compare_agents(
    alphazero: TrainedAgent,
    muzero: TrainedAgent,
    config: ExperimentConfig,
    games: int,
) -> dict[str, float | int]:
    games = games if games % 2 == 0 else games + 1
    paired: list[tuple[bool, float]] = []
    for pair_index in range(games // 2):
        seed = config.train.seed + 100_000 + pair_index
        paired.append(
            (
                True,
                play_arena_game(
                    alphazero,
                    muzero,
                    config,
                    seed,
                    first_is_black=True,
                ),
            )
        )
        paired.append(
            (
                False,
                play_arena_game(
                    alphazero,
                    muzero,
                    config,
                    seed,
                    first_is_black=False,
                ),
            )
        )
    outcomes = [outcome for _, outcome in paired]
    black_outcomes = [outcome for is_black, outcome in paired if is_black]
    white_outcomes = [outcome for is_black, outcome in paired if not is_black]
    return {
        "games": games,
        "alphazero_wins": sum(value > 0 for value in outcomes),
        "muzero_wins": sum(value < 0 for value in outcomes),
        "draws": sum(value == 0 for value in outcomes),
        "alphazero_score": float(np.mean([(value + 1.0) / 2.0 for value in outcomes])),
        "alphazero_as_black_wins": sum(value > 0 for value in black_outcomes),
        "alphazero_as_black_games": len(black_outcomes),
        "alphazero_as_white_wins": sum(value > 0 for value in white_outcomes),
        "alphazero_as_white_games": len(white_outcomes),
    }


def parameter_count(network: nn.Module) -> int:
    return sum(parameter.numel() for parameter in network.parameters())


def save_experiment(
    out: Path,
    config: ExperimentConfig,
    alphazero: TrainedAgent,
    muzero: TrainedAgent,
    arena: dict[str, float | int],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.save(alphazero.network.state_dict(), out / "alphazero.pt")
    torch.save(muzero.network.state_dict(), out / "muzero.pt")
    payload = {
        "config": config.to_dict(),
        "parameters": {
            "alphazero": parameter_count(alphazero.network),
            "muzero": parameter_count(muzero.network),
        },
        "training": {
            "alphazero": alphazero.history,
            "muzero": muzero.history,
        },
        "arena": arena,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
