from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pgx_mcts_bench.config import BraidGameConfig, ExperimentConfig
from pgx_mcts_bench.data import GameRecord, Position, ReplayBuffer
from pgx_mcts_bench.game import GameAdapter, make_game
from pgx_mcts_bench.networks import (
    AlphaZeroNet,
    MuZeroNet,
    PolicyValueNet,
    load_policy_value_state_dict,
    make_braid_network,
)
from pgx_mcts_bench.search import NeuralMCTS


def _observations(items: list[Position], device: torch.device) -> Tensor:
    array = np.stack([item.observation for item in items])
    return (
        torch.from_numpy(array)
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )


def _policies(items: list[Position], device: torch.device) -> Tensor:
    return torch.from_numpy(np.stack([item.policy for item in items])).float().to(device)


def _outcomes(items: list[Position], device: torch.device) -> Tensor:
    return torch.tensor([item.outcome for item in items], dtype=torch.float32, device=device)


def _legal(items: list[Position], device: torch.device) -> Tensor:
    return torch.from_numpy(np.stack([item.legal_actions for item in items])).float().to(device)


def policy_loss(logits: Tensor, target: Tensor) -> Tensor:
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def upper_bound_cost_loss(prediction: Tensor, target: Tensor, shared: Tensor) -> Tensor:
    """Equality loss for own outcomes, one-sided loss for shared witnesses."""
    equality = F.smooth_l1_loss(prediction, target, reduction="none")
    upper_bound = F.smooth_l1_loss(
        torch.relu(prediction - target), torch.zeros_like(prediction), reduction="none"
    )
    return torch.where(shared, upper_bound, equality)


def play_selfplay_game(
    game: GameAdapter,
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
    game: GameAdapter,
    search: NeuralMCTS,
    rngs: list[np.random.Generator],
    seeds: list[int],
    temperature_moves: int,
    random_first_role: bool = False,
) -> list[GameRecord]:
    """Play independent games while batching every neural search evaluation."""
    if len(rngs) != len(seeds):
        raise ValueError("One RNG is required per self-play game")
    transitions = [game.reset(seed) for seed in seeds]
    records: list[GameRecord] = [[] for _ in seeds]
    moves = [0 for _ in seeds]
    first_role = [game.first_role_player(t.state) for t in transitions]

    while True:
        active = [
            index for index, transition in enumerate(transitions) if not transition.terminated
        ]
        if not active:
            break
        # The first role (the Scrambler) plays uniform-random legal moves and its
        # positions are not trained on. A learned adversary measured no better
        # than this over 8 seeds and collapsed to worse on some, so it is a fixed
        # generator now rather than an agent.
        searched = [
            index
            for index in active
            if not (random_first_role and transitions[index].player == first_role[index])
        ]
        results_by_index: dict[int, Any] = {}
        if searched:
            batch = search.run_batch(
                states=[transitions[i].state for i in searched],
                observations=[transitions[i].observation for i in searched],
                legal_actions=[transitions[i].legal_actions for i in searched],
                rngs=[rngs[i] for i in searched],
                temperatures=[
                    1.0 if moves[i] < temperature_moves else 0.0 for i in searched
                ],
                add_root_noise=True,
            )
            results_by_index = dict(zip(searched, batch, strict=True))
        for index in active:
            transition = transitions[index]
            if index in results_by_index:
                result = results_by_index[index]
                position = Position(
                    observation=transition.observation,
                    legal_actions=transition.legal_actions,
                    policy=result.policy.astype(np.float32),
                    action=result.action,
                    player=transition.player,
                    role=0 if transition.player == first_role[index] else 1,
                    episode_seed=int(seeds[index]),
                    value_potential=game.value_potential(
                        transition.state, transition.player
                    ),
                )
                action = result.action
            else:
                position = None
                action = int(
                    rngs[index].choice(np.flatnonzero(transition.legal_actions))
                )
            next_transition = game.step(transition.state, action)
            if position is not None:
                position.reward = search.edge_reward(
                    transition.state, transition.player, next_transition
                )
                position.next_terminated = next_transition.terminated
                records[index].append(position)
            transitions[index] = next_transition
            moves[index] += 1

    for record, transition in zip(records, transitions, strict=True):
        rewards = game.final_rewards(transition.state)
        final = (
            game.unwrap(transition.state)
            if hasattr(game, "unwrap")
            else transition.state
        )
        is_braid = all(
            hasattr(final, field)
            for field in ("_word", "_n", "_crossing_changes", "_budget")
        )
        if is_braid:
            solved = float(
                bool((np.asarray(final._word) == 0).all())
                and int(np.asarray(final._n)) == 1
            )
            crossing_changes = float(np.asarray(final._crossing_changes))
            final_moves = float(
                game.config.simplify_budget - int(np.asarray(final._budget))
            )
        for position in record:
            position.outcome = float(rewards[position.player])
            if search.config.potential_cost_shaping:
                position.outcome -= position.value_potential
            if is_braid:
                position.solved = solved
                position.final_crossing_changes = crossing_changes
                position.final_moves = final_moves
    return records


def second_role_win_rate(records: list[GameRecord]) -> float:
    """Fraction of games the second role (the Simplifier) won.

    The curriculum signal. A run where this stays at zero cannot learn: every
    target says the Simplifier lost, so there is nothing to imitate.
    """
    wins = 0
    counted = 0
    for record in records:
        outcomes = [p.outcome for p in record if p.role == 1]
        if not outcomes:
            continue
        counted += 1
        wins += outcomes[0] > 0
    return wins / counted if counted else 0.0


def train_alphazero_step(
    network: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    network.train()
    batch = replay.sample_positions(batch_size)
    observations = _observations(batch, device)
    logits, values, auxiliary = network.forward_with_auxiliary(observations)
    p_loss = policy_loss(logits, _policies(batch, device))
    v_loss = F.mse_loss(values, _outcomes(batch, device))
    zero = values.sum() * 0.0
    auxiliary_loss = solve_loss = crossings_loss = moves_loss = zero
    solve_brier = shadow_mae = zero
    if auxiliary is not None:
        solve_logits, predicted_crossings, predicted_moves = auxiliary
        members = solve_logits.shape[1]
        solved = torch.tensor(
            [float(getattr(position, "solved", -1.0)) for position in batch],
            device=device,
        )
        roles = torch.tensor([position.role for position in batch], device=device)
        eligible = (solved >= 0.0) & (roles == 1)
        masks = []
        for position in batch:
            seed = int(getattr(position, "episode_seed", 0)) & 0xFFFFFFFF
            row = [
                ((seed * 1_103_515_245 + (member + 1) * 12_345) & 0xFFFFFFFF) % 10 < 8
                for member in range(members)
            ]
            masks.append(row)
        bootstrap = torch.tensor(masks, dtype=torch.bool, device=device)
        solve_mask = bootstrap & eligible[:, None]

        def masked_mean(losses: Tensor, mask: Tensor) -> Tensor:
            count = mask.sum()
            return (
                (losses * mask).sum() / count.clamp(min=1)
                if bool(count)
                else losses.sum() * 0.0
            )

        solve_targets = solved[:, None].expand_as(solve_logits).clamp(0.0, 1.0)
        solve_loss = masked_mean(
            F.binary_cross_entropy_with_logits(
                solve_logits, solve_targets, reduction="none"
            ),
            solve_mask,
        )
        solved_mask = solve_mask & (solve_targets > 0.5)
        crossing_targets = torch.tensor(
            [
                float(getattr(position, "final_crossing_changes", float("nan")))
                for position in batch
            ],
            device=device,
        )[:, None].expand_as(predicted_crossings)
        move_targets = torch.tensor(
            [float(getattr(position, "final_moves", float("nan"))) for position in batch],
            device=device,
        )[:, None].expand_as(predicted_moves)
        budget = float(getattr(network, "auxiliary_budget", 1.0))
        shared = torch.tensor(
            [bool(getattr(position, "shared_witness", False)) for position in batch],
            dtype=torch.bool,
            device=device,
        )[:, None].expand_as(predicted_crossings)
        crossings_loss = masked_mean(
            upper_bound_cost_loss(
                predicted_crossings / budget,
                torch.nan_to_num(crossing_targets) / budget,
                shared,
            ),
            solved_mask & torch.isfinite(crossing_targets),
        )
        moves_loss = masked_mean(
            upper_bound_cost_loss(
                predicted_moves / budget,
                torch.nan_to_num(move_targets) / budget,
                shared,
            ),
            solved_mask & torch.isfinite(move_targets),
        )
        auxiliary_loss = solve_loss + crossings_loss + moves_loss
        if bool(eligible.any()):
            mean_probability = solve_logits.sigmoid().mean(dim=1)
            solve_brier = ((mean_probability[eligible] - solved[eligible]) ** 2).mean()
            composed = network.composed_auxiliary_value(observations, auxiliary)
            shadow_mae = (composed[eligible] - values.detach()[eligible]).abs().mean()
    relation = (
        network.regularization_loss()
        if hasattr(network, "regularization_loss")
        else torch.zeros((), device=device)
    )
    auxiliary_weight = float(getattr(network, "auxiliary_loss_weight", 0.0))
    loss = p_loss + v_loss + auxiliary_weight * auxiliary_loss + 0.1 * relation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 5.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "policy": float(p_loss.item()),
        "value": float(v_loss.item()),
        "auxiliary": float(auxiliary_loss.item()),
        "solve": float(solve_loss.item()),
        "crossings": float(crossings_loss.item()),
        "moves": float(moves_loss.item()),
        "solve_brier": float(solve_brier.item()),
        "shadow_mae": float(shadow_mae.item()),
        "relation": float(relation.item()),
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
            reward_target = torch.tensor([position.reward], dtype=torch.float32, device=device)
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
    network: PolicyValueNet | MuZeroNet
    history: list[dict[str, float]]
    config: ExperimentConfig


def _new_network(kind: str, config: ExperimentConfig) -> PolicyValueNet | MuZeroNet:
    if kind == "alphazero":
        if isinstance(config.game, BraidGameConfig):
            return make_braid_network(config.game, config.model)
        return AlphaZeroNet(config.game, config.model)
    if kind == "muzero":
        return MuZeroNet(config.game, config.model)
    raise ValueError(f"Unknown agent: {kind}")


def _limit_records(records: list[GameRecord], positions: int) -> list[GameRecord]:
    """Keep exactly ``positions`` completed-game positions, preserving sequences."""
    if positions <= 0:
        return records
    limited: list[GameRecord] = []
    remaining = positions
    for record in records:
        if remaining <= 0:
            break
        kept = record[:remaining]
        if kept:
            limited.append(kept)
            remaining -= len(kept)
    if remaining:
        raise ValueError(f"Requested {positions} positions but only found {positions - remaining}")
    return limited


def _checkpoint_path(checkpoint_dir: Path, kind: str, iteration: int) -> Path:
    return checkpoint_dir / f"{kind}-iteration-{iteration:04d}.pt"


def _latest_checkpoint(checkpoint_dir: Path, kind: str) -> Path | None:
    paths = sorted(checkpoint_dir.glob(f"{kind}-iteration-*.pt"))
    return paths[-1] if paths else None


def _save_checkpoint(
    path: Path,
    *,
    kind: str,
    iteration: int,
    config: ExperimentConfig,
    network: PolicyValueNet | MuZeroNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    rng: np.random.Generator,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "version": 1,
            "kind": kind,
            "iteration": iteration,
            "config": config.to_dict(),
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "replay_games": replay.games,
            "rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "history": history,
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    kind: str,
    config: ExperimentConfig,
    network: PolicyValueNet | MuZeroNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[int, list[dict[str, float]]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("version") != 1 or payload.get("kind") != kind:
        raise ValueError(f"Incompatible checkpoint: {path}")
    saved_config = ExperimentConfig.from_dict(payload["config"])
    saved_train = asdict(saved_config.train)
    current_train = asdict(config.train)
    for runtime_field in (
        "iterations",
        "device",
        "checkpoint_iterations",
        "learning_curve_games",
    ):
        saved_train.pop(runtime_field)
        current_train.pop(runtime_field)
    if (
        saved_config.game != config.game
        or saved_config.search != config.search
        or saved_config.model != config.model
        or saved_train != current_train
    ):
        raise ValueError(f"Checkpoint configuration does not match this run: {path}")
    migrated = (
        load_policy_value_state_dict(network, payload["network"])
        if isinstance(network, PolicyValueNet)
        else False
    )
    if not isinstance(network, PolicyValueNet):
        network.load_state_dict(payload["network"])
    try:
        optimizer.load_state_dict(payload["optimizer"])
    except ValueError:
        if not migrated:
            raise
    replay.games = payload["replay_games"]
    replay.position_count = sum(len(game) for game in replay.games)
    rng.bit_generator.state = payload["rng_state"]
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    return int(payload["iteration"]), list(payload["history"])


def train_agent(
    kind: str,
    config: ExperimentConfig,
    *,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    iteration_hook: Callable[[int, PolicyValueNet | MuZeroNet], str | None] | None = None,
) -> TrainedAgent:
    """`iteration_hook(iteration, network)` runs after each iteration; whatever
    string it returns is appended to that iteration's log line. Used to evaluate
    against a frozen anchor set without threading game-specific code through the
    training loop."""
    torch.manual_seed(config.train.seed)
    np_rng = np.random.default_rng(config.train.seed)
    device = torch.device(config.train.device)
    game = make_game(config.game)
    # Curriculum: train at a reduced Scrambler budget and climb toward the
    # target only when the Simplifier is winning. The evaluation anchors stay at
    # the target difficulty, so this changes what the agent trains on and not
    # what it is measured against.
    target_k = getattr(config.game, "scramble_budget", 0)
    current_k = config.train.curriculum_start_k
    use_curriculum = bool(current_k) and isinstance(config.game, BraidGameConfig)
    if use_curriculum:
        current_k = min(current_k, target_k)
        game = make_game(replace(config.game, scramble_budget=current_k))
    network = _new_network(kind, config)
    network.to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    replay = ReplayBuffer(config.train.replay_capacity, np_rng)
    search = NeuralMCTS(game, network, config.search, config.train.device)
    history: list[dict[str, float]] = []
    start_iteration = 0
    if resume and checkpoint_dir is not None:
        latest = _latest_checkpoint(checkpoint_dir, kind)
        if latest is not None:
            start_iteration, history = _load_checkpoint(
                latest,
                kind=kind,
                config=config,
                network=network,
                optimizer=optimizer,
                replay=replay,
                rng=np_rng,
                device=device,
            )
            print(f"{kind} resumed from iteration {start_iteration}: {latest}")

    checkpoint_iterations = set(config.train.checkpoint_iterations)
    checkpoint_iterations.add(config.train.iterations)
    for iteration in range(start_iteration, config.train.iterations):
        started = time.perf_counter()
        records: list[GameRecord] = []
        simulated_positions = 0
        round_index = 0
        target_positions = config.train.selfplay_positions_per_iteration
        while round_index == 0 or (target_positions > 0 and simulated_positions < target_positions):
            base_seed = (
                config.train.seed
                + iteration * 1_000_000
                + round_index * config.train.selfplay_games
            )
            seeds = [base_seed + game_index for game_index in range(config.train.selfplay_games)]
            game_rngs = [np.random.default_rng(seed + 1_000_003) for seed in seeds]
            batch_records = play_selfplay_games(
                game,
                search,
                game_rngs,
                seeds,
                config.train.temperature_moves,
                config.train.random_first_role,
            )
            records.extend(batch_records)
            simulated_positions += sum(len(record) for record in batch_records)
            round_index += 1
        simulated_game_count = len(records)
        simulated_lengths = [len(record) for record in records]
        if target_positions > 0 and config.train.exact_position_budget:
            records = _limit_records(records, target_positions)
        generated_positions = sum(len(record) for record in records)
        kept_lengths = []
        for record in records:
            replay.add(record)
            kept_lengths.append(len(record))
        metrics: dict[str, float] = {}
        for _ in range(config.train.train_steps):
            if kind == "alphazero":
                metrics = train_alphazero_step(
                    network,
                    optimizer,
                    replay,
                    config.train.batch_size,
                    device,
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
        simplifier_wins = second_role_win_rate(records)
        promoted = 0
        if use_curriculum and current_k < target_k:
            if simplifier_wins >= config.train.curriculum_promote_at:
                current_k += 1
                promoted = 1
                game = make_game(replace(config.game, scramble_budget=current_k))
                search = NeuralMCTS(game, network, config.search, config.train.device)
        row = {
            "iteration": float(iteration + 1),
            "simplifier_wins": float(simplifier_wins),
            "scramble_k": float(current_k if use_curriculum else target_k),
            "promoted": float(promoted),
            "positions": float(replay.position_count),
            "positions_generated": float(generated_positions),
            "positions_simulated": float(simulated_positions),
            "games_generated": float(simulated_game_count),
            "records_kept": float(len(records)),
            "mean_game_length": float(np.mean(simulated_lengths)),
            "mean_kept_record_length": float(np.mean(kept_lengths)),
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        history.append(row)
        note = iteration_hook(iteration + 1, network) if iteration_hook is not None else None
        suffix = f" | {note}" if note else ""
        print(f"{kind} iteration {iteration + 1}/{config.train.iterations}: {row}{suffix}")
        completed_iteration = iteration + 1
        if checkpoint_dir is not None and completed_iteration in checkpoint_iterations:
            _save_checkpoint(
                _checkpoint_path(checkpoint_dir, kind, completed_iteration),
                kind=kind,
                iteration=completed_iteration,
                config=config,
                network=network,
                optimizer=optimizer,
                replay=replay,
                rng=np_rng,
                history=history,
            )
    return TrainedAgent(kind, network, history, config)


def play_arena_game(
    first: TrainedAgent,
    second: TrainedAgent,
    seed: int,
    *,
    first_is_black: bool | None = None,
    opening_moves: int | None = None,
) -> float:
    """Return +1 first-agent win, -1 second-agent win, or 0 draw.

    `first_is_black` selects which side the first agent takes: Black in Go, the
    Scrambler in the braid game. Random opening plies keep temperature-0 arena
    games from all being identical; how many is a property of the game.
    """
    if first.config.game != second.config.game:
        raise ValueError("Cross-play agents must use the same game configuration")
    game = make_game(first.config.game)
    if opening_moves is None:
        opening_moves = first.config.game.opening_moves
    rng = np.random.default_rng(seed)
    transition = game.reset(seed)
    for _ in range(opening_moves):
        if transition.terminated:
            break
        actions = np.flatnonzero(transition.legal_actions)
        action = int(rng.choice(actions))
        transition = game.step(transition.state, action)
    # Pgx randomizes the external player-id order, so ask the adapter which id
    # holds the first role rather than assuming it is player 0.
    black_player = game.first_role_player(transition.state)
    if first_is_black is None:
        first_is_black = seed % 2 == 0
    first_player = black_player if first_is_black else 1 - black_player
    searches = {
        first_player: NeuralMCTS(
            game,
            first.network,
            first.config.search,
            first.config.train.device,
        ),
        1 - first_player: NeuralMCTS(
            game,
            second.network,
            second.config.search,
            second.config.train.device,
        ),
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


def compare_pair(
    first: TrainedAgent,
    second: TrainedAgent,
    games: int,
    *,
    seed: int,
) -> dict[str, float | int]:
    games = games if games % 2 == 0 else games + 1
    paired: list[tuple[bool, float]] = []
    for pair_index in range(games // 2):
        opening_seed = seed + pair_index
        paired.append(
            (
                True,
                play_arena_game(
                    first,
                    second,
                    opening_seed,
                    first_is_black=True,
                ),
            )
        )
        paired.append(
            (
                False,
                play_arena_game(
                    first,
                    second,
                    opening_seed,
                    first_is_black=False,
                ),
            )
        )
    outcomes = [outcome for _, outcome in paired]
    black_outcomes = [outcome for is_black, outcome in paired if is_black]
    white_outcomes = [outcome for is_black, outcome in paired if not is_black]
    return {
        "games": games,
        "first_wins": sum(value > 0 for value in outcomes),
        "second_wins": sum(value < 0 for value in outcomes),
        "draws": sum(value == 0 for value in outcomes),
        "first_score": float(np.mean([(value + 1.0) / 2.0 for value in outcomes])),
        "first_as_black_wins": sum(value > 0 for value in black_outcomes),
        "first_as_black_games": len(black_outcomes),
        "first_as_white_wins": sum(value > 0 for value in white_outcomes),
        "first_as_white_games": len(white_outcomes),
    }


def play_baseline_game(
    agent: TrainedAgent,
    seed: int,
    *,
    agent_takes_first_role: bool,
) -> float:
    """Agent against a uniform-random opponent. Returns +1 if the agent wins.

    This is the measurement the braid game actually needs. An arena between two
    trained agents says which is stronger; it does not say whether either has
    learned anything, because a Scrambler that produces trivial instances and a
    Simplifier that cannot solve them look the same as a hard pairing. Playing
    each role against random gives an absolute number with a known baseline:
    a uniform-random Simplifier undoes ~1.6% of K=6 tier-0 scrambles.
    """
    game = make_game(agent.config.game)
    rng = np.random.default_rng(seed)
    transition = game.reset(seed)
    role_player = game.first_role_player(transition.state)
    agent_player = role_player if agent_takes_first_role else 1 - role_player
    search = NeuralMCTS(game, agent.network, agent.config.search, agent.config.train.device)
    while not transition.terminated:
        if transition.player == agent_player:
            result = search.run(
                transition.state,
                transition.observation,
                transition.legal_actions,
                rng,
                temperature=0.0,
                add_root_noise=False,
            )
            action = result.action
        else:
            action = int(rng.choice(np.flatnonzero(transition.legal_actions)))
        transition = game.step(transition.state, action)
    return float(game.final_rewards(transition.state)[agent_player])


def evaluate_against_random(
    agent: TrainedAgent,
    games: int,
    *,
    seed: int,
) -> dict[str, float | int]:
    """Win rate of the agent in each role against a uniform-random opponent."""
    scores: dict[bool, list[float]] = {True: [], False: []}
    for role in (True, False):
        for index in range(games):
            scores[role].append(
                play_baseline_game(agent, seed + index, agent_takes_first_role=role)
            )
    return {
        "games_per_role": games,
        "first_role_win_rate": float(np.mean([s > 0 for s in scores[True]])),
        "second_role_win_rate": float(np.mean([s > 0 for s in scores[False]])),
    }


def compare_agents(
    alphazero: TrainedAgent,
    muzero: TrainedAgent,
    config: ExperimentConfig,
    games: int,
) -> dict[str, float | int]:
    result = compare_pair(
        alphazero,
        muzero,
        games,
        seed=config.train.seed + 100_000,
    )
    return {
        "games": result["games"],
        "alphazero_wins": result["first_wins"],
        "muzero_wins": result["second_wins"],
        "draws": result["draws"],
        "alphazero_score": result["first_score"],
        "alphazero_as_black_wins": result["first_as_black_wins"],
        "alphazero_as_black_games": result["first_as_black_games"],
        "alphazero_as_white_wins": result["first_as_white_wins"],
        "alphazero_as_white_games": result["first_as_white_games"],
    }


def load_agent(
    artifact: Path,
    kind: str,
    *,
    device: str = "cpu",
    checkpoint: Path | None = None,
) -> TrainedAgent:
    results_path = artifact / "results.json"
    payload = json.loads(results_path.read_text())
    config_payload = payload["config"]
    config_payload["train"]["device"] = device
    config = ExperimentConfig.from_dict(config_payload)
    network = _new_network(kind, config).to(torch.device(device))
    weights_path = checkpoint or artifact / f"{kind}.pt"
    weights = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(weights, dict) and "network" in weights:
        history = list(weights.get("history", []))
        weights = weights["network"]
    else:
        history = list(payload.get("training", {}).get(kind, []))
    if isinstance(network, PolicyValueNet):
        load_policy_value_state_dict(network, weights)
    else:
        network.load_state_dict(weights)
    return TrainedAgent(kind, network, history, config)


def evaluate_learning_curve(
    out: Path,
    config: ExperimentConfig,
    games: int,
) -> list[dict[str, Any]]:
    if games <= 0:
        return []
    rows: list[dict[str, Any]] = []
    checkpoint_dir = out / "checkpoints"
    iterations = sorted(
        {
            *config.train.checkpoint_iterations,
            config.train.iterations,
        }
    )
    for iteration in iterations:
        if iteration > config.train.iterations:
            continue
        alphazero_path = _checkpoint_path(checkpoint_dir, "alphazero", iteration)
        muzero_path = _checkpoint_path(checkpoint_dir, "muzero", iteration)
        if not alphazero_path.exists() or not muzero_path.exists():
            continue
        alphazero = load_agent(
            out,
            "alphazero",
            device=config.train.device,
            checkpoint=alphazero_path,
        )
        muzero = load_agent(
            out,
            "muzero",
            device=config.train.device,
            checkpoint=muzero_path,
        )
        arena = compare_agents(alphazero, muzero, config, games)
        row = {"iteration": iteration, "arena": arena}
        rows.append(row)
        print(f"learning curve iteration {iteration}: {arena}")
    (out / "learning_curve.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def parameter_count(network: nn.Module) -> int:
    return sum(parameter.numel() for parameter in network.parameters())


def save_braid_experiment(
    out: Path,
    config: ExperimentConfig,
    agent: TrainedAgent,
    baseline: dict[str, float | int],
    selfplay_arena: dict[str, float | int] | None = None,
) -> None:
    """Single-agent results file for the braid game (no MuZero counterpart yet)."""
    out.mkdir(parents=True, exist_ok=True)
    torch.save(agent.network.state_dict(), out / "alphazero.pt")
    payload = {
        "config": config.to_dict(),
        "parameters": {"alphazero": parameter_count(agent.network)},
        "history": {"alphazero": agent.history},
        "baseline_vs_random": baseline,
    }
    if selfplay_arena is not None:
        payload["selfplay_arena"] = selfplay_arena
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


def save_experiment(
    out: Path,
    config: ExperimentConfig,
    alphazero: TrainedAgent,
    muzero: TrainedAgent,
    arena: dict[str, float | int],
    learning_curve: list[dict[str, Any]] | None = None,
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
        "learning_curve": learning_curve or [],
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
