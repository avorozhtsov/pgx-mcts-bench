from __future__ import annotations

import copy
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


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[nn.Parameter]:
    """Return each parameter owned by an optimizer exactly once."""
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def _native_forward_with_auxiliary(
    network: PolicyValueNet,
    observations: Tensor,
    optimized_parameter_ids: set[int],
) -> tuple[Tensor, Tensor, Any]:
    """Run the base scientist without a separately optimized sharing residual."""
    sharing_parameters = [
        parameter
        for attribute in ("option_policy_adapter", "option_policy_gate")
        if (module := getattr(network, attribute, None)) is not None
        for parameter in module.parameters()
    ]
    separately_optimized = bool(sharing_parameters) and {
        id(parameter) for parameter in sharing_parameters
    }.isdisjoint(optimized_parameter_ids)
    if not separately_optimized:
        return network.forward_with_auxiliary(observations)
    enabled = bool(getattr(network, "option_adapter_enabled", True))
    network.option_adapter_enabled = False
    try:
        return network.forward_with_auxiliary(observations)
    finally:
        network.option_adapter_enabled = enabled


def attach_policy_value_preservation_teacher(network: PolicyValueNet) -> PolicyValueNet:
    """Attach a frozen pre-update teacher without registering it in checkpoints."""
    teacher = copy.deepcopy(network).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    # A registered child module would duplicate the teacher in every state dict.
    object.__setattr__(network, "_policy_value_preservation_teacher", teacher)
    return teacher


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

    representation_id = str(getattr(getattr(game, "knot", None), "name", ""))
    objective_cap = getattr(game, "objective_cap", None)
    if (
        objective_cap is None
        and bool(getattr(game.config, "objective_budget_channel", False))
        and hasattr(game, "_global_cap")
    ):
        objective_cap = game._global_cap()
    action_horizon = int(getattr(game.config, "simplify_budget", 0))

    def residual_word_length(state: Any) -> int:
        raw = game.unwrap(state) if hasattr(game, "unwrap") else state
        word = getattr(raw, "_word", None)
        if word is None:
            return -1
        return int(np.count_nonzero(np.asarray(word)))

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
                temperatures=[1.0 if moves[i] < temperature_moves else 0.0 for i in searched],
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
                    value_potential=game.value_potential(transition.state, transition.player),
                    representation_id=representation_id,
                    objective_ratio=float(getattr(game, "ratio", float("nan"))),
                    objective_cap=(
                        float(objective_cap) if objective_cap is not None else float("nan")
                    ),
                    action_horizon=action_horizon,
                    residual_word_length=residual_word_length(transition.state),
                    mcts_root_value=float(result.root_value),
                    mcts_visit_count=int(np.asarray(result.visits).sum()),
                    episode_position_index=len(records[index]),
                )
                action = result.action
            else:
                position = None
                action = int(rngs[index].choice(np.flatnonzero(transition.legal_actions)))
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
        final = game.unwrap(transition.state) if hasattr(game, "unwrap") else transition.state
        is_braid = all(
            hasattr(final, field) for field in ("_word", "_n", "_crossing_changes", "_budget")
        )
        if is_braid:
            solved = float(
                bool((np.asarray(final._word) == 0).all()) and int(np.asarray(final._n)) == 1
            )
            crossing_changes = float(np.asarray(final._crossing_changes))
            final_moves = float(game.semantic_move_count(transition.state))
            final_native_plies = float(game.native_ply_count(transition.state))
            final_internal_plies = float(game.internal_ply_count(transition.state))
        objective_censored = transition.termination_reason == "objective_budget_exhausted"
        terminal_residual = residual_word_length(transition.state)
        known_residuals = [
            position.residual_word_length
            for position in record
            if position.residual_word_length >= 0
        ]
        if terminal_residual >= 0:
            known_residuals.append(terminal_residual)
        best_residual = min(known_residuals) if known_residuals else -1
        for position in record:
            position.outcome = float(rewards[position.player])
            if search.config.potential_cost_shaping:
                position.outcome -= position.value_potential
            if is_braid:
                position.solved = solved
                position.final_crossing_changes = crossing_changes
                position.final_moves = final_moves
                position.final_native_plies = final_native_plies
                position.final_internal_plies = final_internal_plies
                position.objective_censored = objective_censored
            position.termination_reason = transition.termination_reason
            position.best_residual_word_length = best_residual
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
    *,
    collaboration_replay: bool = False,
    shared_fraction: float = 0.1,
    policy_value_success_only: bool = False,
    replay_current_representation: str = "",
    replay_current_fraction: float = 0.0,
    replay_similar_fraction: float = 0.0,
    replay_similar_representation_count: int = 8,
    replay_positions_per_episode: int = 1,
    replay_max_position_uses: int = 0,
    continual_replay: bool = False,
    replay_rehearsal_representations: set[str] | None = None,
    replay_rehearsal_fraction: float = 0.25,
    replay_ratio_outcome_balance: tuple[float, float] | None = None,
    relative_trajectory_weight: float = 0.0,
) -> dict[str, float]:
    network.train()
    optimized_parameters = _optimizer_parameters(optimizer)
    optimized_parameter_ids = {id(parameter) for parameter in optimized_parameters}
    if bool(getattr(network, "freeze_batchnorm_stats", False)):
        for module in network.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    batch = (
        replay.sample_continual_positions(
            batch_size,
            current_representation=replay_current_representation,
            rehearsal_representations=replay_rehearsal_representations or set(),
            rehearsal_fraction=replay_rehearsal_fraction,
            positions_per_episode=replay_positions_per_episode,
            max_position_uses=replay_max_position_uses,
        )
        if continual_replay
        else replay.sample_ratio_outcome_balanced_positions(
            batch_size,
            ratios=replay_ratio_outcome_balance,
            positions_per_episode=replay_positions_per_episode,
            max_position_uses=replay_max_position_uses,
        )
        if replay_ratio_outcome_balance is not None
        else replay.sample_collaboration_positions(
            batch_size,
            shared_fraction,
            current_representation=replay_current_representation,
            current_fraction=replay_current_fraction,
            similar_fraction=replay_similar_fraction,
            similar_representation_count=replay_similar_representation_count,
            positions_per_episode=replay_positions_per_episode,
            max_position_uses=replay_max_position_uses,
        )
        if collaboration_replay
        else replay.sample_positions(batch_size)
    )
    observations = _observations(batch, device)
    logits, values, auxiliary = _native_forward_with_auxiliary(
        network, observations, optimized_parameter_ids
    )
    shared_positions = torch.tensor(
        [bool(getattr(position, "shared_witness", False)) for position in batch],
        dtype=torch.bool,
        device=device,
    )
    censored_positions = torch.tensor(
        [bool(getattr(position, "objective_censored", False)) for position in batch],
        dtype=torch.bool,
        device=device,
    )
    solved = torch.tensor(
        [float(getattr(position, "solved", -1.0)) for position in batch],
        device=device,
    )
    roles = torch.tensor([position.role for position in batch], device=device)
    native_targets = (
        ~shared_positions
        if bool(getattr(network, "shared_auxiliary_only", False))
        else torch.ones_like(shared_positions)
    ) & ~censored_positions
    if policy_value_success_only:
        native_targets &= solved > 0.5
    policy_losses = -(_policies(batch, device) * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    value_losses = (values - _outcomes(batch, device)) ** 2
    p_loss = (
        policy_losses[native_targets].mean()
        if bool(native_targets.any())
        else policy_losses.sum() * 0.0
    )
    advantages = torch.tensor(
        [float(getattr(position, "relative_trajectory_advantage", 0.0)) for position in batch],
        dtype=torch.float32,
        device=device,
    )
    relative_mask = advantages != 0.0
    chosen_log_probability = (
        F.log_softmax(logits, dim=-1)
        .gather(
            1,
            torch.tensor([position.action for position in batch], device=device)[:, None],
        )
        .squeeze(1)
    )
    if bool(relative_mask.any()):
        selected_advantages = advantages[relative_mask]
        selected_log_probability = chosen_log_probability[relative_mask]
        positive_loss = -selected_log_probability
        negative_loss = -torch.log1p(-selected_log_probability.exp().clamp(max=1.0 - 1e-6))
        contrastive_loss = torch.where(selected_advantages > 0.0, positive_loss, negative_loss)
        weights = selected_advantages.abs()
        relative_policy_loss = (weights * contrastive_loss).sum() / weights.sum()
    else:
        relative_policy_loss = chosen_log_probability.sum() * 0.0
    v_loss = (
        value_losses[native_targets].mean()
        if bool(native_targets.any())
        else value_losses.sum() * 0.0
    )
    zero = values.sum() * 0.0
    preservation_policy = preservation_value = zero
    preservation_weight = float(getattr(network, "policy_value_preservation_weight", 0.0))
    preservation_teacher = getattr(network, "_policy_value_preservation_teacher", None)
    if preservation_weight > 0.0 and preservation_teacher is not None:
        preservation_teacher.eval()
        with torch.no_grad():
            teacher_logits, teacher_values = preservation_teacher(observations)
        legal = _legal(batch, device).bool()
        floor = torch.finfo(logits.dtype).min
        student_log_probability = F.log_softmax(logits.masked_fill(~legal, floor), dim=-1)
        teacher_log_probability = F.log_softmax(teacher_logits.masked_fill(~legal, floor), dim=-1)
        teacher_probability = teacher_log_probability.exp()
        preservation_policy = (
            (teacher_probability * (teacher_log_probability - student_log_probability))
            .sum(dim=-1)
            .mean()
        )
        preservation_value = F.mse_loss(values, teacher_values)
    auxiliary_loss = solve_loss = crossings_loss = moves_loss = monotonic_loss = zero
    solve_brier = shadow_mae = zero
    crossing_target_count = move_target_count = 0
    if auxiliary is not None:
        solve_logits, predicted_crossings, predicted_moves = auxiliary
        members = solve_logits.shape[1]
        # A cap-exhausted trajectory is not a failure under the environment's
        # full clock, so it remains masked above for policy/value. It *is* an
        # observed failure of the conditional event "solve within this encoded
        # budget", and is therefore a negative solve-probability label.
        eligible = (roles == 1) & ((solved >= 0.0) | censored_positions)
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
            return (losses * mask).sum() / count.clamp(min=1) if bool(count) else losses.sum() * 0.0

        conditional_solved = torch.where(censored_positions, 0.0, solved).clamp(0.0, 1.0)
        solve_targets = conditional_solved[:, None].expand_as(solve_logits)
        solve_loss = masked_mean(
            F.binary_cross_entropy_with_logits(solve_logits, solve_targets, reduction="none"),
            solve_mask,
        )
        solved_mask = solve_mask & (solve_targets > 0.5) & ~censored_positions[:, None]
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
        shared = shared_positions[:, None].expand_as(predicted_crossings)
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
        crossing_target_count = int(
            ((solved > 0.5) & eligible & torch.isfinite(crossing_targets[:, 0])).sum().item()
        )
        move_target_count = int(
            ((solved > 0.5) & eligible & torch.isfinite(move_targets[:, 0])).sum().item()
        )
        monotonic_weight = float(getattr(network, "auxiliary_budget_monotonic_weight", 0.0))
        budget_channel = getattr(network, "objective_budget_channel", None)
        if monotonic_weight > 0.0 and budget_channel is not None:
            lower = observations.clone()
            higher = observations.clone()
            current = observations[:, budget_channel, :, :].clamp(0.0, 1.0)
            lower[:, budget_channel, :, :] = (current - 0.25).clamp(0.0, 1.0)
            higher[:, budget_channel, :, :] = (current + 0.25).clamp(0.0, 1.0)
            paired = torch.cat([lower, higher], dim=0)
            paired_solve = _native_forward_with_auxiliary(network, paired, optimized_parameter_ids)[
                2
            ][0]
            lower_logits, higher_logits = paired_solve.chunk(2, dim=0)
            margin = float(getattr(network, "auxiliary_budget_monotonic_margin", 0.0))
            monotonic_loss = F.relu(lower_logits - higher_logits + margin).mean()
        auxiliary_loss = (
            solve_loss + crossings_loss + moves_loss + monotonic_weight * monotonic_loss
        )
        if bool(eligible.any()):
            mean_probability = solve_logits.sigmoid().mean(dim=1)
            solve_brier = ((mean_probability[eligible] - conditional_solved[eligible]) ** 2).mean()
            composed = network.composed_auxiliary_value(observations, auxiliary)
            shadow_mae = (composed[eligible] - values.detach()[eligible]).abs().mean()
    relation = (
        network.regularization_loss()
        if hasattr(network, "regularization_loss")
        else torch.zeros((), device=device)
    )
    auxiliary_weight = float(getattr(network, "auxiliary_loss_weight", 0.0))
    loss = (
        p_loss
        + v_loss
        + float(relative_trajectory_weight) * relative_policy_loss
        + auxiliary_weight * auxiliary_loss
        + preservation_weight * (preservation_policy + preservation_value)
        + 0.1 * relation
    )
    # The option controller is attached after the native optimizer is created.
    # Clear the whole network so stale adapter gradients cannot accumulate, then
    # clip only the parameters this optimizer can actually update.
    network.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(optimized_parameters, 5.0)
    optimizer.step()
    replay_strata: dict[str, int] = {}
    for row in replay.last_collaboration_sample_trace:
        stratum = str(row.get("requested_stratum", "unknown"))
        replay_strata[stratum] = replay_strata.get(stratum, 0) + int(row.get("positions", 0))
    return {
        "loss": float(loss.item()),
        "policy": float(p_loss.item()),
        "relative_policy": float(relative_policy_loss.item()),
        "relative_policy_targets": float(relative_mask.sum().item()),
        "value": float(v_loss.item()),
        "auxiliary": float(auxiliary_loss.item()),
        "solve": float(solve_loss.item()),
        "crossings": float(crossings_loss.item()),
        "moves": float(moves_loss.item()),
        "budget_monotonic": float(monotonic_loss.item()),
        "solve_brier": float(solve_brier.item()),
        "shadow_mae": float(shadow_mae.item()),
        "preservation_policy": float(preservation_policy.item()),
        "preservation_value": float(preservation_value.item()),
        "relation": float(relation.item()),
        "policy_value_targets": float(native_targets.sum().item()),
        "solve_targets": float(eligible.sum().item()) if auxiliary is not None else 0.0,
        "crossing_targets": float(crossing_target_count),
        "move_targets": float(move_target_count),
        "replay_success_fraction": float((solved > 0.5).float().mean().item()),
        "replay_censored_fraction": float(censored_positions.float().mean().item()),
        "replay_shared_fraction": float(shared_positions.float().mean().item()),
        "replay_unique_representations": float(
            len(
                {
                    str(getattr(position, "representation_id", ""))
                    for position in batch
                    if str(getattr(position, "representation_id", ""))
                }
            )
        ),
        "replay_mean_position_uses": float(
            np.mean([int(getattr(position, "replay_position_uses", 0)) for position in batch])
        ),
        "replay_current_success_positions": float(replay_strata.get("current-success", 0)),
        "replay_rehearsal_success_positions": float(replay_strata.get("rehearsal-success", 0)),
        "replay_ordinary_failure_positions": float(replay_strata.get("ordinary-failure", 0)),
        "replay_budget_censored_failure_positions": float(
            replay_strata.get("budget-censored-failure", 0)
        ),
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
