"""Bounded option-route learning for serial braid scientists."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pgx_mcts_bench.data import Position, ReplayBuffer
from pgx_mcts_bench.serial_braid import SerialBraidGame

STUDENT_NAMES = ("d-head128-u1", "d-gru128-u1", "d-fsa32-u1", "d-tape4-u1")


def _view_at(
    game: SerialBraidGame,
    pgx_state: Any,
    head: int,
    tape: np.ndarray,
    *,
    registers: np.ndarray | None = None,
    colours: np.ndarray | None = None,
    colour: int = 0,
    internal_steps: int = 0,
    semantic_moves: int = 0,
):
    return game._view(  # noqa: SLF001 - deliberate adapter between action spaces
        pgx_state,
        head,
        game._no_registers() if registers is None else registers,  # noqa: SLF001
        game._no_colours() if colours is None else colours,  # noqa: SLF001
        colour,
        tape,
        reward=0.0,
        internal_steps=internal_steps,
        semantic_moves=semantic_moves,
    )


def _option_fingerprint(state: Any) -> tuple[Any, ...]:
    return (
        int(state.head),
        state.registers.tobytes(),
        state.colours.tobytes(),
        int(state.colour),
        state.tape.tobytes(),
    )


def _target_actions(
    game: SerialBraidGame, state: Any, teacher_action: int
) -> list[int]:
    length = int(np.asarray(state.pgx._word).astype(bool).sum())
    legal = game._legal(state.pgx, state.head, state.internal_steps)  # noqa: SLF001
    return [
        action
        for action in range(game._shift_base)  # noqa: SLF001
        if legal[action]
        and game.underlying_action(action, state.head, length) == teacher_action
    ]


def bounded_option_loss(
    network,
    game: SerialBraidGame,
    positions: list[Position],
    *,
    horizon: int = 5,
    beam_width: int = 8,
    device: torch.device,
) -> Tensor:
    """Negative log probability of teacher edit through <=H internal actions.

    Beam membership is selected with detached probabilities, while every path's
    probability remains differentiable. The shortest navigation action is always
    retained beside student-preferred actions, so a randomly initialized policy
    still receives a route to the target instead of an all-zero signal.
    """
    losses: list[Tensor] = []
    for position in positions:
        if position.option_state is None or position.target_external_action < 0:
            continue
        branches: list[tuple[Any, Tensor, frozenset[tuple[Any, ...]]]] = [
            (
                position.option_state,
                torch.zeros((), device=device),
                frozenset({_option_fingerprint(position.option_state)}),
            )
        ]
        successes: list[Tensor] = []
        for depth in range(horizon + 1):
            observations = np.stack(
                [
                    game._view(  # noqa: SLF001
                        state.pgx,
                        state.head,
                        state.registers,
                        state.colours,
                        state.colour,
                        state.tape,
                        reward=0.0,
                        internal_steps=state.internal_steps,
                        semantic_moves=state.semantic_moves,
                    ).observation
                    for state, _, _ in branches
                ]
            )
            tensor = (
                torch.from_numpy(observations)
                .permute(0, 3, 1, 2)
                .float()
                .to(device)
            )
            logits, _ = network(tensor)
            candidates: list[
                tuple[Any, Tensor, frozenset[tuple[Any, ...]], float, bool]
            ] = []
            for branch_index, (state, path_logp, seen) in enumerate(branches):
                legal = game._legal(  # noqa: SLF001
                    state.pgx, state.head, state.internal_steps
                )
                masked = logits[branch_index].masked_fill(
                    ~torch.from_numpy(legal).to(device), -torch.inf
                )
                logp = F.log_softmax(masked, dim=0)
                targets = _target_actions(game, state, position.target_external_action)
                if targets:
                    successes.append(path_logp + torch.logsumexp(logp[targets], dim=0))
                if depth == horizon or state.internal_steps >= horizon:
                    continue
                internal = [
                    int(action)
                    for action in np.flatnonzero(legal)
                    if action >= game._shift_base  # noqa: SLF001
                ]
                if not internal:
                    continue
                chosen = set(
                    sorted(internal, key=lambda action: float(logp[action].detach()), reverse=True)[
                        :beam_width
                    ]
                )
                length = int(np.asarray(state.pgx._word).astype(bool).sum())
                destination = _best_destination(
                    game, position.target_external_action, state.head, length
                )
                oracle_action = None
                if destination is not None and destination[0]:
                    oracle_action = destination[0][0]
                    chosen.add(oracle_action)
                for action in chosen:
                    transition = game.step(state, action)
                    fingerprint = _option_fingerprint(transition.state)
                    if fingerprint in seen or transition.terminated:
                        continue
                    next_logp = path_logp + logp[action]
                    candidates.append(
                        (
                            transition.state,
                            next_logp,
                            seen | {fingerprint},
                            float(next_logp.detach()),
                            action == oracle_action,
                        )
                    )
            if not candidates:
                break
            candidates.sort(key=lambda item: item[3], reverse=True)
            mandatory = [item for item in candidates if item[4]][:beam_width]
            optional = [item for item in candidates if not item[4]]
            selected = mandatory + optional[: beam_width - len(mandatory)]
            branches = [
                (state, logp, seen) for state, logp, seen, _, _ in selected
            ]
        if successes:
            losses.append(-torch.logsumexp(torch.stack(successes), dim=0))
    if not losses:
        parameter = next(network.parameters())
        return parameter.sum() * 0.0
    return torch.stack(losses).mean()


def stable_option_route_loss(
    network,
    game: SerialBraidGame,
    positions: list[Position],
    *,
    horizon: int = 5,
    device: torch.device,
) -> Tensor:
    """Teacher-force a canonical option whose route never depends on logits.

    For each certified semantic edit, choose the deterministic shortest route
    produced by ``_best_destination`` and then execute the corresponding local
    external action. Unlike the legacy beam objective, neither target selection
    nor route membership can change when the student policy changes.
    """
    targets = _stable_option_route_targets(game, positions, horizon=horizon)
    if not targets:
        parameter = next(network.parameters())
        return parameter.sum() * 0.0
    observations = np.stack([item[0] for item in targets])
    legal_actions = np.stack([item[1] for item in targets])
    actions = torch.tensor([item[2] for item in targets], device=device)
    groups = [item[3] for item in targets]
    tensor = torch.from_numpy(observations).permute(0, 3, 1, 2).float().to(device)
    legal = torch.from_numpy(legal_actions).to(device)
    logits, _ = network(tensor)
    logp = F.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=1)
    action_losses = -logp[torch.arange(actions.shape[0], device=device), actions]
    losses = [
        action_losses[
            torch.tensor(
                [index for index, group in enumerate(groups) if group == group_index],
                device=device,
            )
        ].sum()
        for group_index in range(max(groups) + 1)
    ]
    return torch.stack(losses).mean()


def _stable_option_route_targets(
    game: SerialBraidGame,
    positions: list[Position],
    *,
    horizon: int,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """Materialize canonical route states independently of network outputs."""
    targets: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    group = 0
    for position in positions:
        state = position.option_state
        if state is None or position.target_external_action < 0:
            continue
        length = int(np.asarray(state.pgx._word).astype(bool).sum())
        destination = _best_destination(
            game, position.target_external_action, state.head, length
        )
        if destination is None:
            continue
        route, _, external_action = destination
        remaining_horizon = max(horizon - int(state.internal_steps), 0)
        if len(route) > remaining_horizon:
            continue
        actions = [*route, external_action]
        valid = True
        pending: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        for action in actions:
            transition = game._view(  # noqa: SLF001 - exact option state view
                state.pgx,
                state.head,
                state.registers,
                state.colours,
                state.colour,
                state.tape,
                reward=0.0,
                internal_steps=state.internal_steps,
                semantic_moves=state.semantic_moves,
            )
            if not transition.legal_actions[action]:
                valid = False
                break
            pending.append(
                (transition.observation, transition.legal_actions, action, group)
            )
            state = game.step(state, action).state
        if valid and pending:
            targets.extend(pending)
            group += 1
    return targets


def train_bounded_option_step(
    network,
    optimizer: torch.optim.Optimizer,
    game: SerialBraidGame,
    replay: ReplayBuffer,
    rng: np.random.Generator,
    *,
    batch_size: int,
    horizon: int,
    beam_width: int,
    device: torch.device,
    positions: list[Position] | None = None,
    learning_rate_scale: float = 1.0,
    adapter_only: bool = False,
    stable_routes: bool | None = None,
    preservation_positions: list[Position] | None = None,
    route_gate_weight: float = 0.0,
    off_route_kl_weight: float = 0.0,
    off_route_gate_weight: float = 0.0,
    diagnostics: dict[str, float | str] | None = None,
) -> float:
    if learning_rate_scale <= 0.0:
        raise ValueError("learning_rate_scale must be positive")
    if positions is None:
        # Adapter-only training is the collaboration policy path. Make stale
        # filtering the safe default so a caller cannot accidentally resample
        # an obsolete donation from the full replay buffer.
        records = (
            replay.active_distillation_records() if adapter_only else replay.games
        )
        eligible = [
            position
            for record in records
            for position in record
            if position.option_state is not None and position.target_external_action >= 0
        ]
        if not eligible:
            return 0.0
        picks = rng.integers(0, len(eligible), size=batch_size)
        batch = [eligible[int(index)] for index in picks]
    else:
        batch = [
            position
            for position in positions
            if position.option_state is not None and position.target_external_action >= 0
        ]
        if not batch:
            return 0.0
    network.train()
    adapter = getattr(network, "option_policy_adapter", None)
    gate = getattr(network, "option_policy_gate", None)
    frozen_requires_grad: list[tuple[nn.Parameter, bool]] = []
    if adapter_only and adapter is None:
        raise ValueError("adapter-only option training requires an attached adapter")
    if adapter_only:
        # The base scientist is a frozen feature provider for sharing. Keep its
        # BatchNorm buffers fixed as well as its parameters; GroupNorm inside
        # the adapter has no running state and remains fully trainable.
        network.eval()
        adapter.train()
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        sharing_parameters = list(adapter.parameters())
        if gate is not None:
            gate.train()
            sharing_parameters.extend(gate.parameters())
        sharing_parameter_ids = {id(parameter) for parameter in sharing_parameters}
        if optimized != sharing_parameter_ids:
            raise ValueError(
                "adapter-only optimizer must contain exactly adapter and gate parameters"
            )
        for parameter in network.parameters():
            if id(parameter) not in sharing_parameter_ids:
                frozen_requires_grad.append((parameter, parameter.requires_grad))
                parameter.requires_grad_(False)
    if bool(getattr(network, "freeze_batchnorm_stats", False)):
        for module in network.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    if stable_routes is None:
        stable_routes = adapter_only
    loss_function = stable_option_route_loss if stable_routes else bounded_option_loss
    loss_kwargs = {"horizon": horizon, "device": device}
    if not stable_routes:
        loss_kwargs["beam_width"] = beam_width
    option_loss = loss_function(network, game, batch, **loss_kwargs)
    route_gate_loss = option_loss.sum() * 0.0
    if adapter_only and gate is not None and route_gate_weight > 0.0:
        route_targets = _stable_option_route_targets(game, batch, horizon=horizon)
        if route_targets:
            route_observations = torch.from_numpy(
                np.stack([item[0] for item in route_targets])
            ).permute(0, 3, 1, 2).float().to(device)
            route_gate_loss = -gate(route_observations).clamp_min(1e-6).log().mean()

    off_route_kl = option_loss.sum() * 0.0
    off_route_gate = option_loss.sum() * 0.0
    if adapter_only and preservation_positions and (
        off_route_kl_weight > 0.0 or off_route_gate_weight > 0.0
    ):
        off_route_observations = torch.from_numpy(
            np.stack([position.observation for position in preservation_positions])
        ).permute(0, 3, 1, 2).float().to(device)
        off_route_legal = torch.from_numpy(
            np.stack([position.legal_actions for position in preservation_positions])
        ).to(device)
        combined_logits, _ = network(off_route_observations)
        applied_residual, off_route_probabilities = network.option_policy_components(
            off_route_observations
        )
        base_logits = combined_logits - applied_residual
        floor = torch.finfo(combined_logits.dtype).min
        combined_logp = F.log_softmax(
            combined_logits.masked_fill(~off_route_legal, floor), dim=-1
        )
        base_logp = F.log_softmax(
            base_logits.detach().masked_fill(~off_route_legal, floor), dim=-1
        )
        off_route_kl = (
            base_logp.exp() * (base_logp - combined_logp)
        ).sum(dim=-1).mean()
        off_route_gate = off_route_probabilities.square().mean()
    preservation = option_loss.sum() * 0.0
    preservation_weight = 0.0 if adapter_only else float(
        getattr(network, "policy_value_preservation_weight", 0.0)
    )
    teacher = getattr(network, "_policy_value_preservation_teacher", None)
    if preservation_weight > 0.0 and teacher is not None:
        observations = np.stack(
            [
                game._view(  # noqa: SLF001 - exact option-start observation
                    position.option_state.pgx,
                    position.option_state.head,
                    position.option_state.registers,
                    position.option_state.colours,
                    position.option_state.colour,
                    position.option_state.tape,
                    reward=0.0,
                    internal_steps=position.option_state.internal_steps,
                    semantic_moves=position.option_state.semantic_moves,
                ).observation
                for position in batch
            ]
        )
        tensor = torch.from_numpy(observations).permute(0, 3, 1, 2).float().to(device)
        logits, values = network(tensor)
        teacher.eval()
        with torch.no_grad():
            teacher_logits, teacher_values = teacher(tensor)
        legal = torch.from_numpy(np.stack([position.legal_actions for position in batch])).to(
            device
        )
        floor = torch.finfo(logits.dtype).min
        student_logp = F.log_softmax(logits.masked_fill(~legal, floor), dim=-1)
        teacher_logp = F.log_softmax(teacher_logits.masked_fill(~legal, floor), dim=-1)
        preservation = (
            (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1).mean()
            + F.mse_loss(values, teacher_values)
        )
    loss = (
        option_loss
        + preservation_weight * preservation
        + route_gate_weight * route_gate_loss
        + off_route_kl_weight * off_route_kl
        + off_route_gate_weight * off_route_gate
    )
    # Native training may have left gradients on the frozen base while this
    # optimizer owns only the option modules. Clear the entire network so the
    # two optimizers never exchange stale gradients.
    network.zero_grad(set_to_none=True)
    loss.backward()
    clipped_parameters = sharing_parameters if adapter_only else network.parameters()
    nn.utils.clip_grad_norm_(clipped_parameters, 5.0)
    original_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    try:
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * learning_rate_scale
        optimizer.step()
    finally:
        for group, learning_rate in zip(
            optimizer.param_groups, original_learning_rates, strict=True
        ):
            group["lr"] = learning_rate
    if diagnostics is not None:
        with torch.no_grad():
            post_loss = loss_function(network, game, batch, **loss_kwargs)
        diagnostics.update(
            {
                "target": "canonical-shortest-route" if stable_routes else "policy-beam",
                "loss_before": float(option_loss.item()),
                "loss_after": float(post_loss.item()),
                "loss_delta": float(post_loss.item() - option_loss.item()),
                "positions": float(len(batch)),
                "action_targets": float(
                    len(_stable_option_route_targets(game, batch, horizon=horizon))
                    if stable_routes
                    else len(batch)
                ),
                "route_gate_loss": float(route_gate_loss.item()),
                "off_route_kl": float(off_route_kl.item()),
                "off_route_gate": float(off_route_gate.item()),
                "off_route_positions": float(len(preservation_positions or ())),
            }
        )
    for parameter, requires_grad in frozen_requires_grad:
        parameter.requires_grad_(requires_grad)
    return float(option_loss.item())


def _shift_actions(game: SerialBraidGame) -> list[int]:
    # For tape students use WRITE(0), the neutral initial symbol.  Distillation
    # teaches navigation and edits; inventing semantic tape labels would inject a
    # hand-written oracle rather than transfer teacher knowledge.
    return [
        game._shift_base + 2 * index * game._tape_variants + direction  # noqa: SLF001
        for index in range(len(game.strides))
        for direction in (0, 1)
    ]


def _route(game: SerialBraidGame, start: int, goal: int, length: int) -> list[int]:
    if length <= 1 or start % length == goal % length:
        return []
    queue = deque([(start % length, [])])
    seen = {start % length}
    actions = _shift_actions(game)
    while queue:
        head, path = queue.popleft()
        for action in actions:
            displacement = game.shift_of(action)
            assert displacement is not None
            nxt = (head + displacement) % length
            if nxt == goal % length:
                return path + [action]
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [action]))
    raise RuntimeError(f"Cannot route serial head from {start} to {goal} on length {length}")


def _serial_action_for(
    game: SerialBraidGame, teacher_action: int, head: int, length: int
) -> int | None:
    for action in range(game._shift_base):  # noqa: SLF001
        if game.underlying_action(action, head, length) == teacher_action:
            return action
    return None


def _best_destination(
    game: SerialBraidGame, teacher_action: int, start: int, length: int
) -> tuple[list[int], int, int] | None:
    best: tuple[list[int], int, int] | None = None
    for head in range(max(length, 1)):
        action = _serial_action_for(game, teacher_action, head, length)
        if action is None:
            continue
        route = _route(game, start, head, length)
        if best is None or len(route) < len(best[0]):
            best = route, head, action
    return best


def _project_policy(
    game: SerialBraidGame,
    teacher_policy: np.ndarray,
    head: int,
    length: int,
    fallback: int,
) -> np.ndarray:
    policy = np.zeros(game.num_actions, dtype=np.float32)
    for teacher_action, probability in enumerate(teacher_policy):
        if probability <= 0:
            continue
        serial_action = _serial_action_for(game, teacher_action, head, length)
        if serial_action is not None:
            policy[serial_action] += float(probability)
    total = float(policy.sum())
    if total <= 0:
        policy[fallback] = 1.0
    else:
        policy /= total
    return policy
