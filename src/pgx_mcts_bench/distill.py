"""Behavioural distillation from the parallel braid agent into serial students.

The teacher and students do not share an action space.  A teacher edit at a global
word position is represented as a short option: shortest head shifts followed by
the same underlying braid edit.  Shift targets are one-hot; at the destination,
the teacher MCTS visit distribution is projected onto the edits executable there.
"""

from __future__ import annotations

import hashlib
import json
import signal
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pgx_mcts_bench.data import Position, ReplayBuffer
from pgx_mcts_bench.game import BraidUnknotGame, make_game
from pgx_mcts_bench.ladder import STAGES, Candidate, _config, candidates
from pgx_mcts_bench.networks import load_policy_value_state_dict, make_braid_network
from pgx_mcts_bench.search import NeuralMCTS
from pgx_mcts_bench.serial_braid import SerialBraidGame
from pgx_mcts_bench.training import train_alphazero_step

STUDENT_NAMES = ("d-head128-u1", "d-gru128-u1", "d-fsa32-u1", "d-tape4-u1")


@dataclass
class DistillReport:
    teacher: str
    teacher_sha256: str
    episodes: int
    teacher_positions: int
    mapped_positions: int
    skipped_positions: int
    serial_samples: dict[str, int]
    mean_option_length: dict[str, float]
    solved_episodes: int
    training_steps: dict[str, int]
    internal_horizon: int
    option_beam_width: int
    final_option_loss: dict[str, float]
    elapsed_seconds: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate(name: str) -> Candidate:
    try:
        return next(candidate for candidate in candidates() if candidate.name == name)
    except StopIteration as error:
        raise ValueError(f"Unknown distillation student {name}") from error


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


def _episode_samples(
    teacher_game: BraidUnknotGame,
    teacher_search: NeuralMCTS,
    student_games: dict[str, SerialBraidGame],
    seed: int,
    rng: np.random.Generator,
) -> tuple[dict[str, list[Position]], dict[str, int], bool]:
    transition = teacher_game.reset(seed)
    raw: dict[str, list[tuple[Position, int]]] = {name: [] for name in student_games}
    controllers: dict[str, Any | None] = {name: None for name in student_games}
    teacher_positions = 0
    skipped = 0
    while not transition.terminated:
        result = teacher_search.run(
            transition.state,
            transition.observation,
            transition.legal_actions,
            rng,
            temperature=0.0,
            add_root_noise=False,
        )
        teacher_positions += 1
        pgx_state = transition.state
        length = int(np.asarray(pgx_state._word).astype(bool).sum())
        mapped_any = False
        for name, game in student_games.items():
            previous = controllers[name]
            serial = _view_at(
                game,
                pgx_state,
                0 if previous is None else previous.head,
                game._no_tape() if previous is None else previous.tape,  # noqa: SLF001
                registers=None if previous is None else previous.registers,
                colours=None if previous is None else previous.colours,
                colour=0 if previous is None else previous.colour,
                semantic_moves=0 if previous is None else previous.semantic_moves,
            )
            destination = _best_destination(
                game, result.action, serial.state.head, length
            )
            if destination is None:
                continue
            route, head, final_action = destination
            if len(route) + 1 > int(np.asarray(pgx_state._budget)):
                # The teacher has too little real action budget left to pay for
                # serial navigation plus its requested edit.
                continue
            for shift in route:
                policy = np.zeros(game.num_actions, dtype=np.float32)
                policy[shift] = 1.0
                raw[name].append(
                    (
                        Position(
                            serial.observation,
                            serial.legal_actions,
                            policy,
                            shift,
                            serial.player,
                            role=1,
                            outcome=float(result.root_value),
                            episode_seed=seed,
                            option_state=serial.state,
                            target_external_action=result.action,
                        ),
                        len(route),
                    )
                )
                serial = game.step(serial.state, shift)
            assert serial.state.head == head
            raw[name].append(
                (
                    Position(
                        serial.observation,
                        serial.legal_actions,
                        _project_policy(
                            game, result.policy, head, length, final_action
                        ),
                        final_action,
                        serial.player,
                        role=1,
                        outcome=float(result.root_value),
                        episode_seed=seed,
                        option_state=serial.state,
                        target_external_action=result.action,
                    ),
                    len(route) + 1,
                )
            )
            controllers[name] = game.step(serial.state, final_action).state
            mapped_any = True
        if not mapped_any:
            skipped += 1
        transition = teacher_game.step(transition.state, result.action)

    final = transition.state
    solved = bool((np.asarray(final._word) == 0).all()) and int(np.asarray(final._n)) == 1
    crossings = float(np.asarray(final._crossing_changes))
    teacher_moves = float(
        teacher_game.config.simplify_budget - int(np.asarray(final._budget))
    )
    output: dict[str, list[Position]] = {}
    option_lengths: dict[str, int] = {}
    for name, rows in raw.items():
        navigation = sum(max(option_length - 1, 0) for _, option_length in rows)
        # Rows repeat option length; use the actual number of shift-labelled rows.
        navigation = sum(
            int(np.count_nonzero(position.policy[student_games[name]._shift_base :]))
            for position, _ in rows
        )
        for position, _ in rows:
            position.solved = float(solved)
            position.final_crossing_changes = crossings
            position.final_moves = teacher_moves
            position.final_native_plies = teacher_moves + navigation
            position.final_internal_plies = float(navigation)
        output[name] = [position for position, _ in rows]
        option_lengths[name] = len(rows)
    return output, {"teacher": teacher_positions, "skipped": skipped, **option_lengths}, solved


def run_distillation(
    teacher_checkpoint: Path,
    output: Path,
    *,
    episodes: int = 128,
    train_steps: int = 2_000,
    seed: int = 0,
    device: str = "cpu",
    internal_horizon: int = 5,
    option_beam_width: int = 8,
    option_batch_size: int = 4,
    log=print,
) -> DistillReport:
    """Generate teacher options, train all four students, and save ladder seeds."""
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    termination = {"requested": False}

    def request_termination(_signum, _frame):
        termination["requested"] = True

    signal.signal(signal.SIGTERM, request_termination)
    teacher_saved = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    teacher_candidate = _candidate("u1-puct")
    teacher_config = _config(teacher_candidate, STAGES[0], seed, device)
    teacher_game = make_game(teacher_config.game)
    assert isinstance(teacher_game, BraidUnknotGame)
    teacher_network = make_braid_network(teacher_config.game, teacher_config.model).to(device)
    load_policy_value_state_dict(teacher_network, teacher_saved["network"])
    teacher_network.eval()
    teacher_search = NeuralMCTS(teacher_game, teacher_network, teacher_config.search, device)

    student_candidates = {
        name: replace(_candidate(name), serial_internal_horizon=internal_horizon)
        for name in STUDENT_NAMES
    }
    student_configs = {
        name: _config(candidate, STAGES[0], seed, device)
        for name, candidate in student_candidates.items()
    }
    student_games = {name: make_game(config.game) for name, config in student_configs.items()}
    assert all(isinstance(game, SerialBraidGame) for game in student_games.values())
    records: dict[str, list[list[Position]]] = {name: [] for name in STUDENT_NAMES}
    teacher_positions = skipped = solved_episodes = 0
    option_totals = {name: 0 for name in STUDENT_NAMES}
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        # Cycle through every rung known to the teacher instead of distilling only
        # its frontier.  The teacher's promoted stage list is the auditable scope.
        promoted = [row for row in teacher_saved.get("stages", []) if row.get("promoted")]
        stage = (
            promoted[episode % len(promoted)]
            if promoted
            else {"source": STAGES[0][0], "scramble": STAGES[0][1]}
        )
        config = _config(
            teacher_candidate,
            (stage["source"], int(stage["scramble"])),
            seed + episode,
            device,
        )
        teacher_game = make_game(config.game)
        assert isinstance(teacher_game, BraidUnknotGame)
        teacher_search = NeuralMCTS(teacher_game, teacher_network, config.search, device)
        samples, counts, solved = _episode_samples(
            teacher_game, teacher_search, student_games, seed + episode, rng
        )
        teacher_positions += counts["teacher"]
        skipped += counts["skipped"]
        solved_episodes += int(solved)
        for name, positions in samples.items():
            records[name].append(positions)
            option_totals[name] += counts[name]
        log(
            f"dataset {episode + 1}/{episodes}: teacher positions={teacher_positions}, "
            f"solved={solved_episodes}, skipped={skipped}"
        )
        if termination["requested"]:
            torch.save(records, output / "dataset.interrupt.pt")
            raise SystemExit(143)

    torch.save(records, output / "dataset.pt")
    completed_steps: dict[str, int] = {}
    final_option_loss: dict[str, float] = {}
    for name in STUDENT_NAMES:
        config = student_configs[name]
        network = make_braid_network(config.game, config.model).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
        replay = ReplayBuffer(
            max(20_000, sum(map(len, records[name]))), np.random.default_rng(seed)
        )
        for record in records[name]:
            replay.add(record)
        start_step = 0
        option_rng = np.random.default_rng(seed + 91_337)
        option_loss = float("nan")
        progress_path = output / name / "distill-progress.pt"
        if progress_path.exists():
            saved = torch.load(progress_path, map_location=device, weights_only=False)
            network.load_state_dict(saved["network"])
            optimizer.load_state_dict(saved["optimizer"])
            start_step = int(saved["step"])
        for step in range(start_step, train_steps):
            metrics = train_alphazero_step(network, optimizer, replay, 64, torch.device(device))
            option_loss = train_bounded_option_step(
                network,
                optimizer,
                student_games[name],
                replay,
                option_rng,
                batch_size=option_batch_size,
                horizon=internal_horizon,
                beam_width=option_beam_width,
                device=torch.device(device),
            )
            if (step + 1) % 100 == 0 or step == start_step:
                log(
                    f"{name} step {step + 1}/{train_steps}: loss={metrics['loss']:.4f}, "
                    f"policy={metrics['policy']:.4f}, solve={metrics['solve']:.4f}, "
                    f"cc={metrics['crossings']:.4f}, moves={metrics['moves']:.4f}, "
                    f"option={option_loss:.4f}"
                )
            if (step + 1) % 100 == 0 or termination["requested"]:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "network": network.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step + 1,
                    },
                    progress_path,
                )
            if termination["requested"]:
                raise SystemExit(143)
        checkpoint_dir = output / name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "candidate": name,
                "candidate_spec": asdict(student_candidates[name]),
                "stages": [],
                "highest_stage": -1,
                "distilled_from": str(teacher_checkpoint),
                "teacher_sha256": _sha256(teacher_checkpoint),
                "distill_steps": train_steps,
            },
            checkpoint_dir / f"{name}.pt",
        )
        completed_steps[name] = train_steps
        final_option_loss[name] = option_loss

    report = DistillReport(
        teacher=str(teacher_checkpoint),
        teacher_sha256=_sha256(teacher_checkpoint),
        episodes=episodes,
        teacher_positions=teacher_positions,
        mapped_positions=teacher_positions - skipped,
        skipped_positions=skipped,
        serial_samples={name: sum(map(len, records[name])) for name in STUDENT_NAMES},
        mean_option_length={
            name: option_totals[name] / max(teacher_positions - skipped, 1)
            for name in STUDENT_NAMES
        },
        solved_episodes=solved_episodes,
        training_steps=completed_steps,
        internal_horizon=internal_horizon,
        option_beam_width=option_beam_width,
        final_option_loss=final_option_loss,
        elapsed_seconds=time.perf_counter() - started,
    )
    (output / "distill-report.json").write_text(json.dumps(asdict(report), indent=2) + "\n")
    return report
