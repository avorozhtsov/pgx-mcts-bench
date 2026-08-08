from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pgx_mcts_bench.config import AnyGameConfig, BraidGameConfig, ModelConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.body(x))


class Representation(nn.Module):
    def __init__(self, game: AnyGameConfig, model: ModelConfig, output_channels: int):
        super().__init__()
        blocks = [ResidualBlock(output_channels) for _ in range(model.residual_blocks)]
        self.net = nn.Sequential(
            nn.Conv2d(game.observation_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            *blocks,
        )

    def forward(self, observation: Tensor) -> Tensor:
        return self.net(observation)


class PredictionHead(nn.Module):
    def __init__(
        self,
        channels: int,
        game: AnyGameConfig,
        *,
        include_legal: bool,
        include_terminal: bool,
    ):
        super().__init__()
        cells = game.cells
        self.include_legal = include_legal
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * cells, game.action_size),
        )
        self.value = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cells, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self.legal = (
            nn.Sequential(
                nn.Conv2d(channels, 1, 1),
                nn.Flatten(),
                nn.Linear(cells, game.action_size),
            )
            if include_legal
            else None
        )
        self.terminal = (
            nn.Sequential(
                nn.Conv2d(channels, 1, 1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(cells, 1),
            )
            if include_terminal
            else None
        )
        if self.terminal is not None:
            # At initialization imagined states should continue unless the
            # model has evidence for termination.
            nn.init.constant_(self.terminal[-1].bias, -3.0)

    def forward(
        self, hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        legal_logits = self.legal(hidden) if self.legal is not None else None
        terminal_logits = (
            self.terminal(hidden).squeeze(-1) if self.terminal is not None else None
        )
        return (
            self.policy(hidden),
            self.value(hidden).squeeze(-1),
            legal_logits,
            terminal_logits,
        )


class PolicyValueNet(nn.Module):
    """Anything the AlphaZero search path accepts: observation -> (policy, value).

    `NeuralMCTS` dispatches on this rather than on `AlphaZeroNet` so that a game
    can supply its own head shape without being mistaken for a MuZero network.
    """

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward_with_auxiliary(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor] | None]:
        policy, value = self(observation)
        return policy, value, None

    def composed_auxiliary_value(
        self,
        observation: Tensor,
        auxiliary: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        raise NotImplementedError


class AlphaZeroNet(PolicyValueNet):
    def __init__(self, game: AnyGameConfig, model: ModelConfig):
        super().__init__()
        self.representation = Representation(game, model, model.channels)
        self.prediction = PredictionHead(
            model.channels,
            game,
            include_legal=False,
            include_terminal=False,
        )

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        policy, value, _, _ = self.prediction(self.representation(observation))
        return policy, value


class FiLM(nn.Module):
    """Feature-wise Linear Modulation on log(A/B)  (Perez et al., 2018).

    Appending the ratio as an input channel makes the network *free* to use it,
    and a small trunk will most likely learn one compromise policy that is
    mediocre at both extremes. FiLM instead makes the conditioning
    **multiplicative**: a per-channel gain and shift generated from log(A/B),
    applied after each residual block. At log(A/B) = -5 and +5 the gains gate
    different feature subsets, so one set of weights behaves like genuinely
    different networks at the two ends of the Pareto front -- which is the point
    of conditioning at all.
    """

    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 2 * channels)
        )
        # Start as the identity, so conditioning is learned rather than imposed.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.channels = channels

    def forward(self, hidden: Tensor, log_ratio: Tensor) -> Tensor:
        gamma, beta = self.net(log_ratio[:, None]).chunk(2, dim=1)
        return (1.0 + gamma)[:, :, None, None] * hidden + beta[:, :, None, None]


def _inverse_softplus(value: float) -> float:
    return float(torch.log(torch.expm1(torch.tensor(value))).item())


class AuxiliaryValueMember(nn.Module):
    """One bootstrap member for solve probability and conditional costs."""

    def __init__(self, features: int, width: int):
        super().__init__()
        hidden = max(width // 2, 16)
        self.solve = nn.Sequential(
            nn.Linear(features, width),
            nn.SiLU(),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.cost = nn.Sequential(
            nn.Linear(features, width),
            nn.SiLU(),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        self.cost_budget = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, 2),
        )
        self.solve_conditioning = nn.Sequential(
            nn.Linear(features + 4, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        # Neutral solve prior and modest positive cost priors. Small weights keep
        # members distinct without injecting large random outputs at migration.
        nn.init.normal_(self.solve[-1].weight, std=1e-3)
        nn.init.zeros_(self.solve[-1].bias)
        nn.init.normal_(self.cost[-1].weight, std=1e-3)
        nn.init.zeros_(self.cost_budget[-1].weight)
        nn.init.zeros_(self.cost_budget[-1].bias)
        nn.init.zeros_(self.solve_conditioning[-1].weight)
        nn.init.zeros_(self.solve_conditioning[-1].bias)
        with torch.no_grad():
            self.cost[-1].bias.copy_(
                torch.tensor([_inverse_softplus(1.0), _inverse_softplus(8.0)])
            )

    def forward(
        self,
        solve_features: Tensor,
        cost_features: Tensor | None = None,
        conditioning: tuple[Tensor, Tensor, float] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if cost_features is None:
            cost_features = solve_features
        cost_logits = self.cost(cost_features)
        if conditioning is not None:
            remaining_budget, ratio, scale = conditioning
            cost_logits = cost_logits + self.cost_budget(remaining_budget[:, None])
        costs = F.softplus(cost_logits)
        solve_logit = self.solve(solve_features).squeeze(-1)
        if conditioning is not None:
            remaining_budget, ratio, scale = conditioning
            # Costs enter numerically but are stop-gradient inputs: their own
            # supervised heads retain the meanings cc and moves rather than
            # becoming arbitrary hidden variables optimized by solve BCE.
            normalized_cc = costs[:, 0].detach() / scale
            normalized_moves = costs[:, 1].detach() / scale
            normalized_l = (
                ratio * costs[:, 0].detach() + costs[:, 1].detach()
            ) / ((ratio + 1.0) * scale)
            conditioned = torch.cat(
                [
                    solve_features,
                    remaining_budget[:, None],
                    normalized_cc[:, None],
                    normalized_moves[:, None],
                    normalized_l[:, None],
                ],
                dim=1,
            )
            solve_logit = solve_logit + self.solve_conditioning(conditioned).squeeze(-1)
        return solve_logit, costs[:, 0], costs[:, 1]


class AuxiliaryValueHeads(nn.Module):
    def __init__(self, features: int, width: int, members: int):
        super().__init__()
        if members < 1:
            raise ValueError("auxiliary_value_members must be positive")
        self.members = nn.ModuleList(
            AuxiliaryValueMember(features, width) for _ in range(members)
        )
        self.body_budget_skip = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, features),
        )
        self.legacy_budget_skip = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        for branch in (self.body_budget_skip, self.legacy_budget_skip):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(
        self,
        solve_features: Tensor,
        cost_features: Tensor | None = None,
        conditioning: tuple[Tensor, Tensor, float] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        outputs = [
            member(solve_features, cost_features, conditioning) for member in self.members
        ]
        return tuple(torch.stack(items, dim=1) for items in zip(*outputs, strict=True))  # type: ignore[return-value]


class BraidPolicyValueNet(PolicyValueNet):
    """Common shadow-critic plumbing for every braid representation."""

    def _set_auxiliary_training_controls(
        self, game: BraidGameConfig, model: ModelConfig
    ) -> None:
        self.auxiliary_loss_weight = model.auxiliary_value_loss_weight
        self.auxiliary_backprop = model.auxiliary_backprop_to_encoder
        self.auxiliary_solve_backprop = model.auxiliary_solve_backprop_to_encoder
        self.auxiliary_budget_monotonic_weight = model.auxiliary_budget_monotonic_weight
        self.auxiliary_budget_monotonic_margin = model.auxiliary_budget_monotonic_margin
        self.auxiliary_budget_conditioning = model.auxiliary_budget_conditioning
        self.freeze_batchnorm_stats = model.freeze_batchnorm_stats
        self.policy_value_preservation_weight = model.policy_value_preservation_weight
        self.objective_budget_channel = (
            game.observation_channels - 1 if game.objective_budget_channel else None
        )
        self.auxiliary_budget = float(game.simplify_budget)
        self.auxiliary_ratio_channel = 2 * (game.max_strands - 1) + 1 + 1 + 6
        self.use_auxiliary_value = model.use_auxiliary_value
        self.option_policy_adapter: OptionPolicyAdapter | None = None
        self.option_policy_gate: OptionPolicyGate | None = None
        self.option_adapter_enabled = True
        self._option_adapter_spec = (
            game.observation_channels,
            game.action_size,
            (
                game.observation_channels
                - int(game.objective_budget_channel)
                - 1
                if game.serial_internal_horizon
                else None
            ),
        )

    def attach_option_policy_adapter(
        self, *, width: int = 32, residual_blocks: int = 2
    ) -> OptionPolicyAdapter:
        """Attach the isolated sharing controller without changing logits."""
        if self.option_policy_adapter is None:
            channels, actions, internal_budget_channel = self._option_adapter_spec
            self.option_policy_adapter = OptionPolicyAdapter(
                channels,
                actions,
                internal_budget_channel=internal_budget_channel,
                width=width,
                residual_blocks=residual_blocks,
            )
        return self.option_policy_adapter

    def attach_option_policy_gate(
        self, *, width: int = 32, residual_blocks: int = 2, initial_probability: float = 0.1
    ) -> OptionPolicyGate:
        """Attach a conservative state-dependent applicability gate."""
        if self.option_policy_adapter is None:
            self.attach_option_policy_adapter()
        if self.option_policy_gate is None:
            channels, _, internal_budget_channel = self._option_adapter_spec
            self.option_policy_gate = OptionPolicyGate(
                channels,
                internal_budget_channel=internal_budget_channel,
                width=width,
                residual_blocks=residual_blocks,
                initial_probability=initial_probability,
            )
        return self.option_policy_gate

    def option_policy_components(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Return the applied adapter residual and its scalar state gate."""
        if self.option_policy_adapter is None:
            return (
                observation.new_zeros((observation.shape[0], self._option_adapter_spec[1])),
                observation.new_zeros((observation.shape[0], 1)),
            )
        residual = self.option_policy_adapter(observation)
        gate = (
            self.option_policy_gate(observation)
            if self.option_policy_gate is not None
            else observation.new_ones((observation.shape[0], 1))
        )
        return residual * gate, gate

    def _apply_option_policy_adapter(
        self, observation: Tensor, logits: Tensor
    ) -> Tensor:
        if self.option_policy_adapter is None or not self.option_adapter_enabled:
            return logits
        residual, _ = self.option_policy_components(observation)
        return logits + residual


    def _init_auxiliary(
        self, features: int, game: BraidGameConfig, model: ModelConfig
    ) -> None:
        self.auxiliary = AuxiliaryValueHeads(
            features, model.auxiliary_value_width, model.auxiliary_value_members
        )
        self.auxiliary_members = model.auxiliary_value_members
        self._set_auxiliary_training_controls(game, model)

    def _budget_conditioning(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, float] | None:
        if not self.auxiliary_budget_conditioning or self.objective_budget_channel is None:
            return None
        remaining = observation[:, self.objective_budget_channel, 0, 0].clamp(0.0, 1.0)
        ratio = torch.exp(5.0 * observation[:, self.auxiliary_ratio_channel, 0, 0])
        return remaining, ratio, self.auxiliary_budget

    def _auxiliary(
        self, features: Tensor, observation: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        conditioning = self._budget_conditioning(observation) if observation is not None else None
        if self.auxiliary_backprop:
            return self.auxiliary(features, conditioning=conditioning)
        solve_features = features if self.auxiliary_solve_backprop else features.detach()
        return self.auxiliary(solve_features, features.detach(), conditioning)

    def composed_auxiliary_value(
        self,
        observation: Tensor,
        auxiliary: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        solve_logits, crossings, moves = auxiliary
        probability = solve_logits.sigmoid()
        ratio = torch.exp(
            5.0 * observation[:, self.auxiliary_ratio_channel, 0, 0]
        )[:, None]
        normalized_cost = (ratio * crossings + moves) / (
            (ratio + 1.0) * self.auxiliary_budget
        )
        normalized_cost = normalized_cost.clamp(0.0, 1.0)
        member_values = -1.0 + 2.0 * probability * (1.0 - normalized_cost)
        return member_values.mean(dim=1)

    def _search_value(
        self, observation: Tensor, legacy: Tensor, features: Tensor
    ) -> Tensor:
        if not self.use_auxiliary_value:
            return legacy
        return self.composed_auxiliary_value(
            observation, self._auxiliary(features, observation)
        )


class _OptionResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.GroupNorm(4, width)
        self.body = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=1),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.norm(hidden + self.body(hidden))


class OptionPolicyAdapter(nn.Module):
    """Zero-initialized, head-relative residual policy for bounded options."""

    def __init__(
        self,
        observation_channels: int,
        action_size: int,
        *,
        internal_budget_channel: int | None,
        width: int = 32,
        residual_blocks: int = 2,
    ):
        super().__init__()
        if width < 16 or width % 4:
            raise ValueError("option adapter width must be >=16 and divisible by 4")
        self.internal_budget_channel = internal_budget_channel
        self.input = nn.Conv1d(observation_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            _OptionResidualBlock(width) for _ in range(residual_blocks)
        )
        self.readout = nn.Sequential(
            nn.Linear(3 * width + 1, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, action_size),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, observation: Tensor) -> Tensor:
        sequence = observation[:, :, 0, :]
        hidden = F.silu(self.input(sequence))
        for block in self.blocks:
            hidden = block(hidden)
        internal_budget = (
            observation[:, self.internal_budget_channel, 0, 0:1]
            if self.internal_budget_channel is not None
            else observation.new_ones((observation.shape[0], 1))
        )
        summary = torch.cat(
            [
                hidden.mean(dim=2),
                hidden.amax(dim=2),
                hidden[:, :, hidden.shape[2] // 2],
                internal_budget,
            ],
            dim=1,
        )
        return self.readout(summary)


class OptionPolicyGate(nn.Module):
    """State-dependent probability that a shared-option residual is applicable."""

    def __init__(
        self,
        observation_channels: int,
        *,
        internal_budget_channel: int | None,
        width: int = 32,
        residual_blocks: int = 2,
        initial_probability: float = 0.1,
    ):
        super().__init__()
        if width < 16 or width % 4:
            raise ValueError("option gate width must be >=16 and divisible by 4")
        if not 0.0 < initial_probability < 1.0:
            raise ValueError("initial gate probability must be between zero and one")
        self.internal_budget_channel = internal_budget_channel
        self.input = nn.Conv1d(observation_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            _OptionResidualBlock(width) for _ in range(residual_blocks)
        )
        self.readout = nn.Sequential(
            nn.Linear(3 * width + 1, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.constant_(
            self.readout[-1].bias,
            math.log(initial_probability / (1.0 - initial_probability)),
        )

    def forward(self, observation: Tensor) -> Tensor:
        sequence = observation[:, :, 0, :]
        hidden = F.silu(self.input(sequence))
        for block in self.blocks:
            hidden = block(hidden)
        internal_budget = (
            observation[:, self.internal_budget_channel, 0, 0:1]
            if self.internal_budget_channel is not None
            else observation.new_ones((observation.shape[0], 1))
        )
        summary = torch.cat(
            [
                hidden.mean(dim=2),
                hidden.amax(dim=2),
                hidden[:, :, hidden.shape[2] // 2],
                internal_budget,
            ],
            dim=1,
        )
        return self.readout(summary).sigmoid()


class BraidPolicyHead(nn.Module):
    """Positional policy head matching the braid action-space layout exactly.

    The action space is blocked as::

        [REDUCE L][COMMUTE L][BRAID L][INSERT 2(N-1)L][singletons][CROSSING L]

    and a `Conv2d(channels, k, 1)` on a `1 x L` latent, flattened channel-major,
    reproduces `k` consecutive per-position blocks in exactly that order. So the
    head is three pieces: one convolution for the blocks that come before the
    singletons, a pooled linear for the six singleton actions, and one more
    convolution for the crossing-change block that comes after them.

    The convolutions carry no dependence on `L`, so widening the word length is a
    data change rather than an architecture change -- which is the property the
    curriculum needs.
    """

    def __init__(self, channels: int, game: BraidGameConfig):
        super().__init__()
        self.action_size = game.action_size
        self.leading_blocks = 3 + 2 * (game.max_strands - 1)
        # Derived, never assumed: the singleton block shrank from 6 to 4 when the
        # word became cyclic and the two rotation moves were deleted.
        self.singleton_actions = game.action_size - (self.leading_blocks + 1) * game.max_len
        if self.singleton_actions < 1:
            raise ValueError(f"action space {game.action_size} too small for the head")
        self.positional = nn.Conv2d(channels, self.leading_blocks, 1)
        self.singletons = nn.Linear(channels, self.singleton_actions)
        self.crossing = nn.Conv2d(channels, 1, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        positional = self.positional(hidden).flatten(1)
        singletons = self.singletons(hidden.mean(dim=(2, 3)))
        crossing = self.crossing(hidden).flatten(1)
        logits = torch.cat([positional, singletons, crossing], dim=1)
        assert logits.shape[1] == self.action_size
        return logits


class BraidAlphaZeroNet(BraidPolicyValueNet):
    """AlphaZero network for the braid environment.

    The observation is a `1 x L` one-row image, so the shared residual stack
    applies unchanged; only the policy head differs.
    """

    def __init__(self, game: BraidGameConfig, model: ModelConfig):
        super().__init__()
        self.representation = Representation(game, model, model.channels)
        # log(A/B) is the 7th scalar plane; it is constant across positions, so
        # one value per batch element suffices.
        self.ratio_channel = 2 * (game.max_strands - 1) + 1 + 1 + 6
        self.film = FiLM(model.channels) if model.film_on_ratio else None
        self.policy_head = BraidPolicyHead(model.channels, game)
        # Pooled, not flattened. `Flatten -> Linear(L, ...)` would tie the value
        # head to one word capacity, defeating the point: every other parameter
        # here depends on the receptive field (11 letters), not on L, so weights
        # trained at one capacity load and run at any other. Mean pooling keeps
        # the scalar planes exact -- they are constant along the word -- and max
        # pooling preserves "does any position have this feature", which mean
        # alone washes out on a mostly padded array.
        # The padding plane is the last of the letter one-hots.
        self.padding_channel = 2 * (game.max_strands - 1)
        self.value_project = nn.Conv2d(model.channels, model.channels, 1)
        self.value_head = nn.Sequential(
            nn.Linear(2 * model.channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self._init_auxiliary(2 * model.channels, game, model)

    def _forward_core(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.representation(observation)
        if self.film is not None:
            log_ratio = observation[:, self.ratio_channel, 0, 0]
            hidden = self.film(hidden, log_ratio)
        projected = torch.relu(self.value_project(hidden))
        # Pool over occupied positions only. Averaging over all L slots dilutes a
        # 5-letter word by ~6x at tier 0, and an A/B over 6 seeds showed that
        # dilution is enough to make runs collapse (0.778 against 0.944).
        occupied = 1.0 - observation[:, self.padding_channel : self.padding_channel + 1]
        count = occupied.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        masked_mean = (projected * occupied).sum(dim=(2, 3), keepdim=True) / count
        masked_max = (projected + (occupied - 1.0) * 1e4).amax(dim=(2, 3))
        summary = torch.cat([masked_mean.flatten(1), masked_max], dim=1)
        return self.policy_head(hidden), self.value_head(summary).squeeze(-1), summary

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        policy, legacy, features = self._forward_core(observation)
        return policy, self._search_value(observation, legacy, features)

    def forward_with_auxiliary(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        policy, value, features = self._forward_core(observation)
        return policy, value, self._auxiliary(features)


class SerialBraidNet(BraidPolicyValueNet):
    """Network for the moving-window formulation.

    The observation is a `1 x w` window and the action space is `O(1)` in `L`,
    neither of which depends on the word length. So this is a small convolution
    over the window and two heads.

    **The policy head is positional, like the parallel one.** The first version
    pooled the window with mean+max and read every logit off that vector, which
    made the readout near-invariant to *where in the window* a feature sat --
    exactly the one question this formulation exists to answer. A trained
    `serial-w7-head` moved its policy by 0.14 when the window contents were
    cyclically rolled against 0.40 for a genuinely different state, so a third of
    its positional signal was leaking through the convolution's zero padding
    rather than being represented. Here the per-offset logits come from a 1x1
    convolution over the window cells, and the head cell's own features are
    concatenated to the pooled summary that feeds the position-free actions --
    shifts included, since "should I move, and which way" is a question about the
    head's neighbourhood, not about the window's average.
    """

    def __init__(self, game: BraidGameConfig, model: ModelConfig):
        super().__init__()
        self.representation = Representation(game, model, model.channels)
        self.ratio_channel = 2 * (game.max_strands - 1) + 1 + 1 + 6
        self.film = FiLM(model.channels) if model.film_on_ratio else None

        self.act_width = game.serial_width
        self.per_offset = 3 + 2 * (game.max_strands - 1) + 1
        self.n_global = game.action_size - self.act_width * self.per_offset
        # The actionable offsets are centred on the head, and so is the window,
        # so the actionable cells are the middle `act_width` columns.
        self.act_start = (game.serial_window - self.act_width) // 2
        self.head_cell = game.serial_window // 2

        self.positional = nn.Conv2d(model.channels, self.per_offset, 1)
        self.body = nn.Sequential(
            nn.Linear(3 * model.channels, 64),
            nn.ReLU(),
        )
        self.global_policy = nn.Linear(64, self.n_global)
        self.value = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        self._init_auxiliary(64, game, model)

    def _forward_core(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.representation(observation)
        if self.film is not None:
            hidden = self.film(hidden, observation[:, self.ratio_channel, 0, 0])

        cells = self.positional(hidden)[:, :, 0, :]  # (B, per_offset, w)
        acting = cells[:, :, self.act_start : self.act_start + self.act_width]
        # (B, act_width, per_offset) -> flat, matching the offset-major layout of
        # `SerialBraidGame.underlying_action`.
        positional = acting.permute(0, 2, 1).flatten(1)

        summary = torch.cat(
            [
                hidden.mean(dim=(2, 3)),
                hidden.amax(dim=(2, 3)),
                hidden[:, :, 0, self.head_cell],
            ],
            dim=1,
        )
        features = self.body(summary)
        conditioning = self._budget_conditioning(observation)
        if conditioning is not None:
            remaining_budget, _, _ = conditioning
            features = features + self.auxiliary.body_budget_skip(remaining_budget[:, None])
        logits = torch.cat([positional, self.global_policy(features)], dim=1)
        value_logit = self.value[0](features)
        if conditioning is not None:
            value_logit = value_logit + self.auxiliary.legacy_budget_skip(
                remaining_budget[:, None]
            )
        logits = self._apply_option_policy_adapter(observation, logits)
        return logits, torch.tanh(value_logit).squeeze(-1), features

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        policy, legacy, features = self._forward_core(observation)
        return policy, self._search_value(observation, legacy, features)

    def forward_with_auxiliary(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        policy, value, features = self._forward_core(observation)
        return policy, value, self._auxiliary(features, observation)


def _mod_centered(value: Tensor, prime: int) -> Tensor:
    """Centered finite-field representative with a straight-through gradient."""
    half = prime // 2
    return torch.remainder(value + half, prime) - half


class SequenceBraidNet(BraidPolicyValueNet):
    """Serial controller with an automatic head-relative whole-word scan.

    The environment and search are identical across these arms. Only the
    accumulator differs: GRU, soft finite automaton, learned modular matrices, or
    a fixed Burau representation supplied as a human-knowledge oracle.
    """

    def __init__(self, game: BraidGameConfig, model: ModelConfig):
        super().__init__()
        self.kind = game.serial_encoder
        self.alphabet = 2 * (game.max_strands - 1)
        self.states = game.serial_encoder_states
        self.prime = game.serial_encoder_prime
        self.max_strands = game.max_strands
        metadata = game.observation_channels

        if self.kind == "gru":
            self.gru = nn.GRU(game.observation_channels, self.states, batch_first=True)
            encoded = self.states
        elif self.kind == "scan-gru":
            self.scan_window = game.serial_window
            # The final feature is an explicit "forced scan" bit. It separates
            # these virtual perception updates from ordinary decision features
            # without adding an environment action or consuming move budget.
            scan_features = game.observation_channels * self.scan_window + 1
            self.gru = nn.GRU(scan_features, self.states, batch_first=True)
            encoded = self.states
        elif self.kind == "fsa":
            self.transitions = nn.Parameter(
                torch.eye(self.states).repeat(self.alphabet, 1, 1)
                + 0.05 * torch.randn(self.alphabet, self.states, self.states)
            )
            encoded = self.states
        elif self.kind == "finite-field":
            self.field_matrices = nn.Parameter(
                torch.eye(self.states).repeat(self.alphabet, 1, 1)
                + 0.25 * torch.randn(self.alphabet, self.states, self.states)
            )
            encoded = self.states * self.states
        elif self.kind == "burau":
            # Two fixed evaluations of the unreduced Burau representation. These
            # are deliberately an oracle arm: known knot-theoretic algebra enters
            # the model, unlike the learned transition arms.
            matrices = torch.stack(
                [self._burau_generators(game.max_strands, t) for t in (-1.0, 0.5)]
            )
            self.register_buffer("burau_matrices", matrices)
            encoded = 2 * game.max_strands * game.max_strands
        else:
            raise ValueError(f"Unknown serial sequence encoder: {self.kind}")

        self.body = nn.Sequential(
            nn.Linear(encoded + metadata, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.policy = nn.Linear(64, game.action_size)
        self.value = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        self._init_auxiliary(64, game, model)

    @staticmethod
    def _burau_generators(strands: int, t: float) -> Tensor:
        out = []
        for token in range(2 * (strands - 1)):
            generator = token % (strands - 1)
            positive = token < strands - 1
            matrix = torch.eye(strands)
            if positive:
                block = torch.tensor([[1.0 - t, t], [1.0, 0.0]])
            else:
                block = torch.tensor([[0.0, 1.0], [1.0 / t, 1.0 - 1.0 / t]])
            matrix[generator : generator + 2, generator : generator + 2] = block
            out.append(matrix)
        return torch.stack(out)

    def _letters_and_mask(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        letters = observation[:, : self.alphabet, 0, :].permute(0, 2, 1)
        mask = letters.sum(dim=-1)
        return letters, mask

    def _scan_fsa(self, letters: Tensor, mask: Tensor) -> Tensor:
        batch = letters.shape[0]
        state = torch.zeros(batch, self.states, device=letters.device, dtype=letters.dtype)
        state[:, 0] = 1.0
        transitions = torch.softmax(self.transitions, dim=-1)
        for index in range(letters.shape[1]):
            selected = torch.einsum("ba,aij->bij", letters[:, index], transitions)
            advanced = torch.bmm(state[:, None, :], selected).squeeze(1)
            active = mask[:, index : index + 1]
            state = active * advanced + (1.0 - active) * state
        return state

    def _field_operators(self) -> Tensor:
        rounded = self.field_matrices + (
            torch.round(self.field_matrices) - self.field_matrices
        ).detach()
        return _mod_centered(rounded, self.prime)

    def _scan_field(self, letters: Tensor, mask: Tensor) -> Tensor:
        batch = letters.shape[0]
        state = torch.eye(
            self.states, device=letters.device, dtype=letters.dtype
        ).expand(batch, -1, -1)
        operators = self._field_operators()
        for index in range(letters.shape[1]):
            selected = torch.einsum("ba,aij->bij", letters[:, index], operators)
            advanced = _mod_centered(torch.bmm(state, selected), self.prime)
            active = mask[:, index, None, None]
            state = active * advanced + (1.0 - active) * state
        return state.flatten(1) / max(self.prime // 2, 1)

    def _scan_burau(self, letters: Tensor, mask: Tensor) -> Tensor:
        batch = letters.shape[0]
        states = torch.eye(
            self.max_strands, device=letters.device, dtype=letters.dtype
        ).expand(2, batch, -1, -1).clone()
        operators = self.burau_matrices.to(dtype=letters.dtype)
        for index in range(letters.shape[1]):
            selected = torch.einsum("ba,eaij->ebij", letters[:, index], operators)
            advanced = torch.matmul(states, selected)
            active = mask[:, index][None, :, None, None]
            states = active * advanced + (1.0 - active) * states
        # Prevent long words at t=-1 from dominating the learned heads.
        states = torch.sign(states) * torch.log1p(torch.abs(states))
        return states.permute(1, 0, 2, 3).flatten(1)

    def encode(self, observation: Tensor) -> Tensor:
        letters, mask = self._letters_and_mask(observation)
        if self.kind == "gru":
            sequence = observation[:, :, 0, :].permute(0, 2, 1)
            output, _ = self.gru(sequence)
            lengths = mask.sum(dim=1).long().clamp(min=1) - 1
            return output[torch.arange(output.shape[0], device=output.device), lengths]
        if self.kind == "scan-gru":
            sequence = observation[:, :, 0, :].permute(0, 2, 1)
            batch, capacity, channels = sequence.shape
            lengths = mask.sum(dim=1).long().clamp(min=1, max=capacity)
            centres = torch.arange(capacity, device=sequence.device)[None, :, None]
            offsets = torch.arange(
                -(self.scan_window // 2),
                self.scan_window // 2 + 1,
                device=sequence.device,
            )[None, None, :]
            indexes = torch.remainder(centres + offsets, lengths[:, None, None])
            source = sequence[:, None, :, :].expand(-1, capacity, -1, -1)
            gathered = torch.gather(
                source,
                2,
                indexes[:, :, :, None].expand(-1, -1, -1, channels),
            ).flatten(2)
            scan_bit = torch.ones(
                batch, capacity, 1, device=sequence.device, dtype=sequence.dtype
            )
            output, _ = self.gru(torch.cat([gathered, scan_bit], dim=-1))
            final = lengths - 1
            return output[torch.arange(batch, device=output.device), final]
        if self.kind == "fsa":
            return self._scan_fsa(letters, mask)
        if self.kind == "finite-field":
            return self._scan_field(letters, mask)
        return self._scan_burau(letters, mask)

    def regularization_loss(self) -> Tensor:
        """Braid-group relation residual for the learned algebraic arms."""
        if self.kind not in {"fsa", "finite-field"}:
            return next(self.parameters()).new_zeros(())
        operators = (
            torch.softmax(self.transitions, dim=-1)
            if self.kind == "fsa"
            else self._field_operators()
        )
        generators = self.max_strands - 1
        identity = torch.eye(
            operators.shape[-1], device=operators.device, dtype=operators.dtype
        )

        def product(*items: Tensor) -> Tensor:
            value = items[0]
            for item in items[1:]:
                value = value @ item
                if self.kind == "finite-field":
                    value = _mod_centered(value, self.prime)
            return value

        residuals = []
        for i in range(generators):
            residuals.extend(
                [product(operators[i], operators[i + generators]) - identity,
                 product(operators[i + generators], operators[i]) - identity]
            )
        for i in range(generators - 1):
            residuals.append(
                product(operators[i], operators[i + 1], operators[i])
                - product(operators[i + 1], operators[i], operators[i + 1])
            )
        for i in range(generators):
            for j in range(i + 2, generators):
                residuals.append(
                    product(operators[i], operators[j])
                    - product(operators[j], operators[i])
                )
        return torch.stack([r.square().mean() for r in residuals]).mean()

    def _forward_core(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.encode(observation)
        head = observation[:, :, 0, 0]
        features = self.body(torch.cat([encoded, head], dim=1))
        policy = self._apply_option_policy_adapter(observation, self.policy(features))
        return policy, self.value(features).squeeze(-1), features

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        policy, legacy, features = self._forward_core(observation)
        return policy, self._search_value(observation, legacy, features)

    def forward_with_auxiliary(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        policy, value, features = self._forward_core(observation)
        return policy, value, self._auxiliary(features)


class CyclicMemoryBraidNet(BraidPolicyValueNet):
    """Pretrained local controller plus a cyclic, full-word memory encoder.

    The window tower can be loaded exactly from ``s-window-128``. The second
    tower performs dilated message passing on the occupied cyclic word and pools
    without an absolute seam, giving the critic a rotation-invariant receptive
    field over the whole braid. A transported eight-symbol tape is visible only
    to this global tower. Zero-initialized residual heads make parent import
    function-preserving on every action the parent represents.
    """

    def __init__(self, game: BraidGameConfig, model: ModelConfig):
        super().__init__()
        if game.serial_encoder != "cyclic-memory":
            raise ValueError(f"Unknown cyclic-memory encoder: {game.serial_encoder}")
        if game.serial_window != 7 or game.serial_width != 7:
            raise ValueError("cyclic-memory requires a seven-cell action window")
        if game.serial_tape_symbols != 8 or not game.serial_tape_preserve_shift:
            raise ValueError("cyclic-memory requires eight tape symbols plus preserve shifts")
        self.tape_symbols = game.serial_tape_symbols
        self.base_channels = game.observation_channels - self.tape_symbols
        window_game = replace(
            game,
            serial_encoder="",
            serial_encoder_states=0,
            serial_tape_symbols=0,
            serial_tape_preserve_shift=False,
        )
        self.window = SerialBraidNet(window_game, model)
        self.padding_channel = 2 * (game.max_strands - 1)
        self.local_radius = game.serial_window // 2
        self.action_size = game.action_size
        self.width = game.serial_encoder_states
        if self.width < 16:
            raise ValueError("cyclic-memory encoder width must be at least 16")
        self.input_project = nn.Conv1d(game.observation_channels, self.width, 1)
        self.dilations = (1, 2, 4, 8, 16)
        self.cyclic_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(3 * self.width, 2 * self.width, 1),
                nn.SiLU(),
                nn.Conv1d(2 * self.width, self.width, 1),
            )
            for _ in self.dilations
        )
        features = 64 + 2 * self.width
        self.fusion_norm = nn.LayerNorm(features)
        self.policy_residual = nn.Linear(features, game.action_size)
        self.value_residual = nn.Linear(features, 1)
        members = model.auxiliary_value_members
        self.solve_residual = nn.Linear(features, members)
        self.cost_residual = nn.Linear(features, 2 * members)
        for layer in (
            self.policy_residual,
            self.value_residual,
            self.solve_residual,
            self.cost_residual,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.register_buffer(
            "window_action_map", self._window_action_mapping(game, window_game)
        )
        self.auxiliary_members = members
        self._set_auxiliary_training_controls(game, model)

    @staticmethod
    def _window_action_mapping(
        game: BraidGameConfig, window_game: BraidGameConfig
    ) -> Tensor:
        per_offset = 3 + 2 * (game.max_strands - 1) + 1
        base = game.serial_width * per_offset + 4
        variants = game.serial_tape_symbols + 1
        shifts = 2 * len(game.serial_strides)
        mapping = list(range(base))
        mapping.extend(base + shift * variants for shift in range(shifts))
        if len(mapping) != window_game.action_size:
            raise ValueError(
                f"window action map has {len(mapping)} entries, expected "
                f"{window_game.action_size}"
            )
        return torch.tensor(mapping, dtype=torch.long)

    def load_window_state_dict(self, state: dict[str, Tensor]) -> None:
        load_policy_value_state_dict(self.window, state)

    def _lengths(self, observation: Tensor) -> Tensor:
        occupied = 1.0 - observation[:, self.padding_channel, 0, :]
        return occupied.sum(dim=1).long().clamp(min=1, max=observation.shape[-1])

    def _local_view(self, observation: Tensor, lengths: Tensor) -> Tensor:
        offsets = torch.arange(
            -self.local_radius,
            self.local_radius + 1,
            device=observation.device,
        )[None, :]
        indexes = torch.remainder(offsets, lengths[:, None])
        local = torch.gather(
            observation,
            3,
            indexes[:, None, None, :].expand(-1, observation.shape[1], 1, -1),
        )
        return local[:, : self.base_channels]

    @staticmethod
    def _cyclic_neighbour(hidden: Tensor, lengths: Tensor, offset: int) -> Tensor:
        positions = torch.arange(hidden.shape[-1], device=hidden.device)[None, :]
        indexes = torch.remainder(positions + offset, lengths[:, None])
        return torch.gather(hidden, 2, indexes[:, None, :].expand(-1, hidden.shape[1], -1))

    def encode_global(self, observation: Tensor) -> Tensor:
        lengths = self._lengths(observation)
        positions = torch.arange(observation.shape[-1], device=observation.device)[None, :]
        occupied = (positions < lengths[:, None]).to(observation.dtype)[:, None, :]
        hidden = F.silu(self.input_project(observation[:, :, 0, :])) * occupied
        for dilation, block in zip(self.dilations, self.cyclic_blocks, strict=True):
            neighbours = torch.cat(
                [
                    self._cyclic_neighbour(hidden, lengths, -dilation),
                    hidden,
                    self._cyclic_neighbour(hidden, lengths, dilation),
                ],
                dim=1,
            )
            hidden = (hidden + block(neighbours)) * occupied
        count = occupied.sum(dim=2).clamp(min=1.0)
        mean = hidden.sum(dim=2) / count
        maximum = (hidden + (occupied - 1.0) * 1e4).amax(dim=2)
        return torch.cat([mean, maximum], dim=1)

    def _forward_core(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        lengths = self._lengths(observation)
        local = self._local_view(observation, lengths)
        window_logits, window_value, window_features = self.window._forward_core(local)
        global_features = self.encode_global(observation)
        features = self.fusion_norm(torch.cat([window_features, global_features], dim=1))
        # New tape-write actions start with negligible mass, so importing the
        # parent does not immediately dilute its established policy. Root noise
        # still explores them and the residual head can raise them during training.
        logits = window_logits.new_full((window_logits.shape[0], self.action_size), -8.0)
        logits[:, self.window_action_map] = window_logits
        logits = logits + self.policy_residual(features)
        logits = self._apply_option_policy_adapter(observation, logits)
        value = (window_value + self.value_residual(features).squeeze(-1)).clamp(-1.0, 1.0)
        parent_auxiliary = self.window._auxiliary(window_features)
        solve_features = (
            features
            if self.auxiliary_backprop or self.auxiliary_solve_backprop
            else features.detach()
        )
        cost_features = features if self.auxiliary_backprop else features.detach()
        solve_delta = self.solve_residual(solve_features)
        cost_delta = self.cost_residual(cost_features).view(
            observation.shape[0], self.auxiliary_members, 2
        )
        auxiliary = (
            parent_auxiliary[0] + solve_delta,
            (parent_auxiliary[1] + cost_delta[:, :, 0]).clamp(min=0.0),
            (parent_auxiliary[2] + cost_delta[:, :, 1]).clamp(min=0.0),
        )
        return logits, value, features, auxiliary

    def composed_auxiliary_value(
        self,
        observation: Tensor,
        auxiliary: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        solve_logits, crossings, moves = auxiliary
        probability = solve_logits.sigmoid()
        ratio = torch.exp(
            5.0 * observation[:, self.auxiliary_ratio_channel, 0, 0]
        )[:, None]
        normalized_cost = (ratio * crossings + moves) / (
            (ratio + 1.0) * self.auxiliary_budget
        )
        member_values = -1.0 + 2.0 * probability * (1.0 - normalized_cost.clamp(0.0, 1.0))
        return member_values.mean(dim=1)

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        policy, legacy, _, auxiliary = self._forward_core(observation)
        value = (
            self.composed_auxiliary_value(observation, auxiliary)
            if self.use_auxiliary_value
            else legacy
        )
        return policy, value

    def forward_with_auxiliary(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        policy, legacy, _, auxiliary = self._forward_core(observation)
        value = (
            self.composed_auxiliary_value(observation, auxiliary)
            if self.use_auxiliary_value
            else legacy
        )
        return policy, value, auxiliary


class TriadBraidNet(BraidPolicyValueNet):
    """Frozen window/scan/tape experts with a learned normalized policy mixer.

    The composite environment exposes the full head-relative word and a four-
    symbol tape.  This module derives the three observations expected by the
    original parents, preserving their architectures and checkpoint tensors:

    * ``s-window-128`` receives a centred tape-free seven-cell window;
    * ``s-scan-gru`` receives the complete tape-free head-relative word; and
    * ``s-tape4`` receives the centred seven-cell window including its tape.

    Parent logits are centred and scaled independently before a zero-initialized
    state/action router averages the experts that actually represent each
    semantic action.  A zero-initialized residual can subsequently learn
    interactions that no convex mixture can express.
    """

    PARENT_NAMES = ("s-window-128", "s-scan-gru", "s-tape4")

    def __init__(self, game: BraidGameConfig, model: ModelConfig):
        super().__init__()
        if game.serial_ensemble != "window-scan-tape":
            raise ValueError(f"Unknown serial ensemble: {game.serial_ensemble}")
        if game.serial_window != 7 or game.serial_width != 7:
            raise ValueError("window-scan-tape requires a seven-cell action window")
        if game.serial_tape_symbols != 4 or not game.serial_tape_preserve_shift:
            raise ValueError("window-scan-tape requires four tape symbols plus preserve shifts")

        common = dict(
            serial_ensemble="",
            serial_tape_preserve_shift=False,
            serial_registers=0,
            serial_colours=0,
        )
        window_game = replace(
            game,
            serial_act_width=7,
            serial_tape_symbols=0,
            serial_encoder="",
            serial_encoder_states=0,
            **common,
        )
        scan_game = replace(
            game,
            serial_act_width=1,
            serial_tape_symbols=0,
            serial_encoder="scan-gru",
            serial_encoder_states=128,
            **common,
        )
        tape_game = replace(
            game,
            serial_act_width=1,
            serial_tape_symbols=4,
            serial_encoder="",
            serial_encoder_states=0,
            **common,
        )
        self.window = SerialBraidNet(window_game, model)
        self.scan = SequenceBraidNet(scan_game, model)
        self.tape = SerialBraidNet(tape_game, model)
        self.towers = (self.window, self.scan, self.tape)

        self.base_channels = game.observation_channels - game.serial_tape_symbols
        self.padding_channel = 2 * (game.max_strands - 1)
        self.local_radius = game.serial_window // 2
        self.action_size = game.action_size
        self.feature_size = 64 * len(self.towers)
        self.fusion_norm = nn.LayerNorm(self.feature_size)
        self.router = nn.Linear(self.feature_size, self.action_size * len(self.towers))
        self.policy_residual = nn.Linear(self.feature_size, self.action_size)
        self.value_residual = nn.Linear(self.feature_size, 1)
        members = model.auxiliary_value_members
        self.solve_residual = nn.Linear(self.feature_size, members)
        self.cost_residual = nn.Linear(self.feature_size, 2 * members)
        for layer in (
            self.router,
            self.policy_residual,
            self.value_residual,
            self.solve_residual,
            self.cost_residual,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        mappings = self._action_mappings(game, window_game, scan_game, tape_game)
        support = torch.zeros(self.action_size, len(self.towers), dtype=torch.bool)
        for tower, mapping in enumerate(mappings):
            support[mapping, tower] = True
            self.register_buffer(f"action_map_{tower}", mapping)
        if not bool(support.any(dim=1).all()):
            raise ValueError("Composite action space contains an action with no parent expert")
        self.register_buffer("action_support", support)

        self.auxiliary_members = members
        self._set_auxiliary_training_controls(game, model)
        self._freeze_towers()

    @staticmethod
    def _action_mappings(
        game: BraidGameConfig,
        window_game: BraidGameConfig,
        scan_game: BraidGameConfig,
        tape_game: BraidGameConfig,
    ) -> tuple[Tensor, Tensor, Tensor]:
        per_offset = 3 + 2 * (game.max_strands - 1) + 1
        union_singleton = game.serial_width * per_offset
        union_shift = union_singleton + 4
        union_variants = game.serial_tape_symbols + 1  # preserve, then WRITE(0..3)
        shifts = 2 * len(game.serial_strides)

        window = list(range(window_game.serial_width * per_offset + 4))
        window.extend(union_shift + shift * union_variants for shift in range(shifts))

        centre = (game.serial_width // 2) * per_offset
        head = [centre + action for action in range(per_offset)]
        head.extend(union_singleton + action for action in range(4))
        scan = head + [union_shift + shift * union_variants for shift in range(shifts)]

        tape = list(head)
        tape.extend(
            union_shift + shift * union_variants + 1 + symbol
            for shift in range(shifts)
            for symbol in range(game.serial_tape_symbols)
        )
        mappings = tuple(torch.tensor(items, dtype=torch.long) for items in (window, scan, tape))
        expected = (window_game.action_size, scan_game.action_size, tape_game.action_size)
        actual = tuple(len(mapping) for mapping in mappings)
        if actual != expected:
            raise ValueError(f"Composite action maps {actual} do not match parents {expected}")
        return mappings  # type: ignore[return-value]

    def _freeze_towers(self) -> None:
        for tower in self.towers:
            tower.eval()
            for parameter in tower.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen BatchNorm statistics are part of each parent checkpoint.
        for tower in self.towers:
            tower.eval()
        return self

    def load_parent_state_dicts(
        self,
        window: dict[str, Tensor],
        scan: dict[str, Tensor],
        tape: dict[str, Tensor],
    ) -> None:
        for tower, state in zip(self.towers, (window, scan, tape), strict=True):
            tower.load_state_dict(state)
        self._freeze_towers()

    def _views(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        occupied = 1.0 - observation[:, self.padding_channel, 0, :]
        lengths = occupied.sum(dim=1).long().clamp(min=1, max=observation.shape[-1])
        offsets = torch.arange(
            -self.local_radius,
            self.local_radius + 1,
            device=observation.device,
        )[None, :]
        indexes = torch.remainder(offsets, lengths[:, None])
        local = torch.gather(
            observation,
            3,
            indexes[:, None, None, :].expand(-1, observation.shape[1], 1, -1),
        )
        return (
            local[:, : self.base_channels],
            observation[:, : self.base_channels],
            local,
        )

    @staticmethod
    def _normalize_logits(logits: Tensor) -> Tensor:
        centred = logits - logits.mean(dim=1, keepdim=True)
        scale = centred.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-4)
        return centred / scale

    def _forward_core(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        views = self._views(observation)
        outputs = [
            tower._forward_core(view)
            for tower, view in zip(self.towers, views, strict=True)
        ]
        logits = [self._normalize_logits(output[0]) for output in outputs]
        legacy = torch.stack([output[1] for output in outputs], dim=1).mean(dim=1)
        features = self.fusion_norm(torch.cat([output[2] for output in outputs], dim=1))

        scattered = observation.new_zeros(
            observation.shape[0], self.action_size, len(self.towers)
        )
        for tower, tower_logits in enumerate(logits):
            mapping = getattr(self, f"action_map_{tower}")
            scattered[:, mapping, tower] = tower_logits
        router = self.router(features).view(
            observation.shape[0], self.action_size, len(self.towers)
        )
        router = router.masked_fill(~self.action_support[None], torch.finfo(router.dtype).min)
        weights = torch.softmax(router, dim=2)
        policy = (weights * scattered).sum(dim=2) + self.policy_residual(features)
        policy = self._apply_option_policy_adapter(observation, policy)
        value = (legacy + 0.25 * torch.tanh(self.value_residual(features).squeeze(-1))).clamp(
            -1.0, 1.0
        )

        parent_auxiliary = [
            tower._auxiliary(output[2]) for tower, output in zip(self.towers, outputs, strict=True)
        ]
        solve = torch.stack([item[0] for item in parent_auxiliary], dim=0).mean(dim=0)
        crossings = torch.stack([item[1] for item in parent_auxiliary], dim=0).mean(dim=0)
        moves = torch.stack([item[2] for item in parent_auxiliary], dim=0).mean(dim=0)
        solve_features = (
            features
            if self.auxiliary_backprop or self.auxiliary_solve_backprop
            else features.detach()
        )
        cost_features = features if self.auxiliary_backprop else features.detach()
        solve = solve + self.solve_residual(solve_features)
        cost_scale = torch.exp(
            0.5
            * torch.tanh(self.cost_residual(cost_features)).view(
                observation.shape[0], self.auxiliary_members, 2
            )
        )
        crossings = crossings * cost_scale[:, :, 0]
        moves = moves * cost_scale[:, :, 1]
        return policy, value, features, (solve, crossings, moves)

    def composed_auxiliary_value(
        self,
        observation: Tensor,
        auxiliary: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        solve_logits, crossings, moves = auxiliary
        probability = solve_logits.sigmoid()
        ratio = torch.exp(
            5.0 * observation[:, self.auxiliary_ratio_channel, 0, 0]
        )[:, None]
        normalized_cost = ((ratio * crossings + moves) / (
            (ratio + 1.0) * self.auxiliary_budget
        )).clamp(0.0, 1.0)
        return (-1.0 + 2.0 * probability * (1.0 - normalized_cost)).mean(dim=1)

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        policy, legacy, _, auxiliary = self._forward_core(observation)
        value = (
            self.composed_auxiliary_value(observation, auxiliary)
            if self.use_auxiliary_value
            else legacy
        )
        return policy, value

    def forward_with_auxiliary(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        policy, legacy, _, auxiliary = self._forward_core(observation)
        value = (
            self.composed_auxiliary_value(observation, auxiliary)
            if self.use_auxiliary_value
            else legacy
        )
        return policy, value, auxiliary


def load_policy_value_state_dict(
    network: PolicyValueNet, state_dict: dict[str, Tensor]
) -> bool:
    """Load compatible old braid checkpoints with narrowly defined migrations.

    Returns ``True`` when an old checkpoint was migrated. Any missing or extra
    parameter outside ``auxiliary.*`` remains an error. Newly appended
    observation features are initialized as ignored inputs (all zero weights)
    for the representation layers used by the serial candidates.
    """
    if not isinstance(network, BraidPolicyValueNet):
        network.load_state_dict(state_dict)
        return False
    migrated_state = dict(state_dict)
    if (
        network.option_policy_adapter is None
        and any(key.startswith("option_policy_adapter.") for key in migrated_state)
    ):
        network.attach_option_policy_adapter()
    if (
        network.option_policy_gate is None
        and any(key.startswith("option_policy_gate.") for key in migrated_state)
    ):
        network.attach_option_policy_gate()
    target_state = network.state_dict()
    expandable_inputs = {
        "representation.net.0.weight",
        "window.representation.net.0.weight",
        "input_project.weight",
        "gru.weight_ih_l0",
        "body.0.weight",
        "scan.gru.weight_ih_l0",
        "scan.body.0.weight",
        "tape.representation.net.0.weight",
    }
    expanded = False
    for key in expandable_inputs:
        source = migrated_state.get(key)
        target = target_state.get(key)
        if source is None or target is None or source.shape == target.shape:
            continue
        compatible = (
            source.ndim == target.ndim
            and source.ndim >= 2
            and target.shape[1] > source.shape[1]
            and source.shape[:1] == target.shape[:1]
            and source.shape[2:] == target.shape[2:]
        )
        if not compatible:
            continue
        padded = torch.zeros_like(target)
        index = [slice(None)] * source.ndim
        index[1] = slice(0, source.shape[1])
        padded[tuple(index)] = source
        migrated_state[key] = padded
        expanded = True

    # The repaired option controller adds an explicit head-cell summary between
    # the old mean/max pools and budget scalar. Preserve old adapter behavior by
    # inserting zero weights for that new summary and moving the budget column.
    for key in (
        "option_policy_adapter.readout.0.weight",
        "option_policy_gate.readout.0.weight",
    ):
        source = migrated_state.get(key)
        target = target_state.get(key)
        if (
            source is None
            or target is None
            or source.shape == target.shape
            or source.ndim != 2
            or source.shape[0] != target.shape[0]
            or target.shape[1] <= source.shape[1]
        ):
            continue
        old_pooled_width = source.shape[1] - 1
        if old_pooled_width <= 0 or target.shape[1] != source.shape[1] + old_pooled_width // 2:
            continue
        padded = torch.zeros_like(target)
        padded[:, :old_pooled_width] = source[:, :old_pooled_width]
        padded[:, -1] = source[:, -1]
        migrated_state[key] = padded
        expanded = True

    incompatible = network.load_state_dict(migrated_state, strict=False)
    auxiliary_prefixes = ("auxiliary.", "window.auxiliary.")
    bad_missing = [
        key for key in incompatible.missing_keys if not key.startswith(auxiliary_prefixes)
    ]
    bad_unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(auxiliary_prefixes)
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint; missing={bad_missing}, unexpected={bad_unexpected}"
        )
    return expanded or bool(incompatible.missing_keys)


def make_braid_network(game: BraidGameConfig, model: ModelConfig) -> PolicyValueNet:
    if game.serial_ensemble:
        return TriadBraidNet(game, model)
    if game.serial_encoder == "cyclic-memory":
        return CyclicMemoryBraidNet(game, model)
    if game.serial_encoder:
        return SequenceBraidNet(game, model)
    if game.serial_window:
        return SerialBraidNet(game, model)
    return BraidAlphaZeroNet(game, model)


class Dynamics(nn.Module):
    """Learned latent transition. Go-specific: the action is encoded as a board
    point plus a pass plane, which has no analogue in a 1158-action braid space."""

    def __init__(self, game: AnyGameConfig, model: ModelConfig):
        super().__init__()
        if isinstance(game, BraidGameConfig):
            raise NotImplementedError(
                "MuZero's learned dynamics are Go-specific; the braid environment "
                "needs an action-embedding transition instead (roadmap M2 ablation)"
            )
        channels = model.latent_channels
        self.board_size = game.board_size
        self.action_size = game.action_size
        self.transition = nn.Sequential(
            nn.Conv2d(channels + 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            *[ResidualBlock(channels) for _ in range(model.residual_blocks)],
        )
        self.reward = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(game.cells, 1),
            nn.Tanh(),
        )

    def forward(self, hidden: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        batch = hidden.shape[0]
        point = torch.zeros(
            batch, 1, self.board_size, self.board_size, device=hidden.device, dtype=hidden.dtype
        )
        is_point = action < self.board_size**2
        if bool(is_point.any()):
            rows = action[is_point] // self.board_size
            cols = action[is_point] % self.board_size
            point[is_point, 0, rows, cols] = 1.0
        is_pass = (action == self.board_size**2).to(hidden.dtype)
        pass_plane = is_pass[:, None, None, None].expand(
            batch, 1, self.board_size, self.board_size
        )
        next_hidden = self.transition(torch.cat([hidden, point, pass_plane], dim=1))
        reward = self.reward(next_hidden).squeeze(-1)
        return next_hidden, reward


class MuZeroNet(nn.Module):
    """Compact MuZero with learned latent dynamics and an auxiliary legality head."""

    def __init__(self, game: AnyGameConfig, model: ModelConfig):
        super().__init__()
        self.representation = Representation(game, model, model.latent_channels)
        self.prediction = PredictionHead(
            model.latent_channels,
            game,
            include_legal=True,
            include_terminal=True,
        )
        self.dynamics = Dynamics(game, model)

    def initial_inference(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        hidden = self.representation(observation)
        policy, value, legal, terminal = self.prediction(hidden)
        assert legal is not None and terminal is not None
        return hidden, policy, value, legal, terminal

    def recurrent_inference(
        self, hidden: Tensor, action: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        next_hidden, reward = self.dynamics(hidden, action)
        policy, value, legal, terminal = self.prediction(next_hidden)
        assert legal is not None and terminal is not None
        return next_hidden, reward, policy, value, legal, terminal
