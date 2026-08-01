from __future__ import annotations

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
        # Neutral solve prior and modest positive cost priors. Small weights keep
        # members distinct without injecting large random outputs at migration.
        nn.init.normal_(self.solve[-1].weight, std=1e-3)
        nn.init.zeros_(self.solve[-1].bias)
        nn.init.normal_(self.cost[-1].weight, std=1e-3)
        with torch.no_grad():
            self.cost[-1].bias.copy_(
                torch.tensor([_inverse_softplus(1.0), _inverse_softplus(8.0)])
            )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        solve_logit = self.solve(features).squeeze(-1)
        costs = F.softplus(self.cost(features))
        return solve_logit, costs[:, 0], costs[:, 1]


class AuxiliaryValueHeads(nn.Module):
    def __init__(self, features: int, width: int, members: int):
        super().__init__()
        if members < 1:
            raise ValueError("auxiliary_value_members must be positive")
        self.members = nn.ModuleList(
            AuxiliaryValueMember(features, width) for _ in range(members)
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        outputs = [member(features) for member in self.members]
        return tuple(torch.stack(items, dim=1) for items in zip(*outputs, strict=True))  # type: ignore[return-value]


class BraidPolicyValueNet(PolicyValueNet):
    """Common shadow-critic plumbing for every braid representation."""

    def _init_auxiliary(
        self, features: int, game: BraidGameConfig, model: ModelConfig
    ) -> None:
        self.auxiliary = AuxiliaryValueHeads(
            features, model.auxiliary_value_width, model.auxiliary_value_members
        )
        self.auxiliary_members = model.auxiliary_value_members
        self.auxiliary_loss_weight = model.auxiliary_value_loss_weight
        self.auxiliary_backprop = model.auxiliary_backprop_to_encoder
        self.use_auxiliary_value = model.use_auxiliary_value
        self.auxiliary_budget = float(game.simplify_budget)
        self.auxiliary_ratio_channel = 2 * (game.max_strands - 1) + 1 + 1 + 6

    def _auxiliary(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not self.auxiliary_backprop:
            features = features.detach()
        return self.auxiliary(features)

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
            observation, self._auxiliary(features)
        )


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
        logits = torch.cat([positional, self.global_policy(features)], dim=1)
        return logits, self.value(features).squeeze(-1), features

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        policy, legacy, features = self._forward_core(observation)
        return policy, self._search_value(observation, legacy, features)

    def forward_with_auxiliary(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        policy, value, features = self._forward_core(observation)
        return policy, value, self._auxiliary(features)


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
        return self.policy(features), self.value(features).squeeze(-1), features

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        policy, legacy, features = self._forward_core(observation)
        return policy, self._search_value(observation, legacy, features)

    def forward_with_auxiliary(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        policy, value, features = self._forward_core(observation)
        return policy, value, self._auxiliary(features)


def load_policy_value_state_dict(
    network: PolicyValueNet, state_dict: dict[str, Tensor]
) -> bool:
    """Load old braid checkpoints while initializing only new auxiliary towers.

    Returns ``True`` when an old checkpoint was migrated. Any missing or extra
    parameter outside ``auxiliary.*`` remains an error.
    """
    if not isinstance(network, BraidPolicyValueNet):
        network.load_state_dict(state_dict)
        return False
    incompatible = network.load_state_dict(state_dict, strict=False)
    bad_missing = [key for key in incompatible.missing_keys if not key.startswith("auxiliary.")]
    bad_unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith("auxiliary.")
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint; missing={bad_missing}, unexpected={bad_unexpected}"
        )
    return bool(incompatible.missing_keys)


def make_braid_network(game: BraidGameConfig, model: ModelConfig) -> PolicyValueNet:
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
