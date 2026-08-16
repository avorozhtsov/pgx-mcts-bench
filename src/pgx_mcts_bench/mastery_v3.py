"""Function-preserving mastery-v3 architecture candidates.

The two candidates deliberately contain the admitted 12-strand cyclic-memory
controller as a nested parent.  New policy, value, and factorized-critic routes
are zero initialized, so a checkpoint fork has an auditable exact function at
step zero.  The parent is not frozen: after the migration gate, the registered
curriculum may train the complete child.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pgx_mcts_bench.config import BraidGameConfig, ModelConfig
from pgx_mcts_bench.networks import (
    BraidPolicyValueNet,
    CyclicMemoryBraidNet,
    load_policy_value_state_dict,
)

PARENT_ENCODER_WIDTH = 64
PARENT_RESIDUAL_BLOCKS = 4
V3_ENCODER_WIDTH = 128
V3_RESIDUAL_BLOCKS = 10
V3_DILATIONS = (1, 2, 4, 8, 16, 1, 2, 4, 8, 16)
V3_ENCODERS = {"cyclic-memory-deep-v3", "cyclic-graph-dual-v3"}
V3_ORDINAL_MAX_U = 12


@dataclass(frozen=True)
class V3ProofDiagnostics:
    """Proof-supervised outputs kept separate from operational solve odds."""

    feasibility_logit: Tensor
    lower_bound: Tensor
    upper_bound: Tensor
    ordinal_logits: Tensor
    budget_linear: Tensor
    budget_log: Tensor


@dataclass(frozen=True)
class V3Diagnostics:
    """Outputs that are trained and audited outside the legacy MCTS API."""

    row_pair_logits: Tensor
    invalid_logit: Tensor
    capacity_logit: Tensor


class _ZeroLayerScaleCyclicBlock(nn.Module):
    """Cyclic residual block whose new route is exactly zero at construction."""

    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(width)
        self.body = nn.Sequential(
            nn.Conv1d(3 * width, 2 * width, 1),
            nn.SiLU(),
            nn.Conv1d(2 * width, width, 1),
        )
        self.layer_scale = nn.Parameter(torch.zeros(width))

    def forward(self, hidden: Tensor, lengths: Tensor, occupied: Tensor) -> Tensor:
        normalized = self.norm(hidden.transpose(1, 2)).transpose(1, 2)
        neighbours = torch.cat(
            [
                CyclicMemoryBraidNet._cyclic_neighbour(
                    normalized, lengths, -self.dilation
                ),
                normalized,
                CyclicMemoryBraidNet._cyclic_neighbour(
                    normalized, lengths, self.dilation
                ),
            ],
            dim=1,
        )
        residual = self.body(neighbours) * self.layer_scale[None, :, None]
        return (hidden + residual) * occupied


class _BoundedStrandGraphBlock(nn.Module):
    """Message passing on the bounded Artin strand path (never a cyclic seam)."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.body = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.layer_scale = nn.Parameter(torch.zeros(width))

    def forward(self, nodes: Tensor, active: Tensor) -> Tensor:
        normalized = self.norm(nodes)
        zeros = torch.zeros_like(normalized[:, :1])
        left = torch.cat([zeros, normalized[:, :-1]], dim=1)
        right = torch.cat([normalized[:, 1:], zeros], dim=1)
        residual = self.body(torch.cat([left, normalized, right], dim=2))
        residual = residual * self.layer_scale[None, None, :]
        return (nodes + residual) * active[:, :, None]


def _zero_linear(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class CyclicMemoryDeepV3(BraidPolicyValueNet):
    """Exact cyclic-memory parent plus a 128x10 zero-LayerScale tower."""

    encoder_name = "cyclic-memory-deep-v3"

    def __init__(self, game: BraidGameConfig, model: ModelConfig):
        super().__init__()
        if game.serial_encoder != self.encoder_name:
            raise ValueError(f"expected {self.encoder_name}, got {game.serial_encoder!r}")
        if game.serial_encoder_states != V3_ENCODER_WIDTH:
            raise ValueError(f"{self.encoder_name} requires {V3_ENCODER_WIDTH} channels")
        if model.residual_blocks != V3_RESIDUAL_BLOCKS:
            raise ValueError(f"{self.encoder_name} requires {V3_RESIDUAL_BLOCKS} blocks")

        parent_game = self._parent_game(game)
        parent_model = replace(model, residual_blocks=PARENT_RESIDUAL_BLOCKS)
        self.parent = CyclicMemoryBraidNet(parent_game, parent_model)
        self.game = game
        self.width = V3_ENCODER_WIDTH
        self.padding_channel = 2 * (game.max_strands - 1)
        self.input_project = nn.Conv1d(game.observation_channels, self.width, 1)
        self.blocks = nn.ModuleList(
            _ZeroLayerScaleCyclicBlock(self.width, dilation) for dilation in V3_DILATIONS
        )
        features = 2 * self.width
        self.feature_norm = nn.LayerNorm(features)
        self.policy_residual = _zero_linear(nn.Linear(features, game.action_size))
        self.value_residual = _zero_linear(nn.Linear(features, 1))
        members = model.auxiliary_value_members
        self.solve_residual = _zero_linear(nn.Linear(features, members))
        self.cost_residual = _zero_linear(nn.Linear(features, 2 * members))
        self.budget_encoder = nn.Sequential(
            nn.Linear(2, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
            nn.SiLU(),
        )
        proof_features = features + self.width
        self.feasibility_norm = nn.LayerNorm(proof_features)
        self.feasibility_head = _zero_linear(nn.Linear(proof_features, 1))
        self.bound_head = _zero_linear(nn.Linear(features, 2))
        self.ordinal_head = _zero_linear(
            nn.Linear(features, V3_ORDINAL_MAX_U + 1)
        )
        self.auxiliary_members = members
        self._set_auxiliary_training_controls(game, model)

    @staticmethod
    def _parent_game(game: BraidGameConfig) -> BraidGameConfig:
        return replace(
            game,
            serial_encoder="cyclic-memory",
            serial_encoder_states=PARENT_ENCODER_WIDTH,
        )

    def load_parent_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load the exact admitted parent, including optional policy adapters."""

        load_policy_value_state_dict(self.parent, state)

    def _branch_observation(self, observation: Tensor) -> Tensor:
        return observation

    def _parent_observation(self, observation: Tensor) -> Tensor:
        return observation

    def _lengths_and_mask(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        occupied = 1.0 - observation[:, self.padding_channel, 0, :]
        lengths = occupied.sum(dim=1).long().clamp(min=1, max=observation.shape[-1])
        positions = torch.arange(observation.shape[-1], device=observation.device)[None, :]
        mask = (positions < lengths[:, None]).to(observation.dtype)
        return lengths, mask[:, None, :]

    def encode_v3(self, observation: Tensor) -> Tensor:
        source = self._branch_observation(observation)
        lengths, occupied = self._lengths_and_mask(source)
        hidden = F.silu(self.input_project(source[:, :, 0, :])) * occupied
        for block in self.blocks:
            hidden = block(hidden, lengths, occupied)
        count = occupied.sum(dim=2).clamp(min=1.0)
        mean = hidden.sum(dim=2) / count
        maximum = (hidden + (occupied - 1.0) * 1e4).amax(dim=2)
        return self.feature_norm(torch.cat([mean, maximum], dim=1))

    def normalized_budget_features(self, observation: Tensor) -> Tensor:
        """Return bounded linear and log-scaled remaining-L features.

        The legacy observation channel remains unchanged for exact checkpoint
        migration.  The logarithmic feature is v3-only and makes small L10 and
        L1000 budgets numerically visible even against the fixed global cap.
        """

        if self.objective_budget_channel is None:
            raise ValueError("mastery-v3 proof heads require objective_budget_channel")
        linear = observation[:, self.objective_budget_channel, 0, 0].clamp(0.0, 1.0)
        ratio = torch.exp(
            5.0 * observation[:, self.auxiliary_ratio_channel, 0, 0]
        )
        global_cap = (ratio + 1.0) * self.auxiliary_budget
        absolute = linear * global_cap
        logarithmic = torch.log1p(absolute) / torch.log1p(global_cap).clamp(min=1e-6)
        return torch.stack([linear, logarithmic.clamp(0.0, 1.0)], dim=1)

    def _proof_representation(self, observation: Tensor) -> Tensor:
        return self.encode_v3(observation)

    def proof_diagnostics(self, observation: Tensor) -> V3ProofDiagnostics:
        """Predict feasibility and certified-u bounds without redefining p_solve."""

        budget = self.normalized_budget_features(observation)
        representation = self._proof_representation(observation)
        feasibility_features = self.feasibility_norm(
            torch.cat(
                [representation, self.budget_encoder(budget)],
                dim=1,
            )
        )
        bounds = F.softplus(self.bound_head(representation))
        lower = bounds[:, 0]
        upper = lower + bounds[:, 1]
        return V3ProofDiagnostics(
            feasibility_logit=self.feasibility_head(feasibility_features).squeeze(-1),
            lower_bound=lower,
            upper_bound=upper,
            ordinal_logits=self.ordinal_head(representation),
            budget_linear=budget[:, 0],
            budget_log=budget[:, 1],
        )

    def _deltas(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        features = self.encode_v3(observation)
        solve_features = (
            features
            if self.auxiliary_backprop or self.auxiliary_solve_backprop
            else features.detach()
        )
        cost_features = features if self.auxiliary_backprop else features.detach()
        cost = self.cost_residual(cost_features).view(
            observation.shape[0], self.auxiliary_members, 2
        )
        return (
            self.policy_residual(features),
            self.value_residual(features).squeeze(-1),
            (
                self.solve_residual(solve_features),
                cost[:, :, 0],
                cost[:, :, 1],
            ),
        )

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        parent_policy, parent_value = self.parent(self._parent_observation(observation))
        policy_delta, value_delta, _ = self._deltas(observation)
        return parent_policy + policy_delta, (parent_value + value_delta).clamp(-1.0, 1.0)

    def forward_with_auxiliary(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        parent_policy, parent_value, parent_auxiliary = self.parent.forward_with_auxiliary(
            self._parent_observation(observation)
        )
        policy_delta, value_delta, auxiliary_delta = self._deltas(observation)
        return (
            parent_policy + policy_delta,
            (parent_value + value_delta).clamp(-1.0, 1.0),
            (
                parent_auxiliary[0] + auxiliary_delta[0],
                (parent_auxiliary[1] + auxiliary_delta[1]).clamp(min=0.0),
                (parent_auxiliary[2] + auxiliary_delta[2]).clamp(min=0.0),
            ),
        )

    def parameter_report(self) -> dict[str, int]:
        parent_ids = {id(parameter) for parameter in self.parent.parameters()}
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "parent": sum(parameter.numel() for parameter in self.parent.parameters()),
            "new": sum(
                parameter.numel()
                for parameter in self.parameters()
                if id(parameter) not in parent_ids
            ),
        }


class CyclicGraphDualV3(CyclicMemoryDeepV3):
    """Cyclic word memory plus bounded strand graph and invariant conditioning."""

    encoder_name = "cyclic-graph-dual-v3"

    def __init__(self, game: BraidGameConfig, model: ModelConfig):
        if game.max_strands != 12:
            raise ValueError("cyclic-graph-dual-v3 is pinned to 12-strand capacity")
        if not game.invariant_features or game.invariant_fusion != "dual":
            raise ValueError("cyclic-graph-dual-v3 requires explicit dual invariant features")
        super().__init__(game, model)
        from pgx_mcts_bench.invariant_features import invariant_feature_size

        self.max_strands = game.max_strands
        self.generators = game.max_strands - 1
        self.invariant_dim = invariant_feature_size(game.invariant_features)
        trailing = 1 + int(game.objective_budget_channel)
        self.invariant_end = game.observation_channels - trailing
        self.invariant_start = self.invariant_end - self.invariant_dim

        # The explicit graph is built from signed generator incidence.  Every
        # ordinary Artin generator connects exactly one adjacent row pair.
        incidence = torch.zeros(self.max_strands, self.generators)
        for generator in range(self.generators):
            incidence[generator, generator] = 1.0
            incidence[generator + 1, generator] = 1.0
        self.register_buffer("strand_generator_incidence", incidence)
        self.node_input = nn.Linear(4, self.width)
        self.graph_blocks = nn.ModuleList(
            _BoundedStrandGraphBlock(self.width) for _ in range(6)
        )
        self.invariant_encoder = nn.Sequential(
            nn.Linear(self.invariant_dim, self.width),
            nn.SiLU(),
            _zero_linear(nn.Linear(self.width, self.width)),
        )
        self.graph_fusion = nn.Sequential(
            nn.LayerNorm(5 * self.width),
            nn.Linear(5 * self.width, 2 * self.width),
            nn.SiLU(),
        )
        self.row_pair_policy = nn.Sequential(
            nn.Linear(3 * self.width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, 1),
        )
        self.graph_policy_residual = _zero_linear(nn.Linear(2 * self.width, game.action_size))
        self.graph_value_residual = _zero_linear(nn.Linear(2 * self.width, 1))
        self.graph_solve_residual = _zero_linear(
            nn.Linear(2 * self.width, self.auxiliary_members)
        )
        self.graph_cost_residual = _zero_linear(
            nn.Linear(2 * self.width, 2 * self.auxiliary_members)
        )
        self.row_pair_gate = nn.Parameter(torch.zeros(self.generators))
        self.invalid_capacity_head = nn.Linear(2 * self.width, 2)

        # Generator-specific INSERT actions are the semantic place where the
        # shared row-pair scorer enters the legacy action space.
        per_offset = 3 + 2 * self.generators + 1
        action_rows = []
        for offset in range(game.serial_width):
            base = offset * per_offset + 3
            action_rows.append([base + 2 * generator for generator in range(self.generators)])
            action_rows.append(
                [base + 2 * generator + 1 for generator in range(self.generators)]
            )
        self.register_buffer(
            "row_pair_action_rows", torch.tensor(action_rows, dtype=torch.long)
        )

    @staticmethod
    def _parent_game(game: BraidGameConfig) -> BraidGameConfig:
        return replace(
            game,
            serial_encoder="cyclic-memory",
            serial_encoder_states=PARENT_ENCODER_WIDTH,
            invariant_features="",
            invariant_fusion="late",
        )

    def _parent_observation(self, observation: Tensor) -> Tensor:
        return torch.cat(
            [
                observation[:, : self.invariant_start],
                observation[:, self.invariant_end :],
            ],
            dim=1,
        )

    def _strand_features(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        lengths, _ = self._lengths_and_mask(observation)
        positive = observation[:, : self.generators, 0, :].sum(dim=2)
        negative = observation[:, self.generators : 2 * self.generators, 0, :].sum(dim=2)
        scale = lengths.to(observation.dtype)[:, None].clamp(min=1.0)
        positive = positive / scale
        negative = negative / scale
        unsigned = positive + negative
        signed = positive - negative
        node_unsigned = unsigned @ self.strand_generator_incidence.T
        node_signed = signed @ self.strand_generator_incidence.T
        strand_index = torch.linspace(
            -1.0, 1.0, self.max_strands, device=observation.device, dtype=observation.dtype
        )[None, :].expand(observation.shape[0], -1)
        strands = 1 + (
            observation[:, : 2 * self.generators, 0, :].argmax(dim=1)
            % self.generators
        ).amax(dim=1)
        active = (
            torch.arange(self.max_strands, device=observation.device)[None, :]
            < strands[:, None]
        ).to(observation.dtype)
        inputs = torch.stack([node_unsigned, node_signed, strand_index, active], dim=2)
        nodes = F.silu(self.node_input(inputs)) * active[:, :, None]
        for block in self.graph_blocks:
            nodes = block(nodes, active)
        return nodes, active

    def _graph_features(
        self, observation: Tensor
    ) -> tuple[Tensor, V3Diagnostics]:
        nodes, active = self._strand_features(observation)
        count = active.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = nodes.sum(dim=1) / count
        maximum = (nodes + (active[:, :, None] - 1.0) * 1e4).amax(dim=1)
        invariant = observation[:, self.invariant_start : self.invariant_end, 0, 0]
        invariant_features = self.invariant_encoder(invariant)
        cyclic = self.encode_v3(observation)
        features = self.graph_fusion(
            torch.cat([cyclic, mean, maximum, invariant_features], dim=1)
        )
        left = nodes[:, :-1]
        right = nodes[:, 1:]
        pair_context = features[:, None, : self.width].expand(-1, self.generators, -1)
        row_pairs = self.row_pair_policy(torch.cat([left, right, pair_context], dim=2)).squeeze(-1)
        invalid_capacity = self.invalid_capacity_head(features)
        return features, V3Diagnostics(
            row_pair_logits=row_pairs,
            invalid_logit=invalid_capacity[:, 0],
            capacity_logit=invalid_capacity[:, 1],
        )

    def _graph_deltas(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor], V3Diagnostics]:
        features, diagnostics = self._graph_features(observation)
        policy = self.graph_policy_residual(features)
        gated_pairs = diagnostics.row_pair_logits * self.row_pair_gate[None, :]
        for rows in self.row_pair_action_rows:
            policy[:, rows] = policy[:, rows] + gated_pairs
        costs = self.graph_cost_residual(features).view(
            observation.shape[0], self.auxiliary_members, 2
        )
        return (
            policy,
            self.graph_value_residual(features).squeeze(-1),
            (
                self.graph_solve_residual(features),
                costs[:, :, 0],
                costs[:, :, 1],
            ),
            diagnostics,
        )

    def _proof_representation(self, observation: Tensor) -> Tensor:
        return self._graph_features(observation)[0]

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        policy, value = super().forward(observation)
        policy_delta, value_delta, _, _ = self._graph_deltas(observation)
        return policy + policy_delta, (value + value_delta).clamp(-1.0, 1.0)

    def forward_with_auxiliary(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        policy, value, auxiliary = super().forward_with_auxiliary(observation)
        policy_delta, value_delta, auxiliary_delta, _ = self._graph_deltas(observation)
        return (
            policy + policy_delta,
            (value + value_delta).clamp(-1.0, 1.0),
            (
                auxiliary[0] + auxiliary_delta[0],
                (auxiliary[1] + auxiliary_delta[1]).clamp(min=0.0),
                (auxiliary[2] + auxiliary_delta[2]).clamp(min=0.0),
            ),
        )

    def diagnostics(self, observation: Tensor) -> V3Diagnostics:
        """Return the registered invalid/capacity and shared row-pair heads."""

        return self._graph_features(observation)[1]


def make_mastery_v3_network(
    game: BraidGameConfig, model: ModelConfig
) -> CyclicMemoryDeepV3 | CyclicGraphDualV3:
    if game.serial_encoder == CyclicMemoryDeepV3.encoder_name:
        return CyclicMemoryDeepV3(game, model)
    if game.serial_encoder == CyclicGraphDualV3.encoder_name:
        return CyclicGraphDualV3(game, model)
    raise ValueError(f"unknown mastery-v3 encoder: {game.serial_encoder!r}")


def migration_max_abs_difference(
    parent: BraidPolicyValueNet,
    child: CyclicMemoryDeepV3,
    parent_observation: Tensor,
    child_observation: Tensor | None = None,
) -> dict[str, float]:
    """Compare policy, value, and factorized heads for the migration gate."""

    if child_observation is None:
        child_observation = parent_observation
    parent.eval()
    child.eval()
    with torch.inference_mode():
        old_policy, old_value, old_auxiliary = parent.forward_with_auxiliary(
            parent_observation
        )
        new_policy, new_value, new_auxiliary = child.forward_with_auxiliary(
            child_observation
        )
    values: dict[str, Any] = {
        "policy": (old_policy - new_policy).abs().max(),
        "value": (old_value - new_value).abs().max(),
        "p_solve_logits": (old_auxiliary[0] - new_auxiliary[0]).abs().max(),
        "conditional_crossings": (old_auxiliary[1] - new_auxiliary[1]).abs().max(),
        "conditional_moves": (old_auxiliary[2] - new_auxiliary[2]).abs().max(),
    }
    return {name: float(value.item()) for name, value in values.items()}
