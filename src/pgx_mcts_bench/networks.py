from __future__ import annotations

import torch
from torch import Tensor, nn

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


class BraidAlphaZeroNet(PolicyValueNet):
    """AlphaZero network for the braid environment.

    The observation is a `1 x L` one-row image, so the shared residual stack
    applies unchanged; only the policy head differs.
    """

    def __init__(self, game: BraidGameConfig, model: ModelConfig):
        super().__init__()
        self.representation = Representation(game, model, model.channels)
        self.policy_head = BraidPolicyHead(model.channels, game)
        # Pooled, not flattened. `Flatten -> Linear(L, ...)` would tie the value
        # head to one word capacity, defeating the point: every other parameter
        # here depends on the receptive field (11 letters), not on L, so weights
        # trained at one capacity load and run at any other. Mean pooling keeps
        # the scalar planes exact -- they are constant along the word -- and max
        # pooling preserves "does any position have this feature", which mean
        # alone washes out on a mostly padded array.
        self.value_mode = model.braid_value_head
        if self.value_mode not in ("flat", "pooled", "masked"):
            raise ValueError(f"unknown braid_value_head: {self.value_mode}")
        # The padding plane is the last of the letter one-hots.
        self.padding_channel = 2 * (game.max_strands - 1)
        if self.value_mode != "flat":
            self.value_project = nn.Conv2d(model.channels, model.channels, 1)
            self.value_head = nn.Sequential(
                nn.Linear(2 * model.channels, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh(),
            )
        else:
            self.value_project = nn.Conv2d(model.channels, 1, 1)
            self.value_head = nn.Sequential(
                nn.Linear(game.cells, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh(),
            )

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.representation(observation)
        projected = torch.relu(self.value_project(hidden))
        if self.value_mode == "flat":
            summary = projected.flatten(1)
        elif self.value_mode == "pooled":
            summary = torch.cat(
                [projected.mean(dim=(2, 3)), projected.amax(dim=(2, 3))], dim=1
            )
        else:
            # Average over occupied positions only. Averaging over all L slots
            # dilutes a 5-letter word by ~6x at tier 0, and the A/B showed that
            # dilution is enough to make runs collapse.
            occupied = 1.0 - observation[:, self.padding_channel : self.padding_channel + 1]
            count = occupied.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
            masked_mean = (projected * occupied).sum(dim=(2, 3), keepdim=True) / count
            masked_max = (projected + (occupied - 1.0) * 1e4).amax(dim=(2, 3))
            summary = torch.cat([masked_mean.flatten(1), masked_max], dim=1)
        return self.policy_head(hidden), self.value_head(summary).squeeze(-1)


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
