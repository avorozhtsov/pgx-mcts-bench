from __future__ import annotations

import torch
from torch import Tensor, nn

from pgx_mcts_bench.config import GameConfig, ModelConfig


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
    def __init__(self, game: GameConfig, model: ModelConfig, output_channels: int):
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
        game: GameConfig,
        *,
        include_legal: bool,
        include_terminal: bool,
    ):
        super().__init__()
        cells = game.board_size**2
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


class AlphaZeroNet(nn.Module):
    def __init__(self, game: GameConfig, model: ModelConfig):
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


class Dynamics(nn.Module):
    def __init__(self, game: GameConfig, model: ModelConfig):
        super().__init__()
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
            nn.Linear(game.board_size**2, 1),
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

    def __init__(self, game: GameConfig, model: ModelConfig):
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
