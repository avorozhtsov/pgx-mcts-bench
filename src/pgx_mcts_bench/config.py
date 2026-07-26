from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

AgentKind = Literal["alphazero", "muzero"]
ExplorationKind = Literal["u1", "u2", "u3", "u4", "u5"]


@dataclass(frozen=True)
class GameConfig:
    board_size: int = 6
    komi: float = 3.5
    history_length: int = 4
    max_moves: int = 72

    @property
    def action_size(self) -> int:
        return self.board_size**2 + 1

    @property
    def observation_channels(self) -> int:
        return self.history_length * 2 + 1


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 32
    exploration: ExplorationKind = "u1"
    cpuct: float = 1.5
    c1: float = 1.25
    c2: float = 19652.0
    discount: float = 1.0
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 32
    residual_blocks: int = 2
    latent_channels: int = 32


@dataclass(frozen=True)
class TrainConfig:
    iterations: int = 3
    selfplay_games: int = 2
    train_steps: int = 16
    batch_size: int = 32
    replay_capacity: int = 10_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    unroll_steps: int = 3
    temperature_moves: int = 12
    seed: int = 0
    device: str = "cpu"


@dataclass(frozen=True)
class ExperimentConfig:
    game: GameConfig = GameConfig()
    search: SearchConfig = SearchConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

    def to_dict(self) -> dict:
        return asdict(self)


def artifact_dir(root: Path, label: str) -> Path:
    out = root / "artifacts" / label
    out.mkdir(parents=True, exist_ok=True)
    return out
