from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

AgentKind = Literal["alphazero", "muzero"]
ExplorationKind = Literal["u1", "u2", "u3", "u4", "u5"]
GameKind = Literal["go6x6", "braid"]


@dataclass(frozen=True)
class GameConfig:
    kind: GameKind = "go6x6"
    board_size: int = 6
    komi: float = 3.5
    history_length: int = 4
    max_moves: int = 72
    min_moves_before_pass: int = 24

    @property
    def action_size(self) -> int:
        return self.board_size**2 + 1

    @property
    def observation_channels(self) -> int:
        # Pgx planes plus consecutive-pass and normalized move-count metadata.
        # The latter two are required by learned dynamics to model termination.
        return self.history_length * 2 + 3

    # Spatial layout of the observation, so networks and search do not have to
    # assume a square board.
    @property
    def height(self) -> int:
        return self.board_size

    @property
    def width(self) -> int:
        return self.board_size

    @property
    def cells(self) -> int:
        return self.height * self.width

    @property
    def terminal_action(self) -> int:
        """Action that MuZero's learned-rules path treats as a pass."""
        return self.board_size**2

    @property
    def opening_moves(self) -> int:
        """Random plies played before an arena game, to diversify openings."""
        return 6


@dataclass(frozen=True)
class BraidGameConfig:
    """Scrambler vs. Simplifier on braid words (the `rf_knots` environment).

    Mirrors `rf_knots.BraidConfig`; kept as a separate dataclass so that
    `ExperimentConfig` stays serialisable and checkpoint comparison keeps working.
    """

    kind: GameKind = "braid"
    max_len: int = 32
    max_strands: int = 5
    scramble_budget: int = 6
    simplify_budget: int = 24
    allow_crossing_change: bool = False
    multi_objective: bool = False
    log_ratio_range: tuple[float, float] = (0.0, 0.0)
    # > 0 selects the serial (moving-window) formulation: the agent sees a window
    # of this width around a head it must move, and the action space stops
    # depending on L. See serial_braid.py.
    serial_window: int = 0
    # How many window offsets the agent may act at. 1 = only at the head
    # (position lives in the state); serial_window = act anywhere it can see.
    serial_act_width: int = 1

    def to_braid_config(self):
        from rf_knots.config import BraidConfig

        return BraidConfig(
            max_len=self.max_len,
            max_strands=self.max_strands,
            scramble_budget=self.scramble_budget,
            simplify_budget=self.simplify_budget,
            allow_crossing_change=self.allow_crossing_change,
            multi_objective=self.multi_objective,
            log_ratio_range=self.log_ratio_range,
        )

    @property
    def _spec(self):
        from rf_knots.actions import ActionSpec

        return ActionSpec(max_len=self.max_len, max_strands=self.max_strands)

    @property
    def action_size(self) -> int:
        if self.serial_window:
            from pgx_mcts_bench.serial_braid import serial_action_size

            return serial_action_size(
                self.max_strands, min(max(self.serial_act_width, 1), self.serial_window)
            )
        return self._spec.num_actions

    @property
    def observation_channels(self) -> int:
        # letter one-hot (+/- each generator), padding, the top-generator
        # marker that makes DESTABILIZE legality locally visible, and eight
        # broadcast scalars (the last two are log(A/B) and crossing changes so far)
        return 2 * (self.max_strands - 1) + 1 + 1 + 8

    @property
    def height(self) -> int:
        return 1

    @property
    def width(self) -> int:
        return self.serial_window or self.max_len

    @property
    def cells(self) -> int:
        return self.serial_window or self.max_len

    @property
    def max_moves(self) -> int:
        return self.scramble_budget + self.simplify_budget

    @property
    def terminal_action(self) -> int:
        from rf_knots.actions import PASS

        if self.serial_window:
            width = min(max(self.serial_act_width, 1), self.serial_window)
            return width * (3 + 2 * (self.max_strands - 1) + 1) + 3  # PASS
        return self._spec.start_of(PASS)

    @property
    def opening_moves(self) -> int:
        # A few random scramble plies so arena games are not all identical at
        # temperature 0; small enough to leave the Scrambler most of its budget.
        return max(1, self.scramble_budget // 4)


AnyGameConfig = GameConfig | BraidGameConfig


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
    muzero_exact_rules: bool = True


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 32
    residual_blocks: int = 2
    latent_channels: int = 32
    # Braid value head. Pooling makes the network length-agnostic, which is
    # what the curriculum needs; flattening reads every position directly and
    # may be more accurate at a fixed L. Which one wins is an empirical
    # question, so it is a switch rather than an assumption.
    # Condition the braid network on log(A/B) multiplicatively rather than by
    # appending a channel, so the two ends of the Pareto front can behave like
    # different networks inside one set of weights.
    film_on_ratio: bool = True


@dataclass(frozen=True)
class TrainConfig:
    iterations: int = 3
    selfplay_games: int = 2
    selfplay_positions_per_iteration: int = 0
    train_steps: int = 16
    batch_size: int = 32
    replay_capacity: int = 10_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    unroll_steps: int = 3
    temperature_moves: int = 12
    # Braid game: raise the Scrambler's budget K only once the Simplifier is
    # actually winning. Without this a run whose Simplifier never wins a single
    # self-play game has every training target reading "Simplifier loses" -- the
    # value head learns a constant and no gradient points toward solving. That
    # was 4 of 42 runs at a fixed K=6, and they were unrecoverable from
    # iteration 1.
    # The Scrambler is a fixed uniform-random policy, not a learned adversary:
    # measured over 8 seeds it was indistinguishable from random (+0.15, 95% CI
    # [-0.06, +0.35]) and collapsed to worse than random on some seeds. Only
    # Simplifier positions are trained on.
    random_first_role: bool = False
    curriculum_start_k: int = 0          # 0 disables; else start here and climb
    curriculum_promote_at: float = 0.5   # Simplifier self-play win rate to promote
    seed: int = 0
    device: str = "cpu"
    exact_position_budget: bool = True
    checkpoint_iterations: tuple[int, ...] = ()
    learning_curve_games: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    game: AnyGameConfig = GameConfig()
    search: SearchConfig = SearchConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        train = dict(payload["train"])
        train["checkpoint_iterations"] = tuple(train.get("checkpoint_iterations", ()))
        game_payload = dict(payload["game"])
        # Checkpoints written before the braid environment existed have no kind.
        game_kind = game_payload.get("kind", "go6x6")
        game: AnyGameConfig
        if game_kind == "braid":
            game = BraidGameConfig(**game_payload)
        else:
            game = GameConfig(**game_payload)
        return cls(
            game=game,
            search=SearchConfig(**payload["search"]),
            model=ModelConfig(**payload["model"]),
            train=TrainConfig(**train),
        )


def artifact_dir(root: Path, label: str) -> Path:
    out = root / "artifacts" / label
    out.mkdir(parents=True, exist_ok=True)
    return out
