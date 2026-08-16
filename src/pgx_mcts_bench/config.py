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
    """Bounded serial unknotting on braid words (the `rf_knots` environment)."""

    kind: GameKind = "braid"
    max_len: int = 32
    max_strands: int = 5
    scramble_budget: int = 6
    simplify_budget: int = 24
    allow_crossing_change: bool = False
    multi_objective: bool = False
    log_ratio_range: tuple[float, float] = (0.0, 0.0)
    # Optional discrete objective curriculum. When non-empty, generated training
    # games sample exactly these A:B ratios rather than every log-ratio in the
    # enclosing range. Weights are normalized at sampling time.
    objective_ratio_choices: tuple[float, ...] = ()
    objective_ratio_weights: tuple[float, ...] = ()
    # Instances come from the graded generator rather than from a Scrambler
    # phase: a torus knot of at most this crossing number, plus up to this many
    # scramble moves. u(T(p,q)) = (p-1)(q-1)/2 is a theorem, so the optimal
    # number of crossing changes is known exactly for every instance.
    generator_max_crossings: int = 0
    generator_max_scramble: int = 0
    # Random positive braids per (strands, crossings) grade, with u known exactly
    # from u = (c - s + 1) / 2. Torus knots give one diagram per unknotting
    # number; this gives as many as asked for, which is the axis the ladder runs
    # out of first -- four arms cleared all 17 rungs. 0 keeps the torus-only
    # source list, so an existing run is unaffected.
    generator_positive_braids: int = 0
    generator_positive_seed: int = 0
    # Random **mixed-sign** knots, whose u is not known. These carry no label and
    # are scored against the ratcheting best-known bound instead. They exist
    # because every labelled family is a structured one, and an agent can master
    # all of them without learning anything general.
    generator_random_crossings: tuple[int, ...] = ()
    generator_random_per_grade: int = 1
    generator_random_seed: int = 0
    # Pin the generator to one source knot and scramble depth. Set by the ladder
    # runner so a stage is a fixed difficulty rather than a mixture. This stays
    # the *frontier* stage even when training mixes: promotion is measured here.
    stage_source: str = ""
    stage_scramble: int = 0
    # Training-instance mixture as (source, scramble, weight). Empty pins training
    # to the frontier, which is what the first ladder did. Promoting used to mean
    # abandoning a stage, and the measurement said that costs real quality: with
    # the frontier-only rule, `s-window-128` promoted stage 3 at 4.18 crossing
    # changes against an optimum of 1, and its final weights -- after eight more
    # stages, none of them stage 3 -- still scored 1.17. Transfer from harder
    # stages improves the easier ones but does not converge them. Mixing keeps the
    # cleared stages in the training distribution so that gap closes.
    # Evaluation never uses this; it always pins to the frontier.
    stage_mix: tuple[tuple[str, int, float], ...] = ()
    # The agent sees a window of this width around a moving head.  The retired
    # position-indexed formulation is intentionally not selectable here.
    serial_window: int = 7
    # How many window offsets the agent may act at. 1 = only at the head
    # (position lives in the state); serial_window = act anywhere it can see.
    serial_act_width: int = 1
    # Head strides, in letters, each giving a left and a right action. Empty
    # takes the powers of two that fit `max_len`, so any site is reachable in
    # O(log L) plies. A single stride reproduces the original one-step tape.
    serial_shift_strides: tuple[int, ...] = ()
    # Binary registers carried in the head, with one TOGGLE action each. This is
    # the *finite control state* of a Turing machine -- the half that a memoryless
    # scanning head is missing, and the half that costs nothing to make exact. The
    # agent writes them; nothing is computed for it, so this stays a fair arm under
    # the zero-human-knowledge constraint. K registers give 2^K control states and
    # add K actions and K observation channels, both O(1) in word length.
    serial_registers: int = 0
    # Colours the agent may paint onto strands. Unlike a register, a colour is
    # attached to a *strand* rather than to a slot, so the environment transports
    # it: when the head crosses a letter the two strands it swaps carry their
    # colours with them. That makes the colour follow the thread with no seam
    # dependence and no compaction problem -- a mark on the tape would have to be
    # re-indexed every time the word compacts, and a colour indexed from the word
    # start would move under conjugation, which is a free move.
    #
    # Three actions regardless of how many colours: PAINT_LOW, PAINT_HIGH, CYCLE.
    # Deliberately not `strands x colours` paint actions -- `s-reg8` showed that
    # actions which cannot change the word dilute the search, and it cost the arm
    # everything above rung 0.
    serial_colours: int = 0
    # An explicit symbol tape aligned one-for-one with the occupied braid word.
    # A shift writes one of K symbols at the cell it leaves, then moves the head.
    # Rewrites transport, insert, and delete symbols with their corresponding
    # crossings, so the annotation remains a replayable part of the Markov state.
    # 0 disables it; useful experimental alphabets are 2 and 4 (symbol 0 blank).
    serial_tape_symbols: int = 0
    # A tape shift normally writes one of the K symbols, including blank.  A
    # composite controller also needs the shift used by a non-tape parent: move
    # the head while leaving the existing annotation untouched.  Keeping that
    # as a distinct action avoids silently translating "preserve" into "erase".
    serial_tape_preserve_shift: bool = False
    # Optional automatic whole-tape accumulator. The action space remains the
    # serial O(1) head action space, but the observation is the occupied braid
    # word rotated to start at the head and padded to max_len. This is the common
    # interface for recurrent, finite-state, learned finite-field, and known-
    # representation oracle arms.
    serial_encoder: str = ""
    serial_encoder_states: int = 0
    serial_encoder_prime: int = 5
    # Append a lossless strand-by-word raster to a serial window.  Each row has
    # the user's three route bits plus an explicit active-strand mask.  The word
    # axis is cyclic; the strand axis is deliberately bounded (an ordinary
    # Artin braid is a cylinder, not an affine braid on a torus).
    # Values: "joint" (3x3 blocks), "axial" (separate horizontal/vertical
    # interactions), "recurrent" (one axial block reused four times), or
    # "scalable" (recurrent trunk plus a shared row-pair policy scorer).
    serial_raster: str = ""
    # Make the strand axis circular inside the raster trunk.  This is only
    # semantically appropriate for an explicitly cyclic-band (B*) action
    # alphabet; ordinary Artin braids retain a bounded strand axis.
    serial_raster_wrap_strands: bool = False
    # On a scalable full raster, encode unused word-capacity columns as active
    # all-straight identity slices instead of inactive zero padding.  This is a
    # representation augmentation, not a semantic action: decoding deletes the
    # slices and recovers the same Artin word.
    serial_raster_identity_padding: bool = False
    # Normalise the raster trunk over live cells only.  Plain GroupNorm averages
    # over the padded canvas, which on a 50-82% inactive raster is mostly zeros.
    serial_raster_masked_norm: bool = False
    # Residual formulation for raster encoders. ``standard`` preserves the
    # published baseline; ``layerscale`` stabilizes a deeper axial stack with a
    # learnable per-channel residual gate initialized near the identity.
    serial_raster_residual_style: str = "standard"
    # Optional Markov-equivalent taller initial representation.  Each step adds
    # one new strand together with the required terminal sigma_k crossing; a
    # bare active 010 row would instead add an unknot component.
    serial_initial_markov_stabilizations: int = 0
    serial_initial_markov_sign: int = 1
    # Frozen-window + full-scan + writable-tape policy ensemble.  The non-empty
    # value selects the composite network and makes the environment expose the
    # complete head-relative word from which all three parent views are derived.
    serial_ensemble: str = ""
    # Maximum consecutive head/memory/tape operations before an external braid
    # action is required.  This bound is mandatory: an unbounded controller can
    # create non-terminating search paths consisting only of native actions.
    serial_internal_horizon: int = 5
    # Historical distilled checkpoints encode the fraction of internal steps
    # already spent.  New budget-aware arms can instead expose the equivalent
    # fraction remaining without changing those old checkpoint semantics.
    serial_internal_budget_remaining: bool = True
    # Append a broadcast remaining-objective-budget plane. The collaboration
    # wrapper supplies a task-specific cap; ordinary games expose the equivalent
    # global move-derived cap. Kept opt-in so historical checkpoint schemas stay
    # exact unless an experiment explicitly migrates them.
    objective_budget_channel: bool = False
    # Clamp search values to a theorem-certified lower bound on the remaining
    # number of crossing changes. This changes search, not the observation or
    # trainable network, and requires the rf-knots ``bounds`` extra.
    certified_value_floor: bool = False
    # Human-computed knot invariants broadcast into the serial observation.
    # This is an explicitly labelled oracle arm, never a silent addition to the
    # zero-knowledge scientists. Values: classical, alexander, jones, combined.
    invariant_features: str = ""
    # How the raster controller consumes the global invariant vector: ``late``
    # (value/global heads), ``film`` (feature modulation), or ``dual`` (a
    # separate tower fused into positional policy and value).
    invariant_fusion: str = "late"
    # Extend the Artin alphabet with the verified cyclic seam band a_{1,n}.
    # Witnesses compile back to ordinary B_n; see rf_knots.reference.
    cyclic_band_generators: bool = False

    def __post_init__(self) -> None:
        if self.serial_window < 3:
            raise ValueError("serial_window must be at least 3")
        if self.serial_internal_horizon < 1:
            raise ValueError("serial_internal_horizon must be positive")
        from pgx_mcts_bench.invariant_features import invariant_feature_size

        invariant_feature_size(self.invariant_features)
        if self.invariant_fusion not in {"late", "film", "dual"}:
            raise ValueError(f"unknown invariant fusion {self.invariant_fusion!r}")
        invariant_serial_encoders = {"cyclic-graph-dual-v3"}
        if (
            self.invariant_features
            and not self.serial_raster
            and self.serial_encoder not in invariant_serial_encoders
        ):
            raise ValueError(
                "invariant oracle features require a raster scientist or a registered "
                "invariant-aware serial encoder"
            )

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
            cyclic_band_generators=self.cyclic_band_generators,
        )

    @property
    def _spec(self):
        from rf_knots.actions import ActionSpec

        return ActionSpec(
            max_len=self.max_len,
            max_strands=self.max_strands,
            cyclic_band_generators=self.cyclic_band_generators,
        )

    @property
    def generator_capacity(self) -> int:
        return self.max_strands - 1 + int(self.cyclic_band_generators)

    @property
    def serial_strides(self) -> tuple[int, ...]:
        from pgx_mcts_bench.serial_braid import shift_strides

        return shift_strides(self.serial_window, self.max_len, self.serial_shift_strides)

    @property
    def serial_width(self) -> int:
        return min(max(self.serial_act_width, 1), self.serial_window)

    @property
    def action_size(self) -> int:
        from pgx_mcts_bench.serial_braid import serial_action_size

        return serial_action_size(
            self.max_strands,
            self.serial_width,
            len(self.serial_strides),
            self.serial_registers,
            self.serial_colours,
            self.serial_tape_symbols,
            self.serial_tape_preserve_shift,
            self.cyclic_band_generators,
        )

    @property
    def observation_channels(self) -> int:
        # letter one-hot (+/- each generator), padding, the top-generator
        # marker that makes DESTABILIZE legality locally visible, and eight
        # broadcast scalars (the last two are log(A/B) and crossing changes so far),
        # plus one broadcast plane per head register when the serial formulation
        # carries them.
        # Colours are one-hot per height plus the colour the agent is holding.
        # One-hot rather than a scalar: colour ids are labels, and a scalar would
        # tell the network that colour 1 is nearer colour 2 than colour 3.
        colours = self.serial_colours * (self.max_strands + 1) if self.serial_colours else 0
        tape = max(self.serial_tape_symbols, 0)
        # The strand-graph encoder receives four pointer planes per crossing:
        # previous/next crossing along each of the two physical strands.  They
        # are constructed by the serial adapter's deterministic full-word scan.
        strand_graph = 4 if self.serial_encoder.startswith("strand-graph") else 0
        raster = 4 * self.max_strands if self.serial_raster else 0
        from pgx_mcts_bench.invariant_features import invariant_feature_size

        invariants = invariant_feature_size(self.invariant_features)
        return (
            2 * self.generator_capacity
            + 1
            + 1
            + 8
            + self.serial_registers
            + colours
            + tape
            + strand_graph
            + raster
            + invariants
            + 1
            + int(self.objective_budget_channel)
        )

    @property
    def height(self) -> int:
        return 1

    @property
    def width(self) -> int:
        if self.serial_encoder or self.serial_ensemble or self.serial_raster == "scalable":
            return self.max_len
        return self.serial_window

    @property
    def cells(self) -> int:
        return self.width

    @property
    def max_moves(self) -> int:
        return self.scramble_budget + self.simplify_budget

    @property
    def terminal_action(self) -> int:
        return self.serial_width * (3 + 2 * self.generator_capacity + 1) + 3  # PASS

    @property
    def opening_moves(self) -> int:
        # A few random scramble plies so arena games are not all identical at
        # temperature 0; small enough to leave the Scrambler most of its budget.
        return max(1, self.scramble_budget // 4)


AnyGameConfig = GameConfig | BraidGameConfig


def pick_stage(config: BraidGameConfig, generator: Any, rng: Any):
    """`(source, scramble)` for one training instance.

    Three regimes, in precedence order: an explicit mixture over cleared stages,
    a single pinned stage, or the generator's own full grading.
    """
    if config.stage_mix:
        import numpy as np

        weights = np.array([w for _, _, w in config.stage_mix], dtype=float)
        index = int(rng.choice(len(weights), p=weights / weights.sum()))
        name, scramble, _ = config.stage_mix[index]
        return next(s for s in generator.sources if s.name == name), scramble
    if config.stage_source:
        source = next(s for s in generator.sources if s.name == config.stage_source)
        return source, config.stage_scramble
    levels = generator.levels(config.generator_max_scramble)
    return levels[int(rng.integers(len(levels)))]


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
    # Potential-based cost shaping for the braid objective. The environment's
    # terminal return is unchanged, but search receives exact incremental cost
    # and the value head learns the corresponding remaining-return target.
    # Disabled by default so historical checkpoints retain their value meaning.
    potential_cost_shaping: bool = False


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
    # Function-preserving architecture mutations for invariant-oracle children.
    # Each residual block has a zero-initialized final projection; enabling a
    # block therefore leaves a migrated parent's outputs exact at fork time.
    invariant_residual_blocks: int = 0
    # Add a zero-initialized FiLM correction to a dual-fusion invariant tower.
    # This tests whether the successful separate tower also benefits from
    # modulating the visual trunk without replacing either learned parent path.
    invariant_dual_film: bool = False
    # Shadow factorized critic. Four independently initialized towers are trained
    # with deterministic per-episode bootstrap masks while MCTS continues to use
    # the legacy scalar value. Auxiliary gradients are initially kept out of the
    # shared encoder so checkpoint migration cannot damage an established policy.
    auxiliary_value_members: int = 4
    auxiliary_value_width: int = 64
    auxiliary_value_loss_weight: float = 0.1
    auxiliary_backprop_to_encoder: bool = False
    # Budget-aware training may let only the solve-probability loss update the
    # shared representation. Conditional cost targets stay detached: a failed
    # cap is a valid label for "solve within this budget", but says nothing
    # about the eventual crossing/move cost of a successful trajectory.
    auxiliary_solve_backprop_to_encoder: bool = False
    # Paired copies of the same state at lower/higher remaining budgets enforce
    # p(solve | lower budget) <= p(solve | higher budget). Disabled for legacy
    # runs and enabled explicitly when the objective-budget input is present.
    auxiliary_budget_monotonic_weight: float = 0.0
    auxiliary_budget_monotonic_margin: float = 0.05
    # First-stage prototype for s-window-128. Predict conditional costs first,
    # construct L=A*cc+B*moves exactly, and feed those quantities plus the
    # remaining-budget input into a residual solve-probability branch.
    auxiliary_budget_conditioning: bool = False
    # Per-task budget fine-tuning has a much narrower state distribution than
    # ladder training. Preserve the promoted checkpoint's BatchNorm statistics
    # while still allowing gradients through its affine parameters and encoder.
    freeze_batchnorm_stats: bool = False
    # Optional functional trust region used by narrow budget-critic curricula.
    # The frozen teacher is attached at runtime and is deliberately not part of
    # the checkpoint state.  This lets solve loss improve the shared encoder
    # while penalizing changes to the established policy and scalar value.
    policy_value_preservation_weight: float = 0.0
    # Deliberate, checkpointed cut-over switch. False keeps MCTS on the legacy
    # scalar while the auxiliary critic trains and is evaluated in shadow mode.
    use_auxiliary_value: bool = False


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
    curriculum_start_k: int = 0  # 0 disables; else start here and climb
    curriculum_promote_at: float = 0.5  # Simplifier self-play win rate to promote
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
