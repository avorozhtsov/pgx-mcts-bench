"""Curriculum mechanics: stage grading, mixing, resume, and the promotion rule.

These four are the parts of the ladder that measurements changed, and each has a
failure mode that produces plausible numbers rather than an error -- a resume onto
the wrong rung, a mixture that quietly drops a stage, a plateau test that fires on
the first flat evaluation. So they are tested directly rather than through a run.
"""

from __future__ import annotations

import numpy as np

from pgx_mcts_bench.config import pick_stage
from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import (
    STAGES,
    _config,
    parallel_arms,
    promotion_reason,
    resume_point,
    stage_mixture,
)


def test_stages_are_monotone_in_unknotting_number_and_scramble() -> None:
    """The ladder's only structural promise. A stage that is easier than the one
    below it makes "highest stage reached" meaningless as a score."""
    game = make_game(_config(parallel_arms()[1], STAGES[0], 0, "cpu").game)
    by_name = {s.name: s for s in game.generator.sources}
    seen: list[tuple[int, int]] = []
    for source, scramble in STAGES:
        u = by_name[source].unknotting_number
        assert not seen or u >= seen[-1][0], (source, scramble, u, seen[-1])
        if seen and u == seen[-1][0] and source == STAGES[len(seen) - 1][0]:
            assert scramble > seen[-1][1], (source, scramble)
        seen.append((u, scramble))


def test_stage_grading_is_fine_in_scramble_where_the_difficulty_is() -> None:
    """Every `+0` stage in the first ladder promoted in 2 iterations at exactly the
    proved unknotting number; every `+4` overshot. The knot is nearly free and the
    scramble is the problem, so most knots carry more than two rungs."""
    depths: dict[str, list[int]] = {}
    for source, scramble in STAGES:
        depths.setdefault(source, []).append(scramble)
    graded = [d for d in depths.values() if len(d) >= 3]
    assert len(graded) >= 3, depths
    # and no knot appears at only one depth without reason -- +0 alone is a rung
    # that measures the source knot, not the agent
    assert max(len(d) for d in depths.values()) >= 4, depths


def test_stage_mixture_concentrates_on_the_frontier_and_keeps_every_rung() -> None:
    frontier = 6
    mix = stage_mixture(frontier, 0.5)
    assert len(mix) == frontier + 1, "a cleared stage must not drop out entirely"
    assert abs(sum(w for _, _, w in mix) - 1.0) < 1e-9
    weights = [w for _, _, w in mix]
    assert weights == sorted(weights), "mass must increase toward the frontier"
    assert weights[-1] > 0.5, weights[-1]
    assert min(weights) > 0.0, "zero weight is dropping the rung by another name"
    assert [(n, s) for n, s, _ in mix] == STAGES[: frontier + 1]


def test_zero_decay_reproduces_frontier_only_training() -> None:
    """The change has to be ablatable, not assumed: decay 0 is the original rule."""
    assert stage_mixture(6, 0.0) == ()


def test_pick_stage_samples_the_mixture_and_pins_without_one() -> None:
    config = _config(parallel_arms()[1], STAGES[4], 0, "cpu", frontier=4, mix_decay=0.5)
    game = make_game(config.game)
    rng = np.random.default_rng(0)
    counts: dict[tuple[str, int], int] = {}
    for _ in range(600):
        source, scramble = pick_stage(config.game, game.generator, rng)
        counts[(source.name, scramble)] = counts.get((source.name, scramble), 0) + 1
    assert len(counts) == 5, counts
    frontier = counts[STAGES[4]] / 600
    assert 0.40 < frontier < 0.62, frontier

    # Evaluation must stay pinned even while training mixes, or the promotion
    # signal stops measuring the stage it claims to.
    assert config.game.stage_source == STAGES[4][0]
    assert config.game.stage_scramble == STAGES[4][1]

    pinned = _config(parallel_arms()[1], STAGES[4], 0, "cpu")
    assert pinned.game.stage_mix == ()
    for _ in range(20):
        source, scramble = pick_stage(pinned.game, game.generator, rng)
        assert (source.name, scramble) == STAGES[4]


def test_resume_matches_stage_identity_not_index() -> None:
    """The regression this guards: rungs get inserted between runs, so a
    checkpoint's stage *indices* refer to a ladder that no longer exists."""
    old_ladder = [
        {"source": "unknot", "scramble": 2, "promoted": True},
        {"source": "unknot", "scramble": 6, "promoted": True},
        {"source": "T(2,3)", "scramble": 0, "promoted": True},
        {"source": "T(2,3)", "scramble": 4, "promoted": True},
        {"source": "T(2,5)", "scramble": 0, "promoted": True},
        {"source": "T(2,5)", "scramble": 4, "promoted": True},
        {"source": "T(2,7)", "scramble": 0, "promoted": True},  # knot since dropped
        {"source": "T(2,7)", "scramble": 4, "promoted": True},
        {"source": "T(3,4)", "scramble": 0, "promoted": True},
        {"source": "T(3,4)", "scramble": 4, "promoted": False},  # capped, not cleared
    ]
    start, cleared, gaps = resume_point(old_ladder)

    # Index 8 in the old list was T(3,4)+0; naive index arithmetic would restart at
    # 9, which is now T(3,4)+0 itself -- a rung it has already cleared.
    assert STAGES[start] == ("T(2,3)", 2), STAGES[start]
    assert "T(2,3)+2" in gaps
    assert ("T(2,7)", 4) in cleared, "dropped rungs stay recorded as cleared"
    assert ("T(3,4)", 4) not in cleared, "a capped stage is not cleared"


def test_resume_from_nothing_and_from_a_complete_ladder() -> None:
    assert resume_point([])[0] == 0
    done = [{"source": s, "scramble": k, "promoted": True} for s, k in STAGES]
    assert resume_point(done)[0] == len(STAGES)


def test_promotion_needs_feasibility_first() -> None:
    assert promotion_reason(
        0.5, 1.0, [1.0], 1, promote_at=0.8, tolerance=0.25, window=3
    ) is None


def test_promotion_on_reaching_the_objective() -> None:
    assert promotion_reason(
        1.0, 1.2, [1.2], 1, promote_at=0.8, tolerance=0.25, window=3
    ) == "objective"
    # 4.18 against an optimum of 1 is what the old rule promoted on
    assert promotion_reason(
        1.0, 4.18, [4.18], 1, promote_at=0.8, tolerance=0.25, window=3
    ) is None


def test_plateau_needs_two_windows_before_it_can_fire() -> None:
    """One flat window is not a plateau -- it is a stage that has just started."""
    flat = [5.0, 5.0, 5.0]
    assert promotion_reason(
        1.0, 5.0, flat, 1, promote_at=0.8, tolerance=0.25, window=3
    ) is None
    assert promotion_reason(
        1.0, 5.0, flat * 2, 1, promote_at=0.8, tolerance=0.25, window=3
    ) == "plateau"


def test_still_improving_is_not_a_plateau() -> None:
    improving = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
    assert promotion_reason(
        1.0, 4.0, improving, 1, promote_at=0.8, tolerance=0.25, window=3
    ) is None


def test_unsolved_evaluations_count_as_worst_not_as_missing() -> None:
    """A stage that stops solving must read as "not improving" rather than falling
    out of the plateau test, which `nan` would do silently."""
    history = [3.0, 3.0, 3.0, float("inf"), float("inf"), float("inf")]
    assert promotion_reason(
        1.0, float("nan"), history, 1, promote_at=0.8, tolerance=0.25, window=3
    ) == "plateau"


# -- head registers ------------------------------------------------------------


def test_written_registers_are_a_finite_control_state() -> None:
    """The half of a Turing machine a memoryless scanning head is missing.

    Registers ride in the state and are broadcast into the observation, which is
    what lets them exist without touching search, the replay buffer or the training
    step: the register value is part of the observation that gets stored. There is
    no gradient through the memory -- a TOGGLE is an action and takes its credit
    from MCTS like any other.
    """
    from dataclasses import replace as _replace

    from pgx_mcts_bench.ladder import memory_arms

    for candidate in memory_arms():
        config = _config(candidate, STAGES[5], 0, "cpu")
        game = make_game(config.game)
        k = candidate.serial_registers

        # One action and one observation channel per register, both O(1) in L.
        plain = _replace(config.game, serial_registers=0)
        assert config.game.action_size == plain.action_size + k
        assert config.game.observation_channels == plain.observation_channels + k
        assert game.num_actions == config.game.action_size

        transition = game.reset(5)
        assert transition.observation.shape[-1] == config.game.observation_channels
        assert np.all(transition.state[2] == 0.0), "registers start clear"

        toggle = game.num_actions - 1
        assert game.describe(toggle) == f"TOGGLE(r{k - 1})"
        assert transition.legal_actions[toggle], "the control state is always writable"

        budget_before = int(np.asarray(game.unwrap(transition.state)._budget))
        after = game.step(transition.state, toggle)
        assert after.state[2][k - 1] == 1.0
        # It must reach the network, not just the state.
        assert np.all(after.observation[0, :, -1] == 1.0)
        # And it must cost a ply: free writes would let the agent set up an
        # arbitrary control state between two edits, which is an oracle, not a
        # machine.
        assert int(np.asarray(game.unwrap(after.state)._budget)) == budget_before - 1
        assert np.array_equal(
            np.asarray(game.unwrap(after.state)._word),
            np.asarray(game.unwrap(transition.state)._word),
        ), "a write must not touch the tape"

        assert game.step(after.state, toggle).state[2][k - 1] == 0.0, "involution"

        # Edits and shifts must carry the control state through unchanged.
        shift = next(
            a for a in np.flatnonzero(after.legal_actions) if game.shift_of(int(a)) is not None
        )
        assert np.array_equal(game.step(after.state, int(shift)).state[2], after.state[2])


def test_registers_do_not_disturb_the_default_serial_arms() -> None:
    """The register work must be inert when unused, or every earlier serial number
    silently stops being comparable."""
    from pgx_mcts_bench.ladder import serial_arms

    for candidate in serial_arms():
        config = _config(candidate, STAGES[5], 0, "cpu")
        assert config.game.serial_registers == 0
        game = make_game(config.game)
        assert game.registers == 0
        transition = game.reset(1)
        assert transition.observation.shape[-1] == config.game.observation_channels
        assert len(transition.state) == 3 and transition.state[2].size == 0


def test_central_benchmark_contains_comparable_memory_arms() -> None:
    import torch

    from pgx_mcts_bench.ladder import central_benchmark_arms
    from pgx_mcts_bench.networks import SequenceBraidNet, make_braid_network

    arms = central_benchmark_arms()
    assert [arm.name for arm in arms] == [
        "s-head-128",
        "s-reg4",
        "s-reg8",
        "s-gru128",
        "s-fsa32",
        "s-ff4-p5",
        "s-burau-oracle",
    ]
    for arm in arms[3:]:
        config = _config(arm, STAGES[0], 0, "cpu")
        game = make_game(config.game)
        transition = game.from_word([1, 2, -1], strands=3)
        assert transition.observation.shape == (
            1,
            config.game.max_len,
            config.game.observation_channels,
        )
        network = make_braid_network(config.game, config.model)
        assert isinstance(network, SequenceBraidNet)
        observation = torch.from_numpy(transition.observation).permute(2, 0, 1)[None]
        policy, value = network(observation)
        assert policy.shape == (1, config.game.action_size)
        assert value.shape == (1,)
        assert torch.isfinite(policy).all() and torch.isfinite(value).all()


def test_learned_algebra_arms_have_finite_relation_objective() -> None:
    import torch

    from pgx_mcts_bench.ladder import invariant_learning_arms
    from pgx_mcts_bench.networks import SequenceBraidNet, make_braid_network

    for arm in invariant_learning_arms():
        if arm.serial_encoder not in {"fsa", "finite-field"}:
            continue
        config = _config(arm, STAGES[0], 0, "cpu")
        network = make_braid_network(config.game, config.model)
        assert isinstance(network, SequenceBraidNet)
        relation = network.regularization_loss()
        assert relation.ndim == 0 and torch.isfinite(relation)
        relation.backward()
        parameters = (
            network.transitions
            if arm.serial_encoder == "fsa"
            else network.field_matrices
        )
        assert parameters.grad is not None and torch.isfinite(parameters.grad).all()
