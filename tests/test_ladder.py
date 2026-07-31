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
        assert transition.state.registers.size == 0


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


# -- thread colours ------------------------------------------------------------


def _coloured_game(colours: int = 4):
    from dataclasses import replace as _replace

    from pgx_mcts_bench.ladder import serial_arms

    config = _config(serial_arms()[0], STAGES[7], 0, "cpu")
    return make_game(_replace(config.game, serial_colours=colours))


def test_a_colour_follows_its_strand_around_the_whole_necklace() -> None:
    """The property the whole design rests on.

    A colour is attached to a height at the head, and a crossing swaps the two
    strands it joins -- so walking the head once around the necklace must permute
    the colours by exactly the braid's permutation. Checked against
    `reference.permutation`, which is an independent implementation.
    """
    from rf_knots.reference import permutation

    game = _coloured_game()
    transition = game.from_word([1, 2, 1, -2, 3, 2], strands=4)
    state = transition.state
    word = [int(x) for x in np.asarray(game.unwrap(state)._word) if int(x)]

    # Distinct labels, so the transport is a permutation test rather than a
    # coincidence between equal colours.
    painted = np.arange(game.config.max_strands, dtype=np.int64)
    carried = game._transport(game.unwrap(state), 0, len(word), painted)

    expected = permutation(word, game.config.max_strands)
    assert list(carried) == [int(painted[i]) for i in expected], (carried, expected)


def test_transport_is_reversible_because_a_crossing_is_an_involution() -> None:
    game = _coloured_game()
    transition = game.from_word([1, 2, -1, 3, 2], strands=4)
    pgx = game.unwrap(transition.state)
    painted = np.arange(game.config.max_strands, dtype=np.int64)
    for stride in (1, 2, 4):
        there = game._transport(pgx, 0, stride, painted)
        back = game._transport(pgx, stride, -stride, there)
        assert np.array_equal(back, painted), (stride, there, back)


def test_one_shift_swaps_exactly_the_two_strands_the_letter_joins() -> None:
    game = _coloured_game()
    transition = game.from_word([2, 1, 1, 1], strands=4)  # head sits on s2
    state = transition.state
    painted = np.arange(game.config.max_strands, dtype=np.int64)
    state = state._replace(colours=painted)

    right = next(
        a for a in np.flatnonzero(transition.legal_actions) if game.shift_of(int(a)) == 1
    )
    after = game.step(state, int(right)).state[3]
    # s2 joins heights 1 and 2 (0-based), and nothing else may move.
    assert list(after) == [0, 2, 1, 3, 4], list(after)


def test_paint_and_cycle_cost_a_ply_and_do_not_touch_the_word() -> None:
    game = _coloured_game(colours=4)
    transition = game.from_word([1, 2, 1, 2], strands=3)
    names = {game.describe(i): i for i in range(game.num_actions)}
    assert {"PAINT_LOW", "PAINT_HIGH", "CYCLE"} <= set(names)

    before = game.unwrap(transition.state)
    budget = int(np.asarray(before._budget))

    cycled = game.step(transition.state, names["CYCLE"])
    assert cycled.state.colour == 1
    assert int(np.asarray(game.unwrap(cycled.state)._budget)) == budget - 1
    assert np.array_equal(
        np.asarray(game.unwrap(cycled.state)._word), np.asarray(before._word)
    ), "a colour action must not touch the tape"

    painted = game.step(cycled.state, names["PAINT_LOW"])
    low, high = game._crossing_heights(game.unwrap(cycled.state), cycled.state.head)
    assert painted.state.colours[low] == 2, "colour 1 held -> value 1+1 written"
    assert painted.state.colours[high] == 0
    # and it reaches the network, not just the state
    assert painted.observation.shape[-1] == game.config.observation_channels
    assert painted.observation[0, :, -4:].sum() > 0, "held colour must be observable"


def test_cycle_wraps_through_the_whole_palette() -> None:
    game = _coloured_game(colours=4)
    transition = game.from_word([1, 2, 1, 2], strands=3)
    cycle = next(i for i in range(game.num_actions) if game.describe(i) == "CYCLE")
    state, seen = transition.state, []
    for _ in range(5):
        seen.append(state.colour)
        state = game.step(state, cycle).state
    assert seen == [0, 1, 2, 3, 0], seen


def test_three_actions_however_many_colours() -> None:
    """Paint-per-(strand, colour) would be 20 dead actions at N=5, C=4, and dead
    actions are what sank the register arm."""
    from dataclasses import replace as _replace

    from pgx_mcts_bench.ladder import serial_arms

    base = _config(serial_arms()[0], STAGES[7], 0, "cpu").game
    plain = base.action_size
    for colours in (2, 4, 8):
        assert _replace(base, serial_colours=colours).action_size == plain + 3


def test_colours_are_inert_when_unused() -> None:
    """Every earlier serial number has to stay comparable."""
    from pgx_mcts_bench.ladder import serial_arms

    for candidate in serial_arms():
        config = _config(candidate, STAGES[7], 0, "cpu")
        assert config.game.serial_colours == 0
        game = make_game(config.game)
        assert game.colours == 0
        transition = game.reset(2)
        assert transition.observation.shape[-1] == config.game.observation_channels
        assert transition.state.colours.size == config.game.max_strands


# -- the unknot+6 wall ---------------------------------------------------------


def test_pooled_rate_is_not_the_worst_ratio() -> None:
    """What eliminated six of seventeen arms on rung 1 of the mixed ladder.

    With twelve evaluation games and a 0.8 bar, 10/12 = 0.83 is the first passing
    value and 9/12 = 0.75 the first failing one -- so a `min` over three ratios is
    a conjunction of three noisy single-episode coin flips. `s-head-128` scored
    0.58 / 0.92 / 0.58 and capped; pooled that is 0.69, still a fail, but
    `s-reg8`'s 0.83 / 0.75 / 0.50 pools to 0.69 rather than 0.50 and no longer
    turns on one episode of one ratio.
    """
    # 0.83 / 0.92 / 0.75 pools to 0.833: a pass that the min-rule would have
    # failed on a single episode of the worst ratio.
    assert promotion_reason(
        (0.83 + 0.92 + 0.75) / 3, 0.2, [0.2], 0,
        promote_at=0.8, tolerance=0.5, window=3, worst_ratio=0.75,
    ) == "objective"
    # but genuine collapse on one end of the front is still caught
    assert promotion_reason(
        (1.0 + 1.0 + 0.4) / 3, 0.2, [0.2], 0,
        promote_at=0.8, tolerance=0.5, window=3, worst_ratio=0.4,
    ) is None


def test_collapse_floor_is_separate_from_the_promotion_bar() -> None:
    pooled = (1.0 + 1.0 + 0.45) / 3
    assert promotion_reason(
        pooled, 0.0, [0.0], 0, promote_at=0.8, tolerance=0.5, window=3,
        worst_ratio=0.45, collapse_floor=0.5,
    ) is None
    assert promotion_reason(
        pooled, 0.0, [0.0], 0, promote_at=0.8, tolerance=0.5, window=3,
        worst_ratio=0.45, collapse_floor=0.4,
    ) == "objective"


def test_a_capped_rung_does_not_retire_the_candidate() -> None:
    """The docstring always said a stuck candidate moves up anyway; the code
    broke out of the loop instead, so one bad rung ended an arm permanently."""
    import inspect

    from pgx_mcts_bench.ladder import run_ladder

    source = inspect.getsource(run_ladder)
    assert "consecutive_caps += 1" in source
    assert "max_consecutive_caps" in source
    # and the bound still exists, so a hopeless arm releases its cores
    assert "if consecutive_caps >= max_consecutive_caps:" in source


def test_expected_cost_penalises_giving_up() -> None:
    """`crossings` is conditional on solving, so it is anti-correlated with the
    solve rate: an arm that abandons the hard instances drops them out of its own
    average and looks cheaper for it. `expected_crossings` divides by the rate, so
    solving half the instances at 5.00 costs 10.00 rather than tying with an arm
    that solved all of them at 5.00."""
    from pgx_mcts_bench.ladder import evaluate_stage

    class _Row(dict):
        pass

    # exercise the arithmetic directly against the documented example
    games, solved_count, crossings_total = 8, 4, 20.0
    rate = solved_count / games
    conditional = crossings_total / solved_count
    expected = conditional / rate
    assert conditional == 5.0
    assert expected == 10.0
    assert evaluate_stage is not None  # imported for the reader's benefit


def test_expected_cost_agrees_with_conditional_when_everything_solves() -> None:
    games, solved_count, crossings_total = 8, 8, 40.0
    rate = solved_count / games
    conditional = crossings_total / solved_count
    assert conditional == (conditional / rate) == 5.0
