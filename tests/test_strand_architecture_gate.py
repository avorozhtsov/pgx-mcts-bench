import pytest

from pgx_mcts_bench.strand_architecture_gate import (
    EARLY_MIXED_STRAND_STAGES,
    run_strand_architecture_gate,
)


def test_early_gate_introduces_four_strands_before_second_u2_task() -> None:
    assert EARLY_MIXED_STRAND_STAGES[3] == ("P(4,5)#0", 0)
    assert EARLY_MIXED_STRAND_STAGES[-1] == ("P(4,7)#0", 0)


def test_gate_rejects_an_empty_stage_prefix(tmp_path) -> None:
    with pytest.raises(ValueError, match="stage_limit"):
        run_strand_architecture_gate(
            tmp_path,
            candidate_names=["s-head-128"],
            seeds=[1],
            stage_limit=0,
        )


def test_gate_rejects_an_invalid_evaluation_cadence(tmp_path) -> None:
    with pytest.raises(ValueError, match="eval_every"):
        run_strand_architecture_gate(
            tmp_path,
            candidate_names=["s-head-128"],
            seeds=[1],
            eval_every=0,
        )


def test_run_row_records_its_seed(monkeypatch, tmp_path) -> None:
    from pgx_mcts_bench import ladder
    from pgx_mcts_bench import strand_architecture_gate as gate

    candidate = next(
        item
        for item in ladder.candidates()
        if item.name == "conv-cylinder-recurrent-idcols-128-bstar"
    )
    monkeypatch.setattr(ladder, "candidates", lambda: [candidate])
    monkeypatch.setattr(
        ladder,
        "run_ladder",
        lambda *_args, **_kwargs: ladder.LadderResult(
            name=candidate.name,
            rationale="test",
            highest_stage=-1,
            seconds=0.0,
        ),
    )

    row = gate._run_one(
        {
            "candidate": candidate.name,
            "seed": 73,
            "output": str(tmp_path),
            "simulations": 1,
            "stage_limit": 1,
            "device": "cpu",
            "max_iterations": 1,
            "selfplay_games": 1,
            "eval_every": 1,
            "eval_games": 1,
            "promote_at": 0.8,
            "rehearsal_games_per_cleared_stage": 1,
            "adaptive_rehearsal": True,
            "rehearsal_target": 0.8,
            "max_rehearsal_games_per_stage": 8,
            "max_consecutive_caps": 1,
            "retry_capped_on_resume": False,
            "retro_games": 1,
        }
    )

    assert row["seed"] == 73


def test_gate_rejects_a_negative_rehearsal_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="rehearsal"):
        run_strand_architecture_gate(
            tmp_path,
            candidate_names=["s-head-128"],
            seeds=[1],
            rehearsal_games_per_cleared_stage=-1,
        )


def test_adaptive_gate_rejects_zero_initial_rehearsal(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive initial"):
        run_strand_architecture_gate(
            tmp_path,
            candidate_names=["s-head-128"],
            seeds=[1],
            rehearsal_games_per_cleared_stage=0,
        )


def test_gate_rejects_zero_consecutive_cap_allowance(tmp_path) -> None:
    with pytest.raises(ValueError, match="consecutive caps"):
        run_strand_architecture_gate(
            tmp_path,
            candidate_names=["window-local"],
            seeds=[1],
            max_consecutive_caps=0,
        )


def test_gate_records_explicit_capped_stage_retry_policy(monkeypatch, tmp_path) -> None:
    from pgx_mcts_bench import strand_architecture_gate as gate

    monkeypatch.setattr(gate, "_run_one", lambda payload: payload)
    report = gate.run_strand_architecture_gate(
        tmp_path,
        candidate_names=["window-local"],
        seeds=[71],
        retry_capped_on_resume=True,
    )
    assert report["settings"]["retry_capped_on_resume"] is True
    assert report["runs"][0]["retry_capped_on_resume"] is True
