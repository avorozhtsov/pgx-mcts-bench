from pathlib import Path

import pytest
import torch

from pgx_mcts_bench.rapid_adaptation import (
    adaptation_blocks,
    next_f,
    promoted_checkpoint_metadata,
)


def test_adaptation_blocks_follow_20_then_40_schedule() -> None:
    assert adaptation_blocks(200) == [
        (0, 20),
        (20, 60),
        (60, 100),
        (100, 140),
        (140, 180),
        (180, 200),
    ]


def test_f_advances_only_after_subthreshold_block_and_caps_at_16() -> None:
    assert next_f(5, 0.79) == 8
    assert next_f(8, 0.8) == 8
    assert next_f(14, 0.2) == 16
    assert next_f(16, 0.0) == 16


def _checkpoint(path: Path, *, promoted: bool, candidate: str = "s-window-128") -> None:
    torch.save(
        {
            "candidate": candidate,
            "stage_result": {
                "stage": 21,
                "source": "R(3,12)#0",
                "scramble": 4,
                "solve_rate": 1.0 if promoted else 0.0,
                "reason": "plateau" if promoted else "capped",
                "promoted": promoted,
            },
        },
        path,
    )


def test_promoted_checkpoint_metadata_accepts_last_cleared_rung(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage21-after.pt"
    _checkpoint(checkpoint, promoted=True)

    metadata = promoted_checkpoint_metadata(checkpoint, "s-window-128")

    assert metadata["stage"] == 21
    assert metadata["embedded_solve_rate"] == 1.0


def test_promoted_checkpoint_metadata_rejects_capped_snapshot(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage22-after.pt"
    _checkpoint(checkpoint, promoted=False)

    with pytest.raises(ValueError, match="requires a promoted checkpoint"):
        promoted_checkpoint_metadata(checkpoint, "s-window-128")
