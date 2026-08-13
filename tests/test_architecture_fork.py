from __future__ import annotations

import torch
from torch import nn

from pgx_mcts_bench.architecture_fork import migrate_optimizer_state_by_name


class Parent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.shared(inputs)


class Child(Parent):
    def __init__(self) -> None:
        super().__init__()
        self.new = nn.Linear(4, 4)


def test_optimizer_migration_preserves_old_moments_and_leaves_new_state_fresh() -> None:
    parent, child = Parent(), Child()
    child.shared.load_state_dict(parent.shared.state_dict())
    parent_optimizer = torch.optim.AdamW(parent.parameters(), lr=3e-4, weight_decay=2e-5)
    child_optimizer = torch.optim.AdamW(child.parameters(), lr=1e-3)
    parent(torch.ones(2, 3)).sum().backward()
    parent_optimizer.step()

    report = migrate_optimizer_state_by_name(parent, parent_optimizer, child, child_optimizer)

    assert set(report["copied"]) == {"shared.weight", "shared.bias"}
    assert set(report["fresh"]) == {"new.weight", "new.bias"}
    assert child_optimizer.param_groups[0]["lr"] == 3e-4
    assert child_optimizer.param_groups[0]["weight_decay"] == 2e-5
    assert child_optimizer.state[child.shared.weight]["step"].item() == 1
    assert child.new.weight not in child_optimizer.state
