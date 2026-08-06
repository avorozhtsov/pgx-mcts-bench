import torch
from torch import nn

from pgx_mcts_bench.training import attach_policy_value_preservation_teacher


def test_preservation_teacher_is_frozen_and_not_serialized() -> None:
    network = nn.Linear(2, 2)

    teacher = attach_policy_value_preservation_teacher(network)

    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert not any("preservation_teacher" in key for key in network.state_dict())
    with torch.no_grad():
        network.weight.add_(1.0)
    assert not torch.equal(network.weight, teacher.weight)
