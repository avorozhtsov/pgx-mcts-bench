from __future__ import annotations

import pytest
import torch
from torch import nn

from pgx_mcts_bench.proof_guidance import (
    adapter_only_set_objective,
    conservative_set_policy_loss,
    replayed_action_frontier,
)


def test_min_cc_frontier_accepts_different_sequences_and_targets() -> None:
    frontier = replayed_action_frontier(
        torch.tensor([[2.0, 2.0, 3.0, 0.0]]),
        torch.tensor([[40.0, 9.0, 1.0, 0.0]]),
        torch.tensor([[True, True, True, False]]),
    )
    assert frontier.accepted.tolist() == [[True, True, False, False]]
    assert frontier.compared.tolist() == [[True, True, True, False]]


def test_ratio_conditioned_frontier_can_break_same_cc_move_tie() -> None:
    frontier = replayed_action_frontier(
        torch.tensor([[2.0, 2.0, 3.0]]),
        torch.tensor([[40.0, 9.0, 1.0]]),
        torch.tensor([[True, True, True]]),
        objective_ratio=10.0,
    )
    assert frontier.accepted.tolist() == [[False, True, False]]


def test_unknown_actions_receive_exactly_zero_gradient() -> None:
    logits = torch.tensor([[0.2, 1.1, 7.0, -0.4]], requires_grad=True)
    loss = conservative_set_policy_loss(
        logits,
        accepted=torch.tensor([[True, True, False, False]]),
        compared=torch.tensor([[True, True, False, True]]),
    )
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 2].item() == 0.0
    assert logits.grad[0, 3].item() > 0.0
    assert logits.grad[0, :2].sum().item() < 0.0


def test_acceptable_action_order_does_not_change_loss() -> None:
    first = conservative_set_policy_loss(
        torch.tensor([[0.1, 1.3, -0.2]]),
        accepted=torch.tensor([[True, True, False]]),
        compared=torch.tensor([[True, True, True]]),
    )
    swapped = conservative_set_policy_loss(
        torch.tensor([[1.3, 0.1, -0.2]]),
        accepted=torch.tensor([[True, True, False]]),
        compared=torch.tensor([[True, True, True]]),
    )
    assert torch.allclose(first, swapped)


def test_loss_abstains_without_a_replayed_comparison() -> None:
    logits = torch.tensor([[0.2, 1.1, 7.0]], requires_grad=True)
    loss = conservative_set_policy_loss(
        logits,
        accepted=torch.tensor([[True, True, False]]),
        compared=torch.tensor([[True, True, False]]),
    )
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_masks_must_be_consistent() -> None:
    with pytest.raises(ValueError, match="subset"):
        conservative_set_policy_loss(
            torch.zeros((1, 2)),
            accepted=torch.tensor([[True, False]]),
            compared=torch.tensor([[False, True]]),
        )

    with pytest.raises(ValueError, match="at least one"):
        replayed_action_frontier(
            torch.zeros((1, 2)),
            torch.zeros((1, 2)),
            torch.tensor([[False, False]]),
        )


class _TinyAdaptedPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 4)
        self.option_policy_adapter = nn.Linear(3, 4, bias=False)
        nn.init.zeros_(self.option_policy_adapter.weight)
        self.option_policy_gate = None
        self.option_adapter_enabled = True

    def option_policy_components(self, observation: torch.Tensor):
        residual = self.option_policy_adapter(observation)
        return residual, observation.new_ones((observation.shape[0], 1))

    def forward(self, observation: torch.Tensor):
        logits = self.base(observation)
        if self.option_adapter_enabled:
            logits = logits + self.option_policy_adapter(observation)
        return logits, logits[:, 0]


def test_adapter_objective_has_no_gradient_path_to_base_policy() -> None:
    network = _TinyAdaptedPolicy()
    loss = adapter_only_set_objective(
        network,
        torch.tensor([[1.0, 2.0, 3.0]]),
        accepted=torch.tensor([[True, True, False, False]]),
        compared=torch.tensor([[True, True, True, False]]),
    )
    loss.backward()
    assert network.base.weight.grad is None
    assert network.base.bias.grad is None
    assert network.option_policy_adapter.weight.grad is not None
    assert network.option_policy_adapter.weight.grad.abs().sum().item() > 0.0
