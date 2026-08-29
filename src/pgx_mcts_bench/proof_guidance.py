"""Conservative policy supervision from replayed proof/search outcomes.

Proof witnesses are paths, not unique action labels.  This module deliberately
does not implement ordinary behavioural cloning.  It groups all actions whose
replayed continuations reach the same observed objective frontier and optimizes
probability mass of the group as a whole.  Actions which search did not resolve
remain unknown and are absent from both sides of the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ReplayedActionFrontier:
    """Three-way action judgement produced by bounded exact replay/search.

    ``accepted`` actions reach the best observed frontier. ``compared`` also
    includes replayed, completed actions which reached a worse observed result.
    Legal actions outside ``compared`` are unknown, not negative examples.
    """

    accepted: Tensor
    compared: Tensor

    @property
    def unknown(self) -> Tensor:
        return ~self.compared


def replayed_action_frontier(
    crossing_changes: Tensor,
    semantic_moves: Tensor,
    completed_and_replayed: Tensor,
    *,
    objective_ratio: float | None = None,
) -> ReplayedActionFrontier:
    """Classify replayed action outcomes without inventing failed negatives.

    Inputs have shape ``[batch, actions]``. Costs are meaningful only where
    ``completed_and_replayed`` is true. With ``objective_ratio=None``, every
    action reaching the minimum observed crossing-change count is accepted,
    irrespective of semantic move count. This is the safest supervision for an
    unknotting-number policy: a different sequence or target node with the same
    CC bound is not treated as an error.

    Passing a ratio selects the minimum observed ``ratio * CC + moves`` instead
    and is suitable only for an explicitly ratio-conditioned L10/L1000 head.
    A completed higher-cost route is an observed comparison, not a proof that
    the action cannot have a better continuation under a larger search budget.
    """

    if crossing_changes.shape != semantic_moves.shape:
        raise ValueError("crossing-change and semantic-move costs must have equal shape")
    if crossing_changes.shape != completed_and_replayed.shape:
        raise ValueError("costs and completion mask must have equal shape")
    if crossing_changes.ndim != 2:
        raise ValueError("action outcomes must have shape [batch, actions]")
    completed = completed_and_replayed.to(dtype=torch.bool)
    if bool((completed.sum(dim=1) == 0).any()):
        raise ValueError("every row needs at least one completed replayed action")
    if not bool(torch.isfinite(crossing_changes[completed]).all()):
        raise ValueError("completed crossing-change costs must be finite")
    if not bool(torch.isfinite(semantic_moves[completed]).all()):
        raise ValueError("completed semantic-move costs must be finite")

    if objective_ratio is None:
        score = crossing_changes
    else:
        if objective_ratio <= 0.0:
            raise ValueError("objective ratio must be positive")
        score = objective_ratio * crossing_changes + semantic_moves
    best = score.masked_fill(~completed, torch.inf).min(dim=1, keepdim=True).values
    accepted = completed & torch.isclose(score, best, rtol=0.0, atol=1e-6)
    return ReplayedActionFrontier(accepted=accepted, compared=completed)


def conservative_set_policy_loss(
    logits: Tensor,
    accepted: Tensor,
    compared: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Increase accepted-set mass only relative to adjudicated comparisons.

    For each supervised row this is

    ``logsumexp(logits[compared]) - logsumexp(logits[accepted])``.

    Consequently the loss is invariant to the identity/order of acceptable
    actions, does not choose one canonical witness, and has exactly zero
    derivative for unknown actions outside ``compared``. Rows with no observed
    non-frontier comparison abstain and contribute zero.
    """

    if logits.shape != accepted.shape or logits.shape != compared.shape:
        raise ValueError("logits and action masks must have identical shapes")
    if logits.ndim != 2:
        raise ValueError("policy tensors must have shape [batch, actions]")
    accepted = accepted.to(device=logits.device, dtype=torch.bool)
    compared = compared.to(device=logits.device, dtype=torch.bool)
    if bool((accepted & ~compared).any()):
        raise ValueError("accepted actions must be a subset of compared actions")

    eligible = accepted.any(dim=1) & (compared & ~accepted).any(dim=1)
    floor = torch.finfo(logits.dtype).min
    accepted_mass = torch.logsumexp(logits.masked_fill(~accepted, floor), dim=1)
    compared_mass = torch.logsumexp(logits.masked_fill(~compared, floor), dim=1)
    per_row = torch.where(eligible, compared_mass - accepted_mass, 0.0)
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return per_row.sum()
    if reduction == "mean":
        if not bool(eligible.any()):
            return logits.sum() * 0.0
        return per_row[eligible].mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def adapter_only_set_objective(
    network,
    observations: Tensor,
    accepted: Tensor,
    compared: Tensor,
) -> Tensor:
    """Apply set supervision through the option adapter, never the base trunk.

    The base logits are evaluated with the adapter bypassed and detached.  The
    differentiable path is only ``option_policy_components``.  This remains true
    even if callers forgot to set ``requires_grad=False`` on base parameters.
    Retention KL belongs on a separate native-MCTS batch; it is intentionally
    absent here so unknown proof actions receive no indirect graph-batch signal.
    """

    if getattr(network, "option_policy_adapter", None) is None:
        raise ValueError("proof set supervision requires an attached option adapter")
    previous = bool(getattr(network, "option_adapter_enabled", True))
    network.option_adapter_enabled = False
    try:
        with torch.no_grad():
            base_logits, _ = network(observations)
    finally:
        network.option_adapter_enabled = previous
    residual, _ = network.option_policy_components(observations)
    if residual.shape != base_logits.shape:
        raise ValueError("option adapter residual and base policy logits differ in shape")
    return conservative_set_policy_loss(
        base_logits.detach() + residual,
        accepted,
        compared,
    )
