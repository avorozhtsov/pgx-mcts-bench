"""Function-preserving scientist architecture forks."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


def migrate_optimizer_state_by_name(
    source_network: nn.Module,
    source_optimizer: torch.optim.Optimizer,
    target_network: nn.Module,
    target_optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Copy optimizer state for unchanged named parameters into a child model.

    New or shape-changed parameters intentionally retain fresh optimizer state.
    Parameter-group hyperparameters are copied, but the target group's own
    parameter membership is preserved.
    """
    source_named = dict(source_network.named_parameters())
    target_named = dict(target_network.named_parameters())
    copied: list[str] = []
    fresh: list[str] = []
    for name, target in target_named.items():
        source = source_named.get(name)
        if source is None or source.shape != target.shape or source not in source_optimizer.state:
            fresh.append(name)
            continue
        target_optimizer.state[target] = copy.deepcopy(source_optimizer.state[source])
        copied.append(name)

    if len(source_optimizer.param_groups) != len(target_optimizer.param_groups):
        raise ValueError("source and target optimizers have different parameter-group counts")
    for source_group, target_group in zip(
        source_optimizer.param_groups, target_optimizer.param_groups, strict=True
    ):
        for key, value in source_group.items():
            if key != "params":
                target_group[key] = copy.deepcopy(value)
    return {"copied": copied, "fresh": fresh}
