"""Proof-aware L10/L1000 distillation for mastery-v3.

This stage deliberately keeps mathematical feasibility separate from the
operational ``p_solve`` head.  Certified lower bounds provide only negative
budget labels; positive budget and policy labels require a replay-verified
witness from the exact registered starting braid.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rf_knots.evidence import UnknotWitness
from torch import Tensor
from torch.nn import functional as F

from pgx_mcts_bench.adaptive_scientists import FixedWordGame, KnotItem, load_scientist
from pgx_mcts_bench.collaborative_scientists import translate_semantic_record
from pgx_mcts_bench.mastery_v3 import (
    V3_ORDINAL_MAX_U,
    CyclicMemoryDeepV3,
)
from pgx_mcts_bench.mastery_v3_curriculum import file_sha256

PROOF_SCHEMA = "mastery-v3-proof-distillation-v1"
RATIOS = (10, 1000)
MINIMUM_TRAINING_STEPS = 500
MINIMUM_SAMPLES_PER_SIDE = 2


@dataclass(frozen=True)
class CertifiedBudgetLabel:
    identity: str
    representation_id: str
    ratio: int
    budget: int
    feasible: int
    certified_lower_bound: int
    certified_upper_bound: int
    evidence_id: str | None


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _identity(row: dict[str, Any]) -> str:
    return str(row.get("name") or row.get("id"))


def _bounds(row: dict[str, Any]) -> tuple[int, int]:
    lower = row.get("certified_unknotting_lower_bound")
    upper = row.get(
        "certified_unknotting_upper_bound", row.get("known_unknotting_number")
    )
    if lower is None or upper is None:
        raise ValueError(f"{_identity(row)} lacks a certified two-sided interval")
    lower, upper = int(lower), int(upper)
    if lower < 0 or upper < lower:
        raise ValueError(f"invalid certified interval [{lower}, {upper}]")
    return lower, upper


def ordinal_targets(lower: int, upper: int) -> tuple[list[float], list[float]]:
    """Targets for q_k=P(u<=k), masking the uncertified open interval."""

    targets, mask = [], []
    for threshold in range(V3_ORDINAL_MAX_U + 1):
        if threshold < lower:
            targets.append(0.0)
            mask.append(1.0)
        elif threshold >= upper:
            targets.append(1.0)
            mask.append(1.0)
        else:
            targets.append(0.0)
            mask.append(0.0)
    return targets, mask


def certified_budget_labels(
    row: dict[str, Any],
    ratio: int,
    rng: np.random.Generator,
    *,
    samples_per_side: int,
    witness: dict[str, Any] | None,
) -> list[CertifiedBudgetLabel]:
    """Generate only logically certified feasibility labels.

    ``B < ratio * lower`` is impossible.  A positive is admitted only at or
    above the exact replayed witness cost ``ratio * crossings + moves``.
    No label is emitted in between these thresholds.
    """

    lower, upper = _bounds(row)
    identity = _identity(row)
    representation_id = str(row["representation_id"])
    labels: list[CertifiedBudgetLabel] = []
    impossible_stop = ratio * lower
    if impossible_stop > 0:
        for budget in rng.integers(0, impossible_stop, size=samples_per_side):
            labels.append(
                CertifiedBudgetLabel(
                    identity,
                    representation_id,
                    ratio,
                    int(budget),
                    0,
                    lower,
                    upper,
                    None,
                )
            )
    if witness is not None:
        crossing_changes = int(witness["crossing_changes"])
        moves = int(witness["moves"])
        witness_cost = ratio * crossing_changes + moves
        for slack in rng.integers(100, 601, size=samples_per_side):
            labels.append(
                CertifiedBudgetLabel(
                    identity,
                    representation_id,
                    ratio,
                    witness_cost + int(slack),
                    1,
                    lower,
                    upper,
                    str(witness.get("evidence_id")),
                )
            )
    return labels


def _exact_witnesses(evidence: dict[str, Any]) -> dict[tuple[tuple[int, ...], int], dict[str, Any]]:
    verified = evidence.get("verified", {})
    candidates: list[dict[str, Any]] = []
    candidates.extend((verified.get("best_by_representation") or {}).values())
    candidates.extend(verified.get("best_by_representation_and_solver_version") or [])
    exact: dict[tuple[tuple[int, ...], int], dict[str, Any]] = {}
    for item in candidates:
        witness_row = item.get("witness")
        if not isinstance(witness_row, dict):
            continue
        witness = UnknotWitness.from_dict(witness_row)
        witness.verify()
        key = (tuple(int(value) for value in witness.start.word), int(witness.start.strands))
        normalized = {
            "evidence_id": item.get("evidence_id"),
            "crossing_changes": int(witness.crossing_changes),
            "moves": int(witness.moves),
            "witness": witness,
        }
        previous = exact.get(key)
        if previous is None or (
            1000 * normalized["crossing_changes"] + normalized["moves"]
            < 1000 * previous["crossing_changes"] + previous["moves"]
        ):
            exact[key] = normalized
    return exact


def _observation(game: FixedWordGame, seed: int) -> np.ndarray:
    return np.asarray(game.reset(seed).observation)


def _tensor(observations: list[np.ndarray], device: str) -> Tensor:
    return (
        torch.from_numpy(np.stack(observations))
        .permute(0, 3, 1, 2)
        .float()
        .to(device)
    )


def _evaluate(
    network: CyclicMemoryDeepV3,
    observations: Tensor,
    feasible: Tensor,
    lower: Tensor,
    upper: Tensor,
    ordinal: Tensor,
    ordinal_mask: Tensor,
    monotone_pairs: Tensor,
) -> dict[str, float]:
    network.eval()
    with torch.inference_mode():
        diagnostics = network.proof_diagnostics(observations)
        feasibility_accuracy = float(
            ((diagnostics.feasibility_logit >= 0) == (feasible >= 0.5))
            .float()
            .mean()
            .item()
        )
        ordinal_correct = (
            ((diagnostics.ordinal_logits >= 0) == (ordinal >= 0.5)).float()
            * ordinal_mask
        ).sum()
        ordinal_accuracy = float(
            (ordinal_correct / ordinal_mask.sum().clamp(min=1.0)).item()
        )
        bound_mae = float(
            (
                (diagnostics.lower_bound - lower).abs()
                + (diagnostics.upper_bound - upper).abs()
            )
            .mul(0.5)
            .mean()
            .item()
        )
        if monotone_pairs.numel():
            low = diagnostics.feasibility_logit[monotone_pairs[:, 0]]
            high = diagnostics.feasibility_logit[monotone_pairs[:, 1]]
            monotone_violation_rate = float((low > high + 1e-6).float().mean().item())
        else:
            monotone_violation_rate = 0.0
    return {
        "feasibility_accuracy": feasibility_accuracy,
        "ordinal_accuracy": ordinal_accuracy,
        "bound_mae": bound_mae,
        "budget_monotonic_violation_rate": monotone_violation_rate,
    }


def distill_mastery_v3(
    checkpoint: Path,
    curriculum_path: Path,
    evidence_path: Path,
    output: Path,
    *,
    candidate_name: str,
    steps: int = 2_000,
    samples_per_side: int = 4,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    policy_weight: float = 0.5,
    preservation_weight: float = 0.1,
    seed: int = 2026081701,
    device: str = "cuda",
) -> dict[str, Any]:
    """Distill certified bounds and exact witnesses into a v3 checkpoint."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    curriculum = json.loads(curriculum_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    declared_evidence_sha = curriculum["sources"]["evidence_snapshot"]["sha256"]
    actual_evidence_sha = file_sha256(evidence_path)
    if actual_evidence_sha != declared_evidence_sha:
        raise ValueError("evidence snapshot hash differs from the frozen curriculum")
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not source_payload.get("mastery_v3_pretraining"):
        raise ValueError("proof distillation requires a mastery-v3 pretraining checkpoint")
    scientist = load_scientist(
        candidate_name,
        checkpoint,
        seed=seed,
        device=device,
        require_factorized=True,
        objective_budget_channel=True,
    )
    network = scientist.network
    if not isinstance(network, CyclicMemoryDeepV3):
        raise TypeError(type(network).__name__)
    rows = [
        row
        for stage in ("simple_adaptation", "heavy_capacity")
        for row in curriculum["stages"][stage]["rows"]
    ]
    exact = _exact_witnesses(evidence)
    rng = np.random.default_rng(seed)
    labels: list[CertifiedBudgetLabel] = []
    observations: list[np.ndarray] = []
    ordinal_rows: list[list[float]] = []
    ordinal_masks: list[list[float]] = []
    groups: dict[tuple[str, int], list[int]] = {}
    row_by_representation = {str(row["representation_id"]): row for row in rows}
    exact_for_row: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = (tuple(int(value) for value in row["word"]), int(row["strands"]))
        witness = exact.get(key)
        if witness is not None:
            exact_for_row[str(row["representation_id"])] = witness
        for ratio in RATIOS:
            generated = certified_budget_labels(
                row,
                ratio,
                rng,
                samples_per_side=samples_per_side,
                witness=witness,
            )
            for label in generated:
                knot = KnotItem(
                    label.representation_id,
                    len(row["word"]),
                    tuple(int(value) for value in row["word"]),
                    int(row["strands"]),
                )
                game = FixedWordGame(
                    scientist.game,
                    knot,
                    float(ratio),
                    objective_cap=float(label.budget),
                )
                observations.append(_observation(game, seed + len(labels)))
                target, mask = ordinal_targets(
                    label.certified_lower_bound, label.certified_upper_bound
                )
                ordinal_rows.append(target)
                ordinal_masks.append(mask)
                index = len(labels)
                labels.append(label)
                groups.setdefault((label.representation_id, ratio), []).append(index)
    if not labels or not any(label.feasible for label in labels):
        raise ValueError("frozen evidence contains no exact positive witness for this curriculum")
    monotone_pairs = []
    for indexes in groups.values():
        ordered = sorted(indexes, key=lambda index: labels[index].budget)
        monotone_pairs.extend(zip(ordered, ordered[1:], strict=False))

    proof_observations = _tensor(observations, device)
    feasible = torch.tensor(
        [label.feasible for label in labels], dtype=torch.float32, device=device
    )
    lower = torch.tensor(
        [label.certified_lower_bound for label in labels], dtype=torch.float32, device=device
    )
    upper = torch.tensor(
        [label.certified_upper_bound for label in labels], dtype=torch.float32, device=device
    )
    ordinal = torch.tensor(ordinal_rows, dtype=torch.float32, device=device)
    ordinal_mask = torch.tensor(ordinal_masks, dtype=torch.float32, device=device)
    pair_tensor = torch.tensor(monotone_pairs, dtype=torch.long, device=device).reshape(-1, 2)

    policy_observations: list[np.ndarray] = []
    policy_actions: list[int] = []
    for representation_id, witness_item in sorted(exact_for_row.items()):
        row = row_by_representation[representation_id]
        witness: UnknotWitness = witness_item["witness"]
        for ratio in RATIOS:
            cost = ratio * witness.crossing_changes + witness.moves
            budget = cost + int(rng.integers(100, 601))
            knot = KnotItem(
                representation_id,
                len(row["word"]),
                tuple(int(value) for value in row["word"]),
                int(row["strands"]),
            )
            game = FixedWordGame(
                scientist.game, knot, float(ratio), objective_cap=float(budget)
            )
            semantic_actions = [
                step.action.to_flat(scientist.game.config._spec)
                for step in witness.steps
            ]
            translated = translate_semantic_record(
                scientist,
                knot,
                float(ratio),
                semantic_actions,
                seed=seed + len(policy_actions),
            )
            if translated is None:
                raise ValueError(
                    f"exact witness for {representation_id} cannot be routed by {candidate_name}"
                )
            transition = game.reset(seed + len(policy_actions))
            for position in translated:
                action = int(position.action)
                policy_observations.append(np.asarray(transition.observation))
                policy_actions.append(action)
                transition = game.step(transition.state, action)
    policy_tensor = _tensor(policy_observations, device)
    policy_targets = torch.tensor(policy_actions, dtype=torch.long, device=device)

    for parameter in network.parameters():
        parameter.requires_grad_(False)
    excluded = (
        "value_residual",
        "solve_residual",
        "cost_residual",
        "invalid_capacity_head",
    )
    trainable = []
    for name, parameter in network.named_parameters():
        if not name.startswith("parent.") and not any(token in name for token in excluded):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    torch_rng = torch.Generator(device="cpu").manual_seed(seed + 91)
    losses: list[float] = []
    positive_weight = ((feasible.numel() - feasible.sum()) / feasible.sum()).clamp(min=1.0)
    network.train()
    network.parent.eval()
    for _ in range(steps):
        indexes = torch.randint(
            0,
            len(labels),
            (min(batch_size, len(labels)),),
            generator=torch_rng,
        ).to(device)
        batch = proof_observations[indexes]
        diagnostics = network.proof_diagnostics(batch)
        feasibility_loss = F.binary_cross_entropy_with_logits(
            diagnostics.feasibility_logit,
            feasible[indexes],
            pos_weight=positive_weight,
        )
        bound_loss = 0.5 * (
            F.smooth_l1_loss(diagnostics.lower_bound, lower[indexes])
            + F.smooth_l1_loss(diagnostics.upper_bound, upper[indexes])
        )
        ordinal_loss_raw = F.binary_cross_entropy_with_logits(
            diagnostics.ordinal_logits, ordinal[indexes], reduction="none"
        )
        ordinal_loss = (
            ordinal_loss_raw * ordinal_mask[indexes]
        ).sum() / ordinal_mask[indexes].sum().clamp(min=1.0)
        if pair_tensor.numel():
            pair_indexes = torch.randint(
                0,
                pair_tensor.shape[0],
                (min(batch_size // 2, pair_tensor.shape[0]),),
                generator=torch_rng,
            ).to(device)
            selected_pairs = pair_tensor[pair_indexes]
            pair_logits = network.proof_diagnostics(
                proof_observations[selected_pairs.reshape(-1)]
            ).feasibility_logit.reshape(-1, 2)
            monotonic_loss = F.relu(pair_logits[:, 0] - pair_logits[:, 1] + 0.05).mean()
        else:
            monotonic_loss = feasibility_loss.new_zeros(())
        policy_indexes = torch.randint(
            0,
            len(policy_actions),
            (min(batch_size, len(policy_actions)),),
            generator=torch_rng,
        ).to(device)
        policy_batch = policy_tensor[policy_indexes]
        child_policy, _ = network(policy_batch)
        with torch.no_grad():
            parent_policy, _ = network.parent(network._parent_observation(policy_batch))
        imitation_loss = F.cross_entropy(child_policy, policy_targets[policy_indexes])
        preservation_loss = F.kl_div(
            F.log_softmax(child_policy, dim=1),
            F.softmax(parent_policy, dim=1),
            reduction="batchmean",
        )
        loss = (
            feasibility_loss
            + bound_loss
            + ordinal_loss
            + monotonic_loss
            + policy_weight * imitation_loss
            + preservation_weight * preservation_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        losses.append(float(loss.item()))

    metrics = _evaluate(
        network,
        proof_observations,
        feasible,
        lower,
        upper,
        ordinal,
        ordinal_mask,
        pair_tensor,
    )
    network.eval()
    with torch.inference_mode():
        child_policy, _ = network(policy_tensor)
        parent_policy, _ = network.parent(network._parent_observation(policy_tensor))
        parent_policy_kl = float(
            F.kl_div(
                F.log_softmax(child_policy, dim=1),
                F.softmax(parent_policy, dim=1),
                reduction="batchmean",
            ).item()
        )
    unsafe_labels = sum(
        (not label.feasible and label.budget >= label.ratio * label.certified_lower_bound)
        for label in labels
    )
    metrics["parent_policy_kl"] = parent_policy_kl
    passed = bool(
        steps >= MINIMUM_TRAINING_STEPS
        and samples_per_side >= MINIMUM_SAMPLES_PER_SIDE
        and unsafe_labels == 0
        and metrics["feasibility_accuracy"] >= 0.80
        and metrics["ordinal_accuracy"] >= 0.80
        and metrics["bound_mae"] <= 1.0
        and metrics["budget_monotonic_violation_rate"] <= 0.05
        and metrics["parent_policy_kl"] <= 0.25
        and all(math.isfinite(value) for value in metrics.values())
    )
    report = {
        "schema": PROOF_SCHEMA,
        "status": "passed" if passed else "failed",
        "candidate": candidate_name,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "curriculum": str(curriculum_path.resolve()),
        "curriculum_sha256": file_sha256(curriculum_path),
        "evidence_snapshot": str(evidence_path.resolve()),
        "evidence_snapshot_sha256": actual_evidence_sha,
        "training_rows": len(rows),
        "exact_replayed_witness_rows": len(exact_for_row),
        "negative_budget_examples": sum(not label.feasible for label in labels),
        "positive_budget_examples": sum(label.feasible for label in labels),
        "witness_policy_examples": len(policy_actions),
        "unsafe_labels": unsafe_labels,
        "ratios": list(RATIOS),
        "samples_per_side": samples_per_side,
        "steps": steps,
        "minimum_training_steps": MINIMUM_TRAINING_STEPS,
        "training_dose_complete": steps >= MINIMUM_TRAINING_STEPS,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "policy_weight": policy_weight,
        "preservation_weight": preservation_weight,
        "seed": seed,
        "device": device,
        "final_loss": losses[-1] if losses else None,
        "mean_last_100_loss": sum(losses[-100:]) / len(losses[-100:]) if losses else None,
        "metrics": metrics,
        "operational_p_solve_trained": False,
        "positive_label_rule": "B >= ratio * replayed_crossing_changes + replayed_moves",
        "negative_label_rule": "B < ratio * certified_lower_bound",
        "ambiguous_interval_masked": True,
    }
    repo_root = Path(__file__).resolve().parents[2]
    report["source_hashes"] = {
        str(path.relative_to(repo_root)): file_sha256(path)
        for path in (
            Path(__file__).resolve(),
            repo_root / "src/pgx_mcts_bench/mastery_v3.py",
            repo_root / "research/mastery-v3-curriculum/protocol-spec.json",
        )
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        payload = dict(source_payload)
        payload["network"] = network.state_dict()
        payload["mastery_v3_proof_distillation"] = report
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report["checkpoint"] = str(output.resolve())
    report["checkpoint_sha256"] = file_sha256(output)
    _atomic_json(output.with_suffix(output.suffix + ".json"), report)
    _atomic_json(
        output.with_suffix(output.suffix + ".proof-dataset.json"),
        {
            "schema": "mastery-v3-proof-dataset-v1",
            "curriculum_sha256": report["curriculum_sha256"],
            "evidence_snapshot_sha256": actual_evidence_sha,
            "labels": [asdict(label) for label in labels],
        },
    )
    return report
