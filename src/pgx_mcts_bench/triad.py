from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from pgx_mcts_bench.game import make_game
from pgx_mcts_bench.ladder import RATIOS, STAGES, _config, ensemble_arms, evaluate_stage
from pgx_mcts_bench.networks import TriadBraidNet, make_braid_network


@dataclass(frozen=True)
class ParentSnapshot:
    candidate: str
    rung: int
    source: str
    scramble: int
    sha256: str
    path: str


@dataclass(frozen=True)
class TriadBuildReport:
    candidate: str
    checkpoint: str
    parents: tuple[ParentSnapshot, ...]
    initialization: str


EXPECTED_PARENTS = (
    ("s-window-128", 18),
    ("s-scan-gru", 10),
    ("s-tape4", 8),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parent(path: Path, expected_name: str, expected_rung: int) -> tuple[dict, ParentSnapshot]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("candidate") != expected_name or saved.get("stage") != expected_rung:
        raise ValueError(
            f"{path} is {saved.get('candidate')}@r{saved.get('stage')}, "
            f"expected {expected_name}@r{expected_rung}"
        )
    if saved.get("when") != "after" or "network" not in saved:
        raise ValueError(f"{path} is not a completed-rung network snapshot")
    return saved["network"], ParentSnapshot(
        candidate=expected_name,
        rung=expected_rung,
        source=str(saved.get("source", "")),
        scramble=int(saved.get("scramble", 0)),
        sha256=_sha256(path),
        path=str(path),
    )


def build_triad_checkpoint(
    window: Path,
    scan: Path,
    tape: Path,
    output: Path,
    *,
    device: str = "cpu",
) -> TriadBuildReport:
    paths = (window, scan, tape)
    loaded = [
        _load_parent(path, name, rung)
        for path, (name, rung) in zip(paths, EXPECTED_PARENTS, strict=True)
    ]
    candidate = ensemble_arms()[0]
    config = _config(candidate, STAGES[0], seed=0, device=device)
    network = make_braid_network(config.game, config.model).to(torch.device(device))
    if not isinstance(network, TriadBraidNet):
        raise TypeError("s-triad-wst did not construct TriadBraidNet")
    network.load_parent_state_dicts(*(state for state, _ in loaded))
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
    parents = tuple(snapshot for _, snapshot in loaded)
    payload = {
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stages": [],
        "candidate": candidate.name,
        "triad_parents": [asdict(parent) for parent in parents],
        "triad_initialization": (
            "frozen parents; per-parent centered/RMS-normalized logits; "
            "uniform available-parent average; zero router and residuals"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    report = TriadBuildReport(
        candidate=candidate.name,
        checkpoint=str(output),
        parents=parents,
        initialization=payload["triad_initialization"],
    )
    report_path = output.parent.parent / "triad-provenance.json"
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n")
    return report


def evaluate_frozen_triad(
    checkpoint: Path,
    output: Path,
    *,
    eval_games: int = 12,
    seed: int = 0,
    device: str = "cpu",
    stop_at_first_failure: bool = True,
    start_rung: int = 0,
    stop_rung: int = -1,
    crossing_tolerance: float = 0.25,
    promote_at: float = 0.8,
    collapse_floor: float = 0.5,
    log=print,
) -> dict:
    candidate = ensemble_arms()[0]
    first = _config(candidate, STAGES[0], seed, device)
    network = make_braid_network(first.game, first.model).to(torch.device(device))
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    network.load_state_dict(saved["network"])
    network.eval()

    rows = []
    recommended = None
    for index, stage in enumerate(STAGES):
        if index < start_rung:
            continue
        if 0 <= stop_rung < index:
            break
        config = _config(candidate, stage, seed, device)
        game = make_game(config.game)
        source = next(item for item in game.generator.sources if item.name == stage[0])
        by_ratio = evaluate_stage(
            game,
            network,
            config,
            eval_games,
            seed + 500_000 + index * 997,
        )
        rates = [float(value["solved"]) for value in by_ratio.values()]
        solve_rate = sum(rates) / len(rates)
        worst_ratio = min(rates)
        crossings = float(by_ratio[max(RATIOS)]["crossings"])
        optimum = int(source.unknotting_number)
        objective_ok = optimum < 0 or (
            np.isfinite(crossings) and crossings <= optimum + crossing_tolerance
        )
        clears = solve_rate >= promote_at and worst_ratio >= collapse_floor and objective_ok
        row = {
            "rung": index,
            "source": stage[0],
            "scramble": stage[1],
            "solve_rate": solve_rate,
            "worst_ratio_solve_rate": worst_ratio,
            "crossings_at_1000_1": crossings,
            "optimal_crossings": optimum,
            "clears_without_training": clears,
            "by_ratio": by_ratio,
        }
        rows.append(row)
        log(
            f"rung {index:2d} {stage[0]}+{stage[1]}: sr={solve_rate:.2f}, "
            f"worst={worst_ratio:.2f}, cc={crossings:.2f}, "
            f"u={optimum}, {'clear' if clears else 'TRAIN'}"
        )
        if not clears and recommended is None:
            recommended = index
            if stop_at_first_failure:
                break

    report = {
        "candidate": candidate.name,
        "checkpoint": str(checkpoint),
        "training_updates": 0,
        "eval_games_per_ratio": eval_games,
        "thresholds": {
            "solve_rate": promote_at,
            "worst_ratio_solve_rate": collapse_floor,
            "crossing_tolerance": crossing_tolerance,
        },
        "recommended_training_rung": recommended,
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen-frontier.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Frozen triad frontier",
        "",
        f"No training updates; {eval_games} held-out games per A:B ratio.",
        "",
        "| rung | instance | sr | worst sr | cc@1000:1 | u | verdict |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rung']} | {row['source']}+{row['scramble']} | "
            f"{row['solve_rate']:.2f} | {row['worst_ratio_solve_rate']:.2f} | "
            f"{row['crossings_at_1000_1']:.2f} | {row['optimal_crossings']} | "
            f"{'clear' if row['clears_without_training'] else 'start training'} |"
        )
    lines.extend(["", f"Recommended first training rung: **{recommended}**.", ""])
    (output / "frozen-frontier.md").write_text("\n".join(lines))
    return report
