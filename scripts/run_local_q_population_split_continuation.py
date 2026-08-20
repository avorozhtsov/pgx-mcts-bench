#!/usr/bin/env python3
"""Resume the Q population as independent branches from durable checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
RUN = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "local-q-skm-ablation-20260817"
)
ARCHIVE = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "nebius-main32-final-20260817/artifacts/q4000-strand12-20260814"
)
MASTERY = Path(
    "/Users/artemvorozhtsov/projects/pgx-mcts-bench/artifacts/"
    "nebius-main32-final-20260817/artifacts"
)
QROOT = RUN / "inputs/q4000-v1"
ROOT = RUN / "continuation/q4000-v1-population-20260818"
SPLIT = ROOT / "split-20260819"
TAIL = ROOT / "q20-recovery-tail-static-no-sharing"
RECOVERY = ROOT / "q20-recovery"
STATUS = SPLIT / "launcher-status.json"
INVARIANT_TIMEOUT_SECONDS = 3600
MAX_EXPERIMENT_CORES = 6

CHECKPOINTS = {
    "raster-axial-12": ARCHIVE / "migrated/raster-axial-12/checkpoint.pt",
    "raster-invariant-combined-dual-12": (
        ARCHIVE / "migrated/raster-invariant-combined-dual-12/checkpoint.pt"
    ),
    "strand-graph-12": ARCHIVE / "migrated/strand-graph-12/checkpoint.pt",
    "cyclic-memory-12": ARCHIVE / "migrated/cyclic-memory-12/checkpoint.pt",
}
SELECTIONS = {
    "raster-axial-12": REPO
    / "research/local-q-skm-ablation/single-raster-axial-12-selection.json",
    "raster-invariant-combined-dual-12": REPO
    / "research/local-q-skm-ablation/"
    "single-raster-invariant-combined-dual-12-selection.json",
    "strand-graph-12": REPO
    / "research/local-q-skm-ablation/single-strand-graph-12-selection.json",
    "cyclic-memory-12": REPO
    / "research/local-q-skm-ablation/single-cyclic-memory-12-selection.json",
}

_status_lock = threading.Lock()
_status: dict[str, object] = {
    "schema": "q4000-split-population-launcher-v1",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "max_experiment_cores": MAX_EXPERIMENT_CORES,
    "invariant_timeout_seconds": INVARIANT_TIMEOUT_SECONDS,
    "branches": {},
}


def _write_status() -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(_status, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATUS)


def _set_status(label: str, state: str, stage: str, detail: str | None = None) -> None:
    with _status_lock:
        branches = _status["branches"]
        assert isinstance(branches, dict)
        branches[label] = {
            "state": state,
            "stage": stage,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_status()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(f"hash mismatch for {path}: {observed} != {expected}")


def _run(command: list[str], *, log: Path | None = None) -> None:
    print("EXEC", " ".join(command), flush=True)
    if log is None:
        subprocess.run(command, cwd=REPO, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        subprocess.run(
            command,
            cwd=REPO,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def _audit(bank: Path, output: Path, log: Path) -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/audit_local_q4000_group.py",
            "--group",
            str(bank),
            "--output",
            str(output),
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with log.open("a") as handle:
        handle.write(result.stdout)
    if result.returncode == 0:
        return
    audit = json.loads((output / "local-group-audit.json").read_text())
    allowed = (
        "retention ",
        "no native self-play success on a 6+ strand representation",
    )
    unexpected = [
        failure
        for failure in audit.get("failures", [])
        if not any(str(failure).startswith(prefix) for prefix in allowed)
    ]
    if unexpected:
        raise RuntimeError(f"unexpected structural audit failures: {unexpected}")


def _export(state: Path, scientist: str, output: Path, log: Path) -> Path:
    destination = output / scientist / "state.pt.gz"
    if not destination.is_file():
        _run(
            [
                "uv",
                "run",
                "python",
                "scripts/export_sv2_scientist_states.py",
                str(state),
                str(SELECTIONS[scientist]),
                str(output),
            ],
            log=log,
        )
    return destination


def _run_stage(
    *,
    label: str,
    stage: str,
    output: Path,
    bank: Path,
    prior_bank: Path | None,
    scientist: str,
    initial_state: Path,
    simulations: int,
    seed: int,
    block_size: int,
    timeout: bool,
) -> None:
    log = SPLIT / "logs" / f"{label}.log"
    if (output / "report.json").is_file():
        _set_status(label, "COMPLETED", stage, "existing durable report")
        return
    command = [
        "uv",
        "run",
        "pgx-mcts-bench",
        "braid-sv2-coordinated",
        "--output",
        str(output),
        "--bank",
        str(bank),
    ]
    if prior_bank is not None:
        command += ["--prior-bank", str(prior_bank)]
    command += [
        "--scientist",
        f"{scientist}={CHECKPOINTS[scientist]}",
        "--initial-state",
        f"{scientist}={initial_state}",
        "--arm",
        "static-no-sharing",
        "--ratios",
        "10,1000",
        "--simulations",
        str(simulations),
        "--qualification-simulations",
        str(simulations),
        "--qualification-attempts",
        "1",
        "--f-native",
        "5",
        "--selfplay-games",
        "8",
        "--train-steps",
        "96",
        "--batch-size",
        "64",
        "--evaluation-attempts",
        "4",
        "--block-size",
        str(block_size),
        "--retention-target",
        "0.8",
        "--action-horizon",
        "128",
        "--rungs",
        "0",
        "--seed",
        str(seed),
        "--torch-threads",
        "1",
        "--parallel-scientists",
        "--adaptive-compute",
        "--device",
        "cpu",
    ]
    if timeout:
        command += [
            "--scientist-task-timeout-seconds",
            str(INVARIANT_TIMEOUT_SECONDS),
        ]
    if (output / "manifest.json").is_file():
        command.append("--resume")
    _set_status(label, "LAUNCHED", stage, "awaiting live PID and artifact verification")
    _run(command, log=log)
    _audit(bank, output, log)
    _set_status(label, "COMPLETED", stage)


def _run_split_branch(
    label: str,
    scientist: str,
    simulations: int,
    remaining_seed: int,
    q40_seed: int,
    *,
    timeout: bool = False,
) -> None:
    branch = SPLIT / "branches" / label
    log = SPLIT / "logs" / f"{label}.log"
    initial = SPLIT / "source-exports" / scientist / "state.pt.gz"
    _run_stage(
        label=label,
        stage="Q20 remaining 4/4",
        output=branch / "q20-remaining-static-no-sharing",
        bank=SPLIT / "protocol/q20-remaining.json",
        prior_bank=SPLIT / "protocol/prior-q20-remaining.json",
        scientist=scientist,
        initial_state=initial,
        simulations=simulations,
        seed=remaining_seed,
        block_size=4,
        timeout=timeout,
    )
    exported = _export(
        branch / "q20-remaining-static-no-sharing/state.pt.gz",
        scientist,
        branch / "q20-export",
        log,
    )
    _run_stage(
        label=label,
        stage="Q40-1",
        output=branch / "q40-1-static-no-sharing",
        bank=QROOT / "q40-1.json",
        prior_bank=QROOT / "prior-q40-1.json",
        scientist=scientist,
        initial_state=exported,
        simulations=simulations,
        seed=q40_seed,
        block_size=10,
        timeout=timeout,
    )
    (branch / "Q60_COMPLETE").touch()


def _run_cyclic_recovery() -> None:
    label = "q-grown-cyclic-memory-12"
    scientist = "cyclic-memory-12"
    branch = SPLIT / "branches" / label
    log = SPLIT / "logs" / f"{label}.log"
    _run_stage(
        label=label,
        stage="Q20 recovery 10/10",
        output=branch / "q20-recovery-tail-static-no-sharing",
        bank=RECOVERY / "q20-tail.json",
        prior_bank=RECOVERY / "prior-q20-tail.json",
        scientist=scientist,
        initial_state=ROOT / "source-exports/cyclic-memory-12/state.pt.gz",
        simulations=128,
        seed=2026081841,
        block_size=10,
        timeout=False,
    )
    exported = _export(
        branch / "q20-recovery-tail-static-no-sharing/state.pt.gz",
        scientist,
        branch / "q20-export",
        log,
    )
    _run_stage(
        label=label,
        stage="Q40-1",
        output=branch / "q40-1-static-no-sharing",
        bank=QROOT / "q40-1.json",
        prior_bank=QROOT / "prior-q40-1.json",
        scientist=scientist,
        initial_state=exported,
        simulations=128,
        seed=2026081842,
        block_size=10,
        timeout=False,
    )
    (branch / "Q60_COMPLETE").touch()


def _run_full_branch(
    label: str,
    scientist: str,
    initial_state: Path,
    simulations: int,
    seed: int,
    *,
    timeout: bool = False,
) -> None:
    branch = SPLIT / "branches" / label
    log = SPLIT / "logs" / f"{label}.log"
    _run_stage(
        label=label,
        stage="Q20",
        output=branch / "q20-static-no-sharing",
        bank=QROOT / "q20.json",
        prior_bank=None,
        scientist=scientist,
        initial_state=initial_state,
        simulations=simulations,
        seed=seed,
        block_size=10,
        timeout=timeout,
    )
    exported = _export(
        branch / "q20-static-no-sharing/state.pt.gz",
        scientist,
        branch / "q20-export",
        log,
    )
    _run_stage(
        label=label,
        stage="Q40-1",
        output=branch / "q40-1-static-no-sharing",
        bank=QROOT / "q40-1.json",
        prior_bank=QROOT / "prior-q40-1.json",
        scientist=scientist,
        initial_state=exported,
        simulations=simulations,
        seed=seed + 1,
        block_size=10,
        timeout=timeout,
    )
    (branch / "Q60_COMPLETE").touch()


def _prepare() -> None:
    SPLIT.mkdir(parents=True, exist_ok=True)
    _require_hash(TAIL / "state.pt.gz", "9152411003b02f0b97aa94916664ee1073b014770d0af918eecff28917bbee4f")
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/build_q20_split_remaining.py",
            "--source-bank",
            str(QROOT / "q20.json"),
            "--initial-prior-bank",
            str(RECOVERY / "prior-q20-tail.json"),
            "--tail-bank",
            str(RECOVERY / "q20-tail.json"),
            "--tail-state",
            str(TAIL / "state.pt.gz"),
            "--remaining-bank",
            str(SPLIT / "protocol/q20-remaining.json"),
            "--prior-bank",
            str(SPLIT / "protocol/prior-q20-remaining.json"),
            "--manifest",
            str(SPLIT / "protocol/q20-split-manifest.json"),
        ]
    )
    if not (SPLIT / "source-exports/raster-axial-12/state.pt.gz").is_file():
        _run(
            [
                "uv",
                "run",
                "python",
                "scripts/export_sv2_scientist_states.py",
                str(TAIL / "state.pt.gz"),
                str(
                    REPO
                    / "research/local-q-skm-ablation/"
                    "q4000-population-three-selection.json"
                ),
                str(SPLIT / "source-exports"),
            ]
        )

    bridge_root = SPLIT / "skm-bridges"
    bridges = [
        (
            MASTERY
            / "multi-knot-mastery-v2-20260815/scientists/"
            "high-combined-dual/scientist-state.pt.gz",
            bridge_root / "skm-v2-high-combined-dual.pt.gz",
            "raster-invariant-combined-dual-12",
            "2",
            "5",
            "32",
            "skm-v2-high-combined-dual",
        ),
        (
            MASTERY
            / "multi-knot-mastery-v2-20260815/scientists/"
            "high-cyclic-memory/scientist-state.pt.gz",
            bridge_root / "skm-v2-high-cyclic-memory.pt.gz",
            "cyclic-memory-12",
            "2",
            "5",
            "128",
            "skm-v2-high-cyclic-memory",
        ),
        (
            MASTERY
            / "multi-knot-mastery-v1-20260815/scientists/"
            "simple-raster-axial/scientist-state.pt.gz",
            bridge_root / "skm-v1-simple-raster-axial.pt.gz",
            "raster-axial-12",
            "2",
            "5",
            "64",
            "skm-v1-simple-raster-axial",
        ),
        (
            MASTERY
            / "multi-knot-mastery-v1-20260815/scientists/"
            "simple-strand-graph/scientist-state.pt.gz",
            bridge_root / "skm-v1-simple-strand-graph.pt.gz",
            "strand-graph-12",
            "8",
            "5",
            "64",
            "skm-v1-simple-strand-graph",
        ),
    ]
    for source, destination, scientist, f_old, f_native, simulations, lineage in bridges:
        if destination.is_file():
            continue
        _run(
            [
                "uv",
                "run",
                "python",
                "scripts/bridge_mastery_state_to_sv2.py",
                str(source),
                str(destination),
                "--scientist",
                scientist,
                "--f-old",
                f_old,
                "--f-native",
                f_native,
                "--simulations",
                simulations,
                "--lineage",
                lineage,
            ]
        )


def main() -> None:
    os.environ.update(
        {
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    _write_status()
    _prepare()
    bridge = SPLIT / "skm-bridges"
    jobs: list[tuple[str, Callable[[], None]]] = [
        (
            "q-grown-raster-invariant-combined-dual-12",
            lambda: _run_split_branch(
                "q-grown-raster-invariant-combined-dual-12",
                "raster-invariant-combined-dual-12",
                64,
                2136081831,
                2126081832,
                timeout=True,
            ),
        ),
        (
            "q-grown-raster-axial-12",
            lambda: _run_split_branch(
                "q-grown-raster-axial-12",
                "raster-axial-12",
                64,
                2036081831,
                2026081832,
            ),
        ),
        (
            "q-grown-strand-graph-12",
            lambda: _run_split_branch(
                "q-grown-strand-graph-12",
                "strand-graph-12",
                128,
                2236081831,
                2226081832,
            ),
        ),
        ("q-grown-cyclic-memory-12", _run_cyclic_recovery),
        (
            "skm-v2-high-cyclic-memory",
            lambda: _run_full_branch(
                "skm-v2-high-cyclic-memory",
                "cyclic-memory-12",
                bridge / "skm-v2-high-cyclic-memory.pt.gz",
                128,
                2026081853,
            ),
        ),
        (
            "skm-v1-simple-raster-axial",
            lambda: _run_full_branch(
                "skm-v1-simple-raster-axial",
                "raster-axial-12",
                bridge / "skm-v1-simple-raster-axial.pt.gz",
                64,
                2026081855,
            ),
        ),
        (
            "skm-v1-simple-strand-graph",
            lambda: _run_full_branch(
                "skm-v1-simple-strand-graph",
                "strand-graph-12",
                bridge / "skm-v1-simple-strand-graph.pt.gz",
                64,
                2026081857,
            ),
        ),
        (
            "skm-v2-high-combined-dual",
            lambda: _run_full_branch(
                "skm-v2-high-combined-dual",
                "raster-invariant-combined-dual-12",
                bridge / "skm-v2-high-combined-dual.pt.gz",
                32,
                2026081851,
                timeout=True,
            ),
        ),
    ]
    for label, _job in jobs:
        _set_status(label, "QUEUED", "awaiting one of six experiment slots")

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_EXPERIMENT_CORES) as executor:
        future_labels = {}
        for label, job in jobs:
            def wrapped(label: str = label, job: Callable[[], None] = job) -> None:
                try:
                    job()
                    _set_status(label, "COMPLETED", "Q60")
                except Exception as error:
                    _set_status(label, "BLOCKED", "failed", repr(error))
                    raise

            future_labels[executor.submit(wrapped)] = label
        for future in as_completed(future_labels):
            label = future_labels[future]
            try:
                future.result()
            except Exception as error:
                failures.append(f"{label}: {error!r}")

    with _status_lock:
        _status["finished_at"] = datetime.now(timezone.utc).isoformat()
        _status["state"] = "BLOCKED" if failures else "COMPLETED"
        _status["failures"] = failures
        _write_status()
    if failures:
        raise SystemExit("; ".join(failures))
    (SPLIT / "ALL_PROMISING_LINEAGES_Q60_COMPLETE").touch()


if __name__ == "__main__":
    main()
