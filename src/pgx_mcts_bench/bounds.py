"""Best-known unknotting bounds, as a ratchet rather than a theorem.

Requiring an exact `u` is what forced the instance family to be torus knots and
positive braids -- those are the knots we can *label*, and the price is that every
one of them is fibred, chiral, positive-signature, and satisfies `u = g3 = g4`. An
agent can learn "reduce monotonically, crossing changes always pay", be right on
the entire family, and have learned nothing that transfers.

So the label is dropped. A knot's reference value is the fewest crossing changes
**anyone has ever used** to unknot it. That is an upper bound on `u`, it is a
witness rather than an estimate, and it improves whenever any agent beats it. Any
knot becomes usable -- including random mixed-sign words, where the positivity
constraint and all of its consequences are gone.

The reference is the minimum over every agent and every run, deliberately not one
designated agent's answer: ranking by "gap to reference" against a single agent's
number would define that agent to have gap zero.

Storage is an append-only log rather than a mutable table. Many worker processes
claim concurrently, and a small `O_APPEND` write is atomic on POSIX, so there is
no lock, no read-modify-write race, and nothing is ever lost -- a superseded claim
just stops being the minimum. `best()` folds the log on read.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def knot_id(word: tuple[int, ...] | list[int], strands: int) -> str:
    """Identity of the *knot*, not of the diagram it arrived in.

    Every instance the generator produces from a source is the same knot as that
    source -- the scramble moves are type-preserving -- so all of them claim
    against one identity. For a source that is itself a random word, the word is
    the identity.
    """
    letters = ",".join(str(int(x)) for x in word if int(x))
    return f"b{strands}:{letters}" if letters else f"b{strands}:e"


@dataclass(frozen=True)
class Bound:
    knot: str
    crossings: int
    moves: int
    agent: str
    witness: list[int]
    strands: int

    def beats(self, other: Bound | None) -> bool:
        if other is None:
            return True
        # Crossing changes first: that is the quantity bounding u. Moves break
        # ties, so a shorter route to the same bound still counts as progress.
        return (self.crossings, self.moves) < (other.crossings, other.moves)


def claim(path: Path, bound: Bound) -> None:
    """Append a claim. Never blocks, never overwrites, never loses a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "knot": bound.knot,
            "crossings": bound.crossings,
            "moves": bound.moves,
            "agent": bound.agent,
            "witness": list(bound.witness),
            "strands": bound.strands,
        }
    )
    # O_APPEND so concurrent workers interleave whole lines instead of racing.
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(handle, (line + "\n").encode())
    finally:
        os.close(handle)


def best(path: Path) -> dict[str, Bound]:
    """Fold the log: the standing record for every knot anyone has unknotted."""
    records: dict[str, Bound] = {}
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A torn line from a crashed writer is skipped rather than fatal --
            # the log is evidence, and one damaged record should not cost the rest.
            continue
        bound = Bound(
            knot=row["knot"], crossings=row["crossings"], moves=row["moves"],
            agent=row["agent"], witness=row.get("witness", []),
            strands=row.get("strands", 0),
        )
        if bound.beats(records.get(bound.knot)):
            records[bound.knot] = bound
    return records


def report(path: Path) -> str:
    records = best(path)
    lines = [
        "# Best known unknotting bounds",
        "",
        f"{len(records)} knots, each showing the fewest crossing changes any agent",
        "has used. These are **upper bounds** on u(K) carrying an explicit witness,",
        "not proved values: a smaller number may exist and would replace this one.",
        "",
        "| knot | strands | crossings | moves | held by |",
        "|---|---:|---:|---:|---|",
    ]
    for bound in sorted(records.values(), key=lambda b: (-b.crossings, b.knot)):
        lines.append(
            f"| `{bound.knot[:44]}` | {bound.strands} | {bound.crossings} "
            f"| {bound.moves} | `{bound.agent}` |"
        )
    return "\n".join(lines) + "\n"
