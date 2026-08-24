#!/usr/bin/env python3
"""Build small proof-carrying tails for recursive DKT unknotting chains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


TEN_129_PD = [
    [1, 7, 2, 6],
    [3, 17, 4, 16],
    [5, 9, 6, 8],
    [7, 3, 8, 2],
    [10, 19, 11, 20],
    [12, 17, 13, 18],
    [14, 9, 15, 10],
    [15, 5, 16, 4],
    [18, 11, 19, 12],
    [20, 13, 1, 14],
]


class UnionFind:
    def __init__(self, elements: Iterable[int]) -> None:
        self.parent = {int(x): int(x) for x in elements}

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        x, y = self.find(x), self.find(y)
        if x != y:
            self.parent[max(x, y)] = min(x, y)


def successor(label: int, edge_count: int) -> int:
    return label % edge_count + 1


def change_crossing(labels: Sequence[int], edge_count: int) -> list[int]:
    incoming, first_upper, outgoing, second_upper = map(int, labels)
    if successor(incoming, edge_count) != outgoing:
        raise ValueError("PD under-strand orientation is inconsistent")
    if successor(second_upper, edge_count) == first_upper:
        return [second_upper, incoming, first_upper, outgoing]
    if successor(first_upper, edge_count) == second_upper:
        return [first_upper, outgoing, second_upper, incoming]
    raise ValueError("PD upper-strand orientation is inconsistent")


def inverse(word: Sequence[int]) -> list[int]:
    return [-x for x in reversed(word)]


def reduce_word(word: Sequence[int], cyclic: bool = True) -> list[int]:
    result: list[int] = []
    for letter in word:
        if result and result[-1] == -letter:
            result.pop()
        else:
            result.append(int(letter))
    if cyclic:
        while len(result) > 1 and result[0] == -result[-1]:
            result = result[1:-1]
    return result


def substitute(word: Sequence[int], generator: int, replacement: list[int]) -> list[int]:
    out: list[int] = []
    for letter in word:
        if letter == generator:
            out.extend(replacement)
        elif letter == -generator:
            out.extend(inverse(replacement))
        else:
            out.append(letter)
    return reduce_word(out)


def wirtinger_presentation(pd: Sequence[Sequence[int]]) -> dict[str, object]:
    edge_count = 2 * len(pd)
    labels = list(range(1, edge_count + 1))
    arcs = UnionFind(labels)
    for crossing in pd:
        arcs.union(int(crossing[1]), int(crossing[3]))
    roots = sorted({arcs.find(x) for x in labels})
    root_generator = {root: i + 1 for i, root in enumerate(roots)}
    generator = {x: root_generator[arcs.find(x)] for x in labels}
    relators: list[list[int]] = []
    for incoming, first_upper, outgoing, second_upper in pd:
        inc, out, upper = generator[incoming], generator[outgoing], generator[first_upper]
        if successor(second_upper, edge_count) == first_upper:
            relator = [-out, -upper, inc, upper]
        elif successor(first_upper, edge_count) == second_upper:
            relator = [-out, upper, inc, -upper]
        else:
            raise ValueError("PD upper-strand orientation is inconsistent")
        relators.append(reduce_word(relator))
    return {"generators": list(range(1, len(roots) + 1)), "relators": relators}


def tietze_reduce(presentation: dict[str, object]) -> dict[str, object]:
    generators = set(map(int, presentation["generators"]))
    relators = [reduce_word(x) for x in presentation["relators"]]
    relators = [x for x in relators if x]
    steps: list[dict[str, object]] = []
    while True:
        choice = None
        for ri, relator in enumerate(relators):
            for gen in sorted(generators):
                positions = [i for i, letter in enumerate(relator) if abs(letter) == gen]
                if len(positions) == 1:
                    choice = (ri, gen, positions[0])
                    break
            if choice:
                break
        if not choice:
            break
        ri, gen, pos = choice
        relator = relators[ri]
        rotated = relator[pos:] + relator[:pos]
        replacement = inverse(rotated[1:]) if rotated[0] == gen else rotated[1:]
        replacement = reduce_word(replacement, cyclic=False)
        steps.append({
            "eliminated_generator": gen,
            "using_relator": relator,
            "replacement_word": replacement,
        })
        relators = [
            substitute(other, gen, replacement)
            for i, other in enumerate(relators)
            if i != ri
        ]
        relators = [x for x in relators if x]
        generators.remove(gen)
    return {
        "remaining_generators": sorted(generators),
        "remaining_relators": relators,
        "elimination_steps": steps,
        "reduces_to_infinite_cyclic": len(generators) == 1 and not relators,
    }


def certificate(pd: Sequence[Sequence[int]], crossing_index: int) -> dict[str, object]:
    changed = [
        change_crossing(crossing, 2 * len(pd)) if i == crossing_index else list(crossing)
        for i, crossing in enumerate(pd)
    ]
    presentation = wirtinger_presentation(changed)
    reduction = tietze_reduce(presentation)
    if not reduction["reduces_to_infinite_cyclic"]:
        raise RuntimeError("candidate tail did not reduce to the infinite cyclic group")
    return {
        "crossing_index_0_based": crossing_index,
        "crossing_position_1_based": crossing_index + 1,
        "source_pd": [list(x) for x in pd],
        "changed_pd": changed,
        "wirtinger_presentation": presentation,
        "tietze_reduction": reduction,
    }


def build() -> dict[str, object]:
    tail = certificate(TEN_129_PD, 0)
    payload = {
        "schema": "dkt-recursive-solution-seeds-v1",
        "status": "PREPARED",
        "scope": "evaluation and curriculum design only; not clean DKT benchmark input",
        "sources": {
            "unknot_repository": {
                "url": "https://github.com/dtubbenhauer/unknot",
                "commit": "f93552d55d02718049fc8696641565aae4ab08ae",
                "upstream_certificate_script_sha256": "a086fc0ee689688558e08a8bce645d984cd1c72e798f76cd348a581c6e744068",
                "identification_rows_sha256": "3bc8af396f0352c83ef7c502c189d90ad51cda8cecfcc540b396d7051a4e5de4",
            },
            "paper": {
                "url": "https://arxiv.org/abs/2603.07955",
                "reported_edge": "11a_14 --one crossing change after increase/shuffle--> 10_129",
            },
        },
        "chains": [
            {
                "start": "11a_14",
                "reported_upper_bound": 2,
                "edges": [
                    {
                        "from": "11a_14",
                        "to": "10_129",
                        "crossing_changes": 1,
                        "certificate_level": "paper-figure-only",
                        "machine_replayable": False,
                    },
                    {
                        "from": "10_129",
                        "to": "unknot",
                        "crossing_changes": 1,
                        "certificate_level": "exact-wirtinger-tietze",
                        "machine_replayable_pd_tail": True,
                        "certificate": tail,
                    },
                ],
                "end_to_end_machine_replayable": False,
                "missing_piece": "a machine-readable PD and isotopy/action trace for the paper's first edge",
            }
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["semantic_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
