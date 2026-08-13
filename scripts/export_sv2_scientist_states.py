#!/usr/bin/env python3
"""Export per-scientist continuation payloads from a coordinated SV2 state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgx_mcts_bench.sv2_curriculum import _load_state, _save_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coordinated_state", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    state = _load_state(args.coordinated_state)
    selection = json.loads(args.selection.read_text())
    rows = selection.get("selected", selection.get("scientists"))
    if rows is None:
        raise SystemExit("selection must contain a selected or scientists list")
    names = [row["name"] for row in rows]
    if set(names) != set(state["scientists"]):
        raise SystemExit("selection and coordinated-state scientists differ")
    for name in names:
        destination = args.output / name / "state.pt.gz"
        _save_state(
            destination,
            {
                "schema": "semantic-v2-exported-scientist-state-v1",
                "scientist": state["scientists"][name],
                "f_old": int(state["f_old"][name]),
                "rehearsal_exposure": state["rehearsal_exposure"][name],
                "f_native": int(state.get("f_native", {}).get(name, 5)),
                "simulations": int(state.get("simulations", {}).get(name, 64)),
                "donation_dose": int(state.get("donation_dose", 1)),
                "donation_healthy_streak": int(state.get("donation_healthy_streak", 0)),
            },
        )
        print(f"{name}={destination.resolve()}")


if __name__ == "__main__":
    main()
