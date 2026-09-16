#!/usr/bin/env python3
"""Run the gated fourth Fast descendant from the frozen Q304 parent."""

import run_q305_same_root_tournament_pilot as base

base.GATE = base.ROOT / "Q305_TRIMMED_DIVERGENCE_PILOT_V3_VERIFIED.json"
base.STATUS = base.ROOT / "q305-tournament-v1/q305-pilot-launcher-status-v3.json"
base.LOCK = base.ROOT / "q305-tournament-v1/q305-pilot-launcher-v3.lock"

if __name__ == "__main__":
    base.main()
