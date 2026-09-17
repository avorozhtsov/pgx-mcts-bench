#!/usr/bin/env python3
"""Run roster-name recovery for the fourth Fast Q305 descendant."""

import run_q305_same_root_tournament_pilot as base

base.GATE = base.ROOT / "Q305_TRIMMED_DIVERGENCE_RECOVERY_V4_VERIFIED.json"
base.STATUS = base.ROOT / "q305-tournament-v1/q305-pilot-launcher-status-v4.json"
base.LOCK = base.ROOT / "q305-tournament-v1/q305-pilot-launcher-v4.lock"

if __name__ == "__main__":
    base.main()
