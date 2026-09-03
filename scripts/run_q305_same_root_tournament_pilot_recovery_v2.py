#!/usr/bin/env python3
"""Run the stable-loss Q305 tournament recovery from the frozen Q304 parent."""

import run_q305_same_root_tournament_pilot as base

base.GATE = base.ROOT / "Q305_SAME_ROOT_TOURNAMENT_PILOT_RECOVERY_V2_VERIFIED.json"
base.STATUS = base.ROOT / "q305-tournament-v1/q305-pilot-launcher-status-recovery-v2.json"
base.LOCK = base.ROOT / "q305-tournament-v1/q305-pilot-launcher-recovery-v2.lock"

if __name__ == "__main__":
    base.main()
