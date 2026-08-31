#!/usr/bin/env python3
"""Fail-closed policy helpers for the focused post-Q254/Q154 successor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path("/Users/artemvorozhtsov/projects/pgx-mcts-bench-local-ablation")
POLICY = REPO / "research/local-q-skm-ablation/focused-successor-v1-policy.json"
SCHEMA = "focused-successor-v1-policy"


def load_policy(path: Path = POLICY) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("focused successor policy schema differs")
    return payload


def assert_legacy_q304_launch_authorized(path: Path = POLICY) -> dict[str, Any]:
    payload = load_policy(path)
    legacy = payload.get("legacy_q304", {})
    if legacy.get("launch_authorized") is not True:
        raise RuntimeError(
            "legacy fast-6 Q304 is superseded and is not authorized to launch; "
            "use the focused successor program"
        )
    return payload
