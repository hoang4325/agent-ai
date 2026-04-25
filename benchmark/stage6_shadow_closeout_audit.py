from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .io import dump_json, load_json


def run_stage6_shadow_closeout_audit(repo_root: str | Path) -> dict:
    repo_root = Path(repo_root)
    stop_readiness = load_json(repo_root / "benchmark" / "stage6_stop_target_readiness.json", default={}) or {}
    preflight = load_json(repo_root / "benchmark" / "stage6_preflight_revalidation.json", default={}) or {}
    shadow_readiness = load_json(repo_root / "benchmark" / "stage6_shadow_readiness.json", default={}) or {}

    stop_alignment_blocker = "none"
    if stop_readiness.get("status") != "ready":
        blockers = list(stop_readiness.get("remaining_blockers") or [])
        stop_alignment_blocker = blockers[0] if blockers else "stop_alignment_not_ready"

    fallback_blocker = "none"
    remaining = list(preflight.get("remaining_blockers") or []) + list(shadow_readiness.get("remaining_blockers") or [])
    for blocker in remaining:
        if "fallback" in str(blocker):
            fallback_blocker = str(blocker)
            break

    payload = {
        "schema_version": "stage6_shadow_closeout_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stop_alignment_blocker": stop_alignment_blocker,
        "fallback_blocker": fallback_blocker,
        "recommended_stop_case": "stop_follow_ambiguity_core",
        "recommended_fallback_probe_case": "ml_right_positive_core",
        "notes": [
            "stop blocker is considered open whenever stage6_stop_target_readiness is not ready.",
            "fallback blocker is considered open whenever Stage 6 readiness still reports missing online takeover evidence.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_shadow_closeout_audit.json", payload)
    return payload
