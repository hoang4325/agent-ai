"""
Stage 7 — Agent Shadow Metric Pack
=====================================
PHASE A7.4

Computes all 10 Stage 7 agent shadow metrics from:
  - agent_shadow_intent.jsonl
  - agent_shadow_summary.json
  - stage3b_behavior_timeline (baseline reference)

Metrics:
  1.  agent_contract_validity_rate          — hard_gate_ready
  2.  agent_timeout_rate                    — hard_gate_ready
  3.  agent_fallback_to_baseline_rate       — hard_gate_ready
  4.  agent_baseline_disagreement_rate      — soft_guardrail
  5.  agent_useful_disagreement_rate        — soft_guardrail
  6.  agent_route_alignment_rate            — soft_guardrail
  7.  agent_stop_context_correctness_rate   — diagnostic_only
  8.  agent_lane_change_intent_validity_rate— hard_gate_ready
  9.  agent_forbidden_intent_count          — hard_gate_ready
  10. agent_shadow_artifact_completeness    — hard_gate_ready
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_json, load_jsonl, resolve_repo_path

ALLOWED_AGENT_INTENTS = frozenset({
    "keep_lane", "follow", "slow_down", "stop", "yield",
    "prepare_lane_change_left", "prepare_lane_change_right",
    "commit_lane_change_left", "commit_lane_change_right",
    "keep_route_through_junction",
})

LC_LEFT_INTENTS = {"prepare_lane_change_left", "commit_lane_change_left"}
LC_RIGHT_INTENTS = {"prepare_lane_change_right", "commit_lane_change_right"}
LC_INTENTS = LC_LEFT_INTENTS | LC_RIGHT_INTENTS


def _safe_div(n: float, d: float) -> float | None:
    return None if d <= 0 else float(n / d)


def _load_agent_intents(case_run_dir: Path) -> list[dict[str, Any]]:
    """Load agent_shadow_intent.jsonl from case run directory."""
    path = case_run_dir / "stage7" / "agent_shadow_intent.jsonl"
    if not path.exists():
        return []
    return load_jsonl(path)


def _load_agent_summary(case_run_dir: Path) -> dict[str, Any]:
    """Load agent_shadow_summary.json from case run directory."""
    path = case_run_dir / "stage7" / "agent_shadow_summary.json"
    if not path.exists():
        return {}
    return load_json(path) or {}


# ── Metric calculators ────────────────────────────────────────────────────────

def agent_contract_validity_rate(intents: list[dict[str, Any]]) -> float | None:
    """Fraction of frames where agent output passes full contract validation (status == 'valid')."""
    if not intents:
        return None
    valid = sum(1 for r in intents if r.get("validation_status") == "valid")
    return _safe_div(float(valid), float(len(intents)))


def agent_timeout_rate(intents: list[dict[str, Any]]) -> float | None:
    """Fraction of frames where agent timed out."""
    if not intents:
        return None
    timeouts = sum(1 for r in intents if bool(r.get("timeout_flag", False)))
    return _safe_div(float(timeouts), float(len(intents)))


def agent_fallback_to_baseline_rate(intents: list[dict[str, Any]]) -> float | None:
    """Fraction of frames where agent fell back to baseline (timeout or invalid)."""
    if not intents:
        return None
    fallbacks = sum(1 for r in intents if bool(r.get("fallback_to_baseline", False)))
    return _safe_div(float(fallbacks), float(len(intents)))


def agent_baseline_disagreement_rate(intents: list[dict[str, Any]]) -> float | None:
    """
    Fraction of valid frames where agent disagrees with baseline.
    Excludes fallback frames (would be forced agreement anyway).
    """
    valid_frames = [r for r in intents if r.get("validation_status") == "valid" and not r.get("fallback_to_baseline")]
    if not valid_frames:
        return None
    disagreements = sum(1 for r in valid_frames if not bool(r.get("agrees_with_baseline", True)))
    return _safe_div(float(disagreements), float(len(valid_frames)))


def agent_useful_disagreement_rate(intents: list[dict[str, Any]]) -> float | None:
    """
    Among frames where agent disagrees with baseline (valid, non-fallback, different intent),
    fraction where the disagreement is marked as structurally useful.
    """
    disagreement_frames = [
        r for r in intents
        if r.get("validation_status") == "valid"
        and not r.get("fallback_to_baseline")
        and not r.get("agrees_with_baseline", True)
    ]
    if not disagreement_frames:
        return None
    useful = sum(1 for r in disagreement_frames if bool(r.get("disagreement_useful", False)))
    return _safe_div(float(useful), float(len(disagreement_frames)))


def agent_route_alignment_rate(intents: list[dict[str, Any]]) -> float | None:
    """
    Fraction of valid (non-fallback) frames where agent's target_lane aligns with
    the route context's preferred_lane.
    """
    valid_frames = [r for r in intents if r.get("validation_status") == "valid" and not r.get("fallback_to_baseline")]
    if not valid_frames:
        return None
    aligned = 0
    eligible = 0
    for r in valid_frames:
        rcs = r.get("route_context_summary") or {}
        preferred = rcs.get("preferred_lane")
        agent_lane = r.get("target_lane", "current")
        if not preferred:
            continue
        eligible += 1
        if preferred == agent_lane or preferred == "current":
            aligned += 1
    return _safe_div(float(aligned), float(eligible))


def agent_stop_context_correctness_rate(intents: list[dict[str, Any]]) -> float | None:
    """
    Fraction of stop/yield intents issued by the agent where the stop_context_summary
    contains at least a binding_status field (confirming agent consumed stop context).
    """
    stop_intents = [
        r for r in intents
        if r.get("tactical_intent") in {"stop", "yield", "slow_down"}
        and r.get("validation_status") == "valid"
        and not r.get("fallback_to_baseline")
    ]
    if not stop_intents:
        return None
    with_context = sum(
        1 for r in stop_intents
        if (r.get("stop_context_summary") or {}).get("stop_target_binding") is not None
    )
    return _safe_div(float(with_context), float(len(stop_intents)))


def agent_lane_change_intent_validity_rate(intents: list[dict[str, Any]]) -> float | None:
    """
    Fraction of agent LC intents (prepare/commit) that are:
    - NOT in fallback mode
    - Pass validation
    - Do NOT violate lane_change_permission
    
    Note: the adapter already blocks permission violations via contract enforcement,
    so this is a secondary check on the artifact record itself.
    """
    lc_intents = [
        r for r in intents
        if r.get("tactical_intent") in LC_INTENTS
    ]
    if not lc_intents:
        return None
    valid_lc = sum(
        1 for r in lc_intents
        if r.get("validation_status") == "valid" and not r.get("fallback_to_baseline")
    )
    return _safe_div(float(valid_lc), float(len(lc_intents)))


def agent_forbidden_intent_count(intents: list[dict[str, Any]]) -> int:
    """
    Count of agent output records containing a forbidden field or invalid intent.
    This should always be 0 if the adapter contract enforcement is working.
    """
    forbidden_statuses = {"invalid_intent", "forbidden_field", "malformed"}
    return int(sum(1 for r in intents if r.get("validation_status") in forbidden_statuses))


def agent_shadow_artifact_completeness(
    intents: list[dict[str, Any]],
    summary: dict[str, Any],
    comparison_path: Path | None,
    disagreement_path: Path | None,
) -> float | None:
    """
    Checks completeness of all 4 required Stage 7 shadow artifacts:
    1. agent_shadow_intent.jsonl (non-empty)
    2. agent_shadow_summary.json (non-empty)
    3. agent_baseline_comparison.json (exists)
    4. agent_disagreement_events.jsonl (exists)
    Returns fraction of the 4 artifacts that are present and non-empty.
    """
    score = 0.0
    if intents:
        score += 1.0
    if summary:
        score += 1.0
    if comparison_path and comparison_path.exists():
        score += 1.0
    if disagreement_path and disagreement_path.exists():
        score += 1.0
    return score / 4.0


# ── Summary builder ────────────────────────────────────────────────────────────

def compute_agent_shadow_metrics(
    *,
    intents: list[dict[str, Any]],
    summary: dict[str, Any],
    comparison_path: Path | None = None,
    disagreement_path: Path | None = None,
) -> dict[str, Any]:
    """Compute all 10 Stage 7 agent shadow metrics."""

    results = {
        "agent_contract_validity_rate": {
            "value": agent_contract_validity_rate(intents),
            "maturity": "hard_gate_ready",
            "gate_usage": "hard_gate",
            "threshold": {"min": 0.95},
            "trend_preference": "higher_better",
            "description": "Fraction of frames where agent output passes full contract validation",
        },
        "agent_timeout_rate": {
            "value": agent_timeout_rate(intents),
            "maturity": "hard_gate_ready",
            "gate_usage": "hard_gate",
            "threshold": {"max": 0.05},
            "trend_preference": "lower_better",
            "description": "Fraction of frames where agent timed out",
        },
        "agent_fallback_to_baseline_rate": {
            "value": agent_fallback_to_baseline_rate(intents),
            "maturity": "hard_gate_ready",
            "gate_usage": "hard_gate",
            "threshold": {"max": 0.10},
            "trend_preference": "lower_better",
            "description": "Fraction of frames where agent fell back to baseline (timeout or invalid)",
        },
        "agent_baseline_disagreement_rate": {
            "value": agent_baseline_disagreement_rate(intents),
            "maturity": "soft_guardrail",
            "gate_usage": "soft_guardrail",
            "threshold": {"max": 0.40},
            "trend_preference": "lower_better",
            "description": "Fraction of valid frames where agent disagrees with baseline tactical intent",
        },
        "agent_useful_disagreement_rate": {
            "value": agent_useful_disagreement_rate(intents),
            "maturity": "soft_guardrail",
            "gate_usage": "soft_guardrail",
            "threshold": {"min": 0.30},
            "trend_preference": "higher_better",
            "description": "Among disagreements, fraction that are structurally useful (e.g. caution vs. risky LC)",
        },
        "agent_route_alignment_rate": {
            "value": agent_route_alignment_rate(intents),
            "maturity": "soft_guardrail",
            "gate_usage": "soft_guardrail",
            "threshold": {"min": 0.70},
            "trend_preference": "higher_better",
            "description": "Fraction of valid frames where agent target_lane aligns with route preferred_lane",
        },
        "agent_stop_context_correctness_rate": {
            "value": agent_stop_context_correctness_rate(intents),
            "maturity": "diagnostic_only",
            "gate_usage": "diagnostic_only",
            "threshold": None,
            "trend_preference": "higher_better",
            "description": "Fraction of stop-intent frames where agent consumed stop context",
        },
        "agent_lane_change_intent_validity_rate": {
            "value": agent_lane_change_intent_validity_rate(intents),
            "maturity": "hard_gate_ready",
            "gate_usage": "hard_gate",
            "threshold": {"min": 1.0},
            "trend_preference": "higher_better",
            "description": "Fraction of agent LC intents that pass validation and respect lane_change_permission",
        },
        "agent_forbidden_intent_count": {
            "value": agent_forbidden_intent_count(intents),
            "maturity": "hard_gate_ready",
            "gate_usage": "hard_gate",
            "threshold": {"max": 0},
            "trend_preference": "lower_better",
            "description": "Count of records with forbidden_field or invalid_intent — must be 0",
        },
        "agent_shadow_artifact_completeness": {
            "value": agent_shadow_artifact_completeness(intents, summary, comparison_path, disagreement_path),
            "maturity": "hard_gate_ready",
            "gate_usage": "hard_gate",
            "threshold": {"min": 1.0},
            "trend_preference": "higher_better",
            "description": "Fraction of 4 required Stage 7 shadow artifacts that are present",
        },
    }

    # Add threshold pass/fail
    for metric_name, result in results.items():
        threshold = result.get("threshold")
        value = result.get("value")
        if threshold is None or value is None:
            result["threshold_passed"] = None
            result["threshold_reason"] = "no_threshold_or_unavailable"
        else:
            passed, reason = _check_threshold(value, threshold)
            result["threshold_passed"] = passed
            result["threshold_reason"] = reason

    return results


def _check_threshold(value: Any, threshold: dict[str, Any]) -> tuple[bool, str]:
    if "min" in threshold and float(value) < float(threshold["min"]):
        return False, f"value {value} < min {threshold['min']}"
    if "max" in threshold and float(value) > float(threshold["max"]):
        return False, f"value {value} > max {threshold['max']}"
    return True, "ok"


def build_agent_shadow_summary(
    *,
    case_id: str,
    intents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build agent_shadow_summary.json from collected intent records."""
    if not intents:
        return {
            "case_id": case_id,
            "total_frames": 0,
            "schema_version": "agent_shadow_summary_v1",
            "notes": ["no_agent_intent_records"],
        }

    total = len(intents)
    agreement_count = sum(1 for r in intents if bool(r.get("agrees_with_baseline", True)))
    valid_count = sum(1 for r in intents if r.get("validation_status") == "valid")
    fallback_count = sum(1 for r in intents if bool(r.get("fallback_to_baseline", False)))
    timeout_count = sum(1 for r in intents if bool(r.get("timeout_flag", False)))
    intent_counts = Counter(str(r.get("tactical_intent", "unknown")) for r in intents)

    latencies = [
        float((r.get("provenance") or {}).get("call_latency_ms", 0.0))
        for r in intents
        if (r.get("provenance") or {}).get("call_latency_ms") is not None
    ]

    return {
        "case_id": case_id,
        "schema_version": "agent_shadow_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "total_frames": total,
        "valid_frames": valid_count,
        "fallback_frames": fallback_count,
        "timeout_frames": timeout_count,
        "agreement_frames": agreement_count,
        "disagreement_frames": total - agreement_count,
        "agent_contract_validity_rate": _safe_div(float(valid_count), float(total)),
        "agent_timeout_rate": _safe_div(float(timeout_count), float(total)),
        "agent_fallback_to_baseline_rate": _safe_div(float(fallback_count), float(total)),
        "agent_baseline_disagreement_rate": _safe_div(float(total - agreement_count), float(total)),
        "intent_counts": dict(intent_counts),
        "model_id": str((intents[0].get("provenance") or {}).get("model_id", "unknown")),
        "mean_call_latency_ms": (float(sum(latencies) / len(latencies)) if latencies else None),
        "max_call_latency_ms": (float(max(latencies)) if latencies else None),
    }


def build_agent_baseline_comparison(
    *,
    case_id: str,
    intents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build agent_baseline_comparison.json aggregating agreement/disagreement stats."""
    if not intents:
        return {"case_id": case_id, "total_frames": 0, "notes": ["no_intents"]}

    total = len(intents)
    by_baseline: dict[str, Counter] = {}
    for r in intents:
        b = str(r.get("baseline_intent", "unknown"))
        a = str(r.get("tactical_intent", "unknown"))
        if b not in by_baseline:
            by_baseline[b] = Counter()
        by_baseline[b][a] += 1

    disagreement_types: Counter = Counter()
    for r in intents:
        if not r.get("agrees_with_baseline", True) and r.get("validation_status") == "valid":
            disagree_key = f"{r.get('baseline_intent')} → {r.get('tactical_intent')}"
            disagreement_types[disagree_key] += 1

    useful_disagreements = sum(
        1 for r in intents
        if not r.get("agrees_with_baseline", True)
        and bool(r.get("disagreement_useful", False))
    )

    return {
        "case_id": case_id,
        "schema_version": "agent_baseline_comparison_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "total_frames": total,
        "agreement_frames": sum(1 for r in intents if r.get("agrees_with_baseline", True)),
        "disagreement_frames": sum(1 for r in intents if not r.get("agrees_with_baseline", True)),
        "useful_disagreement_frames": useful_disagreements,
        "baseline_to_agent_intent_matrix": {
            baseline: dict(agent_counts)
            for baseline, agent_counts in by_baseline.items()
        },
        "disagreement_type_counts": dict(disagreement_types),
    }
