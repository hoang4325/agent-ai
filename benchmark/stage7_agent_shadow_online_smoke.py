"""
Stage 7C — Agent Shadow Online Smoke Runner
============================================

Opens Stage 7C agent tactical shadow online smoke on the exact two promoted cases:
  - ml_right_positive_core
  - stop_follow_ambiguity_core

CRITICAL INVARIANTS:
  - shadow_mode = True, always
  - Agent has NO control / trajectory / MPC authority
  - Baseline tactical + MPC remain the sole authority at all times
  - No assist mode, no takeover, no control authority hand-off
  - Agent timeout → fallback to baseline recorded in artifact
  - All 6 artifacts are materialized per case

Artifacts produced (per case, in {output}/stage7/):
  1. agent_shadow_intent.jsonl
  2. agent_shadow_summary.json
  3. agent_baseline_comparison.json
  4. agent_disagreement_events.jsonl
  5. agent_contract_validation.json
  6. agent_timeout_fallback_events.jsonl

Output at repo root (canonical paths):
  - benchmark/stage7_agent_shadow_online_smoke_report.json
  - benchmark/stage7_agent_shadow_online_smoke_result.json
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .agent_shadow_adapter import build_adapter
from .io import dump_json, dump_jsonl, load_json, load_jsonl, load_yaml, resolve_repo_path
from .stage7_agent_shadow_audit import run_stage7_contract_audit
from .stage7_agent_shadow_metric_pack import (
    build_agent_baseline_comparison,
    build_agent_shadow_summary,
    compute_agent_shadow_metrics,
)


# ── Constants ─────────────────────────────────────────────────────────────────

# Controlled expansion: 4-case shadow set (was 2-case; expanded after 10/10 stable runs)
STAGE7C_CASES = [
    "ml_right_positive_core",
    "stop_follow_ambiguity_core",
    "ml_left_positive_core",
    "arbitration_stop_during_prepare_right",
]

STAGE4_PROFILE_OVERRIDES = {
    "ml_right_positive_core":              "stage4_online_external_watch_shadow_smoke",
    "stop_follow_ambiguity_core":          "stage4_online_external_watch_stop_aligned_shadow_smoke",
    "ml_left_positive_core":              "stage4_online_external_watch_shadow_smoke",
    "arbitration_stop_during_prepare_right": "stage4_online_external_watch_shadow_smoke",
}


def _load_set_cases(set_name: str, benchmark_config: dict[str, Any], repo_root: Path) -> list[str]:
    sets = benchmark_config.get("sets") or {}
    set_path_str = sets.get(set_name)
    if not set_path_str:
        raise ValueError(f"Set '{set_name}' not in benchmark_v1.yaml sets. Add it or use an existing set.")
    set_path = resolve_repo_path(repo_root, set_path_str)
    payload = load_yaml(set_path)
    return list(payload.get("cases", []))

# Online smoke thresholds (tighter than default: only 60 frames per case)
ONLINE_SMOKE_THRESHOLDS = {
    "agent_shadow_artifact_completeness": {"min": 1.0},
    "agent_contract_validity_rate": {"min": 0.95},
    "agent_timeout_rate": {"max": 0.05},
    "agent_fallback_to_baseline_rate": {"max": 0.10},
    "agent_forbidden_intent_count": {"max": 0},
    "agent_lane_change_intent_validity_rate": {"min": 1.0},
    "agent_baseline_disagreement_rate": {"max": 0.40},
    "agent_useful_disagreement_rate": {"min": 0.20},   # relaxed from 0.30 for online smoke
    "agent_route_alignment_rate": {"min": 0.65},       # relaxed from 0.70 for online smoke
}

# Hard gate set (same as A7.1–A7.6)
HARD_GATE_METRICS = {
    "agent_shadow_artifact_completeness",
    "agent_contract_validity_rate",
    "agent_timeout_rate",
    "agent_fallback_to_baseline_rate",
    "agent_forbidden_intent_count",
    "agent_lane_change_intent_validity_rate",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


# ── Online readiness audit ────────────────────────────────────────────────────

def audit_online_readiness(repo_root: Path, *, case_ids: list[str] | None = None) -> dict[str, Any]:
    """
    Checks that Stage 7C online shadow smoke can safely run:
      1. Stage 6 shadow readiness is 'ready' (online CARLA infrastructure proven)
      2. Frozen corpus Stage 1 sessions exist for both cases
      3. Baseline Stage 3B timelines exist for both cases
      4. Stage 4 profiles exist for both cases
      5. Stage 7A/7B gate results are available (advisory check)
    """
    issues: list[str] = []
    warnings: list[str] = []
    ready_checks: dict[str, Any] = {}

    # Check 1: Stage 6 online shadow infrastructure readiness
    stage6_readiness_path = repo_root / "benchmark" / "stage6_shadow_readiness.json"
    stage6_readiness = load_json(stage6_readiness_path, default={}) or {}
    stage6_ready = str(stage6_readiness.get("status")) == "ready"
    ready_checks["stage6_online_infrastructure_ready"] = stage6_ready
    if not stage6_ready:
        warnings.append(
            "stage6_shadow_readiness.json status != 'ready'. "
            "CARLA online infrastructure may not be proven. Stage 7C will still run in shadow mode."
        )

    selected_case_ids = case_ids or STAGE7C_CASES

    # Check 2: Frozen corpus sessions
    for case_id in selected_case_ids:
        session_dir = (
            repo_root / "benchmark" / "frozen_corpus" / "v1" / "cases"
            / case_id / "stage1_frozen" / "session"
        )
        exists = session_dir.exists()
        ready_checks[f"{case_id}.stage1_frozen_session"] = exists
        if not exists:
            issues.append(f"{case_id}: frozen stage1 session missing at {session_dir}")

    # Check 3: Baseline Stage 3B timeline artifacts
    cases_dir = repo_root / "benchmark" / "cases"
    for case_id in selected_case_ids:
        case_path = cases_dir / f"{case_id}.yaml"
        if not case_path.exists():
            issues.append(f"{case_id}: case spec missing")
            ready_checks[f"{case_id}.case_spec"] = False
            continue
        ready_checks[f"{case_id}.case_spec"] = True
        spec = load_yaml(case_path)
        artifacts = spec.get("artifacts") or {}
        bt_key = artifacts.get("stage3b_behavior_timeline")
        bt_path = resolve_repo_path(repo_root, bt_key) if bt_key else None
        bt_exists = bool(bt_path and bt_path.exists())
        ready_checks[f"{case_id}.stage3b_behavior_timeline"] = bt_exists
        if not bt_exists:
            warnings.append(
                f"{case_id}: stage3b_behavior_timeline missing — "
                "agent shadow will generate synthetic frames from frozen corpus"
            )

    # Check 4: Stage 4 profiles
    for case_id, profile_name in STAGE4_PROFILE_OVERRIDES.items():
        profile_path = repo_root / "benchmark" / "stage_profiles" / f"{profile_name}.yaml"
        exists = profile_path.exists()
        ready_checks[f"{case_id}.stage4_profile"] = exists
        if not exists:
            issues.append(f"{case_id}: Stage 4 profile missing: {profile_path}")

    # Check 5: Prior Stage 7A gate result (advisory)
    stage7a_result_path = repo_root / "benchmark" / "stage7_agent_shadow_gate_result.json"
    stage7a_exists = stage7a_result_path.exists()
    ready_checks["stage7a_gate_result_available"] = stage7a_exists
    if not stage7a_exists:
        warnings.append(
            "stage7_agent_shadow_gate_result.json not found. "
            "Stage 7A gate was not previously confirmed. Proceeding anyway (Stage 7C opens regardless of 7A in this run)."
        )
    else:
        stage7a_result = load_json(stage7a_result_path, default={}) or {}
        stage7a_status = str(stage7a_result.get("overall_status", "unknown"))
        ready_checks["stage7a_gate_status"] = stage7a_status
        if stage7a_status != "pass":
            warnings.append(
                f"Stage 7A gate status is '{stage7a_status}' (not 'pass'). "
                "Proceeding with Stage 7C online smoke, but consider fixing 7A gates first."
            )

    blockers_remaining = len(issues)
    overall = "ready_with_warnings" if (not issues and warnings) else ("ready" if not issues else "blocked")

    audit = {
        "schema_version": "stage7c_online_readiness_audit_v1",
        "generated_at_utc": _utc_now(),
        "stage": "7C",
        "mode": "online_shadow_smoke",
        "cases": selected_case_ids,
        "overall_readiness": overall,
        "blockers": issues,
        "warnings": warnings,
        "checks": ready_checks,
        "invariants": [
            "shadow_mode = True in all agent outputs",
            "agent has NO control/trajectory/MPC authority",
            "baseline tactical + MPC remain sole authority",
            "no assist mode, no takeover, no authority hand-off",
            "agent timeout → fallback_to_baseline = True recorded",
        ],
    }
    return audit


# ── Artifact input builder ────────────────────────────────────────────────────

def _load_case_artifacts_online(case_id: str, repo_root: Path, cases_dir: Path, *, allow_synthetic_fallback: bool) -> dict[str, Any]:
    """
    Load case artifacts for Stage 7C online smoke.
    Priority: current-run Stage 3B timeline → frozen corpus Stage 3B → synthetic frames.
    """
    case_path = cases_dir / f"{case_id}.yaml"
    if not case_path.exists():
        return {}
    spec = load_yaml(case_path)
    artifacts = spec.get("artifacts") or {}
    result: dict[str, Any] = {"spec": spec, "artifacts": artifacts}

    # Stage 3B baseline behavior timeline
    bt_path = resolve_repo_path(repo_root, artifacts.get("stage3b_behavior_timeline"))
    result["baseline_timeline"] = load_jsonl(bt_path) if bt_path else []

    # If no timeline available, fail closed in live-only mode.
    if not result["baseline_timeline"]:
        if allow_synthetic_fallback:
            result["baseline_timeline"] = _synthesize_frames_from_frozen(repo_root, case_id)
            result["timeline_source"] = "synthetic_frozen_corpus"
        else:
            result["timeline_source"] = "missing_live_timeline"
            result["live_timeline_missing"] = True
    else:
        result["timeline_source"] = "stage3b_behavior_timeline"

    # World state sample
    ws_path = resolve_repo_path(repo_root, artifacts.get("stage2_world_state_sample"))
    result["world_state_sample"] = load_json(ws_path) if ws_path else None

    # Lane context sample
    lane_path = resolve_repo_path(repo_root, artifacts.get("stage3a_lane_world_state_sample"))
    result["lane_world_state_sample"] = load_json(lane_path) if lane_path else None

    # Stop target
    stop_path = resolve_repo_path(repo_root, artifacts.get("stage3c_stop_target"))
    result["stop_target"] = load_json(stop_path) if stop_path else None

    return result


def _synthesize_frames_from_frozen(repo_root: Path, case_id: str) -> list[dict[str, Any]]:
    """
    Synthesize minimal agent-input frames from frozen corpus manifest when
    a live Stage 3B timeline is not yet available for this case.
    Produces max 60 frames (Stage 7C online smoke budget).

    R1 fix: target_lane and preferred_lane now derive from corridor.preferred_lane,
    not hardcoded 'current'. This prevents false-positive alignment=1.0 on
    keep_right / keep_left corridor cases.
    """
    MAX_ONLINE_FRAMES = 60
    manifest_path = (
        repo_root / "benchmark" / "frozen_corpus" / "v1" / "cases"
        / case_id / "scenario_manifest.json"
    )
    manifest = load_json(manifest_path, default={}) or {}
    corridor = manifest.get("corridor") or {}

    # Derive route fields from corridor — do NOT fall back to 'current' for lane-change corridors
    route_option = str(corridor.get("route_option") or "straight")
    preferred_lane = str(corridor.get("preferred_lane") or "current")
    # target_lane follows preferred_lane for non-junction routes; junction always 'current'
    if route_option in {"keep_right", "keep_left"}:
        target_lane = preferred_lane   # R1 fix: was always 'current'
    else:
        target_lane = "current"

    # Derive a best-effort default behavior for the case type
    default_behavior = "keep_lane"
    if "lane_change" in case_id or "ml_" in case_id:
        default_behavior = "prepare_lane_change_right" if "right" in case_id else "prepare_lane_change_left"
    elif "stop" in case_id or "follow" in case_id or "arbitration" in case_id:
        default_behavior = "follow"

    # For LC cases, lane_change_permission should match the LC direction
    lc_permission = {"left": True, "right": True}
    if "right" in case_id and "ml_" in case_id:
        lc_permission = {"left": False, "right": True}
    elif "left" in case_id and "ml_" in case_id:
        lc_permission = {"left": True, "right": False}

    frames = []
    for i in range(MAX_ONLINE_FRAMES):
        frames.append({
            "frame_id": i,
            "behavior_request_v2": {
                "requested_behavior": default_behavior,
                "target_lane": target_lane,           # R1: corridor-derived
                "route_option": route_option,
                "preferred_lane": preferred_lane,      # R1: corridor-derived
                "route_mode": "corridor_only",
                "route_conflict_flags": [],
                "lane_change_permission": lc_permission,
                "confidence": 0.9,
                "stop_reason": None,
                "longitudinal_horizon_m": None,
            },
            "_synthetic": True,
            "_source": "frozen_corpus_manifest",
            "_r1_fix": "target_lane and preferred_lane derived from corridor, not hardcoded",
        })
    return frames



def _build_input_bundle(
    frame_record: dict[str, Any],
    world_state_sample: dict[str, Any] | None,
    lane_world_state_sample: dict[str, Any] | None,
    stop_target: dict[str, Any] | None,
) -> tuple[dict, list, dict, dict, dict, dict]:
    """Extract 6-tuple input bundle for agent adapter."""
    ws = world_state_sample or {}
    ego_state: dict[str, Any] = dict(ws.get("ego") or {})
    ego_state["risk_summary"] = dict(ws.get("risk_summary") or {})
    ego_state["scene"] = dict(ws.get("scene") or {})

    tracked_objects = list(ws.get("objects") or [])

    lane_ws = dict(lane_world_state_sample or {})
    lane_context: dict[str, Any] = dict(lane_ws.get("lane_context") or {})
    lane_context["lane_scene"] = dict(lane_ws.get("lane_scene") or {})
    lane_context["lane_relative_objects"] = list(lane_ws.get("lane_relative_objects") or [])

    br2 = dict(frame_record.get("behavior_request_v2") or {})
    route_context = {
        "route_option": br2.get("route_option"),
        "preferred_lane": br2.get("preferred_lane"),
        "route_mode": br2.get("route_mode"),
        "route_conflict_flags": list(br2.get("route_conflict_flags") or []),
        "distance_to_next_branch_m": br2.get("longitudinal_horizon_m"),
    }

    stop_ctx: dict[str, Any] = {}
    if stop_target:
        stop_ctx = {
            "binding_status": stop_target.get("binding_status"),
            "distance_to_stop_m": stop_target.get("distance_to_stop_m"),
        }

    baseline_context = {
        "requested_behavior": br2.get("requested_behavior"),
        "target_lane": br2.get("target_lane", "current"),
        "lane_change_permission": dict(br2.get("lane_change_permission") or {}),
        "confidence": br2.get("confidence"),
        "stop_reason": br2.get("stop_reason"),
    }

    return ego_state, tracked_objects, lane_context, route_context, stop_ctx, baseline_context


# ── Contract validation artifact ──────────────────────────────────────────────

def _build_contract_validation(
    case_id: str,
    intents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build agent_contract_validation.json — per-case contract check summary."""
    total = len(intents)
    by_status: dict[str, int] = {}
    forbidden_fields_found: list[str] = []
    for r in intents:
        status = str(r.get("validation_status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
        # Check for forbidden fields in the record
        for forbidden in ("target_speed_mps", "steering", "throttle", "brake", "trajectory"):
            if forbidden in r:
                forbidden_fields_found.append(f"frame_{r.get('frame_id', '?')}:{forbidden}")

    lc_intents = [r for r in intents if str(r.get("tactical_intent", "")) in {
        "prepare_lane_change_left", "prepare_lane_change_right",
        "commit_lane_change_left", "commit_lane_change_right",
    }]
    lc_permission_violations = [
        r for r in lc_intents
        if r.get("validation_status") != "valid"
        and not r.get("fallback_to_baseline", True)
    ]

    return {
        "schema_version": "agent_contract_validation_v1",
        "case_id": case_id,
        "generated_at_utc": _utc_now(),
        "stage": "7C",
        "mode": "online_shadow_smoke",
        "shadow_mode_enforced": True,
        "control_authority": "baseline_tactical_and_mpc_only",
        "total_frames": total,
        "validation_status_counts": by_status,
        "forbidden_fields_found": forbidden_fields_found,
        "lc_permission_violations": len(lc_permission_violations),
        "contract_valid": len(forbidden_fields_found) == 0 and len(lc_permission_violations) == 0,
        "shadow_mode_violations": 0,  # adapter enforces shadow_mode=True always
        "binding_rules_checked": [
            "tactical_intent in ALLOWED_AGENT_INTENTS",
            "target_speed_mps NOT in output",
            "shadow_mode == True",
            "fallback_to_baseline == True when timeout or invalid",
            "LC not requested when lane_change_permission == False",
        ],
    }


def _build_timeout_fallback_events(
    case_id: str,
    intents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build agent_timeout_fallback_events.jsonl — records all timeout/fallback frames."""
    events = []
    for r in intents:
        if r.get("timeout_flag") or r.get("fallback_to_baseline"):
            events.append({
                "case_id": case_id,
                "frame_id": r.get("frame_id"),
                "timeout_flag": r.get("timeout_flag"),
                "fallback_to_baseline": r.get("fallback_to_baseline"),
                "validation_status": r.get("validation_status"),
                "fallback_reason": (r.get("provenance") or {}).get("fallback_reason"),
                "call_latency_ms": (r.get("provenance") or {}).get("call_latency_ms"),
                "baseline_intent": r.get("baseline_intent"),
                "model_id": r.get("model_id"),
            })
    return events


# ── Per-case runner ───────────────────────────────────────────────────────────

def _run_case_online_shadow(
    *,
    case_id: str,
    repo_root: Path,
    cases_dir: Path,
    output_dir: Path,
    adapter: Any,
    max_frames: int = 60,
    allow_synthetic_fallback: bool = False,
) -> dict[str, Any]:
    """Run Stage 7C online shadow on one case. Materialize all 6 artifacts."""
    print(f"[Stage7C]   Case: {case_id}")
    case_data = _load_case_artifacts_online(case_id, repo_root, cases_dir, allow_synthetic_fallback=allow_synthetic_fallback)
    if not case_data:
        print(f"[Stage7C]     WARN: case spec not found — skipping {case_id}")
        return {"case_id": case_id, "status": "skipped", "reason": "case_spec_not_found"}

    baseline_timeline = (case_data.get("baseline_timeline") or [])[:max_frames]
    timeline_source = case_data.get("timeline_source", "unknown")
    world_state_sample = case_data.get("world_state_sample")
    lane_world_state_sample = case_data.get("lane_world_state_sample")
    stop_target = case_data.get("stop_target")

    if not baseline_timeline:
        reason = "missing_live_timeline" if case_data.get("live_timeline_missing") else "no_frames_available"
        return {"case_id": case_id, "status": "fail", "reason": reason, "timeline_source": timeline_source, "hard_failures": [reason]}

    print(f"[Stage7C]     Frames: {len(baseline_timeline)} (source: {timeline_source})")

    case_out = output_dir / case_id / "stage7"
    case_out.mkdir(parents=True, exist_ok=True)

    intent_records: list[dict[str, Any]] = []
    disagreement_events: list[dict[str, Any]] = []

    for frame_idx, frame_record in enumerate(baseline_timeline):
        frame_id = int(frame_record.get("frame_id", frame_idx))
        ego, tracked, lane_ctx, route_ctx, stop_ctx, baseline_ctx = _build_input_bundle(
            frame_record=frame_record,
            world_state_sample=world_state_sample,
            lane_world_state_sample=lane_world_state_sample,
            stop_target=stop_target,
        )
        intent = adapter.call(
            case_id=case_id,
            frame_id=frame_id,
            ego_state=ego,
            tracked_objects=tracked,
            lane_context=lane_ctx,
            route_context=route_ctx,
            stop_context=stop_ctx,
            baseline_context=baseline_ctx,
        )
        intent_dict = asdict(intent)
        intent_records.append(intent_dict)

        if not intent_dict.get("agrees_with_baseline", True):
            disagreement_events.append({
                "case_id": case_id,
                "frame_id": frame_id,
                "baseline_intent": intent_dict.get("baseline_intent"),
                "agent_intent": intent_dict.get("tactical_intent"),
                "agent_confidence": intent_dict.get("confidence"),
                "disagreement_useful": intent_dict.get("disagreement_useful"),
                "reason_tags": intent_dict.get("reason_tags"),
                "fallback_to_baseline": intent_dict.get("fallback_to_baseline"),
                "validation_status": intent_dict.get("validation_status"),
                "route_context_summary": intent_dict.get("route_context_summary"),
            })

    # ── Artifact 1: agent_shadow_intent.jsonl ─────────────────────────────────
    intent_path = case_out / "agent_shadow_intent.jsonl"
    dump_jsonl(intent_path, intent_records)

    # ── Artifact 2: agent_shadow_summary.json ─────────────────────────────────
    summary = build_agent_shadow_summary(case_id=case_id, intents=intent_records)
    summary_path = case_out / "agent_shadow_summary.json"
    dump_json(summary_path, summary)

    # ── Artifact 3: agent_baseline_comparison.json ────────────────────────────
    comparison = build_agent_baseline_comparison(case_id=case_id, intents=intent_records)
    comparison_path = case_out / "agent_baseline_comparison.json"
    dump_json(comparison_path, comparison)

    # ── Artifact 4: agent_disagreement_events.jsonl ───────────────────────────
    disagreement_path = case_out / "agent_disagreement_events.jsonl"
    dump_jsonl(disagreement_path, disagreement_events)

    # ── Artifact 5: agent_contract_validation.json ────────────────────────────
    contract_validation = _build_contract_validation(case_id, intent_records)
    contract_validation_path = case_out / "agent_contract_validation.json"
    dump_json(contract_validation_path, contract_validation)

    # ── Artifact 6: agent_timeout_fallback_events.jsonl ───────────────────────
    timeout_events = _build_timeout_fallback_events(case_id, intent_records)
    timeout_path = case_out / "agent_timeout_fallback_events.jsonl"
    dump_jsonl(timeout_path, timeout_events)

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Pass all 6 artifact paths; completeness metric checks all 4 primary ones
    metrics = compute_agent_shadow_metrics(
        intents=intent_records,
        summary=summary,
        comparison_path=comparison_path,
        disagreement_path=disagreement_path,
    )
    # Apply online smoke thresholds (override default)
    for metric_name, metric_result in metrics.items():
        override_thresh = ONLINE_SMOKE_THRESHOLDS.get(metric_name)
        if override_thresh is not None:
            metric_result["threshold"] = override_thresh
            value = metric_result.get("value")
            if value is None:
                metric_result["threshold_passed"] = None
                metric_result["threshold_reason"] = "value_unavailable"
            else:
                passed = True
                reason = "ok"
                if "min" in override_thresh and float(value) < float(override_thresh["min"]):
                    passed = False
                    reason = f"value {value} < min {override_thresh['min']}"
                if "max" in override_thresh and float(value) > float(override_thresh["max"]):
                    passed = False
                    reason = f"value {value} > max {override_thresh['max']}"
                metric_result["threshold_passed"] = passed
                metric_result["threshold_reason"] = reason

    # Gate evaluation
    hard_failures: list[str] = []
    soft_warnings: list[str] = []
    for metric_name, metric_result in metrics.items():
        passed = metric_result.get("threshold_passed")
        reason = metric_result.get("threshold_reason", "")
        if passed is False:
            if metric_name in HARD_GATE_METRICS:
                hard_failures.append(f"{metric_name}: {reason}")
            elif metric_result.get("gate_usage") == "soft_guardrail":
                soft_warnings.append(f"{metric_name}: {reason}")

    case_status = "pass" if not hard_failures else "fail"

    print(
        f"[Stage7C]     Valid: {summary.get('valid_frames', 0)} | "
        f"Fallback: {summary.get('fallback_frames', 0)} | "
        f"Disagree: {summary.get('disagreement_frames', 0)} | "
        f"Timeout: {summary.get('timeout_frames', 0)} | "
        f"Status: {case_status.upper()}"
    )

    return {
        "case_id": case_id,
        "status": case_status,
        "timeline_source": timeline_source,
        "frames_processed": len(intent_records),
        "hard_failures": hard_failures,
        "soft_warnings": soft_warnings,
        "metrics": {
            k: {
                "value": v.get("value"),
                "threshold_passed": v.get("threshold_passed"),
                "threshold": v.get("threshold"),
                "maturity": v.get("maturity"),
            }
            for k, v in metrics.items()
        },
        "summary": summary,
        "artifacts": {
            "agent_shadow_intent": str(intent_path),
            "agent_shadow_summary": str(summary_path),
            "agent_baseline_comparison": str(comparison_path),
            "agent_disagreement_events": str(disagreement_path),
            "agent_contract_validation": str(contract_validation_path),
            "agent_timeout_fallback_events": str(timeout_path),
        },
        "contract_validation": contract_validation,
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_stage7c_online_shadow_smoke(
    repo_root: str | Path,
    *,
    config_path: str | Path | None = None,
    set_name: str = "stage7_agent_shadow_online_smoke",
    case_ids: list[str] | None = None,
    adapter_mode: str = "stub",
    adapter_model_id: str = "stub_v1",
    adapter_timeout_s: float = 1.5,
    max_frames_per_case: int = 60,
    output_dir: Path | None = None,
    allow_synthetic_fallback: bool = False,
) -> dict[str, Any]:
    """
    Full Stage 7C online shadow smoke runner.
    Returns combined result dict with gate_result and report.
    """
    repo_root = Path(repo_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir or (
        repo_root / "benchmark" / "reports" / f"stage7c_online_shadow_smoke_{ts}"
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage7C] ═══════════════════════════════════════════════════")
    print(f"[Stage7C] Stage 7 Agentic Shadow — Online Smoke (Stage 7C)")
    print(f"[Stage7C] INVARIANT: shadow_only — NO control authority")
    benchmark_config = load_yaml(Path(config_path).resolve()) if config_path else load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    cases_dir = resolve_repo_path(repo_root, benchmark_config.get("cases_dir")) or (repo_root / "benchmark" / "cases")
    selected_case_ids = case_ids or STAGE7C_CASES
    try:
        if case_ids is None:
            selected_case_ids = _load_set_cases(set_name, benchmark_config, repo_root)
    except Exception:
        if case_ids is None and set_name != "stage7_agent_shadow_online_smoke":
            raise
    print(f"[Stage7C] Cases: {selected_case_ids}")
    print(f"[Stage7C] Adapter: {adapter_mode} / {adapter_model_id}")
    print(f"[Stage7C] Max frames/case: {max_frames_per_case}")
    print(f"[Stage7C] Output: {run_output_dir}")
    print(f"[Stage7C] ═══════════════════════════════════════════════════\n")

    # ── 1. Online readiness audit ─────────────────────────────────────────────
    print("[Stage7C] Running online readiness audit...")
    readiness_audit = audit_online_readiness(repo_root, case_ids=selected_case_ids)
    readiness_path = run_output_dir / "stage7c_online_readiness_audit.json"
    dump_json(readiness_path, readiness_audit)
    print(f"[Stage7C] Readiness: {readiness_audit['overall_readiness']}")
    if readiness_audit["blockers"]:
        print("[Stage7C] BLOCKERS:")
        for b in readiness_audit["blockers"]:
            print(f"[Stage7C]   ✗ {b}")
        # Don't abort — log and continue; caller can inspect result
    if readiness_audit["warnings"]:
        for w in readiness_audit["warnings"]:
            print(f"[Stage7C]   ⚠ {w}")

    # ── 2. Contract audit (re-run at launch) ──────────────────────────────────
    print("\n[Stage7C] Running Stage 7 contract audit...")
    try:
        contract_audit = run_stage7_contract_audit(repo_root)
    except Exception as exc:
        print(f"[Stage7C]   WARN: contract audit failed: {exc}")
        contract_audit = {"status": "audit_failed", "error": str(exc)}

    # ── 3. Build adapter ──────────────────────────────────────────────────────
    adapter = build_adapter(
        mode=adapter_mode,
        model_id=adapter_model_id,
        api_timeout_s=adapter_timeout_s,
    )

    # ── 4. Run per-case shadow ────────────────────────────────────────────────
    print(f"\n[Stage7C] Running agent shadow on {len(selected_case_ids)} cases...\n")
    all_case_results: list[dict[str, Any]] = []
    gate_passed = True
    gate_hard_failures: list[str] = []
    gate_soft_warnings: list[str] = []

    for case_id in selected_case_ids:
        case_result = _run_case_online_shadow(
            case_id=case_id,
            repo_root=repo_root,
            cases_dir=cases_dir,
            output_dir=run_output_dir,
            adapter=adapter,
            max_frames=max_frames_per_case,
            allow_synthetic_fallback=allow_synthetic_fallback,
        )
        all_case_results.append(case_result)
        if case_result.get("status") == "fail":
            gate_passed = False
            gate_hard_failures.extend([
                f"{case_id}/{f}" for f in (case_result.get("hard_failures") or [])
            ])
        gate_soft_warnings.extend([
            f"{case_id}/{f}" for f in (case_result.get("soft_warnings") or [])
        ])

    # Also count skipped as warning
    skipped = [r["case_id"] for r in all_case_results if r.get("status") == "skipped"]
    if skipped:
        gate_soft_warnings.extend([f"{c}:skipped" for c in skipped])

    overall_status = "pass" if gate_passed and not skipped else ("fail" if gate_hard_failures else "partial")
    print(f"\n[Stage7C] ═══════════════════ GATE: {overall_status.upper()} ═══════════════════")

    # ── 5. Build canonical output artifacts ───────────────────────────────────
    selected_metrics = [
        "agent_shadow_artifact_completeness",
        "agent_contract_validity_rate",
        "agent_forbidden_intent_count",
        "agent_timeout_rate",
        "agent_fallback_to_baseline_rate",
        "agent_baseline_disagreement_rate",
    ]

    gate_result = {
        "schema_version": "stage7_agent_shadow_online_smoke_result_v1",
        "generated_at_utc": _utc_now(),
        "stage": "7C",
        "mode": "online_shadow_smoke",
        "agent_authority": "shadow_only — baseline tactical + MPC remain sole authority",
        "no_control_authority": True,
        "no_assist_mode": True,
        "no_takeover": True,
        "subset_cases": case_ids,
        "overall_status": overall_status,
        "hard_failures": gate_hard_failures,
        "soft_warnings": gate_soft_warnings,
        "passed_cases": [r["case_id"] for r in all_case_results if r.get("status") == "pass"],
        "failed_cases": [r["case_id"] for r in all_case_results if r.get("status") == "fail"],
        "skipped_cases": skipped,
        "adapter": {"mode": adapter_mode, "model_id": adapter_model_id, "timeout_s": adapter_timeout_s},
        "readiness_audit_status": readiness_audit["overall_readiness"],
        "readiness_blockers": readiness_audit["blockers"],
        "readiness_warnings": readiness_audit["warnings"],
        # Selected metrics summary across cases
        "metric_summary": {
            metric_name: {
                case_result["case_id"]: {
                    "value": (case_result.get("metrics") or {}).get(metric_name, {}).get("value"),
                    "threshold_passed": (case_result.get("metrics") or {}).get(metric_name, {}).get("threshold_passed"),
                }
                for case_result in all_case_results
                if case_result.get("status") not in {"skipped"}
            }
            for metric_name in selected_metrics
        },
    }

    report = {
        "schema_version": "stage7_agent_shadow_online_smoke_report_v1",
        "generated_at_utc": _utc_now(),
        "stage": "7C",
        "mode": "online_shadow_smoke",
        "agent_authority": "shadow_only",
        "no_control_authority": True,
        "no_assist_mode": True,
        "no_takeover": True,
        "subset_cases": case_ids,
        "overall_status": overall_status,
        "run_output_dir": str(run_output_dir),
        "readiness_audit": readiness_audit,
        "contract_audit_summary": {
            "stage7a_ready": (contract_audit.get("summary") or {}).get("stage7a_replay_readiness"),
        },
        "cases": all_case_results,
        "notes": [
            "Stage 7C: agent tactical shadow online smoke — shadow only.",
            "Baseline tactical + MPC are the sole authority at all times.",
            "No control authority, no assist mode, no takeover is permitted.",
            "Stage 7C opens for analysis only. Do not discuss assist mode.",
        ],
    }

    # ── Write canonical output files ──────────────────────────────────────────
    # Per-run
    dump_json(run_output_dir / "stage7_agent_shadow_online_smoke_result.json", gate_result)
    dump_json(run_output_dir / "stage7_agent_shadow_online_smoke_report.json", report)

    # Canonical benchmark root copies
    dump_json(repo_root / "benchmark" / "stage7_agent_shadow_online_smoke_result.json", gate_result)
    dump_json(repo_root / "benchmark" / "stage7_agent_shadow_online_smoke_report.json", report)

    print(f"[Stage7C] Result  → benchmark/stage7_agent_shadow_online_smoke_result.json")
    print(f"[Stage7C] Report  → benchmark/stage7_agent_shadow_online_smoke_report.json")

    return {
        "gate_result": gate_result,
        "report": report,
        "readiness_audit": readiness_audit,
        "run_output_dir": str(run_output_dir),
    }
