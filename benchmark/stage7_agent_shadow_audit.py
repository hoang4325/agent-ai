"""
Stage 7 Agent Shadow — Contract Audit Engine
=============================================
PHASE A7.1

Audits the *existing* baseline tactical layer contracts (Stage 3B behavior_request_v2,
Stage 3A lane_context, Stage 3C execution) to identify exactly what the agent shadow
can read and what new contract fields it must emit.

Output: benchmark/stage7_agent_shadow_audit.json
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_json, load_jsonl, load_yaml, resolve_repo_path


# ── Canonical allowed agent tactical intents ─────────────────────────────────
ALLOWED_AGENT_INTENTS = {
    "keep_lane",
    "follow",
    "slow_down",
    "stop",
    "yield",
    "prepare_lane_change_left",
    "prepare_lane_change_right",
    "commit_lane_change_left",
    "commit_lane_change_right",
    "keep_route_through_junction",
}

FORBIDDEN_AGENT_OUTPUTS = {
    "direct_steering",
    "direct_throttle",
    "direct_brake",
    "trajectory_override",
    "mpc_override",
    "bypass_fallback",
    "control_authority_claim",
}


# ── What Stage 3B baseline tactical emits (per-frame contract) ────────────────
BASELINE_TACTICAL_CONTRACT = {
    "artifact": "stage3b_behavior_request_v2",
    "fields": {
        "requested_behavior": {
            "role": "selected tactical intent (canonical behavior set)",
            "required": True,
            "reusable_by_agent": True,
            "notes": "agent shadow must align its intent to same canonical set",
        },
        "target_lane": {
            "role": "target lane for lane-change maneuvers (current/left/right)",
            "required": True,
            "reusable_by_agent": True,
        },
        "route_mode": {
            "role": "routing mode (corridor_only / graph / none)",
            "required": True,
            "reusable_by_agent": True,
        },
        "route_option": {
            "role": "selected route option (straight/left/right/unknown)",
            "required": True,
            "reusable_by_agent": True,
        },
        "preferred_lane": {
            "role": "preferred lane from route (current/left/right)",
            "required": True,
            "reusable_by_agent": True,
        },
        "lane_change_permission.left": {
            "role": "left lane-change safety gate (bool)",
            "required": True,
            "reusable_by_agent": True,
            "notes": "agent MUST respect this; forbidden to request LC when permission is False",
        },
        "lane_change_permission.right": {
            "role": "right lane-change safety gate (bool)",
            "required": True,
            "reusable_by_agent": True,
        },
        "lane_change_stage": {
            "role": "LC lifecycle phase (idle/prepare/commit/completing/aborting)",
            "required": False,
            "reusable_by_agent": True,
        },
        "target_speed_mps": {
            "role": "target speed command from baseline tactical",
            "required": True,
            "reusable_by_agent": False,
            "notes": "agent shadow must NOT emit speed as a control command",
        },
        "confidence": {
            "role": "baseline tactical confidence in its own decision",
            "required": False,
            "reusable_by_agent": True,
            "notes": "agent emits its own confidence field separately",
        },
        "stop_reason": {
            "role": "reason for stop/yield behavior",
            "required": False,
            "reusable_by_agent": True,
        },
        "route_conflict_flags": {
            "role": "conflict issues detected in route binding",
            "required": True,
            "reusable_by_agent": True,
        },
    },
}

# ── What Stage 3A emits that agent can read ───────────────────────────────────
LANE_CONTEXT_AGENT_INPUTS = {
    "artifact": "stage3a_lane_aware_world_state",
    "fields": {
        "lane_context.ego_lane_id": {"role": "current lane ID", "required": True},
        "lane_context.left_lane.exists": {"role": "left lane physically exists", "required": True},
        "lane_context.right_lane.exists": {"role": "right lane physically exists", "required": True},
        "lane_context.junction_context": {"role": "junction topology (type/distance/branch count)", "required": False},
        "lane_scene.current_lane_blocked": {"role": "blocking indicator for current lane", "required": True},
        "lane_relative_objects": {"role": "per-object lane assignments + distances", "required": True},
    },
}

# ── What Stage 2 world model emits that agent can read ────────────────────────
WORLD_STATE_AGENT_INPUTS = {
    "artifact": "stage2_world_state",
    "fields": {
        "ego.speed_mps": {"role": "current ego speed", "required": True},
        "ego.route_progress_m": {"role": "distance travelled along route", "required": True},
        "objects": {"role": "tracked objects list (track_id, class, position, velocity)", "required": True},
        "risk_summary.highest_risk_level": {"role": "categorical risk level (none/low/medium/high)", "required": True},
        "risk_summary.front_hazard_track_id": {"role": "track id of front hazard object if any", "required": False},
        "risk_summary.nearest_front_vehicle_distance_m": {"role": "gap to nearest front vehicle", "required": False},
        "scene.front_free_space_m": {"role": "free space ahead in meters", "required": False},
    },
}

# ── Stop/fallback context available to agent ──────────────────────────────────
STOP_FALLBACK_AGENT_INPUTS = {
    "artifact": "stage3c_stop_target + stage3c_fallback_events",
    "fields": {
        "stop_target.binding_status": {"role": "stop target contract binding status", "required": False},
        "stop_target.distance_to_stop_m": {"role": "remaining distance to stop target", "required": False},
        "fallback_events": {"role": "recent fallback activations and reasons", "required": False},
    },
}

# ── Agent shadow new output contract fields (to be emitted by agent) ──────────
AGENT_SHADOW_OUTPUT_CONTRACT = {
    "artifact": "agent_shadow_intent.jsonl / agent_shadow_summary.json",
    "fields": {
        "case_id": {"role": "benchmark case identifier", "required": True, "agent_emits": True},
        "frame_id": {"role": "per-frame tick identifier", "required": True, "agent_emits": True},
        "shadow_mode": {"role": "always True — marks this as non-authoritative", "required": True, "agent_emits": True},
        "model_id": {"role": "model or provider identifier (e.g. gemini-2.5-pro / stub_v1)", "required": True, "agent_emits": True},
        "tactical_intent": {"role": "selected tactical intent from ALLOWED_AGENT_INTENTS", "required": True, "agent_emits": True},
        "target_lane": {"role": "target lane for LC intents (current/left/right)", "required": False, "agent_emits": True},
        "target_speed_mps": {
            "role": "FORBIDDEN — agent must NOT emit direct speed control",
            "required": False,
            "agent_emits": False,
            "forbidden": True,
        },
        "reason_tags": {"role": "list of structured reason codes explaining intent choice", "required": True, "agent_emits": True},
        "route_context_summary": {"role": "brief structured summary of route context agent consumed", "required": True, "agent_emits": True},
        "stop_context_summary": {"role": "brief structured summary of stop context agent consumed", "required": False, "agent_emits": True},
        "confidence": {"role": "agent confidence in its intent [0.0 – 1.0]", "required": True, "agent_emits": True},
        "timeout_flag": {"role": "True if agent did not respond within budget; force-fallback", "required": True, "agent_emits": True},
        "validation_status": {"role": "contract validation result (valid/invalid_intent/forbidden_field/malformed)", "required": True, "agent_emits": True},
        "fallback_to_baseline": {"role": "True if agent timed-out/invalid → baseline is authority", "required": True, "agent_emits": True},
        "provenance": {
            "role": "dict: model_id, provider, call_latency_ms, input_token_count, fallback_reason",
            "required": True,
            "agent_emits": True,
        },
        "baseline_intent": {"role": "copied from Stage 3B baseline for comparison", "required": True, "agent_emits": True},
        "agrees_with_baseline": {"role": "True if agent intent == baseline intent", "required": True, "agent_emits": True},
        "disagreement_useful": {
            "role": "True if disagreement is structurally valid and on a clean contract boundary",
            "required": False,
            "agent_emits": True,
        },
    },
    "binding_rules": [
        "tactical_intent MUST be one of ALLOWED_AGENT_INTENTS",
        "target_speed_mps MUST NOT appear in agent output — forbidden field",
        "shadow_mode MUST be True always",
        "fallback_to_baseline MUST be True when timeout_flag is True",
        "fallback_to_baseline MUST be True when validation_status != 'valid'",
        "agent MUST NOT request prepare/commit_lane_change when lane_change_permission is False",
        "reason_tags MUST be a non-empty list of strings",
        "provenance MUST include model_id and call_latency_ms",
    ],
}


@dataclass
class AgentContextAvailability:
    artifact: str
    fields_available: list[str] = field(default_factory=list)
    fields_missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _check_world_state(ws: dict[str, Any]) -> AgentContextAvailability:
    avail = AgentContextAvailability("stage2_world_state")
    checks = {
        "ego.speed_mps": bool((ws.get("ego") or {}).get("speed_mps") is not None),
        "ego.route_progress_m": bool((ws.get("ego") or {}).get("route_progress_m") is not None),
        "objects": bool(ws.get("objects")),
        "risk_summary.highest_risk_level": bool((ws.get("risk_summary") or {}).get("highest_risk_level")),
        "risk_summary.front_hazard_track_id": bool((ws.get("risk_summary") or {}).get("front_hazard_track_id") is not None),
        "scene.front_free_space_m": bool((ws.get("scene") or {}).get("front_free_space_m") is not None),
    }
    for k, v in checks.items():
        if v:
            avail.fields_available.append(k)
        else:
            avail.fields_missing.append(k)
    return avail


def _check_behavior_request(br: dict[str, Any]) -> AgentContextAvailability:
    avail = AgentContextAvailability("stage3b_behavior_request_v2")
    checks = {
        "requested_behavior": bool(br.get("requested_behavior")),
        "target_lane": bool(br.get("target_lane")),
        "route_mode": bool(br.get("route_mode")),
        "route_option": bool(br.get("route_option")),
        "preferred_lane": bool(br.get("preferred_lane")),
        "lane_change_permission.left": bool((br.get("lane_change_permission") or {}).get("left") is not None),
        "lane_change_permission.right": bool((br.get("lane_change_permission") or {}).get("right") is not None),
        "confidence": bool(br.get("confidence") is not None),
        "route_conflict_flags": bool(br.get("route_conflict_flags") is not None),
    }
    for k, v in checks.items():
        if v:
            avail.fields_available.append(k)
        else:
            avail.fields_missing.append(k)
    return avail


def _check_lane_context(lane_ws: dict[str, Any]) -> AgentContextAvailability:
    avail = AgentContextAvailability("stage3a_lane_aware_world_state")
    lc = lane_ws.get("lane_context") or {}
    checks = {
        "lane_context.ego_lane_id": bool(lc.get("ego_lane_id")),
        "lane_context.left_lane.exists": bool((lc.get("left_lane") or {}).get("exists") is not None),
        "lane_context.right_lane.exists": bool((lc.get("right_lane") or {}).get("exists") is not None),
        "lane_context.junction_context": bool(lc.get("junction_context")),
        "lane_scene.current_lane_blocked": bool((lane_ws.get("lane_scene") or {}).get("current_lane_blocked") is not None),
        "lane_relative_objects": bool(lane_ws.get("lane_relative_objects")),
    }
    for k, v in checks.items():
        if v:
            avail.fields_available.append(k)
        else:
            avail.fields_missing.append(k)
    return avail


def _audit_case(case_spec: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    artifacts = case_spec.get("artifacts") or {}
    results: dict[str, Any] = {}

    ws_path = resolve_repo_path(repo_root, artifacts.get("stage2_world_state_sample"))
    ws = load_json(ws_path) if ws_path else None
    if ws:
        results["world_state"] = asdict(_check_world_state(ws))
    else:
        results["world_state"] = {"artifact": "stage2_world_state", "status": "artifact_missing"}

    lane_path = resolve_repo_path(repo_root, artifacts.get("stage3a_lane_world_state_sample"))
    lane_ws = load_json(lane_path) if lane_path else None
    if lane_ws:
        results["lane_context"] = asdict(_check_lane_context(lane_ws))
    else:
        results["lane_context"] = {"artifact": "stage3a_lane_aware_world_state", "status": "artifact_missing"}

    br_path = resolve_repo_path(repo_root, artifacts.get("stage3b_behavior_request_sample"))
    br = load_json(br_path) if br_path else None
    if br:
        results["behavior_request"] = asdict(_check_behavior_request(br))
    else:
        results["behavior_request"] = {"artifact": "stage3b_behavior_request_v2", "status": "artifact_missing"}

    stop_path = resolve_repo_path(repo_root, artifacts.get("stage3c_stop_target"))
    results["stop_target_available"] = bool(stop_path and stop_path.exists())

    fallback_path = resolve_repo_path(repo_root, artifacts.get("stage3c_fallback_events"))
    results["fallback_events_available"] = bool(fallback_path and fallback_path.exists())

    return results


def run_stage7_contract_audit(
    repo_root: str | Path,
    config_path: str = "benchmark/benchmark_v1.yaml",
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_config = load_yaml(repo_root / config_path)
    cases_dir = resolve_repo_path(repo_root, benchmark_config["cases_dir"])

    case_files = sorted(cases_dir.glob("*.yaml"))
    cases: list[dict[str, Any]] = []
    for cf in case_files:
        spec = load_yaml(cf)
        if spec.get("status") in {"ready", "provisional"}:
            cases.append(spec)

    per_case: dict[str, Any] = {}
    for case_spec in cases:
        per_case[case_spec["scenario_id"]] = _audit_case(case_spec, repo_root)

    # Tally coverage
    ws_avail = sum(1 for v in per_case.values() if "artifact_missing" not in str(v.get("world_state")))
    lc_avail = sum(1 for v in per_case.values() if "artifact_missing" not in str(v.get("lane_context")))
    br_avail = sum(1 for v in per_case.values() if "artifact_missing" not in str(v.get("behavior_request")))
    total = len(per_case)

    audit_result = {
        "schema_version": "stage7_agent_shadow_contract_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "audit_scope": "stage7_agentic_shadow_precheck",

        # ── What baseline tactical currently emits ──────────────────────────
        "baseline_tactical_output": BASELINE_TACTICAL_CONTRACT,

        # ── What agent can read ─────────────────────────────────────────────
        "agent_readable_inputs": {
            "world_state": WORLD_STATE_AGENT_INPUTS,
            "lane_context": LANE_CONTEXT_AGENT_INPUTS,
            "stop_fallback_context": STOP_FALLBACK_AGENT_INPUTS,
        },

        # ── What agent must emit ────────────────────────────────────────────
        "agent_shadow_output_contract": AGENT_SHADOW_OUTPUT_CONTRACT,

        # ── Allowed / forbidden intents ─────────────────────────────────────
        "allowed_agent_intents": sorted(ALLOWED_AGENT_INTENTS),
        "forbidden_agent_outputs": sorted(FORBIDDEN_AGENT_OUTPUTS),

        # ── Minimum input bundle for agent shadow ───────────────────────────
        "minimum_input_bundle_for_agent": [
            "ego_state (ego.speed_mps, ego.route_progress_m)",
            "tracked_objects_summary (objects list simplified)",
            "lane_context (ego_lane_id, left/right exists, junction_context)",
            "route_context_summary (route_option, preferred_lane, route_mode, route_conflict_flags)",
            "stop_context_summary (stop_target.distance_to_stop_m if available)",
            "baseline_tactical_context (baseline requested_behavior + lane_change_permission)",
        ],

        # ── Contracts that MUST be reused ───────────────────────────────────
        "contracts_to_reuse": [
            "lane_change_permission gate (must not request LC when permission=False)",
            "fallback spine — if agent timeout/invalid → baseline authority unchanged",
            "route/session binding — agent reads route_option but does NOT rebind route",
            "stop contract — agent reads stop context but does NOT set stop target",
            "MPC safety spine — agent has NO path to modify trajectory authority",
        ],

        # ── New contracts needed for Stage 7 ────────────────────────────────
        "new_contracts_for_stage7": [
            "agent_shadow_intent.jsonl — per-frame agent shadow intent with full provenance",
            "agent_shadow_summary.json — per-case summary incl. timeout/validity/disagreement rates",
            "agent_baseline_comparison.json — per-case baseline vs agent intent comparison",
            "agent_disagreement_events.jsonl — frames where agent != baseline, with structured context",
            "agent_shadow_intent.schema.json — JSON Schema for contract enforcement",
        ],

        # ── Per-case artifact coverage ──────────────────────────────────────
        "per_case_artifact_coverage": per_case,

        "summary": {
            "total_cases_audited": total,
            "world_state_artifact_available": ws_avail,
            "lane_context_artifact_available": lc_avail,
            "behavior_request_artifact_available": br_avail,
            "fraction_world_state_ready": round(ws_avail / max(total, 1), 3),
            "fraction_lane_context_ready": round(lc_avail / max(total, 1), 3),
            "fraction_behavior_request_ready": round(br_avail / max(total, 1), 3),
            "stage7a_replay_readiness": "ready" if br_avail > 0 else "no_ready_cases",
            "stage7b_scenario_replay_readiness": "design_complete_pending_execution",
            "stage7c_online_readiness": "design_only_not_rollout_yet",
        },
    }

    output_path = repo_root / "benchmark" / "stage7_agent_shadow_audit.json"
    dump_json(output_path, audit_result)
    print(f"[Stage7] Contract audit written to {output_path}")
    return audit_result
