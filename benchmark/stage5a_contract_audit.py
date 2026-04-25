"""
Stage 5A Contract Audit Engine
===============================
Audits frozen corpus + benchmark cases to identify:
- contract clarity per layer (world_state, lane, route, behavior, execution, arbitration)
- ambiguous / missing fields
- hardening recommendations

Output: stage5a_contract_audit.json
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_json, load_jsonl, load_yaml, resolve_repo_path


# ── Contract field definitions ────────────────────────────────────────────────

WORLD_STATE_CONTRACT = {
    "fields": {
        "ego.speed_mps": {"role": "ego velocity", "required": True, "stage": "stage2"},
        "ego.route_progress_m": {"role": "route progress", "required": True, "stage": "stage2"},
        "objects": {"role": "tracked objects list", "required": True, "stage": "stage2"},
        "risk_summary.highest_risk_level": {"role": "risk level", "required": True, "stage": "stage2"},
        "risk_summary.front_hazard_track_id": {"role": "front hazard binding", "required": False, "stage": "stage2"},
        "risk_summary.nearest_front_vehicle_distance_m": {"role": "front gap", "required": False, "stage": "stage2"},
        "scene.front_free_space_m": {"role": "free space ahead", "required": False, "stage": "stage2"},
    },
    "binding_rules": [
        "front_hazard_track_id must refer to an existing tracked object if present",
        "objects must include track_id, class_name, position_m, velocity_mps at minimum",
    ],
}

LANE_TOPOLOGY_CONTRACT = {
    "fields": {
        "lane_context.ego_lane_id": {"role": "current lane identifier", "required": True, "stage": "stage3a"},
        "lane_context.left_lane_available": {"role": "left lane passability", "required": True, "stage": "stage3a"},
        "lane_context.right_lane_available": {"role": "right lane passability", "required": True, "stage": "stage3a"},
        "lane_context.junction_context": {"role": "junction topology", "required": False, "stage": "stage3a"},
        "lane_relative_objects[].lane_relation": {"role": "per-object lane assignment", "required": True, "stage": "stage3a"},
        "lane_relative_objects[].lane_tag": {"role": "lane tag string", "required": False, "stage": "stage3a"},
        "lane_scene.current_lane_blocked": {"role": "blocking flag", "required": True, "stage": "stage3a"},
    },
    "binding_rules": [
        "lane_relation must be one of: current_lane, left_lane, right_lane, cross_lane, opposite_lane",
        "current_lane_blocked must be consistent with blocker binding",
        "junction_context must include junction_type if near junction",
    ],
}

ROUTE_SESSION_CONTRACT = {
    "fields": {
        "route_option": {"role": "selected route option", "required": True, "stage": "stage3b"},
        "preferred_lane": {"role": "preferred lane from route", "required": True, "stage": "stage3b"},
        "route_mode": {"role": "routing mode (corridor/graph/none)", "required": True, "stage": "stage3b"},
        "source_of_route_binding": {"role": "provenance of route binding", "required": True, "stage": "stage3b"},
        "route_conflict_flags": {"role": "conflict indicators", "required": True, "stage": "stage3b"},
        "route_priority": {"role": "route priority level", "required": False, "stage": "stage3b"},
        "route_confidence": {"role": "confidence in route binding", "required": False, "stage": "stage3b"},
        "session_binding_source": {"role": "session data provenance", "required": True, "stage": "benchmark"},
    },
    "binding_rules": [
        "route_option must match case_spec.route_option if defined",
        "preferred_lane must be derivable from route_option",
        "source_of_route_binding must be one of: stage3b_route_context, case_route_option, unknown",
        "route_conflict_flags must be empty for a clean pass",
    ],
}

BEHAVIOR_REQUEST_CONTRACT = {
    "fields": {
        "requested_behavior": {"role": "behavior command", "required": True, "stage": "stage3b"},
        "target_lane": {"role": "target lane for maneuver", "required": True, "stage": "stage3b"},
        "lane_change_permission.left": {"role": "left LC safety gate", "required": True, "stage": "stage3b"},
        "lane_change_permission.right": {"role": "right LC safety gate", "required": True, "stage": "stage3b"},
        "lane_change_stage": {"role": "LC lifecycle phase", "required": False, "stage": "stage3b"},
        "route_influence": {"role": "route-biased override flag", "required": False, "stage": "stage3b"},
    },
    "binding_rules": [
        "requested_behavior must be one of the canonical behavior set",
        "lane_change_permission must gate prepare/commit behaviors",
        "lane_change_stage must be one of: idle, prepare, commit, completing, aborting",
    ],
}

EXECUTION_CONTRACT = {
    "fields": {
        "execution_request.requested_behavior": {"role": "behavior fed to controller", "required": True, "stage": "stage3c"},
        "execution_state.current_behavior": {"role": "controller active behavior", "required": True, "stage": "stage3c"},
        "execution_state.execution_status": {"role": "controller status", "required": True, "stage": "stage3c"},
        "execution_state.notes": {"role": "controller notes/flags", "required": False, "stage": "stage3c"},
        "lane_change_events[].result": {"role": "LC outcome", "required": True, "stage": "stage3c"},
        "lane_change_events[].direction": {"role": "LC direction", "required": True, "stage": "stage3c"},
        "lane_change_events[].duration_s": {"role": "LC duration", "required": False, "stage": "stage3c"},
        "stop_success": {"role": "stop outcome flag", "required": True, "stage": "stage3c"},
        "stop_overshoot": {"role": "overshoot flag", "required": True, "stage": "stage3c"},
    },
    "binding_rules": [
        "execution_status must be one of: active, success, fail, aborted",
        "lane_change result must be one of: success, failure, abort",
        "stop_overshoot true should only accompany stop_attempts > 0",
    ],
}

ARBITRATION_CONTRACT = {
    "fields": {
        "arbitration_action": {"role": "arbitration decision", "required": True, "stage": "stage3c/stage4"},
        "arbitration_reason": {"role": "reason string", "required": True, "stage": "stage3c/stage4"},
        "requested_behavior": {"role": "what was requested", "required": True, "stage": "stage3c/stage4"},
        "executing_behavior": {"role": "what is actually executing", "required": True, "stage": "stage3c/stage4"},
        "classification": {"role": "expected/unexpected/diagnostic", "required": True, "stage": "stage3c/stage4"},
    },
    "binding_rules": [
        "arbitration_action must be one of: continue, cancel, downgrade, emergency_override",
        "continue: requested == executing, no conflict",
        "cancel: active behavior terminated before completion (abort/fail)",
        "downgrade: requested LC but executing keep_lane/slow_down/follow",
        "emergency_override: stop_before_obstacle preempts any other behavior",
        "classification must be consistent with case_spec expected/forbidden arbitration actions",
    ],
}

ALL_CONTRACTS = {
    "world_state": WORLD_STATE_CONTRACT,
    "lane_topology": LANE_TOPOLOGY_CONTRACT,
    "route_binding": ROUTE_SESSION_CONTRACT,
    "behavior_request": BEHAVIOR_REQUEST_CONTRACT,
    "execution": EXECUTION_CONTRACT,
    "arbitration": ARBITRATION_CONTRACT,
}


# ── Audit logic ───────────────────────────────────────────────────────────────

@dataclass
class FieldAuditResult:
    field: str
    contract_layer: str
    role: str
    required: bool
    status: str  # clear | ambiguous | missing | misused
    notes: list[str] = field(default_factory=list)


@dataclass
class ContractLayerAudit:
    layer: str
    total_fields: int
    clear_fields: int
    ambiguous_fields: int
    missing_fields: int
    misused_fields: int
    field_details: list[dict[str, Any]] = field(default_factory=list)
    binding_rules: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _audit_case_contract_coverage(
    case_spec: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Check which contract fields are actually populated for a single case."""
    artifacts = case_spec.get("artifacts") or {}
    results: dict[str, str] = {}

    # Check world state
    ws_path = resolve_repo_path(repo_root, artifacts.get("stage2_world_state_sample"))
    ws = load_json(ws_path) if ws_path else None
    if ws:
        results["world_state.ego"] = "present" if ws.get("ego") else "missing"
        results["world_state.objects"] = "present" if ws.get("objects") else "missing"
        risk = ws.get("risk_summary") or {}
        results["world_state.risk_summary"] = "present" if risk else "missing"
        results["world_state.front_hazard_track_id"] = "present" if risk.get("front_hazard_track_id") is not None else "absent"
    else:
        results["world_state"] = "artifact_missing"

    # Check lane context
    lane_path = resolve_repo_path(repo_root, artifacts.get("stage3a_lane_world_state_sample"))
    lane_ws = load_json(lane_path) if lane_path else None
    if lane_ws:
        results["lane_context"] = "present" if lane_ws.get("lane_context") else "missing"
        results["lane_relative_objects"] = "present" if lane_ws.get("lane_relative_objects") else "missing"
    else:
        results["lane_topology"] = "artifact_missing"

    # Check behavior request
    br_path = resolve_repo_path(repo_root, artifacts.get("stage3b_behavior_request_sample"))
    br = load_json(br_path) if br_path else None
    if br:
        results["behavior_request.requested_behavior"] = "present" if br.get("requested_behavior") else "missing"
        results["behavior_request.lane_change_permission"] = "present" if br.get("lane_change_permission") else "missing"
    else:
        results["behavior_request"] = "artifact_missing"

    # Check route session binding
    rsb_path = resolve_repo_path(repo_root, artifacts.get("route_session_binding"))
    rsb = load_json(rsb_path) if rsb_path else None
    if rsb:
        rb = rsb.get("route_binding") or {}
        results["route_binding.route_option"] = "present" if rb.get("route_option") else "missing"
        results["route_binding.preferred_lane"] = "present" if rb.get("preferred_lane") else "missing"
        results["route_binding.source_of_route_binding"] = "present" if rb.get("source_of_route_binding") else "missing"
        results["route_binding.route_conflict_flags"] = "clean" if not rb.get("route_conflict_flags") else "has_conflicts"
    else:
        results["route_binding"] = "artifact_missing"

    # Check arbitration events
    arb_path = resolve_repo_path(repo_root, artifacts.get("behavior_arbitration_events"))
    arb = load_jsonl(arb_path) if arb_path else []
    if arb:
        actions = {e.get("arbitration_action") for e in arb}
        results["arbitration.actions_observed"] = ",".join(sorted(actions))
        results["arbitration.event_count"] = str(len(arb))
        unexpected = sum(1 for e in arb if e.get("classification") == "unexpected")
        results["arbitration.unexpected_count"] = str(unexpected)
    else:
        results["arbitration"] = "artifact_missing"

    # Check execution
    exec_path = resolve_repo_path(repo_root, artifacts.get("stage3c_execution_timeline"))
    exec_records = load_jsonl(exec_path) if exec_path else []
    if exec_records:
        results["execution.record_count"] = str(len(exec_records))
    else:
        results["execution"] = "artifact_missing"

    # Check critical actor binding
    cab_path = resolve_repo_path(repo_root, artifacts.get("critical_actor_binding"))
    cab = load_jsonl(cab_path) if cab_path else []
    if cab:
        statuses = {r.get("status") for r in cab}
        results["critical_actor_binding.statuses"] = ",".join(sorted(str(s) for s in statuses))
    else:
        results["critical_actor_binding"] = "artifact_missing"

    return results


def _classify_field_status(
    field_name: str,
    contract_def: dict[str, Any],
    case_coverages: list[dict[str, Any]],
) -> str:
    """Classify a contract field as clear/ambiguous/missing/misused across cases."""
    field_info = contract_def["fields"].get(field_name, {})
    required = field_info.get("required", False)

    # Simple heuristic: check if any case audit indicates the field
    present_count = 0
    missing_count = 0
    for coverage in case_coverages:
        for key, value in coverage.items():
            if field_name.split(".")[0] in key:
                if "missing" in str(value) or "artifact_missing" in str(value):
                    missing_count += 1
                else:
                    present_count += 1

    if present_count == 0 and missing_count == 0:
        return "not_audited"
    if missing_count > 0 and present_count == 0:
        return "missing"
    if missing_count > 0 and present_count > 0:
        return "ambiguous"
    return "clear"


def run_contract_audit(
    repo_root: str | Path,
    config_path: str = "benchmark/benchmark_v1.yaml",
) -> dict[str, Any]:
    """Execute full Stage 5A contract audit across all ready/provisional cases."""
    repo_root = Path(repo_root)
    benchmark_config = load_yaml(repo_root / config_path)
    cases_dir = resolve_repo_path(repo_root, benchmark_config["cases_dir"])

    # Load all cases
    case_files = sorted(cases_dir.glob("*.yaml"))
    cases: list[dict[str, Any]] = []
    for cf in case_files:
        spec = load_yaml(cf)
        if spec.get("status") in {"ready", "provisional"}:
            cases.append(spec)

    # Audit each case
    case_coverages: list[dict[str, Any]] = []
    per_case_audits: dict[str, dict[str, Any]] = {}
    for case_spec in cases:
        coverage = _audit_case_contract_coverage(case_spec, repo_root)
        case_coverages.append(coverage)
        per_case_audits[case_spec["scenario_id"]] = coverage

    # Build per-layer audit
    layer_audits: dict[str, dict[str, Any]] = {}
    all_ambiguous: list[dict[str, str]] = []
    all_recommendations: list[str] = []

    for layer_name, contract_def in ALL_CONTRACTS.items():
        clear = 0
        ambiguous = 0
        missing = 0
        misused = 0
        field_details: list[dict[str, Any]] = []

        for field_name, field_info in contract_def["fields"].items():
            status = _classify_field_status(field_name, contract_def, case_coverages)
            if status == "clear":
                clear += 1
            elif status == "ambiguous":
                ambiguous += 1
                all_ambiguous.append({"field": field_name, "layer": layer_name})
            elif status == "missing":
                missing += 1
            elif status == "misused":
                misused += 1

            field_details.append({
                "field": field_name,
                "role": field_info.get("role", "unknown"),
                "required": field_info.get("required", False),
                "stage": field_info.get("stage", "unknown"),
                "status": status,
            })

        layer_audits[layer_name] = {
            "total_fields": len(contract_def["fields"]),
            "clear_fields": clear,
            "ambiguous_fields": ambiguous,
            "missing_fields": missing,
            "misused_fields": misused,
            "field_details": field_details,
            "binding_rules": contract_def.get("binding_rules", []),
        }

    # Generate hardening recommendations
    all_recommendations = [
        "HARDEN: blocker binding — ensure front_hazard_track_id consistency with lane_relative_objects blocker binding",
        "HARDEN: arbitration classification — add expected/forbidden arbitration action contracts to ALL ready cases",
        "HARDEN: route_session_binding — add route_conflict_flags validation to hard gate path",
        "HARDEN: stop semantics — distinguish stop_before_obstacle vs emergency_stop vs follow_to_stop in execution timeline",
        "HARDEN: lane_change semantics — add prepare→commit→complete lifecycle validation",
        "HARDEN: stale binding — add max_stale_frames threshold to critical_actor_binding contract",
        "HARDEN: session_binding_source — require non-unknown source for all ready cases",
    ]

    audit_result = {
        "schema_version": "stage5a_contract_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "contracts": layer_audits,
        "per_case_audits": per_case_audits,
        "ambiguous_fields": all_ambiguous,
        "hardening_recommendations": all_recommendations,
        "summary": {
            "total_cases_audited": len(cases),
            "total_contracts": len(ALL_CONTRACTS),
            "total_ambiguous_fields": len(all_ambiguous),
            "total_recommendations": len(all_recommendations),
        },
    }

    output_path = repo_root / "benchmark" / "stage5a_contract_audit.json"
    dump_json(output_path, audit_result)
    print(f"[Stage5A] Contract audit written to {output_path}")
    return audit_result
