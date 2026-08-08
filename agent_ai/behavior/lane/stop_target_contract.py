from __future__ import annotations

from typing import Any, Iterable


DEFAULT_STOP_TARGET_DISTANCE_M = 3.0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _lane_id_from_context(lane_context: dict[str, Any] | None, target_lane: str) -> int | None:
    payload = dict(lane_context or {})
    lane_key = {
        "left": "left_lane",
        "right": "right_lane",
        "current": "current_lane",
    }.get(str(target_lane), "current_lane")
    lane = payload.get(lane_key)
    if isinstance(lane, dict) and lane.get("lane_id") is not None:
        return _safe_int(lane.get("lane_id"))
    current_lane = payload.get("current_lane")
    if isinstance(current_lane, dict):
        return _safe_int(current_lane.get("lane_id"))
    return None


def build_stop_target_contract(
    *,
    requested_behavior: str,
    target_lane: str,
    stop_reason: str | None,
    blocking_object: dict[str, Any] | None,
    lane_context: dict[str, Any] | None,
    route_context: dict[str, Any] | None = None,
    behavior_constraints: Iterable[str] | None = None,
    scene_constraints: Iterable[str] | None = None,
    target_distance_m: float = DEFAULT_STOP_TARGET_DISTANCE_M,
    provenance_stage: str,
    provenance_artifact: str,
) -> dict[str, Any] | None:
    requested_behavior = str(requested_behavior or "")
    stop_reason = None if stop_reason is None else str(stop_reason)
    behavior_constraints = {str(item) for item in (behavior_constraints or []) if item}
    scene_constraints = {str(item) for item in (scene_constraints or []) if item}
    blocking_object = dict(blocking_object or {})
    route_context = dict(route_context or {})
    lane_context = dict(lane_context or {})

    should_define_stop_target = requested_behavior in {"stop_before_obstacle", "yield"} or bool(stop_reason)
    if not should_define_stop_target:
        return None

    current_lane_id = _lane_id_from_context(lane_context, target_lane)
    blocking_track_id = _safe_int(blocking_object.get("track_id"))
    blocking_distance_m = _safe_float(
        blocking_object.get("distance_m", blocking_object.get("longitudinal_m"))
    )

    source_type = "unavailable"
    binding_status = "unavailable"
    source_detail = "stop_target_unavailable"
    source_role = None
    initial_distance_to_target_m = None
    confidence = 0.0
    target_pose = None

    reason_text = str(stop_reason or "")
    if blocking_track_id is not None or blocking_distance_m is not None:
        source_type = "obstacle"
        binding_status = "exact"
        source_detail = "front_lane_blocker_track"
        source_role = "current_lane_blocker"
        initial_distance_to_target_m = blocking_distance_m
        confidence = 0.95 if blocking_distance_m is not None else 0.85
    elif "stop_line" in reason_text or "stop_line_required" in scene_constraints:
        source_type = "stop_line"
        binding_status = "heuristic"
        source_detail = "stop_reason_stop_line"
        confidence = 0.80
    else:
        route_distance_m = _safe_float(route_context.get("distance_to_next_route_decision_m"))
        junction_distance_m = _safe_float(
            ((lane_context.get("junction_context") or {}) if isinstance(lane_context.get("junction_context"), dict) else {}).get("distance_to_junction_m")
        )
        if "junction" in reason_text or route_distance_m is not None or junction_distance_m is not None:
            source_type = "route_junction"
            binding_status = "heuristic"
            source_detail = "route_or_junction_stop_hint"
            initial_distance_to_target_m = route_distance_m if route_distance_m is not None else junction_distance_m
            confidence = 0.70 if initial_distance_to_target_m is not None else 0.55
        elif "fallback" in reason_text:
            source_type = "fallback_safety"
            binding_status = "exact"
            source_detail = "fallback_stop_reason"
            confidence = 0.95

    return {
        "source_type": source_type,
        "binding_status": binding_status,
        "source_detail": source_detail,
        "source_role": source_role,
        "source_track_id": blocking_track_id,
        "source_actor_id": None,
        "target_distance_m": None if source_type == "unavailable" else float(target_distance_m),
        "initial_distance_to_target_m": initial_distance_to_target_m,
        "target_lane_id": current_lane_id,
        "target_lane_relation": "current_lane" if current_lane_id is not None else None,
        "target_pose": target_pose,
        "target_confidence": float(confidence),
        "stop_reason": reason_text,
        "behavior_constraints": sorted(behavior_constraints),
        "scene_constraints": sorted(scene_constraints),
        "provenance": {
            "source_stage": provenance_stage,
            "source_artifact": provenance_artifact,
            "derivation_mode": binding_status,
        },
    }
