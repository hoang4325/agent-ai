from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

from .schema import LaneAwareObject, LaneContext, ManeuverValidation

LOGGER = logging.getLogger(__name__)


def _nearest_front_object(
    lane_objects: Iterable[LaneAwareObject],
    *,
    lane_relation: str,
) -> LaneAwareObject | None:
    candidates = [
        item
        for item in lane_objects
        if item.lane_relation == lane_relation
        and item.longitudinal_m is not None
        and item.longitudinal_m > 0.0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: float(item.longitudinal_m or 1e9))
    return candidates[0]


def _nearest_rear_object(
    lane_objects: Iterable[LaneAwareObject],
    *,
    lane_relation: str,
) -> LaneAwareObject | None:
    candidates = [
        item
        for item in lane_objects
        if item.lane_relation == lane_relation
        and item.longitudinal_m is not None
        and item.longitudinal_m < 0.0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: abs(float(item.longitudinal_m or 1e9)))
    return candidates[0]


def _front_vru_hazard(lane_objects: Iterable[LaneAwareObject]) -> LaneAwareObject | None:
    candidates = [
        item
        for item in lane_objects
        if item.class_group == "vru"
        and item.longitudinal_m is not None
        and item.longitudinal_m > 0.0
        and item.lane_relation in {"current_lane", "cross_lane", "left_lane", "right_lane"}
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: float(item.longitudinal_m or 1e9))
    return candidates[0]


def _invalidate(invalid: List[Dict[str, str]], maneuver: str, reason: str) -> None:
    invalid.append({"maneuver": str(maneuver), "reason": str(reason)})


def _lane_change_feasibility(
    *,
    direction: str,
    lane_context: LaneContext,
    lane_objects: List[LaneAwareObject],
) -> Tuple[bool, str | None]:
    lane_descriptor = lane_context.left_lane if direction == "left" else lane_context.right_lane
    relation = "left_lane" if direction == "left" else "right_lane"

    if not lane_descriptor.exists:
        return False, f"no_{direction}_lane"
    if not lane_descriptor.same_direction_as_ego:
        return False, f"{direction}_lane_opposite_direction"
    if not lane_descriptor.lane_change_allowed_from_current:
        return False, f"{direction}_lane_change_not_permitted"
    if lane_context.junction_context.is_in_junction:
        return False, "ego_in_junction"
    if (
        lane_context.junction_context.distance_to_junction_m is not None
        and lane_context.junction_context.distance_to_junction_m < 15.0
    ):
        return False, "junction_too_close"

    front_target = _nearest_front_object(lane_objects, lane_relation=relation)
    if front_target is not None and float(front_target.longitudinal_m or 1e9) < 15.0:
        return False, f"{direction}_lane_front_gap_too_small"
    rear_target = _nearest_rear_object(lane_objects, lane_relation=relation)
    if rear_target is not None and abs(float(rear_target.longitudinal_m or -1e9)) < 10.0:
        return False, f"{direction}_lane_rear_gap_too_small"
    return True, None


def validate_maneuvers(
    *,
    world_state: Dict[str, Any],
    stage2_decision_intent: Dict[str, Any],
    lane_context: LaneContext,
    lane_objects: Iterable[LaneAwareObject],
) -> ManeuverValidation:
    lane_objects = list(lane_objects)
    invalid: List[Dict[str, str]] = []
    valid: List[str] = []
    reasoning_tags: List[str] = []

    front_blocker = _nearest_front_object(lane_objects, lane_relation="current_lane")
    front_distance = None if front_blocker is None else float(front_blocker.longitudinal_m or 0.0)
    front_vru = _front_vru_hazard(lane_objects)
    front_vru_distance = None if front_vru is None else float(front_vru.longitudinal_m or 0.0)
    highest_risk = str(world_state["risk_summary"]["highest_risk_level"])
    junction_near = bool(lane_context.junction_context.is_in_junction) or (
        lane_context.junction_context.distance_to_junction_m is not None
        and lane_context.junction_context.distance_to_junction_m < 15.0
    )

    valid.append("keep_lane")
    if front_blocker is None:
        _invalidate(invalid, "follow", "no_front_object_in_current_lane")
    else:
        valid.append("follow")
    if front_blocker is not None or highest_risk in {"medium", "high", "critical"} or junction_near:
        valid.append("slow_down")
    else:
        _invalidate(invalid, "slow_down", "scene_not_constrained")
    if (
        (front_distance is not None and front_distance < 10.0)
        or highest_risk in {"high", "critical"}
        or (front_vru_distance is not None and front_vru_distance < 8.0)
    ):
        valid.append("stop")
    else:
        _invalidate(invalid, "stop", "no_immediate_stop_condition")
    if front_vru is not None and ((front_vru_distance is not None and front_vru_distance < 20.0) or junction_near):
        valid.append("yield")
    else:
        _invalidate(invalid, "yield", "no_vru_yield_condition")

    left_ok, left_reason = _lane_change_feasibility(
        direction="left",
        lane_context=lane_context,
        lane_objects=lane_objects,
    )
    right_ok, right_reason = _lane_change_feasibility(
        direction="right",
        lane_context=lane_context,
        lane_objects=lane_objects,
    )
    if left_ok:
        valid.append("lane_change_left")
    else:
        _invalidate(invalid, "lane_change_left", left_reason or "left_lane_unavailable")
    if right_ok:
        valid.append("lane_change_right")
    else:
        _invalidate(invalid, "lane_change_right", right_reason or "right_lane_unavailable")

    selected_behavior = "keep_lane"
    selected_maneuver = "keep_lane"

    if front_distance is not None and front_distance < 8.0 and "stop" in valid:
        selected_behavior = "stop_before_obstacle"
        selected_maneuver = "stop"
        reasoning_tags.append("current_lane_stop_zone")
    elif front_vru_distance is not None and front_vru_distance < 15.0 and "yield" in valid:
        selected_behavior = "yield"
        selected_maneuver = "yield"
        reasoning_tags.append("vru_yield_condition")
    elif front_distance is not None and front_distance < 25.0:
        if left_ok:
            selected_behavior = "prepare_lane_change_left"
            selected_maneuver = "lane_change_left"
            reasoning_tags.append("left_lane_escape_available")
        elif right_ok:
            selected_behavior = "prepare_lane_change_right"
            selected_maneuver = "lane_change_right"
            reasoning_tags.append("right_lane_escape_available")
        elif "follow" in valid:
            selected_behavior = "follow"
            selected_maneuver = "follow"
            reasoning_tags.append("follow_front_object")
        else:
            selected_behavior = "slow_down"
            selected_maneuver = "slow_down"
            reasoning_tags.append("front_blocker_without_escape")
    else:
        requested = str(stage2_decision_intent.get("maneuver", "keep_lane"))
        if requested == "yield" and "yield" in valid:
            selected_behavior = "yield"
            selected_maneuver = "yield"
        elif requested in {"stop", "emergency_stop"} and "stop" in valid:
            selected_behavior = "stop_before_obstacle"
            selected_maneuver = "stop"
        elif requested == "slow_down" and "slow_down" in valid:
            selected_behavior = "slow_down"
            selected_maneuver = "slow_down"
        else:
            selected_behavior = "keep_lane"
            selected_maneuver = "keep_lane"

    lane_change_permission = {
        "left": bool(left_ok),
        "right": bool(right_ok),
        "left_reason": None if left_ok else left_reason,
        "right_reason": None if right_ok else right_reason,
    }
    LOGGER.info(
        "Maneuver validation frame=%d sample=%s requested=%s selected=%s valid=%s",
        int(world_state["frame"]),
        str(world_state["sample_name"]),
        stage2_decision_intent.get("maneuver"),
        selected_behavior,
        valid,
    )
    return ManeuverValidation(
        frame_id=int(world_state["frame"]),
        requested_intent=dict(stage2_decision_intent),
        valid_maneuvers=valid,
        invalid_maneuvers=invalid,
        selected_behavior=selected_behavior,
        selected_maneuver=selected_maneuver,
        lane_change_permission=lane_change_permission,
        reasoning_tags=reasoning_tags,
    )
