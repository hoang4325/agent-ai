"""Maneuver validation with speed-adaptive time-headway gaps for lane changes."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

from agent_ai.world_state.kinematics import time_headway_gap_m
from agent_ai.world_state.soft_constraint_arbiter import SoftConstraintArbiter

from .lane_change_cost import evaluate_lane_change_candidates
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


def _ego_speed_mps(world_state: Dict[str, Any]) -> float:
    ego = world_state.get("ego") or {}
    if isinstance(ego, dict):
        for key in ("speed_mps", "ego_speed_mps", "v_mps"):
            if key in ego and ego[key] is not None:
                return max(0.0, float(ego[key]))
    return 0.0


def _track_speed_mps(obj: LaneAwareObject | None) -> float | None:
    if obj is None:
        return None
    src = obj.source_track or {}
    if "speed_mps" in src and src["speed_mps"] is not None:
        return max(0.0, float(src["speed_mps"]))
    vel = src.get("velocity_ego")
    if isinstance(vel, (list, tuple)) and len(vel) >= 2:
        return float((float(vel[0]) ** 2 + float(vel[1]) ** 2) ** 0.5)
    return None


def _lane_change_feasibility(
    *,
    direction: str,
    lane_context: LaneContext,
    lane_objects: List[LaneAwareObject],
    ego_speed_mps: float,
    front_gap_t_s: float = 1.5,
    front_gap_d0_m: float = 8.0,
    rear_gap_t_s: float = 1.2,
    rear_gap_d0_m: float = 6.0,
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

    # Speed-adaptive geometric gaps (constant time headway + standstill buffer).
    min_front_gap = time_headway_gap_m(
        ego_speed_mps,
        t_gap_s=front_gap_t_s,
        d0_m=front_gap_d0_m,
    )
    # Rear gap scales with the faster of ego / rear vehicle (closing threat).
    front_target = _nearest_front_object(lane_objects, lane_relation=relation)
    rear_target = _nearest_rear_object(lane_objects, lane_relation=relation)
    rear_speed = _track_speed_mps(rear_target)
    rear_ref_speed = max(ego_speed_mps, rear_speed if rear_speed is not None else 0.0)
    min_rear_gap = time_headway_gap_m(
        rear_ref_speed,
        t_gap_s=rear_gap_t_s,
        d0_m=rear_gap_d0_m,
    )

    if front_target is not None and float(front_target.longitudinal_m or 1e9) < min_front_gap:
        return False, f"{direction}_lane_front_gap_too_small"
    if rear_target is not None and abs(float(rear_target.longitudinal_m or -1e9)) < min_rear_gap:
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

    ego_speed = _ego_speed_mps(world_state)
    front_blocker = _nearest_front_object(lane_objects, lane_relation="current_lane")
    front_distance = None if front_blocker is None else float(front_blocker.longitudinal_m or 0.0)
    front_vru = _front_vru_hazard(lane_objects)
    front_vru_distance = None if front_vru is None else float(front_vru.longitudinal_m or 0.0)
    highest_risk = str(world_state["risk_summary"]["highest_risk_level"])
    junction_near = bool(lane_context.junction_context.is_in_junction) or (
        lane_context.junction_context.distance_to_junction_m is not None
        and lane_context.junction_context.distance_to_junction_m < 15.0
    )

    # Follow / stop thresholds also scale mildly with speed.
    stop_gap_m = time_headway_gap_m(ego_speed, t_gap_s=0.8, d0_m=6.0)
    follow_interest_m = time_headway_gap_m(ego_speed, t_gap_s=2.5, d0_m=18.0)
    vru_yield_m = time_headway_gap_m(ego_speed, t_gap_s=1.5, d0_m=12.0)

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
        (front_distance is not None and front_distance < stop_gap_m)
        or highest_risk in {"high", "critical"}
        or (front_vru_distance is not None and front_vru_distance < 8.0)
    ):
        valid.append("stop")
    else:
        _invalidate(invalid, "stop", "no_immediate_stop_condition")
    if front_vru is not None and (
        (front_vru_distance is not None and front_vru_distance < vru_yield_m) or junction_near
    ):
        valid.append("yield")
    else:
        _invalidate(invalid, "yield", "no_vru_yield_condition")

    left_ok, left_reason = _lane_change_feasibility(
        direction="left",
        lane_context=lane_context,
        lane_objects=lane_objects,
        ego_speed_mps=ego_speed,
    )
    right_ok, right_reason = _lane_change_feasibility(
        direction="right",
        lane_context=lane_context,
        lane_objects=lane_objects,
        ego_speed_mps=ego_speed,
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
    cost_eval: Dict[str, Any] | None = None

    # Hard safety first (not cost-traded).
    if front_distance is not None and front_distance < stop_gap_m * 0.85 and "stop" in valid:
        selected_behavior = "stop_before_obstacle"
        selected_maneuver = "stop"
        reasoning_tags.append("current_lane_stop_zone")
        reasoning_tags.append("speed_adaptive_stop_gap")
    elif front_vru_distance is not None and front_vru_distance < vru_yield_m and "yield" in valid:
        selected_behavior = "yield"
        selected_maneuver = "yield"
        reasoning_tags.append("vru_yield_condition")
    else:
        # Cost-based keep vs LC (and follow/slow when keep wins under front pressure).
        requested = str(stage2_decision_intent.get("maneuver", "keep_lane"))
        route_prefer = None
        lane_pref = str(stage2_decision_intent.get("lane_preference", "keep_current"))
        if lane_pref == "prefer_left" or requested == "lane_change_left":
            route_prefer = "left"
        elif lane_pref == "prefer_right" or requested == "lane_change_right":
            route_prefer = "right"

        scene = world_state.get("scene") or {}
        cost_eval = evaluate_lane_change_candidates(
            lane_context=lane_context,
            lane_objects=lane_objects,
            ego_speed_mps=ego_speed,
            left_ok=left_ok,
            left_reason=left_reason,
            right_ok=right_ok,
            right_reason=right_reason,
            route_prefer=route_prefer,
            current_front_gap_m=front_distance,
            left_occupancy=float(scene.get("left_side_occupancy", 0.5) or 0.5),
            right_occupancy=float(scene.get("right_side_occupancy", 0.5) or 0.5),
            highest_risk=highest_risk,
            prev_maneuver=requested if requested.startswith("lane_change_") else None,
            arbiter=SoftConstraintArbiter(),
        )
        reasoning_tags.extend(list(cost_eval.get("reasoning_tags") or []))
        reasoning_tags.append("cost_based_lc")

        stage = str(cost_eval.get("stage", "none"))
        if stage in {"prepare", "commit"} and cost_eval.get("selected_maneuver", "").startswith("lane_change_"):
            selected_maneuver = str(cost_eval["selected_maneuver"])
            selected_behavior = str(cost_eval.get("selected_behavior") or selected_maneuver)
            reasoning_tags.append(f"lc_stage_{stage}")
        elif front_distance is not None and front_distance < follow_interest_m:
            if "follow" in valid:
                selected_behavior = "follow"
                selected_maneuver = "follow"
                reasoning_tags.append("follow_front_object")
            elif "slow_down" in valid:
                selected_behavior = "slow_down"
                selected_maneuver = "slow_down"
                reasoning_tags.append("front_blocker_without_escape")
            reasoning_tags.append("speed_adaptive_follow_horizon")
        else:
            if requested == "yield" and "yield" in valid:
                selected_behavior = "yield"
                selected_maneuver = "yield"
            elif requested in {"stop", "emergency_stop"} and "stop" in valid:
                selected_behavior = "stop_before_obstacle"
                selected_maneuver = "stop"
            elif requested == "slow_down" and "slow_down" in valid:
                selected_behavior = "slow_down"
                selected_maneuver = "slow_down"
            elif requested == "follow" and "follow" in valid:
                selected_behavior = "follow"
                selected_maneuver = "follow"
            else:
                selected_behavior = "keep_lane"
                selected_maneuver = "keep_lane"

    lane_change_permission = {
        "left": bool(left_ok),
        "right": bool(right_ok),
        "left_reason": None if left_ok else left_reason,
        "right_reason": None if right_ok else right_reason,
        "min_front_gap_m": time_headway_gap_m(ego_speed, t_gap_s=1.5, d0_m=8.0),
        "min_rear_gap_m": time_headway_gap_m(ego_speed, t_gap_s=1.2, d0_m=6.0),
        "ego_speed_mps": ego_speed,
        "cost_eval": cost_eval,
    }
    LOGGER.info(
        "Maneuver validation frame=%d sample=%s requested=%s selected=%s valid=%s ego_v=%.2f",
        int(world_state["frame"]),
        str(world_state["sample_name"]),
        stage2_decision_intent.get("maneuver"),
        selected_behavior,
        valid,
        ego_speed,
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
