from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .schema import BehaviorRequest, LaneAwareObject, LaneContext, ManeuverValidation
from .stop_target_contract import build_stop_target_contract


def _current_lane_front_blocker(lane_objects: Iterable[LaneAwareObject]) -> LaneAwareObject | None:
    candidates = [
        item
        for item in lane_objects
        if item.is_front_in_current_lane and item.longitudinal_m is not None
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: float(item.longitudinal_m or 1e9))
    return candidates[0]


def build_behavior_request(
    *,
    world_state: Dict[str, Any],
    stage2_decision_intent: Dict[str, Any],
    lane_context: LaneContext,
    lane_objects: Iterable[LaneAwareObject],
    maneuver_validation: ManeuverValidation,
) -> BehaviorRequest:
    lane_objects = list(lane_objects)
    ego_speed = float(world_state["ego"]["speed_mps"])
    stage2_target_speed = float(stage2_decision_intent.get("target_speed_mps", ego_speed))
    front_blocker = _current_lane_front_blocker(lane_objects)
    front_distance = None if front_blocker is None else float(front_blocker.longitudinal_m or 0.0)

    requested_behavior = maneuver_validation.selected_behavior
    target_lane = "current"
    stop_reason = None
    blocking_object_id = None if front_blocker is None else int(front_blocker.track_id)
    constraints: List[str] = list(world_state.get("decision_context", {}).get("hard_constraints", []))
    scene_constraints: List[str] = []

    if lane_context.junction_context.is_in_junction:
        scene_constraints.append("in_junction")
    elif lane_context.junction_context.distance_to_junction_m is not None:
        scene_constraints.append(
            f"junction_ahead_{lane_context.junction_context.distance_to_junction_m:.1f}m"
        )

    if requested_behavior == "keep_lane":
        target_speed = stage2_target_speed
    elif requested_behavior == "slow_down":
        target_speed = min(stage2_target_speed, max(2.0, ego_speed - 1.0))
        scene_constraints.append("reduce_speed_due_to_lane_context")
    elif requested_behavior == "follow":
        blocker_speed = 0.0 if front_blocker is None else float(front_blocker.source_track.get("speed_mps", 0.0))
        target_speed = min(stage2_target_speed, max(0.0, blocker_speed + 0.5))
        scene_constraints.append("front_object_following")
    elif requested_behavior == "yield":
        target_speed = min(stage2_target_speed, max(0.0, ego_speed * 0.35))
        stop_reason = "yield_to_vru_or_crossing_flow"
        scene_constraints.append("yield_condition")
    elif requested_behavior == "stop_before_obstacle":
        target_speed = 0.0
        stop_reason = "front_obstacle_or_high_risk"
        scene_constraints.append("stop_required")
    elif requested_behavior == "prepare_lane_change_left":
        target_speed = min(stage2_target_speed, max(4.0, ego_speed - 0.5))
        target_lane = "left"
        scene_constraints.append("prepare_left_lane_change")
    elif requested_behavior == "prepare_lane_change_right":
        target_speed = min(stage2_target_speed, max(4.0, ego_speed - 0.5))
        target_lane = "right"
        scene_constraints.append("prepare_right_lane_change")
    else:
        target_speed = stage2_target_speed

    if requested_behavior in {"prepare_lane_change_left", "prepare_lane_change_right"}:
        constraints.append("require_lane_change_gap_confirmation")
    if front_blocker is not None and requested_behavior in {"follow", "slow_down", "stop_before_obstacle"}:
        constraints.append("respect_front_object_buffer")
    if requested_behavior == "yield":
        constraints.append("protect_vru_and_crossing_agents")
    if not lane_context.left_lane.exists:
        constraints.append("no_left_lane_available")
    if not lane_context.right_lane.exists:
        constraints.append("no_right_lane_available")

    longitudinal_horizon_m = 40.0
    if front_distance is not None:
        longitudinal_horizon_m = max(10.0, min(60.0, front_distance))
    elif lane_context.junction_context.distance_to_junction_m is not None:
        longitudinal_horizon_m = max(15.0, min(50.0, float(lane_context.junction_context.distance_to_junction_m)))

    confidence = float(stage2_decision_intent.get("confidence", 0.6))
    if requested_behavior != "keep_lane":
        confidence = min(0.85, confidence + 0.05)
    if lane_context.current_lane.exists:
        confidence = min(0.9, confidence + 0.05)
    else:
        confidence = max(0.35, confidence - 0.2)
    if lane_context.junction_context.is_in_junction and requested_behavior.startswith("prepare_lane_change"):
        confidence = max(0.3, confidence - 0.2)

    stop_target = build_stop_target_contract(
        requested_behavior=requested_behavior,
        target_lane=target_lane,
        stop_reason=stop_reason,
        blocking_object=(None if front_blocker is None else front_blocker.to_dict()),
        lane_context=lane_context.to_dict(),
        behavior_constraints=constraints,
        scene_constraints=scene_constraints,
        provenance_stage="stage3a",
        provenance_artifact="behavior_request_builder",
    )

    return BehaviorRequest(
        frame=int(world_state["frame"]),
        sample_name=str(world_state["sample_name"]),
        requested_behavior=requested_behavior,
        target_lane=target_lane,
        target_speed_mps=float(target_speed),
        longitudinal_horizon_m=float(longitudinal_horizon_m),
        blocking_object_id=blocking_object_id,
        stop_reason=stop_reason,
        stop_target=stop_target,
        lane_change_permission=dict(maneuver_validation.lane_change_permission),
        constraints=constraints,
        confidence=float(confidence),
        scene_constraints=scene_constraints,
    )
