from __future__ import annotations

from typing import Any, Dict, Iterable

from stage3.stop_target_contract import build_stop_target_contract

from .schema import BehaviorRequestV2, BehaviorSelection, RouteConditionedScene, RouteContext


def _dedupe(items: Iterable[str]) -> list[str]:
    return sorted(set(item for item in items if item))


def _current_lane_blocker(stage3_record: Dict[str, Any]) -> Dict[str, Any] | None:
    blocker = stage3_record["lane_scene"].get("front_object_in_current_lane")
    if blocker:
        return dict(blocker)
    nearest = stage3_record["world_state"]["scene"].get("nearest_front_vehicle")
    if nearest and nearest.get("track_id") is not None:
        return {"track_id": int(nearest["track_id"]), "distance_m": nearest.get("distance_m")}
    return None


def build_behavior_request_v2(
    *,
    stage3_record: Dict[str, Any],
    route_context: RouteContext,
    route_conditioned_scene: RouteConditionedScene,
    behavior_selection: BehaviorSelection,
) -> BehaviorRequestV2:
    stage3_behavior_request = dict(stage3_record["behavior_request"])
    stage2_decision = dict(stage3_record["stage2_decision"])
    world_state = dict(stage3_record["world_state"])
    ego_speed = float(world_state["ego"]["speed_mps"])
    stage2_target_speed = float(stage2_decision.get("target_speed_mps", stage3_behavior_request["target_speed_mps"]))
    base_target_speed = float(stage3_behavior_request["target_speed_mps"])
    selected_behavior = behavior_selection.selected_behavior

    current_lane_blocker = _current_lane_blocker(stage3_record)
    front_route_blocker = route_conditioned_scene.front_route_blocker

    if selected_behavior == "stop_before_obstacle":
        target_speed = 0.0
        stop_reason = stage3_behavior_request.get("stop_reason") or "front_obstacle_or_high_risk"
        blocking_object_id = None if current_lane_blocker is None else int(current_lane_blocker["track_id"])
    elif selected_behavior == "yield":
        target_speed = min(base_target_speed, max(0.0, ego_speed * 0.35))
        stop_reason = stage3_behavior_request.get("stop_reason") or "yield_to_vru_or_crossing_flow"
        blocking_object_id = None if current_lane_blocker is None else int(current_lane_blocker["track_id"])
    elif selected_behavior.startswith("prepare_lane_change_"):
        target_speed = min(stage2_target_speed, max(4.0, ego_speed - 0.5))
        stop_reason = None
        blocking_object_id = (
            int(front_route_blocker["track_id"])
            if front_route_blocker is not None and front_route_blocker.get("track_id") is not None
            else (None if current_lane_blocker is None else int(current_lane_blocker["track_id"]))
        )
    elif selected_behavior.startswith("commit_lane_change_"):
        target_speed = min(stage2_target_speed, max(5.0, ego_speed))
        stop_reason = None
        blocking_object_id = (
            int(front_route_blocker["track_id"])
            if front_route_blocker is not None and front_route_blocker.get("track_id") is not None
            else None
        )
    elif selected_behavior == "keep_route_through_junction":
        target_speed = min(stage2_target_speed, max(3.0, ego_speed))
        stop_reason = None
        blocking_object_id = None
    else:
        target_speed = base_target_speed
        stop_reason = stage3_behavior_request.get("stop_reason")
        blocking_object_id = None if current_lane_blocker is None else int(current_lane_blocker["track_id"])

    longitudinal_horizon_m = float(stage3_behavior_request.get("longitudinal_horizon_m", 40.0))
    if (
        selected_behavior in {"keep_route_through_junction"}
        or selected_behavior.startswith("prepare_lane_change_")
        or selected_behavior.startswith("commit_lane_change_")
    ) and route_context.distance_to_next_route_decision_m is not None:
        longitudinal_horizon_m = max(
            12.0,
            min(longitudinal_horizon_m, float(route_context.distance_to_next_route_decision_m) + 10.0),
        )

    confidence = float(stage3_behavior_request.get("confidence", 0.6))
    if behavior_selection.route_influence:
        if behavior_selection.lane_change_stage == "commit":
            confidence = min(0.90, confidence + 0.05)
        elif behavior_selection.lane_change_stage == "prepare":
            confidence = min(0.85, confidence)
        else:
            confidence = min(0.88, max(confidence, 0.62))
    if route_context.route_mode == "corridor_only" and behavior_selection.route_influence:
        confidence = max(0.45, confidence - 0.05)

    behavior_constraints = _dedupe(
        list(stage3_behavior_request.get("constraints", [])) + list(behavior_selection.behavior_constraints)
    )
    scene_constraints = _dedupe(
        list(stage3_behavior_request.get("scene_constraints", []))
        + list(behavior_selection.scene_constraints)
        + [f"route_option_{route_context.route_option}"]
    )
    stop_target = build_stop_target_contract(
        requested_behavior=selected_behavior,
        target_lane=str(behavior_selection.target_lane),
        stop_reason=stop_reason,
        blocking_object=current_lane_blocker,
        lane_context=dict(stage3_record.get("lane_context") or {}),
        route_context=route_context.to_dict(),
        behavior_constraints=behavior_constraints,
        scene_constraints=scene_constraints,
        provenance_stage="stage3b",
        provenance_artifact="behavior_request_v2_builder",
    )

    return BehaviorRequestV2(
        frame=int(stage3_record["frame_id"]),
        sample_name=str(stage3_record["sample_name"]),
        requested_behavior=selected_behavior,
        target_lane=str(behavior_selection.target_lane),
        route_mode=route_context.route_mode,
        route_option=route_context.route_option,
        preferred_lane=route_context.preferred_lane,
        route_priority=route_context.route_priority,
        lane_change_stage=behavior_selection.lane_change_stage,
        target_speed_mps=float(target_speed),
        longitudinal_horizon_m=float(longitudinal_horizon_m),
        blocking_object_id=blocking_object_id,
        stop_reason=stop_reason,
        stop_target=stop_target,
        lane_change_permission=dict(stage3_record["maneuver_validation"]["lane_change_permission"]),
        behavior_constraints=behavior_constraints,
        scene_constraints=scene_constraints,
        route_conflict_flags=list(route_context.route_conflict_flags),
        confidence=float(confidence),
    )
