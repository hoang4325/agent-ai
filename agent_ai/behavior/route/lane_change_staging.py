"""Route-conditioned lane-change staging with cost-based commit/prepare (P1)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from agent_ai.behavior.lane.lane_change_cost import evaluate_lane_change_candidates
from agent_ai.behavior.lane.schema import JunctionContext, LaneAwareObject, LaneContext, LaneDescriptor
from agent_ai.world_state.soft_constraint_arbiter import SoftConstraintArbiter


def route_demand_direction(route_context: Dict[str, Any]) -> str | None:
    preferred_lane = str(route_context.get("preferred_lane", "unknown"))
    if preferred_lane in {"left", "right"}:
        return preferred_lane
    route_option = str(route_context.get("route_option", "unknown"))
    if route_option in {"left", "right"}:
        return route_option
    return None


def _nearest_target_lane_object(
    route_relative_objects: Iterable[Dict[str, Any]],
    *,
    direction: str,
    ahead: bool,
) -> Dict[str, Any] | None:
    relation = f"{direction}_lane"
    candidates = []
    for item in route_relative_objects:
        if str(item.get("lane_relation")) != relation:
            continue
        longitudinal = item.get("longitudinal_m")
        if longitudinal is None:
            continue
        longitudinal = float(longitudinal)
        if ahead and longitudinal <= 0.0:
            continue
        if not ahead and longitudinal >= 0.0:
            continue
        candidates.append((abs(longitudinal), item))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return dict(candidates[0][1])


def _lane_descriptor_from_dict(payload: Dict[str, Any] | None, role: str) -> LaneDescriptor:
    data = payload or {}
    return LaneDescriptor(
        exists=bool(data.get("exists", False)),
        role=str(data.get("role", role)),
        road_id=data.get("road_id"),
        section_id=data.get("section_id"),
        lane_id=data.get("lane_id"),
        lane_width_m=data.get("lane_width_m"),
        lane_type=data.get("lane_type"),
        lane_change=data.get("lane_change"),
        same_direction_as_ego=bool(data.get("same_direction_as_ego", True)),
        transform_carla=data.get("transform_carla"),
        transform_bevfusion_world=data.get("transform_bevfusion_world"),
        lane_change_allowed_from_current=bool(data.get("lane_change_allowed_from_current", False)),
    )


def _lane_context_from_world(world_state: Dict[str, Any], lane_change_permission: Dict[str, Any]) -> LaneContext:
    """
    Reconstruct a minimal LaneContext for cost evaluation.

    Prefers embedded lane_context on world_state; falls back to permission flags.
    """
    embedded = world_state.get("lane_context") or {}
    if embedded.get("current_lane") is not None:
        jc = embedded.get("junction_context") or {}
        return LaneContext(
            frame_id=int(world_state.get("frame") or world_state.get("frame_id") or 0),
            sample_name=str(world_state.get("sample_name", "")),
            current_lane=_lane_descriptor_from_dict(embedded.get("current_lane"), "current"),
            left_lane=_lane_descriptor_from_dict(embedded.get("left_lane"), "left"),
            right_lane=_lane_descriptor_from_dict(embedded.get("right_lane"), "right"),
            forward_corridor=dict(embedded.get("forward_corridor") or {}),
            junction_context=JunctionContext(
                is_in_junction=bool(jc.get("is_in_junction", False)),
                junction_ahead=bool(jc.get("junction_ahead", False)),
                distance_to_junction_m=jc.get("distance_to_junction_m"),
                branch_count_ahead=int(jc.get("branch_count_ahead") or 0),
                possible_turn_like_options=list(jc.get("possible_turn_like_options") or []),
                branch_distance_m=jc.get("branch_distance_m"),
            ),
        )

    # Fallback synthetic context from permission only.
    def synth(exists: bool, role: str) -> LaneDescriptor:
        return LaneDescriptor(
            exists=exists,
            role=role,
            road_id=1 if exists else None,
            section_id=0 if exists else None,
            lane_id=1 if exists else None,
            lane_width_m=3.5 if exists else None,
            lane_type="Driving" if exists else None,
            lane_change="Both" if exists else None,
            same_direction_as_ego=True,
            transform_carla=None,
            transform_bevfusion_world=None,
            lane_change_allowed_from_current=exists,
        )

    return LaneContext(
        frame_id=int(world_state.get("frame") or 0),
        sample_name=str(world_state.get("sample_name", "")),
        current_lane=synth(True, "current"),
        left_lane=synth(bool(lane_change_permission.get("left", False)), "left"),
        right_lane=synth(bool(lane_change_permission.get("right", False)), "right"),
        forward_corridor={},
        junction_context=JunctionContext(
            is_in_junction=False,
            junction_ahead=False,
            distance_to_junction_m=80.0,
            branch_count_ahead=0,
            possible_turn_like_options=[],
            branch_distance_m=None,
        ),
    )


def _lane_objects_from_route_scene(route_conditioned_scene: Dict[str, Any]) -> List[LaneAwareObject]:
    objects: List[LaneAwareObject] = []
    for raw in route_conditioned_scene.get("route_relative_objects", []) or []:
        objects.append(
            LaneAwareObject(
                track_id=int(raw.get("track_id") or 0),
                class_name=str(raw.get("class_name") or raw.get("class") or "unknown"),
                class_group=str(raw.get("class_group") or "vehicle"),
                position_world_carla=list(raw.get("position_world_carla") or [0.0, 0.0, 0.0]),
                lane_relation=str(raw.get("lane_relation") or "unknown"),
                lane_tag=str(raw.get("lane_tag") or raw.get("lane_relation") or "unknown"),
                object_lane_id=raw.get("object_lane_id"),
                object_road_id=raw.get("object_road_id"),
                longitudinal_m=raw.get("longitudinal_m"),
                lateral_m=raw.get("lateral_m"),
                is_front_in_current_lane=bool(raw.get("is_front_in_current_lane", False)),
                is_rear_in_current_lane=bool(raw.get("is_rear_in_current_lane", False)),
                is_blocking_current_lane=bool(raw.get("is_blocking_current_lane", False)),
                same_direction_as_ego_lane=bool(raw.get("same_direction_as_ego_lane", True)),
                distance_to_lane_center_m=raw.get("distance_to_lane_center_m"),
                source_track=dict(raw.get("source_track") or raw),
            )
        )
    return objects


def determine_lane_change_stage(
    *,
    stage2_decision: Dict[str, Any],
    stage3_behavior: str,
    world_state: Dict[str, Any],
    route_context: Dict[str, Any],
    route_conditioned_scene: Dict[str, Any],
    lane_change_permission: Dict[str, Any],
) -> Dict[str, Any]:
    direction = route_demand_direction(route_context)
    if direction is None:
        return {
            "behavior": None,
            "stage": "none",
            "target_lane": "current",
            "override_reason": None,
            "reasoning_tags": [],
            "behavior_constraints": [],
            "scene_constraints": [],
            "cost_eval": None,
        }

    if not bool(lane_change_permission.get(direction, False)):
        return {
            "behavior": None,
            "stage": "none",
            "target_lane": "current",
            "override_reason": f"route_prefers_{direction}_but_lane_change_not_permitted",
            "reasoning_tags": ["route_lane_change_not_permitted"],
            "behavior_constraints": [f"{direction}_lane_change_not_permitted"],
            "scene_constraints": list(route_context.get("route_conflict_flags", [])),
            "cost_eval": None,
        }

    # Legacy pressure signal retained as soft route preference strength.
    stage2_maneuver = str(stage2_decision.get("maneuver", "keep_lane"))
    lane_preference = str(stage2_decision.get("lane_preference", "keep_current"))
    front_free_space = world_state.get("scene", {}).get("front_free_space_m")
    highest_risk = str(world_state.get("risk_summary", {}).get("highest_risk_level", "low"))
    route_option = str(route_context.get("route_option", "unknown"))
    decision_distance = route_context.get("distance_to_next_route_decision_m")

    pressure = 0
    reasoning_tags: list[str] = []

    if route_option == direction:
        pressure += 2
        reasoning_tags.append("route_turn_requires_lane_shift")
    elif route_option == f"keep_{direction}":
        pressure += 1
        reasoning_tags.append("route_lane_preference_active")

    if stage2_maneuver == f"lane_change_{direction}":
        pressure += 2
        reasoning_tags.append("stage2_requested_matching_lane_change")
    elif lane_preference == f"prefer_{direction}":
        pressure += 1
        reasoning_tags.append("stage2_lane_preference_matches_route")

    if stage3_behavior == f"prepare_lane_change_{direction}":
        pressure += 1
        reasoning_tags.append("stage3a_prepare_matches_route")

    if front_free_space is not None:
        front_free_space = float(front_free_space)
        if front_free_space < 15.0:
            pressure += 2
            reasoning_tags.append("front_gap_short")
        elif front_free_space < 25.0:
            pressure += 1
            reasoning_tags.append("front_gap_reduced")

    if highest_risk in {"high", "critical"}:
        pressure += 1
        reasoning_tags.append("high_risk_scene")
    elif highest_risk == "medium":
        pressure += 1
        reasoning_tags.append("medium_risk_scene")

    if route_option in {"left", "right"} and decision_distance is not None and float(decision_distance) <= 20.0:
        pressure += 1
        reasoning_tags.append("branch_decision_near")

    explicit_need = any(
        [
            stage2_maneuver == f"lane_change_{direction}",
            stage3_behavior == f"prepare_lane_change_{direction}",
            highest_risk in {"medium", "high", "critical"},
            front_free_space is not None and float(front_free_space) < 25.0,
            route_option == direction and decision_distance is not None and float(decision_distance) <= 30.0,
            pressure >= 2,
        ]
    )
    if not explicit_need:
        return {
            "behavior": None,
            "stage": "none",
            "target_lane": "current",
            "override_reason": None,
            "reasoning_tags": [],
            "behavior_constraints": [],
            "scene_constraints": [],
            "cost_eval": None,
        }

    # Cost-based evaluation (P1) — replaces fixed 12/8 m gap + pressure thresholds.
    ego = world_state.get("ego") or {}
    ego_speed = float(ego.get("speed_mps") or 0.0)
    scene = world_state.get("scene") or {}
    lane_objects = _lane_objects_from_route_scene(route_conditioned_scene)
    # Also map current-lane front gap from scene if present.
    front_gap = front_free_space
    if front_gap is None and scene.get("nearest_front_vehicle"):
        front_gap = scene["nearest_front_vehicle"].get("distance_m")

    lane_ctx = _lane_context_from_world(world_state, lane_change_permission)
    left_ok = bool(lane_change_permission.get("left", False))
    right_ok = bool(lane_change_permission.get("right", False))

    cost_eval = evaluate_lane_change_candidates(
        lane_context=lane_ctx,
        lane_objects=lane_objects,
        ego_speed_mps=ego_speed,
        left_ok=left_ok,
        left_reason=lane_change_permission.get("left_reason"),
        right_ok=right_ok,
        right_reason=lane_change_permission.get("right_reason"),
        route_prefer=direction,
        current_front_gap_m=None if front_gap is None else float(front_gap),
        left_occupancy=float(scene.get("left_side_occupancy", 0.5) or 0.5),
        right_occupancy=float(scene.get("right_side_occupancy", 0.5) or 0.5),
        highest_risk=highest_risk,
        prev_maneuver=stage2_maneuver if stage2_maneuver.startswith("lane_change_") else None,
        arbiter=SoftConstraintArbiter(
            # Bias route preference slightly harder in staging.
            weights={"route": 1.8, "progress": 1.2},
        ),
    )
    reasoning_tags.extend(list(cost_eval.get("reasoning_tags") or []))
    reasoning_tags.append("cost_based_staging")

    selected = str(cost_eval.get("selected_maneuver", "keep_lane"))
    stage = str(cost_eval.get("stage", "none"))
    margin = float(cost_eval.get("cost_margin_vs_keep") or 0.0)

    # Route demand must match selected direction (do not override opposite LC).
    if selected != f"lane_change_{direction}":
        # If cost prefers keep/other but route pressure is high and direction is feasible,
        # still allow prepare with weaker confidence.
        if pressure >= 3 and bool(lane_change_permission.get(direction, False)):
            stage = "prepare"
            behavior = f"prepare_lane_change_{direction}"
            reasoning_tags.append("route_pressure_prepare_override")
            return {
                "behavior": behavior,
                "stage": stage,
                "target_lane": direction,
                "override_reason": f"route_conditioned_prepare_{direction}",
                "reasoning_tags": reasoning_tags + [f"prepare_{direction}_lane_change"],
                "behavior_constraints": [
                    "route_conditioned_behavior_v2",
                    "prepare_stage_only_no_lateral_commit",
                    "cost_based_lc",
                ],
                "scene_constraints": list(route_context.get("route_conflict_flags", [])),
                "cost_eval": cost_eval,
            }
        return {
            "behavior": None,
            "stage": "none",
            "target_lane": "current",
            "override_reason": None,
            "reasoning_tags": reasoning_tags,
            "behavior_constraints": [],
            "scene_constraints": [],
            "cost_eval": cost_eval,
        }

    # Gap-limited constraints for planner (soft, not hard reject — cost already handled).
    front_target = _nearest_target_lane_object(
        route_conditioned_scene.get("route_relative_objects", []), direction=direction, ahead=True
    )
    rear_target = _nearest_target_lane_object(
        route_conditioned_scene.get("route_relative_objects", []), direction=direction, ahead=False
    )
    behavior_constraints: list[str] = ["route_conditioned_behavior_v2", "cost_based_lc"]
    scene_constraints: list[str] = list(route_context.get("route_conflict_flags", []))
    if front_target is not None and float(front_target.get("longitudinal_m") or 1e9) < 12.0:
        behavior_constraints.append(f"{direction}_target_lane_front_gap_limited")
    if rear_target is not None and abs(float(rear_target.get("longitudinal_m") or -1e9)) < 8.0:
        behavior_constraints.append(f"{direction}_target_lane_rear_gap_limited")

    # Promote commit when route pressure high and cost margin healthy.
    if stage == "prepare" and pressure >= 4 and margin >= 0.25:
        stage = "commit"
        reasoning_tags.append("route_pressure_commit_boost")
    if stage == "commit" or (stage == "prepare" and pressure >= 4 and margin >= 0.45):
        stage = "commit"
        behavior = f"commit_lane_change_{direction}"
        behavior_constraints.append("lane_change_commit_requires_local_planner_support")
        return {
            "behavior": behavior,
            "stage": "commit",
            "target_lane": direction,
            "override_reason": f"route_conditioned_commit_{direction}",
            "reasoning_tags": reasoning_tags + [f"commit_{direction}_lane_change"],
            "behavior_constraints": behavior_constraints,
            "scene_constraints": scene_constraints,
            "cost_eval": cost_eval,
        }

    if stage == "prepare" or margin > 0.0:
        return {
            "behavior": f"prepare_lane_change_{direction}",
            "stage": "prepare",
            "target_lane": direction,
            "override_reason": f"route_conditioned_prepare_{direction}",
            "reasoning_tags": reasoning_tags + [f"prepare_{direction}_lane_change"],
            "behavior_constraints": behavior_constraints + ["prepare_stage_only_no_lateral_commit"],
            "scene_constraints": scene_constraints,
            "cost_eval": cost_eval,
        }

    return {
        "behavior": None,
        "stage": "none",
        "target_lane": "current",
        "override_reason": None,
        "reasoning_tags": reasoning_tags,
        "behavior_constraints": [],
        "scene_constraints": [],
        "cost_eval": cost_eval,
    }
