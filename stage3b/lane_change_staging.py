from __future__ import annotations

from typing import Any, Dict, Iterable


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
        }

    route_relative_objects = list(route_conditioned_scene.get("route_relative_objects", []))
    front_target = _nearest_target_lane_object(route_relative_objects, direction=direction, ahead=True)
    rear_target = _nearest_target_lane_object(route_relative_objects, direction=direction, ahead=False)

    stage2_maneuver = str(stage2_decision.get("maneuver", "keep_lane"))
    lane_preference = str(stage2_decision.get("lane_preference", "keep_current"))
    front_free_space = world_state["scene"].get("front_free_space_m")
    highest_risk = str(world_state["risk_summary"].get("highest_risk_level", "low"))
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
        }

    target_front_clear = front_target is None or float(front_target.get("longitudinal_m") or 1e9) > 12.0
    target_rear_clear = rear_target is None or abs(float(rear_target.get("longitudinal_m") or -1e9)) > 8.0

    behavior_constraints: list[str] = []
    scene_constraints: list[str] = []
    if not target_front_clear:
        behavior_constraints.append(f"{direction}_target_lane_front_gap_limited")
    if not target_rear_clear:
        behavior_constraints.append(f"{direction}_target_lane_rear_gap_limited")

    if pressure >= 4 and target_front_clear and target_rear_clear:
        return {
            "behavior": f"commit_lane_change_{direction}",
            "stage": "commit",
            "target_lane": direction,
            "override_reason": f"route_conditioned_commit_{direction}",
            "reasoning_tags": reasoning_tags + [f"commit_{direction}_lane_change"],
            "behavior_constraints": behavior_constraints + [
                "route_conditioned_behavior_v2",
                "lane_change_commit_requires_local_planner_support",
            ],
            "scene_constraints": scene_constraints,
        }

    if pressure >= 2:
        return {
            "behavior": f"prepare_lane_change_{direction}",
            "stage": "prepare",
            "target_lane": direction,
            "override_reason": f"route_conditioned_prepare_{direction}",
            "reasoning_tags": reasoning_tags + [f"prepare_{direction}_lane_change"],
            "behavior_constraints": behavior_constraints + [
                "route_conditioned_behavior_v2",
                "prepare_stage_only_no_lateral_commit",
            ],
            "scene_constraints": scene_constraints,
        }

    return {
        "behavior": None,
        "stage": "none",
        "target_lane": "current",
        "override_reason": None,
        "reasoning_tags": [],
        "behavior_constraints": [],
        "scene_constraints": [],
    }
