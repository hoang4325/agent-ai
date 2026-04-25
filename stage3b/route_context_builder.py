from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .schema import RouteConditionedScene, RouteContext, RouteRelativeObject


def _route_relation(
    *,
    preferred_lane: str,
    lane_relation: str,
) -> tuple[str, str, bool]:
    target_relation = {
        "current": "current_lane",
        "left": "left_lane",
        "right": "right_lane",
    }.get(preferred_lane)

    if target_relation is None:
        return "route_unknown", "context", False
    if lane_relation == target_relation:
        return "route_lane", "primary", True
    if preferred_lane != "current" and lane_relation == "current_lane":
        return "current_transition_lane", "supporting", False
    if lane_relation in {"left_lane", "right_lane"}:
        return "adjacent_context_lane", "secondary", False
    if lane_relation == "cross_lane":
        return "cross_route_context", "secondary", False
    return "off_route_context", "context", False


def _nearest_front_route_blocker(route_relative_objects: Iterable[RouteRelativeObject]) -> RouteRelativeObject | None:
    candidates = [
        item
        for item in route_relative_objects
        if item.is_route_blocker_candidate
        and item.longitudinal_m is not None
        and float(item.longitudinal_m) > 0.0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: float(item.longitudinal_m or 1e9))
    return candidates[0]


def build_route_context(
    *,
    stage3_record: Dict[str, Any],
    route_hint: Dict[str, Any],
) -> RouteContext:
    lane_context = stage3_record["lane_context"]
    junction_context = lane_context["junction_context"]
    route_conflict_flags = list(route_hint.get("route_conflict_flags", []))

    route_option = str(route_hint["route_option"])
    preferred_lane = str(route_hint["preferred_lane"])

    distance_to_next_branch_m = junction_context.get("branch_distance_m")
    if distance_to_next_branch_m is None and junction_context.get("junction_ahead"):
        distance_to_next_branch_m = junction_context.get("distance_to_junction_m")

    distance_to_next_route_decision_m = None
    if route_option in {"left", "right", "straight"}:
        distance_to_next_route_decision_m = (
            junction_context.get("branch_distance_m")
            if junction_context.get("branch_distance_m") is not None
            else junction_context.get("distance_to_junction_m")
        )
    elif int(junction_context.get("branch_count_ahead", 0)) > 1:
        distance_to_next_route_decision_m = (
            junction_context.get("branch_distance_m")
            if junction_context.get("branch_distance_m") is not None
            else junction_context.get("distance_to_junction_m")
        )

    if preferred_lane == "left" and not bool(lane_context["left_lane"]["exists"]):
        route_conflict_flags.append("preferred_left_lane_unavailable")
    if preferred_lane == "right" and not bool(lane_context["right_lane"]["exists"]):
        route_conflict_flags.append("preferred_right_lane_unavailable")
    if (
        route_option in {"left", "right", "straight"}
        and int(junction_context.get("branch_count_ahead", 0)) > 1
        and distance_to_next_route_decision_m is None
    ):
        route_conflict_flags.append("route_branch_distance_unknown")
    if route_option == "unknown":
        route_conflict_flags.append("route_option_unknown")

    deduped_flags = sorted(set(route_conflict_flags))
    return RouteContext(
        frame_id=int(stage3_record["frame_id"]),
        sample_name=str(stage3_record["sample_name"]),
        route_mode=str(route_hint["route_mode"]),
        route_option=route_option,
        preferred_lane=preferred_lane,
        route_priority=str(route_hint["route_priority"]),
        distance_to_next_branch_m=(
            None if distance_to_next_branch_m is None else float(distance_to_next_branch_m)
        ),
        distance_to_next_route_decision_m=(
            None
            if distance_to_next_route_decision_m is None
            else float(distance_to_next_route_decision_m)
        ),
        route_conflict_flags=deduped_flags,
        hint_source=str(route_hint.get("hint_source")) if route_hint.get("hint_source") else None,
    )


def build_route_conditioned_scene(
    *,
    stage3_record: Dict[str, Any],
    route_context: RouteContext,
) -> RouteConditionedScene:
    preferred_lane = route_context.preferred_lane
    lane_objects = stage3_record["lane_relative_objects"]

    route_relative_objects: List[RouteRelativeObject] = []
    for item in lane_objects:
        route_relation, route_relevance, is_in_route_lane = _route_relation(
            preferred_lane=preferred_lane,
            lane_relation=str(item.get("lane_relation", "unknown")),
        )
        is_front_in_route_lane = bool(is_in_route_lane and float(item.get("longitudinal_m") or 0.0) > 0.0)
        is_route_blocker_candidate = bool(
            is_front_in_route_lane
            and item.get("class_group") in {"vehicle", "vru"}
            and bool(item.get("same_direction_as_ego_lane", True))
        )
        route_relative_objects.append(
            RouteRelativeObject(
                track_id=int(item["track_id"]),
                class_name=str(item["class_name"]),
                class_group=str(item["class_group"]),
                lane_relation=str(item["lane_relation"]),
                route_relation=route_relation,
                route_relevance=route_relevance,
                longitudinal_m=(
                    None if item.get("longitudinal_m") is None else float(item["longitudinal_m"])
                ),
                lateral_m=None if item.get("lateral_m") is None else float(item["lateral_m"]),
                is_in_route_lane=is_in_route_lane,
                is_front_in_route_lane=is_front_in_route_lane,
                is_route_blocker_candidate=is_route_blocker_candidate,
                source_object=dict(item),
            )
        )

    front_route_blocker = _nearest_front_route_blocker(route_relative_objects)
    if preferred_lane in {"left", "right"}:
        route_relevant_adjacent_lane = preferred_lane
    else:
        route_relevant_adjacent_lane = "current"

    return RouteConditionedScene(
        frame_id=int(stage3_record["frame_id"]),
        sample_name=str(stage3_record["sample_name"]),
        lane_context=dict(stage3_record["lane_context"]),
        route_context=route_context.to_dict(),
        route_relative_objects=[item.to_dict() for item in route_relative_objects],
        front_route_blocker=None if front_route_blocker is None else front_route_blocker.to_dict(),
        route_relevant_adjacent_lane=route_relevant_adjacent_lane,
        route_conflict_flags=list(route_context.route_conflict_flags),
    )
