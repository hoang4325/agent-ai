from __future__ import annotations

from typing import Any, Dict, List

from .lane_change_staging import determine_lane_change_stage
from .schema import BehaviorSelection, RouteConditionedScene, RouteContext


def _dedupe(items: List[str]) -> List[str]:
    return sorted(set(item for item in items if item))


def select_behavior(
    *,
    stage3_record: Dict[str, Any],
    route_context: RouteContext,
    route_conditioned_scene: RouteConditionedScene,
) -> BehaviorSelection:
    stage2_decision = dict(stage3_record["stage2_decision"])
    maneuver_validation = dict(stage3_record["maneuver_validation"])
    stage3_behavior = str(stage3_record["behavior_request"]["requested_behavior"])
    world_state = dict(stage3_record["world_state"])
    lane_context = dict(stage3_record["lane_context"])
    junction_context = dict(lane_context["junction_context"])

    selected_behavior = stage3_behavior
    route_conditioned_behavior = stage3_behavior
    lane_change_stage = "none"
    target_lane = str(stage3_record["behavior_request"].get("target_lane", "current"))
    override_reason = None
    route_influence = False
    reasoning_tags: List[str] = []
    behavior_constraints: List[str] = [f"route_mode_{route_context.route_mode}"]
    scene_constraints: List[str] = list(route_context.route_conflict_flags)

    safety_behaviors = {"stop_before_obstacle", "yield"}
    if stage3_behavior in safety_behaviors:
        override_reason = "stage3a_safety_behavior_preserved"
    else:
        route_option = route_context.route_option
        decision_distance = route_context.distance_to_next_route_decision_m
        in_route_junction_window = route_option in {"left", "right", "straight"} and (
            bool(junction_context.get("is_in_junction"))
            or (
                decision_distance is not None
                and float(decision_distance) <= 25.0
            )
        )

        if in_route_junction_window and stage3_behavior in {"keep_lane", "slow_down", "follow"}:
            selected_behavior = "keep_route_through_junction"
            route_conditioned_behavior = selected_behavior
            target_lane = "current"
            route_influence = True
            override_reason = f"route_option_{route_option}_through_junction"
            reasoning_tags.append("route_through_junction")
            behavior_constraints.append("respect_route_option_at_branch")

        staging = determine_lane_change_stage(
            stage2_decision=stage2_decision,
            stage3_behavior=stage3_behavior,
            world_state=world_state,
            route_context=route_context.to_dict(),
            route_conditioned_scene=route_conditioned_scene.to_dict(),
            lane_change_permission=maneuver_validation["lane_change_permission"],
        )
        if staging["behavior"] is not None and stage3_behavior not in safety_behaviors:
            selected_behavior = str(staging["behavior"])
            route_conditioned_behavior = selected_behavior
            lane_change_stage = str(staging["stage"])
            target_lane = str(staging["target_lane"])
            route_influence = True
            override_reason = str(staging["override_reason"])
            reasoning_tags.extend(staging["reasoning_tags"])
            behavior_constraints.extend(staging["behavior_constraints"])
            scene_constraints.extend(staging["scene_constraints"])

    if selected_behavior == stage3_behavior and stage3_behavior in {"prepare_lane_change_left", "prepare_lane_change_right"}:
        lane_change_stage = "prepare"
        target_lane = "left" if stage3_behavior.endswith("left") else "right"
    elif selected_behavior.startswith("commit_lane_change_"):
        lane_change_stage = "commit"
        target_lane = "left" if selected_behavior.endswith("left") else "right"
    elif selected_behavior.startswith("prepare_lane_change_"):
        lane_change_stage = "prepare"
        target_lane = "left" if selected_behavior.endswith("left") else "right"

    if route_influence and selected_behavior != stage3_behavior:
        reasoning_tags.append("route_conditioned_override")
    if route_context.route_mode == "corridor_only":
        behavior_constraints.append("minimal_route_context_only")
    elif route_context.route_mode == "minimal_route_hint":
        behavior_constraints.append("minimal_route_hint_applied")
    elif route_context.route_mode == "replay_route_manifest":
        behavior_constraints.append("replay_route_manifest_applied")

    return BehaviorSelection(
        frame_id=int(stage3_record["frame_id"]),
        sample_name=str(stage3_record["sample_name"]),
        requested_intent_stage2=stage2_decision,
        validated_maneuver_stage3a=maneuver_validation,
        route_conditioned_behavior=route_conditioned_behavior,
        lane_change_stage=lane_change_stage,
        selected_behavior=selected_behavior,
        target_lane=target_lane,
        override_reason=override_reason,
        route_influence=route_influence,
        reasoning_tags=_dedupe(reasoning_tags),
        behavior_constraints=_dedupe(behavior_constraints),
        scene_constraints=_dedupe(scene_constraints),
    )
