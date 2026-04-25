from __future__ import annotations

from typing import List

from .object_schema import DecisionIntent, PlannerInterfacePayload, WorldState


def build_planner_interface_payload(
    world_state: WorldState,
    decision_intent: DecisionIntent,
    *,
    max_objects: int = 8,
) -> PlannerInterfacePayload:
    tracked_objects = sorted(
        world_state.objects,
        key=lambda item: (-float(item.risk_score), float(item.distance_m)),
    )
    selected_objects = tracked_objects[:max_objects]
    return PlannerInterfacePayload(
        frame_id=world_state.frame_id,
        timestamp=world_state.timestamp,
        ego={
            "speed_mps": float(world_state.ego.speed_mps),
            "yaw_deg": float(world_state.ego.yaw_deg),
            "route_progress_m": float(world_state.ego.route_progress_m),
            "position_world": list(world_state.ego.position_world),
        },
        decision_intent=decision_intent.to_dict(),
        hard_constraints=list(world_state.decision_context["hard_constraints"]),
        soft_constraints=list(world_state.decision_context["soft_constraints"]),
        tracked_objects=[item.to_dict() for item in selected_objects],
        scene_summary=world_state.scene.to_dict(),
        risk_summary=world_state.risk_summary.to_dict(),
    )
