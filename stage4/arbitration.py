from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List

from stage3c.execution_contract import ExecutionRequest


@dataclass
class ArbitrationDecision:
    action: str
    effective_request: ExecutionRequest
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["effective_request"] = self.effective_request.to_dict()
        return payload


def build_fallback_request(
    *,
    frame: int,
    sample_name: str,
    behavior: str,
    target_speed_mps: float,
    route_option: str = "unknown",
    priority: str = "safety",
    confidence: float = 0.25,
    notes: List[str] | None = None,
) -> ExecutionRequest:
    constraints = list(notes or [])
    stop_reason = "stage4_fallback" if behavior == "stop_before_obstacle" else None
    return ExecutionRequest(
        frame=int(frame),
        sample_name=str(sample_name),
        requested_behavior=str(behavior),
        target_lane="current",
        target_speed_mps=float(target_speed_mps),
        lane_change_stage="none",
        blocking_object_id=None,
        constraints=constraints,
        execution_mode="local_planner_bridge",
        priority=str(priority),
        route_option=str(route_option),
        stop_reason=stop_reason,
        stop_target=(
            {
                "source_type": "fallback_safety",
                "binding_status": "exact",
                "source_detail": "stage4_fallback",
                "source_role": "deterministic_fallback",
                "source_track_id": None,
                "source_actor_id": None,
                "target_distance_m": 0.0,
                "initial_distance_to_target_m": 0.0,
                "target_lane_id": None,
                "target_lane_relation": "current_lane",
                "target_pose": None,
                "target_confidence": 0.95,
                "stop_reason": stop_reason,
                "behavior_constraints": [],
                "scene_constraints": list(constraints),
                "provenance": {
                    "source_stage": "stage4_arbitration",
                    "source_artifact": "build_fallback_request",
                    "derivation_mode": "exact",
                },
            }
            if behavior == "stop_before_obstacle"
            else None
        ),
        confidence=float(confidence),
    )


def downgrade_request(request: ExecutionRequest, *, target_speed_mps: float) -> ExecutionRequest:
    return ExecutionRequest(
        frame=int(request.frame),
        sample_name=str(request.sample_name),
        requested_behavior="slow_down",
        target_lane="current",
        target_speed_mps=float(max(0.0, target_speed_mps)),
        lane_change_stage="none",
        blocking_object_id=request.blocking_object_id,
        constraints=sorted(set(list(request.constraints) + ["degraded_from_stale_or_conflict"])),
        execution_mode=str(request.execution_mode),
        priority=str(request.priority),
        route_option=str(request.route_option),
        stop_reason=request.stop_reason,
        stop_target=request.stop_target,
        confidence=min(0.5, float(request.confidence)),
    )


def arbitrate_request(
    *,
    proposed_request: ExecutionRequest | None,
    last_effective_request: ExecutionRequest | None,
    active_lane_change: Any | None,
    perception_stale_ticks: int,
    soft_stale_ticks: int,
    hard_stale_ticks: int,
    last_behavior_frame: int | None,
) -> ArbitrationDecision:
    fallback_frame = int(last_effective_request.frame) if last_effective_request is not None else int(last_behavior_frame or 0)
    fallback_sample = (
        str(last_effective_request.sample_name)
        if last_effective_request is not None
        else f"stale_frame_{fallback_frame:06d}"
    )

    if proposed_request is None:
        if last_effective_request is None:
            return ArbitrationDecision(
                action="emergency_override",
                effective_request=build_fallback_request(
                    frame=fallback_frame,
                    sample_name=fallback_sample,
                    behavior="stop_before_obstacle",
                    target_speed_mps=0.0,
                    notes=["waiting_for_first_valid_perception"],
                ),
                notes=["no_behavior_request_available"],
            )
        if perception_stale_ticks >= hard_stale_ticks:
            return ArbitrationDecision(
                action="emergency_override",
                effective_request=build_fallback_request(
                    frame=int(last_effective_request.frame),
                    sample_name=str(last_effective_request.sample_name),
                    behavior="stop_before_obstacle",
                    target_speed_mps=0.0,
                    route_option=str(last_effective_request.route_option),
                    notes=["hard_perception_timeout"],
                ),
                notes=["hard_perception_timeout"],
            )
        if perception_stale_ticks >= soft_stale_ticks and active_lane_change is None:
            return ArbitrationDecision(
                action="downgrade",
                effective_request=downgrade_request(
                    last_effective_request,
                    target_speed_mps=min(2.0, float(last_effective_request.target_speed_mps)),
                ),
                notes=["soft_perception_timeout"],
            )
        return ArbitrationDecision(
            action="continue",
            effective_request=last_effective_request,
            notes=["reusing_last_effective_request"],
        )

    if active_lane_change is not None and getattr(active_lane_change, "status", None) == "executing":
        active_direction = str(getattr(active_lane_change, "direction", "unknown"))
        requested_behavior = str(proposed_request.requested_behavior)
        if requested_behavior == "stop_before_obstacle":
            return ArbitrationDecision(
                action="emergency_override",
                effective_request=proposed_request,
                notes=["stop_request_supersedes_active_lane_change"],
            )
        if requested_behavior.startswith(("prepare_lane_change_", "commit_lane_change_")):
            requested_direction = "left" if requested_behavior.endswith("left") else "right"
            if requested_direction != active_direction and last_effective_request is not None:
                return ArbitrationDecision(
                    action="cancel",
                    effective_request=last_effective_request,
                    notes=["conflicting_lane_change_ignored_until_current_completes"],
                )
        if last_effective_request is not None and not requested_behavior.startswith("stop_before_obstacle"):
            return ArbitrationDecision(
                action="continue",
                effective_request=last_effective_request,
                notes=["continuing_active_lane_change"],
            )

    if "route_option_unknown" in proposed_request.constraints and proposed_request.requested_behavior == "keep_route_through_junction":
        return ArbitrationDecision(
            action="downgrade",
            effective_request=downgrade_request(
                proposed_request,
                target_speed_mps=min(3.0, float(proposed_request.target_speed_mps)),
            ),
            notes=["unknown_route_cannot_force_junction_behavior"],
        )

    if "route_option_unknown" in proposed_request.constraints and proposed_request.requested_behavior != "stop_before_obstacle":
        return ArbitrationDecision(
            action="downgrade",
            effective_request=downgrade_request(
                proposed_request,
                target_speed_mps=min(1.5, float(proposed_request.target_speed_mps)),
            ),
            notes=["route_uncertainty_requires_speed_downgrade"],
        )

    return ArbitrationDecision(
        action="continue",
        effective_request=proposed_request,
        notes=["proposed_request_accepted"],
    )
