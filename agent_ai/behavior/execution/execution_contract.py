from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


def _to_dict(value: Any) -> Dict[str, Any]:
    return asdict(value)


@dataclass
class ExecutionRequest:
    frame: int
    sample_name: str
    requested_behavior: str
    target_lane: str
    target_speed_mps: float
    lane_change_stage: str
    blocking_object_id: int | None
    constraints: List[str]
    execution_mode: str
    priority: str
    route_option: str
    stop_reason: str | None
    stop_target: Dict[str, Any] | None
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


@dataclass
class ExecutionState:
    frame: int
    sample_name: str
    requested_behavior: str
    current_behavior: str
    execution_status: str
    current_lane_id: int | None
    target_lane_id: int | None
    lane_change_progress: float
    speed_mps: float
    target_speed_mps: float
    distance_to_target_lane_center: float | None
    distance_to_current_lane_center: float | None
    lateral_shift_m: float
    is_in_junction: bool = False
    distance_to_obstacle_m: float | None = None
    distance_to_stop_target_m: float | None = None
    stop_target_binding_status: str | None = None
    notes: List[str] = field(default_factory=list)
    carla_frame: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


def build_execution_request(stage3b_record: Dict[str, Any]) -> ExecutionRequest:
    behavior_request = dict(stage3b_record["behavior_request_v2"])
    constraints = sorted(
        set(
            list(behavior_request.get("behavior_constraints", []))
            + list(behavior_request.get("scene_constraints", []))
            + list(behavior_request.get("route_conflict_flags", []))
        )
    )
    return ExecutionRequest(
        frame=int(behavior_request["frame"]),
        sample_name=str(behavior_request["sample_name"]),
        requested_behavior=str(behavior_request["requested_behavior"]),
        target_lane=str(behavior_request.get("target_lane", "current")),
        target_speed_mps=float(behavior_request.get("target_speed_mps", 0.0)),
        lane_change_stage=str(behavior_request.get("lane_change_stage", "none")),
        blocking_object_id=(
            None
            if behavior_request.get("blocking_object_id") is None
            else int(behavior_request["blocking_object_id"])
        ),
        constraints=constraints,
        execution_mode="local_planner_bridge",
        priority=str(behavior_request.get("route_priority", "normal")),
        route_option=str(behavior_request.get("route_option", "unknown")),
        stop_reason=behavior_request.get("stop_reason"),
        stop_target=(
            dict(behavior_request.get("stop_target") or {})
            if behavior_request.get("stop_target") is not None
            else None
        ),
        confidence=float(behavior_request.get("confidence", 0.5)),
    )
