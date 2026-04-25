from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


def _to_dict(value: Any) -> Dict[str, Any]:
    return asdict(value)


@dataclass
class LaneDescriptor:
    exists: bool
    role: str
    road_id: int | None
    section_id: int | None
    lane_id: int | None
    lane_width_m: float | None
    lane_type: str | None
    lane_change: str | None
    same_direction_as_ego: bool
    transform_carla: Dict[str, Any] | None
    transform_bevfusion_world: Dict[str, Any] | None
    lane_change_allowed_from_current: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


@dataclass
class JunctionContext:
    is_in_junction: bool
    junction_ahead: bool
    distance_to_junction_m: float | None
    branch_count_ahead: int
    possible_turn_like_options: List[str]
    branch_distance_m: float | None

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


@dataclass
class LaneContext:
    frame_id: int
    sample_name: str
    current_lane: LaneDescriptor
    left_lane: LaneDescriptor
    right_lane: LaneDescriptor
    forward_corridor: Dict[str, Any]
    junction_context: JunctionContext

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "sample_name": self.sample_name,
            "current_lane": self.current_lane.to_dict(),
            "left_lane": self.left_lane.to_dict(),
            "right_lane": self.right_lane.to_dict(),
            "forward_corridor": self.forward_corridor,
            "junction_context": self.junction_context.to_dict(),
        }


@dataclass
class LaneAwareObject:
    track_id: int
    class_name: str
    class_group: str
    position_world_carla: List[float]
    lane_relation: str
    lane_tag: str
    object_lane_id: int | None
    object_road_id: int | None
    longitudinal_m: float | None
    lateral_m: float | None
    is_front_in_current_lane: bool
    is_rear_in_current_lane: bool
    is_blocking_current_lane: bool
    same_direction_as_ego_lane: bool
    distance_to_lane_center_m: float | None
    source_track: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


@dataclass
class ManeuverValidation:
    frame_id: int
    requested_intent: Dict[str, Any]
    valid_maneuvers: List[str]
    invalid_maneuvers: List[Dict[str, str]]
    selected_behavior: str
    selected_maneuver: str
    lane_change_permission: Dict[str, Any]
    reasoning_tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


@dataclass
class BehaviorRequest:
    frame: int
    sample_name: str
    requested_behavior: str
    target_lane: str
    target_speed_mps: float
    longitudinal_horizon_m: float
    blocking_object_id: int | None
    stop_reason: str | None
    stop_target: Dict[str, Any] | None
    lane_change_permission: Dict[str, Any]
    constraints: List[str]
    confidence: float
    scene_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


@dataclass
class LaneAwareWorldState:
    frame: int
    timestamp: float
    sample_name: str
    ego: Dict[str, Any]
    scene: Dict[str, Any]
    risk_summary: Dict[str, Any]
    decision_context: Dict[str, Any]
    lane_context: Dict[str, Any]
    lane_relative_objects: List[Dict[str, Any]]
    behavior_request_preview: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)
