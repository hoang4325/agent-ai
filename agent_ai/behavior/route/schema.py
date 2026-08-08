from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


def _to_dict(value: Any) -> Dict[str, Any]:
    return asdict(value)


@dataclass
class RouteContext:
    frame_id: int
    sample_name: str
    route_mode: str
    route_option: str
    preferred_lane: str
    route_priority: str
    distance_to_next_branch_m: float | None
    distance_to_next_route_decision_m: float | None
    route_conflict_flags: List[str] = field(default_factory=list)
    hint_source: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


@dataclass
class RouteRelativeObject:
    track_id: int
    class_name: str
    class_group: str
    lane_relation: str
    route_relation: str
    route_relevance: str
    longitudinal_m: float | None
    lateral_m: float | None
    is_in_route_lane: bool
    is_front_in_route_lane: bool
    is_route_blocker_candidate: bool
    source_object: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


@dataclass
class RouteConditionedScene:
    frame_id: int
    sample_name: str
    lane_context: Dict[str, Any]
    route_context: Dict[str, Any]
    route_relative_objects: List[Dict[str, Any]]
    front_route_blocker: Dict[str, Any] | None
    route_relevant_adjacent_lane: str
    route_conflict_flags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


@dataclass
class BehaviorSelection:
    frame_id: int
    sample_name: str
    requested_intent_stage2: Dict[str, Any]
    validated_maneuver_stage3a: Dict[str, Any]
    route_conditioned_behavior: str
    lane_change_stage: str
    selected_behavior: str
    target_lane: str
    override_reason: str | None
    route_influence: bool
    reasoning_tags: List[str] = field(default_factory=list)
    behavior_constraints: List[str] = field(default_factory=list)
    scene_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


@dataclass
class BehaviorRequestV2:
    frame: int
    sample_name: str
    requested_behavior: str
    target_lane: str
    route_mode: str
    route_option: str
    preferred_lane: str
    route_priority: str
    lane_change_stage: str
    target_speed_mps: float
    longitudinal_horizon_m: float
    blocking_object_id: int | None
    stop_reason: str | None
    stop_target: Dict[str, Any] | None
    lane_change_permission: Dict[str, Any]
    behavior_constraints: List[str]
    scene_constraints: List[str]
    route_conflict_flags: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)
