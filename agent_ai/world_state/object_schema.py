from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence


def dataclass_to_dict(value: Any) -> Dict[str, Any]:
    return asdict(value)


@dataclass
class EgoState:
    frame_id: int
    timestamp: float
    sample_name: str
    town: str | None
    weather: Dict[str, float]
    position_world: List[float]
    velocity_world: List[float]
    speed_mps: float
    yaw_deg: float
    route_progress_m: float
    world_from_ego_bevfusion: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass
class NormalizedDetection:
    detection_id: str
    class_id: int
    class_name: str
    class_group: str
    score: float
    position_ego: List[float]
    size_xyz: List[float]
    yaw_rad: float
    velocity_ego: List[float]
    speed_mps: float
    distance_m: float
    bearing_deg: float
    source_confidence: float
    raw_box_9d: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass
class NormalizedFramePrediction:
    sample_name: str
    sequence_index: int
    frame_id: int
    timestamp: float
    prediction_variant: str
    source_sample_dir: str
    source_prediction_dir: str
    ego: EgoState
    detections: List[NormalizedDetection]
    class_names: List[str]
    raw_prediction_count: int
    filtered_prediction_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_name": self.sample_name,
            "sequence_index": self.sequence_index,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "prediction_variant": self.prediction_variant,
            "source_sample_dir": self.source_sample_dir,
            "source_prediction_dir": self.source_prediction_dir,
            "ego": self.ego.to_dict(),
            "detections": [item.to_dict() for item in self.detections],
            "class_names": list(self.class_names),
            "raw_prediction_count": self.raw_prediction_count,
            "filtered_prediction_count": self.filtered_prediction_count,
        }


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    class_group: str
    latest_detection_id: str | None
    age_frames: int
    hits: int
    missed_frames: int
    is_occluded_est: bool
    score: float
    mean_score: float
    position_ego: List[float]
    velocity_ego: List[float]
    speed_mps: float
    bbox: List[float]
    size_xyz: List[float]
    yaw_rad: float
    distance_m: float
    bearing_deg: float
    ttc_seconds: float | None
    relative_sector: str
    source_confidence: float
    risk_score: float = 0.0
    risk_level: str = "low"
    reasoning_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass
class SceneSummary:
    active_object_count: int
    front_free_space_m: float | None
    left_side_occupancy: float
    right_side_occupancy: float
    rear_gap_m: float | None
    nearest_front_vehicle: Dict[str, Any] | None
    nearest_vru: Dict[str, Any] | None
    nearest_any_object: Dict[str, Any] | None
    abnormal_flags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass
class RiskSummary:
    highest_risk_level: str
    highest_risk_score: float
    urgent_track_ids: List[int]
    front_hazard_track_id: int | None
    nearest_front_vehicle_distance_m: float | None
    nearest_vru_distance_m: float | None
    minimum_ttc_seconds: float | None
    flags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass
class DecisionIntent:
    frame_id: int
    timestamp: float
    maneuver: str
    target_speed_mps: float
    lane_preference: str
    constraints: List[str]
    confidence: float
    reasoning_tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass
class PlannerInterfacePayload:
    frame_id: int
    timestamp: float
    ego: Dict[str, Any]
    decision_intent: Dict[str, Any]
    hard_constraints: List[str]
    soft_constraints: List[str]
    tracked_objects: List[Dict[str, Any]]
    scene_summary: Dict[str, Any]
    risk_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass
class WorldState:
    frame_id: int
    timestamp: float
    sample_name: str
    sequence_index: int
    ego: EgoState
    objects: List[TrackedObject]
    scene: SceneSummary
    risk_summary: RiskSummary
    decision_context: Dict[str, Sequence[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame_id,
            "timestamp": self.timestamp,
            "sample_name": self.sample_name,
            "sequence_index": self.sequence_index,
            "ego": self.ego.to_dict(),
            "objects": [item.to_dict() for item in self.objects],
            "scene": self.scene.to_dict(),
            "risk_summary": self.risk_summary.to_dict(),
            "decision_context": {
                key: list(value) for key, value in self.decision_context.items()
            },
        }
