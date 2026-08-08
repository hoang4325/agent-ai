from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class PerceptionFrameResult:
    sample_name: str
    sequence_index: int
    frame_id: int
    timestamp: float
    capture_tick_id: int | None
    capture_carla_frame: int | None
    capture_monotonic_s: float | None
    source_sample_dir: str
    source_prediction_dir: str
    stage1_status: str
    prediction_variant: str
    adapt_latency_ms: float | None
    infer_latency_ms: float | None
    visualize_latency_ms: float | None
    perception_latency_ms: float | None
    summary: Dict[str, Any]
    normalized_prediction: Dict[str, Any]
    raw_prediction: Any | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_name": self.sample_name,
            "sequence_index": self.sequence_index,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "capture_tick_id": self.capture_tick_id,
            "capture_carla_frame": self.capture_carla_frame,
            "source_sample_dir": self.source_sample_dir,
            "source_prediction_dir": self.source_prediction_dir,
            "stage1_status": self.stage1_status,
            "prediction_variant": self.prediction_variant,
            "adapt_latency_ms": self.adapt_latency_ms,
            "infer_latency_ms": self.infer_latency_ms,
            "visualize_latency_ms": self.visualize_latency_ms,
            "perception_latency_ms": self.perception_latency_ms,
            "summary": dict(self.summary),
            "normalized_prediction": dict(self.normalized_prediction),
        }


@dataclass
class BehaviorFrameState:
    sample_name: str
    sequence_index: int
    frame_id: int
    timestamp: float
    perception_latency_ms: float | None
    world_update_latency_ms: float
    behavior_latency_ms: float
    normalized_prediction: Dict[str, Any]
    tracked_objects: List[Dict[str, Any]]
    risk_summary: Dict[str, Any]
    world_state: Dict[str, Any]
    decision_intent: Dict[str, Any]
    planner_interface_payload: Dict[str, Any]
    lane_context: Dict[str, Any]
    lane_scene: Dict[str, Any]
    lane_relative_objects: List[Dict[str, Any]]
    maneuver_validation: Dict[str, Any]
    behavior_request: Dict[str, Any]
    route_context: Dict[str, Any]
    route_conditioned_scene: Dict[str, Any]
    behavior_selection: Dict[str, Any]
    behavior_request_v2: Dict[str, Any]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OnlineTickRecord:
    tick_id: int
    frame_id: int
    timestamp: float
    perception_latency_ms: float | None
    world_update_latency_ms: float | None
    behavior_latency_ms: float | None
    execution_latency_ms: float | None
    planner_prepare_latency_ms: float | None
    execution_finalize_latency_ms: float | None
    planner_execution_latency_ms: float | None
    decision_to_execution_latency_ms: float | None
    total_loop_latency_ms: float | None
    latency_budget_ms: float | None
    over_budget: bool | None
    selected_behavior: str | None
    execution_status: str | None
    ego_lane_id: int | None
    speed_mps: float | None
    blocker_distance_m: float | None
    perception_frame_id: int | None
    behavior_frame_id: int | None
    arbitration_action: str | None
    fallback_required: bool | None
    fallback_activated: bool | None
    fallback_reason: str | None
    fallback_source: str | None
    fallback_mode: str | None
    fallback_action: str | None
    fallback_takeover_latency_ms: float | None
    fallback_success: bool | None
    health_flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OnlineStackState:
    tick_id: int
    carla_frame: int | None
    last_capture_sample: str | None
    last_perception_frame_id: int | None
    last_behavior_frame_id: int | None
    tracker_state: Dict[str, Any]
    current_behavior: Dict[str, Any] | None
    execution_state: Dict[str, Any] | None
    route_context: Dict[str, Any] | None
    last_valid_world_state: Dict[str, Any] | None
    health_flags: List[str] = field(default_factory=list)
    dropped_frames: int = 0
    skipped_updates: int = 0
    state_resets: int = 0
    behavior_override_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
