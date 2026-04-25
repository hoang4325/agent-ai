from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from stage3c_coverage.lane_change_completion import (
    LaneChangeCompletionCriteria,
    evaluate_lane_change_step,
)


@dataclass
class LaneChangeTracker:
    request_frame: int
    sample_name: str
    requested_behavior: str
    direction: str
    start_carla_frame: int
    start_timestamp: float
    start_lane_id: int | None
    target_lane_id: int | None
    lane_width_m: float
    start_location_xyz: List[float]
    start_right_vector_xyz: List[float]
    completion_criteria: LaneChangeCompletionCriteria
    status: str = "executing"
    elapsed_ticks: int = 0
    stable_ticks: int = 0
    progress: float = 0.0
    distance_to_target_lane_center: float | None = None
    distance_to_current_lane_center: float | None = None
    lateral_shift_m: float = 0.0
    max_lateral_shift_m: float = 0.0
    required_lateral_shift_m: float = 0.0
    completion_time_s: float | None = None
    completion_reason: str | None = None
    notes: List[str] = field(default_factory=list)

    def abort(self, reason: str) -> None:
        self.status = "aborted"
        self.completion_reason = str(reason or "aborted")
        if reason:
            self.notes.append(str(reason))

    def update(self, *, vehicle: Any, map_inst: Any, snapshot: Any) -> None:
        location = vehicle.get_location()
        waypoint = map_inst.get_waypoint(location)
        if waypoint is None:
            self.elapsed_ticks += 1
            self.notes.append("waypoint_lookup_failed")
            if self.elapsed_ticks >= self.completion_criteria.max_duration_ticks:
                self.status = "fail"
                self.completion_reason = "waypoint_lookup_timeout"
            return

        dx = float(location.x) - float(self.start_location_xyz[0])
        dy = float(location.y) - float(self.start_location_xyz[1])
        dz = float(location.z) - float(self.start_location_xyz[2])
        lateral_axis = self.start_right_vector_xyz
        self.lateral_shift_m = dx * lateral_axis[0] + dy * lateral_axis[1] + dz * lateral_axis[2]
        self.max_lateral_shift_m = max(float(self.max_lateral_shift_m), abs(float(self.lateral_shift_m)))
        self.distance_to_current_lane_center = float(location.distance(waypoint.transform.location))

        target_wp = None
        if self.target_lane_id is not None and waypoint.lane_id != self.target_lane_id:
            if self.direction == "right":
                target_wp = waypoint.get_right_lane()
            elif self.direction == "left":
                target_wp = waypoint.get_left_lane()
            if target_wp is not None:
                self.distance_to_target_lane_center = float(location.distance(target_wp.transform.location))
        else:
            self.distance_to_target_lane_center = float(location.distance(waypoint.transform.location))

        completion = evaluate_lane_change_step(
            current_lane_id=int(waypoint.lane_id),
            target_lane_id=self.target_lane_id,
            lateral_shift_m=float(self.lateral_shift_m),
            lane_width_m=float(self.lane_width_m),
            stable_ticks=int(self.stable_ticks),
            criteria=self.completion_criteria,
        )
        self.required_lateral_shift_m = float(completion["required_lateral_shift_m"])
        self.progress = float(completion["progress"])
        previous_stable_ticks = int(self.stable_ticks)
        self.stable_ticks = int(completion["stable_ticks"])
        if completion["reason"]:
            if completion["reason"] not in self.notes:
                self.notes.append(str(completion["reason"]))
        if previous_stable_ticks > 0 and self.stable_ticks == 0 and self.status == "executing":
            self.notes.append("target_lane_reverted_before_completion")

        self.elapsed_ticks += 1
        if bool(completion["success"]):
            self.status = "success"
            self.progress = 1.0
            self.completion_reason = "stable_target_lane_reached"
            if self.completion_time_s is None:
                self.completion_time_s = float(snapshot.timestamp.elapsed_seconds) - float(self.start_timestamp)

        if self.status == "executing" and self.elapsed_ticks >= self.completion_criteria.max_duration_ticks:
            self.status = "fail"
            self.completion_reason = "lane_change_timeout"
            self.notes.append("lane_change_timeout")

    def to_event(self, event_type: str, *, current_lane_id: int | None = None) -> Dict[str, Any]:
        return {
            "event_type": event_type,
            "sample_name": self.sample_name,
            "request_frame": int(self.request_frame),
            "requested_behavior": self.requested_behavior,
            "direction": self.direction,
            "start_lane_id": self.start_lane_id,
            "target_lane_id": self.target_lane_id,
            "current_lane_id": current_lane_id,
            "status": self.status,
            "lane_change_progress": float(self.progress),
            "lateral_shift_m": float(self.lateral_shift_m),
            "max_lateral_shift_m": float(self.max_lateral_shift_m),
            "required_lateral_shift_m": float(self.required_lateral_shift_m),
            "stable_ticks": int(self.stable_ticks),
            "success_hold_ticks": int(self.completion_criteria.success_hold_ticks),
            "completion_time_s": self.completion_time_s,
            "completion_reason": self.completion_reason,
            "notes": list(self.notes),
        }
