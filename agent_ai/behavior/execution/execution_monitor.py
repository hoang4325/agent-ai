from __future__ import annotations

import math
from typing import Any, Dict, List

from .execution_contract import ExecutionRequest, ExecutionState
from .local_planner_bridge import PlannerStepResult
from .lane_change_executor import LaneChangeTracker
from .stop_target_runtime import measure_stop_target_distance


def _dedupe(items: List[str]) -> List[str]:
    return sorted(set(item for item in items if item))


class ExecutionMonitor:
    def __init__(self) -> None:
        self.timeline_records: List[Dict[str, Any]] = []
        self.lane_change_events: List[Dict[str, Any]] = []

    def record(
        self,
        *,
        execution_request: ExecutionRequest,
        planner_step: PlannerStepResult,
        snapshot: Any,
        vehicle: Any,
        map_inst: Any,
        lane_change_tracker: LaneChangeTracker | None,
    ) -> Dict[str, Any]:
        location = vehicle.get_location()
        transform = vehicle.get_transform()
        waypoint = map_inst.get_waypoint(location)
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration() if hasattr(vehicle, "get_acceleration") else None
        angular_velocity = vehicle.get_angular_velocity() if hasattr(vehicle, "get_angular_velocity") else None
        control = vehicle.get_control() if hasattr(vehicle, "get_control") else None
        speed_mps = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
        current_lane_id = None if waypoint is None else int(waypoint.lane_id)
        is_in_junction = False if waypoint is None else bool(waypoint.is_junction)
        distance_to_current_lane_center = (
            None if waypoint is None else float(location.distance(waypoint.transform.location))
        )
        obstacle_distance_m = None
        stop_target_distance_m = None
        stop_target_binding_status = None
        stop_target = dict(execution_request.stop_target or {})
        if stop_target:
            stop_target_binding_status = str(stop_target.get("binding_status") or "unavailable")
            stop_target_distance_m, measurement_mode = measure_stop_target_distance(vehicle, stop_target)
            if str(stop_target.get("source_type") or "") == "obstacle":
                obstacle_distance_m = stop_target_distance_m
            if measurement_mode:
                notes_measurement = f"stop_target_measurement={measurement_mode}"
            else:
                notes_measurement = None
        else:
            notes_measurement = None

        notes = list(planner_step.notes)
        if execution_request.blocking_object_id is not None and stop_target_distance_m is None:
            notes.append("blocking_object_id_not_bound_to_live_actor")
        if notes_measurement:
            notes.append(notes_measurement)
        execution_status = planner_step.execution_status
        lane_change_progress = 0.0
        distance_to_target_lane_center = None
        target_lane_id = planner_step.target_lane_id
        lateral_shift_m = 0.0
        if lane_change_tracker is not None:
            execution_status = lane_change_tracker.status
            lane_change_progress = float(lane_change_tracker.progress)
            distance_to_target_lane_center = lane_change_tracker.distance_to_target_lane_center
            target_lane_id = lane_change_tracker.target_lane_id
            lateral_shift_m = float(lane_change_tracker.lateral_shift_m)
            notes.extend(lane_change_tracker.notes)

        state = ExecutionState(
            frame=int(execution_request.frame),
            sample_name=str(execution_request.sample_name),
            requested_behavior=str(execution_request.requested_behavior),
            current_behavior=str(planner_step.executed_behavior),
            execution_status=str(execution_status),
            current_lane_id=current_lane_id,
            target_lane_id=target_lane_id,
            lane_change_progress=float(lane_change_progress),
            speed_mps=float(speed_mps),
            target_speed_mps=float(execution_request.target_speed_mps),
            distance_to_target_lane_center=distance_to_target_lane_center,
            distance_to_current_lane_center=distance_to_current_lane_center,
            lateral_shift_m=float(lateral_shift_m),
            is_in_junction=bool(is_in_junction),
            distance_to_obstacle_m=obstacle_distance_m,
            distance_to_stop_target_m=stop_target_distance_m,
            stop_target_binding_status=stop_target_binding_status,
            notes=_dedupe(notes),
            carla_frame=int(snapshot.frame),
        )
        timeline_record = {
            "request_frame": int(execution_request.frame),
            "sample_name": str(execution_request.sample_name),
            "carla_frame": int(snapshot.frame),
            "timestamp": float(snapshot.timestamp.elapsed_seconds),
            "execution_request": execution_request.to_dict(),
            "execution_state": state.to_dict(),
            "applied_control": (
                None
                if control is None
                else {
                    "throttle": float(getattr(control, "throttle", 0.0)),
                    "steer": float(getattr(control, "steer", 0.0)),
                    "brake": float(getattr(control, "brake", 0.0)),
                    "hand_brake": bool(getattr(control, "hand_brake", False)),
                    "reverse": bool(getattr(control, "reverse", False)),
                    "manual_gear_shift": bool(getattr(control, "manual_gear_shift", False)),
                }
            ),
            "vehicle_pose": {
                "x_m": float(location.x),
                "y_m": float(location.y),
                "z_m": float(location.z),
                "pitch_deg": float(transform.rotation.pitch),
                "yaw_deg": float(transform.rotation.yaw),
                "roll_deg": float(transform.rotation.roll),
            },
            "vehicle_kinematics": {
                "velocity_x_mps": float(velocity.x),
                "velocity_y_mps": float(velocity.y),
                "velocity_z_mps": float(velocity.z),
                "speed_mps": float(speed_mps),
                "acceleration_x_mps2": None if acceleration is None else float(acceleration.x),
                "acceleration_y_mps2": None if acceleration is None else float(acceleration.y),
                "acceleration_z_mps2": None if acceleration is None else float(acceleration.z),
                "angular_velocity_x_dps": None if angular_velocity is None else float(angular_velocity.x),
                "angular_velocity_y_dps": None if angular_velocity is None else float(angular_velocity.y),
                "angular_velocity_z_dps": None if angular_velocity is None else float(angular_velocity.z),
            },
            "planner_diagnostics": {
                "planner_status": str(planner_step.execution_status),
                "planner_notes": _dedupe(list(planner_step.notes)),
                "execution_mode": str(execution_request.execution_mode),
            },
        }
        self.timeline_records.append(timeline_record)
        return timeline_record

    def record_lane_change_start(self, *, snapshot: Any, tracker: LaneChangeTracker) -> Dict[str, Any]:
        payload = tracker.to_event("start", current_lane_id=tracker.start_lane_id)
        payload["carla_frame"] = int(snapshot.frame)
        payload["timestamp"] = float(snapshot.timestamp.elapsed_seconds)
        self.lane_change_events.append(payload)
        return payload

    def record_lane_change_terminal(
        self,
        *,
        snapshot: Any,
        tracker: LaneChangeTracker,
        current_lane_id: int | None,
    ) -> Dict[str, Any]:
        payload = tracker.to_event(str(tracker.status), current_lane_id=current_lane_id)
        payload["carla_frame"] = int(snapshot.frame)
        payload["timestamp"] = float(snapshot.timestamp.elapsed_seconds)
        self.lane_change_events.append(payload)
        return payload
