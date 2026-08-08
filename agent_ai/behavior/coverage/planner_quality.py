from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from .evaluation_stage3c_coverage import (
    derive_lane_change_events,
    derive_stop_events,
    load_json,
    load_jsonl,
)
from .stop_controller_tuner import StopControllerConfig


LANE_CHANGE_BEHAVIORS = {
    "prepare_lane_change_left",
    "prepare_lane_change_right",
    "commit_lane_change_left",
    "commit_lane_change_right",
}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(mean(cleaned))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {value.__class__.__name__}")


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=_json_default)
        handle.write("\n")


def _dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=_json_default))
            handle.write("\n")


def _case_id_from_output_dir(output_dir: str | Path) -> str | None:
    output_dir = Path(output_dir)
    parent = output_dir.parent
    if parent.name and parent.name not in {"case_runs", "stage3c", "stage4"}:
        return parent.name
    return None


def _normalize_angle_delta_deg(current_yaw_deg: float, previous_yaw_deg: float) -> float:
    delta = float(current_yaw_deg) - float(previous_yaw_deg)
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return float(delta)


def _event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("sample_name"),
        event.get("request_frame"),
        event.get("direction"),
        event.get("start_lane_id"),
        event.get("target_lane_id"),
    )


def _find_last_event_frame(event: dict[str, Any], rows: list[dict[str, Any]]) -> int | None:
    start_frame = _safe_int(event.get("start_frame"))
    end_frame = _safe_int(event.get("end_frame"))
    if end_frame is not None:
        return end_frame
    request_frame = _safe_int(event.get("request_frame"))
    sample_name = str(event.get("sample_name", ""))
    matching = [
        _safe_int(row.get("carla_frame"))
        for row in rows
        if _safe_int(row.get("request_frame")) == request_frame or str(row.get("sample_name", "")) == sample_name
    ]
    matching = [value for value in matching if value is not None and (start_frame is None or value >= start_frame)]
    return None if not matching else max(matching)


def derive_trajectory_trace(execution_records: Iterable[Dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(execution_records)
    trace: list[dict[str, Any]] = []
    previous_timestamp = None
    previous_yaw = None
    for row in rows:
        execution_request = dict(row.get("execution_request") or {})
        execution_state = dict(row.get("execution_state") or {})
        vehicle_pose = dict(row.get("vehicle_pose") or {})
        planner_diagnostics = dict(row.get("planner_diagnostics") or {})
        timestamp = _safe_float(row.get("timestamp"))
        yaw_deg = _safe_float(vehicle_pose.get("yaw_deg"))
        entry = {
            "request_frame": _safe_int(row.get("request_frame")),
            "sample_name": str(row.get("sample_name", "")),
            "carla_frame": _safe_int(row.get("carla_frame")),
            "timestamp": timestamp,
            "requested_behavior": str(execution_request.get("requested_behavior", "")),
            "executing_behavior": str(execution_state.get("current_behavior", "")),
            "execution_status": str(execution_state.get("execution_status", "")),
            "current_lane_id": execution_state.get("current_lane_id"),
            "target_lane_id": execution_state.get("target_lane_id"),
            "lane_change_progress": _safe_float(execution_state.get("lane_change_progress")),
            "distance_to_target_lane_center": _safe_float(execution_state.get("distance_to_target_lane_center")),
            "distance_to_current_lane_center": _safe_float(execution_state.get("distance_to_current_lane_center")),
            "lateral_shift_m": _safe_float(execution_state.get("lateral_shift_m")),
            "speed_mps": _safe_float(execution_state.get("speed_mps")),
            "target_speed_mps": _safe_float(execution_state.get("target_speed_mps")),
            "is_in_junction": bool(execution_state.get("is_in_junction", False)),
            "trajectory_source": str(execution_request.get("execution_mode", "local_planner_bridge")),
            "planner_status": str(planner_diagnostics.get("planner_status", execution_state.get("execution_status", ""))),
            "pose_supported": bool(vehicle_pose),
            "x_m": _safe_float(vehicle_pose.get("x_m")),
            "y_m": _safe_float(vehicle_pose.get("y_m")),
            "z_m": _safe_float(vehicle_pose.get("z_m")),
            "yaw_deg": yaw_deg,
            "pitch_deg": _safe_float(vehicle_pose.get("pitch_deg")),
            "roll_deg": _safe_float(vehicle_pose.get("roll_deg")),
            "notes": list(execution_state.get("notes") or []),
        }
        if (
            timestamp is not None
            and previous_timestamp is not None
            and yaw_deg is not None
            and previous_yaw is not None
            and timestamp > previous_timestamp
        ):
            delta = _normalize_angle_delta_deg(yaw_deg, previous_yaw)
            entry["yaw_rate_deg_per_s"] = float(delta / (timestamp - previous_timestamp))
        else:
            entry["yaw_rate_deg_per_s"] = None
        trace.append(entry)
        previous_timestamp = timestamp
        previous_yaw = yaw_deg
    return trace


def derive_control_trace(execution_records: Iterable[Dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(execution_records)
    trace: list[dict[str, Any]] = []
    previous_timestamp = None
    previous_speed = None
    previous_accel = None
    for row in rows:
        execution_request = dict(row.get("execution_request") or {})
        execution_state = dict(row.get("execution_state") or {})
        control = dict(row.get("applied_control") or {})
        timestamp = _safe_float(row.get("timestamp"))
        speed_mps = _safe_float(execution_state.get("speed_mps"))
        acceleration_mps2 = None
        jerk_mps3 = None
        if (
            timestamp is not None
            and previous_timestamp is not None
            and speed_mps is not None
            and previous_speed is not None
            and timestamp > previous_timestamp
        ):
            dt = float(timestamp - previous_timestamp)
            acceleration_mps2 = float((speed_mps - previous_speed) / dt)
            if previous_accel is not None:
                jerk_mps3 = float((acceleration_mps2 - previous_accel) / dt)
        entry = {
            "request_frame": _safe_int(row.get("request_frame")),
            "sample_name": str(row.get("sample_name", "")),
            "carla_frame": _safe_int(row.get("carla_frame")),
            "timestamp": timestamp,
            "requested_behavior": str(execution_request.get("requested_behavior", "")),
            "executing_behavior": str(execution_state.get("current_behavior", "")),
            "execution_status": str(execution_state.get("execution_status", "")),
            "control_mode": str(execution_request.get("execution_mode", "local_planner_bridge")),
            "speed_mps": speed_mps,
            "target_speed_mps": _safe_float(execution_state.get("target_speed_mps")),
            "speed_error_mps": None if speed_mps is None else float(speed_mps - float(execution_state.get("target_speed_mps", 0.0))),
            "throttle": _safe_float(control.get("throttle")),
            "steer": _safe_float(control.get("steer")),
            "brake": _safe_float(control.get("brake")),
            "hand_brake": bool(control.get("hand_brake", False)) if control else False,
            "reverse": bool(control.get("reverse", False)) if control else False,
            "acceleration_proxy_mps2": acceleration_mps2,
            "jerk_proxy_mps3": jerk_mps3,
            "support_level": "direct" if control else "approximate_from_speed_profile",
        }
        trace.append(entry)
        previous_timestamp = timestamp
        previous_speed = speed_mps
        previous_accel = acceleration_mps2
    return trace


def derive_lane_change_phase_trace(
    execution_records: Iterable[Dict[str, Any]],
    lane_change_events: Iterable[Dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(execution_records)
    events = list(lane_change_events)
    if not rows or not events:
        return []

    phase_rows: list[dict[str, Any]] = []
    rows_by_frame = {
        _safe_int(row.get("carla_frame")): row
        for row in rows
        if _safe_int(row.get("carla_frame")) is not None
    }

    for episode_index, event in enumerate(events, start=1):
        start_frame = _safe_int(event.get("start_frame"))
        end_frame = _find_last_event_frame(event, rows)
        event_rows = [
            row
            for row in rows
            if (
                (_safe_int(row.get("carla_frame")) is not None)
                and (start_frame is None or int(row["carla_frame"]) >= start_frame)
                and (end_frame is None or int(row["carla_frame"]) <= end_frame)
                and (
                    str((row.get("execution_request") or {}).get("requested_behavior", "")) in LANE_CHANGE_BEHAVIORS
                    or str((row.get("execution_state") or {}).get("current_behavior", "")) in LANE_CHANGE_BEHAVIORS
                    or float((row.get("execution_state") or {}).get("lane_change_progress", 0.0) or 0.0) > 0.0
                )
            )
        ]
        if not event_rows:
            event_rows = [
                row
                for row in rows
                if _safe_int(row.get("request_frame")) == _safe_int(event.get("request_frame"))
            ]

        last_frame = None if not event_rows else _safe_int(event_rows[-1].get("carla_frame"))
        for row in event_rows:
            request = dict(row.get("execution_request") or {})
            state = dict(row.get("execution_state") or {})
            requested_behavior = str(request.get("requested_behavior", ""))
            progress = float(state.get("lane_change_progress", 0.0) or 0.0)
            current_lane_id = state.get("current_lane_id")
            target_lane_id = state.get("target_lane_id")
            execution_status = str(state.get("execution_status", ""))

            if requested_behavior.startswith("prepare_lane_change_"):
                phase = "prepare"
            elif last_frame is not None and _safe_int(row.get("carla_frame")) == last_frame:
                phase = "completed" if str(event.get("result")) == "success" else "aborted"
            elif target_lane_id is not None and current_lane_id == target_lane_id and progress >= 1.0:
                phase = "stabilizing"
            else:
                phase = "commit"

            phase_rows.append(
                {
                    "episode_id": f"lc_episode_{episode_index:03d}",
                    "request_frame": _safe_int(row.get("request_frame")),
                    "sample_name": str(row.get("sample_name", "")),
                    "carla_frame": _safe_int(row.get("carla_frame")),
                    "timestamp": _safe_float(row.get("timestamp")),
                    "direction": str(event.get("direction", "unknown")),
                    "requested_behavior": requested_behavior,
                    "executing_behavior": str(state.get("current_behavior", "")),
                    "phase": phase,
                    "current_lane_id": current_lane_id,
                    "target_lane_id": target_lane_id,
                    "lane_change_progress": progress,
                    "execution_status": execution_status,
                    "terminal_result": event.get("result") if phase in {"completed", "aborted"} else None,
                    "abort_reason": event.get("reason") if phase == "aborted" else None,
                    "completion_reason": event.get("reason") if phase == "completed" else None,
                }
            )

        if not event_rows:
            frame_row = rows_by_frame.get(start_frame) if start_frame is not None else None
            phase_rows.append(
                {
                    "episode_id": f"lc_episode_{episode_index:03d}",
                    "request_frame": _safe_int(event.get("request_frame")),
                    "sample_name": str(event.get("sample_name", "")),
                    "carla_frame": start_frame,
                    "timestamp": _safe_float((frame_row or {}).get("timestamp")),
                    "direction": str(event.get("direction", "unknown")),
                    "requested_behavior": str(event.get("requested_behavior", "")),
                    "executing_behavior": str(((frame_row or {}).get("execution_state") or {}).get("current_behavior", "")),
                    "phase": "completed" if str(event.get("result")) == "success" else "aborted",
                    "current_lane_id": event.get("current_lane_id"),
                    "target_lane_id": event.get("target_lane_id"),
                    "lane_change_progress": _safe_float(event.get("lane_change_progress")),
                    "execution_status": str(event.get("status", "")),
                    "terminal_result": event.get("result"),
                    "abort_reason": event.get("reason") if str(event.get("result")) != "success" else None,
                    "completion_reason": event.get("reason") if str(event.get("result")) == "success" else None,
                }
            )
    return phase_rows


def derive_stop_target(
    execution_records: Iterable[Dict[str, Any]],
    stop_events: Iterable[Dict[str, Any]],
    *,
    config: StopControllerConfig | None = None,
) -> dict[str, Any]:
    config = config or StopControllerConfig()
    rows = list(execution_records)
    events = list(stop_events)
    stop_rows = []
    for index, event in enumerate(events, start=1):
        request_frame = _safe_int(event.get("request_frame"))
        window = [
            row
            for row in rows
            if request_frame is not None and _safe_int(row.get("request_frame")) == request_frame
        ]
        first_request = dict((window[0].get("execution_request") or {})) if window else {}
        first_state = dict((window[0].get("execution_state") or {})) if window else {}
        stop_reason = str(first_request.get("stop_reason") or event.get("reason") or "")
        stop_target_request = dict(first_request.get("stop_target") or {})
        target_source_type = str(stop_target_request.get("source_type") or "")
        if not target_source_type:
            if "fallback" in stop_reason:
                target_source_type = "fallback_safety"
            elif "stop_line" in stop_reason:
                target_source_type = "stop_line"
            elif "route" in stop_reason or "junction" in stop_reason:
                target_source_type = "route_junction"
            else:
                target_source_type = "unavailable"
        binding_status = str(stop_target_request.get("binding_status") or "unavailable")
        target_source_detail = str(
            stop_target_request.get("source_detail")
            or stop_target_request.get("target_source")
            or stop_reason
            or "stop_target_unavailable"
        )
        target_distance = _safe_float(stop_target_request.get("target_distance_m"))
        target_confidence = _safe_float(stop_target_request.get("target_confidence"))
        if target_confidence is None:
            target_confidence = 0.0 if target_source_type == "unavailable" else 0.55

        target_distances = [
            _safe_float(((row.get("execution_state") or {}).get("distance_to_stop_target_m")))
            for row in window
        ]
        obstacle_distances = [
            _safe_float(((row.get("execution_state") or {}).get("distance_to_obstacle_m")))
            for row in window
        ]
        target_distances = [value for value in target_distances if value is not None]
        obstacle_distances = [value for value in obstacle_distances if value is not None]
        measured_distances = target_distances or obstacle_distances
        first_distance = None if not measured_distances else float(measured_distances[0])
        min_distance = None if not measured_distances else float(min(measured_distances))
        final_distance = None if not measured_distances else float(measured_distances[-1])

        provenance = dict(stop_target_request.get("provenance") or {})
        derivation_mode = str(provenance.get("derivation_mode") or binding_status or "missing")
        runtime_binding_source = provenance.get("runtime_binding_source")
        actor_binding_source = provenance.get("actor_binding_source")
        materialization_status = provenance.get("materialization_status")
        runtime_actor_id = _safe_int(provenance.get("runtime_actor_id"))
        if final_distance is not None:
            if target_source_type == "fallback_safety":
                measurement_mode = "explicit_fallback_stop"
            elif runtime_binding_source and binding_status == "exact":
                measurement_mode = f"exact_runtime_bound:{runtime_binding_source}"
            elif runtime_binding_source:
                measurement_mode = f"runtime_bound:{runtime_binding_source}"
            elif target_source_type in {"obstacle", "stop_line", "route_junction"}:
                measurement_mode = f"{binding_status}_bound_target_distance"
            else:
                measurement_mode = derivation_mode
        else:
            if runtime_binding_source:
                measurement_mode = f"runtime_binding_without_distance:{runtime_binding_source}"
            else:
                measurement_mode = derivation_mode if target_source_type != "unavailable" else "missing"

        if target_source_type == "unavailable" and target_distance is None and measured_distances:
            target_source_type = "obstacle"
            target_source_detail = "distance_to_obstacle_m_proxy"
            target_distance = float(config.stop_success_distance_m)
            target_confidence = 0.55
            binding_status = "approximate"
            measurement_mode = "approximate_configured_standoff"

        initial_distance_to_target = _safe_float(stop_target_request.get("initial_distance_to_target_m"))
        runtime_alignment_error = (
            None
            if initial_distance_to_target is None or first_distance is None
            else float(abs(first_distance - initial_distance_to_target))
        )
        runtime_alignment_consistent = bool(
            runtime_alignment_error is not None and runtime_alignment_error <= 10.0
        )

        target_defined = (
            target_source_type != "unavailable"
            and (
                target_distance is not None
                or stop_target_request.get("target_pose") is not None
                or stop_target_request.get("source_actor_id") is not None
                or stop_target_request.get("source_track_id") is not None
            )
        )
        stop_final_error_supported = target_defined and target_distance is not None and final_distance is not None
        stop_final_error = (
            None
            if not stop_final_error_supported
            else float(abs(float(final_distance) - float(target_distance)))
        )
        overshoot_distance_supported = bool(measured_distances)
        overshoot_distance = (
            None
            if not overshoot_distance_supported or min_distance is None
            else float(
                max(
                    0.0,
                    (float(target_distance) if target_distance is not None else float(config.overshoot_distance_m))
                    - float(min_distance),
                )
            )
        )
        stop_rows.append(
            {
                "stop_event_id": f"stop_event_{index:03d}",
                "request_frame": request_frame,
                "sample_name": str(event.get("sample_name", "")),
                "result": str(event.get("result", "")),
                "tick_id": _safe_int(event.get("tick_id")),
                "target_source": target_source_detail,
                "target_source_type": target_source_type,
                "binding_status": binding_status,
                "target_defined": bool(target_defined),
                "target_confidence": float(target_confidence),
                "measurement_mode": measurement_mode,
                "distance_measurement_supported": bool(measured_distances),
                "obstacle_binding_status": (
                    "runtime_exact_actor_bound"
                    if target_source_type == "obstacle" and runtime_binding_source and binding_status == "exact"
                    else (
                        "runtime_bound"
                        if target_source_type == "obstacle" and runtime_binding_source
                        else (
                            "track_bound"
                            if target_source_type == "obstacle" and stop_target_request.get("source_track_id") is not None
                            else ("explicit_fallback" if target_source_type == "fallback_safety" else "unbound")
                        )
                    )
                ),
                "obstacle_binding_id": first_request.get("blocking_object_id"),
                "source_track_id": _safe_int(stop_target_request.get("source_track_id")),
                "source_actor_id": _safe_int(stop_target_request.get("source_actor_id")) or runtime_actor_id,
                "source_role": stop_target_request.get("source_role"),
                "target_distance_m": target_distance,
                "initial_distance_to_target_m": initial_distance_to_target,
                "first_distance_to_target_m": first_distance,
                "runtime_alignment_error_m": runtime_alignment_error,
                "runtime_alignment_consistent": runtime_alignment_consistent,
                "target_pose": stop_target_request.get("target_pose"),
                "target_lane_id": (
                    stop_target_request.get("target_lane_id")
                    if stop_target_request.get("target_lane_id") is not None
                    else first_state.get("current_lane_id")
                ),
                "target_lane_relation": (
                    stop_target_request.get("target_lane_relation")
                    or ("current_lane" if first_state.get("current_lane_id") is not None else None)
                ),
                "stop_reason": stop_reason or str(event.get("reason", "")),
                "final_distance_to_obstacle_m": (
                    None if not obstacle_distances else float(obstacle_distances[-1])
                ),
                "min_distance_to_obstacle_m": min_distance,
                "final_distance_to_target_m": final_distance,
                "min_distance_to_target_m": min_distance,
                "stop_final_error_supported": bool(stop_final_error_supported),
                "stop_final_error_m": stop_final_error,
                "overshoot_distance_supported": bool(overshoot_distance_supported),
                "stop_overshoot_distance_m": overshoot_distance,
                "hold_ticks": _safe_int(event.get("hold_ticks")),
                "reason": str(event.get("reason", "")),
                "provenance": {
                    "source_stage": "stage3c",
                    "source_artifact": "execution_timeline.jsonl",
                    "derivation_mode": derivation_mode,
                    "runtime_binding_source": runtime_binding_source,
                    "actor_binding_source": actor_binding_source,
                    "materialization_status": materialization_status,
                    "runtime_actor_id": runtime_actor_id,
                },
            }
        )

    supported_stop_rows = [row for row in stop_rows if row.get("distance_measurement_supported")]
    target_defined_rows = [row for row in stop_rows if row.get("target_defined")]
    stop_final_rows = [row for row in stop_rows if row.get("stop_final_error_supported")]
    explicit_rows = [row for row in stop_rows if str(row.get("binding_status")) == "exact"]
    heuristic_rows = [row for row in stop_rows if str(row.get("binding_status")) == "heuristic"]
    approximate_rows = [row for row in stop_rows if str(row.get("binding_status")) == "approximate"]
    stop_final_support = (
        "present"
        if stop_final_rows and any(str(row.get("binding_status")) == "exact" for row in stop_final_rows)
        else ("approximate" if stop_final_rows else "missing")
    )
    return {
        "schema_version": "stage6.stop_target.v1",
        "measurement_support": {
            "stop_target_defined_rate": "present" if stop_rows else "missing",
            "stop_final_error_m": stop_final_support,
            "stop_overshoot_distance_m": (
                "present"
                if supported_stop_rows and any(str(row.get("binding_status")) == "exact" for row in supported_stop_rows)
                else ("approximate" if supported_stop_rows else "missing")
            ),
        },
        "summary": {
            "num_stop_events": len(stop_rows),
            "stop_target_defined_events": len(target_defined_rows),
            "stop_target_defined_rate": _safe_div(float(len(target_defined_rows)), float(len(stop_rows))),
            "explicit_target_events": len(explicit_rows),
            "heuristic_target_events": len(heuristic_rows),
            "approximate_target_events": len(approximate_rows),
            "unavailable_target_events": sum(1 for row in stop_rows if str(row.get("binding_status")) == "unavailable"),
            "distance_measurement_supported_events": len(supported_stop_rows),
            "stop_final_error_supported": bool(stop_final_rows),
            "stop_final_error_supported_events": len(stop_final_rows),
            "stop_overshoot_distance_supported": bool(supported_stop_rows),
            "stop_overshoot_distance_supported_events": len(supported_stop_rows),
            "mean_stop_final_error_m": _mean_or_none(
                row.get("stop_final_error_m") for row in stop_final_rows
            ),
            "mean_stop_overshoot_distance_m": _mean_or_none(
                row.get("stop_overshoot_distance_m") for row in supported_stop_rows
            ),
            "alignment_consistent_events": sum(1 for row in stop_rows if bool(row.get("runtime_alignment_consistent"))),
            "mean_runtime_alignment_error_m": _mean_or_none(
                row.get("runtime_alignment_error_m") for row in stop_rows
            ),
        },
        "stop_events": stop_rows,
    }


def derive_stop_target_trace(
    execution_records: Iterable[Dict[str, Any]],
    *,
    case_id: str | None,
) -> list[dict[str, Any]]:
    trace_rows: list[dict[str, Any]] = []
    for row in execution_records:
        execution_request = dict(row.get("execution_request") or {})
        stop_target = dict(execution_request.get("stop_target") or {})
        requested_behavior = str(execution_request.get("requested_behavior") or "")
        if not stop_target and requested_behavior not in {"stop_before_obstacle", "yield"}:
            continue
        provenance = dict(stop_target.get("provenance") or {})
        execution_state = dict(row.get("execution_state") or {})
        trace_rows.append(
            {
                "tick_id": _safe_int(row.get("tick_id")),
                "case_id": case_id,
                "sample_name": str(row.get("sample_name") or ""),
                "carla_frame": _safe_int(row.get("carla_frame")),
                "timestamp": _safe_float(row.get("timestamp")),
                "requested_behavior": requested_behavior,
                "target_source_type": str(stop_target.get("source_type") or "unavailable"),
                "target_actor_id": _safe_int(stop_target.get("source_actor_id")),
                "target_lane_id": _safe_int(stop_target.get("target_lane_id")),
                "target_distance_m": _safe_float(stop_target.get("target_distance_m")),
                "target_confidence": _safe_float(stop_target.get("target_confidence")),
                "target_lane_relation": stop_target.get("target_lane_relation"),
                "binding_status": str(stop_target.get("binding_status") or "unavailable"),
                "distance_to_stop_target_m": _safe_float(execution_state.get("distance_to_stop_target_m")),
                "distance_to_obstacle_m": _safe_float(execution_state.get("distance_to_obstacle_m")),
                "provenance": {
                    "runtime_binding_source": provenance.get("runtime_binding_source"),
                    "actor_binding_source": provenance.get("actor_binding_source"),
                    "materialization_status": provenance.get("materialization_status"),
                    "runtime_actor_id": provenance.get("runtime_actor_id"),
                },
                "notes": list(execution_state.get("notes") or []),
            }
        )
    return trace_rows


def derive_stop_binding_summary(stop_target: dict[str, Any], *, case_id: str | None) -> dict[str, Any]:
    summary = dict(stop_target.get("summary") or {})
    stop_events = list(stop_target.get("stop_events") or [])
    mean_alignment_error = summary.get("mean_runtime_alignment_error_m")
    alignment_consistent = bool((summary.get("alignment_consistent_events") or 0) == len(stop_events) and stop_events)
    notes = []
    if bool(summary.get("stop_final_error_supported")):
        notes.append("stop_final_error_m_supported")
    else:
        notes.append("stop_final_error_m_missing")
    if alignment_consistent:
        notes.append("runtime_stop_alignment_consistent")
    elif mean_alignment_error is not None:
        notes.append("runtime_stop_alignment_mismatch")
    return {
        "case_id": case_id,
        "defined_rate": summary.get("stop_target_defined_rate"),
        "exact_rate": _safe_div(
            float(sum(1 for row in stop_events if str(row.get("binding_status")) == "exact")),
            float(len(stop_events)),
        ),
        "heuristic_rate": _safe_div(
            float(sum(1 for row in stop_events if str(row.get("binding_status")) == "heuristic")),
            float(len(stop_events)),
        ),
        "missing_rate": _safe_div(
            float(sum(1 for row in stop_events if str(row.get("binding_status")) in {"unavailable", "missing"})),
            float(len(stop_events)),
        ),
        "stop_final_error_available": bool(summary.get("stop_final_error_supported")),
        "mean_runtime_alignment_error_m": mean_alignment_error,
        "alignment_consistent": alignment_consistent,
        "notes": notes,
    }


def derive_planner_control_latency(
    execution_records: Iterable[Dict[str, Any]],
    *,
    online_tick_records: Iterable[Dict[str, Any]] | None = None,
    total_loop_budget_ms: float = 150.0,
    source_stage: str = "stage3c_replay",
) -> list[dict[str, Any]]:
    execution_rows = list(execution_records)
    online_rows = list(online_tick_records or [])
    execution_by_frame = {
        _safe_int(row.get("carla_frame")): row
        for row in execution_rows
        if _safe_int(row.get("carla_frame")) is not None
    }

    rows: list[dict[str, Any]] = []
    if online_rows:
        for tick in online_rows:
            frame_id = _safe_int(tick.get("frame_id"))
            execution_row = execution_by_frame.get(frame_id, {})
            planner_prepare_latency_ms = _safe_float(tick.get("planner_prepare_latency_ms"))
            execution_finalize_latency_ms = _safe_float(tick.get("execution_finalize_latency_ms"))
            planner_execution_latency_ms = _safe_float(tick.get("planner_execution_latency_ms"))
            if planner_execution_latency_ms is None:
                planner_execution_latency_ms = _safe_float(tick.get("execution_latency_ms"))
            decision_to_execution_latency_ms = _safe_float(tick.get("decision_to_execution_latency_ms"))
            if decision_to_execution_latency_ms is None and planner_execution_latency_ms is not None:
                decision_to_execution_latency_ms = float(
                    (_safe_float(tick.get("world_update_latency_ms")) or 0.0)
                    + (_safe_float(tick.get("behavior_latency_ms")) or 0.0)
                    + planner_execution_latency_ms
                )
            total_loop_latency_ms = _safe_float(tick.get("total_loop_latency_ms"))
            if total_loop_latency_ms is None:
                components = [
                    _safe_float(tick.get("perception_latency_ms")),
                    _safe_float(tick.get("world_update_latency_ms")),
                    _safe_float(tick.get("behavior_latency_ms")),
                    decision_to_execution_latency_ms,
                ]
                if any(value is not None for value in components):
                    total_loop_latency_ms = float(sum(value or 0.0 for value in components))
            over_budget = None if total_loop_latency_ms is None else bool(total_loop_latency_ms > float(total_loop_budget_ms))
            rows.append(
                {
                    "tick_id": _safe_int(tick.get("tick_id")),
                    "request_frame": _safe_int(execution_row.get("request_frame")),
                    "sample_name": str(execution_row.get("sample_name") or ""),
                    "carla_frame": frame_id,
                    "timestamp": _safe_float(tick.get("timestamp")),
                    "source_stage": "stage4_online",
                    "measurement_mode": "direct_online_tick_contract",
                    "perception_latency_ms": _safe_float(tick.get("perception_latency_ms")),
                    "world_update_latency_ms": _safe_float(tick.get("world_update_latency_ms")),
                    "behavior_latency_ms": _safe_float(tick.get("behavior_latency_ms")),
                    "planner_prepare_latency_ms": planner_prepare_latency_ms,
                    "execution_finalize_latency_ms": execution_finalize_latency_ms,
                    "planner_execution_latency_ms": planner_execution_latency_ms,
                    "decision_to_execution_latency_ms": decision_to_execution_latency_ms,
                    "total_loop_latency_ms": total_loop_latency_ms,
                    "latency_budget_ms": float(total_loop_budget_ms),
                    "over_budget": over_budget,
                    "over_budget_component": "total_loop_latency_ms" if over_budget else None,
                    "provenance": {
                        "source_artifact": "online_tick_record.jsonl",
                        "source_stage": "stage4",
                    },
                }
            )
        return rows

    for row in execution_rows:
        planner_diag = dict(row.get("planner_diagnostics") or {})
        planner_prepare_latency_ms = _safe_float(planner_diag.get("planner_prepare_latency_ms"))
        execution_finalize_latency_ms = _safe_float(planner_diag.get("execution_finalize_latency_ms"))
        planner_execution_latency_ms = _safe_float(planner_diag.get("planner_execution_latency_ms"))
        if planner_execution_latency_ms is None and (
            planner_prepare_latency_ms is not None or execution_finalize_latency_ms is not None
        ):
            planner_execution_latency_ms = float(
                (planner_prepare_latency_ms or 0.0) + (execution_finalize_latency_ms or 0.0)
            )
        over_budget = None if planner_execution_latency_ms is None else bool(planner_execution_latency_ms > float(total_loop_budget_ms))
        rows.append(
            {
                "tick_id": None,
                "request_frame": _safe_int(row.get("request_frame")),
                "sample_name": str(row.get("sample_name") or ""),
                "carla_frame": _safe_int(row.get("carla_frame")),
                "timestamp": _safe_float(row.get("timestamp")),
                "source_stage": source_stage,
                "measurement_mode": (
                    "direct_replay_planner_timing" if planner_execution_latency_ms is not None else "missing"
                ),
                "perception_latency_ms": None,
                "world_update_latency_ms": None,
                "behavior_latency_ms": None,
                "planner_prepare_latency_ms": planner_prepare_latency_ms,
                "execution_finalize_latency_ms": execution_finalize_latency_ms,
                "planner_execution_latency_ms": planner_execution_latency_ms,
                "decision_to_execution_latency_ms": planner_execution_latency_ms,
                "total_loop_latency_ms": planner_execution_latency_ms,
                "latency_budget_ms": float(total_loop_budget_ms),
                "over_budget": over_budget,
                "over_budget_component": "planner_execution_latency_ms" if over_budget else None,
                "provenance": {
                    "source_artifact": "execution_timeline.jsonl",
                    "source_stage": source_stage,
                },
            }
        )
    return rows


def derive_fallback_events(
    execution_records: Iterable[Dict[str, Any]],
    *,
    planner_control_latency: Iterable[Dict[str, Any]] | None = None,
    online_tick_records: Iterable[Dict[str, Any]] | None = None,
    source_stage: str = "stage3c_replay",
) -> list[dict[str, Any]]:
    execution_rows = list(execution_records)
    latency_rows = list(planner_control_latency or [])
    online_rows = list(online_tick_records or [])
    latency_by_frame = {
        _safe_int(row.get("carla_frame")): row
        for row in latency_rows
        if _safe_int(row.get("carla_frame")) is not None
    }
    execution_by_frame = {
        _safe_int(row.get("carla_frame")): row
        for row in execution_rows
        if _safe_int(row.get("carla_frame")) is not None
    }

    if online_rows:
        events: list[dict[str, Any]] = []
        for tick in online_rows:
            frame_id = _safe_int(tick.get("frame_id"))
            execution_row = execution_by_frame.get(frame_id, {})
            arbitration_action = str(tick.get("arbitration_action") or "")
            health_flags = [str(flag) for flag in (tick.get("health_flags") or [])]
            notes = [str(note) for note in (tick.get("notes") or [])]
            fallback_required = bool(
                arbitration_action in {"emergency_override", "downgrade", "cancel"}
                or any(flag in {"stage1_worker_failed", "no_behavior_state_yet", "perception_hard_stale"} for flag in health_flags)
            )
            fallback_activated = bool(arbitration_action in {"emergency_override", "downgrade", "cancel"})
            reason_parts = []
            if arbitration_action in {"emergency_override", "downgrade", "cancel"}:
                reason_parts.append(arbitration_action)
            reason_parts.extend(health_flags)
            reason_parts.extend(notes)
            mode = {
                "emergency_override": "deterministic_emergency_stop",
                "downgrade": "deterministic_speed_downgrade",
                "cancel": "deterministic_cancel_to_last_effective_request",
            }.get(arbitration_action)
            events.append(
                {
                    "tick_id": _safe_int(tick.get("tick_id")),
                    "request_frame": _safe_int(execution_row.get("request_frame")),
                    "sample_name": str(execution_row.get("sample_name") or ""),
                    "carla_frame": frame_id,
                    "timestamp": _safe_float(tick.get("timestamp")),
                    "requested_behavior": str(tick.get("selected_behavior") or ""),
                    "executing_behavior": str(((execution_row.get("execution_state") or {}).get("current_behavior")) or ""),
                    "planner_status": str(((execution_row.get("planner_diagnostics") or {}).get("planner_status")) or tick.get("execution_status") or ""),
                    "fallback_required": fallback_required,
                    "fallback_activated": fallback_activated,
                    "fallback_reason": ";".join(part for part in reason_parts if part),
                    "fallback_source": "stage4_arbitration",
                    "fallback_mode": mode,
                    "fallback_action": str(((execution_row.get("execution_state") or {}).get("current_behavior")) or ""),
                    "takeover_latency_ms": (
                        _safe_float(tick.get("planner_execution_latency_ms"))
                        if _safe_float(tick.get("planner_execution_latency_ms")) is not None
                        else _safe_float(tick.get("execution_latency_ms"))
                    ),
                    "fallback_success": (
                        None
                        if not fallback_activated
                        else bool(str(tick.get("execution_status") or "") in {"success", "executing"})
                    ),
                    "missing_when_required": bool(fallback_required and not fallback_activated),
                    "notes": notes,
                }
            )
        return events

    events: list[dict[str, Any]] = []
    safe_fallback_behaviors = {"keep_lane", "slow_down", "stop_before_obstacle"}
    for row in execution_rows:
        execution_request = dict(row.get("execution_request") or {})
        execution_state = dict(row.get("execution_state") or {})
        planner_diag = dict(row.get("planner_diagnostics") or {})
        notes = sorted(
            {
                str(note)
                for note in [
                    *((execution_state.get("notes") or [])),
                    *((planner_diag.get("planner_notes") or [])),
                ]
                if note
            }
        )
        requested_behavior = str(execution_request.get("requested_behavior") or "")
        executing_behavior = str(execution_state.get("current_behavior") or "")
        execution_status = str(execution_state.get("execution_status") or planner_diag.get("planner_status") or "")
        fallback_required = bool(
            requested_behavior != executing_behavior
            or execution_status in {"fail", "aborted"}
            or any(
                note in {
                    "lane_change_plan_unavailable",
                    "current_waypoint_not_found",
                    "active_lane_change_aborted_for_stop",
                    "blocking_object_id_not_bound_to_live_actor",
                }
                for note in notes
            )
        )
        fallback_activated = fallback_required
        frame_id = _safe_int(row.get("carla_frame"))
        latency_row = latency_by_frame.get(frame_id, {})
        reason = ";".join(notes) if notes else (
            "requested_vs_executed_mismatch" if requested_behavior != executing_behavior else execution_status
        )
        events.append(
            {
                "tick_id": None,
                "request_frame": _safe_int(row.get("request_frame")),
                "sample_name": str(row.get("sample_name") or ""),
                "carla_frame": frame_id,
                "timestamp": _safe_float(row.get("timestamp")),
                "requested_behavior": requested_behavior,
                "executing_behavior": executing_behavior,
                "planner_status": str(planner_diag.get("planner_status") or execution_status),
                "fallback_required": fallback_required,
                "fallback_activated": fallback_activated,
                "fallback_reason": reason if fallback_required else "",
                "fallback_source": "execution_bridge",
                "fallback_mode": (
                    "deterministic_local_planner_bridge" if fallback_activated else None
                ),
                "fallback_action": executing_behavior if fallback_activated else None,
                "takeover_latency_ms": _safe_float(latency_row.get("planner_execution_latency_ms")),
                "fallback_success": (
                    None if not fallback_activated else bool(executing_behavior in safe_fallback_behaviors)
                ),
                "missing_when_required": False,
                "notes": notes,
            }
        )
    return events


def _latency_summary(latency_rows: list[dict[str, Any]]) -> dict[str, Any]:
    planner_values = [_safe_float(row.get("planner_execution_latency_ms")) for row in latency_rows]
    planner_values = [value for value in planner_values if value is not None]
    decision_values = [_safe_float(row.get("decision_to_execution_latency_ms")) for row in latency_rows]
    decision_values = [value for value in decision_values if value is not None]
    total_values = [_safe_float(row.get("total_loop_latency_ms")) for row in latency_rows]
    total_values = [value for value in total_values if value is not None]
    over_budget_count = sum(1 for row in latency_rows if row.get("over_budget") is True)
    has_online_rows = any(str(row.get("source_stage")) == "stage4_online" for row in latency_rows)
    return {
        "measurement_support": {
            "planner_execution_latency_ms": "present" if planner_values else "missing",
            "decision_to_execution_latency_ms": (
                "approximate" if decision_values and not has_online_rows else ("present" if decision_values else "missing")
            ),
            "total_loop_latency_ms": (
                "approximate" if total_values and not has_online_rows else ("present" if total_values else "missing")
            ),
        },
        "num_records": len(latency_rows),
        "direct_latency_records": len(planner_values),
        "mean_planner_execution_latency_ms": _mean_or_none(planner_values),
        "mean_decision_to_execution_latency_ms": _mean_or_none(decision_values),
        "mean_total_loop_latency_ms": _mean_or_none(total_values),
        "max_total_loop_latency_ms": None if not total_values else float(max(total_values)),
        "over_budget_count": int(over_budget_count),
        "over_budget_rate": _safe_div(float(over_budget_count), float(len(total_values))),
    }


def _fallback_summary(fallback_events: list[dict[str, Any]]) -> dict[str, Any]:
    activated = [event for event in fallback_events if event.get("fallback_activated")]
    required = [event for event in fallback_events if event.get("fallback_required")]
    reason_covered = [
        event for event in activated
        if str(event.get("fallback_reason") or "").strip()
    ]
    takeover_latencies = [
        _safe_float(event.get("takeover_latency_ms"))
        for event in activated
    ]
    takeover_latencies = [value for value in takeover_latencies if value is not None]
    return {
        "measurement_support": {
            "fallback_activation_rate": "present" if fallback_events else "missing",
            "fallback_reason_coverage": "present" if activated else "approximate",
            "fallback_takeover_latency_ms": "present" if takeover_latencies else "missing",
        },
        "num_events": len(fallback_events),
        "required_events": len(required),
        "activated_events": len(activated),
        "fallback_activation_rate": _safe_div(float(len(activated)), float(len(fallback_events))),
        "fallback_reason_coverage": _safe_div(float(len(reason_covered)), float(len(activated))),
        "fallback_takeover_latency_ms": _mean_or_none(takeover_latencies),
        "fallback_missing_when_required_count": int(
            sum(1 for event in fallback_events if event.get("missing_when_required"))
        ),
    }


def _lane_change_completion_validity(
    lane_change_events: list[dict[str, Any]],
) -> float | None:
    if not lane_change_events:
        return None
    successful_events = [event for event in lane_change_events if str(event.get("result")) == "success"]
    if not successful_events:
        return 0.0
    valid_successes = 0
    for event in successful_events:
        target_lane_id = event.get("target_lane_id")
        start_lane_id = event.get("start_lane_id")
        max_shift = _safe_float(event.get("max_lateral_shift_m")) or 0.0
        required_shift = _safe_float(event.get("required_lateral_shift_m"))
        reason = str(event.get("reason", ""))
        shift_ok = True
        if required_shift is not None:
            shift_ok = max_shift + 1e-6 >= max(required_shift * 0.8, 0.25)
        lane_ok = target_lane_id is not None and target_lane_id != start_lane_id
        reason_ok = "reverted" not in reason and "timeout" not in reason
        if lane_ok and shift_ok and reason_ok:
            valid_successes += 1
    return _safe_div(float(valid_successes), float(len(successful_events)))


def _heading_settle_times(
    trajectory_trace: list[dict[str, Any]],
    lane_change_events: list[dict[str, Any]],
) -> list[float]:
    if not trajectory_trace:
        return []
    settle_times: list[float] = []
    for event in lane_change_events:
        if str(event.get("result")) != "success":
            continue
        end_frame = _safe_int(event.get("end_frame"))
        target_lane_id = event.get("target_lane_id")
        if end_frame is None:
            continue
        candidate_rows = [
            row
            for row in trajectory_trace
            if _safe_int(row.get("carla_frame")) is not None and int(row["carla_frame"]) >= end_frame
        ]
        if not candidate_rows:
            continue
        end_timestamp = _safe_float(candidate_rows[0].get("timestamp"))
        consecutive = 0
        for row in candidate_rows:
            yaw_rate = abs(_safe_float(row.get("yaw_rate_deg_per_s")) or 0.0)
            distance_to_center = _safe_float(row.get("distance_to_current_lane_center"))
            on_target_lane = (target_lane_id is None) or row.get("current_lane_id") == target_lane_id
            settled = (
                bool(row.get("pose_supported"))
                and on_target_lane
                and distance_to_center is not None
                and distance_to_center <= 0.60
                and yaw_rate <= 10.0
            )
            consecutive = consecutive + 1 if settled else 0
            if consecutive >= 3 and end_timestamp is not None:
                timestamp = _safe_float(row.get("timestamp"))
                if timestamp is not None and timestamp >= end_timestamp:
                    settle_times.append(float(timestamp - end_timestamp))
                break
    return settle_times


def _lateral_oscillation_rate(control_trace: list[dict[str, Any]]) -> float | None:
    direct_rows = [row for row in control_trace if row.get("support_level") == "direct" and row.get("steer") is not None]
    if len(direct_rows) < 3:
        return None
    flips = 0
    eligible = 0
    previous_sign = None
    for row in direct_rows:
        steer = float(row.get("steer") or 0.0)
        if abs(steer) < 0.05:
            continue
        sign = 1 if steer > 0.0 else -1
        if previous_sign is not None:
            eligible += 1
            if sign != previous_sign:
                flips += 1
        previous_sign = sign
    return _safe_div(float(flips), float(eligible))


def _trajectory_smoothness_proxy(control_trace: list[dict[str, Any]]) -> float | None:
    if len(control_trace) < 2:
        return None
    steer_deltas: list[float] = []
    accel_terms: list[float] = []
    previous_steer = None
    for row in control_trace:
        steer = _safe_float(row.get("steer"))
        if steer is not None and previous_steer is not None:
            steer_deltas.append(abs(steer - previous_steer))
        previous_steer = steer if steer is not None else previous_steer
        accel = _safe_float(row.get("acceleration_proxy_mps2"))
        if accel is not None:
            accel_terms.append(abs(accel))
    if not steer_deltas and not accel_terms:
        return None
    return float((_mean_or_none(steer_deltas) or 0.0) + 0.05 * (_mean_or_none(accel_terms) or 0.0))


def _control_jerk_proxy(control_trace: list[dict[str, Any]]) -> float | None:
    jerks = [_safe_float(row.get("jerk_proxy_mps3")) for row in control_trace]
    jerks = [abs(value) for value in jerks if value is not None]
    return None if not jerks else float(mean(jerks))


def build_planner_quality_bundle(
    execution_records: Iterable[Dict[str, Any]],
    raw_lane_change_events: Iterable[Dict[str, Any]],
    *,
    config: StopControllerConfig | None = None,
    online_tick_records: Iterable[Dict[str, Any]] | None = None,
    source_stage: str = "stage3c_replay",
    total_loop_budget_ms: float = 150.0,
    case_id: str | None = None,
) -> dict[str, Any]:
    config = config or StopControllerConfig()
    execution_rows = list(execution_records)
    online_tick_rows = list(online_tick_records or [])
    lane_change_events = derive_lane_change_events(
        execution_records=execution_rows,
        raw_lane_change_events=list(raw_lane_change_events),
    )
    stop_events = derive_stop_events(execution_rows, config=config)
    trajectory_trace = derive_trajectory_trace(execution_rows)
    control_trace = derive_control_trace(execution_rows)
    lane_change_phase_trace = derive_lane_change_phase_trace(execution_rows, lane_change_events)
    stop_target = derive_stop_target(execution_rows, stop_events, config=config)
    stop_target_trace = derive_stop_target_trace(execution_rows, case_id=case_id)
    stop_binding_summary = derive_stop_binding_summary(stop_target, case_id=case_id)
    planner_control_latency = derive_planner_control_latency(
        execution_rows,
        online_tick_records=online_tick_rows,
        total_loop_budget_ms=float(total_loop_budget_ms),
        source_stage=source_stage,
    )
    fallback_events = derive_fallback_events(
        execution_rows,
        planner_control_latency=planner_control_latency,
        online_tick_records=online_tick_rows,
        source_stage=source_stage,
    )
    latency_summary = _latency_summary(planner_control_latency)
    fallback_summary = _fallback_summary(fallback_events)

    lane_change_attempts = len(lane_change_events)
    lane_change_successes = sum(1 for event in lane_change_events if str(event.get("result")) == "success")
    lane_change_aborts = sum(1 for event in lane_change_events if str(event.get("result")) == "abort")
    heading_settle_times = _heading_settle_times(trajectory_trace, lane_change_events)

    planner_quality_summary = {
        "schema_version": "stage5b.planner_quality_summary.v1",
        "measurement_support": {
            "trajectory_trace": (
                "present"
                if any(bool(row.get("pose_supported")) for row in trajectory_trace)
                else ("approximate" if trajectory_trace else "missing")
            ),
            "control_trace": (
                "present"
                if any(str(row.get("support_level")) == "direct" for row in control_trace)
                else ("approximate" if control_trace else "missing")
            ),
            "lane_change_phase_trace": "present" if lane_change_phase_trace else "missing",
            "stop_target": str((stop_target.get("measurement_support") or {}).get("stop_overshoot_distance_m", "missing")),
            "lane_change_events_enriched": "present" if lane_change_events else "missing",
            "stop_events": "present" if stop_events else "missing",
            "planner_control_latency": str((latency_summary.get("measurement_support") or {}).get("planner_execution_latency_ms", "missing")),
            "fallback_events": str((fallback_summary.get("measurement_support") or {}).get("fallback_activation_rate", "missing")),
        },
        "lane_change_completion_rate": _safe_div(float(lane_change_successes), float(lane_change_attempts)),
        "mean_lane_change_completion_time_s": _mean_or_none(
            _safe_float(event.get("completion_time_s"))
            for event in lane_change_events
            if str(event.get("result")) == "success"
        ),
        "lane_change_abort_rate": _safe_div(float(lane_change_aborts), float(lane_change_attempts)),
        "lane_change_completion_validity": _lane_change_completion_validity(lane_change_events),
        "stop_hold_stability": _mean_or_none(
            min(float(event.get("hold_ticks", 0)) / max(float(config.hold_ticks), 1.0), 1.0)
            for event in stop_events
        ),
        "stop_target_defined_rate": (stop_target.get("summary") or {}).get("stop_target_defined_rate"),
        "stop_final_error_m": (stop_target.get("summary") or {}).get("mean_stop_final_error_m"),
        "stop_overshoot_distance_m": (stop_target.get("summary") or {}).get("mean_stop_overshoot_distance_m"),
        "heading_settle_time_s": _mean_or_none(heading_settle_times),
        "lateral_oscillation_rate": _lateral_oscillation_rate(control_trace),
        "trajectory_smoothness_proxy": _trajectory_smoothness_proxy(control_trace),
        "control_jerk_proxy": _control_jerk_proxy(control_trace),
        "planner_execution_latency_ms": latency_summary.get("mean_planner_execution_latency_ms"),
        "decision_to_execution_latency_ms": latency_summary.get("mean_decision_to_execution_latency_ms"),
        "over_budget_rate": latency_summary.get("over_budget_rate"),
        "fallback_activation_rate": fallback_summary.get("fallback_activation_rate"),
        "fallback_reason_coverage": fallback_summary.get("fallback_reason_coverage"),
        "fallback_takeover_latency_ms": fallback_summary.get("fallback_takeover_latency_ms"),
        "fallback_missing_when_required_count": fallback_summary.get("fallback_missing_when_required_count"),
        "warnings": [
            "stop_final_error_m_unavailable_without_explicit_stop_target_contract"
            if str((stop_target.get("measurement_support") or {}).get("stop_final_error_m")) == "missing"
            else None,
            "control_trace_approximate_from_speed_profile"
            if control_trace and not any(str(row.get("support_level")) == "direct" for row in control_trace)
            else None,
            "trajectory_trace_missing_pose_support"
            if trajectory_trace and not any(bool(row.get("pose_supported")) for row in trajectory_trace)
            else None,
            "decision_to_execution_latency_ms_approximate_without_online_stage_breakdown"
            if str((latency_summary.get("measurement_support") or {}).get("decision_to_execution_latency_ms")) == "approximate"
            else None,
            "fallback_takeover_latency_ms_unavailable_without_direct_activation_events"
            if str((fallback_summary.get("measurement_support") or {}).get("fallback_takeover_latency_ms")) == "missing"
            else None,
        ],
    }
    planner_quality_summary["warnings"] = [
        warning for warning in planner_quality_summary["warnings"] if warning
    ]

    return {
        "lane_change_events_enriched": lane_change_events,
        "stop_events": stop_events,
        "trajectory_trace": trajectory_trace,
        "control_trace": control_trace,
        "lane_change_phase_trace": lane_change_phase_trace,
        "stop_target": stop_target,
        "stop_target_trace": stop_target_trace,
        "stop_binding_summary": stop_binding_summary,
        "planner_control_latency": planner_control_latency,
        "fallback_events": fallback_events,
        "planner_quality_summary": planner_quality_summary,
    }


def write_planner_quality_artifacts(
    output_dir: str | Path,
    *,
    execution_records: Iterable[Dict[str, Any]] | None = None,
    raw_lane_change_events: Iterable[Dict[str, Any]] | None = None,
    config: StopControllerConfig | None = None,
    online_tick_records: Iterable[Dict[str, Any]] | None = None,
    source_stage: str = "stage3c_replay",
    total_loop_budget_ms: float = 150.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    case_id = _case_id_from_output_dir(output_dir)
    execution_rows = list(execution_records) if execution_records is not None else load_jsonl(output_dir / "execution_timeline.jsonl")
    raw_events = list(raw_lane_change_events) if raw_lane_change_events is not None else load_jsonl(output_dir / "lane_change_events.jsonl")
    bundle = build_planner_quality_bundle(
        execution_rows,
        raw_events,
        config=config,
        online_tick_records=online_tick_records,
        source_stage=source_stage,
        total_loop_budget_ms=float(total_loop_budget_ms),
        case_id=case_id,
    )
    _dump_jsonl(output_dir / "lane_change_events_enriched.jsonl", bundle["lane_change_events_enriched"])
    _dump_jsonl(output_dir / "stop_events.jsonl", bundle["stop_events"])
    _dump_jsonl(output_dir / "trajectory_trace.jsonl", bundle["trajectory_trace"])
    _dump_jsonl(output_dir / "control_trace.jsonl", bundle["control_trace"])
    _dump_jsonl(output_dir / "lane_change_phase_trace.jsonl", bundle["lane_change_phase_trace"])
    _dump_json(output_dir / "stop_target.json", bundle["stop_target"])
    _dump_jsonl(output_dir / "stop_target_trace.jsonl", bundle["stop_target_trace"])
    _dump_json(output_dir / "stop_binding_summary.json", bundle["stop_binding_summary"])
    _dump_jsonl(output_dir / "planner_control_latency.jsonl", bundle["planner_control_latency"])
    _dump_jsonl(output_dir / "fallback_events.jsonl", bundle["fallback_events"])
    _dump_json(output_dir / "planner_quality_summary.json", bundle["planner_quality_summary"])
    return bundle


def load_planner_quality_bundle(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    return {
        "lane_change_events_enriched": load_jsonl(run_dir / "lane_change_events_enriched.jsonl"),
        "stop_events": load_jsonl(run_dir / "stop_events.jsonl"),
        "trajectory_trace": load_jsonl(run_dir / "trajectory_trace.jsonl"),
        "control_trace": load_jsonl(run_dir / "control_trace.jsonl"),
        "lane_change_phase_trace": load_jsonl(run_dir / "lane_change_phase_trace.jsonl"),
        "stop_target": load_json(run_dir / "stop_target.json") if (run_dir / "stop_target.json").exists() else {},
        "stop_target_trace": load_jsonl(run_dir / "stop_target_trace.jsonl"),
        "stop_binding_summary": load_json(run_dir / "stop_binding_summary.json") if (run_dir / "stop_binding_summary.json").exists() else {},
        "planner_control_latency": load_jsonl(run_dir / "planner_control_latency.jsonl"),
        "fallback_events": load_jsonl(run_dir / "fallback_events.jsonl"),
        "planner_quality_summary": load_json(run_dir / "planner_quality_summary.json") if (run_dir / "planner_quality_summary.json").exists() else {},
    }
