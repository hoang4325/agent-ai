from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class StopControllerConfig:
    hold_ticks: int = 10
    full_stop_speed_threshold_mps: float = 0.35
    near_distance_m: float = 8.0
    hard_brake_distance_m: float = 3.0
    overshoot_distance_m: float = 0.5
    stop_success_distance_m: float = 3.0


def compute_stop_command(
    *,
    speed_mps: float,
    distance_to_obstacle_m: float | None,
    config: StopControllerConfig,
) -> Dict[str, Any]:
    distance = None if distance_to_obstacle_m is None else max(0.0, float(distance_to_obstacle_m))
    speed = max(0.0, float(speed_mps))
    notes: list[str] = []

    if distance is None:
        brake = 1.0 if speed > 1.5 else 0.7
        notes.append("stop_without_obstacle_distance")
    elif distance <= config.hard_brake_distance_m:
        brake = 1.0
        notes.append("hard_brake_zone")
    elif distance <= config.near_distance_m:
        ratio = (distance - config.hard_brake_distance_m) / max(
            config.near_distance_m - config.hard_brake_distance_m,
            0.1,
        )
        brake = max(0.45, 1.0 - 0.45 * ratio)
        notes.append("near_stop_zone")
    else:
        brake = 0.45 if speed <= 2.5 else 0.65
        notes.append("approach_stop_zone")

    if speed <= config.full_stop_speed_threshold_mps:
        brake = max(brake, 0.7)
        notes.append("stop_hold_candidate")

    return {
        "throttle": 0.0,
        "steer": 0.0,
        "brake": float(max(0.0, min(1.0, brake))),
        "hand_brake": False,
        "manual_gear_shift": False,
        "notes": notes,
    }


def classify_stop_event(
    *,
    min_speed_mps: float,
    min_distance_to_obstacle_m: float | None,
    hold_ticks: int,
    config: StopControllerConfig,
) -> Dict[str, Any]:
    distance = None if min_distance_to_obstacle_m is None else float(min_distance_to_obstacle_m)
    if min_speed_mps <= config.full_stop_speed_threshold_mps and hold_ticks > 0:
        if distance is not None and distance < config.overshoot_distance_m:
            return {
                "result": "overshoot",
                "reason": "stopped_too_close_to_obstacle",
            }
        return {
            "result": "success",
            "reason": "vehicle_stopped_and_held",
        }
    return {
        "result": "abort",
        "reason": "stop_not_reached",
    }
