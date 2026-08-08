"""
Stage-4 control helpers extracted from the online orchestrator.

Keeps ``Stage4OnlineOrchestrator`` focused on the tick loop while shared
control math / CARLA apply helpers live here for reuse and testing.
"""
from __future__ import annotations

from typing import Any

from agent_ai.shared.numeric import clamp


def vehicle_speed_mps(vehicle: Any) -> float:
    velocity = vehicle.get_velocity()
    return float((velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5)


def apply_neutral_brake(vehicle: Any) -> None:
    import carla  # type: ignore

    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=0.2,
            hand_brake=False,
            manual_gear_shift=False,
        )
    )


def apply_bounded_motion_guard(
    vehicle: Any,
    *,
    current_speed_mps: float,
    speed_limit_mps: float,
) -> None:
    import carla  # type: ignore

    current_control = vehicle.get_control()
    throttle_cap = 0.12
    brake_floor = 0.05
    if current_speed_mps >= max(0.0, speed_limit_mps * 0.85):
        brake_floor = 0.25
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=min(float(current_control.throttle), throttle_cap),
            steer=float(current_control.steer),
            brake=max(float(current_control.brake), brake_floor),
            hand_brake=bool(current_control.hand_brake),
            manual_gear_shift=bool(current_control.manual_gear_shift),
            reverse=bool(current_control.reverse),
            gear=int(current_control.gear),
        )
    )


def build_shadow_candidate_control(
    *,
    carla_module: Any,
    proposal: dict[str, Any],
    current_speed_mps: float,
) -> Any:
    requested_behavior = str(proposal.get("shadow_requested_behavior") or "")
    sampled_path = list((((proposal.get("proposed_trajectory") or {}).get("sampled_path")) or []))
    target_speed_mps = float(proposal.get("shadow_target_speed_mps") or 0.0)
    lateral_offset_m = 0.0
    for point in sampled_path[1:4]:
        offset = float(point.get("lateral_offset_m") or 0.0)
        if abs(offset) >= 0.02:
            lateral_offset_m = offset
            break
    if "lane_change_left" in requested_behavior and lateral_offset_m <= 0.0:
        lateral_offset_m = max(lateral_offset_m, 0.35)
    elif "lane_change_right" in requested_behavior and lateral_offset_m >= 0.0:
        lateral_offset_m = min(lateral_offset_m, -0.35)

    steer = clamp(lateral_offset_m * 0.18, -0.20, 0.20)
    speed_error_mps = float(target_speed_mps - current_speed_mps)
    throttle = 0.0
    brake = 0.0
    if requested_behavior == "stop_before_obstacle" or target_speed_mps <= 0.05:
        brake = 0.35 if current_speed_mps >= 0.20 else 0.20
    elif speed_error_mps > 0.10:
        throttle = clamp(0.05 + speed_error_mps * 0.06, 0.05, 0.14)
    elif speed_error_mps < -0.05:
        brake = clamp(abs(speed_error_mps) * 0.12, 0.05, 0.20)
    else:
        throttle = 0.04 if current_speed_mps < max(target_speed_mps, 0.15) else 0.0
        brake = 0.0 if throttle > 0.0 else 0.05

    return carla_module.VehicleControl(
        throttle=float(throttle),
        steer=float(steer),
        brake=float(brake),
        hand_brake=False,
        manual_gear_shift=False,
    )
