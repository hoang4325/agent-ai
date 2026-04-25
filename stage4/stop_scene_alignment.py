from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from .scenario_actor_materialization import (
    _advance_along_lane,
    _build_expected_actors,
    _transform_from_payload,
)


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


def _transform_to_pose(transform: Any) -> dict[str, Any]:
    return {
        "location": {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
        },
        "rotation": {
            "roll": float(transform.rotation.roll),
            "pitch": float(transform.rotation.pitch),
            "yaw": float(transform.rotation.yaw),
        },
    }


def _actor_gap_m(vehicle: Any, actor: Any) -> float | None:
    vehicle_location = vehicle.get_location()
    actor_location = actor.get_location()
    if vehicle_location is None or actor_location is None:
        return None
    center_distance = float(vehicle_location.distance(actor_location))
    vehicle_extent = getattr(getattr(vehicle, "bounding_box", None), "extent", None)
    actor_extent = getattr(getattr(actor, "bounding_box", None), "extent", None)
    vehicle_radius = 0.0
    actor_radius = 0.0
    if vehicle_extent is not None:
        vehicle_radius = math.sqrt(float(vehicle_extent.x) ** 2 + float(vehicle_extent.y) ** 2)
    if actor_extent is not None:
        actor_radius = math.sqrt(float(actor_extent.x) ** 2 + float(actor_extent.y) ** 2)
    return float(max(0.0, center_distance - vehicle_radius - actor_radius))


def _apply_stationary_brake(vehicle: Any) -> None:
    import carla  # type: ignore

    vehicle.set_autopilot(False)
    if hasattr(vehicle, "set_target_velocity"):
        vehicle.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
    if hasattr(vehicle, "set_target_angular_velocity"):
        vehicle.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=False,
            manual_gear_shift=False,
        )
    )


def _realign_vehicle(vehicle: Any, transform: Any, *, hand_brake: bool) -> None:
    vehicle.set_autopilot(False)
    if hasattr(vehicle, "set_simulate_physics"):
        vehicle.set_simulate_physics(True)
    vehicle.set_transform(transform)
    _apply_stationary_brake(vehicle)
    if hand_brake:
        control = vehicle.get_control()
        control.hand_brake = True
        control.brake = 1.0
        vehicle.apply_control(control)


def _binding_for_role(actor_bindings: dict[str, Any] | None, role: str) -> dict[str, Any]:
    return dict((actor_bindings or {}).get(role) or {})


def resolve_alignment_spec(args: Any) -> dict[str, Any] | None:
    enabled = bool(getattr(args, "stop_alignment_enabled", False))
    if not enabled:
        return None
    expected_gap_m = _safe_float(getattr(args, "stop_alignment_expected_gap_m", None))
    if expected_gap_m is None:
        return None
    sample_name = str(getattr(args, "stop_alignment_sample_name", "") or "")
    request_frame = _safe_int(getattr(args, "stop_alignment_request_frame", None))
    return {
        "enabled": True,
        "expected_gap_m": expected_gap_m,
        "gap_tolerance_m": _safe_float(getattr(args, "stop_alignment_gap_tolerance_m", 3.0)) or 3.0,
        "alignment_source": str(
            getattr(args, "stop_alignment_source", None)
            or "manifest_distance_minus_expected_stop_gap"
        ),
        "alignment_window_id": sample_name or (None if request_frame is None else f"request_frame_{request_frame}"),
        "sample_name": sample_name or None,
        "request_frame": request_frame,
    }


def apply_stop_scene_alignment(
    *,
    world: Any,
    ego_vehicle: Any,
    scenario_manifest_payload: dict[str, Any] | None,
    actor_bindings: dict[str, Any] | None,
    args: Any,
) -> dict[str, Any] | None:
    spec = resolve_alignment_spec(args)
    if spec is None:
        return None

    payload = {
        "schema_version": "stage6.stop_scene_alignment.v1",
        "alignment_source": spec["alignment_source"],
        "alignment_window_id": spec["alignment_window_id"],
        "sample_name": spec["sample_name"],
        "request_frame": spec["request_frame"],
        "expected_blocker_gap_m": spec["expected_gap_m"],
        "gap_tolerance_m": spec["gap_tolerance_m"],
        "expected_ego_pose": None,
        "runtime_ego_pose": None,
        "expected_blocker_pose": None,
        "runtime_blocker_pose": None,
        "expected_blocker_gap_m": spec["expected_gap_m"],
        "runtime_blocker_gap_m": None,
        "alignment_error_m": None,
        "alignment_consistent": False,
        "alignment_ready": False,
        "binding_status": "missing",
        "provenance": {
            "source_stage": "stage4_online_orchestrator",
            "source_artifact": "stop_scene_alignment",
            "derivation_mode": "manifest_window_alignment",
        },
        "notes": [],
    }

    manifest = dict(scenario_manifest_payload or {})
    expected_actors = _build_expected_actors(world, manifest)
    ego_spec = dict(expected_actors.get("ego") or {})
    blocker_spec = dict(expected_actors.get("blocker") or {})
    blocker_binding = _binding_for_role(actor_bindings, "blocker")
    payload["binding_status"] = str(blocker_binding.get("binding_status") or "missing")
    if not ego_spec or not blocker_spec:
        payload["notes"].append("stop_alignment_expected_actor_recipe_missing")
        return payload

    blocker_distance_m = _safe_float(((manifest.get("placements") or {}).get("blocker_distance_m")))
    ego_waypoint = ego_spec.get("expected_waypoint")
    ego_transform = ego_spec.get("expected_transform")
    blocker_transform = blocker_spec.get("expected_transform")
    if blocker_distance_m is None or ego_waypoint is None or ego_transform is None or blocker_transform is None:
        payload["notes"].append("stop_alignment_manifest_incomplete")
        return payload

    advance_distance_m = max(float(blocker_distance_m) - float(spec["expected_gap_m"]), 0.0)
    aligned_waypoint = _advance_along_lane(ego_waypoint, advance_distance_m)
    z_offset = float(ego_transform.location.z) - float(ego_waypoint.transform.location.z)
    aligned_ego_transform = _transform_from_payload(
        {
            "location": {
                "x": float(aligned_waypoint.transform.location.x),
                "y": float(aligned_waypoint.transform.location.y),
                "z": float(aligned_waypoint.transform.location.z),
            },
            "rotation": {
                "roll": float(aligned_waypoint.transform.rotation.roll),
                "pitch": float(aligned_waypoint.transform.rotation.pitch),
                "yaw": float(aligned_waypoint.transform.rotation.yaw),
            },
        },
        z_offset_m=z_offset,
    )
    if aligned_ego_transform is None:
        payload["notes"].append("stop_alignment_ego_transform_unavailable")
        return payload

    _realign_vehicle(ego_vehicle, aligned_ego_transform, hand_brake=False)
    payload["expected_ego_pose"] = _transform_to_pose(aligned_ego_transform)
    payload["runtime_ego_pose"] = _transform_to_pose(ego_vehicle.get_transform())

    runtime_blocker_actor = None
    runtime_actor_id = _safe_int(blocker_binding.get("runtime_actor_id"))
    if runtime_actor_id is not None:
        runtime_blocker_actor = world.get_actor(runtime_actor_id)
    if runtime_blocker_actor is not None:
        _realign_vehicle(runtime_blocker_actor, blocker_transform, hand_brake=True)
        payload["expected_blocker_pose"] = _transform_to_pose(blocker_transform)
        payload["runtime_blocker_pose"] = _transform_to_pose(runtime_blocker_actor.get_transform())
        runtime_gap_m = _actor_gap_m(ego_vehicle, runtime_blocker_actor)
        payload["runtime_blocker_gap_m"] = runtime_gap_m
        if runtime_gap_m is not None:
            alignment_error_m = abs(float(runtime_gap_m) - float(spec["expected_gap_m"]))
            payload["alignment_error_m"] = float(alignment_error_m)
            payload["alignment_consistent"] = bool(alignment_error_m <= float(spec["gap_tolerance_m"]))
            payload["alignment_ready"] = bool(
                payload["alignment_consistent"] and str(blocker_binding.get("binding_status")) == "exact"
            )
        else:
            payload["notes"].append("runtime_blocker_gap_unavailable")
    else:
        payload["notes"].append("runtime_blocker_actor_missing_for_stop_alignment")

    if not payload["alignment_ready"]:
        payload["notes"].append("stop_alignment_not_ready_for_stop_quality_gate")
    else:
        payload["notes"].append("stop_alignment_ready_for_stop_quality_gate")
    return deepcopy(payload)
