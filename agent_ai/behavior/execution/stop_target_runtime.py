from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


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


def enrich_runtime_stop_target(
    stop_target: dict[str, Any] | None,
    scenario_manifest_payload: dict[str, Any] | None,
    actor_bindings: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not stop_target:
        return None
    payload = deepcopy(stop_target)
    manifest = dict(scenario_manifest_payload or {})
    bindings = dict(actor_bindings or {})
    source_type = str(payload.get("source_type") or "")
    binding_status = str(payload.get("binding_status") or "unavailable")
    runtime_binding_source = None

    if source_type == "obstacle":
        blocker_binding = dict(bindings.get("blocker") or {})
        runtime_actor_id = _safe_int(blocker_binding.get("runtime_actor_id"))
        materialization_status = str(blocker_binding.get("materialization_status") or "")
        actor_binding_source = str(blocker_binding.get("binding_source") or "")
        blocker_binding_status = str(blocker_binding.get("binding_status") or "")
        if runtime_actor_id is not None and blocker_binding_status == "exact":
            payload["source_actor_id"] = runtime_actor_id
            payload["source_role"] = payload.get("source_role") or "scenario_blocker_actor"
            runtime_binding_source = f"critical_actor_materialization:{actor_binding_source or materialization_status or 'runtime_actor'}"
            binding_status = "exact"
        elif payload.get("source_actor_id") is None:
            blocker_actor_id = _safe_int(manifest.get("blocker_actor_id"))
            if blocker_actor_id is not None:
                payload["source_actor_id"] = blocker_actor_id
                payload["source_role"] = payload.get("source_role") or "scenario_blocker_actor"
                runtime_binding_source = "scenario_manifest.blocker_actor_id"
                if binding_status == "exact":
                    binding_status = "heuristic"
    if payload.get("target_lane_id") is None:
        corridor = dict(manifest.get("corridor") or {})
        lane_id = _safe_int(corridor.get("lane_id"))
        if lane_id is not None:
            payload["target_lane_id"] = lane_id
            payload["target_lane_relation"] = payload.get("target_lane_relation") or "current_lane"
    if runtime_binding_source:
        provenance = dict(payload.get("provenance") or {})
        provenance["runtime_binding_source"] = runtime_binding_source
        if source_type == "obstacle":
            blocker_binding = dict(bindings.get("blocker") or {})
            provenance["actor_binding_source"] = blocker_binding.get("binding_source")
            provenance["materialization_status"] = blocker_binding.get("materialization_status")
            provenance["runtime_actor_id"] = blocker_binding.get("runtime_actor_id")
        payload["provenance"] = provenance
    payload["binding_status"] = binding_status
    return payload


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


def measure_stop_target_distance(vehicle: Any, stop_target: dict[str, Any] | None) -> tuple[float | None, str | None]:
    if not stop_target:
        return None, None
    source_type = str(stop_target.get("source_type") or "")
    if source_type == "obstacle":
        source_actor_id = _safe_int(stop_target.get("source_actor_id"))
        if source_actor_id is None:
            return None, "missing_runtime_actor_binding"
        world = vehicle.get_world()
        actor = world.get_actor(source_actor_id) if world is not None else None
        if actor is None:
            return None, "runtime_actor_missing"
        return _actor_gap_m(vehicle, actor), "live_actor_gap"
    if source_type in {"stop_line", "route_junction"}:
        target_pose = dict(stop_target.get("target_pose") or {})
        if not target_pose:
            return None, "target_pose_missing"
        ego_location = vehicle.get_location()
        if ego_location is None:
            return None, "ego_location_missing"
        target_x = _safe_float(target_pose.get("x_m"))
        target_y = _safe_float(target_pose.get("y_m"))
        target_z = _safe_float(target_pose.get("z_m"))
        if target_x is None or target_y is None or target_z is None:
            return None, "target_pose_incomplete"
        dx = float(ego_location.x) - target_x
        dy = float(ego_location.y) - target_y
        dz = float(ego_location.z) - target_z
        return float(math.sqrt(dx * dx + dy * dy + dz * dz)), "target_pose_distance"
    if source_type == "fallback_safety":
        return 0.0, "fallback_stop"
    return None, None
