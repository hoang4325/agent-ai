from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .map_adapter import CarlaMapAdapter, WaypointProbe
from .schema import LaneAwareObject, LaneContext

LOGGER = logging.getLogger(__name__)


def _ensure_matrix(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {array.shape}")
    return array


def _transform_points(points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz.reshape(0, 3)
    homogeneous = np.concatenate(
        [points_xyz.astype(np.float32), np.ones((points_xyz.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    transformed = homogeneous @ matrix.T
    return transformed[:, :3].astype(np.float32)


def _lane_key(descriptor: Dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    return (
        descriptor.get("road_id"),
        descriptor.get("section_id"),
        descriptor.get("lane_id"),
    )


def _waypoint_key(probe: WaypointProbe) -> tuple[int | None, int | None, int | None]:
    if probe.waypoint is None:
        return (None, None, None)
    return (int(probe.waypoint.road_id), int(probe.waypoint.section_id), int(probe.waypoint.lane_id))


def _forward_and_left_vectors_from_yaw(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yaw_rad = math.radians(float(yaw_deg))
    forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float32)
    left = np.array([-math.sin(yaw_rad), math.cos(yaw_rad)], dtype=np.float32)
    return forward, left


def _classify_lane_relation(
    *,
    probe: WaypointProbe,
    lane_context: LaneContext,
    longitudinal_m: float,
    lateral_m: float,
    lane_width_m: float,
) -> str:
    probe_key = _waypoint_key(probe)
    current_key = _lane_key(lane_context.current_lane.to_dict())
    left_key = _lane_key(lane_context.left_lane.to_dict())
    right_key = _lane_key(lane_context.right_lane.to_dict())

    if probe_key == current_key and current_key != (None, None, None):
        return "current_lane"
    if probe_key == left_key and left_key != (None, None, None):
        return "left_lane"
    if probe_key == right_key and right_key != (None, None, None):
        return "right_lane"

    abs_lateral = abs(float(lateral_m))
    if lane_context.junction_context.is_in_junction or (probe.waypoint and probe.waypoint.is_junction):
        if abs_lateral <= lane_width_m:
            return "current_lane"
        if lateral_m > 0.0 and abs_lateral <= lane_width_m * 2.25:
            return "left_lane" if lane_context.left_lane.exists else "cross_lane"
        if lateral_m < 0.0 and abs_lateral <= lane_width_m * 2.25:
            return "right_lane" if lane_context.right_lane.exists else "cross_lane"
        return "cross_lane"

    if abs_lateral <= lane_width_m * 0.65:
        return "current_lane"
    if lateral_m > 0.0 and abs_lateral <= lane_width_m * 2.0:
        return "left_lane" if lane_context.left_lane.exists else "cross_lane"
    if lateral_m < 0.0 and abs_lateral <= lane_width_m * 2.0:
        return "right_lane" if lane_context.right_lane.exists else "cross_lane"
    if probe.waypoint is None:
        return "unknown"
    return "cross_lane"


def _lane_tag_from_relation(relation: str) -> str:
    return {
        "current_lane": "current",
        "left_lane": "left",
        "right_lane": "right",
        "cross_lane": "cross",
        "unknown": "unknown",
    }.get(relation, "unknown")


def build_lane_relative_objects(
    *,
    world_state: Dict[str, Any],
    lane_context: LaneContext,
    map_adapter: CarlaMapAdapter,
    max_projection_distance_m: float = 10.0,
    current_lane_blocking_distance_m: float = 55.0,
) -> List[LaneAwareObject]:
    ego_world_matrix = _ensure_matrix(world_state["ego"]["world_from_ego_bevfusion"])
    ego_position_world = np.asarray(world_state["ego"]["position_world"][:3], dtype=np.float32)
    forward_vec, left_vec = _forward_and_left_vectors_from_yaw(float(world_state["ego"]["yaw_deg"]))
    lane_width_m = float(lane_context.current_lane.lane_width_m or 3.5)

    objects = world_state.get("objects", [])
    if not objects:
        return []

    ego_points = np.asarray([item["position_ego"][:3] for item in objects], dtype=np.float32)
    world_points_bev = _transform_points(ego_points, ego_world_matrix)

    lane_objects: List[LaneAwareObject] = []
    for source_track, world_point_bev in zip(objects, world_points_bev):
        probe = map_adapter.closest_waypoint_from_bevfusion_world(world_point_bev.tolist())
        world_point_carla = map_adapter.carla_location_from_bevfusion_world(world_point_bev.tolist())

        delta_xy = np.array(
            [
                float(world_point_bev[0] - ego_position_world[0]),
                float(world_point_bev[1] - ego_position_world[1]),
            ],
            dtype=np.float32,
        )
        longitudinal_m = float(delta_xy @ forward_vec)
        lateral_m = float(delta_xy @ left_vec)

        projection_distance = None if probe.projected_distance_m is None else float(probe.projected_distance_m)
        relation = _classify_lane_relation(
            probe=probe,
            lane_context=lane_context,
            longitudinal_m=longitudinal_m,
            lateral_m=lateral_m,
            lane_width_m=lane_width_m,
        )
        if projection_distance is not None and projection_distance > float(max_projection_distance_m) and relation == "unknown":
            relation = "unknown"

        same_direction = False
        if probe.waypoint is not None and lane_context.current_lane.exists:
            current_probe = map_adapter.closest_waypoint_from_bevfusion_world(world_state["ego"]["position_world"])
            if current_probe.waypoint is not None:
                ref_forward = current_probe.waypoint.transform.get_forward_vector()
                obj_forward = probe.waypoint.transform.get_forward_vector()
                same_direction = bool(ref_forward.x * obj_forward.x + ref_forward.y * obj_forward.y > 0.5)

        is_front_in_current_lane = relation == "current_lane" and longitudinal_m > 0.0
        is_rear_in_current_lane = relation == "current_lane" and longitudinal_m < 0.0
        is_blocking_current_lane = (
            is_front_in_current_lane
            and float(longitudinal_m) <= float(current_lane_blocking_distance_m)
            and abs(float(lateral_m)) <= lane_width_m
        )

        distance_to_lane_center = None
        if probe.projected_distance_m is not None:
            distance_to_lane_center = float(probe.projected_distance_m)

        lane_objects.append(
            LaneAwareObject(
                track_id=int(source_track["track_id"]),
                class_name=str(source_track["class_name"]),
                class_group=str(source_track["class_group"]),
                position_world_carla=[
                    float(world_point_carla.x),
                    float(world_point_carla.y),
                    float(world_point_carla.z),
                ],
                lane_relation=relation,
                lane_tag=_lane_tag_from_relation(relation),
                object_lane_id=None if probe.waypoint is None else int(probe.waypoint.lane_id),
                object_road_id=None if probe.waypoint is None else int(probe.waypoint.road_id),
                longitudinal_m=float(longitudinal_m),
                lateral_m=float(lateral_m),
                is_front_in_current_lane=bool(is_front_in_current_lane),
                is_rear_in_current_lane=bool(is_rear_in_current_lane),
                is_blocking_current_lane=bool(is_blocking_current_lane),
                same_direction_as_ego_lane=bool(same_direction),
                distance_to_lane_center_m=distance_to_lane_center,
                source_track=dict(source_track),
            )
        )

    lane_objects.sort(
        key=lambda item: (
            0 if item.lane_relation == "current_lane" else 1,
            0 if item.is_front_in_current_lane else 1,
            abs(item.longitudinal_m if item.longitudinal_m is not None else 1e9),
        )
    )
    LOGGER.info(
        "Lane reasoning frame=%d sample=%s lane_objects=%d current_lane_front=%d",
        int(world_state["frame"]),
        str(world_state["sample_name"]),
        len(lane_objects),
        sum(1 for item in lane_objects if item.is_front_in_current_lane),
    )
    return lane_objects


def summarize_lane_scene(
    lane_objects: Iterable[LaneAwareObject],
    lane_context: LaneContext,
) -> Dict[str, Any]:
    lane_objects = list(lane_objects)
    current_front = [
        item for item in lane_objects if item.is_front_in_current_lane and item.lane_relation == "current_lane"
    ]
    current_front.sort(key=lambda item: float(item.longitudinal_m or 1e9))
    current_lane_blocker = next((item for item in current_front if item.is_blocking_current_lane), None)
    left_lane_ahead = [
        item for item in lane_objects if item.lane_relation == "left_lane" and (item.longitudinal_m or -1e9) > -10.0
    ]
    right_lane_ahead = [
        item for item in lane_objects if item.lane_relation == "right_lane" and (item.longitudinal_m or -1e9) > -10.0
    ]
    return {
        "current_lane_blocked": current_lane_blocker is not None,
        "front_object_in_current_lane": None if current_lane_blocker is None else current_lane_blocker.to_dict(),
        "front_object_count_current_lane": int(len(current_front)),
        "left_lane_occupancy_count": int(len(left_lane_ahead)),
        "right_lane_occupancy_count": int(len(right_lane_ahead)),
        "left_lane_available": bool(lane_context.left_lane.exists),
        "right_lane_available": bool(lane_context.right_lane.exists),
        "junction_active": bool(lane_context.junction_context.is_in_junction),
        "junction_ahead": bool(lane_context.junction_context.junction_ahead),
    }
