from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import carla

from .schema import JunctionContext, LaneContext, LaneDescriptor

LOGGER = logging.getLogger(__name__)


def bevfusion_world_xyz_to_carla(xyz: Iterable[float]) -> List[float]:
    x, y, z = [float(value) for value in xyz]
    return [x, -y, z]


def carla_xyz_to_bevfusion(xyz: Iterable[float]) -> List[float]:
    x, y, z = [float(value) for value in xyz]
    return [x, -y, z]


def _location_to_dict(location: carla.Location) -> Dict[str, float]:
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
    }


def _rotation_to_dict(rotation: carla.Rotation) -> Dict[str, float]:
    return {
        "roll": float(rotation.roll),
        "pitch": float(rotation.pitch),
        "yaw": float(rotation.yaw),
    }


def _transform_to_dict(transform: carla.Transform) -> Dict[str, Any]:
    return {
        "location": _location_to_dict(transform.location),
        "rotation": _rotation_to_dict(transform.rotation),
    }


def _transform_to_bevfusion_dict(transform: carla.Transform) -> Dict[str, Any]:
    location = carla_xyz_to_bevfusion([transform.location.x, transform.location.y, transform.location.z])
    return {
        "location": {
            "x": float(location[0]),
            "y": float(location[1]),
            "z": float(location[2]),
        },
        "rotation": _rotation_to_dict(transform.rotation),
    }


def _lane_change_allowed(current_waypoint: carla.Waypoint, side: str) -> bool:
    if side == "left":
        return bool(current_waypoint.lane_change & carla.LaneChange.Left)
    if side == "right":
        return bool(current_waypoint.lane_change & carla.LaneChange.Right)
    raise ValueError(f"Unsupported lane side: {side}")


def _forward_xy(waypoint: carla.Waypoint) -> List[float]:
    vector = waypoint.transform.get_forward_vector()
    norm = math.hypot(float(vector.x), float(vector.y))
    if norm <= 1e-6:
        return [1.0, 0.0]
    return [float(vector.x) / norm, float(vector.y) / norm]


def _same_direction(reference: carla.Waypoint, candidate: carla.Waypoint) -> bool:
    ref = _forward_xy(reference)
    other = _forward_xy(candidate)
    dot = ref[0] * other[0] + ref[1] * other[1]
    return bool(dot > 0.5)


def _turn_like_label(current_waypoint: carla.Waypoint, next_waypoint: carla.Waypoint) -> str:
    current_yaw = float(current_waypoint.transform.rotation.yaw) % 360.0
    next_yaw = float(next_waypoint.transform.rotation.yaw) % 360.0
    diff_angle = (next_yaw - current_yaw) % 180.0
    if diff_angle < 35.0 or diff_angle > 145.0:
        return "straight"
    if diff_angle > 90.0:
        return "left"
    return "right"


def _normalize_town_name(town: str) -> str:
    normalized = str(town).replace("\\", "/")
    if "/" in normalized:
        return normalized.split("/")[-1]
    return normalized


@dataclass
class WaypointProbe:
    waypoint: carla.Waypoint | None
    query_location_carla: carla.Location
    projected_distance_m: float | None
    lane_type_name: str | None


class CarlaMapAdapter:
    def __init__(
        self,
        *,
        client: carla.Client,
        world: carla.World,
        map_obj: carla.Map,
        host: str,
        port: int,
    ) -> None:
        self.client = client
        self.world = world
        self.map = map_obj
        self.host = str(host)
        self.port = int(port)

    @classmethod
    def connect(
        cls,
        *,
        host: str,
        port: int,
        timeout_s: float = 10.0,
        town: str | None = None,
        load_town_if_needed: bool = False,
    ) -> "CarlaMapAdapter":
        client = carla.Client(host, int(port))
        client.set_timeout(float(timeout_s))
        world = client.get_world()
        if town:
            expected_world_name = str(town)
            current_world_name = str(world.get_map().name)
            if current_world_name != expected_world_name:
                if not load_town_if_needed:
                    raise RuntimeError(
                        "CARLA world mismatch. "
                        f"Expected {expected_world_name}, got {current_world_name}. "
                        "Pass --load-town-if-needed to switch the simulator world."
                    )
                LOGGER.info(
                    "Loading CARLA town because current world does not match artifact town: %s -> %s",
                    current_world_name,
                    expected_world_name,
                )
                world = client.load_world(_normalize_town_name(expected_world_name))
        map_obj = world.get_map()
        LOGGER.info(
            "Connected to CARLA host=%s port=%d map=%s",
            host,
            int(port),
            map_obj.name,
        )
        return cls(client=client, world=world, map_obj=map_obj, host=host, port=int(port))

    def carla_location_from_bevfusion_world(self, xyz_bevfusion: Iterable[float]) -> carla.Location:
        x, y, z = bevfusion_world_xyz_to_carla(xyz_bevfusion)
        return carla.Location(x=float(x), y=float(y), z=float(z))

    def closest_waypoint_from_bevfusion_world(
        self,
        xyz_bevfusion: Iterable[float],
        *,
        lane_type: carla.LaneType = carla.LaneType.Driving,
    ) -> WaypointProbe:
        query_location = self.carla_location_from_bevfusion_world(xyz_bevfusion)
        waypoint = self.map.get_waypoint(
            query_location,
            project_to_road=True,
            lane_type=lane_type,
        )
        projected_distance = None
        lane_type_name = None
        if waypoint is not None:
            projected_distance = float(query_location.distance(waypoint.transform.location))
            lane_type_name = str(waypoint.lane_type)
        return WaypointProbe(
            waypoint=waypoint,
            query_location_carla=query_location,
            projected_distance_m=projected_distance,
            lane_type_name=lane_type_name,
        )

    def _descriptor_from_waypoint(
        self,
        *,
        role: str,
        waypoint: carla.Waypoint | None,
        ego_waypoint: carla.Waypoint,
        lane_change_allowed_from_current: bool = False,
    ) -> LaneDescriptor:
        if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
            return LaneDescriptor(
                exists=False,
                role=role,
                road_id=None,
                section_id=None,
                lane_id=None,
                lane_width_m=None,
                lane_type=None,
                lane_change=None,
                same_direction_as_ego=False,
                transform_carla=None,
                transform_bevfusion_world=None,
                lane_change_allowed_from_current=False,
            )

        return LaneDescriptor(
            exists=True,
            role=role,
            road_id=int(waypoint.road_id),
            section_id=int(waypoint.section_id),
            lane_id=int(waypoint.lane_id),
            lane_width_m=float(waypoint.lane_width),
            lane_type=str(waypoint.lane_type),
            lane_change=str(waypoint.lane_change),
            same_direction_as_ego=_same_direction(ego_waypoint, waypoint),
            transform_carla=_transform_to_dict(waypoint.transform),
            transform_bevfusion_world=_transform_to_bevfusion_dict(waypoint.transform),
            lane_change_allowed_from_current=bool(lane_change_allowed_from_current),
        )

    def _candidate_side_lane(self, ego_waypoint: carla.Waypoint, side: str) -> carla.Waypoint | None:
        raw_candidate = ego_waypoint.get_left_lane() if side == "left" else ego_waypoint.get_right_lane()
        if raw_candidate is None:
            return None
        if raw_candidate.lane_type != carla.LaneType.Driving:
            return None
        if not _same_direction(ego_waypoint, raw_candidate):
            return None
        return raw_candidate

    def _build_forward_corridor(
        self,
        *,
        ego_waypoint: carla.Waypoint,
        horizon_m: float,
        waypoint_step_m: float,
    ) -> tuple[Dict[str, Any], JunctionContext]:
        samples: List[Dict[str, Any]] = []
        current_waypoint = ego_waypoint
        traveled = 0.0
        branch_count = 1
        branch_distance = None
        possible_turn_options: List[str] = []
        first_junction_distance = 0.0 if ego_waypoint.is_junction else None

        while traveled < float(horizon_m):
            samples.append(
                {
                    "distance_m": float(traveled),
                    "road_id": int(current_waypoint.road_id),
                    "section_id": int(current_waypoint.section_id),
                    "lane_id": int(current_waypoint.lane_id),
                    "is_junction": bool(current_waypoint.is_junction),
                    "yaw_deg": float(current_waypoint.transform.rotation.yaw),
                    "position_carla": _location_to_dict(current_waypoint.transform.location),
                    "position_bevfusion_world": {
                        "x": float(current_waypoint.transform.location.x),
                        "y": float(-current_waypoint.transform.location.y),
                        "z": float(current_waypoint.transform.location.z),
                    },
                }
            )

            next_waypoints = list(current_waypoint.next(float(waypoint_step_m)))
            if not next_waypoints:
                break
            if first_junction_distance is None:
                junction_candidates = [item for item in next_waypoints if item.is_junction]
                if junction_candidates:
                    first_junction_distance = float(traveled + waypoint_step_m)
            if branch_distance is None and len(next_waypoints) > 1:
                branch_count = int(len(next_waypoints))
                branch_distance = float(traveled + waypoint_step_m)
                possible_turn_options = sorted(
                    {
                        _turn_like_label(current_waypoint, item.next(3.0)[0] if item.next(3.0) else item)
                        for item in next_waypoints
                    }
                )

            def continuity_cost(candidate: carla.Waypoint) -> tuple[float, float]:
                label_rank = {"straight": 0.0, "left": 1.0, "right": 1.0}.get(
                    _turn_like_label(current_waypoint, candidate),
                    2.0,
                )
                current_forward = _forward_xy(current_waypoint)
                next_forward = _forward_xy(candidate)
                dot = current_forward[0] * next_forward[0] + current_forward[1] * next_forward[1]
                return (label_rank, -dot)

            next_waypoints.sort(key=continuity_cost)
            current_waypoint = next_waypoints[0]
            traveled += float(waypoint_step_m)

        corridor = {
            "horizon_m": float(horizon_m),
            "sample_step_m": float(waypoint_step_m),
            "chain": samples,
            "branch_count_ahead": int(branch_count),
            "branch_distance_m": None if branch_distance is None else float(branch_distance),
            "possible_turn_like_options": list(possible_turn_options),
            "reference_heading_deg": float(ego_waypoint.transform.rotation.yaw),
        }
        junction_context = JunctionContext(
            is_in_junction=bool(ego_waypoint.is_junction),
            junction_ahead=bool(ego_waypoint.is_junction or first_junction_distance is not None),
            distance_to_junction_m=None if first_junction_distance is None else float(first_junction_distance),
            branch_count_ahead=int(branch_count),
            possible_turn_like_options=list(possible_turn_options),
            branch_distance_m=None if branch_distance is None else float(branch_distance),
        )
        return corridor, junction_context

    def build_lane_context(
        self,
        *,
        frame_id: int,
        sample_name: str,
        ego_position_world_bevfusion: Iterable[float],
        horizon_m: float = 60.0,
        waypoint_step_m: float = 5.0,
    ) -> LaneContext:
        ego_probe = self.closest_waypoint_from_bevfusion_world(ego_position_world_bevfusion)
        ego_waypoint = ego_probe.waypoint
        if ego_waypoint is None:
            raise RuntimeError(
                f"Could not project ego pose to a driving waypoint for sample {sample_name} frame {frame_id}."
            )

        left_waypoint = self._candidate_side_lane(ego_waypoint, "left")
        right_waypoint = self._candidate_side_lane(ego_waypoint, "right")
        forward_corridor, junction_context = self._build_forward_corridor(
            ego_waypoint=ego_waypoint,
            horizon_m=horizon_m,
            waypoint_step_m=waypoint_step_m,
        )
        lane_context = LaneContext(
            frame_id=int(frame_id),
            sample_name=str(sample_name),
            current_lane=self._descriptor_from_waypoint(
                role="current_lane",
                waypoint=ego_waypoint,
                ego_waypoint=ego_waypoint,
                lane_change_allowed_from_current=False,
            ),
            left_lane=self._descriptor_from_waypoint(
                role="left_lane",
                waypoint=left_waypoint,
                ego_waypoint=ego_waypoint,
                lane_change_allowed_from_current=_lane_change_allowed(ego_waypoint, "left") if left_waypoint else False,
            ),
            right_lane=self._descriptor_from_waypoint(
                role="right_lane",
                waypoint=right_waypoint,
                ego_waypoint=ego_waypoint,
                lane_change_allowed_from_current=_lane_change_allowed(ego_waypoint, "right") if right_waypoint else False,
            ),
            forward_corridor=forward_corridor,
            junction_context=junction_context,
        )
        LOGGER.info(
            "Lane context frame=%d sample=%s road=%s lane=%s left=%s right=%s junction=%s",
            frame_id,
            sample_name,
            lane_context.current_lane.road_id,
            lane_context.current_lane.lane_id,
            lane_context.left_lane.exists,
            lane_context.right_lane.exists,
            lane_context.junction_context.is_in_junction,
        )
        return lane_context
