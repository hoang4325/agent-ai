from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import carla

LOGGER = logging.getLogger("stage3_multilane_scenario")


@dataclass
class CorridorCandidate:
    road_id: int
    section_id: int
    lane_id: int
    s: float
    location: List[float]
    yaw_deg: float
    lane_width_m: float
    lane_change: str
    adjacent_side: str
    adjacent_lane_id: int
    adjacent_lane_marking: str
    adjacent_lane_change_allowed: bool
    forward_non_junction_length_m: float
    junction_distance_m: float | None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe and spawn a deterministic non-junction multi-lane CARLA scenario for Stage 3A coverage."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--town", default="Carla/Maps/Town10HD_Opt")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    probe_parser = subparsers.add_parser("probe", help="Find non-junction multi-lane corridors.")
    probe_parser.add_argument("--top-k", type=int, default=12)
    probe_parser.add_argument("--sample-distance-m", type=float, default=5.0)
    probe_parser.add_argument("--min-forward-length-m", type=float, default=80.0)
    probe_parser.add_argument("--adjacent-side", choices=("left", "right", "any"), default="any")
    probe_parser.add_argument("--output-json", default=None)

    spawn_parser = subparsers.add_parser("spawn", help="Spawn ego + blocker + adjacent lane actor.")
    spawn_parser.add_argument("--sample-distance-m", type=float, default=5.0)
    spawn_parser.add_argument("--min-forward-length-m", type=float, default=80.0)
    spawn_parser.add_argument("--adjacent-side", choices=("left", "right"), default="right")
    spawn_parser.add_argument("--road-id", type=int, default=None)
    spawn_parser.add_argument("--section-id", type=int, default=None)
    spawn_parser.add_argument("--lane-id", type=int, default=None)
    spawn_parser.add_argument("--s", type=float, default=None)
    spawn_parser.add_argument("--ego-filter", default="vehicle.lincoln.mkz_2020")
    spawn_parser.add_argument("--blocker-filter", default="vehicle.tesla.model3")
    spawn_parser.add_argument("--adjacent-filter", default="vehicle.audi.tt")
    spawn_parser.add_argument("--blocker-kind", choices=("vehicle", "pedestrian"), default="vehicle")
    spawn_parser.add_argument("--walker-filter", default="walker.pedestrian.*")
    spawn_parser.add_argument("--pedestrian-start-side", choices=("left", "right"), default="left")
    spawn_parser.add_argument("--pedestrian-cross-speed-mps", type=float, default=2.5)
    spawn_parser.add_argument("--pedestrian-cross-lateral-m", type=float, default=5.0)
    spawn_parser.add_argument("--blocker-distance-m", type=float, default=24.0)
    spawn_parser.add_argument("--adjacent-distance-m", type=float, default=42.0)
    spawn_parser.add_argument(
        "--adjacent-position",
        choices=("ahead", "behind", "both", "none"),
        default="ahead",
        help="Place adjacent-lane actor(s) ahead, behind, both around ego, or none for a clear adjacent lane.",
    )
    spawn_parser.add_argument(
        "--adjacent-front-distance-m",
        type=float,
        default=0.0,
        help="Front adjacent-lane actor distance when --adjacent-position=both; defaults to --adjacent-distance-m.",
    )
    spawn_parser.add_argument(
        "--adjacent-rear-distance-m",
        type=float,
        default=0.0,
        help="Rear adjacent-lane actor distance when --adjacent-position=both; defaults to 35m.",
    )
    spawn_parser.add_argument(
        "--moving-adjacent-npcs",
        action="store_true",
        help="Enable Traffic Manager autopilot for adjacent-lane NPCs while keeping the blocker static.",
    )
    spawn_parser.add_argument(
        "--adjacent-speed-diff-percent",
        type=float,
        default=-80.0,
        help="Traffic Manager speed difference for moving adjacent NPCs; positive means slower than speed limit.",
    )
    spawn_parser.add_argument("--ego-z-offset-m", type=float, default=0.4)
    spawn_parser.add_argument("--npc-handbrake", action="store_true")
    spawn_parser.add_argument("--output-manifest", required=True)
    spawn_parser.add_argument(
        "--cleanup-scenario-actors",
        action="store_true",
        help="Destroy stale Stage3/Stage10 scenario vehicles before spawning a fresh case.",
    )

    destroy_parser = subparsers.add_parser("destroy", help="Destroy actors from a previous scenario manifest.")
    destroy_parser.add_argument("--manifest", required=True)

    return parser.parse_args()


def _normalize_town_name(town: str) -> str:
    normalized = str(town).replace("\\", "/")
    if "/" in normalized:
        return normalized.split("/")[-1]
    return normalized


def connect_world(
    *,
    host: str,
    port: int,
    timeout_s: float,
    town: str | None,
) -> tuple[carla.Client, carla.World]:
    client = carla.Client(str(host), int(port))
    client.set_timeout(float(timeout_s))
    world = client.get_world()
    if town and str(world.get_map().name) != str(town):
        LOGGER.info("Loading town %s", town)
        world = client.load_world(_normalize_town_name(str(town)))
    LOGGER.info("Connected to CARLA map=%s", world.get_map().name)
    return client, world


def _forward_xy(waypoint: carla.Waypoint) -> tuple[float, float]:
    vector = waypoint.transform.get_forward_vector()
    norm = math.hypot(float(vector.x), float(vector.y))
    if norm <= 1e-6:
        return (1.0, 0.0)
    return (float(vector.x) / norm, float(vector.y) / norm)


def _same_direction(reference: carla.Waypoint, candidate: carla.Waypoint) -> bool:
    ref_x, ref_y = _forward_xy(reference)
    cand_x, cand_y = _forward_xy(candidate)
    return bool(ref_x * cand_x + ref_y * cand_y > 0.5)


def _carla_enum_name(value: Any) -> str:
    return str(value).split(".")[-1]


def _lane_change_name(waypoint: carla.Waypoint) -> str:
    return _carla_enum_name(getattr(waypoint, "lane_change", "unknown"))


def _lane_marking_type_name(waypoint: carla.Waypoint, side: str) -> str:
    marking = waypoint.left_lane_marking if side == "left" else waypoint.right_lane_marking
    return _carla_enum_name(getattr(marking, "type", "unknown"))


def _lane_change_enum_allows(waypoint: carla.Waypoint, side: str) -> bool:
    lane_change = _lane_change_name(waypoint).lower()
    return bool("both" in lane_change or side.lower() in lane_change)


def _lane_marking_allows_lane_change(waypoint: carla.Waypoint, side: str) -> bool:
    marking_type = _lane_marking_type_name(waypoint, side).lower()
    return bool("broken" in marking_type and "solid" not in marking_type)


def _lane_change_allowed(waypoint: carla.Waypoint, side: str) -> bool:
    return bool(_lane_change_enum_allows(waypoint, side) and _lane_marking_allows_lane_change(waypoint, side))


def _adjacent_lane(waypoint: carla.Waypoint, side: str) -> carla.Waypoint | None:
    if not _lane_change_allowed(waypoint, side):
        return None
    candidate = waypoint.get_left_lane() if side == "left" else waypoint.get_right_lane()
    if candidate is None:
        return None
    if candidate.lane_type != carla.LaneType.Driving:
        return None
    if not _same_direction(waypoint, candidate):
        return None
    return candidate


def _best_next_waypoint(current: carla.Waypoint, step_m: float) -> carla.Waypoint | None:
    next_waypoints = list(current.next(float(step_m)))
    if not next_waypoints:
        return None

    current_forward = _forward_xy(current)

    def continuity_cost(candidate: carla.Waypoint) -> float:
        cand_forward = _forward_xy(candidate)
        dot = current_forward[0] * cand_forward[0] + current_forward[1] * cand_forward[1]
        return -dot

    next_waypoints.sort(key=continuity_cost)
    return next_waypoints[0]


def _forward_non_junction_length(waypoint: carla.Waypoint, step_m: float, horizon_m: float = 120.0) -> float:
    distance = 0.0
    current = waypoint
    visited = set()
    while distance < float(horizon_m):
        key = (int(current.road_id), int(current.section_id), int(current.lane_id), round(float(current.s), 1))
        if key in visited:
            break
        visited.add(key)
        if current.is_junction:
            break
        next_wp = _best_next_waypoint(current, float(step_m))
        if next_wp is None:
            break
        distance += float(step_m)
        current = next_wp
        if current.is_junction:
            break
    return float(distance)


def _distance_to_junction(waypoint: carla.Waypoint, step_m: float, horizon_m: float = 120.0) -> float | None:
    distance = 0.0
    current = waypoint
    visited = set()
    if current.is_junction:
        return 0.0
    while distance < float(horizon_m):
        key = (int(current.road_id), int(current.section_id), int(current.lane_id), round(float(current.s), 1))
        if key in visited:
            break
        visited.add(key)
        next_wp = _best_next_waypoint(current, float(step_m))
        if next_wp is None:
            return None
        distance += float(step_m)
        current = next_wp
        if current.is_junction:
            return float(distance)
    return None


def find_corridor_candidates(
    *,
    world: carla.World,
    sample_distance_m: float,
    min_forward_length_m: float,
    adjacent_side: str = "any",
) -> List[CorridorCandidate]:
    map_obj = world.get_map()
    candidates: List[CorridorCandidate] = []
    for waypoint in map_obj.generate_waypoints(float(sample_distance_m)):
        if waypoint.lane_type != carla.LaneType.Driving or waypoint.is_junction:
            continue

        chosen_side = None
        adjacent = None
        for side in (("left", "right") if adjacent_side == "any" else (adjacent_side,)):
            candidate_lane = _adjacent_lane(waypoint, side)
            if candidate_lane is not None:
                chosen_side = side
                adjacent = candidate_lane
                break
        if adjacent is None or chosen_side is None:
            continue

        forward_length = _forward_non_junction_length(waypoint, float(sample_distance_m))
        if forward_length < float(min_forward_length_m):
            continue

        candidates.append(
            CorridorCandidate(
                road_id=int(waypoint.road_id),
                section_id=int(waypoint.section_id),
                lane_id=int(waypoint.lane_id),
                s=float(waypoint.s),
                location=[
                    float(waypoint.transform.location.x),
                    float(waypoint.transform.location.y),
                    float(waypoint.transform.location.z),
                ],
                yaw_deg=float(waypoint.transform.rotation.yaw),
                lane_width_m=float(waypoint.lane_width),
                lane_change=str(waypoint.lane_change),
                adjacent_side=str(chosen_side),
                adjacent_lane_id=int(adjacent.lane_id),
                adjacent_lane_marking=_lane_marking_type_name(waypoint, str(chosen_side)),
                adjacent_lane_change_allowed=_lane_change_allowed(waypoint, str(chosen_side)),
                forward_non_junction_length_m=float(forward_length),
                junction_distance_m=_distance_to_junction(waypoint, float(sample_distance_m)),
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.forward_non_junction_length_m,
            item.road_id,
            item.section_id,
            item.lane_id,
            item.s,
        )
    )
    return candidates


def _find_matching_waypoint(
    *,
    world: carla.World,
    road_id: int,
    section_id: int | None,
    lane_id: int,
    s: float | None,
    sample_distance_m: float,
) -> carla.Waypoint:
    best = None
    best_cost = None
    for waypoint in world.get_map().generate_waypoints(float(sample_distance_m)):
        if int(waypoint.road_id) != int(road_id):
            continue
        if section_id is not None and int(waypoint.section_id) != int(section_id):
            continue
        if int(waypoint.lane_id) != int(lane_id):
            continue
        cost = 0.0 if s is None else abs(float(waypoint.s) - float(s))
        if best is None or cost < float(best_cost):
            best = waypoint
            best_cost = cost
    if best is None:
        raise RuntimeError(
            f"Could not find waypoint for road_id={road_id} section_id={section_id} lane_id={lane_id} s={s}"
        )
    return best


def _advance_along_lane(start: carla.Waypoint, distance_m: float, step_m: float = 2.0) -> carla.Waypoint:
    current = start
    traveled = 0.0
    while traveled + 1e-6 < float(distance_m):
        next_wp = _best_next_waypoint(current, float(step_m))
        if next_wp is None:
            break
        current = next_wp
        traveled += float(step_m)
    return current


def _best_previous_waypoint(current: carla.Waypoint, step_m: float) -> carla.Waypoint | None:
    previous_waypoints = list(current.previous(float(step_m)))
    if not previous_waypoints:
        return None

    current_forward = _forward_xy(current)

    def continuity_cost(candidate: carla.Waypoint) -> float:
        cand_forward = _forward_xy(candidate)
        dot = current_forward[0] * cand_forward[0] + current_forward[1] * cand_forward[1]
        return -dot

    previous_waypoints.sort(key=continuity_cost)
    return previous_waypoints[0]


def _retreat_along_lane(start: carla.Waypoint, distance_m: float, step_m: float = 2.0) -> carla.Waypoint:
    current = start
    traveled = 0.0
    while traveled + 1e-6 < float(distance_m):
        previous_wp = _best_previous_waypoint(current, float(step_m))
        if previous_wp is None:
            break
        current = previous_wp
        traveled += float(step_m)
    return current


def _vehicle_blueprint(world: carla.World, pattern: str, role_name: str) -> carla.ActorBlueprint:
    blueprints = world.get_blueprint_library().filter(str(pattern))
    if not blueprints:
        raise RuntimeError(f"No blueprint matches {pattern}")
    blueprint = blueprints[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", str(role_name))
    return blueprint


def _walker_blueprint(world: carla.World, pattern: str) -> carla.ActorBlueprint:
    blueprints = world.get_blueprint_library().filter(str(pattern))
    if not blueprints:
        raise RuntimeError(f"No walker blueprint matches {pattern}")
    blueprint = blueprints[0]
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "false")
    return blueprint


def _spawn_vehicle(world: carla.World, blueprint: carla.ActorBlueprint, transform: carla.Transform) -> carla.Vehicle:
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is None:
        raise RuntimeError(
            f"Failed to spawn actor {blueprint.id} at "
            f"({transform.location.x:.2f}, {transform.location.y:.2f}, {transform.location.z:.2f})"
        )
    return actor  # type: ignore[return-value]


def _spawn_walker(
    world: carla.World,
    *,
    blueprint: carla.ActorBlueprint,
    transform: carla.Transform,
    destination: carla.Location,
    speed_mps: float,
) -> tuple[carla.Walker, carla.Actor | None]:
    walker = world.try_spawn_actor(blueprint, transform)
    if walker is None:
        raise RuntimeError(
            f"Failed to spawn walker {blueprint.id} at "
            f"({transform.location.x:.2f}, {transform.location.y:.2f}, {transform.location.z:.2f})"
        )
    controller = None
    try:
        controller_bp = world.get_blueprint_library().find("controller.ai.walker")
        controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
        controller.start()
        controller.go_to_location(destination)
        controller.set_max_speed(max(0.1, float(speed_mps)))
    except Exception:
        if controller is not None:
            controller.destroy()
        walker.destroy()
        raise
    return walker, controller


def _parking_brake(vehicle: carla.Vehicle) -> None:
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))


def _release_parking_brake(vehicle: carla.Vehicle) -> None:
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False))


_SCENARIO_ROLE_NAMES = {
    "hero",
    "stage3_blocker",
    "stage3_adjacent_front",
    "stage3_adjacent_rear",
}


def _cleanup_scenario_role_actors(world: carla.World) -> None:
    destroyed: list[int] = []
    for actor in list(world.get_actors()):
        role_name = str(actor.attributes.get("role_name", ""))
        if role_name not in _SCENARIO_ROLE_NAMES:
            continue
        try:
            actor.destroy()
            destroyed.append(int(actor.id))
        except RuntimeError:
            LOGGER.warning("Failed to cleanup stale scenario actor id=%s role=%s", actor.id, role_name)
    if destroyed:
        LOGGER.info("Cleaned up stale scenario actors before spawn: %s", destroyed)


def _adjacent_desired_speed_kmh(speed_diff_percent: float) -> float:
    faster_pct = max(0.0, -float(speed_diff_percent))
    return max(80.0, min(110.0, 70.0 + faster_pct * 0.25))


def _enable_adjacent_autopilot(
    *,
    vehicle: carla.Vehicle | None,
    traffic_manager: carla.TrafficManager,
    tm_port: int,
    speed_diff_percent: float,
) -> None:
    if vehicle is None:
        return
    _release_parking_brake(vehicle)
    vehicle.set_autopilot(True, int(tm_port))
    try:
        traffic_manager.auto_lane_change(vehicle, False)
    except RuntimeError:
        LOGGER.warning("Failed to disable auto lane change for adjacent actor id=%s", int(vehicle.id))
    try:
        traffic_manager.vehicle_percentage_speed_difference(vehicle, float(speed_diff_percent))
    except RuntimeError:
        LOGGER.warning("Failed to set speed difference for adjacent actor id=%s", int(vehicle.id))
    for action_name in ("ignore_lights_percentage", "ignore_signs_percentage"):
        try:
            getattr(traffic_manager, action_name)(vehicle, 100.0)
        except (AttributeError, RuntimeError):
            LOGGER.warning("Failed to set %s for adjacent actor id=%s", action_name, int(vehicle.id))
    try:
        desired_speed_kmh = _adjacent_desired_speed_kmh(float(speed_diff_percent))
        traffic_manager.set_desired_speed(vehicle, desired_speed_kmh)
        LOGGER.info(
            "Adjacent NPC id=%s autopilot speed_diff=%.1f%% desired_speed=%.1f km/h",
            int(vehicle.id),
            float(speed_diff_percent),
            desired_speed_kmh,
        )
    except (AttributeError, RuntimeError):
        LOGGER.debug("Traffic Manager set_desired_speed unavailable for adjacent actor id=%s", int(vehicle.id))
    try:
        traffic_manager.distance_to_leading_vehicle(vehicle, 8.0)
    except RuntimeError:
        LOGGER.warning("Failed to set following distance for adjacent actor id=%s", int(vehicle.id))


def _actor_id_or_zero(actor: carla.Actor | None) -> int:
    return int(actor.id) if actor is not None else 0


def _right_xy_from_yaw(yaw_deg: float) -> tuple[float, float]:
    yaw_rad = math.radians(float(yaw_deg))
    return (math.sin(yaw_rad), -math.cos(yaw_rad))


def _offset_location(
    base: carla.Location,
    right_xy: tuple[float, float],
    lateral_m: float,
    *,
    z_offset_m: float = 0.0,
) -> carla.Location:
    return carla.Location(
        x=float(base.x) + float(right_xy[0]) * float(lateral_m),
        y=float(base.y) + float(right_xy[1]) * float(lateral_m),
        z=float(base.z) + float(z_offset_m),
    )


def command_probe(args: argparse.Namespace) -> int:
    _, world = connect_world(host=args.host, port=args.port, timeout_s=args.timeout_s, town=args.town)
    candidates = find_corridor_candidates(
        world=world,
        sample_distance_m=float(args.sample_distance_m),
        min_forward_length_m=float(args.min_forward_length_m),
        adjacent_side=str(args.adjacent_side),
    )
    top_candidates = candidates[: int(args.top_k)]
    LOGGER.info("Found %d corridor candidates. Showing top %d.", len(candidates), len(top_candidates))
    for index, item in enumerate(top_candidates):
        print(
            json.dumps(
                {
                    "rank": index,
                    **item.to_dict(),
                },
                indent=2,
            )
        )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([item.to_dict() for item in top_candidates], indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Wrote candidate summary to %s", output_path)
    return 0


def command_spawn(args: argparse.Namespace) -> int:
    client, world = connect_world(host=args.host, port=args.port, timeout_s=args.timeout_s, town=args.town)
    traffic_manager = client.get_trafficmanager(int(args.tm_port))
    traffic_manager.set_synchronous_mode(bool(args.moving_adjacent_npcs))
    if bool(args.cleanup_scenario_actors):
        _cleanup_scenario_role_actors(world)

    if args.road_id is None or args.lane_id is None:
        candidates = find_corridor_candidates(
            world=world,
            sample_distance_m=float(args.sample_distance_m),
            min_forward_length_m=float(args.min_forward_length_m),
            adjacent_side=str(args.adjacent_side),
        )
        if not candidates:
            raise RuntimeError("Could not auto-pick any non-junction multi-lane corridor candidate.")
        chosen = candidates[0]
        road_id = chosen.road_id
        section_id = chosen.section_id
        lane_id = chosen.lane_id
        s = chosen.s
        LOGGER.info("Auto-selected corridor candidate: %s", json.dumps(chosen.to_dict(), indent=2))
    else:
        road_id = int(args.road_id)
        section_id = None if args.section_id is None else int(args.section_id)
        lane_id = int(args.lane_id)
        s = None if args.s is None else float(args.s)

    ego_wp = _find_matching_waypoint(
        world=world,
        road_id=int(road_id),
        section_id=section_id,
        lane_id=int(lane_id),
        s=s,
        sample_distance_m=float(args.sample_distance_m),
    )
    if ego_wp.is_junction:
        raise RuntimeError("Chosen ego waypoint lies inside a junction. Pick another corridor waypoint.")
    adjacent_wp = _adjacent_lane(ego_wp, str(args.adjacent_side))
    if adjacent_wp is None:
        marking = _lane_marking_type_name(ego_wp, str(args.adjacent_side))
        lane_change = _lane_change_name(ego_wp)
        raise RuntimeError(
            f"No legal dashed-line same-direction {args.adjacent_side} driving lane adjacent to ego waypoint "
            f"(marking={marking}, lane_change={lane_change})."
        )

    ego_transform = carla.Transform(
        carla.Location(
            x=float(ego_wp.transform.location.x),
            y=float(ego_wp.transform.location.y),
            z=float(ego_wp.transform.location.z + float(args.ego_z_offset_m)),
        ),
        ego_wp.transform.rotation,
    )
    blocker_wp = _advance_along_lane(ego_wp, float(args.blocker_distance_m))
    adjacent_position = str(args.adjacent_position)
    adjacent_front_distance_m = (
        float(args.adjacent_front_distance_m)
        if float(args.adjacent_front_distance_m) > 0.0
        else float(args.adjacent_distance_m)
    )
    adjacent_rear_distance_m = (
        float(args.adjacent_rear_distance_m)
        if float(args.adjacent_rear_distance_m) > 0.0
        else 35.0
    )
    adjacent_front_wp = _advance_along_lane(adjacent_wp, adjacent_front_distance_m)
    adjacent_rear_wp = _retreat_along_lane(adjacent_wp, adjacent_rear_distance_m)

    spawned_actors: list[carla.Actor] = []
    spawned_controllers: list[carla.Actor] = []
    blocker: carla.Actor | None = None
    blocker_controller: carla.Actor | None = None
    adjacent_vehicle: carla.Vehicle | None = None
    adjacent_front_vehicle: carla.Vehicle | None = None
    adjacent_rear_vehicle: carla.Vehicle | None = None
    try:
        ego = _spawn_vehicle(world, _vehicle_blueprint(world, args.ego_filter, "hero"), ego_transform)
        spawned_actors.append(ego)
        if str(args.blocker_kind) == "pedestrian":
            right_xy = _right_xy_from_yaw(float(blocker_wp.transform.rotation.yaw))
            start_sign = -1.0 if str(args.pedestrian_start_side) == "left" else 1.0
            cross_lateral_m = abs(float(args.pedestrian_cross_lateral_m))
            start_location = _offset_location(
                blocker_wp.transform.location,
                right_xy,
                start_sign * cross_lateral_m,
                z_offset_m=0.1,
            )
            destination = _offset_location(
                blocker_wp.transform.location,
                right_xy,
                -start_sign * cross_lateral_m,
                z_offset_m=0.1,
            )
            cross_yaw = float(blocker_wp.transform.rotation.yaw) + (90.0 if start_sign < 0 else -90.0)
            blocker, blocker_controller = _spawn_walker(
                world,
                blueprint=_walker_blueprint(world, args.walker_filter),
                transform=carla.Transform(start_location, carla.Rotation(yaw=cross_yaw)),
                destination=destination,
                speed_mps=float(args.pedestrian_cross_speed_mps),
            )
            if blocker_controller is not None:
                spawned_controllers.append(blocker_controller)
        else:
            blocker = _spawn_vehicle(
                world,
                _vehicle_blueprint(world, args.blocker_filter, "stage3_blocker"),
                carla.Transform(
                    carla.Location(
                        x=float(blocker_wp.transform.location.x),
                        y=float(blocker_wp.transform.location.y),
                        z=float(blocker_wp.transform.location.z + float(args.ego_z_offset_m)),
                    ),
                    blocker_wp.transform.rotation,
                ),
            )
        spawned_actors.append(blocker)
        if adjacent_position in {"ahead", "both"}:
            adjacent_front_vehicle = _spawn_vehicle(
                world,
                _vehicle_blueprint(world, args.adjacent_filter, "stage3_adjacent_front"),
                carla.Transform(
                    carla.Location(
                        x=float(adjacent_front_wp.transform.location.x),
                        y=float(adjacent_front_wp.transform.location.y),
                        z=float(adjacent_front_wp.transform.location.z + float(args.ego_z_offset_m)),
                    ),
                    adjacent_front_wp.transform.rotation,
                ),
            )
            spawned_actors.append(adjacent_front_vehicle)
        if adjacent_position in {"behind", "both"}:
            adjacent_rear_vehicle = _spawn_vehicle(
                world,
                _vehicle_blueprint(world, args.adjacent_filter, "stage3_adjacent_rear"),
                carla.Transform(
                    carla.Location(
                        x=float(adjacent_rear_wp.transform.location.x),
                        y=float(adjacent_rear_wp.transform.location.y),
                        z=float(adjacent_rear_wp.transform.location.z + float(args.ego_z_offset_m)),
                    ),
                    adjacent_rear_wp.transform.rotation,
                ),
            )
            spawned_actors.append(adjacent_rear_vehicle)
        adjacent_vehicle = adjacent_front_vehicle or adjacent_rear_vehicle
        if adjacent_vehicle is None and adjacent_position != "none":
            raise RuntimeError(f"No adjacent-lane actor spawned for adjacent_position={adjacent_position}")
    except Exception:
        for controller in reversed(spawned_controllers):
            try:
                controller.destroy()
            except RuntimeError:
                LOGGER.warning("Failed to destroy partially spawned controller id=%s", getattr(controller, "id", "unknown"))
        for actor in reversed(spawned_actors):
            try:
                actor.destroy()
            except RuntimeError:
                LOGGER.warning("Failed to destroy partially spawned actor id=%s", getattr(actor, "id", "unknown"))
        raise
    if bool(args.moving_adjacent_npcs):
        _enable_adjacent_autopilot(
            vehicle=adjacent_front_vehicle,
            traffic_manager=traffic_manager,
            tm_port=int(args.tm_port),
            speed_diff_percent=float(args.adjacent_speed_diff_percent),
        )
        _enable_adjacent_autopilot(
            vehicle=adjacent_rear_vehicle,
            traffic_manager=traffic_manager,
            tm_port=int(args.tm_port),
            speed_diff_percent=float(args.adjacent_speed_diff_percent),
        )

    if args.npc_handbrake:
        if isinstance(blocker, carla.Vehicle):
            _parking_brake(blocker)
        if adjacent_front_vehicle is not None and not bool(args.moving_adjacent_npcs):
            _parking_brake(adjacent_front_vehicle)
        if adjacent_rear_vehicle is not None and not bool(args.moving_adjacent_npcs):
            _parking_brake(adjacent_rear_vehicle)

    manifest = {
        "town": world.get_map().name,
        "host": str(args.host),
        "port": int(args.port),
        "tm_port": int(args.tm_port),
        "ego_actor_id": int(ego.id),
        "blocker_actor_id": int(blocker.id if blocker is not None else 0),
        "blocker_kind": str(args.blocker_kind),
        "walker_controller_actor_id": _actor_id_or_zero(blocker_controller),
        "adjacent_actor_id": _actor_id_or_zero(adjacent_vehicle),
        "adjacent_front_actor_id": _actor_id_or_zero(adjacent_front_vehicle),
        "adjacent_rear_actor_id": _actor_id_or_zero(adjacent_rear_vehicle),
        "adjacent_actor_ids": [
            int(actor_id)
            for actor_id in [
                _actor_id_or_zero(adjacent_front_vehicle),
                _actor_id_or_zero(adjacent_rear_vehicle),
            ]
            if int(actor_id) > 0
        ],
        "adjacent_side": str(args.adjacent_side),
        "scenario_type": "stage3a_non_junction_multilane",
        "corridor": {
            "road_id": int(ego_wp.road_id),
            "section_id": int(ego_wp.section_id),
            "lane_id": int(ego_wp.lane_id),
            "s": float(ego_wp.s),
            "lane_width_m": float(ego_wp.lane_width),
            "lane_change": str(ego_wp.lane_change),
            "adjacent_lane_id": int(adjacent_wp.lane_id),
            "adjacent_lane_marking": _lane_marking_type_name(ego_wp, str(args.adjacent_side)),
            "adjacent_lane_change_allowed": _lane_change_allowed(ego_wp, str(args.adjacent_side)),
            "junction_distance_m": _distance_to_junction(ego_wp, float(args.sample_distance_m)),
            "ego_transform": {
                "location": {
                    "x": float(ego_transform.location.x),
                    "y": float(ego_transform.location.y),
                    "z": float(ego_transform.location.z),
                },
                "rotation": {
                    "roll": float(ego_transform.rotation.roll),
                    "pitch": float(ego_transform.rotation.pitch),
                    "yaw": float(ego_transform.rotation.yaw),
                },
            },
        },
        "placements": {
            "blocker_distance_m": float(args.blocker_distance_m),
            "adjacent_distance_m": float(args.adjacent_distance_m),
            "adjacent_position": adjacent_position,
            "adjacent_front_distance_m": adjacent_front_distance_m,
            "adjacent_rear_distance_m": adjacent_rear_distance_m,
            "moving_adjacent_npcs": bool(args.moving_adjacent_npcs),
            "adjacent_speed_diff_percent": float(args.adjacent_speed_diff_percent),
            "pedestrian_start_side": str(args.pedestrian_start_side),
            "pedestrian_cross_speed_mps": float(args.pedestrian_cross_speed_mps),
            "pedestrian_cross_lateral_m": float(args.pedestrian_cross_lateral_m),
        },
    }
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest["recommended_commands"] = {
        "stage1_dump_from_manifest": (
            f"python D:\\Agent-AI\\scripts\\run_carla_dump.py --host {args.host} --port {args.port} "
            f"--tm-port {args.tm_port} --output-root D:\\Agent-AI\\outputs\\samples_stage3a_multilane_right "
            f"--num-samples 80 --fixed-delta-seconds 0.05 --image-width 1600 --image-height 900 "
            f"--camera-fov 70 --scenario-manifest {output_manifest} --autopilot"
        ),
        "stage1_dump_attach_id": (
            f"python D:\\Agent-AI\\scripts\\run_carla_dump.py --host {args.host} --port {args.port} "
            f"--tm-port {args.tm_port} --output-root D:\\Agent-AI\\outputs\\samples_stage3a_multilane_right "
            f"--num-samples 80 --fixed-delta-seconds 0.05 --image-width 1600 --image-height 900 "
            f"--camera-fov 70 --attach-to-actor-id {ego.id} --autopilot"
        ),
        "stage1_dump_stable_from_manifest": (
            f"python D:\\Agent-AI\\scripts\\run_carla_dump.py --host {args.host} --port {args.port} "
            f"--tm-port {args.tm_port} --output-root D:\\Agent-AI\\outputs\\samples_stage3a_multilane_right "
            f"--num-samples 60 --fixed-delta-seconds 0.05 --image-width 1280 --image-height 720 "
            f"--camera-fov 70 --warmup-ticks 8 --scenario-manifest {output_manifest} --autopilot"
        ),
        "stage1_dump_ultrastable_from_manifest": (
            f"python D:\\Agent-AI\\scripts\\run_carla_dump.py --host {args.host} --port {args.port} "
            f"--tm-port {args.tm_port} --output-root D:\\Agent-AI\\outputs\\samples_stage3a_multilane_right "
            f"--num-samples 50 --fixed-delta-seconds 0.10 --image-width 960 --image-height 540 "
            f"--camera-fov 70 --warmup-ticks 10 --shutdown-settle-ticks 4 "
            f"--scenario-manifest {output_manifest} --autopilot"
        ),
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info(
        "Spawned scenario ego=%d blocker=%d adjacent_front=%d adjacent_rear=%d road=%d lane=%d side=%s manifest=%s",
        ego.id,
        blocker.id if blocker is not None else 0,
        _actor_id_or_zero(adjacent_front_vehicle),
        _actor_id_or_zero(adjacent_rear_vehicle),
        ego_wp.road_id,
        ego_wp.lane_id,
        args.adjacent_side,
        output_manifest,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def command_destroy(args: argparse.Namespace) -> int:
    client, world = connect_world(host=args.host, port=args.port, timeout_s=args.timeout_s, town=None)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    actor_ids = []
    for key in [
        "ego_actor_id",
        "blocker_actor_id",
        "walker_controller_actor_id",
        "adjacent_actor_id",
        "adjacent_front_actor_id",
        "adjacent_rear_actor_id",
    ]:
        actor_id = int(manifest.get(key, 0) or 0)
        if actor_id > 0 and actor_id not in actor_ids:
            actor_ids.append(actor_id)
    for actor_id in manifest.get("adjacent_actor_ids", []) or []:
        actor_id = int(actor_id or 0)
        if actor_id > 0 and actor_id not in actor_ids:
            actor_ids.append(actor_id)
    destroyed = []
    for actor_id in actor_ids:
        if actor_id <= 0:
            continue
        actor = world.get_actor(actor_id)
        if actor is None:
            LOGGER.warning("Actor id %d not found", actor_id)
            continue
        actor.destroy()
        destroyed.append(actor_id)
    LOGGER.info("Destroyed actors: %s", destroyed)
    return 0


def main() -> int:
    args = parse_args()
    configure_logging()
    if args.mode == "probe":
        return command_probe(args)
    if args.mode == "spawn":
        return command_spawn(args)
    if args.mode == "destroy":
        return command_destroy(args)
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
