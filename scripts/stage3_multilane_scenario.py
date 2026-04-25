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
    spawn_parser.add_argument("--blocker-distance-m", type=float, default=24.0)
    spawn_parser.add_argument("--adjacent-distance-m", type=float, default=42.0)
    spawn_parser.add_argument("--ego-z-offset-m", type=float, default=0.4)
    spawn_parser.add_argument("--npc-handbrake", action="store_true")
    spawn_parser.add_argument("--output-manifest", required=True)

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


def _adjacent_lane(waypoint: carla.Waypoint, side: str) -> carla.Waypoint | None:
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


def _vehicle_blueprint(world: carla.World, pattern: str, role_name: str) -> carla.ActorBlueprint:
    blueprints = world.get_blueprint_library().filter(str(pattern))
    if not blueprints:
        raise RuntimeError(f"No blueprint matches {pattern}")
    blueprint = blueprints[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", str(role_name))
    return blueprint


def _spawn_vehicle(world: carla.World, blueprint: carla.ActorBlueprint, transform: carla.Transform) -> carla.Vehicle:
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is None:
        raise RuntimeError(
            f"Failed to spawn actor {blueprint.id} at "
            f"({transform.location.x:.2f}, {transform.location.y:.2f}, {transform.location.z:.2f})"
        )
    return actor  # type: ignore[return-value]


def _parking_brake(vehicle: carla.Vehicle) -> None:
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))


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
    traffic_manager.set_synchronous_mode(False)

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
        raise RuntimeError(f"No same-direction {args.adjacent_side} driving lane adjacent to ego waypoint.")

    ego_transform = carla.Transform(
        carla.Location(
            x=float(ego_wp.transform.location.x),
            y=float(ego_wp.transform.location.y),
            z=float(ego_wp.transform.location.z + float(args.ego_z_offset_m)),
        ),
        ego_wp.transform.rotation,
    )
    blocker_wp = _advance_along_lane(ego_wp, float(args.blocker_distance_m))
    adjacent_target_wp = _advance_along_lane(adjacent_wp, float(args.adjacent_distance_m))

    spawned_actors: list[carla.Actor] = []
    try:
        ego = _spawn_vehicle(world, _vehicle_blueprint(world, args.ego_filter, "hero"), ego_transform)
        spawned_actors.append(ego)
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
        adjacent_vehicle = _spawn_vehicle(
            world,
            _vehicle_blueprint(world, args.adjacent_filter, "stage3_adjacent"),
            carla.Transform(
                carla.Location(
                    x=float(adjacent_target_wp.transform.location.x),
                    y=float(adjacent_target_wp.transform.location.y),
                    z=float(adjacent_target_wp.transform.location.z + float(args.ego_z_offset_m)),
                ),
                adjacent_target_wp.transform.rotation,
            ),
        )
        spawned_actors.append(adjacent_vehicle)
    except Exception:
        for actor in reversed(spawned_actors):
            try:
                actor.destroy()
            except RuntimeError:
                LOGGER.warning("Failed to destroy partially spawned actor id=%s", getattr(actor, "id", "unknown"))
        raise
    if args.npc_handbrake:
        _parking_brake(blocker)
        _parking_brake(adjacent_vehicle)

    manifest = {
        "town": world.get_map().name,
        "host": str(args.host),
        "port": int(args.port),
        "tm_port": int(args.tm_port),
        "ego_actor_id": int(ego.id),
        "blocker_actor_id": int(blocker.id),
        "adjacent_actor_id": int(adjacent_vehicle.id),
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
        "Spawned scenario ego=%d blocker=%d adjacent=%d road=%d lane=%d side=%s manifest=%s",
        ego.id,
        blocker.id,
        adjacent_vehicle.id,
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
    actor_ids = [
        int(manifest.get("ego_actor_id", 0)),
        int(manifest.get("blocker_actor_id", 0)),
        int(manifest.get("adjacent_actor_id", 0)),
    ]
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
