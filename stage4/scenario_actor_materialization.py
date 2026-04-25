from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_town_name(town: str) -> str:
    normalized = str(town).replace("\\", "/")
    return normalized.split("/")[-1]


def _forward_xy(waypoint: Any) -> tuple[float, float]:
    vector = waypoint.transform.get_forward_vector()
    norm = math.hypot(float(vector.x), float(vector.y))
    if norm <= 1e-6:
        return (1.0, 0.0)
    return (float(vector.x) / norm, float(vector.y) / norm)


def _same_direction(reference: Any, candidate: Any) -> bool:
    ref_x, ref_y = _forward_xy(reference)
    cand_x, cand_y = _forward_xy(candidate)
    return bool(ref_x * cand_x + ref_y * cand_y > 0.5)


def _adjacent_lane(waypoint: Any, side: str) -> Any | None:
    candidate = waypoint.get_left_lane() if side == "left" else waypoint.get_right_lane()
    if candidate is None:
        return None
    import carla  # type: ignore

    if candidate.lane_type != carla.LaneType.Driving:
        return None
    if not _same_direction(waypoint, candidate):
        return None
    return candidate


def _best_next_waypoint(current: Any, step_m: float) -> Any | None:
    next_waypoints = list(current.next(float(step_m)))
    if not next_waypoints:
        return None
    current_forward = _forward_xy(current)

    def continuity_cost(candidate: Any) -> float:
        cand_forward = _forward_xy(candidate)
        dot = current_forward[0] * cand_forward[0] + current_forward[1] * cand_forward[1]
        return -dot

    next_waypoints.sort(key=continuity_cost)
    return next_waypoints[0]


def _advance_along_lane(start: Any, distance_m: float, step_m: float = 2.0) -> Any:
    current = start
    traveled = 0.0
    while traveled + 1e-6 < float(distance_m):
        next_wp = _best_next_waypoint(current, float(step_m))
        if next_wp is None:
            break
        current = next_wp
        traveled += float(step_m)
    return current


def _find_matching_waypoint(
    *,
    world: Any,
    road_id: int,
    section_id: int | None,
    lane_id: int,
    s: float | None,
    sample_distance_m: float = 2.0,
) -> Any:
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


def _build_transform(location_payload: Dict[str, Any], rotation_payload: Dict[str, Any]) -> Any:
    import carla  # type: ignore

    return carla.Transform(
        carla.Location(
            x=float(location_payload.get("x", location_payload.get("x_m", 0.0))),
            y=float(location_payload.get("y", location_payload.get("y_m", 0.0))),
            z=float(location_payload.get("z", location_payload.get("z_m", 0.0))),
        ),
        carla.Rotation(
            roll=float(rotation_payload.get("roll", 0.0)),
            pitch=float(rotation_payload.get("pitch", 0.0)),
            yaw=float(rotation_payload.get("yaw", rotation_payload.get("yaw_deg", 0.0))),
        ),
    )


def _offset_transform(base_transform: Any, *, forward_m: float = 0.0, lateral_m: float = 0.0) -> Any:
    import carla  # type: ignore

    yaw_rad = math.radians(float(base_transform.rotation.yaw))
    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)
    right_x = math.cos(yaw_rad + (math.pi / 2.0))
    right_y = math.sin(yaw_rad + (math.pi / 2.0))
    return carla.Transform(
        carla.Location(
            x=float(base_transform.location.x + forward_x * float(forward_m) + right_x * float(lateral_m)),
            y=float(base_transform.location.y + forward_y * float(forward_m) + right_y * float(lateral_m)),
            z=float(base_transform.location.z),
        ),
        base_transform.rotation,
    )


def _transform_from_payload(transform_payload: Dict[str, Any] | None, *, z_offset_m: float | None = None) -> Any | None:
    if not isinstance(transform_payload, dict):
        return None
    location = transform_payload.get("location")
    rotation = transform_payload.get("rotation")
    if not isinstance(location, dict) or not isinstance(rotation, dict):
        return None
    if z_offset_m is not None:
        location = dict(location)
        location["z"] = float(location.get("z", location.get("z_m", 0.0))) + float(z_offset_m)
    return _build_transform(location, rotation)


def _transform_to_payload(transform: Any) -> Dict[str, Any]:
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


def _distance_between_transforms(transform_a: Any, transform_b: Any) -> float:
    location_a = transform_a.location
    location_b = transform_b.location
    return float(location_a.distance(location_b))


def _lane_id_for_actor(map_inst: Any, actor: Any) -> int | None:
    if actor is None:
        return None
    waypoint = map_inst.get_waypoint(actor.get_location())
    return None if waypoint is None else int(waypoint.lane_id)


def _actor_distance_to_expected(actor: Any, expected_transform: Any) -> float | None:
    if actor is None or expected_transform is None:
        return None
    try:
        return _distance_between_transforms(actor.get_transform(), expected_transform)
    except RuntimeError:
        return None


def _actor_exact_match(actor: Any, expected_transform: Any, expected_lane_id: int | None, map_inst: Any) -> bool:
    distance = _actor_distance_to_expected(actor, expected_transform)
    observed_lane_id = _lane_id_for_actor(map_inst, actor)
    if distance is None:
        return False
    if distance > 6.0:
        return False
    if expected_lane_id is not None and observed_lane_id != expected_lane_id:
        return False
    return True


def _vehicle_blueprint(world: Any, pattern: str, role_name: str) -> Any:
    blueprints = world.get_blueprint_library().filter(str(pattern))
    if not blueprints:
        blueprints = world.get_blueprint_library().filter("vehicle.*")
    if not blueprints:
        raise RuntimeError(f"No vehicle blueprint matches {pattern}")
    blueprint = blueprints[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", str(role_name))
    return blueprint


def _apply_stationary_brake(vehicle: Any, *, hand_brake: bool) -> None:
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
            hand_brake=bool(hand_brake),
            manual_gear_shift=False,
        )
    )


def _realign_vehicle(vehicle: Any, transform: Any, *, hand_brake: bool) -> None:
    vehicle.set_autopilot(False)
    if hasattr(vehicle, "set_simulate_physics"):
        vehicle.set_simulate_physics(True)
    vehicle.set_transform(transform)
    _apply_stationary_brake(vehicle, hand_brake=hand_brake)


def _spawn_vehicle(world: Any, blueprint_filter: str, role_name: str, transform: Any, *, hand_brake: bool) -> Any:
    blueprint = _vehicle_blueprint(world, blueprint_filter, role_name)
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is None:
        raise RuntimeError(
            f"Failed to spawn {role_name} at "
            f"({transform.location.x:.2f}, {transform.location.y:.2f}, {transform.location.z:.2f})"
        )
    _apply_stationary_brake(actor, hand_brake=hand_brake)
    return actor


def _find_live_scene_candidate(
    world: Any,
    *,
    expected_transform: Any,
    expected_lane_id: int | None,
    map_inst: Any,
    exclude_actor_ids: set[int],
) -> Any | None:
    best_actor = None
    best_distance = None
    for actor in world.get_actors().filter("vehicle.*"):
        actor_id = _safe_int(getattr(actor, "id", None))
        if actor_id is None or actor_id in exclude_actor_ids:
            continue
        if not _actor_exact_match(actor, expected_transform, expected_lane_id, map_inst):
            continue
        distance = _actor_distance_to_expected(actor, expected_transform)
        if distance is None:
            continue
        if best_actor is None or float(distance) < float(best_distance):
            best_actor = actor
            best_distance = float(distance)
    return best_actor


@dataclass
class ActorBinding:
    role: str
    manifest_actor_id: int | None
    runtime_actor_id: int | None
    binding_status: str
    binding_source: str
    materialization_status: str
    target_source_type: str | None = None
    expected_lane_id: int | None = None
    observed_lane_id: int | None = None
    distance_to_expected_m: float | None = None
    blueprint_id: str | None = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


@dataclass
class MaterializationResult:
    ego_vehicle: Any | None
    scenario_manifest_payload: Dict[str, Any] | None
    spawned_ego: bool
    spawned_actors: List[Any] = field(default_factory=list)
    actor_bindings: Dict[str, ActorBinding] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": "stage4.critical_actor_materialization.v1",
            "scenario_type": None if not self.scenario_manifest_payload else self.scenario_manifest_payload.get("scenario_type"),
            "spawned_ego": bool(self.spawned_ego),
            "spawned_actor_ids": [int(actor.id) for actor in self.spawned_actors if getattr(actor, "id", None) is not None],
            "actor_bindings": {role: binding.to_dict() for role, binding in self.actor_bindings.items()},
            "notes": list(self.notes),
        }


def _expected_multilane_specs(manifest_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    corridor = dict(manifest_payload.get("corridor") or {})
    placements = dict(manifest_payload.get("placements") or {})
    ego_transform_payload = corridor.get("ego_transform")
    if not corridor or not placements or not isinstance(ego_transform_payload, dict):
        return {}

    road_id = _safe_int(corridor.get("road_id"))
    lane_id = _safe_int(corridor.get("lane_id"))
    if road_id is None or lane_id is None:
        return {}
    section_id = _safe_int(corridor.get("section_id"))
    s = _safe_float(corridor.get("s"))
    adjacent_side = str(manifest_payload.get("adjacent_side") or "right")
    blocker_distance_m = _safe_float(placements.get("blocker_distance_m"))
    adjacent_distance_m = _safe_float(placements.get("adjacent_distance_m"))
    return {
        "ego": {
            "manifest_actor_id": _safe_int(manifest_payload.get("ego_actor_id")),
            "road_id": road_id,
            "section_id": section_id,
            "lane_id": lane_id,
            "s": s,
            "transform_payload": ego_transform_payload,
            "blueprint_filter": "vehicle.lincoln.mkz_2020",
            "role_name": "hero",
            "hand_brake": False,
        },
        "blocker": {
            "manifest_actor_id": _safe_int(manifest_payload.get("blocker_actor_id")),
            "base_role": "ego",
            "distance_m": blocker_distance_m,
            "lane_side": "current",
            "blueprint_filter": "vehicle.tesla.model3",
            "role_name": "stage4_blocker",
            "hand_brake": True,
        },
        "adjacent": {
            "manifest_actor_id": _safe_int(manifest_payload.get("adjacent_actor_id")),
            "base_role": "ego",
            "distance_m": adjacent_distance_m,
            "lane_side": adjacent_side,
            "blueprint_filter": "vehicle.audi.tt",
            "role_name": "stage4_adjacent",
            "hand_brake": True,
        },
    }


def _build_expected_actors(world: Any, manifest_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    specs = _expected_multilane_specs(manifest_payload)
    ego_spec = specs.get("ego")
    if not ego_spec:
        return {}
    ego_waypoint = _find_matching_waypoint(
        world=world,
        road_id=int(ego_spec["road_id"]),
        section_id=ego_spec["section_id"],
        lane_id=int(ego_spec["lane_id"]),
        s=ego_spec["s"],
    )
    ego_transform = _transform_from_payload(dict(ego_spec.get("transform_payload") or {}))
    if ego_transform is None:
        return {}
    expected: Dict[str, Dict[str, Any]] = {
        "ego": {
            **ego_spec,
            "expected_waypoint": ego_waypoint,
            "expected_lane_id": int(ego_waypoint.lane_id),
            "expected_transform": ego_transform,
        }
    }
    blocker_spec = specs.get("blocker")
    blocker_distance = _safe_float((blocker_spec or {}).get("distance_m"))
    if blocker_spec and blocker_distance is not None:
        blocker_wp = _advance_along_lane(ego_waypoint, blocker_distance)
        z_offset = float(ego_transform.location.z) - float(ego_waypoint.transform.location.z)
        blocker_transform = _transform_from_payload(
            {
                "location": {
                    "x": float(blocker_wp.transform.location.x),
                    "y": float(blocker_wp.transform.location.y),
                    "z": float(blocker_wp.transform.location.z),
                },
                "rotation": {
                    "roll": float(blocker_wp.transform.rotation.roll),
                    "pitch": float(blocker_wp.transform.rotation.pitch),
                    "yaw": float(blocker_wp.transform.rotation.yaw),
                },
            },
            z_offset_m=z_offset,
        )
        expected["blocker"] = {
            **blocker_spec,
            "expected_waypoint": blocker_wp,
            "expected_lane_id": int(blocker_wp.lane_id),
            "expected_transform": blocker_transform,
        }
    adjacent_spec = specs.get("adjacent")
    adjacent_distance = _safe_float((adjacent_spec or {}).get("distance_m"))
    adjacent_side = str((adjacent_spec or {}).get("lane_side") or "right")
    if adjacent_spec and adjacent_distance is not None:
        adjacent_base_wp = _adjacent_lane(ego_waypoint, adjacent_side)
        if adjacent_base_wp is not None:
            adjacent_target_wp = _advance_along_lane(adjacent_base_wp, adjacent_distance)
            z_offset = float(ego_transform.location.z) - float(ego_waypoint.transform.location.z)
            adjacent_transform = _transform_from_payload(
                {
                    "location": {
                        "x": float(adjacent_target_wp.transform.location.x),
                        "y": float(adjacent_target_wp.transform.location.y),
                        "z": float(adjacent_target_wp.transform.location.z),
                    },
                    "rotation": {
                        "roll": float(adjacent_target_wp.transform.rotation.roll),
                        "pitch": float(adjacent_target_wp.transform.rotation.pitch),
                        "yaw": float(adjacent_target_wp.transform.rotation.yaw),
                    },
                },
                z_offset_m=z_offset,
            )
            expected["adjacent"] = {
                **adjacent_spec,
                "expected_waypoint": adjacent_target_wp,
                "expected_lane_id": int(adjacent_target_wp.lane_id),
                "expected_transform": adjacent_transform,
            }
    return expected


def materialize_scenario_actors(*, world: Any, args: Any, scenario_manifest_payload: Dict[str, Any]) -> MaterializationResult:
    result = MaterializationResult(
        ego_vehicle=None,
        scenario_manifest_payload=dict(scenario_manifest_payload or {}),
        spawned_ego=False,
    )
    expected_actors = _build_expected_actors(world, scenario_manifest_payload)
    if not expected_actors:
        result.notes.append("scenario_manifest_does_not_expose_spawn_recipe_for_critical_actor_materialization")
        return result

    map_inst = world.get_map()
    consumed_actor_ids: set[int] = set()

    def bind_role(role: str) -> Any | None:
        spec = dict(expected_actors.get(role) or {})
        if not spec:
            return None
        manifest_actor_id = _safe_int(spec.get("manifest_actor_id"))
        expected_transform = spec.get("expected_transform")
        expected_lane_id = _safe_int(spec.get("expected_lane_id"))
        role_name = str(spec.get("role_name") or role)
        hand_brake = bool(spec.get("hand_brake", role != "ego"))
        notes: List[str] = []
        actor = None
        binding_source = "missing"
        materialization_status = "missing"
        binding_status = "missing"

        if manifest_actor_id is not None:
            candidate = world.get_actor(manifest_actor_id)
            if candidate is not None:
                actor = candidate
                if expected_transform is not None and role == "ego":
                    _realign_vehicle(actor, expected_transform, hand_brake=False)
                    binding_source = "manifest_actor_realigned"
                    materialization_status = "existing_manifest_actor_realigned"
                    binding_status = "exact"
                elif expected_transform is not None and role in {"blocker", "adjacent"}:
                    _realign_vehicle(actor, expected_transform, hand_brake=hand_brake)
                    binding_source = "manifest_actor_realigned"
                    materialization_status = "existing_manifest_actor_realigned"
                    binding_status = "exact"
                else:
                    binding_source = "manifest_actor_attached"
                    materialization_status = "existing_manifest_actor"
                    binding_status = "heuristic"
            else:
                notes.append("manifest_actor_missing_in_live_world")

        if actor is None and expected_transform is not None and role != "ego":
            candidate = _find_live_scene_candidate(
                world,
                expected_transform=expected_transform,
                expected_lane_id=expected_lane_id,
                map_inst=map_inst,
                exclude_actor_ids=consumed_actor_ids,
            )
            if candidate is not None:
                actor = candidate
                _realign_vehicle(actor, expected_transform, hand_brake=hand_brake)
                binding_source = "live_scene_actor_near_manifest_transform"
                materialization_status = "existing_live_scene_actor_reused"
                binding_status = "exact"

        if actor is None and expected_transform is not None:
            if role == "ego" and not bool(getattr(args, "spawn_ego_if_missing", False)):
                notes.append("spawn_ego_if_missing_disabled")
            else:
                try:
                    actor = _spawn_vehicle(
                        world,
                        str(spec.get("blueprint_filter") or "vehicle.*"),
                        role_name,
                        expected_transform,
                        hand_brake=hand_brake,
                    )
                    result.spawned_actors.append(actor)
                    binding_source = "spawned_from_manifest"
                    materialization_status = "spawned_from_manifest"
                    binding_status = "exact"
                    if role == "ego":
                        result.spawned_ego = True
                except Exception as exc:
                    notes.append(f"spawn_failed:{exc}")

        if actor is not None:
            consumed_actor_ids.add(int(actor.id))
        observed_lane_id = _lane_id_for_actor(map_inst, actor)
        distance_to_expected_m = _actor_distance_to_expected(actor, expected_transform)
        result.actor_bindings[role] = ActorBinding(
            role=role,
            manifest_actor_id=manifest_actor_id,
            runtime_actor_id=None if actor is None else int(actor.id),
            binding_status=binding_status,
            binding_source=binding_source,
            materialization_status=materialization_status,
            target_source_type="obstacle_actor" if role == "blocker" else None,
            expected_lane_id=expected_lane_id,
            observed_lane_id=observed_lane_id,
            distance_to_expected_m=distance_to_expected_m,
            blueprint_id=None if actor is None else str(getattr(actor, "type_id", "")),
            notes=notes,
        )
        return actor

    ego_vehicle = bind_role("ego")
    bind_role("blocker")
    bind_role("adjacent")
    result.ego_vehicle = ego_vehicle
    if ego_vehicle is None:
        result.notes.append("ego_vehicle_not_materialized_from_manifest")
    if "blocker" not in result.actor_bindings:
        result.notes.append("blocker_actor_spec_not_available")
    elif result.actor_bindings["blocker"].binding_status != "exact":
        result.notes.append("blocker_actor_not_exactly_bound")
    return result
