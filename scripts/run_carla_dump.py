from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import carla

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_bevfusion_stage1.collector import CarlaSynchronousMode, FrameCollector
from carla_bevfusion_stage1.dumper import SampleDumper
from carla_bevfusion_stage1.rig import default_nuscenes_like_rig, destroy_actors, spawn_rig

LOGGER = logging.getLogger("run_carla_dump")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn CARLA nuScenes-like rig and dump synchronized samples.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default=None, help="Optional town to load before spawning.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--warmup-ticks", type=int, default=5)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--camera-fov", type=float, default=70.0)
    parser.add_argument("--vehicle-filter", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--spawn-point-index", type=int, default=0)
    parser.add_argument("--attach-to-actor-id", type=int, default=None)
    parser.add_argument(
        "--scenario-manifest",
        default=None,
        help="Optional scenario manifest path. If --attach-to-actor-id is omitted, use ego_actor_id from this file.",
    )
    parser.add_argument("--autopilot", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument(
        "--max-frame-retries",
        type=int,
        default=8,
        help="Maximum resync retries when a synchronized frame is missed during warmup or capture.",
    )
    parser.add_argument(
        "--shutdown-settle-ticks",
        type=int,
        default=2,
        help="Extra synchronous ticks after stopping sensors, before destroying them. Helps avoid CARLA camera teardown crashes.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def resolve_next_index(output_root: Path) -> int:
    existing = sorted(output_root.glob("sample_*"))
    if not existing:
        return 0
    return max(int(path.name.split("_")[-1]) for path in existing) + 1


def resolve_attach_actor_id(args: argparse.Namespace) -> tuple[int | None, Path | None]:
    if args.attach_to_actor_id is not None:
        return int(args.attach_to_actor_id), None
    if not args.scenario_manifest:
        return None, None

    manifest_path = Path(args.scenario_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actor_id = manifest.get("ego_actor_id")
    if not isinstance(actor_id, int) or actor_id <= 0:
        raise ValueError(f"Scenario manifest {manifest_path} does not contain a valid ego_actor_id.")
    LOGGER.info("Resolved ego actor id=%s from scenario manifest %s", actor_id, manifest_path)
    return int(actor_id), manifest_path


def spawn_or_get_ego(world: carla.World, args: argparse.Namespace) -> tuple[carla.Actor, bool]:
    attach_actor_id, manifest_path = resolve_attach_actor_id(args)
    if attach_actor_id is not None:
        actor = world.get_actor(attach_actor_id)
        if actor is None:
            if manifest_path is not None:
                raise ValueError(
                    f"Actor id {attach_actor_id} from scenario manifest {manifest_path} does not exist in the current world."
                )
            raise ValueError(f"Actor id {attach_actor_id} does not exist in the current world.")
        return actor, False

    blueprints = world.get_blueprint_library().filter(args.vehicle_filter)
    if not blueprints:
        raise ValueError(f"No vehicle blueprint matches filter '{args.vehicle_filter}'")
    blueprint = blueprints[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("World does not expose spawn points.")
    spawn_index = max(0, min(args.spawn_point_index, len(spawn_points) - 1))
    actor = world.try_spawn_actor(blueprint, spawn_points[spawn_index])
    if actor is None:
        raise RuntimeError(f"Failed to spawn ego vehicle at spawn point index {spawn_index}")
    LOGGER.info("Spawned ego vehicle id=%s blueprint=%s", actor.id, actor.type_id)
    return actor, True


def log_capture_summary(sample_name: str, capture, meta_path: Path) -> None:
    frame_parts = []
    for sensor_name, packet in capture.packets.items():
        point_count = len(packet.measurement) if hasattr(packet.measurement, "__len__") else "-"
        frame_parts.append(f"{sensor_name}:{packet.frame}@{packet.timestamp:.3f}/n={point_count}")
    LOGGER.info("%s -> %s", sample_name, " | ".join(frame_parts))
    LOGGER.info("meta written to %s", meta_path)


def tick_until_capture(
    *,
    sync_mode: CarlaSynchronousMode,
    world: carla.World,
    collector: FrameCollector,
    max_frame_retries: int,
    phase_label: str,
):
    last_error = None
    for attempt in range(max(1, int(max_frame_retries))):
        frame = sync_mode.tick()
        snapshot = world.get_snapshot()
        try:
            capture = collector.wait_for_frame(frame, snapshot)
            if attempt > 0:
                LOGGER.info(
                    "%s resynchronized on frame=%d after %d retry(s)",
                    phase_label,
                    frame,
                    attempt,
                )
            return capture
        except TimeoutError as exc:
            last_error = exc
            LOGGER.warning(
                "%s missed synchronized frame=%d (attempt %d/%d): %s",
                phase_label,
                frame,
                attempt + 1,
                max(1, int(max_frame_retries)),
                exc,
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{phase_label} failed without a captured frame or timeout.")


def main() -> None:
    args = parse_args()
    configure_logging()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    if args.town:
        LOGGER.info("Loading town %s", args.town)
        world = client.load_world(args.town)
    else:
        world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)

    ego_vehicle = None
    spawned_ego = False
    sensor_actors = {}
    collector = None

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    next_index = resolve_next_index(output_root)
    previous_sample_name = None

    try:
        ego_vehicle, spawned_ego = spawn_or_get_ego(world, args)
        rig = default_nuscenes_like_rig(
            image_width=args.image_width,
            image_height=args.image_height,
            camera_fov=args.camera_fov,
            fixed_delta_seconds=args.fixed_delta_seconds,
        )

        with CarlaSynchronousMode(
            world,
            fixed_delta_seconds=args.fixed_delta_seconds,
            traffic_manager=traffic_manager,
        ) as sync_mode:
            sensor_actors = spawn_rig(world, ego_vehicle, rig)
            collector = FrameCollector(sensor_actors, timeout_seconds=args.timeout_seconds)
            collector.start()
            dumper = SampleDumper(output_root=output_root, rig_preset=rig, sensor_actors=sensor_actors)

            if args.autopilot:
                ego_vehicle.set_autopilot(True, args.tm_port)

            try:
                for _ in range(args.warmup_ticks):
                    tick_until_capture(
                        sync_mode=sync_mode,
                        world=world,
                        collector=collector,
                        max_frame_retries=args.max_frame_retries,
                        phase_label="warmup",
                    )

                for sample_offset in range(args.num_samples):
                    capture = tick_until_capture(
                        sync_mode=sync_mode,
                        world=world,
                        collector=collector,
                        max_frame_retries=args.max_frame_retries,
                        phase_label=f"sample[{sample_offset}]",
                    )
                    dump_result = dumper.dump_capture(
                        capture,
                        ego_vehicle=ego_vehicle,
                        world=world,
                        sample_index=next_index + sample_offset,
                        previous_sample_name=previous_sample_name,
                    )
                    previous_sample_name = dump_result.sample_dir.name
                    log_capture_summary(dump_result.sample_dir.name, capture, dump_result.meta_path)
            finally:
                if collector is not None:
                    collector.stop()
                for _ in range(max(0, int(args.shutdown_settle_ticks))):
                    try:
                        sync_mode.tick()
                    except RuntimeError:
                        break

    finally:
        if sensor_actors:
            destroy_actors(sensor_actors.values())
        if spawned_ego and ego_vehicle is not None:
            ego_vehicle.destroy()


if __name__ == "__main__":
    main()
