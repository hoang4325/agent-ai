from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .evaluation_stage3c import summarize_stage3c_session
from .execution_contract import ExecutionRequest, build_execution_request
from .execution_monitor import ExecutionMonitor
from .local_planner_bridge import LocalPlannerBridge, ensure_carla_pythonapi
from .stop_target_runtime import enrich_runtime_stop_target
from .video_recorder import Stage3CVideoRecorder
from .visualization_stage3c import save_stage3c_visualization_bundle
from stage3c_coverage.planner_quality import write_planner_quality_artifacts

LOGGER = logging.getLogger("stage3c.replay_runner")


def _mp4_recording_disabled() -> bool:
    """Temporary safety switch to disable MP4 recording in test runs.

    Set AGENTAI_DISABLE_MP4=0 to re-enable recording.
    """
    return str(os.getenv("AGENTAI_DISABLE_MP4", "1")).strip().lower() not in {"0", "false", "no", "off"}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 3B timeline: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _dedupe_stage3b_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for record in records:
        key = (int(record["frame_id"]), str(record["sample_name"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3C closed-loop execution bridge in CARLA.")
    parser.add_argument("--stage3b-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario-manifest", default=None)
    parser.add_argument("--attach-to-actor-id", type=int, default=0)
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-timeout-s", type=float, default=10.0)
    parser.add_argument("--carla-pythonapi-root", default=r"D:\carla\PythonAPI")
    parser.add_argument("--load-town-if-needed", action="store_true")
    parser.add_argument("--spawn-ego-if-missing", action="store_true")
    parser.add_argument("--ego-vehicle-filter", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--ego-spawn-point-index", type=int, default=0)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.10)
    parser.add_argument("--warmup-ticks", type=int, default=3)
    parser.add_argument("--post-settle-ticks", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--prepare-offset-m", type=float, default=0.75)
    parser.add_argument("--sampling-radius-m", type=float, default=2.0)
    parser.add_argument("--lane-change-success-hold-ticks", type=int, default=4)
    parser.add_argument("--max-lane-change-ticks", type=int, default=45)
    parser.add_argument("--lane-change-min-lateral-shift-m", type=float, default=0.50)
    parser.add_argument("--lane-change-min-lateral-shift-fraction", type=float, default=0.15)
    parser.add_argument("--stop-hold-ticks", type=int, default=10)
    parser.add_argument("--stop-full-stop-speed-threshold-mps", type=float, default=0.35)
    parser.add_argument("--stop-near-distance-m", type=float, default=8.0)
    parser.add_argument("--stop-hard-brake-distance-m", type=float, default=3.0)
    parser.add_argument("--record-mp4", action="store_true")
    parser.add_argument("--recording-path", default=None)
    parser.add_argument("--recording-camera-mode", default="chase", choices=["chase", "hood", "topdown"])
    parser.add_argument("--recording-width", type=int, default=1280)
    parser.add_argument("--recording-height", type=int, default=720)
    parser.add_argument("--recording-fov", type=float, default=90.0)
    parser.add_argument("--recording-fps", type=float, default=0.0)
    parser.add_argument("--recording-post-roll-ticks", type=int, default=0)
    parser.add_argument("--recording-overlay", action="store_true")
    parser.add_argument("--recording-no-overlay", dest="recording_overlay", action="store_false")
    parser.set_defaults(recording_overlay=True)
    args = parser.parse_args()
    if _mp4_recording_disabled() and bool(getattr(args, "record_mp4", False)):
        LOGGER.warning("MP4 recording disabled by AGENTAI_DISABLE_MP4; ignoring --record-mp4")
        args.record_mp4 = False
        args.recording_post_roll_ticks = 0
    return args


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_stage3b_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    timeline_path = Path(args.stage3b_output_dir) / "behavior_timeline_v2.jsonl"
    records = _dedupe_stage3b_records(_load_jsonl(timeline_path))
    if args.max_frames and args.max_frames > 0:
        records = records[: args.max_frames]
    if not records:
        raise RuntimeError(f"No Stage 3B frames found in {timeline_path}")
    return records


def _spawn_fallback_ego_vehicle(world: Any, args: argparse.Namespace) -> Any:
    return _spawn_fallback_ego_vehicle_at_transform(world=world, args=args, transform=None)


def _spawn_fallback_ego_vehicle_at_transform(
    *,
    world: Any,
    args: argparse.Namespace,
    transform: Any | None,
) -> Any:
    blueprint_filter = str(getattr(args, "ego_vehicle_filter", "vehicle.lincoln.mkz_2020"))
    spawn_point_index = max(0, int(getattr(args, "ego_spawn_point_index", 0)))
    blueprint_library = world.get_blueprint_library()
    candidates = blueprint_library.filter(blueprint_filter)
    if not candidates:
        candidates = blueprint_library.filter("vehicle.*")
    if not candidates:
        raise RuntimeError("No vehicle blueprint available for fallback ego spawn.")
    ego_vehicle = None
    resolved_spawn_index = None
    if transform is not None:
        ego_vehicle = world.try_spawn_actor(candidates[0], transform)
        if ego_vehicle is not None:
            LOGGER.info(
                "Spawned fallback ego actor id=%s blueprint=%s from scenario manifest transform",
                int(ego_vehicle.id),
                str(candidates[0].id),
            )
            return ego_vehicle

    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("CARLA map has no spawn points for fallback ego spawn.")
    resolved_spawn_index = spawn_point_index % len(spawn_points)
    for offset in range(len(spawn_points)):
        candidate_index = (spawn_point_index + offset) % len(spawn_points)
        spawn_point = spawn_points[candidate_index]
        ego_vehicle = world.try_spawn_actor(candidates[0], spawn_point)
        if ego_vehicle is not None:
            resolved_spawn_index = candidate_index
            break
    if ego_vehicle is None:
        raise RuntimeError(
            "Failed to spawn fallback ego vehicle; all spawn points appear blocked. "
            f"filter={blueprint_filter} start_spawn_point_index={spawn_point_index}"
        )
    LOGGER.info(
        "Spawned fallback ego actor id=%s blueprint=%s spawn_point_index=%d",
        int(ego_vehicle.id),
        str(candidates[0].id),
        resolved_spawn_index,
    )
    return ego_vehicle


def _manifest_ego_transform_payload(manifest_payload: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(manifest_payload, dict):
        return None
    corridor = manifest_payload.get("corridor")
    if not isinstance(corridor, dict):
        return None
    transform = corridor.get("ego_transform")
    if not isinstance(transform, dict):
        return None
    location = transform.get("location")
    rotation = transform.get("rotation")
    if not isinstance(location, dict) or not isinstance(rotation, dict):
        return None
    return transform


def _build_carla_transform(transform_payload: Dict[str, Any]) -> Any:
    import carla  # type: ignore

    location_payload = transform_payload.get("location", {})
    rotation_payload = transform_payload.get("rotation", {})
    return carla.Transform(
        carla.Location(
            x=float(location_payload.get("x", 0.0)),
            y=float(location_payload.get("y", 0.0)),
            z=float(location_payload.get("z", 0.0)),
        ),
        carla.Rotation(
            roll=float(rotation_payload.get("roll", 0.0)),
            pitch=float(rotation_payload.get("pitch", 0.0)),
            yaw=float(rotation_payload.get("yaw", 0.0)),
        ),
    )


def _set_vehicle_neutral_state(vehicle: Any) -> None:
    import carla  # type: ignore

    vehicle.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
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


def _align_vehicle_to_manifest(vehicle: Any, manifest_payload: Dict[str, Any] | None) -> bool:
    transform_payload = _manifest_ego_transform_payload(manifest_payload)
    if transform_payload is None:
        return False
    transform = _build_carla_transform(transform_payload)
    vehicle.set_autopilot(False)
    if hasattr(vehicle, "set_simulate_physics"):
        vehicle.set_simulate_physics(True)
    vehicle.set_transform(transform)
    _set_vehicle_neutral_state(vehicle)
    LOGGER.info(
        "Aligned ego actor id=%s to scenario manifest transform x=%.2f y=%.2f z=%.2f yaw=%.2f",
        int(vehicle.id),
        float(transform.location.x),
        float(transform.location.y),
        float(transform.location.z),
        float(transform.rotation.yaw),
    )
    return True


def _resolve_ego_vehicle(client: Any, world: Any, args: argparse.Namespace) -> tuple[Any, Any, Dict[str, Any] | None, bool]:
    manifest_payload: Dict[str, Any] | None = None
    actor_id = 0
    if args.scenario_manifest:
        manifest_path = Path(args.scenario_manifest)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest_payload = json.load(handle)
        actor_id = int(manifest_payload["ego_actor_id"])
        current_map = world.get_map().name
        manifest_town = str(manifest_payload.get("town", ""))
        if manifest_town and current_map != manifest_town:
            if not bool(getattr(args, "load_town_if_needed", False)):
                raise RuntimeError(f"Current CARLA map {current_map} does not match manifest town {manifest_town}")
            world = client.load_world(manifest_town)
    elif args.attach_to_actor_id:
        actor_id = int(args.attach_to_actor_id)
    else:
        if bool(getattr(args, "spawn_ego_if_missing", False)):
            ego_vehicle = _spawn_fallback_ego_vehicle_at_transform(
                world=world,
                args=args,
                transform=None if manifest_payload is None else _build_carla_transform(_manifest_ego_transform_payload(manifest_payload) or {}),
            )
            return world, ego_vehicle, manifest_payload, True
        raise ValueError("Provide either --scenario-manifest or --attach-to-actor-id for Stage 3C closed-loop execution.")

    ego_vehicle = world.get_actor(actor_id)
    if ego_vehicle is None:
        if not bool(getattr(args, "spawn_ego_if_missing", False)):
            raise RuntimeError(f"Ego actor id {actor_id} does not exist in the current CARLA world.")
        transform_payload = _manifest_ego_transform_payload(manifest_payload)
        ego_vehicle = _spawn_fallback_ego_vehicle_at_transform(
            world=world,
            args=args,
            transform=None if transform_payload is None else _build_carla_transform(transform_payload),
        )
        _align_vehicle_to_manifest(ego_vehicle, manifest_payload)
        return world, ego_vehicle, manifest_payload, True
    _align_vehicle_to_manifest(ego_vehicle, manifest_payload)
    return world, ego_vehicle, manifest_payload, False


def _enable_sync_mode(world: Any, fixed_delta_seconds: float) -> Any:
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(fixed_delta_seconds)
    world.apply_settings(settings)
    return original_settings


def _apply_neutral_brake(vehicle: Any) -> None:
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


def _current_lane_id(vehicle: Any, map_inst: Any) -> int | None:
    waypoint = map_inst.get_waypoint(vehicle.get_location())
    return None if waypoint is None else int(waypoint.lane_id)


def _persist_frame_artifacts(
    *,
    output_dir: Path,
    stage3b_record: Dict[str, Any],
    timeline_record: Dict[str, Any],
) -> None:
    frame_output_dir = output_dir / "frames" / str(stage3b_record["sample_name"])
    _write_json(frame_output_dir / "behavior_request_v2.json", dict(stage3b_record["behavior_request_v2"]))
    _write_json(frame_output_dir / "execution_request.json", dict(timeline_record["execution_request"]))
    _write_json(frame_output_dir / "execution_state.json", dict(timeline_record["execution_state"]))


def _build_overlay_lines(timeline_record: Dict[str, Any]) -> List[str]:
    execution_request = timeline_record["execution_request"]
    execution_state = timeline_record["execution_state"]
    junction_tag = "junction" if execution_state.get("is_in_junction") else "road"
    return [
        (
            f"sample={timeline_record['sample_name']} req_frame={timeline_record['request_frame']} "
            f"carla_frame={timeline_record['carla_frame']}"
        ),
        (
            f"requested={execution_request['requested_behavior']} "
            f"executed={execution_state['current_behavior']} status={execution_state['execution_status']}"
        ),
        (
            f"lane={execution_state['current_lane_id']} -> {execution_state['target_lane_id']} "
            f"progress={execution_state['lane_change_progress']:.2f} mode={junction_tag}"
        ),
        (
            f"speed={execution_state['speed_mps']:.2f} mps "
            f"target={execution_request['target_speed_mps']:.2f} mps "
            f"route={execution_request.get('route_option', 'unknown')}"
        ),
    ]


def _run_request_tick(
    *,
    world: Any,
    vehicle: Any,
    map_inst: Any,
    bridge: LocalPlannerBridge,
    monitor: ExecutionMonitor,
    execution_request: ExecutionRequest,
    stage3b_record: Dict[str, Any] | None,
    timeline_path: Path,
    lane_change_events_path: Path,
    output_dir: Path,
    persist_frame_artifacts: bool,
    video_recorder: Stage3CVideoRecorder | None,
) -> Dict[str, Any]:
    snapshot_before = world.get_snapshot()
    had_active_lane_change = bridge.active_lane_change is not None
    prepare_start = time.perf_counter()
    planner_step = bridge.step(execution_request, snapshot_before)
    started_lane_change = (not had_active_lane_change) and (bridge.active_lane_change is not None)
    started_tracker = bridge.active_lane_change

    vehicle.apply_control(planner_step.control)
    planner_prepare_latency_ms = (time.perf_counter() - prepare_start) * 1000.0
    world.tick()
    snapshot_after = world.get_snapshot()
    finalize_start = time.perf_counter()

    if started_lane_change and started_tracker is not None:
        event = monitor.record_lane_change_start(snapshot=snapshot_after, tracker=started_tracker)
        _append_jsonl(lane_change_events_path, event)

    completed_tracker = bridge.post_tick_update(snapshot_after)
    tracker_for_state = completed_tracker or bridge.active_lane_change
    timeline_record = monitor.record(
        execution_request=execution_request,
        planner_step=planner_step,
        snapshot=snapshot_after,
        vehicle=vehicle,
        map_inst=map_inst,
        lane_change_tracker=tracker_for_state,
    )
    timeline_record["stage3b_behavior_request"] = (
        None if stage3b_record is None else dict(stage3b_record["behavior_request_v2"])
    )
    planner_diagnostics = dict(timeline_record.get("planner_diagnostics") or {})
    execution_finalize_latency_ms = (time.perf_counter() - finalize_start) * 1000.0
    planner_diagnostics.update(
        {
            "planner_prepare_latency_ms": float(planner_prepare_latency_ms),
            "execution_finalize_latency_ms": float(execution_finalize_latency_ms),
            "planner_execution_latency_ms": float(planner_prepare_latency_ms + execution_finalize_latency_ms),
        }
    )
    timeline_record["planner_diagnostics"] = planner_diagnostics
    _append_jsonl(timeline_path, timeline_record)

    if completed_tracker is not None:
        event = monitor.record_lane_change_terminal(
            snapshot=snapshot_after,
            tracker=completed_tracker,
            current_lane_id=timeline_record["execution_state"]["current_lane_id"],
        )
        _append_jsonl(lane_change_events_path, event)

    if persist_frame_artifacts and stage3b_record is not None:
        _persist_frame_artifacts(
            output_dir=output_dir,
            stage3b_record=stage3b_record,
            timeline_record=timeline_record,
        )

    if video_recorder is not None:
        wrote_video = video_recorder.write_frame(
            carla_frame=int(timeline_record["carla_frame"]),
            overlay_lines=_build_overlay_lines(timeline_record),
        )
        if not wrote_video:
            LOGGER.warning(
                "Stage3C video recorder missed frame carla_frame=%d sample=%s",
                int(timeline_record["carla_frame"]),
                execution_request.sample_name,
            )

    LOGGER.info(
        "Stage3C carla_frame=%d request_frame=%d sample=%s requested=%s executed=%s status=%s lane=%s->%s",
        int(snapshot_after.frame),
        int(execution_request.frame),
        execution_request.sample_name,
        execution_request.requested_behavior,
        timeline_record["execution_state"]["current_behavior"],
        timeline_record["execution_state"]["execution_status"],
        timeline_record["execution_state"]["current_lane_id"],
        timeline_record["execution_state"]["target_lane_id"],
    )
    return timeline_record


def run_closed_loop(args: argparse.Namespace) -> Dict[str, Any]:
    if _mp4_recording_disabled() and bool(getattr(args, "record_mp4", False)):
        LOGGER.warning("MP4 recording disabled by AGENTAI_DISABLE_MP4; forcing record_mp4=False")
        args.record_mp4 = False
        if hasattr(args, "recording_post_roll_ticks"):
            args.recording_post_roll_ticks = 0

    ensure_carla_pythonapi(args.carla_pythonapi_root)
    import carla  # type: ignore

    output_dir = Path(args.output_dir)
    stage3b_records = _load_stage3b_records(args)

    client = carla.Client(args.carla_host, int(args.carla_port))
    client.set_timeout(float(args.carla_timeout_s))
    world = client.get_world()
    world, ego_vehicle, scenario_manifest, spawned_fallback_ego = _resolve_ego_vehicle(client, world, args)
    map_inst = world.get_map()
    ego_vehicle.set_autopilot(False)

    timeline_path = output_dir / "execution_timeline.jsonl"
    lane_change_events_path = output_dir / "lane_change_events.jsonl"
    if timeline_path.exists():
        timeline_path.unlink()
    if lane_change_events_path.exists():
        lane_change_events_path.unlink()
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    lane_change_events_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text("", encoding="utf-8")
    lane_change_events_path.write_text("", encoding="utf-8")

    manifest = {
        "stage3b_output_dir": str(args.stage3b_output_dir),
        "output_dir": str(output_dir),
        "scenario_manifest": args.scenario_manifest,
        "attach_to_actor_id": int(ego_vehicle.id),
        "spawned_fallback_ego": bool(spawned_fallback_ego),
        "carla_host": args.carla_host,
        "carla_port": int(args.carla_port),
        "carla_pythonapi_root": str(args.carla_pythonapi_root),
        "fixed_delta_seconds": float(args.fixed_delta_seconds),
        "prepare_offset_m": float(args.prepare_offset_m),
        "lane_change_success_hold_ticks": int(args.lane_change_success_hold_ticks),
        "max_lane_change_ticks": int(args.max_lane_change_ticks),
        "lane_change_min_lateral_shift_m": float(args.lane_change_min_lateral_shift_m),
        "lane_change_min_lateral_shift_fraction": float(args.lane_change_min_lateral_shift_fraction),
        "stop_hold_ticks": int(args.stop_hold_ticks),
        "stop_full_stop_speed_threshold_mps": float(args.stop_full_stop_speed_threshold_mps),
        "stop_near_distance_m": float(args.stop_near_distance_m),
        "stop_hard_brake_distance_m": float(args.stop_hard_brake_distance_m),
        "num_requests": int(len(stage3b_records)),
        "execution_mode": "local_planner_bridge",
        "backend_note": "CARLA LocalPlanner/PID used as temporary execution backend, not a production local planner.",
    }
    if scenario_manifest is not None:
        manifest["scenario_manifest_payload"] = scenario_manifest
    _write_json(output_dir / "stage3c_manifest.json", manifest)

    bridge = LocalPlannerBridge(
        vehicle=ego_vehicle,
        map_inst=map_inst,
        pythonapi_root=args.carla_pythonapi_root,
        fixed_delta_seconds=float(args.fixed_delta_seconds),
        sampling_radius_m=float(args.sampling_radius_m),
        prepare_offset_m=float(args.prepare_offset_m),
        lane_change_success_hold_ticks=int(args.lane_change_success_hold_ticks),
        max_lane_change_ticks=int(args.max_lane_change_ticks),
        lane_change_min_lateral_shift_m=float(args.lane_change_min_lateral_shift_m),
        lane_change_min_lateral_shift_fraction=float(args.lane_change_min_lateral_shift_fraction),
        stop_hold_ticks=int(args.stop_hold_ticks),
        stop_full_stop_speed_threshold_mps=float(args.stop_full_stop_speed_threshold_mps),
        stop_near_distance_m=float(args.stop_near_distance_m),
        stop_hard_brake_distance_m=float(args.stop_hard_brake_distance_m),
    )
    monitor = ExecutionMonitor()
    recording_path = (
        Path(args.recording_path)
        if args.recording_path
        else output_dir / "video" / "closed_loop.mp4"
    )
    recording_fps = float(args.recording_fps) if float(args.recording_fps) > 0.0 else (1.0 / float(args.fixed_delta_seconds))
    video_recorder: Stage3CVideoRecorder | None = None

    original_settings = _enable_sync_mode(world, float(args.fixed_delta_seconds))
    last_request: ExecutionRequest | None = None
    try:
        if bool(args.record_mp4):
            video_recorder = Stage3CVideoRecorder(
                world=world,
                vehicle=ego_vehicle,
                output_path=recording_path,
                fps=recording_fps,
                width=int(args.recording_width),
                height=int(args.recording_height),
                fov=float(args.recording_fov),
                mode=str(args.recording_camera_mode),
                overlay=bool(args.recording_overlay),
            )
            manifest["recording"] = {
                "enabled": True,
                "path": str(recording_path),
                "camera_mode": str(args.recording_camera_mode),
                "width": int(args.recording_width),
                "height": int(args.recording_height),
                "fov": float(args.recording_fov),
                "fps": float(recording_fps),
                "overlay": bool(args.recording_overlay),
                "post_roll_ticks": int(args.recording_post_roll_ticks),
            }
            _write_json(output_dir / "stage3c_manifest.json", manifest)

        _apply_neutral_brake(ego_vehicle)
        for _ in range(max(0, int(args.warmup_ticks))):
            world.tick()

        for stage3b_record in stage3b_records:
            execution_request = build_execution_request(stage3b_record)
            execution_request.stop_target = enrich_runtime_stop_target(
                execution_request.stop_target,
                scenario_manifest,
            )
            last_request = execution_request
            _run_request_tick(
                world=world,
                vehicle=ego_vehicle,
                map_inst=map_inst,
                bridge=bridge,
                monitor=monitor,
                execution_request=execution_request,
                stage3b_record=stage3b_record,
                timeline_path=timeline_path,
                lane_change_events_path=lane_change_events_path,
                output_dir=output_dir,
                persist_frame_artifacts=True,
                video_recorder=video_recorder,
            )

        for _ in range(max(0, int(args.post_settle_ticks))):
            if bridge.active_lane_change is None or last_request is None:
                break
            last_request.stop_target = enrich_runtime_stop_target(
                last_request.stop_target,
                scenario_manifest,
            )
            _run_request_tick(
                world=world,
                vehicle=ego_vehicle,
                map_inst=map_inst,
                bridge=bridge,
                monitor=monitor,
                execution_request=last_request,
                stage3b_record=None,
                timeline_path=timeline_path,
                lane_change_events_path=lane_change_events_path,
                output_dir=output_dir,
                persist_frame_artifacts=False,
                video_recorder=video_recorder,
            )

        for _ in range(max(0, int(args.recording_post_roll_ticks))):
            if last_request is None:
                break
            last_request.stop_target = enrich_runtime_stop_target(
                last_request.stop_target,
                scenario_manifest,
            )
            _run_request_tick(
                world=world,
                vehicle=ego_vehicle,
                map_inst=map_inst,
                bridge=bridge,
                monitor=monitor,
                execution_request=last_request,
                stage3b_record=None,
                timeline_path=timeline_path,
                lane_change_events_path=lane_change_events_path,
                output_dir=output_dir,
                persist_frame_artifacts=False,
                video_recorder=video_recorder,
            )

        evaluation_summary = summarize_stage3c_session(monitor.timeline_records, monitor.lane_change_events)
        _write_json(output_dir / "execution_summary.json", evaluation_summary)
        planner_quality_bundle = write_planner_quality_artifacts(
            output_dir,
            execution_records=monitor.timeline_records,
            raw_lane_change_events=monitor.lane_change_events,
            source_stage="stage3c_replay",
        )
        manifest["planner_quality_artifacts"] = {
            "trajectory_trace": "trajectory_trace.jsonl",
            "control_trace": "control_trace.jsonl",
            "lane_change_phase_trace": "lane_change_phase_trace.jsonl",
            "lane_change_events_enriched": "lane_change_events_enriched.jsonl",
            "stop_events": "stop_events.jsonl",
            "stop_target": "stop_target.json",
            "planner_control_latency": "planner_control_latency.jsonl",
            "fallback_events": "fallback_events.jsonl",
            "planner_quality_summary": "planner_quality_summary.json",
            "measurement_support": (planner_quality_bundle.get("planner_quality_summary") or {}).get("measurement_support"),
        }
        _write_json(output_dir / "stage3c_manifest.json", manifest)
        if video_recorder is not None:
            _write_json(output_dir / "video_manifest.json", video_recorder.manifest())
        save_stage3c_visualization_bundle(
            output_root=output_dir / "visualization",
            execution_records=monitor.timeline_records,
        )
        LOGGER.info("Stage3C closed-loop execution complete output_dir=%s", output_dir)
        return evaluation_summary
    finally:
        if video_recorder is not None:
            video_recorder.close()
        _apply_neutral_brake(ego_vehicle)
        if spawned_fallback_ego:
            ego_vehicle.destroy()
        world.apply_settings(original_settings)


def main() -> None:
    args = parse_args()
    configure_logging()
    run_closed_loop(args)


if __name__ == "__main__":
    main()
