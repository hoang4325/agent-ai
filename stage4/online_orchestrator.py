from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from stage3c.execution_contract import build_execution_request
from stage3c.local_planner_bridge import ensure_carla_pythonapi
from stage3c.stop_target_runtime import enrich_runtime_stop_target
from stage3c_coverage.planner_quality import write_planner_quality_artifacts

from .evaluation_stage4 import summarize_stage4_session
from .execution_runtime import ExecutionRuntime
from .monitoring import Stage4Monitor
from .perception_online_adapter import PerceptionOnlineAdapter
from .scenario_actor_materialization import materialize_scenario_actors
from .shadow_runtime import Stage4ShadowRuntime
from .stop_scene_alignment import apply_stop_scene_alignment
from .state_store import BehaviorFrameState, OnlineStackState, OnlineTickRecord
from .visualization_stage4 import save_stage4_visualization_bundle
from .world_behavior_runtime import WorldBehaviorRuntime

LOGGER = logging.getLogger("stage4.online_orchestrator")


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
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")


def _normalize_town_name(town: str) -> str:
    normalized = str(town).replace("\\", "/")
    return normalized.split("/")[-1]


def _call_world_api_with_retry(
    *,
    action: Any,
    description: str,
    attempts: int = 6,
    sleep_seconds: float = 2.0,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            return action()
        except RuntimeError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            LOGGER.warning(
                "Stage4 %s not ready on attempt %d/%d: %s",
                description,
                attempt,
                attempts,
                exc,
            )
            time.sleep(float(sleep_seconds))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Stage4 failed to resolve {description}.")


def _resolve_attach_actor_id(args: Any) -> tuple[int | None, Dict[str, Any] | None]:
    if getattr(args, "attach_to_actor_id", None):
        return int(args.attach_to_actor_id), None
    if not getattr(args, "scenario_manifest", None):
        return None, None
    manifest_path = Path(args.scenario_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actor_id = manifest.get("ego_actor_id")
    if not isinstance(actor_id, int) or actor_id <= 0:
        raise ValueError(f"Scenario manifest {manifest_path} does not contain a valid ego_actor_id.")
    return int(actor_id), manifest


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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _vehicle_speed_mps(vehicle: Any) -> float:
    velocity = vehicle.get_velocity()
    return float((velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5)


def _apply_bounded_motion_guard(
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


def _clone_execution_request_with_updates(request: Any, **updates: Any) -> Any:
    payload = request.to_dict()
    payload.update(updates)
    from stage3c.execution_contract import ExecutionRequest

    return ExecutionRequest(**payload)


def _build_shadow_candidate_control(
    *,
    carla_module: Any,
    proposal: Dict[str, Any],
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

    steer = _clamp(lateral_offset_m * 0.18, -0.20, 0.20)
    speed_error_mps = float(target_speed_mps - current_speed_mps)
    throttle = 0.0
    brake = 0.0
    if requested_behavior == "stop_before_obstacle" or target_speed_mps <= 0.05:
        brake = 0.35 if current_speed_mps >= 0.20 else 0.20
    elif speed_error_mps > 0.10:
        throttle = _clamp(0.05 + speed_error_mps * 0.06, 0.05, 0.14)
    elif speed_error_mps < -0.05:
        brake = _clamp(abs(speed_error_mps) * 0.12, 0.05, 0.20)
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


class Stage4OnlineOrchestrator:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.project_root = Path(__file__).resolve().parents[1]
        self.output_dir = Path(args.output_dir).resolve()
        self.samples_root = self.output_dir / "samples"
        self.stage1_output_root = self.output_dir / "stage1"
        self.sample_index = 0
        self.previous_sample_name: str | None = None
        self.state_resets = 0
        self.skipped_updates = 0
        self.retry_dropped_frames = 0
        self.perception_stale_ticks = 0
        self.last_behavior_state: BehaviorFrameState | None = None
        self.last_capture_sample_name: str | None = None
        self._spawned_ego_vehicle = False
        self._spawned_runtime_actors: List[Any] = []
        self.scenario_actor_materialization: Dict[str, Any] | None = None
        self.scenario_actor_bindings: Dict[str, Dict[str, Any]] = {}
        self.stop_scene_alignment: Dict[str, Any] | None = None
        self.shadow_runtime: Stage4ShadowRuntime | None = None
        self.control_authority_sandbox_mode = str(
            getattr(args, "control_authority_sandbox_mode", "disabled") or "disabled"
        )
        self.control_authority_sandbox_handoff_tick = int(
            getattr(args, "control_authority_sandbox_handoff_tick", -1)
        )
        self.control_authority_sandbox_handoff_duration_ticks = int(
            getattr(args, "control_authority_sandbox_handoff_duration_ticks", -1)
        )
        self.control_authority_sandbox_max_speed_mps = float(
            getattr(args, "control_authority_sandbox_max_speed_mps", 0.35)
        )
        self.control_authority_sandbox_min_speed_mps = float(
            getattr(args, "control_authority_sandbox_min_speed_mps", 0.05)
        )
        self.control_authority_sandbox_trace_path = self.output_dir / "control_authority_sandbox_trace.jsonl"
        self.control_authority_sandbox_summary_path = self.output_dir / "control_authority_sandbox_summary.json"
        self.control_authority_sandbox_rows: List[Dict[str, Any]] = []
        self.control_authority_sandbox_trace_path.unlink(missing_ok=True)

        ensure_carla_pythonapi(args.carla_pythonapi_root)
        import carla  # type: ignore
        from carla_bevfusion_stage1.collector import FrameCollector
        from carla_bevfusion_stage1.dumper import SampleDumper
        from carla_bevfusion_stage1.rig import build_rig_preset
        from stage3.map_adapter import CarlaMapAdapter

        self._FrameCollector = FrameCollector
        self._SampleDumper = SampleDumper

        self.carla = carla
        self.client = carla.Client(args.carla_host, int(args.carla_port))
        self.client.set_timeout(float(args.carla_timeout_s))
        self.world = _call_world_api_with_retry(
            action=self.client.get_world,
            description="client.get_world()",
            attempts=max(3, int(getattr(args, "carla_ready_attempts", 6))),
            sleep_seconds=float(getattr(args, "carla_ready_retry_seconds", 2.0)),
        )
        if getattr(args, "town", None):
            if self.world.get_map().name != str(args.town):
                if not bool(getattr(args, "load_town_if_needed", False)):
                    raise RuntimeError(
                        f"Current CARLA town {self.world.get_map().name} does not match requested town {args.town}. "
                        "Pass --load-town-if-needed to switch automatically."
                    )
                self.world = _call_world_api_with_retry(
                    action=lambda: self.client.load_world(_normalize_town_name(str(args.town))),
                    description=f"client.load_world({args.town})",
                    attempts=max(3, int(getattr(args, "carla_ready_attempts", 6))),
                    sleep_seconds=float(getattr(args, "carla_ready_retry_seconds", 2.0)),
                )
        self.map_inst = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager(int(args.tm_port))

        actor_id, scenario_manifest_payload = _resolve_attach_actor_id(args)
        manifest_town = None if scenario_manifest_payload is None else str(scenario_manifest_payload.get("town") or "")
        if manifest_town and self.world.get_map().name != manifest_town:
            if not bool(getattr(args, "load_town_if_needed", False)):
                raise RuntimeError(
                    f"Current CARLA town {self.world.get_map().name} does not match scenario manifest town {manifest_town}. "
                    "Pass --load-town-if-needed to switch automatically."
                )
            self.world = _call_world_api_with_retry(
                action=lambda: self.client.load_world(_normalize_town_name(manifest_town)),
                description=f"client.load_world({manifest_town})",
                attempts=max(3, int(getattr(args, "carla_ready_attempts", 6))),
                sleep_seconds=float(getattr(args, "carla_ready_retry_seconds", 2.0)),
            )
            self.map_inst = self.world.get_map()
            self.traffic_manager = self.client.get_trafficmanager(int(args.tm_port))
        ego_vehicle = None
        if scenario_manifest_payload is not None:
            materialization = materialize_scenario_actors(
                world=self.world,
                args=args,
                scenario_manifest_payload=scenario_manifest_payload,
            )
            self.scenario_actor_materialization = materialization.to_manifest()
            self.scenario_actor_bindings = {
                role: dict(binding)
                for role, binding in self.scenario_actor_materialization.get("actor_bindings", {}).items()
            }
            self._spawned_runtime_actors.extend(materialization.spawned_actors)
            self._spawned_ego_vehicle = bool(materialization.spawned_ego)
            ego_vehicle = materialization.ego_vehicle
        if ego_vehicle is None and actor_id is not None:
            ego_vehicle = self.world.get_actor(actor_id)
        if ego_vehicle is None and bool(getattr(args, "spawn_ego_if_missing", False)):
            ego_vehicle = self._spawn_fallback_ego_vehicle()
        if ego_vehicle is None and actor_id is None:
            raise ValueError(
                "Stage 4 requires --scenario-manifest/--attach-to-actor-id or --spawn-ego-if-missing."
            )
        if ego_vehicle is None:
            raise RuntimeError(f"Ego actor id {actor_id} does not exist in the current CARLA world.")
        ego_vehicle.set_autopilot(False)
        self.ego_vehicle = ego_vehicle
        self.scenario_manifest_payload = scenario_manifest_payload

        self.rig = build_rig_preset(
            str(getattr(args, "sensor_rig_profile", "default")),
            image_width=int(args.image_width),
            image_height=int(args.image_height),
            camera_fov=float(args.camera_fov),
            fixed_delta_seconds=float(args.fixed_delta_seconds),
        )
        self.collector: FrameCollector | None = None
        self.sensor_actors: Dict[str, Any] = {}
        self.dumper: SampleDumper | None = None

        map_adapter = CarlaMapAdapter(
            client=self.client,
            world=self.world,
            map_obj=self.map_inst,
            host=str(args.carla_host),
            port=int(args.carla_port),
        )
        self.perception_adapter = PerceptionOnlineAdapter(
            project_root=self.project_root,
            samples_root=self.samples_root,
            stage1_output_root=self.stage1_output_root,
            session_name=str(args.stage1_session_name),
            num_samples=int(args.stage1_num_samples),
            min_score=float(args.min_score),
            launch_mode=str(args.stage1_launch_mode),
            external_session_root=args.stage1_session_root,
            container_name=str(args.stage1_container_name),
            container_agent_root=str(args.stage1_container_agent_root),
            container_repo_root=str(args.stage1_container_repo_root),
            container_config=str(args.stage1_container_config),
            container_checkpoint=str(args.stage1_container_checkpoint),
            device=str(args.stage1_device),
            radar_bridge=str(args.radar_bridge),
            radar_ablation=str(args.radar_ablation),
            compare_zero_radar=bool(args.compare_zero_radar),
            adapter_lidar_sweeps_test=int(args.adapter_lidar_sweeps_test),
            adapter_radar_sweeps=int(args.adapter_radar_sweeps),
            adapter_lidar_max_points=int(args.adapter_lidar_max_points),
            adapter_radar_max_points=int(args.adapter_radar_max_points),
            prediction_variant=str(args.prediction_variant),
            poll_interval_seconds=float(args.stage1_poll_interval_seconds),
            idle_timeout_seconds=float(args.stage1_idle_timeout_seconds),
            external_watch_start_sequence_index=int(getattr(args, "external_watch_start_sequence_index", -1)),
            external_watch_start_sample_name=getattr(args, "external_watch_start_sample_name", None),
            external_watch_end_sequence_index=int(getattr(args, "external_watch_end_sequence_index", -1)),
            external_watch_end_sample_name=getattr(args, "external_watch_end_sample_name", None),
            external_watch_max_payloads_per_poll=int(getattr(args, "external_watch_max_payloads_per_poll", 0)),
            worker_log_path=self.output_dir / "stage1_worker.log",
            worker_audit_path=self.output_dir / "stage1_worker_audit.json",
            worker_failure_trace_path=self.output_dir / "stage1_worker_failure_trace.txt",
            worker_health_path=self.output_dir / "stage1_worker_health.jsonl",
        )
        self.world_behavior_runtime = WorldBehaviorRuntime(
            map_adapter=map_adapter,
            output_dir=self.output_dir,
            route_option=args.route_option,
            preferred_lane=args.preferred_lane,
            route_priority=str(args.route_priority),
            route_manifest_path=args.route_manifest,
            max_missed_frames=int(args.max_missed_frames),
            vehicle_match_distance_m=float(args.vehicle_match_distance_m),
            vru_match_distance_m=float(args.vru_match_distance_m),
            static_match_distance_m=float(args.static_match_distance_m),
            cruise_speed_mps=float(args.cruise_speed_mps),
            forward_horizon_m=float(args.forward_horizon_m),
            waypoint_step_m=float(args.waypoint_step_m),
        )
        effective_record_mp4 = bool(getattr(args, "record_mp4", False)) and (not _mp4_recording_disabled())
        if bool(getattr(args, "record_mp4", False)) and not effective_record_mp4:
            LOGGER.warning("MP4 recording disabled by AGENTAI_DISABLE_MP4; forcing record_mp4=False")

        self.execution_runtime = ExecutionRuntime(
            vehicle=self.ego_vehicle,
            map_inst=self.map_inst,
            pythonapi_root=args.carla_pythonapi_root,
            output_dir=self.output_dir,
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
            recording_enabled=effective_record_mp4,
            recording_path=args.recording_path,
            recording_fps=float(args.recording_fps) if float(args.recording_fps) > 0.0 else (1.0 / float(args.fixed_delta_seconds)),
            recording_width=int(args.recording_width),
            recording_height=int(args.recording_height),
            recording_fov=float(args.recording_fov),
            recording_camera_mode=str(args.recording_camera_mode),
            recording_overlay=bool(args.recording_overlay),
            soft_stale_ticks=int(args.soft_stale_ticks),
            hard_stale_ticks=int(args.hard_stale_ticks),
        )
        self.monitor = Stage4Monitor(output_dir=self.output_dir)
        self.stop_scene_alignment = apply_stop_scene_alignment(
            world=self.world,
            ego_vehicle=self.ego_vehicle,
            scenario_manifest_payload=self.scenario_manifest_payload,
            actor_bindings=self.scenario_actor_bindings,
            args=args,
        )
        if bool(getattr(args, "shadow_mode", False)):
            self.shadow_runtime = Stage4ShadowRuntime(
                output_dir=self.output_dir,
                case_id=self.output_dir.parent.name or "stage4_online_case",
            )

    def _spawn_fallback_ego_vehicle(self) -> Any:
        blueprint_filter = str(getattr(self.args, "ego_vehicle_filter", "vehicle.lincoln.mkz_2020"))
        spawn_point_index = max(0, int(getattr(self.args, "ego_spawn_point_index", 0)))
        blueprint_library = self.world.get_blueprint_library()
        candidates = blueprint_library.filter(blueprint_filter)
        if not candidates:
            candidates = blueprint_library.filter("vehicle.*")
        if not candidates:
            raise RuntimeError("No vehicle blueprint available for fallback ego spawn.")
        spawn_points = list(self.map_inst.get_spawn_points())
        if not spawn_points:
            raise RuntimeError("CARLA map has no spawn points for fallback ego spawn.")
        ego_vehicle = None
        resolved_spawn_index = spawn_point_index % len(spawn_points)
        for offset in range(len(spawn_points)):
            candidate_index = (spawn_point_index + offset) % len(spawn_points)
            spawn_point = spawn_points[candidate_index]
            ego_vehicle = self.world.try_spawn_actor(candidates[0], spawn_point)
            if ego_vehicle is not None:
                resolved_spawn_index = candidate_index
                break
        if ego_vehicle is None:
            raise RuntimeError(
                "Failed to spawn fallback ego vehicle; all spawn points appear blocked. "
                f"filter={blueprint_filter} start_spawn_point_index={spawn_point_index}"
            )
        self._spawned_ego_vehicle = True
        self._spawned_runtime_actors.append(ego_vehicle)
        LOGGER.info(
            "Spawned fallback ego actor id=%s blueprint=%s spawn_point_index=%d",
            int(ego_vehicle.id),
            str(candidates[0].id),
            resolved_spawn_index,
        )
        return ego_vehicle

    def _stage4_manifest(self) -> Dict[str, Any]:
        return {
            "runtime_architecture": {
                "selected": "split_process_watch_orchestrator",
                "host_role": "CARLA sync loop, capture, Stage2/3A/3B runtime, Stage3C execution bridge",
                "container_role": "BEVFusion Stage1 watch-mode worker over shared samples directory",
                "why": [
                    "reuses existing Stage1 watch mode without forcing host CUDA/model environment parity",
                    "keeps tracker/lane/route/execution state in one host process for simpler debugging",
                    "avoids rewriting Stage 1-3C logic into a monolith",
                ],
            },
            "args": vars(self.args),
            "carla_map": self.map_inst.name,
            "ego_actor_id": int(self.ego_vehicle.id),
            "spawned_fallback_ego": bool(self._spawned_ego_vehicle),
            "stage1_session_root": str(self.perception_adapter.session_root),
            "samples_root": str(self.samples_root),
            "output_dir": str(self.output_dir),
            "scenario_manifest_payload": self.scenario_manifest_payload,
            "scenario_actor_materialization": self.scenario_actor_materialization,
            "stop_scene_alignment": self.stop_scene_alignment,
            "control_authority_sandbox_mode": self.control_authority_sandbox_mode,
        }

    def _control_authority_sandbox_enabled(self) -> bool:
        return self.control_authority_sandbox_mode not in {"", "disabled", "none"}

    def _control_authority_sandbox_end_tick_exclusive(self) -> int | None:
        if self.control_authority_sandbox_handoff_duration_ticks <= 0:
            return None
        if self.control_authority_sandbox_handoff_tick < 0:
            return None
        return int(self.control_authority_sandbox_handoff_tick + self.control_authority_sandbox_handoff_duration_ticks)

    def _control_authority_sandbox_active_for_tick(self, tick_id: int) -> bool:
        if not self._control_authority_sandbox_enabled():
            return False
        if self.control_authority_sandbox_handoff_tick < 0:
            return False
        tick_id = int(tick_id)
        if tick_id < self.control_authority_sandbox_handoff_tick:
            return False
        handoff_end_tick_exclusive = self._control_authority_sandbox_end_tick_exclusive()
        if handoff_end_tick_exclusive is not None and tick_id >= handoff_end_tick_exclusive:
            return False
        return True

    def _full_shadow_authority_mode(self) -> bool:
        return self.control_authority_sandbox_mode == "full_shadow_authority"

    def _record_control_authority_sandbox_event(
        self,
        *,
        tick_id: int,
        carla_frame: int,
        timestamp: float,
        requested_behavior: str | None,
        arbitration_action: str,
        planner_execution_latency_ms: float,
        speed_mps: float | None,
        fallback_activated: bool,
        no_motion_guard_active: bool,
        bounded_motion_guard_active: bool,
        speed_cap_enforced: bool,
        handoff_requested: bool,
        handoff_active: bool,
        shadow_proposal_available: bool,
        actual_shadow_takeover_applied: bool,
        executed_control_owner: str,
        shadow_candidate_control: Dict[str, Any] | None,
        notes: List[str],
    ) -> None:
        payload = {
            "tick_id": int(tick_id),
            "carla_frame": int(carla_frame),
            "timestamp": float(timestamp),
            "sandbox_mode": self.control_authority_sandbox_mode,
            "handoff_requested": bool(handoff_requested),
            "handoff_active": bool(handoff_active),
            "handoff_executed": bool(handoff_active and shadow_proposal_available),
            "requested_behavior": requested_behavior,
            "arbitration_action": arbitration_action,
            "candidate_control_owner": "mpc_shadow_candidate" if self.shadow_runtime is not None else None,
            "executed_control_owner": str(executed_control_owner),
            "actual_shadow_takeover_applied": bool(actual_shadow_takeover_applied),
            "no_motion_guard_active": bool(no_motion_guard_active),
            "bounded_motion_guard_active": bool(bounded_motion_guard_active),
            "speed_cap_enforced": bool(speed_cap_enforced),
            "shadow_proposal_available": bool(shadow_proposal_available),
            "shadow_candidate_control": None if shadow_candidate_control is None else dict(shadow_candidate_control),
            "planner_execution_latency_ms": float(planner_execution_latency_ms),
            "speed_mps": speed_mps,
            "fallback_activated": bool(fallback_activated),
            "notes": list(notes),
        }
        self.control_authority_sandbox_rows.append(payload)
        _append_jsonl(self.control_authority_sandbox_trace_path, payload)

    def _finalize_control_authority_sandbox_summary(self) -> None:
        if not self._control_authority_sandbox_enabled():
            return
        rows = list(self.control_authority_sandbox_rows)
        handoff_rows = [row for row in rows if bool(row.get("handoff_active"))]
        handoff_ticks = [int(row["tick_id"]) for row in handoff_rows if row.get("tick_id") is not None]
        handoff_executed = any(bool(row.get("handoff_executed")) for row in rows)
        no_motion_guard_observed = any(bool(row.get("no_motion_guard_active")) for row in rows)
        bounded_motion_guard_observed = any(bool(row.get("bounded_motion_guard_active")) for row in rows)
        speed_values = [
            float(row["speed_mps"])
            for row in rows
            if row.get("speed_mps") is not None
        ]
        handoff_speed_values = [
            float(row["speed_mps"])
            for row in handoff_rows
            if row.get("speed_mps") is not None
        ]
        max_speed_mps = max(speed_values) if speed_values else None
        max_speed_mps_in_handoff_window = max(handoff_speed_values) if handoff_speed_values else None
        observed_handoff_start_tick = min(handoff_ticks) if handoff_ticks else None
        observed_handoff_end_tick = max(handoff_ticks) if handoff_ticks else None
        handoff_duration_ticks_observed = len(handoff_ticks)
        post_handoff_rows = (
            [
                row
                for row in rows
                if observed_handoff_end_tick is not None and int(row.get("tick_id", -1)) > observed_handoff_end_tick
            ]
            if observed_handoff_end_tick is not None
            else []
        )
        post_handoff_baseline_rows = [
            row
            for row in post_handoff_rows
            if str(row.get("executed_control_owner") or "") == "baseline_execution_runtime"
            and not bool(row.get("actual_shadow_takeover_applied"))
        ]
        rollback_to_baseline_observed = bool(post_handoff_baseline_rows)
        rollback_tick = (
            min(int(row["tick_id"]) for row in post_handoff_baseline_rows if row.get("tick_id") is not None)
            if post_handoff_baseline_rows
            else None
        )
        rollback_delay_ticks = (
            None
            if rollback_tick is None or observed_handoff_end_tick is None
            else int(rollback_tick - observed_handoff_end_tick)
        )
        max_speed_within_limit = (
            True
            if max_speed_mps_in_handoff_window is None
            else bool(max_speed_mps_in_handoff_window <= self.control_authority_sandbox_max_speed_mps)
        )
        motion_observed_in_handoff_window = (
            False
            if max_speed_mps_in_handoff_window is None
            else bool(max_speed_mps_in_handoff_window >= self.control_authority_sandbox_min_speed_mps)
        )
        actual_shadow_takeover_applied = any(bool(row.get("actual_shadow_takeover_applied")) for row in rows)
        executed_control_owners = sorted(
            {
                str(row.get("executed_control_owner") or "")
                for row in rows
                if str(row.get("executed_control_owner") or "").strip()
            }
        )
        executed_control_owner = (
            "mpc_shadow_candidate_sandbox"
            if "mpc_shadow_candidate_sandbox" in executed_control_owners
            else "baseline_execution_runtime"
        )
        speed_cap_enforced_count = sum(1 for row in rows if bool(row.get("speed_cap_enforced")))
        full_authority_tail_rows = (
            [
                row
                for row in rows
                if observed_handoff_start_tick is not None
                and int(row.get("tick_id", -1)) >= observed_handoff_start_tick
                and str(row.get("executed_control_owner") or "") == "mpc_shadow_candidate_sandbox"
                and bool(row.get("actual_shadow_takeover_applied"))
            ]
            if observed_handoff_start_tick is not None
            else []
        )
        full_authority_until_end_observed = bool(
            self._full_shadow_authority_mode()
            and observed_handoff_start_tick is not None
            and rows
            and full_authority_tail_rows
            and len(full_authority_tail_rows)
            == len([row for row in rows if int(row.get("tick_id", -1)) >= observed_handoff_start_tick])
        )
        if self.control_authority_sandbox_mode == "bounded_motion_shadow_authority":
            ready_for_sandbox_discussion = bool(
                handoff_executed
                and bounded_motion_guard_observed
                and motion_observed_in_handoff_window
                and max_speed_within_limit
                and actual_shadow_takeover_applied
                and executed_control_owner == "mpc_shadow_candidate_sandbox"
            )
            notes = [
                "Sandbox handoff transfers actuator authority to a tightly bounded shadow-derived control for a short window.",
                "bounded_motion_shadow_authority applies shadow candidate control and then enforces the same bounded-motion speed guard.",
            ]
        elif self.control_authority_sandbox_mode == "full_shadow_authority":
            ready_for_sandbox_discussion = bool(
                handoff_executed
                and bounded_motion_guard_observed
                and motion_observed_in_handoff_window
                and max_speed_within_limit
                and actual_shadow_takeover_applied
                and executed_control_owner == "mpc_shadow_candidate_sandbox"
                and full_authority_until_end_observed
                and not rollback_to_baseline_observed
            )
            notes = [
                "Sandbox handoff transfers actuator authority to a shadow-derived control for the full remaining runtime window.",
                "full_shadow_authority keeps shadow-derived control active through the end of the run and does not roll back to baseline inside the sandbox segment.",
            ]
        elif self.control_authority_sandbox_mode == "bounded_motion_handoff":
            ready_for_sandbox_discussion = bool(
                handoff_executed
                and bounded_motion_guard_observed
                and motion_observed_in_handoff_window
                and max_speed_within_limit
                and not actual_shadow_takeover_applied
            )
            notes = [
                "Sandbox handoff never transfers actuator ownership away from the baseline controller.",
                "bounded_motion_handoff clamps baseline-applied control inside a capped speed window without handing actuator ownership to shadow MPC.",
            ]
        else:
            ready_for_sandbox_discussion = bool(
                handoff_executed
                and no_motion_guard_observed
                and max_speed_within_limit
                and not actual_shadow_takeover_applied
            )
            notes = [
                "Sandbox handoff never transfers actuator ownership away from the baseline controller.",
                "no_motion_handoff overrides applied vehicle control with neutral brake before the CARLA tick.",
            ]
        summary = {
            "schema_version": "stage6.control_authority_handoff_sandbox_summary.v2",
            "sandbox_mode": self.control_authority_sandbox_mode,
            "handoff_tick": self.control_authority_sandbox_handoff_tick,
            "handoff_duration_ticks_configured": self.control_authority_sandbox_handoff_duration_ticks,
            "handoff_end_tick_exclusive": self._control_authority_sandbox_end_tick_exclusive(),
            "num_events": len(rows),
            "handoff_requested_count": sum(1 for row in rows if bool(row.get("handoff_requested"))),
            "handoff_executed_count": sum(1 for row in rows if bool(row.get("handoff_executed"))),
            "handoff_executed": bool(handoff_executed),
            "handoff_start_tick_observed": observed_handoff_start_tick,
            "handoff_end_tick_observed": observed_handoff_end_tick,
            "handoff_duration_ticks_observed": int(handoff_duration_ticks_observed),
            "no_motion_guard_observed": bool(no_motion_guard_observed),
            "bounded_motion_guard_observed": bool(bounded_motion_guard_observed),
            "max_speed_mps": max_speed_mps,
            "max_speed_mps_in_handoff_window": max_speed_mps_in_handoff_window,
            "max_speed_limit_mps": self.control_authority_sandbox_max_speed_mps,
            "min_speed_limit_mps": self.control_authority_sandbox_min_speed_mps,
            "max_speed_within_limit": bool(max_speed_within_limit),
            "motion_observed_in_handoff_window": bool(motion_observed_in_handoff_window),
            "speed_cap_enforced_count": int(speed_cap_enforced_count),
            "actual_shadow_takeover_applied": bool(actual_shadow_takeover_applied),
            "executed_control_owner": executed_control_owner,
            "bounded_authority_window": not self._full_shadow_authority_mode(),
            "full_authority_until_end_observed": bool(full_authority_until_end_observed),
            "full_authority_tail_tick_count": int(len(full_authority_tail_rows)),
            "candidate_control_owner": "mpc_shadow_candidate" if self.shadow_runtime is not None else None,
            "rollback_to_baseline_observed": bool(rollback_to_baseline_observed),
            "rollback_tick": rollback_tick,
            "rollback_delay_ticks": rollback_delay_ticks,
            "post_handoff_baseline_tick_count": int(len(post_handoff_baseline_rows)),
            "ready_for_sandbox_discussion": bool(ready_for_sandbox_discussion),
            "notes": notes,
        }
        _write_json(self.control_authority_sandbox_summary_path, summary)

    def _tick_until_capture(self, sync_mode: CarlaSynchronousMode, phase_label: str) -> tuple[Any, int]:
        last_error = None
        for attempt in range(max(1, int(self.args.max_frame_retries))):
            frame = sync_mode.tick()
            snapshot = self.world.get_snapshot()
            try:
                if self.collector is None:
                    raise RuntimeError("Collector not initialized.")
                capture = self.collector.wait_for_frame(frame, snapshot)
                if attempt > 0:
                    LOGGER.info(
                        "%s resynchronized on frame=%d after %d retry(s)",
                        phase_label,
                        frame,
                        attempt,
                    )
                return capture, attempt
            except TimeoutError as exc:
                last_error = exc
                LOGGER.warning(
                    "%s missed synchronized frame=%d (attempt %d/%d): %s",
                    phase_label,
                    frame,
                    attempt + 1,
                    max(1, int(self.args.max_frame_retries)),
                    exc,
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{phase_label} failed without a synchronized capture.")

    def _process_available_predictions(self) -> List[BehaviorFrameState]:
        updates: List[BehaviorFrameState] = []
        for perception_frame in self.perception_adapter.poll_predictions():
            try:
                behavior_state = self.world_behavior_runtime.process_prediction(perception_frame)
            except Exception:
                LOGGER.exception(
                    "Stage4 world/behavior update failed for sample=%s frame=%s",
                    perception_frame.sample_name,
                    perception_frame.frame_id,
                )
                if bool(self.args.reset_state_on_stage_failure):
                    self.world_behavior_runtime.reset()
                    self.state_resets += 1
                continue
            self.last_behavior_state = behavior_state
            self.monitor.record_behavior_update(behavior_state)
            updates.append(behavior_state)
        return updates

    def _capture_and_register(self, capture: Any, tick_id: int) -> None:
        if self.dumper is None:
            raise RuntimeError("SampleDumper not initialized.")
        dump_result = self.dumper.dump_capture(
            capture,
            ego_vehicle=self.ego_vehicle,
            world=self.world,
            sample_index=self.sample_index,
            previous_sample_name=self.previous_sample_name,
        )
        self.previous_sample_name = dump_result.sample_dir.name
        self.sample_index += 1
        self.last_capture_sample_name = dump_result.sample_dir.name
        self.perception_adapter.register_captured_sample(
            sample_name=dump_result.sample_dir.name,
            tick_id=tick_id,
            carla_frame=int(capture.frame),
            capture_monotonic_s=time.monotonic(),
        )

    def _current_health_flags(self) -> List[str]:
        flags: List[str] = []
        if self.perception_adapter.worker_failed():
            flags.append("stage1_worker_failed")
        if self.last_behavior_state is None:
            flags.append("no_behavior_state_yet")
        if self.perception_stale_ticks >= int(self.args.hard_stale_ticks):
            flags.append("perception_hard_stale")
        elif self.perception_stale_ticks >= int(self.args.soft_stale_ticks):
            flags.append("perception_soft_stale")
        return sorted(set(flags))

    def _build_tracker_state(self) -> Dict[str, Any]:
        if self.last_behavior_state is None:
            return {
                "num_tracks": 0,
                "track_ids": [],
                "highest_risk_level": None,
            }
        return {
            "num_tracks": int(len(self.last_behavior_state.tracked_objects)),
            "track_ids": [int(item["track_id"]) for item in self.last_behavior_state.tracked_objects],
            "highest_risk_level": self.last_behavior_state.risk_summary.get("highest_risk_level"),
            "front_hazard_track_id": self.last_behavior_state.risk_summary.get("front_hazard_track_id"),
        }

    def _blocker_distance_m(self) -> float | None:
        if self.last_behavior_state is None:
            return None
        blocker = self.last_behavior_state.lane_scene.get("front_object_in_current_lane")
        if blocker and blocker.get("longitudinal_m") is not None:
            return float(blocker["longitudinal_m"])
        nearest = self.last_behavior_state.world_state.get("scene", {}).get("nearest_front_vehicle")
        if nearest and nearest.get("distance_m") is not None:
            return float(nearest["distance_m"])
        return None

    def _build_stack_state(
        self,
        *,
        tick_id: int,
        carla_frame: int,
        execution_state: Dict[str, Any] | None,
        health_flags: List[str],
        behavior_override_counts: Dict[str, int],
    ) -> OnlineStackState:
        return OnlineStackState(
            tick_id=int(tick_id),
            carla_frame=int(carla_frame),
            last_capture_sample=self.last_capture_sample_name,
            last_perception_frame_id=(
                None if self.last_behavior_state is None else int(self.last_behavior_state.frame_id)
            ),
            last_behavior_frame_id=(
                None if self.last_behavior_state is None else int(self.last_behavior_state.frame_id)
            ),
            tracker_state=self._build_tracker_state(),
            current_behavior=(
                None if self.last_behavior_state is None else dict(self.last_behavior_state.behavior_request_v2)
            ),
            execution_state=None if execution_state is None else dict(execution_state),
            route_context=(
                None if self.last_behavior_state is None else dict(self.last_behavior_state.route_context)
            ),
            last_valid_world_state=(
                None if self.last_behavior_state is None else dict(self.last_behavior_state.world_state)
            ),
            health_flags=list(health_flags),
            dropped_frames=int(self.retry_dropped_frames + self.perception_adapter.unresolved_capture_count()),
            skipped_updates=int(self.skipped_updates),
            state_resets=int(self.state_resets),
            behavior_override_counts=dict(behavior_override_counts),
        )

    def run(self) -> Dict[str, Any]:
        from carla_bevfusion_stage1.collector import CarlaSynchronousMode
        from carla_bevfusion_stage1.rig import destroy_actors, spawn_rig

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_root.mkdir(parents=True, exist_ok=True)
        _write_json(self.output_dir / "stage4_manifest.json", self._stage4_manifest())
        if self.scenario_actor_materialization is not None:
            _write_json(
                self.output_dir / "critical_actor_materialization.json",
                dict(self.scenario_actor_materialization),
            )
        if self.stop_scene_alignment is not None:
            _write_json(
                self.output_dir / "stop_scene_alignment.json",
                dict(self.stop_scene_alignment),
            )

        self.perception_adapter.start()
        self.sensor_actors = {}
        behavior_override_counts: Dict[str, int] = {}

        try:
            with CarlaSynchronousMode(
                self.world,
                fixed_delta_seconds=float(self.args.fixed_delta_seconds),
                traffic_manager=self.traffic_manager,
            ) as sync_mode:
                self.sensor_actors = spawn_rig(self.world, self.ego_vehicle, self.rig)
                self.collector = self._FrameCollector(
                    self.sensor_actors,
                    timeout_seconds=float(self.args.timeout_seconds),
                )
                self.collector.start()
                self.dumper = self._SampleDumper(
                    output_root=self.samples_root,
                    rig_preset=self.rig,
                    sensor_actors=self.sensor_actors,
                )
                _apply_neutral_brake(self.ego_vehicle)

                for warmup_index in range(max(0, int(self.args.warmup_ticks))):
                    capture, retry_count = self._tick_until_capture(sync_mode, f"warmup[{warmup_index}]")
                    self.retry_dropped_frames += int(retry_count)
                    self._capture_and_register(capture, tick_id=-(warmup_index + 1))
                    time.sleep(float(self.args.stage1_poll_interval_seconds))
                    self._process_available_predictions()

                for tick_id in range(int(self.args.num_ticks)):
                    pre_updates = self._process_available_predictions()
                    proposed_request = None
                    if self.last_behavior_state is not None:
                        proposed_request = build_execution_request(
                            {"behavior_request_v2": self.last_behavior_state.behavior_request_v2}
                        )
                        proposed_request.stop_target = enrich_runtime_stop_target(
                            proposed_request.stop_target,
                            self.scenario_manifest_payload,
                            self.scenario_actor_bindings,
                        )

                    effective_proposed_request = proposed_request
                    effective_perception_stale_ticks = int(self.perception_stale_ticks)
                    fault_injection_notes: list[str] = []
                    fallback_probe_tick = int(getattr(self.args, "fallback_probe_tick", -1))
                    fallback_probe_trigger = str(getattr(self.args, "fallback_probe_trigger", "") or "")
                    if fallback_probe_tick >= 0 and tick_id == fallback_probe_tick:
                        if fallback_probe_trigger == "hard_timeout":
                            effective_proposed_request = None
                            effective_perception_stale_ticks = max(
                                effective_perception_stale_ticks,
                                int(self.args.hard_stale_ticks),
                            )
                            fault_injection_notes.append("fallback_probe:hard_timeout")
                        elif fallback_probe_trigger == "soft_timeout":
                            effective_proposed_request = None
                            effective_perception_stale_ticks = max(
                                effective_perception_stale_ticks,
                                int(self.args.soft_stale_ticks),
                            )
                            fault_injection_notes.append("fallback_probe:soft_timeout")
                    fallback_perturbation_mode = str(
                        getattr(self.args, "fallback_perturbation_mode", "disabled") or "disabled"
                    )
                    fallback_perturbation_start_tick = int(
                        getattr(self.args, "fallback_perturbation_start_tick", -1)
                    )
                    if (
                        effective_proposed_request is not None
                        and fallback_perturbation_mode == "route_uncertainty"
                        and fallback_perturbation_start_tick >= 0
                        and tick_id >= fallback_perturbation_start_tick
                    ):
                        updated_constraints = sorted(
                            set(list(effective_proposed_request.constraints) + ["route_option_unknown"])
                        )
                        effective_proposed_request = _clone_execution_request_with_updates(
                            effective_proposed_request,
                            constraints=updated_constraints,
                            confidence=min(0.5, float(effective_proposed_request.confidence)),
                        )
                        fault_injection_notes.append("fallback_perturbation:route_uncertainty")

                    snapshot_before = self.world.get_snapshot()
                    prepare_info = self.execution_runtime.prepare_tick(
                        tick_id=tick_id,
                        snapshot_before=snapshot_before,
                        proposed_request=effective_proposed_request,
                        perception_stale_ticks=effective_perception_stale_ticks,
                        last_behavior_frame_id=(
                            None if self.last_behavior_state is None else int(self.last_behavior_state.frame_id)
                        ),
                    )
                    sandbox_handoff_requested = (
                        self._control_authority_sandbox_enabled()
                        and int(tick_id) == self.control_authority_sandbox_handoff_tick
                    )
                    sandbox_handoff_active = self._control_authority_sandbox_active_for_tick(tick_id)
                    no_motion_guard_active = (
                        sandbox_handoff_active
                        and self.control_authority_sandbox_mode == "no_motion_handoff"
                    )
                    bounded_motion_guard_active = (
                        sandbox_handoff_active
                        and self.control_authority_sandbox_mode in {"bounded_motion_handoff", "bounded_motion_shadow_authority", "full_shadow_authority"}
                    )
                    shadow_authority_takeover_active = (
                        sandbox_handoff_active
                        and self.control_authority_sandbox_mode in {"bounded_motion_shadow_authority", "full_shadow_authority"}
                    )
                    shadow_proposal_available = bool(self.shadow_runtime and self.shadow_runtime.proposals)
                    shadow_candidate_control = None
                    actual_shadow_takeover_applied = False
                    executed_control_owner = "baseline_execution_runtime"
                    speed_cap_enforced = False
                    if no_motion_guard_active:
                        _apply_neutral_brake(self.ego_vehicle)
                    elif shadow_authority_takeover_active and shadow_proposal_available and self.shadow_runtime is not None:
                        current_speed_mps = _vehicle_speed_mps(self.ego_vehicle)
                        latest_shadow_proposal = dict(self.shadow_runtime.proposals[-1])
                        candidate_control = _build_shadow_candidate_control(
                            carla_module=self.carla,
                            proposal=latest_shadow_proposal,
                            current_speed_mps=current_speed_mps,
                        )
                        self.ego_vehicle.apply_control(candidate_control)
                        shadow_candidate_control = {
                            "throttle": float(getattr(candidate_control, "throttle", 0.0)),
                            "steer": float(getattr(candidate_control, "steer", 0.0)),
                            "brake": float(getattr(candidate_control, "brake", 0.0)),
                        }
                        actual_shadow_takeover_applied = True
                        executed_control_owner = "mpc_shadow_candidate_sandbox"
                        _apply_bounded_motion_guard(
                            self.ego_vehicle,
                            current_speed_mps=current_speed_mps,
                            speed_limit_mps=self.control_authority_sandbox_max_speed_mps,
                        )
                        speed_cap_enforced = bool(
                            current_speed_mps >= max(
                                self.control_authority_sandbox_min_speed_mps,
                                self.control_authority_sandbox_max_speed_mps * 0.85,
                            )
                        )
                    elif bounded_motion_guard_active:
                        current_speed_mps = _vehicle_speed_mps(self.ego_vehicle)
                        _apply_bounded_motion_guard(
                            self.ego_vehicle,
                            current_speed_mps=current_speed_mps,
                            speed_limit_mps=self.control_authority_sandbox_max_speed_mps,
                        )
                        speed_cap_enforced = bool(
                            current_speed_mps >= max(
                                self.control_authority_sandbox_min_speed_mps,
                                self.control_authority_sandbox_max_speed_mps * 0.85,
                            )
                        )
                    capture, retry_count = self._tick_until_capture(sync_mode, f"tick[{tick_id}]")
                    self.retry_dropped_frames += int(retry_count)
                    self._capture_and_register(capture, tick_id=tick_id)
                    finalize_info = self.execution_runtime.finalize_tick(snapshot_after=capture.world_snapshot)

                    time.sleep(float(self.args.stage1_poll_interval_seconds))
                    post_updates = self._process_available_predictions()
                    latest_update = (post_updates or pre_updates)
                    behavior_updated_this_tick = bool(latest_update)
                    if behavior_updated_this_tick:
                        self.perception_stale_ticks = 0
                    else:
                        self.perception_stale_ticks += 1
                        self.skipped_updates += 1

                    health_flags = self._current_health_flags()
                    arbitration_action = str(finalize_info["arbitration_action"])
                    behavior_override_counts[arbitration_action] = behavior_override_counts.get(arbitration_action, 0) + 1
                    planner_prepare_latency_ms = float(finalize_info.get("planner_prepare_latency_ms") or 0.0)
                    execution_finalize_latency_ms = float(finalize_info.get("execution_finalize_latency_ms") or 0.0)
                    planner_execution_latency_ms = float(finalize_info.get("planner_execution_latency_ms") or 0.0)
                    world_update_latency_ms = (
                        None if not latest_update else latest_update[-1].world_update_latency_ms
                    )
                    behavior_latency_ms = (
                        None if not latest_update else latest_update[-1].behavior_latency_ms
                    )
                    perception_latency_ms = (
                        None if not latest_update else latest_update[-1].perception_latency_ms
                    )
                    decision_to_execution_latency_ms = float(
                        (world_update_latency_ms or 0.0)
                        + (behavior_latency_ms or 0.0)
                        + planner_execution_latency_ms
                    )
                    total_loop_latency_ms = float(
                        (perception_latency_ms or 0.0)
                        + decision_to_execution_latency_ms
                    )
                    latency_budget_ms = float(getattr(self.args, "latency_budget_ms", 150.0))
                    execution_state = dict(finalize_info.get("execution_state") or {})
                    timeline_record = dict(finalize_info.get("timeline_record") or {})
                    planner_diag = dict(timeline_record.get("planner_diagnostics") or {})
                    requested_behavior = (
                        None
                        if self.last_behavior_state is None
                        else str(self.last_behavior_state.behavior_request_v2["requested_behavior"])
                    )
                    executing_behavior = str(execution_state.get("current_behavior") or "")
                    execution_status = str(execution_state.get("execution_status") or "")
                    organic_fallback_notes = sorted(
                        {
                            str(note)
                            for note in [
                                *((execution_state.get("notes") or [])),
                                *((planner_diag.get("planner_notes") or [])),
                            ]
                            if note
                        }
                    )
                    organic_fallback_required = bool(
                        (requested_behavior and requested_behavior != executing_behavior)
                        or execution_status in {"fail", "aborted"}
                        or any(
                            note
                            in {
                                "lane_change_plan_unavailable",
                                "current_waypoint_not_found",
                                "active_lane_change_aborted_for_stop",
                                "blocking_object_id_not_bound_to_live_actor",
                            }
                            for note in organic_fallback_notes
                        )
                    )
                    fallback_required = bool(
                        arbitration_action in {"emergency_override", "downgrade", "cancel"}
                        or organic_fallback_required
                        or any(flag in {"stage1_worker_failed", "no_behavior_state_yet", "perception_hard_stale"} for flag in health_flags)
                    )
                    fallback_activated = bool(
                        arbitration_action in {"emergency_override", "downgrade", "cancel"}
                        or organic_fallback_required
                    )
                    fallback_reason = ";".join(
                        item
                        for item in [
                            arbitration_action if arbitration_action in {"emergency_override", "downgrade", "cancel"} else "",
                            *health_flags,
                            *list(prepare_info["arbitration_notes"]),
                            *list(finalize_info["arbitration_notes"]),
                            *organic_fallback_notes,
                            *fault_injection_notes,
                        ]
                        if item
                    )
                    fallback_mode = ({
                        "emergency_override": "deterministic_emergency_stop",
                        "downgrade": "deterministic_speed_downgrade",
                        "cancel": "deterministic_cancel_to_last_effective_request",
                    }.get(arbitration_action) or ("deterministic_local_planner_bridge" if organic_fallback_required else None))
                    safe_fallback_behaviors = {"keep_lane", "slow_down", "stop_before_obstacle"}
                    tick_record = OnlineTickRecord(
                        tick_id=int(tick_id),
                        frame_id=int(capture.frame),
                        timestamp=float(capture.world_snapshot.timestamp.elapsed_seconds),
                        perception_latency_ms=perception_latency_ms,
                        world_update_latency_ms=world_update_latency_ms,
                        behavior_latency_ms=behavior_latency_ms,
                        execution_latency_ms=float(finalize_info["execution_latency_ms"]),
                        planner_prepare_latency_ms=planner_prepare_latency_ms,
                        execution_finalize_latency_ms=execution_finalize_latency_ms,
                        planner_execution_latency_ms=planner_execution_latency_ms,
                        decision_to_execution_latency_ms=decision_to_execution_latency_ms,
                        total_loop_latency_ms=total_loop_latency_ms,
                        latency_budget_ms=latency_budget_ms,
                        over_budget=bool(total_loop_latency_ms > latency_budget_ms),
                        selected_behavior=requested_behavior,
                        execution_status=execution_status,
                        ego_lane_id=execution_state.get("current_lane_id"),
                        speed_mps=execution_state.get("speed_mps"),
                        blocker_distance_m=self._blocker_distance_m(),
                        perception_frame_id=(
                            None if not latest_update else int(latest_update[-1].frame_id)
                        ),
                        behavior_frame_id=(
                            None if self.last_behavior_state is None else int(self.last_behavior_state.frame_id)
                        ),
                        arbitration_action=arbitration_action,
                        fallback_required=fallback_required,
                        fallback_activated=fallback_activated,
                        fallback_reason=fallback_reason or None,
                        fallback_source=(
                            "stage4_arbitration"
                            if arbitration_action in {"emergency_override", "downgrade", "cancel"}
                            else ("stage4_execution_bridge" if organic_fallback_required else None)
                        ),
                        fallback_mode=fallback_mode,
                        fallback_action=executing_behavior or None,
                        fallback_takeover_latency_ms=(
                            planner_execution_latency_ms if fallback_activated else None
                        ),
                        fallback_success=(
                            None
                            if not fallback_activated
                            else bool(executing_behavior in safe_fallback_behaviors)
                        ),
                        health_flags=health_flags,
                        notes=list(prepare_info["arbitration_notes"])
                        + list(finalize_info["arbitration_notes"])
                        + organic_fallback_notes
                        + list(fault_injection_notes)
                        + ([f"capture_retries={retry_count}"] if retry_count else []),
                    )
                    stack_state = self._build_stack_state(
                        tick_id=tick_id,
                        carla_frame=int(capture.frame),
                        execution_state=finalize_info["execution_state"],
                        health_flags=health_flags,
                        behavior_override_counts=behavior_override_counts,
                    )
                    self.monitor.record_tick(tick_record, stack_state)
                    if self.shadow_runtime is not None:
                        self.shadow_runtime.record_tick(
                            timeline_record=dict(finalize_info["timeline_record"]),
                            planner_execution_latency_ms=planner_execution_latency_ms,
                        )
                    if self._control_authority_sandbox_enabled():
                        self._record_control_authority_sandbox_event(
                            tick_id=tick_id,
                            carla_frame=int(capture.frame),
                            timestamp=float(capture.world_snapshot.timestamp.elapsed_seconds),
                            requested_behavior=(
                                None
                                if self.last_behavior_state is None
                                else str(self.last_behavior_state.behavior_request_v2["requested_behavior"])
                            ),
                            arbitration_action=arbitration_action,
                            planner_execution_latency_ms=planner_execution_latency_ms,
                            speed_mps=tick_record.speed_mps,
                            fallback_activated=fallback_activated,
                            no_motion_guard_active=no_motion_guard_active,
                            bounded_motion_guard_active=bounded_motion_guard_active,
                            speed_cap_enforced=speed_cap_enforced,
                            handoff_requested=sandbox_handoff_requested,
                            handoff_active=sandbox_handoff_active,
                            shadow_proposal_available=shadow_proposal_available,
                            actual_shadow_takeover_applied=actual_shadow_takeover_applied,
                            executed_control_owner=executed_control_owner,
                            shadow_candidate_control=shadow_candidate_control,
                            notes=list(tick_record.notes),
                        )

                drain_deadline = time.monotonic() + float(self.args.stage1_drain_timeout_seconds)
                while (
                    self.perception_adapter.unresolved_capture_count() > 0
                    and time.monotonic() < drain_deadline
                ):
                    time.sleep(float(self.args.stage1_poll_interval_seconds))
                    self._process_available_predictions()

        finally:
            if self.collector is not None:
                self.collector.stop()
            if self.sensor_actors:
                destroy_actors(self.sensor_actors.values())
            try:
                _apply_neutral_brake(self.ego_vehicle)
            except Exception:
                LOGGER.exception("Failed to apply neutral brake during Stage 4 shutdown.")
            spawned_unique: List[Any] = []
            seen_actor_ids: set[int] = set()
            for actor in reversed(self._spawned_runtime_actors):
                actor_id = getattr(actor, "id", None)
                if actor_id is None or int(actor_id) in seen_actor_ids:
                    continue
                seen_actor_ids.add(int(actor_id))
                spawned_unique.append(actor)
            if spawned_unique:
                destroy_actors(spawned_unique)
            self.execution_runtime.close()
            self.perception_adapter.close()

        summary = summarize_stage4_session(
            tick_records=self.monitor.tick_records,
            behavior_updates=self.monitor.behavior_updates,
            execution_records=self.execution_runtime.monitor.timeline_records,
            lane_change_events=self.execution_runtime.monitor.lane_change_events,
            dropped_frames=self.retry_dropped_frames + self.perception_adapter.unresolved_capture_count(),
            skipped_updates=self.skipped_updates,
            state_resets=self.state_resets,
            stage1_summary=self.perception_adapter.build_summary(),
        )
        write_planner_quality_artifacts(
            self.output_dir,
            execution_records=self.execution_runtime.monitor.timeline_records,
            raw_lane_change_events=self.execution_runtime.monitor.lane_change_events,
            online_tick_records=self.monitor.tick_records,
            source_stage="stage4_online",
        )
        if self.shadow_runtime is not None:
            planner_quality_summary = {}
            planner_quality_summary_path = self.output_dir / "planner_quality_summary.json"
            if planner_quality_summary_path.exists():
                with planner_quality_summary_path.open("r", encoding="utf-8") as handle:
                    planner_quality_summary = json.load(handle)
            self.shadow_runtime.finalize(planner_quality_summary=planner_quality_summary)
        self._finalize_control_authority_sandbox_summary()
        if self.stop_scene_alignment is not None:
            stop_target_payload = {}
            stop_target_path = self.output_dir / "stop_target.json"
            if stop_target_path.exists():
                with stop_target_path.open("r", encoding="utf-8") as handle:
                    stop_target_payload = json.load(handle)
            stop_summary = dict(stop_target_payload.get("summary") or {})
            stop_events = list(stop_target_payload.get("stop_events") or [])
            if stop_events:
                representative_event = stop_events[0]
                alignment_consistent = bool(representative_event.get("runtime_alignment_consistent"))
                self.stop_scene_alignment["runtime_blocker_gap_m"] = representative_event.get("first_distance_to_target_m")
                self.stop_scene_alignment["alignment_error_m"] = representative_event.get("runtime_alignment_error_m")
                self.stop_scene_alignment["alignment_consistent"] = alignment_consistent
                self.stop_scene_alignment["alignment_ready"] = alignment_consistent
                self.stop_scene_alignment["notes"] = [
                    note
                    for note in list(self.stop_scene_alignment.get("notes") or [])
                    if not str(note).startswith("stop_alignment_")
                ]
                self.stop_scene_alignment["notes"].append(
                    "stop_alignment_ready_for_stop_quality_gate"
                    if alignment_consistent
                    else "stop_alignment_not_ready_for_stop_quality_gate"
                )
            elif stop_summary:
                alignment_consistent = bool(
                    int(stop_summary.get("alignment_consistent_events") or 0) > 0
                    and int(stop_summary.get("alignment_consistent_events") or 0)
                    == int(stop_summary.get("num_stop_events") or 0)
                )
                self.stop_scene_alignment["alignment_error_m"] = stop_summary.get("mean_runtime_alignment_error_m")
                self.stop_scene_alignment["alignment_consistent"] = alignment_consistent
                self.stop_scene_alignment["alignment_ready"] = alignment_consistent
            _write_json(
                self.output_dir / "stop_scene_alignment.json",
                dict(self.stop_scene_alignment),
            )
        _write_json(self.output_dir / "online_run_summary.json", summary)
        save_stage4_visualization_bundle(
            output_root=self.output_dir / "visualization",
            tick_records=self.monitor.tick_records,
        )
        LOGGER.info("Stage4 online run complete output_dir=%s", self.output_dir)
        return summary
