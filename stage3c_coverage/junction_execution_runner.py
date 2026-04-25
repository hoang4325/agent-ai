from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from stage3c.replay_runner_stage3c import run_closed_loop

from .evaluation_stage3c_coverage import load_json


def run_junction_execution(
    *,
    stage3b_output_dir: str | Path,
    output_dir: str | Path,
    scenario_manifest: str | None = None,
    attach_to_actor_id: int = 0,
    carla_host: str = "127.0.0.1",
    carla_port: int = 2000,
    carla_timeout_s: float = 10.0,
    carla_pythonapi_root: str = r"D:\carla\PythonAPI",
    fixed_delta_seconds: float = 0.10,
    warmup_ticks: int = 3,
    post_settle_ticks: int = 20,
    max_frames: int = 0,
    prepare_offset_m: float = 0.75,
    sampling_radius_m: float = 2.0,
    lane_change_success_hold_ticks: int = 4,
    max_lane_change_ticks: int = 45,
    lane_change_min_lateral_shift_m: float = 0.50,
    lane_change_min_lateral_shift_fraction: float = 0.15,
    stop_hold_ticks: int = 10,
    stop_full_stop_speed_threshold_mps: float = 0.35,
    stop_near_distance_m: float = 8.0,
    stop_hard_brake_distance_m: float = 3.0,
    record_mp4: bool = False,
    recording_path: str | None = None,
    recording_camera_mode: str = "chase",
    recording_width: int = 1280,
    recording_height: int = 720,
    recording_fov: float = 90.0,
    recording_fps: float = 0.0,
    recording_post_roll_ticks: int = 0,
    recording_overlay: bool = True,
) -> Dict[str, Any]:
    stage3b_summary_path = Path(stage3b_output_dir) / "evaluation_summary.json"
    if stage3b_summary_path.exists():
        stage3b_summary = load_json(stage3b_summary_path)
        if int(stage3b_summary.get("junction_frames", 0)) <= 0:
            raise ValueError(f"Stage 3B output does not look like a junction replay: {stage3b_summary_path}")

    args = SimpleNamespace(
        stage3b_output_dir=str(stage3b_output_dir),
        output_dir=str(output_dir),
        scenario_manifest=scenario_manifest,
        attach_to_actor_id=int(attach_to_actor_id),
        carla_host=str(carla_host),
        carla_port=int(carla_port),
        carla_timeout_s=float(carla_timeout_s),
        carla_pythonapi_root=str(carla_pythonapi_root),
        fixed_delta_seconds=float(fixed_delta_seconds),
        warmup_ticks=int(warmup_ticks),
        post_settle_ticks=int(post_settle_ticks),
        max_frames=int(max_frames),
        prepare_offset_m=float(prepare_offset_m),
        sampling_radius_m=float(sampling_radius_m),
        lane_change_success_hold_ticks=int(lane_change_success_hold_ticks),
        max_lane_change_ticks=int(max_lane_change_ticks),
        lane_change_min_lateral_shift_m=float(lane_change_min_lateral_shift_m),
        lane_change_min_lateral_shift_fraction=float(lane_change_min_lateral_shift_fraction),
        stop_hold_ticks=int(stop_hold_ticks),
        stop_full_stop_speed_threshold_mps=float(stop_full_stop_speed_threshold_mps),
        stop_near_distance_m=float(stop_near_distance_m),
        stop_hard_brake_distance_m=float(stop_hard_brake_distance_m),
        record_mp4=bool(record_mp4),
        recording_path=recording_path,
        recording_camera_mode=str(recording_camera_mode),
        recording_width=int(recording_width),
        recording_height=int(recording_height),
        recording_fov=float(recording_fov),
        recording_fps=float(recording_fps),
        recording_post_roll_ticks=int(recording_post_roll_ticks),
        recording_overlay=bool(recording_overlay),
    )
    return run_closed_loop(args)
