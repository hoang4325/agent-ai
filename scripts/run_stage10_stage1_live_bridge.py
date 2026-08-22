"""
scripts/run_stage10_stage1_live_bridge.py
==========================================
Stage 10 — Live CARLA → BEVFusion → Stage 9 bridge.

Connects every module we have built (Stage 1 → Stage 9) into a CARLA
synchronous closed-loop. Slow tactical Agent API calls run on a latest-only
background worker and never block the control loop:

  CARLA sensors ──► CarlaSensorSync
                       │  LiveFrame
                       ▼
             BEVFusionLiveAdapter  (bevfusion_runtime)
                       │  DetectionList
                       ▼
              WorldStateBuilder
                       │  WorldState
                       ▼
          [TODO S9] AuthorityArbiter.step()
                       │  ActuatorCommand
                       ▼
              carla_vehicle.apply_control()

Usage:
  python scripts/run_stage10_stage1_live_bridge.py \\
      --carla-root  "C:/CARLA" \\
      --bev-repo    "C:/bevfusion_repo" \\
      --bev-config  "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/...yaml" \\
      --bev-ckpt    "checkpoints/bevfusion.pth" \\
      --max-frames  500 \\
      --device      cuda

Exit code 0 = completed normally, 1 = error/crash.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from carla_bevfusion_stage1.carla_sensor_sync import LiveFrame
    from carla_bevfusion_stage1.world_state_builder import EgoTelemetry

# ── Ensure project root is on sys.path ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.async_agent_worker import AsyncAgentResult, LatestOnlyAgentWorker

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("stage10_live_bridge")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 10 – CARLA + BEVFusion live bridge")
    p.add_argument("--carla-root",   required=True,  help="Path to CARLA installation root")
    p.add_argument("--carla-host",   default="127.0.0.1")
    p.add_argument("--carla-port",   type=int, default=2000)
    p.add_argument("--tm-port",      type=int, default=8000)
    p.add_argument("--bev-repo",     required=True,  help="BEVFusion git repo root")
    p.add_argument("--bev-config",   required=True,  help="BEVFusion YAML config path")
    p.add_argument("--bev-ckpt",     required=True,  help="BEVFusion checkpoint .pth")
    p.add_argument("--output-root",  required=True,  help="Where to write inference results")
    p.add_argument("--samples-root", default=None,   help="Watch mode: poll this folder for dumped samples instead of connecting to CARLA")
    p.add_argument("--device",       default="cuda",  help="torch device")
    p.add_argument("--max-frames",   type=int, default=500)
    p.add_argument("--delta-t",      type=float, default=0.1,  help="Simulation step (s)")
    p.add_argument("--seed",         type=int, default=0,
                   help="Non-negative Python/NumPy/CARLA/Traffic Manager seed recorded with the run")
    p.add_argument("--rig-profile",  default="default",
                   choices=["default", "low_memory_shadow"],
                   help="Sensor rig quality preset")
    p.add_argument("--image-width",  type=int, default=1600)
    p.add_argument("--image-height", type=int, default=900)
    p.add_argument("--score-thresh", type=float, default=0.35)
    p.add_argument("--adapter-lidar-max-points", type=int, default=0,
                   help="Cap live LiDAR points before BEVFusion inference; 0 keeps all points")
    p.add_argument("--adapter-radar-max-points", type=int, default=2500,
                   help="Cap live radar points before BEVFusion inference")
    p.add_argument("--map",          default="Town01",  help="CARLA map name")
    p.add_argument("--spawn-point",  type=int, default=0, help="Ego spawn-point index")
    p.add_argument("--attach-to-actor-id", type=int, default=None,
                   help="Attach to an existing ego actor instead of spawning a new one")
    p.add_argument("--scenario-manifest", default=None,
                   help="Optional scenario manifest path. If --attach-to-actor-id is omitted, use ego_actor_id and town from this file")
    p.add_argument("--attach-autopilot", action="store_true",
                   help="When attaching to an existing ego actor, enable CARLA autopilot on that actor")
    p.add_argument("--route-target-spawn-point", type=int, default=None,
                   help="Optional destination spawn-point index for true route-completion tracking")
    p.add_argument("--route-distance-m", type=float, default=500.0,
                   help="Fallback mission distance for route-completion proxy when no destination is supplied")
    p.add_argument("--success-rc-threshold", type=float, default=0.95,
                   help="Route-completion threshold used by stage10_driving_metrics.json")
    p.add_argument("--collision-impulse-threshold", type=float, default=0.0,
                   help="Minimum collision impulse magnitude counted as a collision")
    p.add_argument("--enable-radar", action="store_true", default=True)
    p.add_argument("--radar-ablation", choices=("none", "zero_bev"), default="none",
                   help="Ablate radar branch (none=active, zero_bev=disabled)")
    p.add_argument("--log-dir",      default="benchmark/reports/stage10_live",
                   help="Directory to write JSONL authority log")
    p.add_argument("--no-stage9",    action="store_true",
                   help="Skip Stage 9 arbiter (perception-only dry run)")
    p.add_argument("--agent-mode",   default="stub",
                   choices=["stub", "api", "compare"],
                   help="Agent mode: stub=no LLM, api=real LLM, compare=baseline+agent side-by-side logging")
    p.add_argument("--agent-control-mode", default="shadow",
                   choices=["baseline", "shadow", "assist"],
                   help="baseline=no Agent control, shadow=Agent logs only, assist=bounded tactical Agent intent through MPC")
    p.add_argument("--agent-trigger-mode", choices=("every_frame", "risk_or_event"), default="risk_or_event",
                   help="When compare mode calls the LLM agent")
    p.add_argument("--agent-compare-stride", type=int, default=10,
                   help="Minimum frame stride for periodic Agent compare calls in risk_or_event mode")
    p.add_argument("--agent-risk-ttc-threshold", type=float, default=3.0,
                   help="Always query Agent when estimated TTC is below this threshold")
    p.add_argument("--agent-assist-min-confidence", type=float, default=0.50,
                   help="Minimum Agent confidence required before bounded assist can override the baseline request")
    p.add_argument("--agent-max-requests-per-minute", type=float, default=30.0,
                   help="Wall-clock rate limit for real Agent API calls; set <=0 to disable")
    p.add_argument("--agent-max-requests-per-episode", type=int, default=0,
                   help="Maximum submitted Agent requests per episode; 0 disables the cap")
    p.add_argument("--agent-api-timeout-s", type=float, default=5.0,
                   help="Per-request Agent API timeout. The control loop never waits for it")
    p.add_argument("--agent-api-max-retries", type=int, default=0,
                   help="Provider retries in the background Agent worker")
    p.add_argument("--agent-response-max-age-s", type=float, default=3.0,
                   help="Discard an Agent response older than this many simulated seconds")
    p.add_argument("--agent-retry-cooldown-s", type=float, default=2.0,
                   help="Wall-clock cooldown before retrying a failed/stale Agent decision")
    p.add_argument("--agent-maneuver-timeout-s", type=float, default=20.0,
                   help="Maximum duration of an accepted lane-change maneuver; safety is revalidated every frame")
    p.add_argument("--agent-lane-stable-frames", type=int, default=5,
                   help="Consecutive target-lane frames required before lane-change completion")
    p.add_argument("--agent-lane-center-tolerance-m", type=float, default=0.60,
                   help="Maximum target-lane lateral error used for stable completion")
    p.add_argument("--agent-lane-heading-tolerance-rad", type=float, default=0.20,
                   help="Maximum absolute heading error used for stable completion")
    p.add_argument("--agent-post-lane-change-settle-s", type=float, default=2.0,
                   help="Bounded target-lane centering time before control returns to the baseline")
    p.add_argument("--agent-cross-lane-max-steer", type=float, default=0.72,
                   help="Maximum normalized steering while the ego is still outside the target lane")
    p.add_argument("--agent-cross-lane-min-steer", type=float, default=0.42,
                   help="Minimum normalized steering magnitude maintained until the target lane ID is reached")
    p.add_argument("--agent-target-corridor-half-width-m", type=float, default=1.60,
                   help="Half-width of the BEVFusion target-lane safety corridor")
    p.add_argument("--agent-target-rear-clearance-m", type=float, default=8.0,
                   help="Required rear clearance in the target lane during an Agent lane change")
    p.add_argument("--agent-emergency-ttc-floor-s", type=float, default=0.75,
                   help="Hard global TTC floor that target-lane clearance cannot override")
    p.add_argument("--record-mp4", action="store_true",
                   help="Record the Stage10 live CARLA run to an MP4 attached to the ego vehicle")
    p.add_argument("--recording-path", default=None,
                   help="Optional MP4 output path. Defaults to <log-dir>/video/stage10_live.mp4")
    p.add_argument("--recording-camera-mode", choices=("chase", "hood", "topdown", "topdown_wide"), default="chase")
    p.add_argument("--recording-width", type=int, default=1280)
    p.add_argument("--recording-height", type=int, default=720)
    p.add_argument("--recording-fov", type=float, default=100.0)
    p.add_argument("--recording-fps", type=float, default=0.0,
                   help="Recording FPS. Defaults to 1 / --delta-t when <= 0")
    p.add_argument("--recording-overlay", dest="recording_overlay", action="store_true")
    p.add_argument("--recording-no-overlay", dest="recording_overlay", action="store_false")
    p.set_defaults(recording_overlay=True)
    return p.parse_args()


# ── Watch mode (Folder Polling) ──────────────────────────────────────────────

class FolderWatcherSync:
    """
    Mirrors CarlaSensorSync interface but reads from a folder on disk.
    Polls for 'sample_complete.json' or 'meta.json' before processing.
    """
    def __init__(
        self,
        samples_root: Path,
        max_samples: int,
        poll_interval_s: float = 0.5,
    ) -> None:
        self._root = Path(samples_root)
        self._max = max_samples
        self._poll_s = poll_interval_s
        self._processed: Set[str] = set()
        self._frame_counter = 0

    def start(self) -> None:
        if not self._root.exists():
            raise FileNotFoundError(f"Watch root does not exist: {self._root}")
        LOGGER.info("FolderWatcherSync started on %s", self._root)

    def stop(self) -> None:
        LOGGER.info("FolderWatcherSync stopped. Processed %d samples", self._frame_counter)

    def tick(self) -> Optional[LiveFrame]:
        """Poll for the next newest ready sample."""
        from carla_bevfusion_stage1.carla_sensor_sync import (
            LiveFrame, LiveCalibration, _camera_image_to_bgra, _lidar_to_xyzi, _radar_to_array
        )
        from carla_bevfusion_stage1.constants import LIDAR_SENSOR_NAME, MODEL_CAMERA_ORDER, RADAR_SENSOR_ORDER
        from PIL import Image

        while self._frame_counter < self._max:
            # 1. Find all 'sample_xxxxxx' dirs
            candidates = sorted(
                [d for d in self._root.iterdir() if d.is_dir() and d.name.startswith("sample_")],
                key=lambda x: int(x.name.split("_")[-1])
            )
            # 2. Filter new and ready
            new_ready = [d for d in candidates if d.name not in self._processed and (d / "meta.json").exists()]

            if not new_ready:
                time.sleep(self._poll_s)
                continue

            target = new_ready[0]
            self._processed.add(target.name)
            self._frame_counter += 1

            # 3. Load meta
            with (target / "meta.json").open("r") as f:
                meta = json.load(f)

            # 4. Reconstruct LiveFrame
            images_bgra = {}
            for cam in MODEL_CAMERA_ORDER:
                img = Image.open(target / "images" / f"{cam}.png")
                # Convert PIL to BGRA array
                rgb = np.array(img.convert("RGB"))
                bgra = np.zeros((img.height, img.width, 4), dtype=np.uint8)
                bgra[:, :, :3] = rgb[:, :, ::-1] # RGB -> BGR
                bgra[:, :, 3] = 255
                images_bgra[cam] = bgra

            lidar_xyzi = np.load(target / "lidar" / f"{LIDAR_SENSOR_NAME}.npy")

            radar_raw = {}
            for r in RADAR_SENSOR_ORDER:
                radar_path = target / "radar" / f"{r}.npy"
                radar_raw[r] = np.load(radar_path) if radar_path.exists() else np.zeros((0,4), dtype=np.float32)

            # Reconstruct Calibration from meta
            # (Simplification: meta stores matrix_world_from_sensor_bevfusion etc)
            from carla_bevfusion_stage1.coordinate_utils import ensure_numpy_matrix
            s_meta = meta["sensors"]
            calib = LiveCalibration(
                world_from_lidar_bev=ensure_numpy_matrix(s_meta[LIDAR_SENSOR_NAME]["matrix_world_from_sensor_bevfusion"]),
                ego_from_lidar_bev=ensure_numpy_matrix(s_meta[LIDAR_SENSOR_NAME]["matrix_ego_from_sensor_bevfusion"]),
                world_from_cameras_bev={c: ensure_numpy_matrix(s_meta[c]["matrix_world_from_sensor_bevfusion"]) for c in MODEL_CAMERA_ORDER},
                ego_from_cameras_bev={c: ensure_numpy_matrix(s_meta[c]["matrix_ego_from_sensor_bevfusion"]) for c in MODEL_CAMERA_ORDER},
                camera_intrinsics={c: np.array(s_meta[c]["camera_intrinsics"], dtype=np.float32) for c in MODEL_CAMERA_ORDER},
                world_from_radars_bev={
                    r: ensure_numpy_matrix(s_meta[r]["matrix_world_from_sensor_bevfusion"])
                    for r in RADAR_SENSOR_ORDER if r in s_meta
                },
                ego_from_radars_bev={
                    r: ensure_numpy_matrix(s_meta[r]["matrix_ego_from_sensor_bevfusion"])
                    for r in RADAR_SENSOR_ORDER if r in s_meta
                },
            )

            ego_vel = np.array(meta["ego"]["velocity_carla"], dtype=np.float32)

            return LiveFrame(
                frame_id=int(meta["frame_id"]),
                timestamp_s=float(meta["timestamp"]),
                images_bgra=images_bgra,
                lidar_xyzi=lidar_xyzi,
                radar_raw=radar_raw,
                calibration=calib,
                ego_velocity_carla=ego_vel,
            )
        return None



# ── CARLA helpers ─────────────────────────────────────────────────────────────

def _connect_carla(host: str, port: int, timeout_s: float = 30.0):
    """Connect to a running CARLA instance."""
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API not found. Add CARLA PythonAPI to PYTHONPATH."
        ) from exc
    client = carla.Client(host, port)
    client.set_timeout(timeout_s)
    LOGGER.info("Connecting to CARLA @ %s:%d …", host, port)
    world = client.get_world()
    LOGGER.info("Connected. Map: %s", world.get_map().name)
    return client, world


def _normalize_map_name(name: str) -> str:
    normalized = str(name).replace("\\", "/")
    return normalized.split("/")[-1]


def _load_scenario_manifest(manifest_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not manifest_path:
        return None
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Scenario manifest must contain a JSON object: {path}")
    return manifest


def _mp4_recording_disabled() -> bool:
    return str(os.getenv("AGENTAI_DISABLE_MP4", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _stage10_overlay_lines(
    *,
    frame_idx: int,
    live_frame: Any,
    ego_tel: Any,
    route_info: Dict[str, Any],
    det_count: int,
    control_source: str,
    agent_mode: str,
    agent_control_mode: str,
    min_ttc_s: float,
    lane_change_rule: str = "",
) -> List[str]:
    route_pct = route_info.get("route_completion_pct")
    route_label = "n/a" if route_pct is None else f"{float(route_pct):.1f}%"
    return [
        (
            f"stage10 tick={frame_idx} carla_frame={live_frame.frame_id} "
            f"agent={agent_mode}/{agent_control_mode}"
        ),
        (
            f"speed={float(ego_tel.ego_v_mps):.2f} mps dets={int(det_count)} "
            f"ttc={float(min_ttc_s):.1f}s route={route_label}"
        ),
        f"lane={ego_tel.ego_lane_id} change={lane_change_rule or 'n/a'} control={control_source}",
    ]


def _resolve_target_map(args: argparse.Namespace, scenario_manifest: Optional[Dict[str, Any]]) -> str:
    if scenario_manifest and str(args.map) == "Town01":
        manifest_town = scenario_manifest.get("town")
        if manifest_town:
            return _normalize_map_name(str(manifest_town))
    return _normalize_map_name(str(args.map))


def _agent_preferred_lane_from_manifest(scenario_manifest: Optional[Dict[str, Any]]) -> str:
    if not scenario_manifest:
        return "current"

    side = str(scenario_manifest.get("adjacent_side") or "").lower()
    if side not in {"left", "right"}:
        return "current"

    corridor = scenario_manifest.get("corridor") or {}
    if corridor.get("adjacent_lane_change_allowed") is False:
        return "current"

    placements = scenario_manifest.get("placements") or {}
    try:
        blocker_distance_m = float(placements.get("blocker_distance_m", 999.0))
        adjacent_distance_m = float(placements.get("adjacent_distance_m", 0.0))
        adjacent_front_distance_m = float(placements.get("adjacent_front_distance_m", adjacent_distance_m))
        adjacent_rear_distance_m = float(placements.get("adjacent_rear_distance_m", 999.0))
    except (TypeError, ValueError):
        return "current"

    front_clear = max(adjacent_distance_m, adjacent_front_distance_m)
    rear_clear = adjacent_rear_distance_m <= 0.0 or adjacent_rear_distance_m >= 15.0
    if blocker_distance_m <= 25.0 and front_clear >= 25.0 and rear_clear:
        return side
    return "current"


def _agent_target_lane_id_from_manifest(scenario_manifest: Optional[Dict[str, Any]]) -> str:
    if not scenario_manifest:
        return ""
    corridor = scenario_manifest.get("corridor") or {}
    lane_id = corridor.get("adjacent_lane_id")
    return "" if lane_id is None else str(lane_id)


def _agent_origin_lane_id_from_manifest(scenario_manifest: Optional[Dict[str, Any]]) -> str:
    if not scenario_manifest:
        return ""
    corridor = scenario_manifest.get("corridor") or {}
    lane_id = corridor.get("lane_id")
    return "" if lane_id is None else str(lane_id)


def _moving_adjacent_npcs_enabled(scenario_manifest: Optional[Dict[str, Any]]) -> bool:
    if not scenario_manifest:
        return False
    placements = scenario_manifest.get("placements") or {}
    return bool(placements.get("moving_adjacent_npcs", False))


def _scenario_adjacent_actor_ids(scenario_manifest: Optional[Dict[str, Any]]) -> List[int]:
    if not scenario_manifest:
        return []
    actor_ids: List[int] = []
    for key in ("adjacent_actor_id", "adjacent_front_actor_id", "adjacent_rear_actor_id"):
        actor_id = int(scenario_manifest.get(key, 0) or 0)
        if actor_id > 0 and actor_id not in actor_ids:
            actor_ids.append(actor_id)
    for actor_id in scenario_manifest.get("adjacent_actor_ids", []) or []:
        actor_id = int(actor_id or 0)
        if actor_id > 0 and actor_id not in actor_ids:
            actor_ids.append(actor_id)
    return actor_ids


def _adjacent_desired_speed_kmh(speed_diff_percent: float) -> float:
    faster_pct = max(0.0, -float(speed_diff_percent))
    return max(80.0, min(110.0, 70.0 + faster_pct * 0.25))


def _activate_moving_adjacent_npcs(
    *,
    world: Any,
    traffic_manager: Any,
    tm_port: int,
    scenario_manifest: Optional[Dict[str, Any]],
) -> None:
    if world is None or traffic_manager is None or not scenario_manifest:
        return
    if not _moving_adjacent_npcs_enabled(scenario_manifest):
        return

    try:
        import carla  # type: ignore
    except ImportError:
        return

    placements = scenario_manifest.get("placements") or {}
    speed_diff_percent = float(placements.get("adjacent_speed_diff_percent", -80.0))
    desired_speed_kmh = _adjacent_desired_speed_kmh(speed_diff_percent)
    actor_ids = _scenario_adjacent_actor_ids(scenario_manifest)
    activated: List[int] = []
    for actor_id in actor_ids:
        actor = world.get_actor(int(actor_id))
        if actor is None:
            LOGGER.warning("Moving adjacent NPC id=%s missing; cannot enable autopilot.", actor_id)
            continue
        try:
            actor.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False))
        except RuntimeError:
            LOGGER.warning("Failed to release adjacent NPC brake id=%s", actor_id)
        try:
            actor.set_autopilot(True, int(tm_port))
        except RuntimeError:
            LOGGER.warning("Failed to enable adjacent NPC autopilot id=%s tm_port=%s", actor_id, tm_port)
            continue
        for action_name, value in (
            ("auto_lane_change", False),
            ("vehicle_percentage_speed_difference", speed_diff_percent),
            ("ignore_lights_percentage", 100.0),
            ("ignore_signs_percentage", 100.0),
            ("distance_to_leading_vehicle", 3.0),
        ):
            try:
                getattr(traffic_manager, action_name)(actor, value)
            except (AttributeError, RuntimeError):
                LOGGER.debug("Traffic Manager %s unavailable for adjacent NPC id=%s", action_name, actor_id)
        try:
            traffic_manager.set_desired_speed(actor, desired_speed_kmh)
        except (AttributeError, RuntimeError):
            LOGGER.debug("Traffic Manager set_desired_speed unavailable for adjacent NPC id=%s", actor_id)
        activated.append(int(actor_id))
    if activated:
        LOGGER.info(
            "Stage10 activated moving adjacent NPCs ids=%s speed_diff=%.1f%% desired_speed=%.1f km/h",
            activated,
            speed_diff_percent,
            desired_speed_kmh,
        )


def _carla_enum_name(value: Any) -> str:
    return str(value).split(".")[-1]


def _waypoint_forward_xy(waypoint: Any) -> Tuple[float, float]:
    vector = waypoint.transform.get_forward_vector()
    norm = float(np.hypot(float(vector.x), float(vector.y)))
    if norm <= 1e-6:
        return (1.0, 0.0)
    return (float(vector.x) / norm, float(vector.y) / norm)


def _same_direction_waypoints(reference: Any, candidate: Any) -> bool:
    ref_x, ref_y = _waypoint_forward_xy(reference)
    cand_x, cand_y = _waypoint_forward_xy(candidate)
    return bool(ref_x * cand_x + ref_y * cand_y > 0.5)


def _lane_marking_type_name(waypoint: Any, side: str) -> str:
    marking = waypoint.left_lane_marking if side == "left" else waypoint.right_lane_marking
    return _carla_enum_name(getattr(marking, "type", "unknown"))


def _lane_change_enum_allows(waypoint: Any, side: str) -> bool:
    lane_change = _carla_enum_name(getattr(waypoint, "lane_change", "unknown")).lower()
    return bool("both" in lane_change or side.lower() in lane_change)


def _lane_marking_allows_lane_change(waypoint: Any, side: str) -> bool:
    marking_type = _lane_marking_type_name(waypoint, side).lower()
    return bool("broken" in marking_type and "solid" not in marking_type)


def _current_lane_change_permission(
    carla_map: Any,
    ego: Any,
    side: str,
    *,
    target_lane_id: str = "",
) -> Tuple[bool, str]:
    if side not in {"left", "right"}:
        return False, "current:no_requested_lane"
    if carla_map is None or ego is None:
        return False, f"{side}:map_or_ego_unavailable"

    try:
        waypoint = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
    except RuntimeError as exc:
        return False, f"{side}:waypoint_error:{exc}"
    if waypoint is None:
        return False, f"{side}:waypoint_unavailable"
    if bool(getattr(waypoint, "is_junction", False)):
        return False, f"{side}:junction"
    current_lane_id = str(getattr(waypoint, "lane_id", "") or "")
    if target_lane_id and current_lane_id == str(target_lane_id):
        return True, f"{side}:target_lane_reached"

    adjacent = waypoint.get_left_lane() if side == "left" else waypoint.get_right_lane()
    if adjacent is None:
        return False, f"{side}:no_adjacent_lane"
    if "driving" not in _carla_enum_name(getattr(adjacent, "lane_type", "unknown")).lower():
        return False, f"{side}:adjacent_not_driving"
    if not _same_direction_waypoints(waypoint, adjacent):
        return False, f"{side}:adjacent_opposite_direction"

    marking = _lane_marking_type_name(waypoint, side)
    lane_change = _carla_enum_name(getattr(waypoint, "lane_change", "unknown"))
    enum_allows = _lane_change_enum_allows(waypoint, side)
    marking_allows = _lane_marking_allows_lane_change(waypoint, side)
    allowed = bool(enum_allows and marking_allows)
    status = "ok" if allowed else "blocked"
    return allowed, f"{side}:{marking}:lane_change={lane_change}:{status}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _lane_id_int(lane_id: Any) -> Optional[int]:
    try:
        return int(str(lane_id))
    except (TypeError, ValueError):
        return None


def _lane_passed_target(current_lane_id: Any, origin_lane_id: Any, target_lane_id: Any) -> bool:
    current = _lane_id_int(current_lane_id)
    origin = _lane_id_int(origin_lane_id)
    target = _lane_id_int(target_lane_id)
    if current is None or origin is None or target is None:
        return False
    if current in {origin, target} or origin == target:
        return False
    return bool((target - origin) * (current - target) > 0)


def _wrap_degrees(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _is_driving_waypoint(waypoint: Any) -> bool:
    return bool("driving" in _carla_enum_name(getattr(waypoint, "lane_type", "unknown")).lower())


def _find_target_lane_waypoint(carla_map: Any, ego: Any, target_lane_id: str) -> Any:
    if carla_map is None or ego is None:
        return None
    try:
        start = carla_map.get_waypoint(ego.get_location(), project_to_road=True)
    except RuntimeError:
        return None
    if start is None:
        return None
    if not target_lane_id or str(getattr(start, "lane_id", "")) == str(target_lane_id):
        return start

    visited: Set[Tuple[int, int, int]] = set()
    frontier: List[Any] = [start]
    for _ in range(4):
        next_frontier: List[Any] = []
        for waypoint in frontier:
            key = (
                int(getattr(waypoint, "road_id", 0)),
                int(getattr(waypoint, "section_id", 0)),
                int(getattr(waypoint, "lane_id", 0)),
            )
            if key in visited:
                continue
            visited.add(key)
            if str(getattr(waypoint, "lane_id", "")) == str(target_lane_id):
                return waypoint
            for adjacent in (waypoint.get_left_lane(), waypoint.get_right_lane()):
                if adjacent is None or not _is_driving_waypoint(adjacent):
                    continue
                if not _same_direction_waypoints(start, adjacent):
                    continue
                if str(getattr(adjacent, "lane_id", "")) == str(target_lane_id):
                    return adjacent
                next_frontier.append(adjacent)
        frontier = next_frontier
    return None


def _target_lane_lateral_offset_bev_m(
    *,
    carla_map: Any,
    ego: Any,
    target_lane_id: str,
) -> Optional[float]:
    """Return target-lane centre offset in BEVFusion ego coordinates.

    CARLA uses local +y to the right while the canonical BEVFusion frame uses
    +y to the left, hence the final sign inversion.
    """
    target_wp = _find_target_lane_waypoint(carla_map, ego, target_lane_id)
    if target_wp is None or ego is None:
        return None
    try:
        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        target_loc = target_wp.transform.location
        yaw_rad = math.radians(float(ego_tf.rotation.yaw))
        dx = float(target_loc.x - ego_loc.x)
        dy = float(target_loc.y - ego_loc.y)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    local_y_carla = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
    return -float(local_y_carla)


def _target_lane_corridor_risk(
    *,
    detections: List[Any],
    ego_v_mps: float,
    lateral_center_m: Optional[float],
    corridor_half_width_m: float,
    rear_clearance_m: float,
    ttc_threshold_s: float,
) -> Dict[str, Any]:
    """Estimate risk only inside the intended target-lane corridor.

    The global TTC remains available for conservative baseline behavior.  This
    projection prevents a stationary blocker in the *origin* lane from being
    misclassified as a target-lane hazard while still rejecting close forward
    and rear objects detected by BEVFusion in the lane being entered.
    """
    if lateral_center_m is None or not math.isfinite(float(lateral_center_m)):
        return {
            "available": False,
            "clear": False,
            "forward_ttc_s": None,
            "rear_clearance_m": None,
            "object_count": 0,
            "lateral_center_m": None,
            "source": "bevfusion_target_corridor_v1",
        }

    center_m = float(lateral_center_m)
    half_width_m = max(0.5, float(corridor_half_width_m))
    required_rear_m = max(0.0, float(rear_clearance_m))
    min_forward_ttc_s = 99.0
    min_rear_clearance_m = 99.0
    object_count = 0

    for detection in detections or []:
        try:
            x_m = float(getattr(detection, "x"))
            y_m = float(getattr(detection, "y"))
            box_width_m = abs(float(getattr(detection, "dy", 0.0)))
            box_length_m = abs(float(getattr(detection, "dx", 0.0)))
        except (AttributeError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (x_m, y_m, box_width_m, box_length_m)):
            continue

        # LiDARInstance3DBoxes stores dimensions, not radii. Bound the padding
        # so a malformed or unusually large box cannot clear/block every lane.
        lateral_padding_m = _clamp(0.5 * box_width_m, 0.35, 1.50)
        if abs(y_m - center_m) > half_width_m + lateral_padding_m:
            continue
        object_count += 1

        longitudinal_padding_m = _clamp(0.5 * box_length_m, 0.50, 3.00)
        if x_m >= 0.0:
            forward_gap_m = max(0.1, x_m - longitudinal_padding_m - 1.0)
            ttc_s = forward_gap_m / max(0.5, float(ego_v_mps))
            min_forward_ttc_s = min(min_forward_ttc_s, ttc_s)
        else:
            rear_gap_m = max(0.0, abs(x_m) - longitudinal_padding_m)
            min_rear_clearance_m = min(min_rear_clearance_m, rear_gap_m)

    clear = bool(
        min_forward_ttc_s >= max(0.0, float(ttc_threshold_s))
        and min_rear_clearance_m >= required_rear_m
    )
    return {
        "available": True,
        "clear": clear,
        "forward_ttc_s": round(min_forward_ttc_s, 6),
        "rear_clearance_m": round(min_rear_clearance_m, 6),
        "object_count": int(object_count),
        "lateral_center_m": round(center_m, 6),
        "source": "bevfusion_target_corridor_v1",
    }


def _finite_difference_longitudinal_kinematics(
    *,
    speed_mps: float,
    timestamp_s: float,
    previous_speed_mps: Optional[float],
    previous_timestamp_s: Optional[float],
    previous_acceleration_mps2: Optional[float],
) -> tuple[float, Optional[float]]:
    """Estimate scalar longitudinal acceleration and jerk from ego speed."""
    if previous_speed_mps is None or previous_timestamp_s is None:
        return 0.0, None
    dt_s = max(float(timestamp_s) - float(previous_timestamp_s), 1e-6)
    acceleration_mps2 = (float(speed_mps) - float(previous_speed_mps)) / dt_s
    jerk_mps3 = (
        None
        if previous_acceleration_mps2 is None
        else (acceleration_mps2 - float(previous_acceleration_mps2)) / dt_s
    )
    return acceleration_mps2, jerk_mps3


def _lane_change_ttc_safety(
    *,
    world_state: Any,
    global_min_ttc_s: float,
    threshold_s: float,
    emergency_floor_s: float = 0.75,
) -> Tuple[bool, str]:
    """Reconcile global TTC with the target-lane BEVFusion corridor.

    Low global TTC may be caused by the blocker the maneuver is intended to
    avoid.  It is safe to continue only when a current, map-aligned BEVFusion
    corridor assessment explicitly marks the target lane clear. Missing target
    corridor evidence remains fail-safe.
    """
    global_ttc_s = float(global_min_ttc_s)
    if global_ttc_s < max(0.0, float(emergency_floor_s)):
        return False, "emergency_global_ttc"
    if global_ttc_s >= max(0.0, float(threshold_s)):
        return True, "global_ttc_clear"
    if not bool(getattr(world_state, "target_lane_risk_available", False)):
        return False, "low_ttc_target_corridor_unavailable"
    if bool(getattr(world_state, "target_lane_corridor_clear", False)):
        return True, "global_low_ttc_target_corridor_clear"
    return False, "low_ttc_target_corridor_unsafe"


def _lane_center_longitudinal_control(
    *,
    current_speed_mps: float,
    requested_speed_mps: float,
    lateral_distance_m: float,
) -> tuple[float, float]:
    """Return mutually exclusive throttle/brake for bounded lane centering."""
    target_speed = float(requested_speed_mps)
    lateral_abs = abs(float(lateral_distance_m))
    if lateral_abs > 2.2:
        target_speed = min(target_speed, 2.0)
    elif lateral_abs > 1.2:
        target_speed = min(target_speed, 2.5)

    speed_error = target_speed - float(current_speed_mps)
    throttle = _clamp(0.20 * speed_error, 0.0, 0.42)
    brake = _clamp(-0.35 * speed_error, 0.0, 0.65)
    if lateral_abs > 2.8:
        throttle = min(throttle, 0.28)
    return throttle, brake


def _lane_center_steering_control(
    *,
    local_x_m: float,
    local_y_m: float,
    heading_error_rad: float,
    max_steer: float,
    cross_lane_max_steer: Optional[float] = None,
    cross_lane_min_steer: float = 0.42,
    lane_change_assist: bool,
    target_lane_reached: bool,
) -> tuple[float, str]:
    """Return bounded steering for either crossing or settling into a lane.

    The lane-change phase has a shorter look-ahead and a small directional
    floor until the HD-map lane ID confirms entry into the target lane.  This
    avoids a false "settle" transition caused solely by ego yaw changing the
    relative look-ahead geometry before the vehicle crosses the lane boundary;
    it never bypasses the live TTC, map-permission, or emergency safety gates
    checked by the caller.
    """
    settle_max_steer_abs = max(0.0, float(max_steer))
    cross_max_steer_abs = (
        settle_max_steer_abs
        if cross_lane_max_steer is None
        else max(0.0, float(cross_lane_max_steer))
    )
    local_x = max(float(local_x_m), 1.0)
    local_y = float(local_y_m)
    if lane_change_assist and not target_lane_reached:
        # While the ego is still in the origin lane, lateral displacement is
        # the primary objective.  A large heading correction here can cancel
        # the turn toward an adjacent-lane centre (especially near a blocker)
        # and leave the vehicle driving forward in the blocked lane.  The
        # target point is already expressed in ego coordinates, so pure
        # pursuit incorporates the useful part of heading alignment.  Retain
        # only a small explicit heading term until the HD-map lane transition
        # has physically occurred.
        pure_pursuit = math.atan2(4.2 * local_y, max(local_x * local_x, 1.0))
        steer = 0.25 * float(heading_error_rad) + 1.85 * pure_pursuit
        phase = "cross_lane"
        crossing_floor = min(
            cross_max_steer_abs,
            max(0.0, float(cross_lane_min_steer)),
        )
        if abs(steer) < crossing_floor and abs(local_y) > 1e-3:
            steer = math.copysign(crossing_floor, local_y)
        steering_limit = cross_max_steer_abs
    elif lane_change_assist:
        pure_pursuit = math.atan2(2.7 * local_y, max(local_x * local_x, 1.0))
        steer = 0.75 * float(heading_error_rad) + 1.25 * pure_pursuit
        phase = "settle_target_lane"
        steering_limit = settle_max_steer_abs
    else:
        pure_pursuit = math.atan2(2.7 * local_y, max(local_x * local_x, 1.0))
        steer = 0.75 * float(heading_error_rad) + 1.25 * pure_pursuit
        phase = "lane_center"
        steering_limit = settle_max_steer_abs
    return _clamp(steer, -steering_limit, steering_limit), phase


def _map_lane_centering_control(
    *,
    carla_map: Any,
    ego: Any,
    target_lane_id: str,
    target_speed_mps: float,
    source: str,
    max_steer: float = 0.42,
    cross_lane_max_steer: Optional[float] = None,
    cross_lane_min_steer: float = 0.42,
    lane_change_assist: bool = False,
) -> Any:
    from stage9.schemas import ActuatorCommand

    target_wp = _find_target_lane_waypoint(carla_map, ego, target_lane_id)
    if target_wp is None:
        return None

    velocity = ego.get_velocity()
    current_speed = float(math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2))
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    try:
        current_wp = carla_map.get_waypoint(ego_loc, project_to_road=True)
        target_lane_reached = bool(
            current_wp is not None
            and str(getattr(current_wp, "lane_id", "")) == str(target_lane_id)
        )
    except (AttributeError, RuntimeError):
        target_lane_reached = False

    if lane_change_assist:
        lookahead_m = _clamp(3.0 + current_speed * 0.55, 3.0, 5.5)
    else:
        lookahead_m = _clamp(4.0 + current_speed * 0.7, 4.0, 8.0)
    next_candidates = [
        wp for wp in target_wp.next(float(lookahead_m))
        if str(getattr(wp, "lane_id", "")) == str(getattr(target_wp, "lane_id", ""))
    ]
    aim_wp = next_candidates[0] if next_candidates else target_wp
    aim_loc = aim_wp.transform.location

    yaw_rad = math.radians(float(ego_tf.rotation.yaw))
    dx = float(aim_loc.x - ego_loc.x)
    dy = float(aim_loc.y - ego_loc.y)
    local_x = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
    local_y = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
    heading_error_rad = math.radians(_wrap_degrees(float(aim_wp.transform.rotation.yaw) - float(ego_tf.rotation.yaw)))
    steer, steering_phase = _lane_center_steering_control(
        local_x_m=local_x,
        local_y_m=local_y,
        heading_error_rad=heading_error_rad,
        max_steer=max_steer,
        cross_lane_max_steer=cross_lane_max_steer,
        cross_lane_min_steer=cross_lane_min_steer,
        lane_change_assist=lane_change_assist,
        target_lane_reached=target_lane_reached,
    )

    lateral_abs = abs(float(local_y))
    # A target in the adjacent lane is normally 3-4 m lateral from ego.
    # Applying throttle and a forced brake at the same time prevented the
    # vehicle from generating enough forward motion for pure pursuit to cross
    # the lane boundary. The helper keeps speed bounded while ensuring the
    # longitudinal actuators remain mutually exclusive.
    throttle, brake = _lane_center_longitudinal_control(
        current_speed_mps=current_speed,
        requested_speed_mps=float(target_speed_mps),
        lateral_distance_m=lateral_abs,
    )

    command = ActuatorCommand(
        steer=steer,
        throttle=throttle,
        brake=brake,
        source=source,
    )
    # Dynamic telemetry is deliberately attached rather than added to the
    # stable Stage 9 command schema; it is emitted into the Stage 10 audit log.
    setattr(command, "target_lane_lateral_error_m", float(local_y))
    setattr(command, "target_lane_heading_error_rad", float(heading_error_rad))
    setattr(command, "steering_phase", steering_phase)
    setattr(command, "target_lane_reached", bool(target_lane_reached))
    return command


def _draw_scenario_actor_labels(world: Any, scenario_manifest: Optional[Dict[str, Any]], *, life_time_s: float) -> None:
    if world is None or not scenario_manifest:
        return
    try:
        import carla  # type: ignore
    except ImportError:
        return

    label_specs = [
        ("ego_actor_id", "EGO", carla.Color(40, 240, 80)),
        ("blocker_actor_id", "BLOCKER", carla.Color(255, 60, 60)),
        ("adjacent_front_actor_id", "FRONT", carla.Color(70, 160, 255)),
        ("adjacent_rear_actor_id", "REAR", carla.Color(255, 210, 40)),
    ]
    for key, label, color in label_specs:
        actor_id = int(scenario_manifest.get(key, 0) or 0)
        if actor_id <= 0:
            continue
        actor = world.get_actor(actor_id)
        if actor is None:
            continue
        try:
            location = actor.get_location()
            location.z += 2.2
            world.debug.draw_string(
                location,
                label,
                draw_shadow=True,
                color=color,
                life_time=max(0.05, float(life_time_s)),
                persistent_lines=False,
            )
        except RuntimeError:
            continue


def _resolve_attach_actor_id(
    args: argparse.Namespace,
    scenario_manifest: Optional[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[Path]]:
    if args.attach_to_actor_id is not None:
        return int(args.attach_to_actor_id), None
    if not scenario_manifest or not args.scenario_manifest:
        return None, None

    actor_id = scenario_manifest.get("ego_actor_id")
    manifest_path = Path(args.scenario_manifest)
    if not isinstance(actor_id, int) or actor_id <= 0:
        raise ValueError(f"Scenario manifest {manifest_path} does not contain a valid ego_actor_id.")
    LOGGER.info("Resolved ego actor id=%s from scenario manifest %s", actor_id, manifest_path)
    return int(actor_id), manifest_path


def _load_map_if_needed(client, world, target_map: str):
    import carla  # type: ignore
    current = world.get_map().name.split("/")[-1]
    if current != target_map:
        LOGGER.info("Loading map %s (was %s) …", target_map, current)
        world = client.load_world(target_map)
        time.sleep(2.0)
    return world


def _spawn_ego(world, spawn_idx: int = 0):
    import carla  # type: ignore
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in this map.")
    sp = spawn_points[spawn_idx % len(spawn_points)]
    ego = world.spawn_actor(vehicle_bp, sp)
    ego.set_autopilot(True)      # Simple traffic-manager autopilot as baseline
    LOGGER.info("Ego spawned at spawn_point=%d actor_id=%d", spawn_idx, ego.id)
    return ego


def _spawn_or_get_ego(
    world,
    args: argparse.Namespace,
    scenario_manifest: Optional[Dict[str, Any]],
) -> Tuple[object, bool, str]:
    attach_actor_id, manifest_path = _resolve_attach_actor_id(args, scenario_manifest)
    if attach_actor_id is not None:
        actor = world.get_actor(int(attach_actor_id))
        if actor is None:
            if manifest_path is not None:
                raise ValueError(
                    f"Actor id {attach_actor_id} from scenario manifest {manifest_path} does not exist in the current world."
                )
            raise ValueError(f"Actor id {attach_actor_id} does not exist in the current world.")
        if bool(args.attach_autopilot):
            actor.set_autopilot(True)
            LOGGER.info("Attached ego actor_id=%d with autopilot enabled", actor.id)
        else:
            LOGGER.info("Attached ego actor_id=%d", actor.id)
        return actor, False, "attached_actor"

    return _spawn_ego(world, args.spawn_point), True, "spawned_actor"


def _apply_actuator_command(ego_actor, cmd) -> None:
    """
    Apply a Stage 9 ActuatorCommand to the CARLA ego vehicle.

    TODO [Stage 9 hook]: Uncomment and implement when wiring up Arbiter.
    cmd fields: steer ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1]
    """
    try:
        import carla  # type: ignore
        control = carla.VehicleControl(
            throttle=float(getattr(cmd, "throttle", getattr(cmd, "throttle_norm", 0.0))),
            steer=float(getattr(cmd, "steer", getattr(cmd, "steer_norm", 0.0))),
            brake=float(getattr(cmd, "brake", getattr(cmd, "brake_norm", 0.0))),
        )
        ego_actor.apply_control(control)
    except Exception as exc:
        LOGGER.warning("apply_control failed: %s", exc)


# ── Driving metric helpers ────────────────────────────────────────────────────

def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(100.0, float(percentile))) / 100.0 * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    alpha = rank - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def _location_to_xyz(location) -> np.ndarray:
    return np.array([float(location.x), float(location.y), float(location.z)], dtype=np.float64)


def _ensure_carla_pythonapi_paths(carla_root: str | Path) -> None:
    root = Path(carla_root)
    candidates = [
        root / "PythonAPI",
        root / "PythonAPI" / "carla",
        root / "PythonAPI" / "carla" / "agents",
    ]
    egg_dir = root / "PythonAPI" / "carla" / "dist"
    if egg_dir.exists():
        candidates.extend(sorted(egg_dir.glob("carla-*.egg")))

    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)


def _collision_actor_key(event: Dict[str, Any]) -> tuple[Any, str]:
    """Return a stable actor key without trusting display-only actor names."""
    actor_id = event.get("other_actor_id")
    actor_type = str(event.get("other_actor_type") or "unknown")
    return actor_id, actor_type


def _group_collision_episodes(
    events: List[Dict[str, Any]],
    *,
    max_frame_gap: int = 5,
) -> List[Dict[str, Any]]:
    """Collapse consecutive sensor callbacks from one physical contact."""
    if max_frame_gap < 0:
        raise ValueError("max_frame_gap must be non-negative")
    ordered = sorted(
        events,
        key=lambda event: (
            int(event.get("frame_id", -1)),
            float(event.get("timestamp_s", 0.0)),
        ),
    )
    episodes: List[Dict[str, Any]] = []
    for event in ordered:
        frame_id = int(event.get("frame_id", -1))
        timestamp_s = float(event.get("timestamp_s", 0.0))
        intensity = float(event.get("intensity", 0.0))
        actor_id, actor_type = _collision_actor_key(event)
        previous = episodes[-1] if episodes else None
        same_contact = bool(
            previous is not None
            and previous["other_actor_id"] == actor_id
            and previous["other_actor_type"] == actor_type
            and frame_id - int(previous["end_frame_id"]) <= max_frame_gap
        )
        if not same_contact:
            episodes.append(
                {
                    "start_frame_id": frame_id,
                    "end_frame_id": frame_id,
                    "start_timestamp_s": timestamp_s,
                    "end_timestamp_s": timestamp_s,
                    "other_actor_id": actor_id,
                    "other_actor_type": actor_type,
                    "sensor_event_count": 1,
                    "peak_intensity": round(intensity, 6),
                    "total_intensity": round(intensity, 6),
                }
            )
            continue
        previous["end_frame_id"] = frame_id
        previous["end_timestamp_s"] = timestamp_s
        previous["sensor_event_count"] = int(previous["sensor_event_count"]) + 1
        previous["peak_intensity"] = round(
            max(float(previous["peak_intensity"]), intensity), 6
        )
        previous["total_intensity"] = round(
            float(previous["total_intensity"]) + intensity, 6
        )
    return episodes


def _classify_lane_crossing(event: Dict[str, Any]) -> str:
    """Classify CARLA lane markings conservatively at event level."""
    markings = list(event.get("crossed_lane_markings") or [])
    if not markings:
        return "unknown"
    types = [str(marking.get("type") or "unknown").lower() for marking in markings]
    permissions = [
        str(marking.get("lane_change") or "unknown").lower()
        for marking in markings
    ]
    if any("solid" in marking_type for marking_type in types):
        return "illegal"
    if any(permission in {"none", "lanechange.none"} for permission in permissions):
        return "illegal"
    if all("broken" in marking_type for marking_type in types) and all(
        permission not in {"unknown", ""} for permission in permissions
    ):
        return "legal"
    return "unknown"


def _lane_crossing_phase_summary(
    events: List[Dict[str, Any]],
    *,
    maneuver_start_timestamp_s: Optional[float],
    maneuver_completion_timestamp_s: Optional[float],
) -> Dict[str, Dict[str, int]]:
    """Split raw CARLA lane-crossing callbacks by maneuver lifecycle phase."""
    summary = {
        phase: {"event_count": 0, "legal": 0, "illegal": 0, "unknown": 0}
        for phase in ("before_maneuver", "during_maneuver", "after_maneuver")
    }
    for event in events:
        timestamp_s = float(event.get("timestamp_s", 0.0))
        if maneuver_start_timestamp_s is None or timestamp_s < maneuver_start_timestamp_s:
            phase = "before_maneuver"
        elif (
            maneuver_completion_timestamp_s is not None
            and timestamp_s > maneuver_completion_timestamp_s
        ):
            phase = "after_maneuver"
        else:
            phase = "during_maneuver"
        classification = str(event.get("classification") or "unknown")
        if classification not in {"legal", "illegal", "unknown"}:
            classification = "unknown"
        summary[phase]["event_count"] += 1
        summary[phase][classification] += 1
    return summary


class CollisionMonitor:
    """Attach a CARLA collision sensor and collect per-run collision events."""

    def __init__(self, world, ego_actor, *, impulse_threshold: float = 0.0) -> None:
        self._world = world
        self._ego = ego_actor
        self._threshold = max(0.0, float(impulse_threshold))
        self._sensor = None
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        try:
            bp = self._world.get_blueprint_library().find("sensor.other.collision")
            self._sensor = self._world.spawn_actor(bp, self._carla_transform(), attach_to=self._ego)
            self._sensor.listen(self._on_collision)
            LOGGER.info("CollisionMonitor attached to ego actor_id=%s", getattr(self._ego, "id", "unknown"))
        except Exception as exc:
            LOGGER.warning("CollisionMonitor unavailable: %s", exc)
            self._sensor = None

    def stop(self) -> None:
        if self._sensor is None:
            return
        try:
            self._sensor.stop()
            self._sensor.destroy()
        except Exception as exc:
            LOGGER.debug("CollisionMonitor stop failed: %s", exc)
        self._sensor = None

    def events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def counted_events(self) -> List[Dict[str, Any]]:
        return [ev for ev in self.events() if float(ev.get("intensity", 0.0)) >= self._threshold]

    def counted_episodes(self) -> List[Dict[str, Any]]:
        return _group_collision_episodes(self.counted_events())

    def _carla_transform(self):
        import carla  # type: ignore
        return carla.Transform()

    def _on_collision(self, event) -> None:
        impulse = getattr(event, "normal_impulse", None)
        ix = float(getattr(impulse, "x", 0.0)) if impulse is not None else 0.0
        iy = float(getattr(impulse, "y", 0.0)) if impulse is not None else 0.0
        iz = float(getattr(impulse, "z", 0.0)) if impulse is not None else 0.0
        intensity = float((ix * ix + iy * iy + iz * iz) ** 0.5)
        other = getattr(event, "other_actor", None)
        record = {
            "frame_id": int(getattr(event, "frame", -1)),
            "timestamp_s": float(getattr(event, "timestamp", 0.0)),
            "other_actor_id": int(getattr(other, "id", -1)) if other is not None else None,
            "other_actor_type": str(getattr(other, "type_id", "")) if other is not None else None,
            "normal_impulse": {"x": ix, "y": iy, "z": iz},
            "intensity": round(intensity, 6),
            "counted": intensity >= self._threshold,
        }
        with self._lock:
            self._events.append(record)


class LaneInvasionMonitor:
    """Attach CARLA's lane-invasion sensor and retain a per-run audit log."""

    def __init__(self, world, ego_actor) -> None:
        self._world = world
        self._ego = ego_actor
        self._sensor = None
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        try:
            import carla  # type: ignore

            bp = self._world.get_blueprint_library().find("sensor.other.lane_invasion")
            self._sensor = self._world.spawn_actor(bp, carla.Transform(), attach_to=self._ego)
            self._sensor.listen(self._on_invasion)
            LOGGER.info("LaneInvasionMonitor attached to ego actor_id=%s", getattr(self._ego, "id", "unknown"))
        except Exception as exc:
            LOGGER.warning("LaneInvasionMonitor unavailable: %s", exc)
            self._sensor = None

    def stop(self) -> None:
        if self._sensor is None:
            return
        try:
            self._sensor.stop()
            self._sensor.destroy()
        except Exception as exc:
            LOGGER.debug("LaneInvasionMonitor stop failed: %s", exc)
        self._sensor = None

    def events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def classified_events(self) -> List[Dict[str, Any]]:
        classified: List[Dict[str, Any]] = []
        for event in self.events():
            record = dict(event)
            record["classification"] = _classify_lane_crossing(record)
            classified.append(record)
        return classified

    def _on_invasion(self, event) -> None:
        markings = []
        for marking in list(getattr(event, "crossed_lane_markings", []) or []):
            markings.append(
                {
                    "type": str(getattr(marking, "type", "unknown")),
                    "color": str(getattr(marking, "color", "unknown")),
                    "lane_change": str(getattr(marking, "lane_change", "unknown")),
                }
            )
        record = {
            "frame_id": int(getattr(event, "frame", -1)),
            "timestamp_s": float(getattr(event, "timestamp", 0.0)),
            "crossed_lane_markings": markings,
        }
        record["classification"] = _classify_lane_crossing(record)
        with self._lock:
            self._events.append(record)


class RouteProgressTracker:
    """
    Tracks route completion for Stage 10.

    If --route-target-spawn-point is supplied and CARLA's GlobalRoutePlanner is
    available, progress follows the planned route. Otherwise it falls back to a
    distance-travelled mission proxy so nominal free-roam runs still emit RC%.
    """

    def __init__(
        self,
        *,
        mode: str,
        target_distance_m: float,
        route_points: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.mode = mode
        self.target_distance_m = max(1.0, float(target_distance_m))
        self.route_points = route_points or []
        self.route_cumulative_m = self._cumulative_distances(self.route_points)
        self.route_length_m = (
            float(self.route_cumulative_m[-1])
            if self.route_cumulative_m
            else self.target_distance_m
        )
        self._last_location: Optional[np.ndarray] = None
        self._distance_traveled_m = 0.0
        self._best_route_progress_m = 0.0

    @classmethod
    def build(
        cls,
        *,
        carla_map,
        spawn_point_index: int,
        target_spawn_point_index: Optional[int],
        fallback_distance_m: float,
        start_location=None,
    ) -> "RouteProgressTracker":
        if target_spawn_point_index is None:
            return cls(mode="distance_proxy", target_distance_m=fallback_distance_m)

        spawn_points = carla_map.get_spawn_points()
        if not spawn_points:
            return cls(mode="distance_proxy_no_spawn_points", target_distance_m=fallback_distance_m)

        start = start_location or spawn_points[int(spawn_point_index) % len(spawn_points)].location
        dest = spawn_points[int(target_spawn_point_index) % len(spawn_points)].location
        route_points = cls._trace_route_points(carla_map, start, dest)
        if len(route_points) >= 2:
            return cls(mode="carla_global_route", route_points=route_points, target_distance_m=fallback_distance_m)

        straight = [_location_to_xyz(start), _location_to_xyz(dest)]
        return cls(mode="straight_line_route", route_points=straight, target_distance_m=fallback_distance_m)

    def update(self, location_xyz: np.ndarray) -> Dict[str, Any]:
        loc = np.asarray(location_xyz, dtype=np.float64)
        if self._last_location is not None:
            step = float(np.linalg.norm(loc[:2] - self._last_location[:2]))
            if step >= 0.0:
                self._distance_traveled_m += step
        self._last_location = loc

        if self.route_points and self.route_cumulative_m:
            xy = np.array([p[:2] for p in self.route_points], dtype=np.float64)
            dists = np.linalg.norm(xy - loc[:2], axis=1)
            nearest_idx = int(np.argmin(dists))
            route_progress_m = max(self._best_route_progress_m, float(self.route_cumulative_m[nearest_idx]))
            self._best_route_progress_m = route_progress_m
            denominator = max(1.0, self.route_length_m)
        else:
            route_progress_m = self._distance_traveled_m
            denominator = self.target_distance_m

        completion = max(0.0, min(1.0, route_progress_m / denominator))
        return {
            "route_mode": self.mode,
            "route_progress_m": round(route_progress_m, 3),
            "route_length_m": round(denominator, 3),
            "route_completion_rate": round(completion, 6),
            "route_completion_pct": round(completion * 100.0, 3),
            "distance_traveled_m": round(self._distance_traveled_m, 3),
        }

    def final_summary(self) -> Dict[str, Any]:
        denominator = max(1.0, self.route_length_m if self.route_points else self.target_distance_m)
        progress = self._best_route_progress_m if self.route_points else self._distance_traveled_m
        completion = max(0.0, min(1.0, progress / denominator))
        return {
            "route_mode": self.mode,
            "route_progress_m": round(progress, 3),
            "route_length_m": round(denominator, 3),
            "route_completion_rate": round(completion, 6),
            "route_completion_pct": round(completion * 100.0, 3),
            "distance_traveled_m": round(self._distance_traveled_m, 3),
        }

    @staticmethod
    def _trace_route_points(carla_map, start_location, end_location) -> List[np.ndarray]:
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore

            planner = GlobalRoutePlanner(carla_map, 2.0)
            if hasattr(planner, "setup"):
                planner.setup()
            route = planner.trace_route(start_location, end_location)
            points: List[np.ndarray] = []
            for item in route:
                waypoint = item[0] if isinstance(item, tuple) else item
                points.append(_location_to_xyz(waypoint.transform.location))
            return points
        except Exception as exc:
            LOGGER.warning("GlobalRoutePlanner unavailable; falling back to straight-line RC: %s", exc)
            return []

    @staticmethod
    def _cumulative_distances(points: List[np.ndarray]) -> List[float]:
        if not points:
            return []
        cumulative = [0.0]
        for prev, curr in zip(points, points[1:]):
            cumulative.append(cumulative[-1] + float(np.linalg.norm(curr[:2] - prev[:2])))
        return cumulative


def _build_driving_metrics(
    *,
    args: argparse.Namespace,
    stats: Dict[str, Any],
    route_tracker: Optional[RouteProgressTracker],
    collision_monitor: Optional[CollisionMonitor],
    lane_invasion_monitor: Optional[LaneInvasionMonitor],
    maneuver_start_timestamp_s: Optional[float] = None,
    maneuver_completion_timestamp_s: Optional[float] = None,
) -> Dict[str, Any]:
    route_summary = (
        route_tracker.final_summary()
        if route_tracker is not None
        else {
            "route_mode": "unavailable_watch_mode",
            "route_progress_m": 0.0,
            "route_length_m": None,
            "route_completion_rate": None,
            "route_completion_pct": None,
            "distance_traveled_m": 0.0,
        }
    )
    collision_events = collision_monitor.counted_events() if collision_monitor is not None else []
    collision_episodes = (
        collision_monitor.counted_episodes() if collision_monitor is not None else []
    )
    lane_crossing_events = (
        lane_invasion_monitor.classified_events()
        if lane_invasion_monitor is not None else []
    )
    legal_lane_crossings = [
        event for event in lane_crossing_events if event.get("classification") == "legal"
    ]
    illegal_lane_invasions = [
        event for event in lane_crossing_events if event.get("classification") == "illegal"
    ]
    unknown_lane_crossings = [
        event for event in lane_crossing_events if event.get("classification") == "unknown"
    ]
    lane_crossing_by_phase = _lane_crossing_phase_summary(
        lane_crossing_events,
        maneuver_start_timestamp_s=maneuver_start_timestamp_s,
        maneuver_completion_timestamp_s=maneuver_completion_timestamp_s,
    )
    distance_km = float(route_summary.get("distance_traveled_m") or 0.0) / 1000.0
    collision_rate_per_km = (
        round(len(collision_episodes) / distance_km, 6)
        if distance_km > 1e-6
        else None
    )
    rc = route_summary.get("route_completion_rate")
    route_ok = bool(rc is not None and float(rc) >= float(args.success_rc_threshold))
    collision_ok = len(collision_episodes) == 0
    lane_rule_ok = len(illegal_lane_invasions) == 0
    runtime_ok = int(stats.get("errors", 0)) == 0 and int(stats.get("frames", 0)) > 0
    success = bool(route_ok and collision_ok and lane_rule_ok and runtime_ok)
    jerk_samples = [float(value) for value in stats.get("longitudinal_jerk_samples_mps3", [])]
    abs_jerk_samples = [abs(value) for value in jerk_samples]
    duration_s = float(stats.get("frames", 0)) * float(args.delta_t)
    offroad_frames = int(stats.get("offroad_frames", 0))
    max_abs_jerk = max(abs_jerk_samples) if abs_jerk_samples else None
    mean_abs_jerk = (
        sum(abs_jerk_samples) / len(abs_jerk_samples)
        if abs_jerk_samples else None
    )
    tick_latencies = [float(value) for value in stats.get("tick_latency_samples_ms", [])]
    lidar_points = [int(value) for value in stats.get("lidar_point_samples", [])]
    radar_points = [int(value) for value in stats.get("radar_point_samples", [])]
    step_budget_ms = max(float(args.delta_t) * 1000.0, 1e-6)

    return {
        "schema_version": "stage10_driving_metrics_v3",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "stage": "10_live_bridge",
        "map": args.map,
        "random_seed": int(args.seed),
        "agent_mode": args.agent_mode,
        "agent_control_mode": args.agent_control_mode,
        "frames": int(stats.get("frames", 0)),
        "route_completion_rate": rc,
        "route_completion_pct": route_summary.get("route_completion_pct"),
        # collision_count is the paper-facing physical-contact count. Raw
        # CARLA callbacks remain available separately for auditability.
        "collision_count": len(collision_episodes),
        "collision_episode_count": len(collision_episodes),
        "collision_sensor_event_count": len(collision_events),
        "collision_episodes": collision_episodes,
        "collision_rate_per_km": collision_rate_per_km,
        # Preserve the raw CARLA lane-invasion event count while exposing the
        # legality-aware safety metric separately. A valid lane change across
        # a broken line is not an illegal invasion.
        "lane_invasion_count": len(lane_crossing_events),
        "lane_crossing_event_count": len(lane_crossing_events),
        "legal_lane_crossing_count": len(legal_lane_crossings),
        "illegal_lane_invasion_count": len(illegal_lane_invasions),
        "unknown_lane_crossing_count": len(unknown_lane_crossings),
        "lane_crossing_by_maneuver_phase": lane_crossing_by_phase,
        "maneuver_illegal_lane_invasion_count": lane_crossing_by_phase[
            "during_maneuver"
        ]["illegal"],
        "post_maneuver_illegal_lane_invasion_count": lane_crossing_by_phase[
            "after_maneuver"
        ]["illegal"],
        "lane_invasion_rate_per_km": (
            round(len(illegal_lane_invasions) / distance_km, 6)
            if distance_km > 1e-6 else None
        ),
        "offroad_frames": offroad_frames,
        "offroad_rate": round(offroad_frames / max(int(stats.get("frames", 0)), 1), 6),
        "episode_duration_s": round(duration_s, 6),
        # Backward-compatible alias. Use episode_duration_s for the complete
        # run and lane_change_maneuver.completion_time_s for maneuver time.
        "maneuver_duration_s": round(duration_s, 6),
        "comfort": {
            "longitudinal_jerk_sample_count": len(jerk_samples),
            "mean_abs_longitudinal_jerk_mps3": (
                round(mean_abs_jerk, 6) if mean_abs_jerk is not None else None
            ),
            "max_abs_longitudinal_jerk_mps3": (
                round(max_abs_jerk, 6) if max_abs_jerk is not None else None
            ),
            "jerk_exceedance_rate_over_3_mps3": (
                round(sum(value > 3.0 for value in abs_jerk_samples) / len(abs_jerk_samples), 6)
                if abs_jerk_samples else None
            ),
        },
        "scenario_success": success,
        "scenario_success_rate": 1.0 if success else 0.0,
        "success_criteria": {
            "route_completion_rate_min": float(args.success_rc_threshold),
            "route_completion_passed": route_ok,
            "collision_count_max": 0,
            "collision_passed": collision_ok,
            "illegal_lane_invasion_count": len(illegal_lane_invasions),
            "illegal_lane_invasion_passed": lane_rule_ok,
            "runtime_errors_max": 0,
            "runtime_passed": runtime_ok,
        },
        "metric_definitions": {
            "collision_episode": {
                "same_actor_required": True,
                "maximum_inter_event_frame_gap": 5,
                "raw_count_field": "collision_sensor_event_count",
            },
            "lane_crossing": {
                "legal": "all markings are broken and lane-change permission is present",
                "illegal": "any marking is solid or explicitly disallows lane change",
                "unknown": "insufficient or unrecognized marking metadata",
                "phase_partition": (
                    "before the first applied lane-change command, during the accepted maneuver "
                    "through stable target-lane completion, or after physical completion"
                ),
            },
            "longitudinal_jerk": (
                "finite difference of scalar ego speed acceleration using CARLA timestamps"
            ),
            "duration": {
                "episode_duration_s": "all simulated frames multiplied by delta_t",
                "lane_change_completion_time_s": (
                    "first accepted lane-change intent to first target-lane observation"
                ),
            },
        },
        "route": route_summary,
        "runtime": {
            "errors": int(stats.get("errors", 0)),
            "avg_bev_inference_ms": round(
                float(stats.get("bev_ms_sum", 0.0)) / max(1, int(stats.get("frames", 0))),
                3,
            ) if int(stats.get("frames", 0)) else None,
            "avg_detections_per_frame": round(
                float(stats.get("total_det", 0)) / max(1, int(stats.get("frames", 0))),
                3,
            ) if int(stats.get("frames", 0)) else None,
            "sensor_input": {
                "mean_lidar_points": (
                    round(sum(lidar_points) / len(lidar_points), 3)
                    if lidar_points else None
                ),
                "mean_radar_points": (
                    round(sum(radar_points) / len(radar_points), 3)
                    if radar_points else None
                ),
                "nonempty_radar_frame_rate": (
                    round(sum(value > 0 for value in radar_points) / len(radar_points), 6)
                    if radar_points else None
                ),
                "zero_radar_frame_count": sum(value == 0 for value in radar_points),
            },
            "control_loop_latency": {
                "sample_count": len(tick_latencies),
                "mean_ms": (
                    round(sum(tick_latencies) / len(tick_latencies), 3)
                    if tick_latencies else None
                ),
                "p50_ms": (
                    round(float(_percentile(tick_latencies, 50.0)), 3)
                    if tick_latencies else None
                ),
                "p95_ms": (
                    round(float(_percentile(tick_latencies, 95.0)), 3)
                    if tick_latencies else None
                ),
                "max_ms": round(max(tick_latencies), 3) if tick_latencies else None,
                "simulation_step_budget_ms": round(step_budget_ms, 3),
                "over_step_budget_rate": (
                    round(
                        sum(value > step_budget_ms for value in tick_latencies)
                        / len(tick_latencies),
                        4,
                    )
                    if tick_latencies else None
                ),
            },
        },
        "artifacts": {
            "ego_trace": str(Path(args.log_dir) / "ego_trace.jsonl"),
            "collision_events": str(Path(args.log_dir) / "collision_events.jsonl"),
            "collision_episodes": str(Path(args.log_dir) / "collision_episodes.jsonl"),
            "lane_invasion_events": str(Path(args.log_dir) / "lane_invasion_events.jsonl"),
        },
    }


def _should_query_agent(
    *,
    args: argparse.Namespace,
    frame_idx: int,
    baseline_intent: str,
    min_ttc_s: float,
    world_state: Any,
) -> tuple[bool, str]:
    if args.agent_trigger_mode == "every_frame":
        return True, "every_frame"

    if float(min_ttc_s) < float(args.agent_risk_ttc_threshold):
        return True, "low_ttc"

    try:
        preferred_lane = str(getattr(world_state, "agent_preferred_lane", "current"))
        if (
            preferred_lane in {"left", "right"}
            and bool(getattr(world_state, "lane_change_permission", False))
            and not bool(getattr(world_state, "agent_lane_change_decision_seen", False))
        ):
            return True, f"scenario_lane_change_{preferred_lane}"
    except Exception:
        pass

    if baseline_intent not in {"keep_lane"}:
        return True, f"baseline_{baseline_intent}"

    try:
        if not bool(getattr(world_state, "corridor_clear", True)):
            return True, "corridor_not_clear"
    except Exception:
        pass

    stride = max(1, int(args.agent_compare_stride))
    if frame_idx % stride == 0:
        return True, f"periodic_stride_{stride}"

    return False, "not_triggered"


def _apply_agent_rate_limit(
    *,
    args: argparse.Namespace,
    requested: bool,
    trigger_reason: str,
    now_s: float,
    last_query_s: Optional[float],
) -> tuple[bool, str]:
    if not requested:
        return False, trigger_reason

    rpm = float(getattr(args, "agent_max_requests_per_minute", 30.0))
    if rpm <= 0.0:
        return True, trigger_reason

    min_interval_s = 60.0 / max(rpm, 1e-6)
    if last_query_s is None or (now_s - last_query_s) >= min_interval_s:
        return True, trigger_reason

    remaining_s = max(0.0, min_interval_s - (now_s - last_query_s))
    return False, f"rate_limited_{rpm:.1f}_rpm_wait_{remaining_s:.2f}s_after_{trigger_reason}"


def _apply_agent_episode_limit(
    *,
    requested: bool,
    trigger_reason: str,
    submitted_count: int,
    max_requests: int,
) -> tuple[bool, str]:
    """Bound high-level Agent calls without affecting the local control loop."""
    if not requested:
        return False, trigger_reason
    if max_requests <= 0 or submitted_count < max_requests:
        return True, trigger_reason
    return False, f"episode_request_cap_{max_requests}_reached"


def _apply_agent_retry_cooldown(
    *,
    requested: bool,
    trigger_reason: str,
    now_s: float,
    last_failure_s: Optional[float],
    cooldown_s: float,
) -> tuple[bool, str]:
    """Delay only replacement calls after a failed/stale Agent response."""
    if not requested or last_failure_s is None or cooldown_s <= 0.0:
        return requested, trigger_reason
    elapsed_s = max(0.0, float(now_s) - float(last_failure_s))
    if elapsed_s >= float(cooldown_s):
        return True, trigger_reason
    remaining_s = max(0.0, float(cooldown_s) - elapsed_s)
    return False, f"retry_cooldown_wait_{remaining_s:.2f}s_after_{trigger_reason}"


def _agent_response_freshness(
    *,
    args: argparse.Namespace,
    result: AsyncAgentResult,
    current_timestamp_s: float,
    current_world_state: Any,
    current_min_ttc_s: float,
) -> tuple[bool, str, float]:
    """Revalidate an asynchronous tactical response against the live scene."""
    response_age_s = max(
        0.0,
        float(current_timestamp_s) - float(result.request.sim_timestamp_s),
    )
    if result.error_type:
        return False, f"async_worker_error_{result.error_type}", response_age_s
    if result.intent_record is None:
        return False, "no_agent_intent", response_age_s
    if response_age_s > float(args.agent_response_max_age_s):
        return False, "stale_response_age", response_age_s

    context = result.request.context
    requested_lane_id = str(context.get("ego_lane_id", "") or "")
    current_lane_id = str(getattr(current_world_state, "ego_lane_id", "") or "")
    if requested_lane_id and current_lane_id and requested_lane_id != current_lane_id:
        return False, "stale_response_lane_changed", response_age_s

    requested_preferred_lane = str(context.get("preferred_lane", "current") or "current")
    current_preferred_lane = str(
        getattr(current_world_state, "agent_preferred_lane", "current") or "current"
    )
    if requested_preferred_lane != current_preferred_lane:
        return False, "stale_response_route_changed", response_age_s

    intent = str(getattr(result.intent_record, "tactical_intent", "keep_lane"))
    if intent in _ASSIST_LANE_CHANGE_INTENTS:
        intent_side = _assist_target_lane(intent)
        if current_preferred_lane in {"left", "right"} and intent_side != current_preferred_lane:
            return False, "stale_response_lane_change_direction_mismatch", response_age_s
        if not bool(getattr(current_world_state, "lane_change_permission", False)):
            return False, "stale_response_lane_change_not_permitted", response_age_s
        ttc_safe, ttc_reason = _lane_change_ttc_safety(
            world_state=current_world_state,
            global_min_ttc_s=current_min_ttc_s,
            threshold_s=float(args.agent_risk_ttc_threshold),
            emergency_floor_s=float(getattr(args, "agent_emergency_ttc_floor_s", 0.75)),
        )
        if not ttc_safe:
            return False, f"stale_response_{ttc_reason}", response_age_s

    return True, "fresh", response_age_s


def _agent_request_context(
    *,
    baseline_intent: str,
    min_ttc_s: float,
    world_state: Any,
    route_info: Dict[str, Any],
    num_detections: int,
) -> Dict[str, Any]:
    return {
        "baseline_intent": str(baseline_intent),
        "min_ttc_s": float(min_ttc_s),
        "ego_v_mps": float(getattr(world_state, "ego_v_mps", 0.0)),
        "ego_lane_id": str(getattr(world_state, "ego_lane_id", "") or ""),
        "preferred_lane": str(
            getattr(world_state, "agent_preferred_lane", "current") or "current"
        ),
        "lane_change_permission": bool(
            getattr(world_state, "lane_change_permission", False)
        ),
        "route_completion_rate": route_info.get("route_completion_rate"),
        "route_progress_m": route_info.get("route_progress_m"),
        "num_detections": int(num_detections),
    }


def _agent_sensor_input(detection_list: Any) -> Dict[str, Any]:
    """Freeze BEVFusion runtime metadata without copying raw sensor payloads."""
    return {
        "inference_time_ms": float(getattr(detection_list, "inference_time_ms", 0.0)),
        "num_detections": len(list(getattr(detection_list, "detections", []) or [])),
        "num_raw_boxes": int(getattr(detection_list, "num_raw_boxes", 0)),
        "lidar_point_count": int(getattr(detection_list, "lidar_point_count", 0)),
        "radar_point_count": int(getattr(detection_list, "radar_point_count", 0)),
    }


_ASSIST_LANE_CHANGE_INTENTS = {
    "prepare_lane_change_left",
    "prepare_lane_change_right",
    "commit_lane_change_left",
    "commit_lane_change_right",
}
_ASSIST_CONSERVATIVE_INTENTS = {"slow_down", "stop", "yield", "follow"}


def _assist_target_lane(agent_intent: str) -> str:
    if "left" in agent_intent:
        return "left"
    if "right" in agent_intent:
        return "right"
    return "current"


def _agent_assist_allowed(
    *,
    args: argparse.Namespace,
    intent_record: Any,
    baseline_intent: str,
    world_state: Any,
    lane_change_completed: bool = False,
) -> Tuple[bool, str]:
    if intent_record is None:
        return False, "no_agent_intent"
    if bool(getattr(intent_record, "fallback_to_baseline", False)):
        return False, "agent_fallback"
    if str(getattr(intent_record, "validation_status", "")) != "valid":
        return False, "agent_invalid"

    agent_intent = str(getattr(intent_record, "tactical_intent", baseline_intent))
    if agent_intent == baseline_intent:
        return False, "same_as_baseline"

    confidence = float(getattr(intent_record, "confidence", 0.0))
    if confidence < float(args.agent_assist_min_confidence):
        return False, "low_confidence"

    if agent_intent in _ASSIST_LANE_CHANGE_INTENTS:
        if lane_change_completed:
            return False, "lane_change_already_completed"
        preferred_lane = str(
            getattr(world_state, "agent_preferred_lane", "current") or "current"
        )
        intent_side = _assist_target_lane(agent_intent)
        if preferred_lane in {"left", "right"} and intent_side != preferred_lane:
            return False, "lane_change_direction_mismatch"
        if not bool(getattr(world_state, "lane_change_permission", False)):
            return False, "lane_change_not_permitted"
        return True, "lane_change_assist"

    if agent_intent in _ASSIST_CONSERVATIVE_INTENTS:
        return True, "conservative_assist"

    return False, "unsupported_intent"


def _trajectory_request_from_agent_intent(
    *,
    baseline_req: Any,
    agent_intent: str,
    world_state: Any,
) -> Any:
    from stage9.schemas import TrajectoryRequest

    ego_v = float(getattr(world_state, "ego_v_mps", 0.0))
    baseline_target_v = float(getattr(baseline_req, "target_v_desired_mps", 0.0))
    if agent_intent in {"stop", "yield"}:
        target_v = 0.0
    elif agent_intent == "slow_down":
        target_v = max(0.5, min(3.0, ego_v * 0.5))
    elif agent_intent in _ASSIST_LANE_CHANGE_INTENTS:
        # Keep blocked-lane assist cautious: enough speed to commit, not enough to ram the blocker.
        if agent_intent.startswith("prepare_lane_change_"):
            target_v = max(0.8, min(max(ego_v, 1.0), 1.5))
        else:
            target_v = max(1.0, min(max(ego_v, 1.5), 2.5))
    else:
        target_v = baseline_target_v
    lateral_bound_m = (
        4.0 if agent_intent in _ASSIST_LANE_CHANGE_INTENTS
        else min(max(float(getattr(baseline_req, "lateral_bound_m", 0.75)), 0.75), 1.0)
    )

    request = TrajectoryRequest(
        source="AGENT_ASSIST",
        tactical_intent=agent_intent,
        v_max_mps=min(float(getattr(baseline_req, "v_max_mps", 8.0)), 8.0),
        a_long_max_mps2=min(float(getattr(baseline_req, "a_long_max_mps2", 2.5)), 2.0),
        a_lat_max_mps2=min(float(getattr(baseline_req, "a_lat_max_mps2", 1.5)), 1.5),
        jerk_max_mps3=min(float(getattr(baseline_req, "jerk_max_mps3", 3.0)), 3.0),
        lateral_bound_m=lateral_bound_m,
        drivable_envelope=getattr(baseline_req, "drivable_envelope", None),
        target_lane_id=_assist_target_lane(agent_intent),
        target_v_desired_mps=target_v,
        horizon_s=min(float(getattr(baseline_req, "horizon_s", 3.0)), 3.0),
        cost_profile="AGENT_ASSIST",
    )
    setattr(request, "current_speed_mps", ego_v)
    setattr(request, "current_lateral_error_m", float(getattr(world_state, "ego_lateral_error_m", 0.0)))
    return request


def _post_lane_change_cruise_request(*, baseline_req: Any, world_state: Any) -> Any:
    from stage9.schemas import TrajectoryRequest

    ego_v = float(getattr(world_state, "ego_v_mps", 0.0))
    target_v = max(3.5, min(max(ego_v + 1.0, 3.5), 5.0))
    request = TrajectoryRequest(
        source="POST_LANE_CHANGE",
        tactical_intent="keep_lane",
        v_max_mps=min(float(getattr(baseline_req, "v_max_mps", 8.0)), 8.0),
        a_long_max_mps2=min(float(getattr(baseline_req, "a_long_max_mps2", 2.5)), 2.0),
        a_lat_max_mps2=min(float(getattr(baseline_req, "a_lat_max_mps2", 1.5)), 1.2),
        jerk_max_mps3=min(float(getattr(baseline_req, "jerk_max_mps3", 3.0)), 3.0),
        lateral_bound_m=1.0,
        drivable_envelope=getattr(baseline_req, "drivable_envelope", None),
        target_lane_id="current",
        target_v_desired_mps=target_v,
        horizon_s=min(float(getattr(baseline_req, "horizon_s", 3.0)), 3.0),
        cost_profile="POST_LANE_CHANGE",
    )
    setattr(request, "current_speed_mps", ego_v)
    setattr(request, "current_lateral_error_m", float(getattr(world_state, "ego_lateral_error_m", 0.0)))
    return request


def _assist_hold_frames(args: argparse.Namespace) -> int:
    """Return the bounded maneuver deadline in frames.

    The legacy name is retained because reports and tests already use it.  The
    duration is now configurable and defaults to 20 s.  The additional bounded
    settling time is needed after the vehicle first crosses the lane boundary
    before the five-frame centre/heading completion check can pass.
    """
    delta_t = max(float(getattr(args, "delta_t", 0.1)), 1e-3)
    timeout_s = max(float(getattr(args, "agent_maneuver_timeout_s", 20.0)), delta_t)
    return max(1, int(math.ceil(timeout_s / delta_t)))


def _assist_lane_stable_required_frames(args: argparse.Namespace) -> int:
    return max(1, int(getattr(args, "agent_lane_stable_frames", 5)))


def _assist_commit_promotion_frames(args: argparse.Namespace) -> int:
    delta_t = max(float(getattr(args, "delta_t", 0.1)), 1e-3)
    return max(5, min(20, int(round(1.0 / delta_t))))


def _assist_lifecycle_action(
    *,
    accepted_new_assist: bool,
    preserve_active_after_fallback: bool,
    response_received: bool,
    can_continue_active: bool,
) -> str:
    """Choose whether to retain, advance, or clear the active maneuver."""
    if accepted_new_assist:
        return "accepted"
    if preserve_active_after_fallback or (not response_received and can_continue_active):
        return "continue"
    return "clear"


def _promote_lane_change_intent(agent_intent: str) -> str:
    if agent_intent == "prepare_lane_change_left":
        return "commit_lane_change_left"
    if agent_intent == "prepare_lane_change_right":
        return "commit_lane_change_right"
    return agent_intent


def _assist_maneuver_phase(
    *,
    intent: Optional[str],
    completed: bool,
    failure_reason: Optional[str],
) -> str:
    if completed:
        return "completed"
    if failure_reason:
        return "failed"
    value = str(intent or "")
    if value.startswith("prepare_lane_change_"):
        return "prepare"
    if value.startswith("commit_lane_change_"):
        return "commit"
    return "idle"


def _merge_active_and_new_agent_intent(active_intent: Optional[str], new_intent: str) -> str:
    active = str(active_intent or "")
    if active.startswith("commit_lane_change_") and new_intent.startswith("prepare_lane_change_"):
        active_side = "left" if "left" in active else "right" if "right" in active else ""
        new_side = "left" if "left" in new_intent else "right" if "right" in new_intent else ""
        if active_side and active_side == new_side:
            return active
    return new_intent


def _retune_active_assist_request(
    *,
    request: Any,
    world_state: Any,
    applied_frames: int,
    args: argparse.Namespace,
) -> str:
    ego_v = float(getattr(world_state, "ego_v_mps", 0.0))
    intent = str(getattr(request, "tactical_intent", "keep_lane"))
    if (
        intent.startswith("prepare_lane_change_")
        and applied_frames >= _assist_commit_promotion_frames(args)
    ):
        intent = _promote_lane_change_intent(intent)
        setattr(request, "tactical_intent", intent)

    if intent in _ASSIST_LANE_CHANGE_INTENTS:
        if intent.startswith("prepare_lane_change_"):
            target_v = max(1.2, min(max(ego_v + 0.2, 1.5), 2.0))
            lateral_bound_m = 4.0
        else:
            target_v = max(2.0, min(max(ego_v + 0.3, 2.6), 3.6))
            lateral_bound_m = 4.5
        setattr(request, "target_v_desired_mps", target_v)
        setattr(request, "lateral_bound_m", lateral_bound_m)
    setattr(request, "current_speed_mps", ego_v)
    setattr(request, "current_lateral_error_m", float(getattr(world_state, "ego_lateral_error_m", 0.0)))
    return intent


def _same_lane_change_family(lhs: Optional[str], rhs: Optional[str]) -> bool:
    left = str(lhs or "")
    right = str(rhs or "")
    if left not in _ASSIST_LANE_CHANGE_INTENTS or right not in _ASSIST_LANE_CHANGE_INTENTS:
        return False
    return _assist_target_lane(left) == _assist_target_lane(right)


def _assist_lane_transition_completed(
    *,
    world_state: Any,
    active_metadata: Dict[str, Any],
    lateral_tolerance_m: float = 0.60,
    heading_tolerance_rad: float = 0.20,
) -> bool:
    """Return whether the current frame is a target-lane stability candidate.

    Consecutive-frame confirmation is maintained by the live loop so this
    helper remains pure and can be reused by unit tests.
    """
    if world_state is None:
        return False
    origin_lane_id = str(active_metadata.get("origin_lane_id") or "")
    if not origin_lane_id:
        return False
    current_lane_id = str(getattr(world_state, "ego_lane_id", "") or "")
    target_lane_id = str(active_metadata.get("target_lane_id") or "")
    if target_lane_id and current_lane_id != target_lane_id:
        return False
    if not target_lane_id and current_lane_id == origin_lane_id:
        return False
    lateral_error_m = abs(float(getattr(world_state, "ego_lateral_error_m", 99.0)))
    heading_error_rad = abs(float(getattr(world_state, "heading_error_rad", 99.0)))
    return bool(current_lane_id) and (
        lateral_error_m <= max(0.0, float(lateral_tolerance_m))
        and heading_error_rad <= max(0.0, float(heading_tolerance_rad))
    )


def _update_lane_transition_stability(
    *,
    candidate: bool,
    previous_frames: int,
    required_frames: int,
) -> tuple[int, bool]:
    stable_frames = int(previous_frames) + 1 if candidate else 0
    required = max(1, int(required_frames))
    return stable_frames, stable_frames >= required


def _post_lane_change_settle_state(
    *,
    completion_timestamp_s: Optional[float],
    current_timestamp_s: float,
    settle_duration_s: float,
) -> tuple[bool, Optional[float]]:
    """Return whether bounded post-maneuver centering still owns control."""
    if completion_timestamp_s is None:
        return False, None
    elapsed_s = max(0.0, float(current_timestamp_s) - float(completion_timestamp_s))
    duration_s = max(0.0, float(settle_duration_s))
    return elapsed_s < duration_s, elapsed_s


def _assist_maneuver_start_timestamp(
    assist_log: List[Dict[str, Any]],
) -> Optional[float]:
    return next(
        (
            float(row["timestamp_s"])
            for row in assist_log
            if row.get("assist_applied")
            and str(row.get("agent_intent", "")) in _ASSIST_LANE_CHANGE_INTENTS
        ),
        None,
    )


def _assist_completion_metadata(
    active_metadata: Dict[str, Any],
    tracked_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep target-lane tracking alive after bounded control authority ends."""
    if active_metadata.get("origin_lane_id"):
        return active_metadata
    return tracked_metadata


def _can_continue_active_assist(
    *,
    args: argparse.Namespace,
    active_request: Any,
    active_metadata: Dict[str, Any],
    baseline_intent: str,
    world_state: Any,
    min_ttc_s: float,
    lane_change_completed: bool = False,
) -> bool:
    if active_request is None or world_state is None:
        return False
    active_intent = str(getattr(active_request, "tactical_intent", ""))
    if active_intent not in _ASSIST_LANE_CHANGE_INTENTS:
        return False
    if baseline_intent not in {"stop_before_obstacle", "follow", "keep_lane"}:
        return False
    ttc_safe, _ = _lane_change_ttc_safety(
        world_state=world_state,
        global_min_ttc_s=min_ttc_s,
        threshold_s=max(1.5, float(args.agent_risk_ttc_threshold) - 0.25),
        emergency_floor_s=float(getattr(args, "agent_emergency_ttc_floor_s", 0.75)),
    )
    if not ttc_safe:
        return False
    if not bool(getattr(world_state, "lane_change_permission", False)):
        return False
    if lane_change_completed:
        return False
    return True


def _active_assist_stop_reason(
    *,
    args: argparse.Namespace,
    active_request: Any,
    baseline_intent: str,
    world_state: Any,
    min_ttc_s: float,
    lane_change_completed: bool,
) -> Optional[str]:
    """Explain why a previously accepted maneuver can no longer continue."""
    if active_request is None or world_state is None:
        return "active_assist_unavailable"
    if lane_change_completed:
        return "lane_change_completed"
    active_intent = str(getattr(active_request, "tactical_intent", ""))
    if active_intent not in _ASSIST_LANE_CHANGE_INTENTS:
        return "active_intent_not_lane_change"
    if baseline_intent not in {"stop_before_obstacle", "follow", "keep_lane"}:
        return "safety_abort_baseline_conflict"
    ttc_safe, ttc_reason = _lane_change_ttc_safety(
        world_state=world_state,
        global_min_ttc_s=min_ttc_s,
        threshold_s=max(1.5, float(args.agent_risk_ttc_threshold) - 0.25),
        emergency_floor_s=float(getattr(args, "agent_emergency_ttc_floor_s", 0.75)),
    )
    if not ttc_safe:
        return f"safety_abort_{ttc_reason}"
    if not bool(getattr(world_state, "lane_change_permission", False)):
        return "safety_abort_lane_change_not_permitted"
    return None


def _summarize_assist_log(
    assist_log: List[Dict[str, Any]],
    stats: Dict[str, Any],
    args: argparse.Namespace,
    worker_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    submitted = [r for r in assist_log if r.get("agent_queried")]
    attempted = [r for r in assist_log if r.get("agent_response_received")]
    fallback_frames = [
        r
        for r in attempted
        if bool(r.get("agent_fallback_to_baseline")) or r.get("agent_worker_error_type")
    ]
    fallback_row_ids = {id(row) for row in fallback_frames}
    timeout_frames = [
        row
        for row in fallback_frames
        if "timeout" in str(
            row.get("agent_worker_error_type")
            or row.get("agent_fallback_reason")
            or ""
        ).lower()
    ]
    arbitration_evaluated = [
        row for row in attempted if id(row) not in fallback_row_ids
    ]
    accepted = [r for r in assist_log if r.get("assist_applied")]
    rejected = [
        r for r in assist_log
        if r.get("agent_response_received") and not r.get("assist_applied")
    ]
    arbitration_rejected = [
        row for row in arbitration_evaluated if not row.get("assist_applied")
    ]
    agent_decision_applied = [
        r
        for r in accepted
        if r.get("agent_response_received") and not r.get("post_lane_change_cruise")
    ]
    assist_hold_applied = [r for r in accepted if r.get("assist_continued")]
    post_lane_change_applied = [r for r in accepted if r.get("post_lane_change_cruise")]
    post_lane_change_handoff = [
        r for r in assist_log if r.get("post_lane_change_handoff_to_baseline")
    ]
    controller_only_applied = [r for r in accepted if not r.get("agent_queried")]
    rejection_counts: Dict[str, int] = {}
    query_rejection_counts: Dict[str, int] = {}
    non_query_reason_counts: Dict[str, int] = {}
    intent_counts: Dict[str, int] = {}
    fallback_reason_counts: Dict[str, int] = {}
    arbitration_rejection_reason_counts: Dict[str, int] = {}
    validation_status_counts: Dict[str, int] = {}
    for row in assist_log:
        if row.get("agent_intent"):
            intent = str(row["agent_intent"])
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        if row.get("assist_reject_reason"):
            reason = str(row["assist_reject_reason"])
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            if row.get("agent_response_received") and not row.get("assist_applied"):
                query_rejection_counts[reason] = query_rejection_counts.get(reason, 0) + 1
            elif not row.get("agent_response_received"):
                non_query_reason_counts[reason] = non_query_reason_counts.get(reason, 0) + 1
        if row.get("agent_validation_status"):
            status = str(row["agent_validation_status"])
            validation_status_counts[status] = validation_status_counts.get(status, 0) + 1
        if row.get("agent_fallback_reason"):
            reason = str(row["agent_fallback_reason"])
            fallback_reason_counts[reason] = fallback_reason_counts.get(reason, 0) + 1
    for row in arbitration_rejected:
        reason = str(row.get("assist_reject_reason") or "unspecified")
        arbitration_rejection_reason_counts[reason] = (
            arbitration_rejection_reason_counts.get(reason, 0) + 1
        )

    sim_frames = int(stats.get("frames", 0))
    call_latencies = [
        float(row["agent_call_latency_ms"])
        for row in attempted
        if row.get("agent_call_latency_ms") is not None
    ]
    response_ages = [
        float(row["agent_response_age_s"])
        for row in attempted
        if row.get("agent_response_age_s") is not None
    ]
    api_attempt_counts = [
        int(row["agent_api_attempt_count"])
        for row in attempted
        if row.get("agent_api_attempt_count") is not None
    ]
    api_payload_variant_counts: Dict[str, int] = {}
    for row in attempted:
        if row.get("agent_api_payload_variant"):
            variant = str(row["agent_api_payload_variant"])
            api_payload_variant_counts[variant] = api_payload_variant_counts.get(variant, 0) + 1
    stale_responses = [
        row for row in attempted if row.get("agent_response_fresh") is False
    ]
    accepted_query_frames = [row for row in attempted if row.get("assist_applied")]
    maneuver_start = _assist_maneuver_start_timestamp(assist_log)
    post_cruise_end = next(
        (
            float(row["timestamp_s"])
            for row in assist_log
            if maneuver_start is not None
            and float(row.get("timestamp_s", -1.0)) >= maneuver_start
            and bool(row.get("post_lane_change_cruise"))
        ),
        None,
    )
    # Lane-transition completion is a physical event and must not depend on
    # the optional post-transition cruise policy. In particular, low TTC can
    # legitimately suppress post-cruise centering even after the ego vehicle
    # has reached the target lane. The live loop records the first stable
    # target-lane frame explicitly for this metric.
    lane_transition_end = next(
        (
            float(row["lane_change_completion_timestamp_s"])
            for row in assist_log
            if maneuver_start is not None
            and row.get("lane_change_completion_timestamp_s") is not None
            and float(row["lane_change_completion_timestamp_s"]) >= maneuver_start
        ),
        None,
    )
    maneuver_end = lane_transition_end if lane_transition_end is not None else post_cruise_end
    explicit_failure_row = next(
        (row for row in assist_log if row.get("maneuver_failure_reason")),
        None,
    )
    explicit_failure_reason = (
        str(explicit_failure_row.get("maneuver_failure_reason"))
        if explicit_failure_row is not None else None
    )
    explicit_failure_timestamp_s = (
        float(explicit_failure_row["maneuver_failure_timestamp_s"])
        if explicit_failure_row is not None
        and explicit_failure_row.get("maneuver_failure_timestamp_s") is not None
        else None
    )
    if maneuver_end is not None:
        maneuver_failure_reason = None
        maneuver_failure_timestamp_s = None
    elif explicit_failure_reason:
        maneuver_failure_reason = explicit_failure_reason
        maneuver_failure_timestamp_s = explicit_failure_timestamp_s
    elif maneuver_start is not None:
        maneuver_failure_reason = "episode_ended_before_completion"
        maneuver_failure_timestamp_s = None
    elif fallback_frames:
        maneuver_failure_reason = "agent_api_failure"
        maneuver_failure_timestamp_s = None
    elif arbitration_rejected:
        maneuver_failure_reason = "safety_arbitration_rejected"
        maneuver_failure_timestamp_s = None
    elif agent_decision_applied:
        maneuver_failure_reason = "no_lane_change_intent"
        maneuver_failure_timestamp_s = None
    elif submitted:
        maneuver_failure_reason = "agent_response_not_applied"
        maneuver_failure_timestamp_s = None
    else:
        maneuver_failure_reason = "agent_not_queried"
        maneuver_failure_timestamp_s = None
    latency_budget_ms = max(float(args.delta_t) * 1000.0, 1e-6)
    control_loop_latencies = [
        float(value) for value in stats.get("tick_latency_samples_ms", [])
    ]
    return {
        "schema_version": "stage10_agent_assist_evaluation_v4",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "sim_frames": sim_frames,
        "random_seed": int(args.seed),
        "policy": {
            "max_requests_per_episode": int(getattr(args, "agent_max_requests_per_episode", 0)),
            "retry_cooldown_s": float(getattr(args, "agent_retry_cooldown_s", 0.0)),
            "api_timeout_s": float(getattr(args, "agent_api_timeout_s", 0.0)),
            "api_max_retries": int(getattr(args, "agent_api_max_retries", 0)),
            "response_max_age_s": float(getattr(args, "agent_response_max_age_s", 0.0)),
            "maneuver_timeout_s": float(getattr(args, "agent_maneuver_timeout_s", 20.0)),
            "lane_stable_frames": int(getattr(args, "agent_lane_stable_frames", 5)),
            "lane_center_tolerance_m": float(
                getattr(args, "agent_lane_center_tolerance_m", 0.60)
            ),
            "lane_heading_tolerance_rad": float(
                getattr(args, "agent_lane_heading_tolerance_rad", 0.20)
            ),
            "post_lane_change_settle_s": float(
                getattr(args, "agent_post_lane_change_settle_s", 2.0)
            ),
            "cross_lane_max_steer": float(
                getattr(args, "agent_cross_lane_max_steer", 0.72)
            ),
            "cross_lane_min_steer": float(
                getattr(args, "agent_cross_lane_min_steer", 0.42)
            ),
            "target_corridor_half_width_m": float(
                getattr(args, "agent_target_corridor_half_width_m", 1.60)
            ),
            "target_rear_clearance_m": float(
                getattr(args, "agent_target_rear_clearance_m", 8.0)
            ),
            "emergency_ttc_floor_s": float(
                getattr(args, "agent_emergency_ttc_floor_s", 0.75)
            ),
        },
        "agent_query_frames": len(submitted),
        "agent_response_frames": len(attempted),
        "agent_fallback_frames": len(fallback_frames),
        "agent_fallback_rate": round(len(fallback_frames) / max(len(attempted), 1), 4),
        "agent_api_failure_frames": len(fallback_frames),
        "agent_api_failure_rate": round(
            len(fallback_frames) / max(len(attempted), 1), 4
        ),
        "agent_timeout_frames": len(timeout_frames),
        "agent_timeout_rate": round(
            len(timeout_frames) / max(len(attempted), 1), 4
        ),
        "safety_arbitration_evaluated_frames": len(arbitration_evaluated),
        "safety_arbitration_rejected_frames": len(arbitration_rejected),
        "safety_arbitration_rejection_rate": (
            round(len(arbitration_rejected) / len(arbitration_evaluated), 4)
            if arbitration_evaluated else None
        ),
        "safety_arbitration_acceptance_rate": (
            round(
                (len(arbitration_evaluated) - len(arbitration_rejected))
                / len(arbitration_evaluated),
                4,
            )
            if arbitration_evaluated else None
        ),
        "assist_applied_frames": len(accepted),
        "agent_decision_applied_frames": len(agent_decision_applied),
        "assist_hold_applied_frames": len(assist_hold_applied),
        "post_lane_change_cruise_frames": len(post_lane_change_applied),
        "post_lane_change_handoff_frames": len(post_lane_change_handoff),
        "post_lane_change_handoff_timestamp_s": (
            float(post_lane_change_handoff[0]["timestamp_s"])
            if post_lane_change_handoff else None
        ),
        "controller_only_applied_frames": len(controller_only_applied),
        "assist_rejected_frames": len(rejected),
        "agent_query_acceptance_rate": round(len(accepted_query_frames) / max(len(attempted), 1), 4),
        "agent_query_rejection_rate": round(len(rejected) / max(len(attempted), 1), 4),
        "assist_intervention_rate": round(len(accepted) / max(sim_frames, 1), 4),
        "agent_decision_intervention_rate": round(len(agent_decision_applied) / max(sim_frames, 1), 4),
        "agent_query_rate": round(len(submitted) / max(sim_frames, 1), 4),
        "end_to_end_query_success_rate": (
            round(len(accepted_query_frames) / len(submitted), 4)
            if submitted else None
        ),
        "agent_response_rate": round(len(attempted) / max(sim_frames, 1), 4),
        "stale_response_count": len(stale_responses),
        "stale_response_discard_rate": round(
            len(stale_responses) / max(len(attempted), 1), 4
        ),
        "agent_intent_distribution": intent_counts,
        "assist_reject_reason_counts": rejection_counts,
        "query_rejection_reason_counts": query_rejection_counts,
        "non_query_reason_counts": non_query_reason_counts,
        "agent_validation_status_counts": validation_status_counts,
        "agent_fallback_reason_counts": fallback_reason_counts,
        "agent_api_attempt_count_total": sum(api_attempt_counts),
        "agent_api_attempt_count_max": max(api_attempt_counts) if api_attempt_counts else None,
        "agent_api_payload_variant_counts": api_payload_variant_counts,
        "maneuver_failure_reason_counts": (
            {maneuver_failure_reason: 1} if maneuver_failure_reason else {}
        ),
        "safety_arbitration_rejection_reason_counts": arbitration_rejection_reason_counts,
        "async_worker": dict(worker_stats or {}),
        "control_loop_latency": {
            "sample_count": len(control_loop_latencies),
            "mean_ms": (
                round(sum(control_loop_latencies) / len(control_loop_latencies), 3)
                if control_loop_latencies else None
            ),
            "p50_ms": (
                round(float(_percentile(control_loop_latencies, 50.0)), 3)
                if control_loop_latencies else None
            ),
            "p95_ms": (
                round(float(_percentile(control_loop_latencies, 95.0)), 3)
                if control_loop_latencies else None
            ),
            "max_ms": (
                round(max(control_loop_latencies), 3)
                if control_loop_latencies else None
            ),
            "simulation_step_budget_ms": round(latency_budget_ms, 3),
            "over_step_budget_rate": (
                round(
                    sum(value > latency_budget_ms for value in control_loop_latencies)
                    / len(control_loop_latencies),
                    4,
                )
                if control_loop_latencies else None
            ),
        },
        "latency": {
            "sample_count": len(call_latencies),
            "mean_api_call_ms": (
                round(sum(call_latencies) / len(call_latencies), 3) if call_latencies else None
            ),
            "p50_api_call_ms": (
                round(float(_percentile(call_latencies, 50.0)), 3) if call_latencies else None
            ),
            "p95_api_call_ms": (
                round(float(_percentile(call_latencies, 95.0)), 3) if call_latencies else None
            ),
            "max_api_call_ms": round(max(call_latencies), 3) if call_latencies else None,
            "simulation_step_budget_ms": round(latency_budget_ms, 3),
            "over_step_budget_rate": (
                round(sum(value > latency_budget_ms for value in call_latencies) / len(call_latencies), 4)
                if call_latencies else None
            ),
            "mean_response_age_s": (
                round(sum(response_ages) / len(response_ages), 6)
                if response_ages else None
            ),
            "p95_response_age_s": (
                round(float(_percentile(response_ages, 95.0)), 6)
                if response_ages else None
            ),
        },
        "lane_change_maneuver": {
            "started_timestamp_s": maneuver_start,
            "completed_timestamp_s": maneuver_end,
            "completion_time_s": (
                round(maneuver_end - maneuver_start, 6)
                if maneuver_start is not None and maneuver_end is not None else None
            ),
            "completed": maneuver_end is not None,
            "failure_reason": maneuver_failure_reason,
            "failure_timestamp_s": maneuver_failure_timestamp_s,
            "completion_source": (
                "lane_transition"
                if lane_transition_end is not None
                else ("post_lane_change_cruise" if post_cruise_end is not None else None)
            ),
            "post_lane_change_cruise_timestamp_s": post_cruise_end,
        },
        "frame_log": assist_log,
    }


# ── Stage 9 Arbiter bootstrap ─────────────────────────────────────────────────
# TODO [Stage 9 hook]: Replace stubs with real Stage 9 components.

def _build_stage9_arbiter(
    log_dir: Path,
    agent_mode: str = "stub",
    agent_control_mode: str = "shadow",
):
    """
    Build and return the Stage 9 AuthorityArbiter with REAL components.
    Returns None if imports fail (graceful degradation).
    """
    try:
        from stage9 import (
            AuthorityArbiter,
            AuthorityStateMachine,
            BaselineDetector,
            ContractResolver,
            HandoffPlanner,
            HumanOverrideMonitor,
            MRMExecutor,
            SafetySupervisor,
            TORManager,
        )
    except ImportError as exc:
        LOGGER.warning("Stage 9 not available: %s — running perception-only.", exc)
        return None

    # ── Real Adapter components (Stage 6 MPC + Stage 7 Agent) ────────────────
    try:
        from carla_bevfusion_stage1.stage9_adapters import (
            RealMPCAdapter,
            RealBaselineAdapter,
            RealAgentAdapter,
        )
        mpc_impl = RealMPCAdapter(dt_s=0.1)
        baseline_impl = RealBaselineAdapter()
        _adapter_mode = "api" if agent_mode in ("api", "compare") else "stub"
        if agent_control_mode in {"baseline", "shadow", "assist"}:
            class _NoopAgent:
                def propose_contract(self, w): return None
                def get_intent(self, w, c): return "keep_lane"
            agent_impl = _NoopAgent()
        else:
            agent_impl = RealAgentAdapter(mode=_adapter_mode)
        LOGGER.info("Stage 10: Real MPC + Baseline + Agent adapters loaded (agent_mode=%s).", _adapter_mode)
    except ImportError as exc:
        LOGGER.warning("stage9_adapters unavailable (%s). Using minimal stubs.", exc)

        class _StubBaseline:
            def plan(self, w): return _stub_traj()
            def degraded_hold(self, w): return _stub_traj()
            def is_healthy(self, w): return True

        class _StubMPC:
            def execute(self, req): return _stub_cmd()
            def preview_feasible(self, req): return True

        class _StubAgent:
            def propose_contract(self, w): return None
            def get_intent(self, w, c): return "keep_lane"

        mpc_impl = _StubMPC()
        baseline_impl = _StubBaseline()
        agent_impl = _StubAgent()

    # ── Stubs still needed for TOR (no real TOR implementation) ──────────────
    class _StubTOR:
        MIN_WAIT_BUDGET_S = 5.0
        def start(self, w): pass
        def tick(self, w): pass
        def timed_out(self): return False

    def _stub_traj():
        from stage9.schemas import TrajectoryRequest
        return TrajectoryRequest()

    def _stub_cmd():
        from stage9.schemas import ActuatorCommand
        return ActuatorCommand(steer=0.0, throttle=0.0, brake=0.0)

    log_dir.mkdir(parents=True, exist_ok=True)
    arbiter = AuthorityArbiter(
        asm=AuthorityStateMachine(),
        baseline_detector=BaselineDetector(),
        human_override=HumanOverrideMonitor(),
        agent=agent_impl,
        supervisor=SafetySupervisor(),
        baseline=baseline_impl,
        contract_resolver=ContractResolver(),
        handoff_planner=HandoffPlanner(),
        mpc=mpc_impl,
        tor=_StubTOR(),
        mrm=MRMExecutor(),
        log_path=log_dir / "stage9_authority_log.jsonl",
    )
    LOGGER.info("Stage 9 AuthorityArbiter initialised with REAL components.")
    return arbiter



# ── Main loop ─────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    if int(args.seed) < 0:
        raise ValueError("--seed must be non-negative")
    if float(args.agent_api_timeout_s) <= 0.0:
        raise ValueError("--agent-api-timeout-s must be positive")
    if int(args.agent_api_max_retries) < 0:
        raise ValueError("--agent-api-max-retries must be non-negative")
    if float(args.agent_response_max_age_s) <= 0.0:
        raise ValueError("--agent-response-max-age-s must be positive")
    if int(args.agent_max_requests_per_episode) < 0:
        raise ValueError("--agent-max-requests-per-episode must be non-negative")
    if float(args.agent_retry_cooldown_s) < 0.0:
        raise ValueError("--agent-retry-cooldown-s must be non-negative")
    if float(args.agent_maneuver_timeout_s) <= 0.0:
        raise ValueError("--agent-maneuver-timeout-s must be positive")
    if int(args.agent_lane_stable_frames) <= 0:
        raise ValueError("--agent-lane-stable-frames must be positive")
    if float(args.agent_lane_center_tolerance_m) < 0.0:
        raise ValueError("--agent-lane-center-tolerance-m must be non-negative")
    if float(args.agent_lane_heading_tolerance_rad) < 0.0:
        raise ValueError("--agent-lane-heading-tolerance-rad must be non-negative")
    if float(args.agent_post_lane_change_settle_s) < 0.0:
        raise ValueError("--agent-post-lane-change-settle-s must be non-negative")
    if not 0.0 < float(args.agent_cross_lane_max_steer) <= 1.0:
        raise ValueError("--agent-cross-lane-max-steer must be in (0, 1]")
    if not 0.0 <= float(args.agent_cross_lane_min_steer) <= float(args.agent_cross_lane_max_steer):
        raise ValueError(
            "--agent-cross-lane-min-steer must be in [0, --agent-cross-lane-max-steer]"
        )
    if float(args.agent_target_corridor_half_width_m) <= 0.0:
        raise ValueError("--agent-target-corridor-half-width-m must be positive")
    if float(args.agent_target_rear_clearance_m) < 0.0:
        raise ValueError("--agent-target-rear-clearance-m must be non-negative")
    if float(args.agent_emergency_ttc_floor_s) < 0.0:
        raise ValueError("--agent-emergency-ttc-floor-s must be non-negative")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    _ensure_carla_pythonapi_paths(args.carla_root)

    from carla_bevfusion_stage1.bevfusion_runtime import build_bevfusion_model
    from carla_bevfusion_stage1.bevfusion_live_adapter import BEVFusionLiveAdapter
    from carla_bevfusion_stage1.carla_sensor_sync import CarlaSensorSync
    from carla_bevfusion_stage1.rig import build_rig_preset
    from carla_bevfusion_stage1.world_state_builder import WorldStateBuilder, EgoTelemetry

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ego_trace_path = log_dir / "ego_trace.jsonl"
    collision_events_path = log_dir / "collision_events.jsonl"
    collision_episodes_path = log_dir / "collision_episodes.jsonl"
    lane_invasion_events_path = log_dir / "lane_invasion_events.jsonl"
    if ego_trace_path.exists():
        ego_trace_path.unlink()
    if collision_events_path.exists():
        collision_events_path.unlink()
    if collision_episodes_path.exists():
        collision_episodes_path.unlink()
    if lane_invasion_events_path.exists():
        lane_invasion_events_path.unlink()

    # ── 1. Build BEVFusion model FIRST (before sensors start) ────────────────
    # IMPORTANT: mmdet3d import takes ~50s. Loading model before attaching CARLA
    # sensors prevents sensor buffer overflow / frame-sync timeouts.
    LOGGER.info("Loading BEVFusion model (before sensor spawn to avoid timeout) …")
    model, cfg, BoxClass = build_bevfusion_model(
        repo_root=args.bev_repo,
        config_path=args.bev_config,
        checkpoint_path=args.bev_ckpt,
        device=args.device,
        radar_ablation=args.radar_ablation,
    )
    bev_adapter = BEVFusionLiveAdapter(
        model, cfg, BoxClass,
        device=args.device,
        score_threshold=args.score_thresh,
        lidar_max_points=args.adapter_lidar_max_points,
        radar_max_points=args.adapter_radar_max_points,
    )
    LOGGER.info("BEVFusion ready.")

    # ── 2. Input Source (CARLA or Watch Folder) ───────────────────────────────
    # Sensors are spawned NOW, after the model is warm, so no frames are missed.
    route_tracker: Optional[RouteProgressTracker] = None
    collision_monitor: Optional[CollisionMonitor] = None
    lane_invasion_monitor: Optional[LaneInvasionMonitor] = None
    video_recorder = None
    world = None
    traffic_manager = None
    scenario_manifest: Optional[Dict[str, Any]] = None
    agent_preferred_lane = "current"
    agent_origin_lane_id = ""
    agent_target_lane_id = ""
    ego_spawned_by_stage10 = False
    if args.samples_root:
        sensor_source = FolderWatcherSync(args.samples_root, args.max_frames)
        ego = None
        carla_map = None
        if bool(args.record_mp4):
            LOGGER.warning("Stage10 --record-mp4 requires live CARLA mode; watch mode has no ego camera to record.")
    else:
        scenario_manifest = _load_scenario_manifest(args.scenario_manifest)
        args.map = _resolve_target_map(args, scenario_manifest)
        agent_preferred_lane = _agent_preferred_lane_from_manifest(scenario_manifest)
        agent_origin_lane_id = _agent_origin_lane_id_from_manifest(scenario_manifest)
        agent_target_lane_id = _agent_target_lane_id_from_manifest(scenario_manifest)
        client, world = _connect_carla(args.carla_host, args.carla_port)
        world = _load_map_if_needed(client, world, args.map)
        try:
            world.set_pedestrians_seed(int(args.seed))
        except (AttributeError, RuntimeError):
            LOGGER.debug("Pedestrian seed configuration unavailable")
        if _moving_adjacent_npcs_enabled(scenario_manifest):
            tm_port = int((scenario_manifest or {}).get("tm_port", args.tm_port))
            traffic_manager = client.get_trafficmanager(tm_port)
            try:
                traffic_manager.set_random_device_seed(int(args.seed))
            except (AttributeError, RuntimeError):
                LOGGER.warning("Traffic Manager seed configuration unavailable")
            LOGGER.info("Stage10 moving adjacent NPCs: Traffic Manager sync enabled on port %d", tm_port)
        carla_map = world.get_map()
        ego, ego_spawned_by_stage10, ego_source = _spawn_or_get_ego(world, args, scenario_manifest)
        if traffic_manager is not None:
            _activate_moving_adjacent_npcs(
                world=world,
                traffic_manager=traffic_manager,
                tm_port=int((scenario_manifest or {}).get("tm_port", args.tm_port)),
                scenario_manifest=scenario_manifest,
            )
        route_tracker = RouteProgressTracker.build(
            carla_map=carla_map,
            spawn_point_index=args.spawn_point,
            target_spawn_point_index=args.route_target_spawn_point,
            fallback_distance_m=args.route_distance_m,
            start_location=None if ego_spawned_by_stage10 else ego.get_location(),
        )
        LOGGER.info(
            "Driving metrics enabled: route_mode=%s route_length=%.1fm success_rc>=%.0f%%",
            route_tracker.mode,
            route_tracker.route_length_m if route_tracker is not None else 0.0,
            float(args.success_rc_threshold) * 100.0,
        )
        LOGGER.info("Stage10 ego source=%s actor_id=%s", ego_source, getattr(ego, "id", "unknown"))
        LOGGER.info(
            "Stage10 Agent route hint: preferred_lane=%s origin_lane_id=%s target_lane_id=%s",
            agent_preferred_lane,
            agent_origin_lane_id or "unknown",
            agent_target_lane_id or "unknown",
        )
        collision_monitor = CollisionMonitor(
            world,
            ego,
            impulse_threshold=float(args.collision_impulse_threshold),
        )
        collision_monitor.start()
        lane_invasion_monitor = LaneInvasionMonitor(world, ego)
        lane_invasion_monitor.start()

        preset = build_rig_preset(
            args.rig_profile,
            image_width=args.image_width,
            image_height=args.image_height,
            camera_fov=70.0,
            fixed_delta_seconds=args.delta_t,
        )
        sensor_source = CarlaSensorSync(  # type: ignore
            world, ego, preset,
            fixed_delta_seconds=args.delta_t,
            image_width=args.image_width,
            image_height=args.image_height,
            enable_radar=args.enable_radar,
            traffic_manager=traffic_manager,
        )

        effective_record_mp4 = bool(args.record_mp4)
        if effective_record_mp4 and _mp4_recording_disabled():
            LOGGER.warning("Stage10 MP4 recording disabled by AGENTAI_DISABLE_MP4; ignoring --record-mp4")
            effective_record_mp4 = False
        if effective_record_mp4:
            from stage3c.video_recorder import Stage3CVideoRecorder

            recording_path = (
                Path(args.recording_path)
                if args.recording_path
                else log_dir / "video" / "stage10_live.mp4"
            )
            recording_fps = (
                float(args.recording_fps)
                if float(args.recording_fps) > 0.0
                else 1.0 / max(float(args.delta_t), 1e-6)
            )
            video_recorder = Stage3CVideoRecorder(
                world=world,
                vehicle=ego,
                output_path=recording_path,
                fps=recording_fps,
                width=int(args.recording_width),
                height=int(args.recording_height),
                fov=float(args.recording_fov),
                mode=str(args.recording_camera_mode),
                overlay=bool(args.recording_overlay),
            )
            LOGGER.info("Stage10 MP4 recording enabled -> %s", recording_path)

    # ── 3. Stage 9 Arbiter (optional) ────────────────────────────────────────
    arbiter = None if args.no_stage9 else _build_stage9_arbiter(
        log_dir,
        agent_mode=args.agent_mode,
        agent_control_mode=args.agent_control_mode,
    )

    # ── 3b. Compare-mode: separate agent adapter for side-by-side logging ────
    compare_agent = None
    compare_agent_worker: Optional[LatestOnlyAgentWorker] = None
    compare_agent_worker_stats: Dict[str, Any] = {}
    compare_baseline = None
    compare_log: List[Dict[str, Any]] = []
    compare_skipped_frames = 0
    last_compare_agent_query_wall_s: Optional[float] = None
    compare_agent_queries_submitted = 0
    if args.agent_mode == "compare":
        try:
            from carla_bevfusion_stage1.stage9_adapters import RealAgentAdapter, RealBaselineAdapter
            compare_agent = RealAgentAdapter(
                mode="api",
                api_timeout_s=float(args.agent_api_timeout_s),
                api_max_retries=int(args.agent_api_max_retries),
            )
            compare_agent_worker = LatestOnlyAgentWorker(
                compare_agent.observe_intent_request,
                name="stage10-compare-agent",
            )
            compare_baseline = RealBaselineAdapter()
            LOGGER.info("Compare mode: side-by-side Agent vs Baseline logging ENABLED.")
        except ImportError as exc:
            LOGGER.warning("Compare mode unavailable (%s). Falling back to normal mode.", exc)
            compare_agent = None

    assist_agent = None
    assist_agent_worker: Optional[LatestOnlyAgentWorker] = None
    assist_agent_worker_stats: Dict[str, Any] = {}
    assist_baseline = None
    assist_mpc = None
    assist_log: List[Dict[str, Any]] = []
    last_assist_agent_query_wall_s: Optional[float] = None
    last_assist_agent_failure_wall_s: Optional[float] = None
    assist_agent_queries_submitted = 0
    assist_agent_valid_decisions = 0
    assist_agent_terminal_decision = False
    active_assist_request: Any = None
    active_assist_intent: Optional[str] = None
    active_assist_hold_remaining = 0
    active_assist_applied_frames = 0
    active_assist_metadata: Dict[str, Any] = {}
    assist_maneuver_tracking_metadata: Dict[str, Any] = {}
    assist_lane_change_completed = False
    assist_lane_change_completed_timestamp_s: Optional[float] = None
    assist_lane_transition_stable_frames = 0
    assist_maneuver_failure_reason: Optional[str] = None
    assist_maneuver_failure_timestamp_s: Optional[float] = None
    agent_lane_change_decision_seen = False
    if args.agent_control_mode == "assist":
        try:
            from carla_bevfusion_stage1.stage9_adapters import (
                RealAgentAdapter,
                RealBaselineAdapter,
                RealMPCAdapter,
            )
            assist_agent = RealAgentAdapter(
                mode="api" if args.agent_mode in ("api", "compare") else "stub",
                api_timeout_s=float(args.agent_api_timeout_s),
                api_max_retries=int(args.agent_api_max_retries),
            )
            assist_agent_worker = LatestOnlyAgentWorker(
                assist_agent.observe_intent_request,
                name="stage10-assist-agent",
            )
            assist_baseline = RealBaselineAdapter()
            assist_mpc = RealMPCAdapter(dt_s=float(args.delta_t))
            LOGGER.info(
                "Agent assist mode ENABLED: async bounded tactical intent -> safety revalidation -> MPC."
            )
        except ImportError as exc:
            LOGGER.warning("Agent assist unavailable (%s). Falling back to baseline control.", exc)
            assist_agent = None
            assist_baseline = None
            assist_mpc = None


    # ── 4. Sensor rig + sync ──────────────────────────────────────────────────
    # (Moved setup to sensor_source initialization above)

    # ── 5. WorldState Builder ─────────────────────────────────────────────────
    ws_builder = WorldStateBuilder(human_driver_available=True)

    # ── Graceful shutdown on Ctrl-C ───────────────────────────────────────────
    _shutdown = [False]
    def _sigint_handler(sig, frame):
        LOGGER.warning("SIGINT received – stopping after current frame.")
        _shutdown[0] = True
    signal.signal(signal.SIGINT, _sigint_handler)

    sensor_source.start()
    LOGGER.info("=" * 60)
    LOGGER.info("Stage 10 Bridge started. source=%s", "WatchMode" if args.samples_root else "LiveMode")
    LOGGER.info("=" * 60)

    stats = {
        "frames": 0,
        "total_det": 0,
        "bev_ms_sum": 0.0,
        "errors": 0,
        "offroad_frames": 0,
        "longitudinal_jerk_samples_mps3": [],
        "tick_latency_samples_ms": [],
        "lidar_point_samples": [],
        "radar_point_samples": [],
    }
    previous_speed_mps: Optional[float] = None
    previous_acceleration_mps2: Optional[float] = None
    previous_ego_timestamp_s: Optional[float] = None

    try:
        for frame_idx in range(args.max_frames):
            if _shutdown[0]:
                break

            t_tick = time.monotonic()

            # ── A. Sensor tick ─────────────────────────────────────────────
            try:
                live_frame = sensor_source.tick()
                if live_frame is None:
                    break
            except TimeoutError as exc:
                LOGGER.error("Sensor timeout frame=%d: %s", frame_idx, exc)
                stats["errors"] += 1
                continue

            # ── B. BEVFusion inference ─────────────────────────────────────
            det_list = bev_adapter.adapt_and_infer(live_frame)
            stats["bev_ms_sum"] += det_list.inference_time_ms
            stats["total_det"] += len(det_list.detections)
            stats["lidar_point_samples"].append(int(det_list.lidar_point_count))
            stats["radar_point_samples"].append(int(det_list.radar_point_count))

            # ── C. Ego telemetry ───────────────────────────────────────────
            if not args.samples_root and ego is not None and carla_map is not None:
                ego_tel = WorldStateBuilder.extract_ego_telemetry(
                    ego, carla_map,
                    frame_id=live_frame.frame_id,
                    timestamp_s=live_frame.timestamp_s,
                    prev_v_mps=float(previous_speed_mps or 0.0),
                )
            else:
                # Reconstruct from meta in watch mode
                ego_tel = EgoTelemetry(
                    frame_id=live_frame.frame_id,
                    timestamp_s=live_frame.timestamp_s,
                    ego_lane_id="watch_mode_unknown",
                    ego_v_mps=float(np.sqrt(np.sum(live_frame.ego_velocity_carla**2))),
                    ego_a_mps2=0.0,
                    ego_lateral_error_m=0.0,
                    heading_error_rad=0.0,
                    ego_location_xyz=np.zeros(3),
                )
            acceleration_mps2, jerk_mps3 = _finite_difference_longitudinal_kinematics(
                speed_mps=float(ego_tel.ego_v_mps),
                timestamp_s=float(live_frame.timestamp_s),
                previous_speed_mps=previous_speed_mps,
                previous_timestamp_s=previous_ego_timestamp_s,
                previous_acceleration_mps2=previous_acceleration_mps2,
            )
            ego_tel.ego_a_mps2 = acceleration_mps2
            if jerk_mps3 is not None:
                stats["longitudinal_jerk_samples_mps3"].append(jerk_mps3)
            previous_speed_mps = float(ego_tel.ego_v_mps)
            previous_acceleration_mps2 = acceleration_mps2
            previous_ego_timestamp_s = float(live_frame.timestamp_s)
            offroad = False
            if not args.samples_root and ego is not None and carla_map is not None:
                try:
                    import carla  # type: ignore

                    driving_waypoint = carla_map.get_waypoint(
                        ego.get_location(),
                        project_to_road=False,
                        lane_type=carla.LaneType.Driving,
                    )
                    offroad = driving_waypoint is None
                except (AttributeError, RuntimeError):
                    offroad = False
            if offroad:
                stats["offroad_frames"] += 1

            # ── D. WorldState ──────────────────────────────────────────────
            world_state = ws_builder.build(det_list, ego_tel)
            if world_state is not None:
                # WorldState's stable public schema predates heading tracking;
                # attach the CARLA-derived value for completion validation.
                setattr(world_state, "heading_error_rad", float(ego_tel.heading_error_rad))
                transition_candidate = _assist_lane_transition_completed(
                    world_state=world_state,
                    active_metadata=_assist_completion_metadata(
                        active_assist_metadata,
                        assist_maneuver_tracking_metadata,
                    ),
                    lateral_tolerance_m=float(args.agent_lane_center_tolerance_m),
                    heading_tolerance_rad=float(args.agent_lane_heading_tolerance_rad),
                )
                (
                    assist_lane_transition_stable_frames,
                    transition_stable,
                ) = _update_lane_transition_stability(
                    candidate=transition_candidate,
                    previous_frames=assist_lane_transition_stable_frames,
                    required_frames=_assist_lane_stable_required_frames(args),
                )
                if transition_stable:
                    if not assist_lane_change_completed:
                        assist_lane_change_completed = True
                        assist_lane_change_completed_timestamp_s = float(live_frame.timestamp_s)
                        assist_maneuver_failure_reason = None
                        assist_maneuver_failure_timestamp_s = None
                    active_assist_request = None
                    active_assist_intent = None
                    active_assist_hold_remaining = 0
                    active_assist_applied_frames = 0
                    active_assist_metadata = {}
                lane_change_overshoot = _lane_passed_target(
                    getattr(world_state, "ego_lane_id", ""),
                    agent_origin_lane_id,
                    agent_target_lane_id,
                )
                preferred_lane_for_frame = (
                    "current" if assist_lane_change_completed else agent_preferred_lane
                )
                lane_change_allowed = False
                lane_change_rule = "not_requested"
                if preferred_lane_for_frame in {"left", "right"}:
                    lane_change_allowed, lane_change_rule = _current_lane_change_permission(
                        carla_map,
                        ego,
                        preferred_lane_for_frame,
                        target_lane_id=agent_target_lane_id,
                    )
                setattr(world_state, "agent_preferred_lane", preferred_lane_for_frame)
                setattr(world_state, "agent_origin_lane_id", agent_origin_lane_id)
                setattr(world_state, "agent_target_lane_id", agent_target_lane_id)
                setattr(world_state, "lane_change_overshoot", lane_change_overshoot)
                setattr(world_state, "lane_change_permission", lane_change_allowed)
                setattr(world_state, "lane_change_rule", lane_change_rule)
                setattr(
                    world_state,
                    "agent_lane_change_decision_seen",
                    agent_lane_change_decision_seen,
                )
                setattr(
                    world_state,
                    "route_conflict_flags",
                    ["blocked_clear_adjacent_lane"]
                    if preferred_lane_for_frame in {"left", "right"} and lane_change_allowed
                    else [],
                )
                target_lateral_offset_m = _target_lane_lateral_offset_bev_m(
                    carla_map=carla_map,
                    ego=ego,
                    target_lane_id=agent_target_lane_id,
                )
                target_lane_risk = _target_lane_corridor_risk(
                    detections=list(det_list.detections or []),
                    ego_v_mps=float(ego_tel.ego_v_mps),
                    lateral_center_m=target_lateral_offset_m,
                    corridor_half_width_m=float(args.agent_target_corridor_half_width_m),
                    rear_clearance_m=float(args.agent_target_rear_clearance_m),
                    ttc_threshold_s=float(args.agent_risk_ttc_threshold),
                )
                setattr(
                    world_state,
                    "target_lane_risk_available",
                    bool(target_lane_risk["available"]),
                )
                setattr(
                    world_state,
                    "target_lane_corridor_clear",
                    bool(target_lane_risk["clear"]),
                )
                setattr(
                    world_state,
                    "target_lane_forward_ttc_s",
                    target_lane_risk["forward_ttc_s"],
                )
                setattr(
                    world_state,
                    "target_lane_rear_clearance_m",
                    target_lane_risk["rear_clearance_m"],
                )
                setattr(
                    world_state,
                    "target_lane_corridor_object_count",
                    int(target_lane_risk["object_count"]),
                )
                setattr(
                    world_state,
                    "target_lane_lateral_offset_m",
                    target_lane_risk["lateral_center_m"],
                )
                setattr(
                    world_state,
                    "target_lane_risk_source",
                    str(target_lane_risk["source"]),
                )
            route_info = (
                route_tracker.update(ego_tel.ego_location_xyz)
                if route_tracker is not None
                else {
                    "route_mode": "unavailable_watch_mode",
                    "route_progress_m": 0.0,
                    "route_length_m": None,
                    "route_completion_rate": None,
                    "route_completion_pct": None,
                    "distance_traveled_m": 0.0,
                }
            )
            _append_jsonl(
                ego_trace_path,
                {
                    "frame_id": live_frame.frame_id,
                    "frame_idx": frame_idx,
                    "timestamp_s": live_frame.timestamp_s,
                    "x": float(ego_tel.ego_location_xyz[0]),
                    "y": float(ego_tel.ego_location_xyz[1]),
                    "z": float(ego_tel.ego_location_xyz[2]),
                    "ego_v_mps": float(ego_tel.ego_v_mps),
                    "ego_a_mps2": float(ego_tel.ego_a_mps2),
                    "ego_lateral_error_m": float(ego_tel.ego_lateral_error_m),
                    "heading_error_rad": float(ego_tel.heading_error_rad),
                    "longitudinal_jerk_mps3": (
                        float(stats["longitudinal_jerk_samples_mps3"][-1])
                        if stats["longitudinal_jerk_samples_mps3"] else None
                    ),
                    "lidar_point_count": int(det_list.lidar_point_count),
                    "radar_point_count": int(det_list.radar_point_count),
                    "offroad": offroad,
                    "ego_lane_id": str(ego_tel.ego_lane_id),
                    **route_info,
                },
            )

            # ── E. Stage 9 Arbiter ─────────────────────────────────────────
            cmd = None
            control_source = "none"
            if arbiter is not None and world_state is not None:
                cmd = arbiter.step(world_state, sim_time_s=live_frame.timestamp_s)
                control_source = "baseline_or_arbiter"

            if (
                args.agent_control_mode == "assist"
                and world_state is not None
                and assist_agent is not None
                and assist_agent_worker is not None
                and assist_baseline is not None
                and assist_mpc is not None
            ):
                maneuver_failure_event: Optional[str] = None
                if active_assist_request is not None and active_assist_hold_remaining <= 0:
                    maneuver_failure_event = "maneuver_timeout"
                    assist_maneuver_failure_reason = maneuver_failure_event
                    assist_maneuver_failure_timestamp_s = float(live_frame.timestamp_s)
                    assist_agent_terminal_decision = True
                    assist_maneuver_tracking_metadata = {}
                    assist_lane_transition_stable_frames = 0
                if active_assist_hold_remaining <= 0:
                    active_assist_request = None
                    active_assist_intent = None
                    active_assist_applied_frames = 0
                    active_assist_metadata = {}
                setattr(
                    world_state,
                    "agent_active_maneuver",
                    active_assist_intent if active_assist_hold_remaining > 0 else None,
                )
                assist_baseline_req = assist_baseline.plan(world_state)
                assist_baseline_intent = str(getattr(assist_baseline_req, "tactical_intent", "keep_lane"))
                assist_ttc = float(
                    ws_builder._prev_detections and
                    _ttc_from_prev(ws_builder._prev_detections, ego_tel.ego_v_mps) or 99.0
                )
                assist_ttc_policy_safe, assist_ttc_policy_reason = _lane_change_ttc_safety(
                    world_state=world_state,
                    global_min_ttc_s=assist_ttc,
                    threshold_s=max(
                        1.5,
                        float(args.agent_risk_ttc_threshold) - 0.25,
                    ),
                    emergency_floor_s=float(args.agent_emergency_ttc_floor_s),
                )
                assist_result = assist_agent_worker.poll()
                worker_state = assist_agent_worker.stats()
                worker_busy = bool(worker_state.get("inflight") or worker_state.get("pending"))
                if assist_lane_change_completed:
                    assist_should_query, assist_trigger_reason = False, "lane_change_completed_post_cruise"
                elif assist_maneuver_failure_reason is not None:
                    assist_should_query, assist_trigger_reason = False, "maneuver_terminal_failure"
                elif active_assist_request is not None:
                    assist_should_query, assist_trigger_reason = False, "active_maneuver_in_progress"
                elif assist_agent_terminal_decision:
                    assist_should_query, assist_trigger_reason = False, "agent_decision_terminal"
                elif assist_result is not None:
                    # Consume and classify the response before scheduling a replacement.
                    assist_should_query, assist_trigger_reason = False, "agent_response_processing"
                elif worker_busy:
                    assist_should_query, assist_trigger_reason = False, "agent_request_inflight"
                else:
                    assist_should_query, assist_trigger_reason = _should_query_agent(
                        args=args,
                        frame_idx=frame_idx,
                        baseline_intent=assist_baseline_intent,
                        min_ttc_s=assist_ttc,
                        world_state=world_state,
                    )
                assist_should_query, assist_trigger_reason = _apply_agent_retry_cooldown(
                    requested=assist_should_query,
                    trigger_reason=assist_trigger_reason,
                    now_s=time.monotonic(),
                    last_failure_s=last_assist_agent_failure_wall_s,
                    cooldown_s=float(args.agent_retry_cooldown_s),
                )
                assist_should_query, assist_trigger_reason = _apply_agent_rate_limit(
                    args=args,
                    requested=assist_should_query,
                    trigger_reason=assist_trigger_reason,
                    now_s=time.monotonic(),
                    last_query_s=last_assist_agent_query_wall_s,
                )
                assist_should_query, assist_trigger_reason = _apply_agent_episode_limit(
                    requested=assist_should_query,
                    trigger_reason=assist_trigger_reason,
                    submitted_count=assist_agent_queries_submitted,
                    max_requests=int(args.agent_max_requests_per_episode),
                )
                assist_submit_outcome = None
                if assist_should_query:
                    assist_submit_outcome = assist_agent_worker.submit(
                        frame_id=int(live_frame.frame_id),
                        frame_idx=frame_idx,
                        sim_timestamp_s=float(live_frame.timestamp_s),
                        payload=assist_agent.build_intent_request(
                            world_state,
                            baseline_intent=assist_baseline_intent,
                            detections=det_list.detections,
                            sensor_input=_agent_sensor_input(det_list),
                        ),
                        context=_agent_request_context(
                            baseline_intent=assist_baseline_intent,
                            min_ttc_s=assist_ttc,
                            world_state=world_state,
                            route_info=route_info,
                            num_detections=len(det_list.detections),
                        ),
                    )
                    last_assist_agent_query_wall_s = time.monotonic()
                    assist_agent_queries_submitted += 1
                (
                    post_lane_change_settle_active,
                    post_lane_change_elapsed_s,
                ) = _post_lane_change_settle_state(
                    completion_timestamp_s=assist_lane_change_completed_timestamp_s,
                    current_timestamp_s=float(live_frame.timestamp_s),
                    settle_duration_s=float(args.agent_post_lane_change_settle_s),
                )
                assist_record: Dict[str, Any] = {
                    "frame_id": live_frame.frame_id,
                    "frame_idx": frame_idx,
                    "timestamp_s": live_frame.timestamp_s,
                    "agent_query_requested": assist_should_query,
                    "agent_queried": assist_submit_outcome is not None,
                    "agent_request_id": (
                        assist_submit_outcome.request_id
                        if assist_submit_outcome is not None else None
                    ),
                    "agent_pending_request_replaced_id": (
                        assist_submit_outcome.replaced_pending_request_id
                        if assist_submit_outcome is not None else None
                    ),
                    "agent_response_received": assist_result is not None,
                    "agent_trigger_reason": assist_trigger_reason,
                    "baseline_intent": assist_baseline_intent,
                    "assist_applied": False,
                    "assist_reject_reason": None,
                    "agent_max_requests_per_minute": float(args.agent_max_requests_per_minute),
                    "agent_max_requests_per_episode": int(args.agent_max_requests_per_episode),
                    "agent_requests_submitted_episode": int(assist_agent_queries_submitted),
                    "agent_valid_decisions_episode": int(assist_agent_valid_decisions),
                    "ego_v_mps": float(ego_tel.ego_v_mps),
                    "min_ttc_s": assist_ttc,
                    "assist_ttc_policy_safe": bool(assist_ttc_policy_safe),
                    "assist_ttc_policy_reason": str(assist_ttc_policy_reason),
                    "target_lane_risk_available": bool(
                        getattr(world_state, "target_lane_risk_available", False)
                    ),
                    "target_lane_corridor_clear": bool(
                        getattr(world_state, "target_lane_corridor_clear", False)
                    ),
                    "target_lane_forward_ttc_s": getattr(
                        world_state, "target_lane_forward_ttc_s", None
                    ),
                    "target_lane_rear_clearance_m": getattr(
                        world_state, "target_lane_rear_clearance_m", None
                    ),
                    "target_lane_corridor_object_count": int(
                        getattr(world_state, "target_lane_corridor_object_count", 0)
                    ),
                    "target_lane_lateral_offset_m": getattr(
                        world_state, "target_lane_lateral_offset_m", None
                    ),
                    "target_lane_risk_source": str(
                        getattr(world_state, "target_lane_risk_source", "unavailable")
                    ),
                    "route_progress_m": route_info.get("route_progress_m"),
                    "active_assist_maneuver": active_assist_intent,
                    "agent_origin_lane_id": agent_origin_lane_id,
                    "agent_target_lane_id": agent_target_lane_id,
                    "lane_change_overshoot": bool(getattr(world_state, "lane_change_overshoot", False)),
                    "lane_change_permission": bool(getattr(world_state, "lane_change_permission", False)),
                    "lane_change_rule": str(getattr(world_state, "lane_change_rule", "unknown")),
                    "lane_change_stability_candidate": bool(transition_candidate),
                    "lane_change_stable_frames": int(assist_lane_transition_stable_frames),
                    "lane_change_required_stable_frames": _assist_lane_stable_required_frames(args),
                    "post_lane_change_settle_active": bool(post_lane_change_settle_active),
                    "post_lane_change_elapsed_s": (
                        round(float(post_lane_change_elapsed_s), 6)
                        if post_lane_change_elapsed_s is not None else None
                    ),
                    "post_lane_change_handoff_to_baseline": False,
                    "maneuver_failure_reason": maneuver_failure_event,
                    "maneuver_failure_timestamp_s": (
                        float(live_frame.timestamp_s) if maneuver_failure_event else None
                    ),
                }
                if assist_result is not None:
                    assist_record.update(
                        {
                            "agent_response_request_id": assist_result.request.request_id,
                            "agent_request_frame_id": assist_result.request.frame_id,
                            "agent_request_frame_idx": assist_result.request.frame_idx,
                            "agent_response_frame_id": int(live_frame.frame_id),
                            "agent_response_frame_idx": frame_idx,
                            "agent_async_latency_ms": round(float(assist_result.latency_ms), 3),
                            "agent_call_latency_ms": round(float(assist_result.latency_ms), 3),
                            "agent_worker_error_type": assist_result.error_type,
                        }
                    )
                if assist_lane_change_completed:
                    active_assist_request = None
                    active_assist_intent = None
                    active_assist_hold_remaining = 0
                    active_assist_applied_frames = 0
                    active_assist_metadata = {}
                    if (
                        post_lane_change_settle_active
                        and assist_ttc >= max(3.0, float(args.agent_risk_ttc_threshold))
                    ):
                        cruise_req = _post_lane_change_cruise_request(
                            baseline_req=assist_baseline_req,
                            world_state=world_state,
                        )
                        cruise_speed_mps = 1.6 if bool(getattr(world_state, "lane_change_overshoot", False)) else 3.8
                        cmd = _map_lane_centering_control(
                            carla_map=carla_map,
                            ego=ego,
                            target_lane_id=agent_target_lane_id,
                            target_speed_mps=cruise_speed_mps,
                            source="MAP_LANE_CENTER_POST",
                            max_steer=0.45,
                        )
                        if cmd is None:
                            cmd = assist_mpc.execute(cruise_req)
                        control_source = "post_lane_change_centering"
                        assist_record.update(
                            {
                                "assist_applied": True,
                                "post_lane_change_cruise": True,
                                "assist_reject_reason": None,
                                "agent_intent": "keep_lane_after_lane_change",
                                "agent_validation_status": "post_lane_change_cruise",
                                "assist_request": {
                                    "tactical_intent": str(getattr(cruise_req, "tactical_intent", "")),
                                    "target_v_desired_mps": float(getattr(cruise_req, "target_v_desired_mps", 0.0)),
                                    "target_lane_id": getattr(cruise_req, "target_lane_id", None),
                                },
                                "applied_command": {
                                    "throttle": float(getattr(cmd, "throttle", 0.0)),
                                    "steer": float(getattr(cmd, "steer", 0.0)),
                                    "brake": float(getattr(cmd, "brake", 0.0)),
                                    "source": str(getattr(cmd, "source", "unknown")),
                                    "target_lane_lateral_error_m": getattr(
                                        cmd, "target_lane_lateral_error_m", None
                                    ),
                                    "target_lane_heading_error_rad": getattr(
                                        cmd, "target_lane_heading_error_rad", None
                                    ),
                                    "steering_phase": getattr(cmd, "steering_phase", None),
                                    "target_lane_reached": bool(
                                        getattr(cmd, "target_lane_reached", False)
                                    ),
                                },
                            }
                        )
                    else:
                        handoff_reason = (
                            "post_lane_change_low_ttc"
                            if post_lane_change_settle_active
                            else "post_lane_change_settle_complete"
                        )
                        control_source = "baseline_post_lane_change_handoff"
                        assist_record.update(
                            {
                                "post_lane_change_handoff_to_baseline": True,
                                "post_lane_change_handoff_reason": handoff_reason,
                                "agent_validation_status": "post_lane_change_handoff",
                            }
                        )
                        if post_lane_change_settle_active:
                            assist_record["assist_reject_reason"] = handoff_reason
                preserve_active_after_agent_fallback = bool(
                    assist_result is not None
                    and active_assist_request is not None
                    and (
                        bool(getattr(assist_result, "error_type", None))
                        or bool(getattr(assist_result.intent_record, "fallback_to_baseline", False))
                    )
                    and _can_continue_active_assist(
                        args=args,
                        active_request=active_assist_request,
                        active_metadata=active_assist_metadata,
                        baseline_intent=assist_baseline_intent,
                        world_state=world_state,
                        min_ttc_s=assist_ttc,
                        lane_change_completed=assist_lane_change_completed,
                    )
                )
                accepted_new_assist = False
                if preserve_active_after_agent_fallback:
                    # A new API request is advisory. Do not tear down a maneuver
                    # that was previously accepted by the safety gate merely
                    # because this later request timed out or could not be parsed.
                    # The existing request is still revalidated every frame and
                    # expires through active_assist_hold_remaining.
                    assist_record.update(
                        {
                            "agent_response_fresh": False,
                            "agent_response_freshness_reason": "agent_fallback_preserved_active_maneuver",
                            "agent_validation_status": "fallback_preserved_active_maneuver",
                            "agent_fallback_to_baseline": True,
                            "agent_fallback_reason": (
                                str(getattr(assist_result, "error_type", ""))
                                or str(
                                    (getattr(assist_result.intent_record, "provenance", {}) or {}).get(
                                        "fallback_reason", "agent_fallback"
                                    )
                                )
                            ),
                        }
                    )
                elif assist_lane_change_completed:
                    active_assist_request = None
                    active_assist_intent = None
                    active_assist_hold_remaining = 0
                    active_assist_applied_frames = 0
                    active_assist_metadata = {}
                elif assist_result is not None:
                    intent_record = assist_result.intent_record
                    response_fresh, freshness_reason, response_age_s = _agent_response_freshness(
                        args=args,
                        result=assist_result,
                        current_timestamp_s=float(live_frame.timestamp_s),
                        current_world_state=world_state,
                        current_min_ttc_s=assist_ttc,
                    )
                    raw_agent_intent = (
                        str(getattr(intent_record, "tactical_intent", assist_baseline_intent))
                        if intent_record is not None
                        else assist_baseline_intent
                    )
                    agent_intent = _merge_active_and_new_agent_intent(active_assist_intent, raw_agent_intent)
                    if response_fresh:
                        assist_allowed, assist_reason = _agent_assist_allowed(
                            args=args,
                            intent_record=intent_record,
                            baseline_intent=assist_baseline_intent,
                            world_state=world_state,
                            lane_change_completed=assist_lane_change_completed,
                        )
                    else:
                        assist_allowed, assist_reason = False, freshness_reason
                    response_is_fallback = bool(
                        assist_result.error_type
                        or intent_record is None
                        or bool(getattr(intent_record, "fallback_to_baseline", False))
                        or not response_fresh
                    )
                    if response_is_fallback:
                        last_assist_agent_failure_wall_s = time.monotonic()
                    else:
                        assist_agent_valid_decisions += 1
                        assist_agent_terminal_decision = True
                        agent_lane_change_decision_seen = True
                    provenance = getattr(intent_record, "provenance", {}) or {}
                    assist_record.update(
                        {
                            "agent_response_age_s": round(response_age_s, 6),
                            "agent_response_fresh": response_fresh,
                            "agent_response_freshness_reason": freshness_reason,
                            "agent_intent": agent_intent,
                            "agent_confidence": (
                                float(getattr(intent_record, "confidence", 0.0))
                                if intent_record is not None else 0.0
                            ),
                            "agent_validation_status": (
                                str(getattr(intent_record, "validation_status", "unavailable"))
                                if intent_record is not None else "unavailable"
                            ),
                            "agent_fallback_to_baseline": (
                                bool(getattr(intent_record, "fallback_to_baseline", True))
                                if intent_record is not None else True
                            ),
                            "agent_model_id": (
                                str(getattr(intent_record, "model_id", "unknown"))
                                if intent_record is not None else "unknown"
                            ),
                            "agent_backend_model_id": provenance.get("backend_model_id"),
                            "agent_reason_tags": (
                                list(getattr(intent_record, "reason_tags", []) or [])
                                if intent_record is not None else []
                            ),
                            "agent_raw_intent": provenance.get("raw_intent_received"),
                            "agent_fallback_reason": (
                                provenance.get("fallback_reason")
                                or getattr(assist_result, "error_type", None)
                            ),
                            "agent_call_latency_ms": (
                                provenance.get("call_latency_ms")
                                if provenance.get("call_latency_ms") is not None
                                else round(float(assist_result.latency_ms), 3)
                            ),
                            "agent_prompt_token_count": provenance.get("prompt_token_count"),
                            "agent_completion_token_count": provenance.get("completion_token_count"),
                            "agent_total_token_count": provenance.get("total_token_count"),
                            "agent_api_attempt_count": provenance.get("api_attempt_count"),
                            "agent_api_payload_variant": provenance.get("api_payload_variant"),
                            "agent_prompt_context_schema": provenance.get("prompt_context_schema"),
                            "agent_prompt_context_object_count": provenance.get(
                                "prompt_context_object_count"
                            ),
                            "assist_reject_reason": None if assist_allowed else assist_reason,
                        }
                    )
                    if assist_allowed:
                        preserve_progress = _same_lane_change_family(active_assist_intent, agent_intent)
                        assist_req = _trajectory_request_from_agent_intent(
                            baseline_req=assist_baseline_req,
                            agent_intent=agent_intent,
                            world_state=world_state,
                        )
                        active_assist_request = assist_req
                        active_assist_intent = agent_intent
                        active_assist_hold_remaining = _assist_hold_frames(args)
                        active_assist_applied_frames = (
                            max(1, active_assist_applied_frames)
                            if preserve_progress else 1
                        )
                        active_assist_metadata = {
                            "agent_confidence": float(getattr(intent_record, "confidence", 0.0))
                            if intent_record is not None else 0.0,
                            "agent_model_id": str(getattr(intent_record, "model_id", "unknown"))
                            if intent_record is not None else "unknown",
                            "agent_reason_tags": list(getattr(intent_record, "reason_tags", []) or [])
                            if intent_record is not None else [],
                            "origin_lane_id": str(getattr(world_state, "ego_lane_id", "") or ""),
                            "target_lane_id": str(agent_target_lane_id or ""),
                        }
                        assist_lane_transition_stable_frames = 0
                        assist_maneuver_failure_reason = None
                        assist_maneuver_failure_timestamp_s = None
                        if not preserve_progress or not assist_maneuver_tracking_metadata:
                            assist_maneuver_tracking_metadata = dict(active_assist_metadata)
                        active_assist_intent = _retune_active_assist_request(
                            request=active_assist_request,
                            world_state=world_state,
                            applied_frames=active_assist_applied_frames,
                            args=args,
                        )
                        cmd = _map_lane_centering_control(
                            carla_map=carla_map,
                            ego=ego,
                            target_lane_id=agent_target_lane_id,
                            target_speed_mps=float(getattr(assist_req, "target_v_desired_mps", 2.0)),
                            source="MAP_LANE_CENTER_ASSIST",
                            max_steer=0.46,
                            cross_lane_max_steer=float(args.agent_cross_lane_max_steer),
                            cross_lane_min_steer=float(args.agent_cross_lane_min_steer),
                            lane_change_assist=True,
                        )
                        if cmd is None:
                            cmd = assist_mpc.execute(assist_req)
                        control_source = "agent_lane_center_assist"
                        assist_record["assist_applied"] = True
                        assist_record["assist_request"] = {
                            "tactical_intent": active_assist_intent,
                            "target_v_desired_mps": float(getattr(assist_req, "target_v_desired_mps", 0.0)),
                            "target_lane_id": getattr(assist_req, "target_lane_id", None),
                        }
                        assist_record["applied_command"] = {
                            "throttle": float(getattr(cmd, "throttle", 0.0)),
                            "steer": float(getattr(cmd, "steer", 0.0)),
                            "brake": float(getattr(cmd, "brake", 0.0)),
                            "source": str(getattr(cmd, "source", "unknown")),
                            "target_lane_lateral_error_m": getattr(
                                cmd, "target_lane_lateral_error_m", None
                            ),
                            "target_lane_heading_error_rad": getattr(
                                cmd, "target_lane_heading_error_rad", None
                            ),
                            "steering_phase": getattr(cmd, "steering_phase", None),
                            "target_lane_reached": bool(
                                getattr(cmd, "target_lane_reached", False)
                            ),
                        }
                        assist_record["agent_intent"] = active_assist_intent
                        accepted_new_assist = True
                    else:
                        active_assist_request = None
                        active_assist_intent = None
                        active_assist_hold_remaining = 0
                        active_assist_applied_frames = 0
                        active_assist_metadata = {}
                can_continue_active = bool(
                    active_assist_request is not None
                    and _can_continue_active_assist(
                        args=args,
                        active_request=active_assist_request,
                        active_metadata=active_assist_metadata,
                        baseline_intent=assist_baseline_intent,
                        world_state=world_state,
                        min_ttc_s=assist_ttc,
                        lane_change_completed=assist_lane_change_completed,
                    )
                )
                continuation_stop_reason = _active_assist_stop_reason(
                    args=args,
                    active_request=active_assist_request,
                    baseline_intent=assist_baseline_intent,
                    world_state=world_state,
                    min_ttc_s=assist_ttc,
                    lane_change_completed=assist_lane_change_completed,
                ) if active_assist_request is not None else None
                lifecycle_action = _assist_lifecycle_action(
                    accepted_new_assist=accepted_new_assist,
                    preserve_active_after_fallback=preserve_active_after_agent_fallback,
                    response_received=assist_result is not None,
                    can_continue_active=can_continue_active,
                )
                if lifecycle_action == "accepted":
                    # The first command was produced above. Keep the accepted
                    # maneuver intact so subsequent frames can continue it.
                    pass
                elif lifecycle_action == "continue":
                    active_assist_hold_remaining = max(0, active_assist_hold_remaining - 1)
                    active_assist_applied_frames += 1
                    active_assist_intent = _retune_active_assist_request(
                        request=active_assist_request,
                        world_state=world_state,
                        applied_frames=active_assist_applied_frames,
                        args=args,
                    )
                    cmd = _map_lane_centering_control(
                        carla_map=carla_map,
                        ego=ego,
                        target_lane_id=agent_target_lane_id,
                        target_speed_mps=float(getattr(active_assist_request, "target_v_desired_mps", 2.5)),
                        source="MAP_LANE_CENTER_ASSIST_HOLD",
                        max_steer=0.46,
                        cross_lane_max_steer=float(args.agent_cross_lane_max_steer),
                        cross_lane_min_steer=float(args.agent_cross_lane_min_steer),
                        lane_change_assist=True,
                    )
                    if cmd is None:
                        cmd = assist_mpc.execute(active_assist_request)
                    control_source = "agent_lane_center_assist_hold"
                    assist_record.update(
                        {
                            "assist_applied": True,
                            "assist_continued": True,
                            "assist_reject_reason": None,
                            "agent_intent": active_assist_intent,
                            "agent_confidence": float(active_assist_metadata.get("agent_confidence", 0.0)),
                            "agent_model_id": str(active_assist_metadata.get("agent_model_id", "unknown")),
                            "agent_reason_tags": list(active_assist_metadata.get("agent_reason_tags", []) or []),
                            "agent_validation_status": "continued_valid",
                            "assist_request": {
                                "tactical_intent": str(getattr(active_assist_request, "tactical_intent", "")),
                                "target_v_desired_mps": float(getattr(active_assist_request, "target_v_desired_mps", 0.0)),
                                "target_lane_id": getattr(active_assist_request, "target_lane_id", None),
                            },
                            "applied_command": {
                                "throttle": float(getattr(cmd, "throttle", 0.0)),
                                "steer": float(getattr(cmd, "steer", 0.0)),
                                "brake": float(getattr(cmd, "brake", 0.0)),
                                "source": str(getattr(cmd, "source", "unknown")),
                                "target_lane_lateral_error_m": getattr(
                                    cmd, "target_lane_lateral_error_m", None
                                ),
                                "target_lane_heading_error_rad": getattr(
                                    cmd, "target_lane_heading_error_rad", None
                                ),
                                "steering_phase": getattr(cmd, "steering_phase", None),
                                "target_lane_reached": bool(
                                    getattr(cmd, "target_lane_reached", False)
                                ),
                            },
                        }
                    )
                else:
                    if (
                        active_assist_request is not None
                        and not assist_lane_change_completed
                        and continuation_stop_reason is not None
                        and continuation_stop_reason.startswith("safety_abort_")
                    ):
                        assist_maneuver_failure_reason = continuation_stop_reason
                        assist_maneuver_failure_timestamp_s = float(live_frame.timestamp_s)
                        assist_maneuver_tracking_metadata = {}
                        assist_lane_transition_stable_frames = 0
                        assist_agent_terminal_decision = True
                        assist_record.update(
                            {
                                "maneuver_failure_reason": continuation_stop_reason,
                                "maneuver_failure_timestamp_s": float(live_frame.timestamp_s),
                                "assist_reject_reason": continuation_stop_reason,
                            }
                        )
                    active_assist_request = None
                    active_assist_intent = None
                    active_assist_hold_remaining = 0
                    active_assist_applied_frames = 0
                    active_assist_metadata = {}
                # Persist the physical completion event on every row after it
                # is detected, including the frame where lifecycle cleanup
                # happens later in this loop.
                assist_record.update(
                    {
                        "lane_change_completed": bool(assist_lane_change_completed),
                        "lane_change_completion_timestamp_s": assist_lane_change_completed_timestamp_s,
                        "assist_maneuver_phase": _assist_maneuver_phase(
                            intent=active_assist_intent,
                            completed=assist_lane_change_completed,
                            failure_reason=assist_maneuver_failure_reason,
                        ),
                        "agent_valid_decisions_episode": int(assist_agent_valid_decisions),
                        "maneuver_failure_reason": (
                            assist_record.get("maneuver_failure_reason")
                            or assist_maneuver_failure_reason
                        ),
                        "maneuver_failure_timestamp_s": (
                            assist_record.get("maneuver_failure_timestamp_s")
                            or assist_maneuver_failure_timestamp_s
                        ),
                    }
                )
                assist_log.append(assist_record)

            if cmd is not None and not args.samples_root and ego is not None:
                _apply_actuator_command(ego, cmd)

            # ── F. Compare mode: side-by-side Agent vs Baseline ───────────
            if (
                compare_agent is not None
                and compare_agent_worker is not None
                and compare_baseline is not None
                and world_state is not None
            ):
                try:
                    bl_req = compare_baseline.plan(world_state)
                    baseline_intent = str(getattr(bl_req, "tactical_intent", "keep_lane"))
                    ttc_val = float(
                        ws_builder._prev_detections and
                        _ttc_from_prev(ws_builder._prev_detections, ego_tel.ego_v_mps) or 99.0
                    )

                    compare_result = compare_agent_worker.poll()
                    if compare_result is not None:
                        request_context = compare_result.request.context
                        request_baseline_intent = str(
                            request_context.get("baseline_intent", "keep_lane")
                        )
                        intent_record = compare_result.intent_record
                        if intent_record is not None:
                            agent_intent = str(
                                getattr(intent_record, "tactical_intent", request_baseline_intent)
                            )
                            agent_confidence = float(getattr(intent_record, "confidence", 0.0))
                            reason_tags = list(getattr(intent_record, "reason_tags", []) or [])
                            agent_reasoning = ",".join(str(tag) for tag in reason_tags)
                            disagreement_useful = bool(
                                getattr(intent_record, "disagreement_useful", False)
                            )
                            agent_validation_status = str(
                                getattr(intent_record, "validation_status", "unknown")
                            )
                            agent_fallback_to_baseline = bool(
                                getattr(intent_record, "fallback_to_baseline", False)
                            )
                            agent_model_id = str(getattr(intent_record, "model_id", "unknown"))
                            provenance = getattr(intent_record, "provenance", {}) or {}
                            agent_raw_intent = provenance.get("raw_intent_received")
                        else:
                            agent_intent = request_baseline_intent
                            agent_confidence = 0.0
                            agent_reasoning = "agent_call_failed"
                            disagreement_useful = False
                            agent_validation_status = "unavailable"
                            agent_fallback_to_baseline = True
                            agent_model_id = "unknown"
                            provenance = {}
                            agent_raw_intent = None

                        response_age_s = max(
                            0.0,
                            float(live_frame.timestamp_s)
                            - float(compare_result.request.sim_timestamp_s),
                        )
                        agrees = request_baseline_intent == agent_intent
                        record = {
                            "frame_id": compare_result.request.frame_id,
                            "frame_idx": compare_result.request.frame_idx,
                            "timestamp_s": compare_result.request.sim_timestamp_s,
                            "response_frame_id": int(live_frame.frame_id),
                            "response_frame_idx": frame_idx,
                            "response_age_s": round(response_age_s, 6),
                            "stale_response": response_age_s > float(args.agent_response_max_age_s),
                            "agent_request_id": compare_result.request.request_id,
                            "agent_queried": True,
                            "agent_trigger_reason": str(
                                request_context.get("trigger_reason", "unknown")
                            ),
                            "ego_v_mps": float(request_context.get("ego_v_mps", 0.0)),
                            "min_ttc_s": float(request_context.get("min_ttc_s", 99.0)),
                            "route_completion_rate": request_context.get("route_completion_rate"),
                            "route_progress_m": request_context.get("route_progress_m"),
                            "num_detections": int(request_context.get("num_detections", 0)),
                            "baseline_intent": request_baseline_intent,
                            "agent_intent": agent_intent,
                            "agent_confidence": agent_confidence,
                            "agent_reasoning": agent_reasoning,
                            "agent_validation_status": agent_validation_status,
                            "agent_fallback_to_baseline": agent_fallback_to_baseline,
                            "agent_model_id": agent_model_id,
                            "agent_backend_model_id": provenance.get("backend_model_id"),
                            "agent_raw_intent": agent_raw_intent,
                            "agent_prompt_token_count": provenance.get("prompt_token_count"),
                            "agent_completion_token_count": provenance.get("completion_token_count"),
                            "agent_total_token_count": provenance.get("total_token_count"),
                            "agent_prompt_context_schema": provenance.get("prompt_context_schema"),
                            "agent_prompt_context_object_count": provenance.get(
                                "prompt_context_object_count"
                            ),
                            "agrees": agrees,
                            "disagreement_useful": disagreement_useful and not agrees,
                            "compare_latency_ms": round(float(compare_result.latency_ms), 1),
                            "worker_error_type": compare_result.error_type,
                        }
                        compare_log.append(record)

                        if not agrees:
                            LOGGER.info(
                                "[COMPARE] request_frame=%d response_frame=%d baseline=%s "
                                "agent=%s conf=%.2f ttc=%.1f age=%.2fs reason=%s",
                                compare_result.request.frame_id,
                                live_frame.frame_id,
                                request_baseline_intent,
                                agent_intent,
                                agent_confidence,
                                float(request_context.get("min_ttc_s", 99.0)),
                                response_age_s,
                                agent_reasoning,
                            )

                    should_query_agent, trigger_reason = _should_query_agent(
                        args=args,
                        frame_idx=frame_idx,
                        baseline_intent=baseline_intent,
                        min_ttc_s=ttc_val,
                        world_state=world_state,
                    )
                    should_query_agent, trigger_reason = _apply_agent_rate_limit(
                        args=args,
                        requested=should_query_agent,
                        trigger_reason=trigger_reason,
                        now_s=time.monotonic(),
                        last_query_s=last_compare_agent_query_wall_s,
                    )
                    should_query_agent, trigger_reason = _apply_agent_episode_limit(
                        requested=should_query_agent,
                        trigger_reason=trigger_reason,
                        submitted_count=compare_agent_queries_submitted,
                        max_requests=int(args.agent_max_requests_per_episode),
                    )
                    if not should_query_agent:
                        compare_skipped_frames += 1
                    else:
                        context = _agent_request_context(
                            baseline_intent=baseline_intent,
                            min_ttc_s=ttc_val,
                            world_state=world_state,
                            route_info=route_info,
                            num_detections=len(det_list.detections),
                        )
                        context["trigger_reason"] = trigger_reason
                        compare_agent_worker.submit(
                            frame_id=int(live_frame.frame_id),
                            frame_idx=frame_idx,
                            sim_timestamp_s=float(live_frame.timestamp_s),
                            payload=compare_agent.build_intent_request(
                                world_state,
                                baseline_intent=baseline_intent,
                                detections=det_list.detections,
                                sensor_input=_agent_sensor_input(det_list),
                            ),
                            context=context,
                        )
                        last_compare_agent_query_wall_s = time.monotonic()
                        compare_agent_queries_submitted += 1
                except Exception as exc:
                    LOGGER.debug("Compare-mode error frame=%d: %s", frame_idx, exc)

            if video_recorder is not None:
                _draw_scenario_actor_labels(
                    world,
                    scenario_manifest,
                    life_time_s=max(float(args.delta_t) * 2.0, 0.15),
                )
                min_ttc_s = float(
                    ws_builder._prev_detections and
                    _ttc_from_prev(ws_builder._prev_detections, ego_tel.ego_v_mps) or 99.0
                )
                wrote_video = video_recorder.write_frame(
                    carla_frame=int(live_frame.frame_id),
                    overlay_lines=_stage10_overlay_lines(
                        frame_idx=frame_idx,
                        live_frame=live_frame,
                        ego_tel=ego_tel,
                        route_info=route_info,
                        det_count=len(det_list.detections),
                        control_source=control_source,
                        agent_mode=str(args.agent_mode),
                        agent_control_mode=str(args.agent_control_mode),
                        min_ttc_s=min_ttc_s,
                        lane_change_rule=str(getattr(world_state, "lane_change_rule", "unknown")) if world_state is not None else "unknown",
                    ),
                )
                if not wrote_video:
                    LOGGER.warning(
                        "Stage10 video recorder missed carla_frame=%d frame_idx=%d",
                        int(live_frame.frame_id),
                        frame_idx,
                    )

            stats["frames"] += 1
            elapsed_ms = (time.monotonic() - t_tick) * 1000
            stats["tick_latency_samples_ms"].append(float(elapsed_ms))

            if frame_idx % 20 == 0:
                avg_bev = stats["bev_ms_sum"] / max(1, stats["frames"])
                LOGGER.info(
                    "[%4d/%4d] frame=%d  dets=%2d  bev=%.0f ms  tick=%.0f ms  v=%.1f m/s  ttc=%.1f s",
                    frame_idx, args.max_frames,
                    live_frame.frame_id,
                    len(det_list.detections),
                    det_list.inference_time_ms,
                    elapsed_ms,
                    ego_tel.ego_v_mps,
                    ws_builder._prev_detections and
                    _ttc_from_prev(ws_builder._prev_detections, ego_tel.ego_v_mps) or 99.0,
                )

    except Exception as exc:
        LOGGER.exception("Unexpected error in main loop: %s", exc)
        stats["errors"] += 1
        return 1
    finally:
        if compare_agent_worker is not None:
            compare_agent_worker.close()
            compare_agent_worker_stats = compare_agent_worker.stats()
        if assist_agent_worker is not None:
            assist_agent_worker.close()
            assist_agent_worker_stats = assist_agent_worker.stats()
        sensor_source.stop()
        if video_recorder is not None:
            try:
                _write_json(log_dir / "stage10_video_manifest.json", video_recorder.manifest())
                LOGGER.info("Stage10 video manifest -> %s", log_dir / "stage10_video_manifest.json")
            finally:
                video_recorder.close()
        if collision_monitor is not None:
            collision_monitor.stop()
            for event in collision_monitor.events():
                _append_jsonl(collision_events_path, event)
            collision_events_path.touch(exist_ok=True)
            for episode in collision_monitor.counted_episodes():
                _append_jsonl(collision_episodes_path, episode)
            collision_episodes_path.touch(exist_ok=True)
        if lane_invasion_monitor is not None:
            lane_invasion_monitor.stop()
            for event in lane_invasion_monitor.events():
                _append_jsonl(lane_invasion_events_path, event)
            lane_invasion_events_path.touch(exist_ok=True)
        driving_metrics = _build_driving_metrics(
            args=args,
            stats=stats,
            route_tracker=route_tracker,
            collision_monitor=collision_monitor,
            lane_invasion_monitor=lane_invasion_monitor,
            maneuver_start_timestamp_s=_assist_maneuver_start_timestamp(assist_log),
            maneuver_completion_timestamp_s=assist_lane_change_completed_timestamp_s,
        )
        _write_json(log_dir / "stage10_driving_metrics.json", driving_metrics)
        if not args.samples_root and ego is not None and ego_spawned_by_stage10:
            try:
                ego.destroy()
            except Exception:
                pass

        # ── Dump compare-mode evaluation report ──────────────────────────
        if compare_log:
            total = len(compare_log)
            agreements = sum(1 for r in compare_log if r["agrees"])
            disagreements = total - agreements
            useful_disagreements = sum(1 for r in compare_log if r["disagreement_useful"])
            agent_latencies = [r["compare_latency_ms"] for r in compare_log]

            # TTC-based analysis: how does agent behave in danger zones?
            low_ttc_frames = [r for r in compare_log if r["min_ttc_s"] < 3.0]
            low_ttc_disagree = [r for r in low_ttc_frames if not r["agrees"]]
            cautious_intents = {"follow", "slow_down", "stop", "yield", "stop_before_obstacle"}
            low_ttc_agent_cautious = [
                r for r in low_ttc_frames if str(r.get("agent_intent")) in cautious_intents
            ]
            low_ttc_baseline_cautious = [
                r for r in low_ttc_frames if str(r.get("baseline_intent")) in cautious_intents
            ]
            fallback_frames = [r for r in compare_log if bool(r.get("agent_fallback_to_baseline"))]
            stale_compare_responses = [r for r in compare_log if bool(r.get("stale_response"))]
            compare_response_ages = [
                float(r["response_age_s"])
                for r in compare_log
                if r.get("response_age_s") is not None
            ]
            validation_counts: Dict[str, int] = {}
            for r in compare_log:
                status = str(r.get("agent_validation_status", "unknown"))
                validation_counts[status] = validation_counts.get(status, 0) + 1

            evaluation = {
                "schema_version": "stage10_agent_live_evaluation_v2",
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                "stage": "10_live_bridge",
                "mode": args.agent_mode,
                "map": args.map,
                "random_seed": int(args.seed),
                "total_frames": total,
                "sim_frames": int(stats.get("frames", 0)),
                "agent_trigger_mode": args.agent_trigger_mode,
                "agent_compare_stride": int(args.agent_compare_stride),
                "agent_risk_ttc_threshold": float(args.agent_risk_ttc_threshold),
                "agent_queried_frames": total,
                "agent_submitted_requests": int(compare_agent_worker_stats.get("submitted", 0)),
                "agent_skipped_frames": int(compare_skipped_frames),
                "stale_response_count": len(stale_compare_responses),
                "stale_response_rate": round(
                    len(stale_compare_responses) / max(total, 1), 4
                ),
                "async_worker": compare_agent_worker_stats,
                "query_ratio": round(total / max(int(stats.get("frames", 0)), 1), 4)
                if int(stats.get("frames", 0)) > 0 else None,
                "agreement_frames": agreements,
                "disagreement_frames": disagreements,
                "agreement_rate": round(agreements / max(total, 1), 4),
                "disagreement_rate": round(disagreements / max(total, 1), 4),
                "useful_disagreement_count": useful_disagreements,
                "useful_disagreement_rate": round(
                    useful_disagreements / max(disagreements, 1), 4
                ) if disagreements > 0 else None,
                "agent_fallback_frames": len(fallback_frames),
                "agent_fallback_rate": round(len(fallback_frames) / max(total, 1), 4),
                "agent_validation_status_counts": validation_counts,
                "low_ttc_analysis": {
                    "total_low_ttc_frames": len(low_ttc_frames),
                    "disagreements_in_low_ttc": len(low_ttc_disagree),
                    "disagreement_rate_in_low_ttc": round(
                        len(low_ttc_disagree) / max(len(low_ttc_frames), 1), 4
                    ) if low_ttc_frames else None,
                    "agent_cautious_frames": len(low_ttc_agent_cautious),
                    "agent_cautious_rate": round(
                        len(low_ttc_agent_cautious) / max(len(low_ttc_frames), 1), 4
                    ) if low_ttc_frames else None,
                    "baseline_cautious_frames": len(low_ttc_baseline_cautious),
                    "baseline_cautious_rate": round(
                        len(low_ttc_baseline_cautious) / max(len(low_ttc_frames), 1), 4
                    ) if low_ttc_frames else None,
                },
                "latency": {
                    "mean_compare_ms": round(sum(agent_latencies) / max(len(agent_latencies), 1), 1),
                    "p50_compare_ms": (
                        round(float(_percentile(agent_latencies, 50.0)), 1) if agent_latencies else None
                    ),
                    "p95_compare_ms": (
                        round(float(_percentile(agent_latencies, 95.0)), 1) if agent_latencies else None
                    ),
                    "max_compare_ms": round(max(agent_latencies), 1) if agent_latencies else 0,
                    "min_compare_ms": round(min(agent_latencies), 1) if agent_latencies else 0,
                    "simulation_step_budget_ms": round(float(args.delta_t) * 1000.0, 1),
                    "over_step_budget_rate": round(
                        sum(value > float(args.delta_t) * 1000.0 for value in agent_latencies)
                        / max(len(agent_latencies), 1),
                        4,
                    ),
                    "mean_response_age_s": (
                        round(sum(compare_response_ages) / len(compare_response_ages), 6)
                        if compare_response_ages else None
                    ),
                    "p95_response_age_s": (
                        round(float(_percentile(compare_response_ages, 95.0)), 6)
                        if compare_response_ages else None
                    ),
                    "control_loop": (
                        (driving_metrics.get("runtime") or {}).get("control_loop_latency")
                    ),
                },
                "intent_distribution": {
                    "baseline": {},
                    "agent": {},
                },
                "perception_summary": {
                    "total_detections": stats["total_det"],
                    "avg_detections_per_frame": round(stats["total_det"] / max(total, 1), 1),
                    "avg_bev_inference_ms": round(stats["bev_ms_sum"] / max(total, 1), 1),
                },
                "driving_metrics": driving_metrics,
                "frame_log": compare_log,
            }

            # Count intent distributions
            for r in compare_log:
                bi = r["baseline_intent"]
                ai = r["agent_intent"]
                evaluation["intent_distribution"]["baseline"][bi] = (
                    evaluation["intent_distribution"]["baseline"].get(bi, 0) + 1
                )
                evaluation["intent_distribution"]["agent"][ai] = (
                    evaluation["intent_distribution"]["agent"].get(ai, 0) + 1
                )

            eval_path = log_dir / "stage10_agent_live_evaluation.json"
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            with open(eval_path, "w") as f:
                json.dump(evaluation, f, indent=2, default=str)
            LOGGER.info("Compare-mode evaluation saved → %s", eval_path)
            LOGGER.info("  Agreement rate   : %.1f%% (%d/%d)", agreements / max(total, 1) * 100, agreements, total)
            LOGGER.info("  Disagreement rate: %.1f%% (%d/%d)", disagreements / max(total, 1) * 100, disagreements, total)
            if disagreements > 0:
                LOGGER.info("  Useful disagree  : %.1f%% (%d/%d)",
                            useful_disagreements / max(disagreements, 1) * 100,
                            useful_disagreements, disagreements)
            if low_ttc_frames:
                LOGGER.info("  Low-TTC frames   : %d, agent cautious: %d",
                            len(low_ttc_frames), len(low_ttc_disagree))

        if assist_log:
            assist_eval = _summarize_assist_log(
                assist_log,
                stats,
                args,
                worker_stats=assist_agent_worker_stats,
            )
            assist_path = log_dir / "stage10_agent_assist_evaluation.json"
            assist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(assist_path, "w") as f:
                json.dump(assist_eval, f, indent=2, default=str)
            LOGGER.info(
                "Agent-assist evaluation saved -> %s (applied=%d/%d)",
                assist_path,
                assist_eval.get("assist_applied_frames", 0),
                assist_eval.get("sim_frames", 0),
            )

        LOGGER.info("=" * 60)
        LOGGER.info("Stage 10 Live Bridge finished.")
        LOGGER.info("  Frames     : %d", stats["frames"])
        LOGGER.info("  Total detections : %d", stats["total_det"])
        if stats["frames"]:
            LOGGER.info("  Avg BEV ms : %.1f", stats["bev_ms_sum"] / stats["frames"])
        LOGGER.info("  Route completion : %s%%", driving_metrics.get("route_completion_pct"))
        LOGGER.info("  Collisions  : %d", driving_metrics.get("collision_count", 0))
        LOGGER.info("  Scenario success : %s", driving_metrics.get("scenario_success"))
        LOGGER.info("  Driving metrics  → %s", log_dir / "stage10_driving_metrics.json")
        LOGGER.info("  Errors     : %d", stats["errors"])
        LOGGER.info("=" * 60)

    return 0 if stats["errors"] == 0 else 1


def _ttc_from_prev(dets, ego_v):
    from carla_bevfusion_stage1.world_state_builder import _estimate_ttc
    return _estimate_ttc(dets, ego_v)


if __name__ == "__main__":
    sys.exit(run(_parse_args()))
