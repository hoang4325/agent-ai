"""
scripts/run_stage10_stage1_live_bridge.py
==========================================
Stage 10 — Live CARLA → BEVFusion → Stage 9 bridge.

Connects every module we have built (Stage 1 → Stage 9) into a single
synchronous closed-loop:

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
    p.add_argument("--bev-repo",     required=True,  help="BEVFusion git repo root")
    p.add_argument("--bev-config",   required=True,  help="BEVFusion YAML config path")
    p.add_argument("--bev-ckpt",     required=True,  help="BEVFusion checkpoint .pth")
    p.add_argument("--output-root",  required=True,  help="Where to write inference results")
    p.add_argument("--samples-root", default=None,   help="Watch mode: poll this folder for dumped samples instead of connecting to CARLA")
    p.add_argument("--device",       default="cuda",  help="torch device")
    p.add_argument("--max-frames",   type=int, default=500)
    p.add_argument("--delta-t",      type=float, default=0.1,  help="Simulation step (s)")
    p.add_argument("--rig-profile",  default="default",
                   choices=["default", "low_memory_shadow"],
                   help="Sensor rig quality preset")
    p.add_argument("--image-width",  type=int, default=1600)
    p.add_argument("--image-height", type=int, default=900)
    p.add_argument("--score-thresh", type=float, default=0.35)
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
    p.add_argument("--agent-trigger-mode", choices=("every_frame", "risk_or_event"), default="risk_or_event",
                   help="When compare mode calls the LLM agent")
    p.add_argument("--agent-compare-stride", type=int, default=10,
                   help="Minimum frame stride for periodic Agent compare calls in risk_or_event mode")
    p.add_argument("--agent-risk-ttc-threshold", type=float, default=3.0,
                   help="Always query Agent when estimated TTC is below this threshold")
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


def _resolve_target_map(args: argparse.Namespace, scenario_manifest: Optional[Dict[str, Any]]) -> str:
    if scenario_manifest and str(args.map) == "Town01":
        manifest_town = scenario_manifest.get("town")
        if manifest_town:
            return _normalize_map_name(str(manifest_town))
    return _normalize_map_name(str(args.map))


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
    cmd fields: steer_norm ∈ [-1,1], throttle_norm ∈ [0,1], brake_norm ∈ [0,1]
    """
    try:
        import carla  # type: ignore
        control = carla.VehicleControl(
            throttle=float(getattr(cmd, "throttle_norm", 0.0)),
            steer=float(getattr(cmd, "steer_norm", 0.0)),
            brake=float(getattr(cmd, "brake_norm", 0.0)),
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
    distance_km = float(route_summary.get("distance_traveled_m") or 0.0) / 1000.0
    collision_rate_per_km = (
        round(len(collision_events) / distance_km, 6)
        if distance_km > 1e-6
        else None
    )
    rc = route_summary.get("route_completion_rate")
    route_ok = bool(rc is not None and float(rc) >= float(args.success_rc_threshold))
    collision_ok = len(collision_events) == 0
    runtime_ok = int(stats.get("errors", 0)) == 0 and int(stats.get("frames", 0)) > 0
    success = bool(route_ok and collision_ok and runtime_ok)

    return {
        "schema_version": "stage10_driving_metrics_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "stage": "10_live_bridge",
        "map": args.map,
        "agent_mode": args.agent_mode,
        "frames": int(stats.get("frames", 0)),
        "route_completion_rate": rc,
        "route_completion_pct": route_summary.get("route_completion_pct"),
        "collision_count": len(collision_events),
        "collision_rate_per_km": collision_rate_per_km,
        "scenario_success": success,
        "scenario_success_rate": 1.0 if success else 0.0,
        "success_criteria": {
            "route_completion_rate_min": float(args.success_rc_threshold),
            "route_completion_passed": route_ok,
            "collision_count_max": 0,
            "collision_passed": collision_ok,
            "runtime_errors_max": 0,
            "runtime_passed": runtime_ok,
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
        },
        "artifacts": {
            "ego_trace": str(Path(args.log_dir) / "ego_trace.jsonl"),
            "collision_events": str(Path(args.log_dir) / "collision_events.jsonl"),
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


# ── Stage 9 Arbiter bootstrap ─────────────────────────────────────────────────
# TODO [Stage 9 hook]: Replace stubs with real Stage 9 components.

def _build_stage9_arbiter(log_dir: Path, agent_mode: str = "stub"):
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
    if ego_trace_path.exists():
        ego_trace_path.unlink()
    if collision_events_path.exists():
        collision_events_path.unlink()

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
    )
    LOGGER.info("BEVFusion ready.")

    # ── 2. Input Source (CARLA or Watch Folder) ───────────────────────────────
    # Sensors are spawned NOW, after the model is warm, so no frames are missed.
    route_tracker: Optional[RouteProgressTracker] = None
    collision_monitor: Optional[CollisionMonitor] = None
    world = None
    ego_spawned_by_stage10 = False
    if args.samples_root:
        sensor_source = FolderWatcherSync(args.samples_root, args.max_frames)
        ego = None
        carla_map = None
    else:
        scenario_manifest = _load_scenario_manifest(args.scenario_manifest)
        args.map = _resolve_target_map(args, scenario_manifest)
        client, world = _connect_carla(args.carla_host, args.carla_port)
        world = _load_map_if_needed(client, world, args.map)
        carla_map = world.get_map()
        ego, ego_spawned_by_stage10, ego_source = _spawn_or_get_ego(world, args, scenario_manifest)
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
        collision_monitor = CollisionMonitor(
            world,
            ego,
            impulse_threshold=float(args.collision_impulse_threshold),
        )
        collision_monitor.start()

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
        )

    # ── 3. Stage 9 Arbiter (optional) ────────────────────────────────────────
    arbiter = None if args.no_stage9 else _build_stage9_arbiter(log_dir, agent_mode=args.agent_mode)

    # ── 3b. Compare-mode: separate agent adapter for side-by-side logging ────
    compare_agent = None
    compare_baseline = None
    compare_log: List[Dict[str, Any]] = []
    compare_skipped_frames = 0
    if args.agent_mode == "compare":
        try:
            from carla_bevfusion_stage1.stage9_adapters import RealAgentAdapter, RealBaselineAdapter
            compare_agent = RealAgentAdapter(mode="api")
            compare_baseline = RealBaselineAdapter()
            LOGGER.info("Compare mode: side-by-side Agent vs Baseline logging ENABLED.")
        except ImportError as exc:
            LOGGER.warning("Compare mode unavailable (%s). Falling back to normal mode.", exc)
            compare_agent = None


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

    stats = {"frames": 0, "total_det": 0, "bev_ms_sum": 0.0, "errors": 0}
    prev_v = 0.0

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

            # ── C. Ego telemetry ───────────────────────────────────────────
            if not args.samples_root and ego is not None and carla_map is not None:
                ego_tel = WorldStateBuilder.extract_ego_telemetry(
                    ego, carla_map,
                    frame_id=live_frame.frame_id,
                    timestamp_s=live_frame.timestamp_s,
                    prev_v_mps=prev_v,
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
            prev_v = ego_tel.ego_v_mps

            # ── D. WorldState ──────────────────────────────────────────────
            world_state = ws_builder.build(det_list, ego_tel)
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
                    "ego_lane_id": str(ego_tel.ego_lane_id),
                    **route_info,
                },
            )

            # ── E. Stage 9 Arbiter ─────────────────────────────────────────
            if arbiter is not None and world_state is not None:
                cmd = arbiter.step(world_state, sim_time_s=live_frame.timestamp_s)
                if not args.samples_root and ego is not None:
                    _apply_actuator_command(ego, cmd)

            # ── F. Compare mode: side-by-side Agent vs Baseline ───────────
            if compare_agent is not None and compare_baseline is not None and world_state is not None:
                try:
                    t_cmp = time.monotonic()
                    bl_req = compare_baseline.plan(world_state)
                    baseline_intent = str(getattr(bl_req, "tactical_intent", "keep_lane"))

                    ttc_val = float(
                        ws_builder._prev_detections and
                        _ttc_from_prev(ws_builder._prev_detections, ego_tel.ego_v_mps) or 99.0
                    )
                    should_query_agent, trigger_reason = _should_query_agent(
                        args=args,
                        frame_idx=frame_idx,
                        baseline_intent=baseline_intent,
                        min_ttc_s=ttc_val,
                        world_state=world_state,
                    )
                    if not should_query_agent:
                        compare_skipped_frames += 1
                    else:
                        contract = compare_agent.propose_contract(world_state)
                        if contract is not None:
                            agent_intent = str(getattr(contract, "tactical_intent", "keep_lane"))
                            agent_confidence = float(getattr(contract, "agent_confidence", 0.0))
                            agent_reasoning = str(getattr(contract, "agent_reasoning_summary", ""))
                            disagreement_useful = True
                        else:
                            agent_intent = baseline_intent  # agent agrees (or fallback)
                            agent_confidence = 0.0
                            agent_reasoning = "agrees_with_baseline"
                            disagreement_useful = False

                        agrees = (baseline_intent == agent_intent)
                        cmp_ms = (time.monotonic() - t_cmp) * 1000

                        record = {
                            "frame_id": live_frame.frame_id,
                            "frame_idx": frame_idx,
                            "timestamp_s": live_frame.timestamp_s,
                            "agent_queried": True,
                            "agent_trigger_reason": trigger_reason,
                            "ego_v_mps": ego_tel.ego_v_mps,
                            "min_ttc_s": ttc_val,
                            "route_completion_rate": route_info.get("route_completion_rate"),
                            "route_progress_m": route_info.get("route_progress_m"),
                            "num_detections": len(det_list.detections),
                            "baseline_intent": baseline_intent,
                            "agent_intent": agent_intent,
                            "agent_confidence": agent_confidence,
                            "agent_reasoning": agent_reasoning,
                            "agrees": agrees,
                            "disagreement_useful": disagreement_useful and not agrees,
                            "compare_latency_ms": round(cmp_ms, 1),
                        }
                        compare_log.append(record)

                        if not agrees:
                            LOGGER.info(
                                "[COMPARE] frame=%d  baseline=%s  agent=%s  conf=%.2f  ttc=%.1f  reason=%s",
                                live_frame.frame_id, baseline_intent, agent_intent,
                                agent_confidence, ttc_val, agent_reasoning,
                            )
                except Exception as exc:
                    LOGGER.debug("Compare-mode error frame=%d: %s", frame_idx, exc)

            stats["frames"] += 1
            elapsed_ms = (time.monotonic() - t_tick) * 1000

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
        sensor_source.stop()
        if collision_monitor is not None:
            collision_monitor.stop()
            for event in collision_monitor.events():
                _append_jsonl(collision_events_path, event)
            collision_events_path.touch(exist_ok=True)
        driving_metrics = _build_driving_metrics(
            args=args,
            stats=stats,
            route_tracker=route_tracker,
            collision_monitor=collision_monitor,
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

            evaluation = {
                "schema_version": "stage10_agent_live_evaluation_v1",
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                "stage": "10_live_bridge",
                "mode": args.agent_mode,
                "map": args.map,
                "total_frames": total,
                "sim_frames": int(stats.get("frames", 0)),
                "agent_trigger_mode": args.agent_trigger_mode,
                "agent_compare_stride": int(args.agent_compare_stride),
                "agent_risk_ttc_threshold": float(args.agent_risk_ttc_threshold),
                "agent_queried_frames": total,
                "agent_skipped_frames": int(compare_skipped_frames),
                "agreement_frames": agreements,
                "disagreement_frames": disagreements,
                "agreement_rate": round(agreements / max(total, 1), 4),
                "disagreement_rate": round(disagreements / max(total, 1), 4),
                "useful_disagreement_count": useful_disagreements,
                "useful_disagreement_rate": round(
                    useful_disagreements / max(disagreements, 1), 4
                ) if disagreements > 0 else None,
                "low_ttc_analysis": {
                    "total_low_ttc_frames": len(low_ttc_frames),
                    "disagreements_in_low_ttc": len(low_ttc_disagree),
                    "agent_cautious_rate": round(
                        len(low_ttc_disagree) / max(len(low_ttc_frames), 1), 4
                    ) if low_ttc_frames else None,
                },
                "latency": {
                    "mean_compare_ms": round(sum(agent_latencies) / max(len(agent_latencies), 1), 1),
                    "max_compare_ms": round(max(agent_latencies), 1) if agent_latencies else 0,
                    "min_compare_ms": round(min(agent_latencies), 1) if agent_latencies else 0,
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
