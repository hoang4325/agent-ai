"""
carla_bevfusion_stage1/carla_sensor_sync.py
============================================
Live sensor synchronisation layer.

Responsibilities:
  1. Spawn the NuScenes-like sensor rig (6 cams + 1 lidar + N radars) on
     an ego vehicle using the existing rig.py / collector.py primitives.
  2. Drive CARLA in synchronous mode so every tick we get exactly one
     fully-assembled FrameCapture (all sensors at the same frame_id).
  3. Extract raw arrays from each SensorPacket measurement and package
     them into a LiveFrame – the "raw sensor bundle" that the adapter
     layer will convert to BEVFusion model inputs.

No BEVFusion or Stage 9 code here – this layer only knows about CARLA.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Existing primitives – already battle-tested
from .collector import CarlaSynchronousMode, FrameCapture, FrameCollector
from .constants import (
    DEFAULT_FIXED_DELTA_SECONDS,
    LIDAR_SENSOR_NAME,
    MODEL_CAMERA_ORDER,
    RADAR_SENSOR_ORDER,
    SENSOR_NAMES,
)
from .coordinate_utils import (
    carla_transform_to_matrix,
    carla_vector_to_bevfusion_xyz,
    camera_intrinsics_from_width_height_fov,
    pose_bundle,
)
from .rig import RigPreset, build_rig_preset, destroy_actors, spawn_rig

LOGGER = logging.getLogger(__name__)


# ── Live sensor bundle (raw, pre-adapter) ────────────────────────────────────

@dataclass
class LiveCalibration:
    """Per-frame calibration matrices in BEVFusion (right-hand) convention."""
    world_from_lidar_bev: np.ndarray          # (4,4)
    ego_from_lidar_bev: np.ndarray            # (4,4)
    world_from_cameras_bev: Dict[str, np.ndarray]   # cam_name → (4,4)
    ego_from_cameras_bev: Dict[str, np.ndarray]     # cam_name → (4,4)
    camera_intrinsics: Dict[str, np.ndarray]        # cam_name → (3,3)


@dataclass
class LiveFrame:
    """
    One fully-synchronised raw sensor snapshot from CARLA.

    images_bgra : cam_name → (H, W, 4) uint8
    lidar_xyzi  : (N, 4) float32  [x, y, z, intensity]  CARLA coords
    radar_raw   : sensor_name → (M, 4) float32  [velocity, altitude, azimuth, depth]
    calibration : per-frame extrinsics/intrinsics
    ego_velocity_carla : (3,) float32 [vx, vy, vz] in CARLA coords
    """
    frame_id: int
    timestamp_s: float
    images_bgra: Dict[str, np.ndarray]
    lidar_xyzi: np.ndarray
    radar_raw: Dict[str, np.ndarray]
    calibration: LiveCalibration
    ego_velocity_carla: np.ndarray          # (3,) float32


# ── Helpers ───────────────────────────────────────────────────────────────────

def _camera_image_to_bgra(measurement) -> np.ndarray:
    """Convert a CARLA Image measurement to (H, W, 4) uint8 BGRA ndarray."""
    import carla  # type: ignore
    raw = np.frombuffer(measurement.raw_data, dtype=np.uint8)
    return raw.reshape(measurement.height, measurement.width, 4).copy()


def _lidar_to_xyzi(measurement) -> np.ndarray:
    """Convert a CARLA LidarMeasurement to (N, 4) float32 [x,y,z,intensity]."""
    raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
    return raw.reshape(-1, 4).copy()


def _radar_to_array(measurement) -> np.ndarray:
    """Convert a CARLA RadarMeasurement to (M, 4) float32 [velocity,alt,azimuth,depth]."""
    if hasattr(measurement, "get_detections"):
        detections = measurement.get_detections()
    else:
        try:
            detections = list(measurement)
        except TypeError:
            raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
            return raw.reshape(-1, 4).copy()
    if not detections:
        return np.zeros((0, 4), dtype=np.float32)
    rows = [
        [d.velocity, d.altitude, d.azimuth, d.depth]
        for d in detections
    ]
    return np.array(rows, dtype=np.float32)


def _build_calibration(
    ego_actor,
    sensor_actors: Dict[str, object],
    preset: RigPreset,
    image_width: int,
    image_height: int,
    camera_fov: float,
) -> LiveCalibration:
    """Derive per-frame calibration from current actor transforms."""
    world_from_ego_carla = carla_transform_to_matrix(ego_actor.get_transform())
    intrinsics_3x3 = camera_intrinsics_from_width_height_fov(
        image_width, image_height, camera_fov
    )

    lidar_actor = sensor_actors[LIDAR_SENSOR_NAME]
    lidar_carla_matrix = carla_transform_to_matrix(lidar_actor.get_transform())
    lidar_bundle = pose_bundle(
        lidar_carla_matrix, world_from_ego_carla, is_camera=False
    )

    world_from_cameras_bev: Dict[str, np.ndarray] = {}
    ego_from_cameras_bev: Dict[str, np.ndarray] = {}
    camera_intrinsics: Dict[str, np.ndarray] = {}

    for cam_name in MODEL_CAMERA_ORDER:
        cam_actor = sensor_actors[cam_name]
        cam_carla_matrix = carla_transform_to_matrix(cam_actor.get_transform())
        cam_bundle = pose_bundle(
            cam_carla_matrix, world_from_ego_carla, is_camera=True
        )
        world_from_cameras_bev[cam_name] = cam_bundle.world_from_sensor_bevfusion
        ego_from_cameras_bev[cam_name] = cam_bundle.ego_from_sensor_bevfusion
        camera_intrinsics[cam_name] = intrinsics_3x3.copy()

    return LiveCalibration(
        world_from_lidar_bev=lidar_bundle.world_from_sensor_bevfusion,
        ego_from_lidar_bev=lidar_bundle.ego_from_sensor_bevfusion,
        world_from_cameras_bev=world_from_cameras_bev,
        ego_from_cameras_bev=ego_from_cameras_bev,
        camera_intrinsics=camera_intrinsics,
    )


# ── Main synchronous loop handle ─────────────────────────────────────────────

class CarlaSensorSync:
    """
    Manages the full CARLA sensor rig lifecycle and provides
    tick() → LiveFrame for each simulation step.

    Usage:
        sync = CarlaSensorSync(world, ego, preset)
        sync.start()
        try:
            for _ in range(max_frames):
                live_frame = sync.tick()
                # → pass to BEVFusionLiveAdapter
        finally:
            sync.stop()
    """

    def __init__(
        self,
        world,                              # carla.World
        ego_actor,                          # carla.Actor (vehicle)
        preset: RigPreset,
        *,
        fixed_delta_seconds: float = DEFAULT_FIXED_DELTA_SECONDS,
        sensor_timeout_s: float = 5.0,
        image_width: int = 1600,
        image_height: int = 900,
        camera_fov: float = 70.0,
        enable_radar: bool = True,
    ) -> None:
        self._world = world
        self._ego = ego_actor
        self._preset = preset
        self._delta_s = fixed_delta_seconds
        self._sensor_timeout = sensor_timeout_s
        self._img_w = image_width
        self._img_h = image_height
        self._cam_fov = camera_fov
        self._enable_radar = enable_radar

        self._sensor_actors: Dict[str, object] = {}
        self._collector: Optional[FrameCollector] = None
        self._sync_ctx: Optional[CarlaSynchronousMode] = None
        self._started = False
        self._frame_counter: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return
        LOGGER.info("Spawning sensor rig '%s'", self._preset.name)
        self._sensor_actors = spawn_rig(self._world, self._ego, self._preset)

        # Wire up synchronous mode
        self._sync_ctx = CarlaSynchronousMode(
            self._world, fixed_delta_seconds=self._delta_s
        )
        self._sync_ctx.__enter__()

        # Wire up frame collector (registers .listen() callbacks)
        self._collector = FrameCollector(
            self._sensor_actors, timeout_seconds=self._sensor_timeout
        )
        self._collector.start()

        # Warm-up: drain first 3 frames so buffer is clean
        for _ in range(3):
            frame_id = self._sync_ctx.tick()
            try:
                snap = self._world.get_snapshot()
                self._collector.wait_for_frame(frame_id, snap)
            except TimeoutError:
                pass

        self._started = True
        LOGGER.info("CarlaSensorSync started (delta=%.3fs)", self._delta_s)

    def stop(self) -> None:
        if not self._started:
            return
        if self._collector:
            self._collector.stop()
        destroy_actors(list(self._sensor_actors.values()))
        if self._sync_ctx:
            self._sync_ctx.__exit__(None, None, None)
        self._started = False
        LOGGER.info("CarlaSensorSync stopped. Total frames: %d", self._frame_counter)

    # ── Per-tick API ──────────────────────────────────────────────────────────

    def tick(self) -> LiveFrame:
        """
        Advance the simulation by one fixed_delta_seconds step and
        return a fully synchronised LiveFrame.
        """
        if not self._started:
            raise RuntimeError("Call start() before tick().")

        frame_id = self._sync_ctx.tick()        # type: ignore[union-attr]
        snap = self._world.get_snapshot()
        capture: FrameCapture = self._collector.wait_for_frame(  # type: ignore[union-attr]
            frame_id, snap
        )
        self._frame_counter += 1

        return self._extract_live_frame(capture)

    # ── Private extraction ────────────────────────────────────────────────────

    def _extract_live_frame(self, capture: FrameCapture) -> LiveFrame:
        """Convert a FrameCapture (raw CARLA measurements) to LiveFrame."""
        images_bgra: Dict[str, np.ndarray] = {}
        for cam_name in MODEL_CAMERA_ORDER:
            pkt = capture.packets[cam_name]
            images_bgra[cam_name] = _camera_image_to_bgra(pkt.measurement)

        lidar_pkt = capture.packets[LIDAR_SENSOR_NAME]
        lidar_xyzi = _lidar_to_xyzi(lidar_pkt.measurement)

        radar_raw: Dict[str, np.ndarray] = {}
        if self._enable_radar:
            for radar_name in RADAR_SENSOR_ORDER:
                if radar_name in capture.packets:
                    radar_raw[radar_name] = _radar_to_array(
                        capture.packets[radar_name].measurement
                    )
                else:
                    radar_raw[radar_name] = np.zeros((0, 4), dtype=np.float32)

        # Per-frame calibration from current actor transforms
        calibration = _build_calibration(
            ego_actor=self._ego,
            sensor_actors=self._sensor_actors,
            preset=self._preset,
            image_width=self._img_w,
            image_height=self._img_h,
            camera_fov=self._cam_fov,
        )

        # Ego velocity
        vel = self._ego.get_velocity()
        ego_vel = np.array([vel.x, vel.y, vel.z], dtype=np.float32)

        return LiveFrame(
            frame_id=capture.frame,
            timestamp_s=float(capture.world_snapshot.timestamp.elapsed_seconds),
            images_bgra=images_bgra,
            lidar_xyzi=lidar_xyzi,
            radar_raw=radar_raw,
            calibration=calibration,
            ego_velocity_carla=ego_vel,
        )
