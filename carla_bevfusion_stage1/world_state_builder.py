"""
carla_bevfusion_stage1/world_state_builder.py
==============================================
Converts a DetectionList (BEVFusion output) + live CARLA telemetry into a
WorldState object that Stage 9 AuthorityArbiter accepts.

Design rules:
  - No BEVFusion or CARLA-specific code here.
  - Computes Stage 9 fields: min_ttc_s, ego_lateral_error_m,
    new_obstacle_score, ODD status, drivable envelope, sensor health.
  - Easily extensible: add more signals by adding fields to the builder.
  - TODO hooks clearly marked for Stage 9 Arbiter integration.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .bevfusion_live_adapter import Detection3D, DetectionList

# Stage 9 WorldState (import here; comment out if running without stage9 package)
try:
    from stage9.schemas import (
        ActuatorCommand,
        AuthorityState,
        DrivableEnvelope,
        MRMStrategy,
        ODDStatus,
        SensorHealth,
        WorldState,
    )
    _STAGE9_AVAILABLE = True
except ImportError:
    _STAGE9_AVAILABLE = False
    WorldState = None   # type: ignore

LOGGER = logging.getLogger(__name__)


# ── Minimal WorldState fields computed here ───────────────────────────────────

@dataclass
class EgoTelemetry:
    """Ego vehicle state from CARLA each tick."""
    frame_id: int
    timestamp_s: float
    ego_lane_id: str                   # from CARLA's map.get_waypoint()
    ego_v_mps: float                   # longitudinal speed
    ego_a_mps2: float                  # longitudinal acceleration estimate
    ego_lateral_error_m: float         # lateral offset from lane centre
    heading_error_rad: float           # yaw error vs waypoint heading
    ego_location_xyz: np.ndarray       # (3,) CARLA coords


@dataclass
class LiveWorldSignals:
    """
    Fused signals computed from DetectionList + EgoTelemetry.
    Used as the direct source for WorldState fields sent to Stage 9.
    """
    frame_id: int
    timestamp_s: float
    world_age_ms: int                  # milliseconds since frame was built

    # Perception signals
    min_ttc_s: float                   # minimum TTC across all detections
    new_obstacle_score: float          # normalised novelty score [0,1]
    closest_obstacle_dist_m: float
    num_detections: int

    # Ego
    ego_v_mps: float
    ego_a_mps2: float
    ego_lateral_error_m: float
    ego_lane_id: str

    # Health
    sensor_health: "SensorHealth"       # type: ignore
    sync_ok: bool
    odd_status: "ODDStatus"             # type: ignore
    time_to_odd_exit_s: Optional[float]
    preview_feasible: bool
    human_driver_available: bool
    human_input_detected: bool

    # Drivable envelope (from lane geometry)
    drivable_envelope: "DrivableEnvelope"  # type: ignore


# ── TTC estimation ────────────────────────────────────────────────────────────

def _estimate_ttc(
    detections: List[Detection3D],
    ego_v_mps: float,
    safety_margin_m: float = 1.0,
) -> float:
    """
    Simple forward-sector TTC using closest detection in the forward cone.
    Returns large value (99.0) if no obstacles in cone.
    """
    min_ttc = 99.0
    for det in detections:
        # Only consider objects ahead (x > 0 in BEVFusion ego frame)
        if det.x < 0.5:
            continue
        dist = float(np.sqrt(det.x**2 + det.y**2))
        dist_net = max(0.1, dist - safety_margin_m)
        closing_v = max(0.5, ego_v_mps)   # assume closing at ego speed (worst case)
        ttc = dist_net / closing_v
        min_ttc = min(min_ttc, ttc)
    return min_ttc


def _new_obstacle_score(
    detections: List[Detection3D],
    prev_detections: List[Detection3D],
    match_dist_threshold_m: float = 5.0,
) -> float:
    """
    Fraction of current detections that have no spatial match in previous frame.
    Ranges [0, 1]. High score = many new obstacles appeared this frame.
    """
    if not detections:
        return 0.0
    if not prev_detections:
        return 1.0
    prev_xy = np.array([[d.x, d.y] for d in prev_detections], dtype=np.float32)
    new_count = 0
    for det in detections:
        xy = np.array([det.x, det.y], dtype=np.float32)
        dists = np.linalg.norm(prev_xy - xy, axis=1)
        if dists.min() > match_dist_threshold_m:
            new_count += 1
    return new_count / len(detections)


def _closest_dist(detections: List[Detection3D]) -> float:
    if not detections:
        return 999.0
    return min(float(np.sqrt(d.x**2 + d.y**2)) for d in detections)


# ── ODD / sensor health stubs ─────────────────────────────────────────────────
# TODO [Stage 9 hook]: Plug in real ODD boundary checks from HD-map or geofence.

def _assess_odd(
    ego_location_xyz: np.ndarray,
    ego_v_mps: float,
    sync_ok: bool,
) -> "tuple[ODDStatus, Optional[float]]":
    """
    Stub: always IN_ODD while sync is OK.
    Replace with real geofencing / HD-map checks.
    """
    if not _STAGE9_AVAILABLE:
        return None, None           # type: ignore
    if not sync_ok:
        return ODDStatus.ODD_EXCEEDED, 0.0
    return ODDStatus.IN_ODD, None


def _assess_sensor_health(
    detection_list: DetectionList,
    max_inference_ms: float = 500.0,
) -> "SensorHealth":
    if not _STAGE9_AVAILABLE:
        return None              # type: ignore
    if detection_list.inference_time_ms > max_inference_ms:
        return SensorHealth.DEGRADED_BUT_VALID
    return SensorHealth.OK


# ── WorldStateBuilder ─────────────────────────────────────────────────────────

class WorldStateBuilder:
    """
    Builds Stage 9 WorldState objects from live BEVFusion detections
    and CARLA ego telemetry.

    Usage:
        builder = WorldStateBuilder()
        world_state = builder.build(detection_list, ego_tel, sync_ok=True)

    TODO [Stage 9 hook]: Pass world_state to stage9.AuthorityArbiter.step()
    """

    def __init__(
        self,
        *,
        lane_width_m: float = 3.5,
        odd_boundary_dist_m: Optional[float] = None,
        human_driver_available: bool = True,
    ) -> None:
        self._lane_width = lane_width_m
        self._odd_boundary = odd_boundary_dist_m
        self._human_driver = human_driver_available
        self._prev_detections: List[Detection3D] = []
        self._prev_ego_v_mps: float = 0.0
        self._prev_timestamp_s: float = 0.0
        self._build_count: int = 0

    # ── Main API ──────────────────────────────────────────────────────────────

    def build(
        self,
        detection_list: DetectionList,
        ego_tel: EgoTelemetry,
        *,
        sync_ok: bool = True,
        preview_feasible: bool = True,
        human_input_detected: bool = False,
    ) -> Optional["WorldState"]:
        """
        Build a WorldState from the latest DetectionList + EgoTelemetry.
        Returns None if Stage 9 schemas are not importable (standalone mode).
        """
        build_start = time.monotonic()

        dets = detection_list.detections
        ego_v = ego_tel.ego_v_mps

        # Compute dt for acceleration estimate
        dt = max(0.01, ego_tel.timestamp_s - self._prev_timestamp_s)
        ego_a = (ego_v - self._prev_ego_v_mps) / dt if self._build_count > 0 else 0.0

        # Perception-derived signals
        min_ttc = _estimate_ttc(dets, ego_v)
        new_obs = _new_obstacle_score(dets, self._prev_detections)
        closest = _closest_dist(dets)
        world_age_ms = max(0, int((time.monotonic() - build_start) * 1000))

        # Health / ODD
        sensor_health = _assess_sensor_health(detection_list)
        odd_status, time_to_exit = _assess_odd(
            ego_tel.ego_location_xyz, ego_v, sync_ok
        )

        # Drivable envelope centred on lane
        if _STAGE9_AVAILABLE:
            envelope = DrivableEnvelope(
                envelope_uuid=f"live_{ego_tel.frame_id}",
                left_bound_m=self._lane_width / 2.0,
                right_bound_m=self._lane_width / 2.0,
                forward_clear_m=max(5.0, closest - 1.0),
            )
        else:
            envelope = None   # type: ignore

        # Update memory
        self._prev_detections = list(dets)
        self._prev_ego_v_mps = ego_v
        self._prev_timestamp_s = ego_tel.timestamp_s
        self._build_count += 1

        signals = LiveWorldSignals(
            frame_id=ego_tel.frame_id,
            timestamp_s=ego_tel.timestamp_s,
            world_age_ms=world_age_ms,
            min_ttc_s=min_ttc,
            new_obstacle_score=new_obs,
            closest_obstacle_dist_m=closest,
            num_detections=len(dets),
            ego_v_mps=ego_v,
            ego_a_mps2=ego_a,
            ego_lateral_error_m=ego_tel.ego_lateral_error_m,
            ego_lane_id=ego_tel.ego_lane_id,
            sensor_health=sensor_health,
            sync_ok=sync_ok,
            odd_status=odd_status,
            time_to_odd_exit_s=time_to_exit,
            preview_feasible=preview_feasible,
            human_driver_available=self._human_driver,
            human_input_detected=human_input_detected,
            drivable_envelope=envelope,
        )

        LOGGER.debug(
            "WorldState frame=%d  n_det=%d  ttc=%.2fs  v=%.1f m/s  lat_err=%.3f m",
            ego_tel.frame_id, len(dets), min_ttc, ego_v, ego_tel.ego_lateral_error_m,
        )

        if not _STAGE9_AVAILABLE:
            LOGGER.warning("Stage 9 not importable – returning LiveWorldSignals only.")
            return None

        # ── Build Stage 9 WorldState ──────────────────────────────────────────
        return WorldState(
            frame_id=ego_tel.frame_id,
            timestamp_s=ego_tel.timestamp_s,
            world_age_ms=world_age_ms,
            sync_ok=sync_ok,
            sensor_health=sensor_health,
            ego_v_mps=ego_v,
            ego_a_mps2=ego_a,
            ego_lane_id=ego_tel.ego_lane_id,
            ego_lateral_error_m=ego_tel.ego_lateral_error_m,
            corridor_clear=(closest > 8.0),
            min_ttc_s=min_ttc,
            new_obstacle_score=new_obs,
            weather_visibility_m=100.0,           # TODO: from CARLA weather API
            lane_change_permission=True,           # TODO: from HD-map rule engine
            drivable_envelope=envelope,
            odd_status=odd_status,
            time_to_odd_exit_s=time_to_exit,
            preview_feasible=preview_feasible,
            human_driver_available=self._human_driver,
            human_input_detected=human_input_detected,
        )

    # ── Ego telemetry helper ──────────────────────────────────────────────────

    @staticmethod
    def extract_ego_telemetry(
        ego_actor,
        carla_map,
        frame_id: int,
        timestamp_s: float,
        prev_v_mps: float = 0.0,
    ) -> EgoTelemetry:
        """
        Pull ego telemetry directly from a CARLA vehicle actor.
        Call once per tick inside the main loop.
        """
        vel = ego_actor.get_velocity()
        ego_v = float(np.sqrt(vel.x**2 + vel.y**2 + vel.z**2))

        loc = ego_actor.get_location()
        ego_xy = np.array([loc.x, loc.y, loc.z], dtype=np.float32)

        # Lane lateral error via nearest waypoint
        waypoint = carla_map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=0xFFFF,       # all lane types
        )
        if waypoint:
            wp_loc = waypoint.transform.location
            lane_id = str(waypoint.lane_id)
            # lateral error = signed distance to waypoint centre projected perpendicular
            wp_fwd = np.array([
                np.cos(np.radians(waypoint.transform.rotation.yaw)),
                np.sin(np.radians(waypoint.transform.rotation.yaw)),
            ], dtype=np.float32)
            delta = np.array([loc.x - wp_loc.x, loc.y - wp_loc.y], dtype=np.float32)
            lateral_error = float(np.cross(wp_fwd, delta))
        else:
            lane_id = "unknown"
            lateral_error = 0.0

        return EgoTelemetry(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            ego_lane_id=lane_id,
            ego_v_mps=ego_v,
            ego_a_mps2=0.0,        # filled by builder using prev_v
            ego_lateral_error_m=lateral_error,
            heading_error_rad=0.0, # TODO: compute from waypoint vs ego heading
            ego_location_xyz=ego_xy,
        )
