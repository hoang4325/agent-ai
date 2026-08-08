"""Multi-object tracker with constant-velocity Kalman filter and gated association."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .kinematics import hypot2, speed_xy, ttc_2d_s, ttc_longitudinal_s
from .schema import NormalizedDetection, NormalizedFramePrediction, TrackedObject

LOGGER = logging.getLogger(__name__)


def _relative_sector(position_ego: List[float]) -> str:
    x, y = position_ego[0], position_ego[1]
    if x >= 0.0 and abs(y) <= 2.5:
        return "front"
    if x >= 0.0 and y > 2.5:
        return "front_left"
    if x >= 0.0 and y < -2.5:
        return "front_right"
    if x < 0.0 and y > 2.5:
        return "rear_left"
    if x < 0.0 and y < -2.5:
        return "rear_right"
    return "rear"


def _bearing_deg(position_ego: List[float]) -> float:
    return float(math.degrees(math.atan2(position_ego[1], position_ego[0])))


@dataclass
class _KalmanCV2D:
    """Constant-velocity Kalman filter in ego xy (state: x, y, vx, vy)."""

    x: np.ndarray  # (4,)
    P: np.ndarray  # (4,4)
    q_pos: float = 0.5
    q_vel: float = 1.5
    r_pos: float = 0.8
    r_vel: float = 2.5

    @classmethod
    def from_detection(cls, detection: NormalizedDetection) -> "_KalmanCV2D":
        x = np.array(
            [
                float(detection.position_ego[0]),
                float(detection.position_ego[1]),
                float(detection.velocity_ego[0]),
                float(detection.velocity_ego[1]),
            ],
            dtype=float,
        )
        P = np.diag([1.5, 1.5, 4.0, 4.0]).astype(float)
        return cls(x=x, P=P)

    def predict(self, dt: float) -> None:
        dt = max(1e-3, float(dt))
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        q = self.q_pos
        qv = self.q_vel
        Q = np.diag(
            [
                q * dt,
                q * dt,
                qv * dt,
                qv * dt,
            ]
        ).astype(float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, detection: NormalizedDetection) -> None:
        """
        Measurement update.

        When detection speed is near-zero (box-only / unreliable velocity), fuse
        position only so the CV model can estimate velocity from motion. Otherwise
        fuse both position and velocity.
        """
        px = float(detection.position_ego[0])
        py = float(detection.position_ego[1])
        use_velocity = float(detection.speed_mps) > 0.25
        if use_velocity:
            z = np.array(
                [px, py, float(detection.velocity_ego[0]), float(detection.velocity_ego[1])],
                dtype=float,
            )
            H = np.eye(4, dtype=float)
            R = np.diag([self.r_pos, self.r_pos, self.r_vel, self.r_vel]).astype(float)
        else:
            z = np.array([px, py], dtype=float)
            H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
            R = np.diag([self.r_pos, self.r_pos]).astype(float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=float)
        self.P = (I - K @ H) @ self.P

    @property
    def position(self) -> List[float]:
        return [float(self.x[0]), float(self.x[1]), 0.0]

    @property
    def velocity(self) -> List[float]:
        return [float(self.x[2]), float(self.x[3])]

    def mahalanobis_pos(self, detection: NormalizedDetection) -> float:
        innov = np.array(
            [
                float(detection.position_ego[0]) - float(self.x[0]),
                float(detection.position_ego[1]) - float(self.x[1]),
            ],
            dtype=float,
        )
        S = self.P[:2, :2] + np.eye(2) * self.r_pos
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        return float(innov.T @ Sinv @ innov)


@dataclass
class _TrackInternal:
    track_id: int
    class_id: int
    class_name: str
    class_group: str
    kf: _KalmanCV2D
    size_xyz: List[float]
    yaw_rad: float
    latest_score: float
    mean_score: float
    hits: int
    age_frames: int
    missed_frames: int
    last_detection_id: str | None
    last_timestamp: float
    trajectory: List[List[float]] = field(default_factory=list)
    z: float = 0.0

    def predicted_position(self, timestamp: float) -> List[float]:
        dt = max(0.0, float(timestamp - self.last_timestamp))
        return [
            float(self.kf.x[0] + self.kf.x[2] * dt),
            float(self.kf.x[1] + self.kf.x[3] * dt),
            float(self.z),
        ]


class SimpleObjectTracker:
    """
    CV-Kalman multi-object tracker with class-gated association.

    Public name kept for API compatibility with existing callers.
    """

    def __init__(
        self,
        *,
        max_missed_frames: int = 5,
        position_alpha: float = 0.65,  # retained for API compat (unused by KF)
        velocity_alpha: float = 0.55,  # retained for API compat
        vehicle_match_distance_m: float = 6.0,
        vru_match_distance_m: float = 3.5,
        static_match_distance_m: float = 2.5,
        min_confirmed_hits: int = 2,
        mahalanobis_gate: float = 9.21,  # ~chi2 99% for 2 dof
    ) -> None:
        self.max_missed_frames = int(max_missed_frames)
        self.position_alpha = float(position_alpha)
        self.velocity_alpha = float(velocity_alpha)
        self.vehicle_match_distance_m = float(vehicle_match_distance_m)
        self.vru_match_distance_m = float(vru_match_distance_m)
        self.static_match_distance_m = float(static_match_distance_m)
        self.min_confirmed_hits = int(min_confirmed_hits)
        self.mahalanobis_gate = float(mahalanobis_gate)
        self._next_track_id = 1
        self._tracks: Dict[int, _TrackInternal] = {}

    def _match_threshold(self, class_group: str) -> float:
        if class_group == "vehicle":
            return self.vehicle_match_distance_m
        if class_group == "vru":
            return self.vru_match_distance_m
        return self.static_match_distance_m

    def _candidate_cost(
        self,
        track: _TrackInternal,
        detection: NormalizedDetection,
        timestamp: float,
    ) -> float | None:
        if detection.class_group != track.class_group:
            return None
        # predict-only cost uses current KF prediction at detection time
        dt = max(0.0, float(timestamp - track.last_timestamp))
        pred = [
            float(track.kf.x[0] + track.kf.x[2] * dt),
            float(track.kf.x[1] + track.kf.x[3] * dt),
        ]
        distance = hypot2(
            pred[0] - float(detection.position_ego[0]),
            pred[1] - float(detection.position_ego[1]),
        )
        if distance > self._match_threshold(track.class_group):
            return None
        # Temporary predict for mahalanobis without mutating permanently
        kf_tmp = _KalmanCV2D(x=track.kf.x.copy(), P=track.kf.P.copy())
        if dt > 0:
            kf_tmp.predict(dt)
        maha = kf_tmp.mahalanobis_pos(detection)
        if maha > self.mahalanobis_gate:
            return None
        speed_penalty = 0.08 * abs(speed_xy(track.kf.velocity) - float(detection.speed_mps))
        return float(distance + 0.15 * math.sqrt(max(0.0, maha)) + speed_penalty)

    def _create_track(self, detection: NormalizedDetection, timestamp: float) -> _TrackInternal:
        track = _TrackInternal(
            track_id=self._next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            class_group=detection.class_group,
            kf=_KalmanCV2D.from_detection(detection),
            size_xyz=list(detection.size_xyz),
            yaw_rad=float(detection.yaw_rad),
            latest_score=float(detection.score),
            mean_score=float(detection.score),
            hits=1,
            age_frames=1,
            missed_frames=0,
            last_detection_id=detection.detection_id,
            last_timestamp=float(timestamp),
            trajectory=[list(detection.position_ego)],
            z=float(detection.position_ego[2]) if len(detection.position_ego) > 2 else 0.0,
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _update_matched_track(
        self,
        track: _TrackInternal,
        detection: NormalizedDetection,
        timestamp: float,
    ) -> None:
        dt = max(1e-3, float(timestamp - track.last_timestamp))
        # Bootstrap velocity from finite difference for young tracks when
        # the detector does not provide a reliable velocity measurement.
        if track.hits < 4 and float(detection.speed_mps) <= 0.25 and track.trajectory:
            prev = track.trajectory[-1]
            vx = (float(detection.position_ego[0]) - float(prev[0])) / dt
            vy = (float(detection.position_ego[1]) - float(prev[1])) / dt
            # Blend into KF state before predict/update (fast lock-on).
            alpha = 0.7 if track.hits == 1 else 0.45
            track.kf.x[2] = (1.0 - alpha) * float(track.kf.x[2]) + alpha * vx
            track.kf.x[3] = (1.0 - alpha) * float(track.kf.x[3]) + alpha * vy
        track.kf.predict(dt)
        track.kf.update(detection)
        track.size_xyz = list(detection.size_xyz)
        track.yaw_rad = float(detection.yaw_rad)
        track.mean_score = float(((track.mean_score * track.hits) + detection.score) / (track.hits + 1))
        track.latest_score = float(detection.score)
        track.hits += 1
        track.age_frames += 1
        track.missed_frames = 0
        track.last_detection_id = detection.detection_id
        track.last_timestamp = float(timestamp)
        track.z = float(detection.position_ego[2]) if len(detection.position_ego) > 2 else track.z
        pos = track.kf.position
        pos[2] = track.z
        track.trajectory.append(pos)
        if len(track.trajectory) > 30:
            track.trajectory = track.trajectory[-30:]

    def update(self, frame_prediction: NormalizedFramePrediction) -> List[TrackedObject]:
        timestamp = float(frame_prediction.timestamp)
        detections = list(frame_prediction.detections)
        track_ids = list(self._tracks.keys())
        candidates: List[Tuple[float, int, int]] = []
        for track_index, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            for detection_index, detection in enumerate(detections):
                cost = self._candidate_cost(track, detection, timestamp)
                if cost is None:
                    continue
                candidates.append((cost, track_index, detection_index))

        # Greedy global nearest neighbor (Hungarian-quality for sparse scenes)
        candidates.sort(key=lambda item: item[0])
        matched_track_indices: set[int] = set()
        matched_detection_indices: set[int] = set()
        matches: List[Tuple[int, int]] = []
        for _cost, track_index, detection_index in candidates:
            if track_index in matched_track_indices or detection_index in matched_detection_indices:
                continue
            matched_track_indices.add(track_index)
            matched_detection_indices.add(detection_index)
            matches.append((track_index, detection_index))

        matched_track_ids: set[int] = set()
        for track_index, detection_index in matches:
            track_id = track_ids[track_index]
            matched_track_ids.add(track_id)
            self._update_matched_track(self._tracks[track_id], detections[detection_index], timestamp)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue
            track = self._create_track(detection, timestamp)
            matched_track_ids.add(track.track_id)

        stale_track_ids: List[int] = []
        for track_id, track in self._tracks.items():
            if track_id in matched_track_ids:
                continue
            dt = max(1e-3, float(timestamp - track.last_timestamp))
            track.kf.predict(dt)
            track.age_frames += 1
            track.missed_frames += 1
            track.last_timestamp = timestamp
            pos = track.kf.position
            pos[2] = track.z
            track.trajectory.append(pos)
            if len(track.trajectory) > 30:
                track.trajectory = track.trajectory[-30:]
            if track.missed_frames > self.max_missed_frames:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            self._tracks.pop(track_id, None)

        visible_tracks: List[TrackedObject] = []
        for track in sorted(self._tracks.values(), key=lambda item: item.track_id):
            if track.hits < self.min_confirmed_hits:
                continue
            is_occluded = track.missed_frames > 0
            pos = [float(track.kf.x[0]), float(track.kf.x[1]), float(track.z)]
            vel = track.kf.velocity
            ttc = ttc_2d_s(pos, vel)
            if ttc is None:
                ttc = ttc_longitudinal_s(pos, vel)
            conf_scale = 0.85 if is_occluded else 1.0
            # lower confidence for young tracks
            if track.hits < 3:
                conf_scale *= 0.9
            visible_tracks.append(
                TrackedObject(
                    track_id=track.track_id,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    class_group=track.class_group,
                    latest_detection_id=track.last_detection_id,
                    age_frames=track.age_frames,
                    hits=track.hits,
                    missed_frames=track.missed_frames,
                    is_occluded_est=is_occluded,
                    score=float(track.latest_score),
                    mean_score=float(track.mean_score),
                    position_ego=pos,
                    velocity_ego=[float(vel[0]), float(vel[1])],
                    speed_mps=speed_xy(vel),
                    bbox=[
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2]),
                        float(track.size_xyz[0]),
                        float(track.size_xyz[1]),
                        float(track.size_xyz[2]),
                        float(track.yaw_rad),
                        float(vel[0]),
                        float(vel[1]),
                    ],
                    size_xyz=list(track.size_xyz),
                    yaw_rad=float(track.yaw_rad),
                    distance_m=hypot2(pos[0], pos[1]),
                    bearing_deg=_bearing_deg(pos),
                    ttc_seconds=ttc,
                    relative_sector=_relative_sector(pos),
                    source_confidence=max(0.05, float(track.latest_score) * conf_scale),
                )
            )

        LOGGER.info(
            "Tracker frame=%d sample=%s detections=%d active_tracks=%d matched=%d",
            frame_prediction.frame_id,
            frame_prediction.sample_name,
            len(detections),
            len(visible_tracks),
            len(matches),
        )
        return visible_tracks
