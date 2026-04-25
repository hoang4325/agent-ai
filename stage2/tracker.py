from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .object_schema import NormalizedDetection, NormalizedFramePrediction, TrackedObject

LOGGER = logging.getLogger(__name__)


def _distance_xy(lhs: List[float], rhs: List[float]) -> float:
    return float(math.hypot(lhs[0] - rhs[0], lhs[1] - rhs[1]))


def _speed_xy(velocity_xy: List[float]) -> float:
    return float(math.hypot(velocity_xy[0], velocity_xy[1]))


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


def _distance_m(position_ego: List[float]) -> float:
    return float(math.hypot(position_ego[0], position_ego[1]))


def _ttc_seconds(position_ego: List[float], velocity_ego: List[float]) -> float | None:
    x = float(position_ego[0])
    if x <= 0.0:
        return None
    closing_speed = max(0.0, -float(velocity_ego[0]))
    if closing_speed <= 0.1:
        return None
    return float(x / closing_speed)


@dataclass
class _TrackInternal:
    track_id: int
    class_id: int
    class_name: str
    class_group: str
    last_position_ego: List[float]
    smoothed_velocity_ego: List[float]
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

    def predicted_position(self, timestamp: float) -> List[float]:
        dt = max(0.0, float(timestamp - self.last_timestamp))
        return [
            float(self.last_position_ego[0] + self.smoothed_velocity_ego[0] * dt),
            float(self.last_position_ego[1] + self.smoothed_velocity_ego[1] * dt),
            float(self.last_position_ego[2]),
        ]


class SimpleObjectTracker:
    def __init__(
        self,
        *,
        max_missed_frames: int = 2,
        position_alpha: float = 0.65,
        velocity_alpha: float = 0.55,
        vehicle_match_distance_m: float = 5.0,
        vru_match_distance_m: float = 3.0,
        static_match_distance_m: float = 2.5,
        min_confirmed_hits: int = 1,
    ) -> None:
        self.max_missed_frames = int(max_missed_frames)
        self.position_alpha = float(position_alpha)
        self.velocity_alpha = float(velocity_alpha)
        self.vehicle_match_distance_m = float(vehicle_match_distance_m)
        self.vru_match_distance_m = float(vru_match_distance_m)
        self.static_match_distance_m = float(static_match_distance_m)
        self.min_confirmed_hits = int(min_confirmed_hits)
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
        predicted_position = track.predicted_position(timestamp)
        distance = _distance_xy(predicted_position, detection.position_ego)
        if distance > self._match_threshold(track.class_group):
            return None
        velocity_penalty = 0.1 * abs(_speed_xy(track.smoothed_velocity_ego) - detection.speed_mps)
        return distance + velocity_penalty

    def _create_track(self, detection: NormalizedDetection, timestamp: float) -> _TrackInternal:
        track = _TrackInternal(
            track_id=self._next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            class_group=detection.class_group,
            last_position_ego=list(detection.position_ego),
            smoothed_velocity_ego=list(detection.velocity_ego),
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
        previous_position = list(track.last_position_ego)
        finite_difference_velocity = [
            float((detection.position_ego[0] - previous_position[0]) / dt),
            float((detection.position_ego[1] - previous_position[1]) / dt),
        ]
        observed_velocity = [
            0.5 * finite_difference_velocity[0] + 0.5 * float(detection.velocity_ego[0]),
            0.5 * finite_difference_velocity[1] + 0.5 * float(detection.velocity_ego[1]),
        ]
        track.smoothed_velocity_ego = [
            self.velocity_alpha * observed_velocity[0]
            + (1.0 - self.velocity_alpha) * float(track.smoothed_velocity_ego[0]),
            self.velocity_alpha * observed_velocity[1]
            + (1.0 - self.velocity_alpha) * float(track.smoothed_velocity_ego[1]),
        ]
        predicted_position = track.predicted_position(timestamp)
        track.last_position_ego = [
            self.position_alpha * float(detection.position_ego[0])
            + (1.0 - self.position_alpha) * float(predicted_position[0]),
            self.position_alpha * float(detection.position_ego[1])
            + (1.0 - self.position_alpha) * float(predicted_position[1]),
            float(detection.position_ego[2]),
        ]
        track.size_xyz = list(detection.size_xyz)
        track.yaw_rad = float(detection.yaw_rad)
        track.latest_score = float(detection.score)
        track.mean_score = float(((track.mean_score * track.hits) + detection.score) / (track.hits + 1))
        track.hits += 1
        track.age_frames += 1
        track.missed_frames = 0
        track.last_detection_id = detection.detection_id
        track.last_timestamp = float(timestamp)
        track.trajectory.append(list(track.last_position_ego))
        if len(track.trajectory) > 20:
            track.trajectory = track.trajectory[-20:]

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

        candidates.sort(key=lambda item: item[0])
        matched_track_indices = set()
        matched_detection_indices = set()
        matches: List[Tuple[int, int]] = []
        for _cost, track_index, detection_index in candidates:
            if track_index in matched_track_indices or detection_index in matched_detection_indices:
                continue
            matched_track_indices.add(track_index)
            matched_detection_indices.add(detection_index)
            matches.append((track_index, detection_index))

        matched_track_ids = set()
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
            track.age_frames += 1
            track.missed_frames += 1
            predicted_position = track.predicted_position(timestamp)
            track.last_position_ego = predicted_position
            track.last_timestamp = timestamp
            track.trajectory.append(list(track.last_position_ego))
            if len(track.trajectory) > 20:
                track.trajectory = track.trajectory[-20:]
            if track.missed_frames > self.max_missed_frames:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            self._tracks.pop(track_id, None)

        visible_tracks: List[TrackedObject] = []
        for track in sorted(self._tracks.values(), key=lambda item: item.track_id):
            if track.hits < self.min_confirmed_hits:
                continue
            is_occluded = track.missed_frames > 0
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
                    position_ego=[float(track.last_position_ego[0]), float(track.last_position_ego[1]), float(track.last_position_ego[2])],
                    velocity_ego=[float(track.smoothed_velocity_ego[0]), float(track.smoothed_velocity_ego[1])],
                    speed_mps=_speed_xy(track.smoothed_velocity_ego),
                    bbox=[
                        float(track.last_position_ego[0]),
                        float(track.last_position_ego[1]),
                        float(track.last_position_ego[2]),
                        float(track.size_xyz[0]),
                        float(track.size_xyz[1]),
                        float(track.size_xyz[2]),
                        float(track.yaw_rad),
                        float(track.smoothed_velocity_ego[0]),
                        float(track.smoothed_velocity_ego[1]),
                    ],
                    size_xyz=list(track.size_xyz),
                    yaw_rad=float(track.yaw_rad),
                    distance_m=_distance_m(track.last_position_ego),
                    bearing_deg=_bearing_deg(track.last_position_ego),
                    ttc_seconds=_ttc_seconds(track.last_position_ego, track.smoothed_velocity_ego),
                    relative_sector=_relative_sector(track.last_position_ego),
                    source_confidence=max(0.05, float(track.latest_score) * (0.85 if is_occluded else 1.0)),
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
