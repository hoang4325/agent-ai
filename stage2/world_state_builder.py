from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from .object_schema import (
    EgoState,
    NormalizedFramePrediction,
    RiskSummary,
    SceneSummary,
    TrackedObject,
    WorldState,
)

LOGGER = logging.getLogger(__name__)


def _object_ref(track: TrackedObject) -> Dict[str, Any]:
    return {
        "track_id": int(track.track_id),
        "class": str(track.class_name),
        "distance_m": float(track.distance_m),
        "bearing_deg": float(track.bearing_deg),
        "position_ego": list(track.position_ego),
        "velocity_ego": list(track.velocity_ego),
        "risk_level": str(track.risk_level),
        "risk_score": float(track.risk_score),
    }


class WorldStateBuilder:
    def __init__(self, *, frontal_corridor_half_width_m: float = 2.5) -> None:
        self.frontal_corridor_half_width_m = float(frontal_corridor_half_width_m)
        self._previous_world_position: List[float] | None = None
        self._route_progress_m = 0.0

    def _update_route_progress(self, ego: EgoState) -> None:
        if self._previous_world_position is None:
            self._previous_world_position = list(ego.position_world)
            ego.route_progress_m = 0.0
            return

        dx = float(ego.position_world[0] - self._previous_world_position[0])
        dy = float(ego.position_world[1] - self._previous_world_position[1])
        step_distance = math.hypot(dx, dy)
        self._route_progress_m += float(step_distance)
        self._previous_world_position = list(ego.position_world)
        ego.route_progress_m = float(self._route_progress_m)

    def _build_scene_summary(self, tracks: List[TrackedObject], risk_summary: RiskSummary) -> SceneSummary:
        front_tracks = [
            track
            for track in tracks
            if track.position_ego[0] > 0.0 and abs(track.position_ego[1]) <= self.frontal_corridor_half_width_m
        ]
        left_tracks = [track for track in tracks if track.position_ego[1] > 1.5]
        right_tracks = [track for track in tracks if track.position_ego[1] < -1.5]
        rear_tracks = [track for track in tracks if track.position_ego[0] < 0.0]

        front_free_space = None
        if front_tracks:
            front_free_space = min(
                max(0.0, float(track.position_ego[0]) - 0.5 * float(track.size_xyz[0]))
                for track in front_tracks
            )

        left_occupancy = min(
            1.0,
            sum(max(0.0, 1.0 - (track.distance_m / 30.0)) for track in left_tracks),
        )
        right_occupancy = min(
            1.0,
            sum(max(0.0, 1.0 - (track.distance_m / 30.0)) for track in right_tracks),
        )

        nearest_front_vehicle = None
        front_vehicles = [track for track in front_tracks if track.class_group == "vehicle"]
        if front_vehicles:
            nearest_front_vehicle = _object_ref(min(front_vehicles, key=lambda item: item.distance_m))

        nearest_vru = None
        vru_tracks = [track for track in tracks if track.class_group == "vru"]
        if vru_tracks:
            nearest_vru = _object_ref(min(vru_tracks, key=lambda item: item.distance_m))

        nearest_any_object = None
        if tracks:
            nearest_any_object = _object_ref(min(tracks, key=lambda item: item.distance_m))

        rear_gap = None
        if rear_tracks:
            rear_gap = min(abs(float(track.position_ego[0])) for track in rear_tracks)

        abnormal_flags: List[str] = []
        if not tracks:
            abnormal_flags.append("no_active_tracks")
        if front_free_space is not None and front_free_space < 8.0:
            abnormal_flags.append("front_blocked")
        if risk_summary.highest_risk_level in {"high", "critical"}:
            abnormal_flags.append("high_risk_scene")
        if nearest_vru is not None and nearest_vru["distance_m"] < 10.0:
            abnormal_flags.append("vru_near")

        return SceneSummary(
            active_object_count=int(len(tracks)),
            front_free_space_m=front_free_space,
            left_side_occupancy=float(left_occupancy),
            right_side_occupancy=float(right_occupancy),
            rear_gap_m=rear_gap,
            nearest_front_vehicle=nearest_front_vehicle,
            nearest_vru=nearest_vru,
            nearest_any_object=nearest_any_object,
            abnormal_flags=abnormal_flags,
        )

    def build(
        self,
        frame_prediction: NormalizedFramePrediction,
        tracks: List[TrackedObject],
        risk_summary: RiskSummary,
    ) -> WorldState:
        ego = frame_prediction.ego
        self._update_route_progress(ego)
        scene = self._build_scene_summary(tracks, risk_summary)

        recommended_maneuvers: List[str] = []
        hard_constraints: List[str] = []
        soft_constraints: List[str] = []

        if scene.front_free_space_m is not None and scene.front_free_space_m < 8.0:
            recommended_maneuvers.append("slow_down")
            hard_constraints.append("respect_front_obstacle_buffer")
        if risk_summary.nearest_vru_distance_m is not None and risk_summary.nearest_vru_distance_m < 10.0:
            recommended_maneuvers.append("yield")
            hard_constraints.append("protect_vru")
        if risk_summary.highest_risk_level == "critical":
            recommended_maneuvers.append("emergency_stop")
            hard_constraints.append("full_stop_if_required")
        if not recommended_maneuvers:
            recommended_maneuvers.append("keep_lane")
        if scene.left_side_occupancy + 0.15 < scene.right_side_occupancy:
            soft_constraints.append("prefer_left_if_replan")
        elif scene.right_side_occupancy + 0.15 < scene.left_side_occupancy:
            soft_constraints.append("prefer_right_if_replan")
        else:
            soft_constraints.append("prefer_keep_lane")

        world_state = WorldState(
            frame_id=frame_prediction.frame_id,
            timestamp=frame_prediction.timestamp,
            sample_name=frame_prediction.sample_name,
            sequence_index=frame_prediction.sequence_index,
            ego=ego,
            objects=tracks,
            scene=scene,
            risk_summary=risk_summary,
            decision_context={
                "recommended_maneuvers": recommended_maneuvers,
                "hard_constraints": hard_constraints,
                "soft_constraints": soft_constraints,
            },
        )
        LOGGER.info(
            "World state frame=%d sample=%s active_objects=%d front_free_space=%s highest_risk=%s",
            frame_prediction.frame_id,
            frame_prediction.sample_name,
            len(tracks),
            "none" if scene.front_free_space_m is None else f"{scene.front_free_space_m:.2f}",
            risk_summary.highest_risk_level,
        )
        return world_state
