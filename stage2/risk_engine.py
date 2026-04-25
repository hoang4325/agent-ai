from __future__ import annotations

import logging
from typing import List, Tuple

from .object_schema import RiskSummary, TrackedObject

LOGGER = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class RiskAssessmentEngine:
    def __init__(
        self,
        *,
        frontal_corridor_half_width_m: float = 2.5,
        emergency_distance_m: float = 4.0,
        caution_distance_m: float = 15.0,
    ) -> None:
        self.frontal_corridor_half_width_m = float(frontal_corridor_half_width_m)
        self.emergency_distance_m = float(emergency_distance_m)
        self.caution_distance_m = float(caution_distance_m)

    def _score_track(self, track: TrackedObject) -> Tuple[float, List[str]]:
        score = 0.0
        tags: List[str] = []
        x = float(track.position_ego[0])
        y = float(track.position_ego[1])
        distance = float(track.distance_m)

        distance_term = _clamp01(1.0 - (distance / 40.0))
        score += 0.30 * distance_term
        if distance < self.caution_distance_m:
            tags.append("near_field")

        in_front = x > 0.0
        in_frontal_corridor = in_front and abs(y) <= self.frontal_corridor_half_width_m
        if in_front:
            score += 0.10
            tags.append("ahead")
        if in_frontal_corridor:
            score += 0.25
            tags.append("frontal_corridor")

        ttc = track.ttc_seconds
        if ttc is not None:
            if ttc < 1.5:
                score += 0.40
                tags.append("ttc_critical")
            elif ttc < 3.0:
                score += 0.25
                tags.append("ttc_high")
            elif ttc < 5.0:
                score += 0.10
                tags.append("ttc_medium")

        if track.class_group == "vru":
            score += 0.15
            tags.append("vru")
        elif track.class_group == "vehicle":
            score += 0.05
            tags.append("vehicle")

        if in_frontal_corridor and distance < self.emergency_distance_m:
            score += 0.35
            tags.append("distance_critical")

        if track.is_occluded_est:
            score *= 0.85
            tags.append("occluded_est")

        score = _clamp01(score)
        return score, tags

    @staticmethod
    def _level(score: float) -> str:
        if score >= 0.85:
            return "critical"
        if score >= 0.60:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    def evaluate(self, tracks: List[TrackedObject]) -> RiskSummary:
        highest_risk_score = 0.0
        highest_risk_level = "low"
        urgent_track_ids: List[int] = []
        front_hazard_track_id = None
        nearest_front_vehicle_distance = None
        nearest_vru_distance = None
        minimum_ttc = None
        flags: List[str] = []

        for track in tracks:
            risk_score, tags = self._score_track(track)
            risk_level = self._level(risk_score)
            track.risk_score = risk_score
            track.risk_level = risk_level
            track.reasoning_tags = tags

            if risk_level in {"high", "critical"}:
                urgent_track_ids.append(int(track.track_id))
            if risk_score > highest_risk_score:
                highest_risk_score = risk_score
                highest_risk_level = risk_level
            if track.relative_sector.startswith("front") and front_hazard_track_id is None and risk_level in {"high", "critical"}:
                front_hazard_track_id = int(track.track_id)

            if track.class_group == "vehicle" and track.position_ego[0] > 0.0:
                nearest_front_vehicle_distance = (
                    track.distance_m
                    if nearest_front_vehicle_distance is None
                    else min(nearest_front_vehicle_distance, track.distance_m)
                )
            if track.class_group == "vru":
                nearest_vru_distance = (
                    track.distance_m if nearest_vru_distance is None else min(nearest_vru_distance, track.distance_m)
                )
            if track.ttc_seconds is not None:
                minimum_ttc = track.ttc_seconds if minimum_ttc is None else min(minimum_ttc, track.ttc_seconds)

        if highest_risk_level in {"high", "critical"}:
            flags.append("hazard_present")
        if minimum_ttc is not None and minimum_ttc < 2.0:
            flags.append("short_ttc")
        if nearest_vru_distance is not None and nearest_vru_distance < 10.0:
            flags.append("vru_nearby")
        if nearest_front_vehicle_distance is not None and nearest_front_vehicle_distance < 10.0:
            flags.append("front_vehicle_close")

        LOGGER.info(
            "Risk summary tracks=%d highest=%s score=%.3f urgent=%s",
            len(tracks),
            highest_risk_level,
            highest_risk_score,
            urgent_track_ids,
        )
        return RiskSummary(
            highest_risk_level=highest_risk_level,
            highest_risk_score=float(highest_risk_score),
            urgent_track_ids=urgent_track_ids,
            front_hazard_track_id=front_hazard_track_id,
            nearest_front_vehicle_distance_m=nearest_front_vehicle_distance,
            nearest_vru_distance_m=nearest_vru_distance,
            minimum_ttc_seconds=minimum_ttc,
            flags=flags,
        )
