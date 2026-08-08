"""Risk assessment with 2-D TTC-aware scoring and speed-adaptive near-field weights."""
from __future__ import annotations

import logging
import math
from typing import List, Tuple

from agent_ai.shared.numeric import clamp01 as _clamp01

from .kinematics import hypot2, ttc_2d_s, ttc_longitudinal_s
from .interaction_predictor import InteractionSummary, interaction_risk_boost
from .motion_predictor import mode_collision_likelihood
from .schema import RiskSummary, TrackedObject

LOGGER = logging.getLogger(__name__)


def _ttc_risk_term(ttc: float | None) -> Tuple[float, List[str]]:
    """Continuous TTC contribution in [0, 0.45] with discrete tags for explainability."""
    if ttc is None:
        return 0.0, []
    tags: List[str] = []
    # Smooth inverse: critical near 0 s, negligible beyond ~8 s.
    term = 0.45 * math.exp(-max(0.0, float(ttc)) / 1.8)
    if ttc < 1.5:
        tags.append("ttc_critical")
    elif ttc < 3.0:
        tags.append("ttc_high")
    elif ttc < 5.0:
        tags.append("ttc_medium")
    else:
        tags.append("ttc_low")
    return float(term), tags


class RiskAssessmentEngine:
    def __init__(
        self,
        *,
        frontal_corridor_half_width_m: float = 2.5,
        emergency_distance_m: float = 4.0,
        caution_distance_m: float = 15.0,
        # Speed-adaptive caution: caution_m = max(base, d0 + v * t_headway)
        caution_time_headway_s: float = 1.6,
        caution_standstill_m: float = 6.0,
        emergency_time_headway_s: float = 0.6,
        emergency_standstill_m: float = 3.0,
        ego_speed_mps: float = 0.0,
    ) -> None:
        self.frontal_corridor_half_width_m = float(frontal_corridor_half_width_m)
        self.emergency_distance_m = float(emergency_distance_m)
        self.caution_distance_m = float(caution_distance_m)
        self.caution_time_headway_s = float(caution_time_headway_s)
        self.caution_standstill_m = float(caution_standstill_m)
        self.emergency_time_headway_s = float(emergency_time_headway_s)
        self.emergency_standstill_m = float(emergency_standstill_m)
        self._ego_speed_mps = float(ego_speed_mps)

    def set_ego_speed(self, speed_mps: float) -> None:
        self._ego_speed_mps = max(0.0, float(speed_mps))

    def _adaptive_caution_m(self) -> float:
        v = self._ego_speed_mps
        adaptive = self.caution_standstill_m + v * self.caution_time_headway_s
        return max(self.caution_distance_m, adaptive)

    def _adaptive_emergency_m(self) -> float:
        v = self._ego_speed_mps
        adaptive = self.emergency_standstill_m + v * self.emergency_time_headway_s
        return max(self.emergency_distance_m, adaptive)

    def _effective_ttc(self, track: TrackedObject) -> float | None:
        """Prefer multi-mode predicted TTC, then tracker TTC, then recompute."""
        if track.predicted_min_ttc_s is not None:
            return float(track.predicted_min_ttc_s)
        if track.ttc_seconds is not None:
            return float(track.ttc_seconds)
        pos = track.position_ego
        vel = track.velocity_ego
        ttc = ttc_2d_s(pos, vel)
        if ttc is None:
            ttc = ttc_longitudinal_s(pos, vel)
        return ttc

    def _closing_speed_term(self, track: TrackedObject) -> Tuple[float, List[str]]:
        """Bonus when object is approaching ego in 2-D."""
        px, py = float(track.position_ego[0]), float(track.position_ego[1])
        vx = float(track.velocity_ego[0]) if track.velocity_ego else 0.0
        vy = float(track.velocity_ego[1]) if len(track.velocity_ego) > 1 else 0.0
        range_m = hypot2(px, py)
        if range_m < 1e-3:
            return 0.15, ["point_blank"]
        range_rate = (px * vx + py * vy) / range_m
        if range_rate >= -0.15:
            return 0.0, []
        closing = -range_rate
        term = 0.12 * _clamp01(closing / 8.0)
        return float(term), ["closing"]

    def _score_track(self, track: TrackedObject) -> Tuple[float, List[str]]:
        score = 0.0
        tags: List[str] = []
        x = float(track.position_ego[0])
        y = float(track.position_ego[1])
        distance = float(track.distance_m)
        caution_m = self._adaptive_caution_m()
        emergency_m = self._adaptive_emergency_m()

        # Distance (horizon scales lightly with speed so high-speed scenes stay sensitive).
        horizon = max(40.0, 25.0 + 2.0 * self._ego_speed_mps)
        distance_term = _clamp01(1.0 - (distance / horizon))
        score += 0.28 * distance_term
        if distance < caution_m:
            tags.append("near_field")

        in_front = x > 0.0
        in_frontal_corridor = in_front and abs(y) <= self.frontal_corridor_half_width_m
        if in_front:
            score += 0.08
            tags.append("ahead")
        if in_frontal_corridor:
            score += 0.22
            tags.append("frontal_corridor")

        ttc = self._effective_ttc(track)
        ttc_term, ttc_tags = _ttc_risk_term(ttc)
        score += ttc_term
        tags.extend(ttc_tags)

        closing_term, closing_tags = self._closing_speed_term(track)
        score += closing_term
        tags.extend(closing_tags)

        # Multi-mode prediction envelope: mass on modes that approach ego.
        mode_mass = mode_collision_likelihood(track)
        if mode_mass > 0.05:
            score += 0.18 * mode_mass
            tags.append("predicted_mode_risk")
            if mode_mass > 0.45:
                tags.append("predicted_conflict_likely")
        if track.predicted_min_range_m is not None and track.predicted_min_range_m < caution_m:
            score += 0.08 * _clamp01(1.0 - track.predicted_min_range_m / max(1.0, caution_m))
            tags.append("predicted_near_miss")

        if track.class_group == "vru":
            score += 0.15
            tags.append("vru")
        elif track.class_group == "vehicle":
            score += 0.05
            tags.append("vehicle")

        if in_frontal_corridor and distance < emergency_m:
            score += 0.35
            tags.append("distance_critical")
        elif in_frontal_corridor and distance < caution_m:
            score += 0.10
            tags.append("distance_caution")

        # Lateral encroachment for objects nearly co-linear.
        if abs(y) < 1.2 and distance < caution_m:
            score += 0.08
            tags.append("lateral_close")

        if track.is_occluded_est:
            score *= 0.85
            tags.append("occluded_est")

        # Slight boost for low-confidence tracks (uncertain is riskier).
        if float(track.source_confidence) < 0.35:
            score = min(1.0, score + 0.05)
            tags.append("low_confidence")

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

    def evaluate(
        self,
        tracks: List[TrackedObject],
        *,
        ego_speed_mps: float | None = None,
        interaction: InteractionSummary | None = None,
    ) -> RiskSummary:
        if ego_speed_mps is not None:
            self.set_ego_speed(ego_speed_mps)

        highest_risk_score = 0.0
        highest_risk_level = "low"
        urgent_track_ids: List[int] = []
        front_hazard_track_id = None
        nearest_front_vehicle_distance = None
        nearest_vru_distance = None
        minimum_ttc = None
        flags: List[str] = []

        caution_m = self._adaptive_caution_m()
        interact_boost = interaction_risk_boost(interaction)

        for track in tracks:
            risk_score, tags = self._score_track(track)
            if interact_boost > 0.0:
                risk_score = _clamp01(risk_score + 0.12 * interact_boost)
                if "scene_interaction" not in tags:
                    tags.append("scene_interaction")
            risk_level = self._level(risk_score)
            track.risk_score = risk_score
            track.risk_level = risk_level
            track.reasoning_tags = tags

            if risk_level in {"high", "critical"}:
                urgent_track_ids.append(int(track.track_id))
            if risk_score > highest_risk_score:
                highest_risk_score = risk_score
                highest_risk_level = risk_level
            if (
                track.relative_sector.startswith("front")
                and front_hazard_track_id is None
                and risk_level in {"high", "critical"}
            ):
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
            ttc = self._effective_ttc(track)
            if ttc is not None:
                minimum_ttc = ttc if minimum_ttc is None else min(minimum_ttc, ttc)

        if highest_risk_level in {"high", "critical"}:
            flags.append("hazard_present")
        if minimum_ttc is not None and minimum_ttc < 2.0:
            flags.append("short_ttc")
        if nearest_vru_distance is not None and nearest_vru_distance < 10.0:
            flags.append("vru_nearby")
        if nearest_front_vehicle_distance is not None and nearest_front_vehicle_distance < caution_m:
            flags.append("front_vehicle_close")
        if self._ego_speed_mps > 12.0 and highest_risk_level in {"medium", "high", "critical"}:
            flags.append("high_speed_risk")
        if interaction is not None and interaction.max_interaction_severity > 0.35:
            flags.append("interaction_scene")
            if interact_boost > 0.0:
                # Mild global uplift so summary reflects multi-agent pressure.
                highest_risk_score = _clamp01(highest_risk_score + 0.08 * interact_boost)
                highest_risk_level = self._level(highest_risk_score)

        LOGGER.info(
            "Risk summary tracks=%d highest=%s score=%.3f urgent=%s ego_v=%.2f caution=%.1f interact=%.2f",
            len(tracks),
            highest_risk_level,
            highest_risk_score,
            urgent_track_ids,
            self._ego_speed_mps,
            caution_m,
            interact_boost,
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
