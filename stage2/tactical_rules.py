from __future__ import annotations

import logging

from .object_schema import DecisionIntent, WorldState

LOGGER = logging.getLogger(__name__)


class RuleBasedTacticalPolicy:
    def __init__(
        self,
        *,
        cruise_speed_mps: float = 8.0,
        minimum_roll_speed_mps: float = 2.0,
    ) -> None:
        self.cruise_speed_mps = float(cruise_speed_mps)
        self.minimum_roll_speed_mps = float(minimum_roll_speed_mps)

    def decide(self, world_state: WorldState) -> DecisionIntent:
        ego_speed = float(world_state.ego.speed_mps)
        scene = world_state.scene
        risk = world_state.risk_summary
        constraints = list(world_state.decision_context["hard_constraints"])
        reasoning_tags: list[str] = []
        lane_preference = "keep_current"
        maneuver = "keep_lane"
        confidence = 0.55
        target_speed = min(self.cruise_speed_mps, max(self.minimum_roll_speed_mps, ego_speed))

        if risk.highest_risk_level == "critical":
            maneuver = "emergency_stop"
            target_speed = 0.0
            confidence = 0.95
            reasoning_tags.extend(["critical_risk", "hard_stop"])
            constraints.append("do_not_accelerate")
        elif risk.nearest_vru_distance_m is not None and risk.nearest_vru_distance_m < 8.0:
            maneuver = "yield"
            target_speed = 0.0 if risk.nearest_vru_distance_m < 5.0 else 1.5
            confidence = 0.85
            reasoning_tags.extend(["vru_priority", "yield_zone"])
            constraints.append("protect_vru")
        elif scene.front_free_space_m is not None and scene.front_free_space_m < 5.0:
            maneuver = "stop"
            target_speed = 0.0
            confidence = 0.88
            reasoning_tags.extend(["front_blocked", "close_obstacle"])
            constraints.append("respect_front_obstacle_buffer")
        elif scene.nearest_front_vehicle is not None and scene.nearest_front_vehicle["distance_m"] < 18.0:
            front_vehicle_distance = float(scene.nearest_front_vehicle["distance_m"])
            if scene.left_side_occupancy + 0.25 < scene.right_side_occupancy and scene.left_side_occupancy < 0.25:
                maneuver = "lane_change_left"
                lane_preference = "prefer_left"
                target_speed = min(self.cruise_speed_mps, max(self.minimum_roll_speed_mps, ego_speed))
                confidence = 0.62
                reasoning_tags.extend(["front_vehicle_slow", "left_gap_preferred"])
                constraints.append("lane_change_requires_validation")
            elif scene.right_side_occupancy + 0.25 < scene.left_side_occupancy and scene.right_side_occupancy < 0.25:
                maneuver = "lane_change_right"
                lane_preference = "prefer_right"
                target_speed = min(self.cruise_speed_mps, max(self.minimum_roll_speed_mps, ego_speed))
                confidence = 0.62
                reasoning_tags.extend(["front_vehicle_slow", "right_gap_preferred"])
                constraints.append("lane_change_requires_validation")
            elif front_vehicle_distance < 10.0:
                maneuver = "follow"
                target_speed = max(0.5, min(self.cruise_speed_mps, 0.35 * front_vehicle_distance))
                confidence = 0.78
                reasoning_tags.extend(["lead_vehicle_present", "follow_gap_control"])
            else:
                maneuver = "slow_down"
                target_speed = max(self.minimum_roll_speed_mps, min(self.cruise_speed_mps, ego_speed * 0.7))
                confidence = 0.70
                reasoning_tags.extend(["front_vehicle_present", "reduce_speed"])
        elif risk.highest_risk_level == "high":
            maneuver = "slow_down"
            target_speed = max(self.minimum_roll_speed_mps, min(self.cruise_speed_mps, ego_speed * 0.6))
            confidence = 0.72
            reasoning_tags.extend(["high_risk_scene", "precautionary_brake"])
        elif risk.highest_risk_level == "medium":
            maneuver = "slow_down"
            target_speed = max(self.minimum_roll_speed_mps, min(self.cruise_speed_mps, ego_speed * 0.8))
            confidence = 0.65
            reasoning_tags.extend(["medium_risk_scene"])
        else:
            maneuver = "keep_lane"
            target_speed = self.cruise_speed_mps
            confidence = 0.60
            reasoning_tags.extend(["cruise"])

        if lane_preference == "keep_current":
            if scene.left_side_occupancy + 0.15 < scene.right_side_occupancy:
                lane_preference = "prefer_left"
            elif scene.right_side_occupancy + 0.15 < scene.left_side_occupancy:
                lane_preference = "prefer_right"

        decision = DecisionIntent(
            frame_id=world_state.frame_id,
            timestamp=world_state.timestamp,
            maneuver=maneuver,
            target_speed_mps=float(target_speed),
            lane_preference=lane_preference,
            constraints=sorted(set(constraints)),
            confidence=float(confidence),
            reasoning_tags=reasoning_tags,
        )
        LOGGER.info(
            "Decision frame=%d maneuver=%s target_speed=%.2f confidence=%.2f",
            decision.frame_id,
            decision.maneuver,
            decision.target_speed_mps,
            decision.confidence,
        )
        return decision
