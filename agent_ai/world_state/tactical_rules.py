"""Rule-based tactical policy with FSM hysteresis and IDM-style follow speed."""
from __future__ import annotations

import logging

from .kinematics import idm_desired_speed_mps, time_headway_gap_m
from .schema import DecisionIntent, WorldState
from .soft_constraint_arbiter import (
    ManeuverCandidate,
    SoftConstraintArbiter,
    build_soft_components,
    inverse_gap_cost,
    ttc_cost,
)

# TYPE_CHECKING-friendly optional import for CostMemory (runtime used only as duck type).
try:
    from .cost_memory import CostMemory
except Exception:  # pragma: no cover
    CostMemory = object  # type: ignore

LOGGER = logging.getLogger(__name__)

# Maneuver priority for hysteresis: higher = more urgent / sticky escape.
_MANEUVER_PRIORITY = {
    "emergency_stop": 100,
    "stop": 90,
    "yield": 80,
    "follow": 50,
    "slow_down": 40,
    "lane_change_left": 35,
    "lane_change_right": 35,
    "keep_lane": 10,
}

# Minimum consecutive frames in a non-critical maneuver before allowing a peer switch.
_DEFAULT_HOLD_FRAMES = 3


class RuleBasedTacticalPolicy:
    """
    Lightweight tactical FSM.

    States map 1:1 to maneuver labels. Hysteresis prevents frame-to-frame
    flip-flops between peer maneuvers (e.g. left/right LC or follow/slow_down)
    while still allowing immediate escalation to stop / yield / emergency.
    """

    def __init__(
        self,
        *,
        cruise_speed_mps: float = 8.0,
        minimum_roll_speed_mps: float = 2.0,
        hold_frames: int = _DEFAULT_HOLD_FRAMES,
        follow_time_gap_s: float = 1.4,
        follow_d0_m: float = 3.5,
        lc_trigger_distance_m: float = 18.0,
        stop_distance_m: float = 5.0,
        vru_hard_m: float = 5.0,
        vru_soft_m: float = 8.0,
    ) -> None:
        self.cruise_speed_mps = float(cruise_speed_mps)
        self.minimum_roll_speed_mps = float(minimum_roll_speed_mps)
        self.hold_frames = int(max(1, hold_frames))
        self.follow_time_gap_s = float(follow_time_gap_s)
        self.follow_d0_m = float(follow_d0_m)
        self.lc_trigger_distance_m = float(lc_trigger_distance_m)
        self.stop_distance_m = float(stop_distance_m)
        self.vru_hard_m = float(vru_hard_m)
        self.vru_soft_m = float(vru_soft_m)

        self._prev_maneuver: str = "keep_lane"
        self._prev_lane_preference: str = "keep_current"
        self._stable_count: int = 0
        self._cost_memory = None

    def reset(self) -> None:
        self._prev_maneuver = "keep_lane"
        self._prev_lane_preference = "keep_current"
        self._stable_count = 0
        self._cost_memory = None

    def _arbiter(self) -> SoftConstraintArbiter:
        mem = self._cost_memory
        if mem is not None and hasattr(mem, "arbiter_for_scene"):
            return mem.arbiter_for_scene()
        return SoftConstraintArbiter()

    def _leader_speed_mps(self, world_state: WorldState) -> float | None:
        front = world_state.scene.nearest_front_vehicle
        if not front:
            return None
        if "speed_mps" in front and front["speed_mps"] is not None:
            return max(0.0, float(front["speed_mps"]))
        # Approximate absolute leader speed from relative ego-frame velocity if present.
        vel = front.get("velocity_ego")
        if isinstance(vel, (list, tuple)) and len(vel) >= 1:
            # In ego frame, leader absolute ≈ ego + relative_x (longitudinal).
            return max(0.0, float(world_state.ego.speed_mps) + float(vel[0]))
        return None

    def _raw_decide(self, world_state: WorldState) -> tuple[str, float, str, list[str], list[str], float]:
        """Return (maneuver, target_speed, lane_pref, reasoning, constraints, confidence)."""
        ego_speed = float(world_state.ego.speed_mps)
        scene = world_state.scene
        risk = world_state.risk_summary
        constraints = list(world_state.decision_context["hard_constraints"])
        reasoning_tags: list[str] = []
        lane_preference = "keep_current"
        maneuver = "keep_lane"
        confidence = 0.55

        # Adaptive geometric thresholds from ego speed.
        stop_dist = max(
            self.stop_distance_m,
            time_headway_gap_m(ego_speed, t_gap_s=0.5, d0_m=self.stop_distance_m),
        )
        lc_trigger = max(
            self.lc_trigger_distance_m,
            time_headway_gap_m(ego_speed, t_gap_s=1.8, d0_m=12.0),
        )
        follow_close = time_headway_gap_m(
            ego_speed, t_gap_s=self.follow_time_gap_s, d0_m=self.follow_d0_m + 4.0
        )

        front_vehicle = scene.nearest_front_vehicle
        front_vehicle_distance = (
            None if front_vehicle is None else float(front_vehicle["distance_m"])
        )
        leader_speed = self._leader_speed_mps(world_state)

        def idm_speed(dist: float | None) -> float:
            return idm_desired_speed_mps(
                ego_speed_mps=ego_speed,
                leader_distance_m=dist,
                leader_speed_mps=leader_speed,
                v0_mps=self.cruise_speed_mps,
                t_gap_s=self.follow_time_gap_s,
                d0_m=self.follow_d0_m,
            )

        if risk.highest_risk_level == "critical":
            maneuver = "emergency_stop"
            target_speed = 0.0
            confidence = 0.95
            reasoning_tags.extend(["critical_risk", "hard_stop"])
            constraints.append("do_not_accelerate")
        elif risk.nearest_vru_distance_m is not None and risk.nearest_vru_distance_m < self.vru_soft_m:
            maneuver = "yield"
            if risk.nearest_vru_distance_m < self.vru_hard_m:
                target_speed = 0.0
            else:
                target_speed = min(1.5, idm_speed(risk.nearest_vru_distance_m))
            confidence = 0.85
            reasoning_tags.extend(["vru_priority", "yield_zone"])
            constraints.append("protect_vru")
        elif scene.front_free_space_m is not None and scene.front_free_space_m < stop_dist:
            maneuver = "stop"
            target_speed = 0.0
            confidence = 0.88
            reasoning_tags.extend(["front_blocked", "close_obstacle", "speed_adaptive_stop"])
            constraints.append("respect_front_obstacle_buffer")
        elif front_vehicle_distance is not None and front_vehicle_distance < lc_trigger:
            # Soft-constraint arbiter: follow / slow_down / LC left / LC right.
            arb = self._arbiter()
            front_ttc = risk.minimum_ttc_seconds
            gap_c = inverse_gap_cost(
                front_vehicle_distance,
                comfortable_m=lc_trigger,
                critical_m=follow_close * 0.7,
            )
            left_occ = float(scene.left_side_occupancy)
            right_occ = float(scene.right_side_occupancy)
            left_clear = left_occ < 0.30
            right_clear = right_occ < 0.30
            escape = front_vehicle_distance < follow_close * 1.15

            candidates = [
                ManeuverCandidate(
                    maneuver="follow",
                    hard_ok=True,
                    components=build_soft_components(
                        safety=ttc_cost(front_ttc) * 0.5,
                        gap=gap_c,
                        progress=0.25 if escape else 0.1,
                        comfort=0.1,
                        risk=0.15 * gap_c,
                        hysteresis=-0.1 if self._prev_maneuver == "follow" else 0.0,
                    ),
                    tags=["arbiter_follow", "idm_speed"],
                ),
                ManeuverCandidate(
                    maneuver="slow_down",
                    hard_ok=True,
                    components=build_soft_components(
                        safety=ttc_cost(front_ttc) * 0.4,
                        gap=gap_c * 0.8,
                        progress=0.2,
                        comfort=0.15,
                        risk=0.1,
                        hysteresis=-0.05 if self._prev_maneuver == "slow_down" else 0.05,
                    ),
                    tags=["arbiter_slow", "idm_speed"],
                ),
                ManeuverCandidate(
                    maneuver="lane_change_left",
                    hard_ok=left_clear and escape,
                    hard_reason=None if (left_clear and escape) else "left_not_clear_or_no_escape",
                    components=build_soft_components(
                        safety=0.1 + 0.4 * left_occ,
                        gap=0.2 + 0.5 * left_occ,
                        progress=0.05 if escape else 0.4,
                        comfort=0.35 + 0.3 * left_occ,
                        risk=0.15 * left_occ,
                        hysteresis=-0.12 if self._prev_maneuver == "lane_change_left" else 0.1,
                        preference=0.05,
                    ),
                    tags=["arbiter_lc_left", "idm_speed"],
                ),
                ManeuverCandidate(
                    maneuver="lane_change_right",
                    hard_ok=right_clear and escape,
                    hard_reason=None if (right_clear and escape) else "right_not_clear_or_no_escape",
                    components=build_soft_components(
                        safety=0.1 + 0.4 * right_occ,
                        gap=0.2 + 0.5 * right_occ,
                        progress=0.05 if escape else 0.4,
                        comfort=0.35 + 0.3 * right_occ,
                        risk=0.15 * right_occ,
                        hysteresis=-0.12 if self._prev_maneuver == "lane_change_right" else 0.1,
                        preference=0.05,
                    ),
                    tags=["arbiter_lc_right", "idm_speed"],
                ),
            ]
            best = arb.select(candidates, fallback="follow")
            maneuver = best.maneuver
            target_speed = idm_speed(front_vehicle_distance)
            reasoning_tags.extend(arb.explain(best))
            reasoning_tags.append("soft_constraint_arbiter")
            if maneuver == "lane_change_left":
                lane_preference = "prefer_left"
                confidence = 0.64
                constraints.append("lane_change_requires_validation")
            elif maneuver == "lane_change_right":
                lane_preference = "prefer_right"
                confidence = 0.64
                constraints.append("lane_change_requires_validation")
            elif maneuver == "follow":
                confidence = 0.78
            else:
                confidence = 0.70
        elif risk.highest_risk_level == "high":
            maneuver = "slow_down"
            target_speed = idm_speed(front_vehicle_distance)
            target_speed = min(target_speed, max(self.minimum_roll_speed_mps, ego_speed * 0.6))
            confidence = 0.72
            reasoning_tags.extend(["high_risk_scene", "precautionary_brake", "idm_speed"])
        elif risk.highest_risk_level == "medium":
            maneuver = "slow_down"
            target_speed = idm_speed(front_vehicle_distance)
            target_speed = min(target_speed, max(self.minimum_roll_speed_mps, ego_speed * 0.8))
            confidence = 0.65
            reasoning_tags.extend(["medium_risk_scene", "idm_speed"])
        else:
            maneuver = "keep_lane"
            target_speed = idm_speed(None)
            confidence = 0.60
            reasoning_tags.extend(["cruise", "idm_free_road"])

        if lane_preference == "keep_current":
            if scene.left_side_occupancy + 0.15 < scene.right_side_occupancy:
                lane_preference = "prefer_left"
            elif scene.right_side_occupancy + 0.15 < scene.left_side_occupancy:
                lane_preference = "prefer_right"

        # Clamp floor for non-stop maneuvers.
        if maneuver not in {"emergency_stop", "stop", "yield"}:
            target_speed = max(self.minimum_roll_speed_mps, min(self.cruise_speed_mps, float(target_speed)))
        else:
            target_speed = max(0.0, min(self.cruise_speed_mps, float(target_speed)))

        return maneuver, float(target_speed), lane_preference, reasoning_tags, constraints, float(confidence)

    def _apply_hysteresis(self, candidate: str, reasoning_tags: list[str]) -> str:
        prev = self._prev_maneuver
        if candidate == prev:
            self._stable_count += 1
            return candidate

        prev_pri = _MANEUVER_PRIORITY.get(prev, 0)
        cand_pri = _MANEUVER_PRIORITY.get(candidate, 0)

        # Immediate escalation always allowed (safety).
        if cand_pri >= prev_pri + 20 or cand_pri >= 80:
            reasoning_tags.append("hysteresis_escalation")
            self._stable_count = 1
            return candidate

        # Immediate de-escalation from emergency/stop when clear.
        if prev_pri >= 80 and cand_pri < 50:
            # Require at least one stable frame of lower urgency before fully dropping
            # from stop/yield — still allow, but tag it.
            reasoning_tags.append("hysteresis_deescalate")
            self._stable_count = 1
            return candidate

        # Peer / minor switch: hold previous until stable long enough.
        if self._stable_count < self.hold_frames and abs(cand_pri - prev_pri) < 20:
            reasoning_tags.append("hysteresis_hold")
            self._stable_count += 1
            return prev

        reasoning_tags.append("hysteresis_switch")
        self._stable_count = 1
        return candidate

    def decide(self, world_state: WorldState, *, cost_memory: "CostMemory | None" = None) -> DecisionIntent:
        # Optional P2 adaptive weights for the soft arbiter path inside _raw_decide.
        self._cost_memory = cost_memory
        maneuver, target_speed, lane_preference, reasoning_tags, constraints, confidence = self._raw_decide(
            world_state
        )
        maneuver = self._apply_hysteresis(maneuver, reasoning_tags)
        self._prev_maneuver = maneuver
        self._prev_lane_preference = lane_preference

        # If hysteresis held a previous LC, keep its lane preference sticky.
        if maneuver == "lane_change_left":
            lane_preference = "prefer_left"
        elif maneuver == "lane_change_right":
            lane_preference = "prefer_right"

        if cost_memory is not None:
            reasoning_tags.append(f"cost_profile_{cost_memory.last_profile}")
            reasoning_tags.append("adaptive_weights")

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
            "Decision frame=%d maneuver=%s target_speed=%.2f confidence=%.2f hold=%d",
            decision.frame_id,
            decision.maneuver,
            decision.target_speed_mps,
            decision.confidence,
            self._stable_count,
        )
        return decision
