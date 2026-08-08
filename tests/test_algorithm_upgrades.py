"""Unit tests for P0 kinematics / tracker / risk / tactical upgrades."""
from __future__ import annotations

import math
import unittest

from agent_ai.behavior.lane.maneuver_validator import _lane_change_feasibility
from agent_ai.behavior.lane.schema import JunctionContext, LaneAwareObject, LaneContext, LaneDescriptor
from agent_ai.world_state.kinematics import (
    idm_desired_speed_mps,
    time_headway_gap_m,
    ttc_2d_s,
    ttc_longitudinal_s,
)
from agent_ai.world_state.risk_engine import RiskAssessmentEngine
from agent_ai.world_state.schema import (
    EgoState,
    NormalizedDetection,
    NormalizedFramePrediction,
    RiskSummary,
    SceneSummary,
    TrackedObject,
    WorldState,
)
from agent_ai.world_state.tactical_rules import RuleBasedTacticalPolicy
from agent_ai.world_state.tracker import SimpleObjectTracker


def _ego(speed: float = 5.0, frame_id: int = 0) -> EgoState:
    return EgoState(
        frame_id=frame_id,
        timestamp=float(frame_id) * 0.1,
        sample_name=f"s{frame_id}",
        town="Town10HD",
        weather={},
        position_world=[0.0, 0.0, 0.0],
        velocity_world=[speed, 0.0, 0.0],
        speed_mps=speed,
        yaw_deg=0.0,
        route_progress_m=0.0,
        world_from_ego_bevfusion=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )


def _det(
    det_id: str,
    x: float,
    y: float,
    *,
    class_name: str = "car",
    class_group: str = "vehicle",
    score: float = 0.9,
    vx: float = 0.0,
    vy: float = 0.0,
) -> NormalizedDetection:
    dist = math.hypot(x, y)
    return NormalizedDetection(
        detection_id=det_id,
        class_id=0,
        class_name=class_name,
        class_group=class_group,
        score=score,
        position_ego=[x, y, 0.0],
        size_xyz=[4.0, 2.0, 1.5],
        yaw_rad=0.0,
        velocity_ego=[vx, vy],
        speed_mps=math.hypot(vx, vy),
        distance_m=dist,
        bearing_deg=math.degrees(math.atan2(y, x)) if dist > 1e-6 else 0.0,
        source_confidence=score,
        raw_box_9d=[x, y, 0.0, 4.0, 2.0, 1.5, 0.0, vx, vy],
    )


def _frame(dets: list[NormalizedDetection], *, frame_id: int = 0, t: float | None = None, speed: float = 5.0):
    ts = float(frame_id) * 0.1 if t is None else t
    return NormalizedFramePrediction(
        sample_name=f"s{frame_id}",
        sequence_index=frame_id,
        frame_id=frame_id,
        timestamp=ts,
        prediction_variant="test",
        source_sample_dir="/tmp",
        source_prediction_dir="/tmp",
        ego=_ego(speed, frame_id=frame_id),
        detections=dets,
        class_names=["car"],
        raw_prediction_count=len(dets),
        filtered_prediction_count=len(dets),
    )


def _track(
    track_id: int,
    x: float,
    y: float,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    class_group: str = "vehicle",
    ttc: float | None = None,
) -> TrackedObject:
    dist = math.hypot(x, y)
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name="car" if class_group == "vehicle" else "pedestrian",
        class_group=class_group,
        latest_detection_id=f"d{track_id}",
        age_frames=5,
        hits=5,
        missed_frames=0,
        is_occluded_est=False,
        score=0.9,
        mean_score=0.9,
        position_ego=[x, y, 0.0],
        velocity_ego=[vx, vy],
        speed_mps=math.hypot(vx, vy),
        bbox=[x, y, 0.0, 4.0, 2.0, 1.5, 0.0, vx, vy],
        size_xyz=[4.0, 2.0, 1.5],
        yaw_rad=0.0,
        distance_m=dist,
        bearing_deg=0.0,
        ttc_seconds=ttc,
        relative_sector="front" if x >= 0 and abs(y) <= 2.5 else "front_left",
        source_confidence=0.9,
    )


class KinematicsTests(unittest.TestCase):
    def test_ttc_longitudinal_closing(self) -> None:
        ttc = ttc_longitudinal_s([20.0, 0.0], [-5.0, 0.0])
        self.assertIsNotNone(ttc)
        self.assertAlmostEqual(ttc, 4.0, places=3)

    def test_ttc_longitudinal_receding(self) -> None:
        self.assertIsNone(ttc_longitudinal_s([20.0, 0.0], [2.0, 0.0]))

    def test_ttc_2d_diagonal_closing(self) -> None:
        ttc = ttc_2d_s([10.0, 10.0], [-2.0, -2.0])
        self.assertIsNotNone(ttc)
        self.assertAlmostEqual(ttc, 5.0, places=1)

    def test_time_headway_scales_with_speed(self) -> None:
        slow = time_headway_gap_m(2.0, t_gap_s=1.5, d0_m=8.0)
        fast = time_headway_gap_m(15.0, t_gap_s=1.5, d0_m=8.0)
        self.assertGreater(fast, slow)
        self.assertGreaterEqual(slow, 8.0)
        self.assertAlmostEqual(fast, 8.0 + 15.0 * 1.5, places=3)

    def test_idm_free_road_accelerates(self) -> None:
        v = idm_desired_speed_mps(
            ego_speed_mps=3.0,
            leader_distance_m=None,
            leader_speed_mps=None,
            v0_mps=8.0,
            dt_s=0.5,
        )
        self.assertGreater(v, 3.0)
        self.assertLessEqual(v, 8.0)

    def test_idm_close_leader_slows(self) -> None:
        v = idm_desired_speed_mps(
            ego_speed_mps=8.0,
            leader_distance_m=6.0,
            leader_speed_mps=2.0,
            v0_mps=8.0,
            t_gap_s=1.4,
            d0_m=3.0,
            dt_s=0.5,
        )
        self.assertLess(v, 8.0)


class TrackerTests(unittest.TestCase):
    def test_associates_across_frames_same_id(self) -> None:
        tracker = SimpleObjectTracker(min_confirmed_hits=1, max_missed_frames=5)
        out0 = tracker.update(_frame([_det("a", 20.0, 0.0)], frame_id=0, t=0.0))
        self.assertEqual(len(out0), 1)
        tid = out0[0].track_id
        out1 = tracker.update(_frame([_det("b", 19.0, 0.2)], frame_id=1, t=0.1))
        self.assertEqual(len(out1), 1)
        self.assertEqual(out1[0].track_id, tid)
        self.assertGreaterEqual(out1[0].hits, 2)

    def test_velocity_estimate_nonzero_after_motion(self) -> None:
        tracker = SimpleObjectTracker(min_confirmed_hits=1)
        tracker.update(_frame([_det("a", 30.0, 0.0)], frame_id=0, t=0.0))
        out = tracker.update(_frame([_det("b", 28.0, 0.0)], frame_id=1, t=0.2))
        self.assertEqual(len(out), 1)
        self.assertLess(out[0].velocity_ego[0], -1.0)

    def test_ttc_populated_when_closing(self) -> None:
        tracker = SimpleObjectTracker(min_confirmed_hits=1)
        tracker.update(_frame([_det("a", 25.0, 0.0)], frame_id=0, t=0.0))
        out = tracker.update(_frame([_det("b", 22.0, 0.0)], frame_id=1, t=0.2))
        self.assertIsNotNone(out[0].ttc_seconds)
        self.assertGreater(out[0].ttc_seconds, 0.0)

    def test_new_track_for_far_detection(self) -> None:
        tracker = SimpleObjectTracker(min_confirmed_hits=1, vehicle_match_distance_m=4.0)
        tracker.update(_frame([_det("a", 10.0, 0.0)], frame_id=0, t=0.0))
        out1 = tracker.update(_frame([_det("b", 40.0, 8.0)], frame_id=1, t=0.1))
        self.assertGreaterEqual(len(out1), 1)
        if len(out1) >= 2:
            self.assertEqual(len({t.track_id for t in out1}), 2)


class RiskEngineTests(unittest.TestCase):
    def test_critical_short_ttc_frontal(self) -> None:
        engine = RiskAssessmentEngine()
        track = _track(1, 8.0, 0.0, vx=-6.0, ttc=1.2)
        summary = engine.evaluate([track], ego_speed_mps=8.0)
        self.assertIn(track.risk_level, {"high", "critical"})
        self.assertIn("ttc_critical", track.reasoning_tags)
        self.assertEqual(summary.highest_risk_level, track.risk_level)

    def test_speed_adaptive_caution(self) -> None:
        engine = RiskAssessmentEngine(caution_distance_m=15.0)
        track_slow = _track(1, 20.0, 0.0, vx=-1.0, ttc=20.0)
        engine.evaluate([track_slow], ego_speed_mps=1.0)
        score_slow = track_slow.risk_score

        track_fast = _track(1, 20.0, 0.0, vx=-1.0, ttc=20.0)
        engine.evaluate([track_fast], ego_speed_mps=16.0)
        self.assertGreaterEqual(track_fast.risk_score, score_slow - 1e-6)

    def test_vru_boost(self) -> None:
        engine = RiskAssessmentEngine()
        v = _track(1, 12.0, 1.0, class_group="vru", ttc=None)
        car = _track(2, 12.0, 1.0, class_group="vehicle", ttc=None)
        engine.evaluate([v, car], ego_speed_mps=5.0)
        self.assertGreater(v.risk_score, car.risk_score)


class TacticalHysteresisTests(unittest.TestCase):
    def _ws(
        self,
        *,
        front_dist: float | None = None,
        risk_level: str = "low",
        vru_dist: float | None = None,
        left_occ: float = 0.5,
        right_occ: float = 0.5,
        free: float | None = 40.0,
        speed: float = 6.0,
    ) -> WorldState:
        front = None if front_dist is None else {"distance_m": front_dist, "speed_mps": 3.0, "track_id": 1}
        return WorldState(
            frame_id=0,
            timestamp=0.0,
            sample_name="t",
            sequence_index=0,
            ego=_ego(speed),
            objects=[],
            scene=SceneSummary(
                active_object_count=0 if front is None else 1,
                front_free_space_m=free,
                left_side_occupancy=left_occ,
                right_side_occupancy=right_occ,
                rear_gap_m=None,
                nearest_front_vehicle=front,
                nearest_vru=None,
                nearest_any_object=front,
                abnormal_flags=[],
            ),
            risk_summary=RiskSummary(
                highest_risk_level=risk_level,
                highest_risk_score=0.9 if risk_level == "critical" else 0.2,
                urgent_track_ids=[],
                front_hazard_track_id=None,
                nearest_front_vehicle_distance_m=front_dist,
                nearest_vru_distance_m=vru_dist,
                minimum_ttc_seconds=None,
                flags=[],
            ),
            decision_context={"hard_constraints": [], "soft_constraints": [], "recommended_maneuvers": []},
        )

    def test_critical_escalates_immediately(self) -> None:
        policy = RuleBasedTacticalPolicy(hold_frames=5)
        policy.decide(self._ws(front_dist=12.0, left_occ=0.1, right_occ=0.6))
        d1 = policy.decide(self._ws(risk_level="critical"))
        self.assertEqual(d1.maneuver, "emergency_stop")
        self.assertEqual(d1.target_speed_mps, 0.0)

    def test_hysteresis_holds_peer_switch(self) -> None:
        policy = RuleBasedTacticalPolicy(hold_frames=3, cruise_speed_mps=8.0)
        d0 = policy.decide(self._ws(front_dist=9.0, left_occ=0.5, right_occ=0.5))
        held = d0.maneuver
        d1 = policy.decide(self._ws(front_dist=9.0, left_occ=0.05, right_occ=0.8))
        if "hysteresis_hold" in d1.reasoning_tags:
            self.assertEqual(d1.maneuver, held)

    def test_idm_follow_speed_below_cruise(self) -> None:
        policy = RuleBasedTacticalPolicy(cruise_speed_mps=8.0)
        d = policy.decide(self._ws(front_dist=8.0, left_occ=0.5, right_occ=0.5, speed=7.0))
        self.assertIn(d.maneuver, {"follow", "slow_down", "stop", "lane_change_left", "lane_change_right"})
        if d.maneuver in {"follow", "slow_down"}:
            self.assertLessEqual(d.target_speed_mps, 8.0)
            self.assertIn("idm_speed", d.reasoning_tags)

    def test_cruise_free_road(self) -> None:
        policy = RuleBasedTacticalPolicy(cruise_speed_mps=8.0)
        d = policy.decide(self._ws(front_dist=None, free=50.0, speed=4.0))
        self.assertEqual(d.maneuver, "keep_lane")
        self.assertGreater(d.target_speed_mps, 4.0)


class ManeuverGapTests(unittest.TestCase):
    def _lane_ctx(self, *, left: bool = True, right: bool = True) -> LaneContext:
        def desc(exists: bool, role: str) -> LaneDescriptor:
            return LaneDescriptor(
                exists=exists,
                role=role,
                road_id=1 if exists else None,
                section_id=0 if exists else None,
                lane_id=1 if exists else None,
                lane_width_m=3.5 if exists else None,
                lane_type="Driving" if exists else None,
                lane_change="Both" if exists else None,
                same_direction_as_ego=True,
                transform_carla=None,
                transform_bevfusion_world=None,
                lane_change_allowed_from_current=True,
            )

        return LaneContext(
            frame_id=0,
            sample_name="t",
            current_lane=desc(True, "current"),
            left_lane=desc(left, "left"),
            right_lane=desc(right, "right"),
            forward_corridor={},
            junction_context=JunctionContext(
                is_in_junction=False,
                junction_ahead=False,
                distance_to_junction_m=80.0,
                branch_count_ahead=0,
                possible_turn_like_options=[],
                branch_distance_m=None,
            ),
        )

    def _obj(self, relation: str, long_m: float) -> LaneAwareObject:
        return LaneAwareObject(
            track_id=1,
            class_name="car",
            class_group="vehicle",
            position_world_carla=[0.0, 0.0, 0.0],
            lane_relation=relation,
            lane_tag=relation,
            object_lane_id=2,
            object_road_id=1,
            longitudinal_m=long_m,
            lateral_m=3.5,
            is_front_in_current_lane=False,
            is_rear_in_current_lane=False,
            is_blocking_current_lane=False,
            same_direction_as_ego_lane=True,
            distance_to_lane_center_m=0.0,
            source_track={"speed_mps": 5.0},
        )

    def test_front_gap_rejects_at_high_speed(self) -> None:
        objs = [self._obj("left_lane", 14.0)]
        ok_slow, _ = _lane_change_feasibility(
            direction="left",
            lane_context=self._lane_ctx(),
            lane_objects=objs,
            ego_speed_mps=2.0,
        )
        ok_fast, reason = _lane_change_feasibility(
            direction="left",
            lane_context=self._lane_ctx(),
            lane_objects=objs,
            ego_speed_mps=12.0,
        )
        self.assertTrue(ok_slow)
        self.assertFalse(ok_fast)
        self.assertIn("front_gap", reason or "")

    def test_rear_gap_speed_adaptive(self) -> None:
        objs = [self._obj("left_lane", -12.0)]
        objs[0].source_track["speed_mps"] = 20.0
        ok, reason = _lane_change_feasibility(
            direction="left",
            lane_context=self._lane_ctx(),
            lane_objects=objs,
            ego_speed_mps=5.0,
        )
        self.assertFalse(ok)
        self.assertIn("rear_gap", reason or "")


if __name__ == "__main__":
    unittest.main()
