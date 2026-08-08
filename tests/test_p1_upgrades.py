"""Unit tests for P1: multi-mode prediction, cost-based LC, soft arbiter."""
from __future__ import annotations

import math
import unittest

from agent_ai.behavior.lane.lane_change_cost import evaluate_lane_change_candidates
from agent_ai.behavior.lane.schema import JunctionContext, LaneAwareObject, LaneContext, LaneDescriptor
from agent_ai.behavior.route.lane_change_staging import determine_lane_change_stage, route_demand_direction
from agent_ai.world_state.motion_predictor import (
    annotate_tracks_with_prediction,
    mode_collision_likelihood,
    predict_modes_for_track,
)
from agent_ai.world_state.schema import TrackedObject
from agent_ai.world_state.soft_constraint_arbiter import (
    ManeuverCandidate,
    SoftConstraintArbiter,
    build_soft_components,
    inverse_gap_cost,
    ttc_cost,
)


def _track(
    track_id: int,
    x: float,
    y: float,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    class_group: str = "vehicle",
) -> TrackedObject:
    dist = math.hypot(x, y)
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name="car" if class_group == "vehicle" else class_group,
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
        ttc_seconds=None,
        relative_sector="front",
        source_confidence=0.9,
    )


def _lane_ctx(left: bool = True, right: bool = True) -> LaneContext:
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


def _obj(relation: str, long_m: float, **src) -> LaneAwareObject:
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
        source_track={"speed_mps": 5.0, **src},
    )


class MotionPredictorTests(unittest.TestCase):
    def test_modes_normalized_and_cover_cv(self) -> None:
        track = _track(1, 20.0, 0.0, vx=-4.0, vy=0.0)
        modes = predict_modes_for_track(track)
        self.assertGreaterEqual(len(modes), 2)
        ids = {m.mode_id for m in modes}
        self.assertIn("cv", ids)
        total_p = sum(m.probability for m in modes)
        self.assertAlmostEqual(total_p, 1.0, places=5)
        for m in modes:
            self.assertGreaterEqual(len(m.waypoints_ego), 2)
            self.assertEqual(len(m.waypoints_ego[0]), 3)

    def test_closing_track_has_ttc_envelope(self) -> None:
        track = _track(1, 15.0, 0.0, vx=-5.0, vy=0.0)
        annotate_tracks_with_prediction([track])
        self.assertTrue(track.predicted_modes)
        self.assertIsNotNone(track.predicted_min_ttc_s)
        self.assertGreater(track.predicted_min_ttc_s, 0.0)
        self.assertLess(track.predicted_min_ttc_s, 10.0)

    def test_static_mode_for_slow_object(self) -> None:
        track = _track(1, 10.0, 2.0, vx=0.0, vy=0.0, class_group="static")
        modes = predict_modes_for_track(track)
        self.assertIn("static", {m.mode_id for m in modes})

    def test_mode_collision_likelihood_high_when_close(self) -> None:
        track = _track(1, 4.0, 0.0, vx=-3.0, vy=0.0)
        annotate_tracks_with_prediction([track])
        mass = mode_collision_likelihood(track, range_threshold_m=6.0, ttc_threshold_s=3.0)
        self.assertGreater(mass, 0.2)


class SoftArbiterTests(unittest.TestCase):
    def test_hard_filters_infeasible(self) -> None:
        arb = SoftConstraintArbiter()
        candidates = [
            ManeuverCandidate(
                maneuver="lane_change_left",
                hard_ok=False,
                hard_reason="blocked",
                components=build_soft_components(gap=0.0),
            ),
            ManeuverCandidate(
                maneuver="keep_lane",
                hard_ok=True,
                components=build_soft_components(gap=0.5),
            ),
        ]
        best = arb.select(candidates)
        self.assertEqual(best.maneuver, "keep_lane")

    def test_lowest_cost_wins(self) -> None:
        arb = SoftConstraintArbiter()
        candidates = [
            ManeuverCandidate(
                maneuver="follow",
                hard_ok=True,
                components=build_soft_components(gap=0.8, risk=0.5),
            ),
            ManeuverCandidate(
                maneuver="lane_change_left",
                hard_ok=True,
                components=build_soft_components(gap=0.1, comfort=0.2),
            ),
        ]
        best = arb.select(candidates)
        self.assertEqual(best.maneuver, "lane_change_left")

    def test_inverse_gap_and_ttc_helpers(self) -> None:
        self.assertEqual(inverse_gap_cost(1.0, comfortable_m=20.0, critical_m=5.0), 1.0)
        self.assertEqual(inverse_gap_cost(25.0, comfortable_m=20.0, critical_m=5.0), 0.0)
        self.assertEqual(ttc_cost(1.0), 1.0)
        self.assertEqual(ttc_cost(10.0), 0.0)
        self.assertEqual(ttc_cost(None), 0.0)


class CostBasedLcTests(unittest.TestCase):
    def test_prefers_clear_left_over_keep_when_front_tight(self) -> None:
        # Empty left lane, tight front on current → LC left should win stage.
        objs: list[LaneAwareObject] = []
        result = evaluate_lane_change_candidates(
            lane_context=_lane_ctx(left=True, right=True),
            lane_objects=objs,
            ego_speed_mps=8.0,
            left_ok=True,
            left_reason=None,
            right_ok=True,
            right_reason=None,
            route_prefer="left",
            current_front_gap_m=8.0,
            left_occupancy=0.05,
            right_occupancy=0.6,
            highest_risk="medium",
        )
        self.assertIn(result["stage"], {"prepare", "commit"})
        self.assertEqual(result["selected_maneuver"], "lane_change_left")
        self.assertGreater(result["cost_margin_vs_keep"], 0.0)

    def test_rejects_blocked_target_lane(self) -> None:
        objs = [
            _obj("left_lane", 5.0),  # front gap too small
            _obj("left_lane", -3.0),  # rear too close
        ]
        result = evaluate_lane_change_candidates(
            lane_context=_lane_ctx(),
            lane_objects=objs,
            ego_speed_mps=10.0,
            left_ok=False,
            left_reason="left_lane_front_gap_too_small",
            right_ok=False,
            right_reason="no_right_lane",
            current_front_gap_m=12.0,
            left_occupancy=0.8,
            right_occupancy=0.8,
        )
        self.assertEqual(result["selected_maneuver"], "keep_lane")
        self.assertEqual(result["stage"], "none")

    def test_route_prefer_biases_direction(self) -> None:
        objs = []
        left = evaluate_lane_change_candidates(
            lane_context=_lane_ctx(),
            lane_objects=objs,
            ego_speed_mps=6.0,
            left_ok=True,
            left_reason=None,
            right_ok=True,
            right_reason=None,
            route_prefer="left",
            current_front_gap_m=10.0,
            left_occupancy=0.1,
            right_occupancy=0.1,
        )
        right = evaluate_lane_change_candidates(
            lane_context=_lane_ctx(),
            lane_objects=objs,
            ego_speed_mps=6.0,
            left_ok=True,
            left_reason=None,
            right_ok=True,
            right_reason=None,
            route_prefer="right",
            current_front_gap_m=10.0,
            left_occupancy=0.1,
            right_occupancy=0.1,
        )
        # With symmetric occupancy, route prefer should select matching side when LC wins.
        if left["stage"] != "none":
            self.assertEqual(left["selected_maneuver"], "lane_change_left")
        if right["stage"] != "none":
            self.assertEqual(right["selected_maneuver"], "lane_change_right")


class StagingTests(unittest.TestCase):
    def test_route_demand_direction(self) -> None:
        self.assertEqual(route_demand_direction({"preferred_lane": "left"}), "left")
        self.assertEqual(route_demand_direction({"route_option": "right"}), "right")
        self.assertIsNone(route_demand_direction({"route_option": "straight"}))

    def test_staging_prepare_with_route_and_permission(self) -> None:
        result = determine_lane_change_stage(
            stage2_decision={"maneuver": "lane_change_left", "lane_preference": "prefer_left"},
            stage3_behavior="prepare_lane_change_left",
            world_state={
                "frame": 1,
                "sample_name": "s",
                "ego": {"speed_mps": 7.0},
                "scene": {
                    "front_free_space_m": 12.0,
                    "left_side_occupancy": 0.05,
                    "right_side_occupancy": 0.5,
                    "nearest_front_vehicle": {"distance_m": 12.0},
                },
                "risk_summary": {"highest_risk_level": "medium"},
                "lane_context": {
                    "current_lane": {
                        "exists": True,
                        "role": "current",
                        "same_direction_as_ego": True,
                        "lane_change_allowed_from_current": True,
                    },
                    "left_lane": {
                        "exists": True,
                        "role": "left",
                        "same_direction_as_ego": True,
                        "lane_change_allowed_from_current": True,
                    },
                    "right_lane": {
                        "exists": True,
                        "role": "right",
                        "same_direction_as_ego": True,
                        "lane_change_allowed_from_current": True,
                    },
                    "junction_context": {
                        "is_in_junction": False,
                        "junction_ahead": False,
                        "distance_to_junction_m": 50.0,
                        "branch_count_ahead": 0,
                        "possible_turn_like_options": [],
                        "branch_distance_m": None,
                    },
                    "forward_corridor": {},
                },
            },
            route_context={
                "preferred_lane": "left",
                "route_option": "left",
                "distance_to_next_route_decision_m": 25.0,
                "route_conflict_flags": [],
            },
            route_conditioned_scene={"route_relative_objects": []},
            lane_change_permission={"left": True, "right": True},
        )
        self.assertIn(result["stage"], {"prepare", "commit"})
        self.assertIsNotNone(result["behavior"])
        self.assertIn("left", str(result["behavior"]))
        self.assertIn("cost_based_staging", result["reasoning_tags"])


if __name__ == "__main__":
    unittest.main()
