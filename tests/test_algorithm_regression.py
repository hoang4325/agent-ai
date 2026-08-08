"""B1 algorithm regression suite + B3 semantic cases (CI-friendly, no CARLA)."""
from __future__ import annotations

import unittest

from agent_ai.benchmark.semantic_case_runner import (
    list_semantic_cases,
    load_case,
    run_all_semantic_cases,
    run_semantic_case,
    summarize_results,
)
from agent_ai.world_state.frenet import (
    default_straight_corridor,
    frenet_to_xy,
    project_to_frenet,
    target_lane_polyline,
)
from agent_ai.world_state.motion_predictor import annotate_tracks_with_prediction, predict_modes_for_track
from agent_ai.world_state.schema import TrackedObject


def _track(x: float, y: float, vx: float = 0.0, vy: float = 0.0, class_group: str = "vehicle") -> TrackedObject:
    import math

    dist = math.hypot(x, y)
    return TrackedObject(
        track_id=1,
        class_id=0,
        class_name="car",
        class_group=class_group,
        latest_detection_id="d1",
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


class FrenetMapAwareTests(unittest.TestCase):
    def test_project_and_sample_roundtrip(self) -> None:
        poly = default_straight_corridor(horizon_m=40.0, step_m=5.0)
        pose = project_to_frenet([12.0, 0.4], poly)
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose.s_m, 12.0, places=1)
        self.assertGreater(abs(pose.d_m), 0.0)
        xy = frenet_to_xy(poly, pose.s_m, 0.0)
        self.assertIsNotNone(xy)
        assert xy is not None
        self.assertAlmostEqual(xy[0], 12.0, places=1)
        self.assertAlmostEqual(xy[1], 0.0, places=1)

    def test_target_lane_offset(self) -> None:
        left = target_lane_polyline(direction="left", lane_width_m=3.5)
        pose = project_to_frenet([10.0, 3.5], left)
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertLess(abs(pose.d_m), 0.3)

    def test_lane_follow_mode_present_with_polyline(self) -> None:
        track = _track(15.0, 0.2, vx=-2.0)
        poly = default_straight_corridor()
        modes = predict_modes_for_track(track, reference_polyline=poly)
        ids = {m.mode_id for m in modes}
        self.assertIn("lane_follow", ids)
        self.assertIn("cv", ids)

    def test_annotate_map_aware_flags(self) -> None:
        tracks = [_track(12.0, 0.0, vx=-3.0)]
        annotate_tracks_with_prediction(tracks, reference_polyline=default_straight_corridor())
        mode_tags = [tag for m in tracks[0].predicted_modes for tag in (m.get("tags") or [])]
        self.assertTrue(any(t == "map_aware" for t in mode_tags))


class SemanticCaseRegressionTests(unittest.TestCase):
    def test_five_cases_present(self) -> None:
        paths = list_semantic_cases()
        ids = {p.stem for p in paths}
        required = {
            "cut_in_front_short_ttc",
            "dense_left_gap_ok",
            "rear_fast_block_lc",
            "vru_cross_yield",
            "junction_no_lc",
        }
        self.assertTrue(required.issubset(ids), msg=f"missing {required - ids}")

    def test_all_semantic_cases_pass(self) -> None:
        results = run_all_semantic_cases()
        summary = summarize_results(results)
        if summary["failed"]:
            details = [
                f"{r.case_id}: {r.failures} obs={r.observations}"
                for r in results
                if not r.passed
            ]
            self.fail("Semantic cases failed:\n" + "\n".join(details))
        self.assertEqual(summary["total"], 5)
        self.assertEqual(summary["passed"], 5)

    def test_each_case_loads(self) -> None:
        for path in list_semantic_cases():
            case = load_case(path)
            self.assertIn("case_id", case)
            self.assertIn("expect", case)
            result = run_semantic_case(case)
            self.assertEqual(result.case_id, case["case_id"])


if __name__ == "__main__":
    unittest.main()
