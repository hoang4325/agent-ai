from __future__ import annotations

import unittest
from types import SimpleNamespace

from carla_bevfusion_stage1.stage9_adapters import RealAgentAdapter


class Stage10AgentLiveContextTests(unittest.TestCase):
    def _request(self, preferred_lane: str) -> dict:
        adapter = RealAgentAdapter.__new__(RealAgentAdapter)
        world = SimpleNamespace(
            frame_id=11,
            ego_v_mps=0.0,
            ego_a_mps2=0.0,
            ego_lateral_error_m=0.0,
            ego_lane_id="-2",
            min_ttc_s=10.0,
            corridor_clear=True,
            new_obstacle_score=0.25,
            sensor_health="OK",
            sync_ok=True,
            world_age_ms=12,
            agent_preferred_lane=preferred_lane,
            agent_active_maneuver="",
            lane_change_permission=True,
            route_conflict_flags=["blocked_clear_adjacent_lane"],
            agent_origin_lane_id="-2",
            agent_target_lane_id="-1",
            target_lane_risk_available=True,
            target_lane_corridor_clear=True,
            target_lane_forward_ttc_s=12.5,
            target_lane_rear_clearance_m=15.0,
            target_lane_corridor_object_count=1,
            target_lane_lateral_offset_m=3.4,
            target_lane_risk_source="bevfusion_target_corridor_v1",
            drivable_envelope=SimpleNamespace(forward_clear_m=10.0),
        )
        detections = [
            SimpleNamespace(
                x=9.0,
                y=0.5,
                z=0.0,
                dx=2.1,
                dy=0.9,
                dz=0.8,
                yaw_rad=0.05,
                score=0.91,
                label_idx=0,
                label_name="car",
            )
        ]
        return adapter.build_intent_request(
            world,
            baseline_intent="keep_lane",
            detections=detections,
            sensor_input={
                "inference_time_ms": 84.5,
                "num_detections": 1,
                "num_raw_boxes": 4,
                "lidar_point_count": 28000,
                "radar_point_count": 612,
            },
        )

    def test_right_blocked_clear_context_is_explicit_and_side_specific(self) -> None:
        request = self._request("right")
        baseline = request["baseline_context"]
        self.assertEqual(baseline["requested_behavior"], "keep_lane")
        self.assertTrue(baseline["current_lane_blocked"])
        self.assertTrue(baseline["adjacent_preferred_lane_clear"])
        self.assertEqual(
            baseline["lane_change_permission"],
            {"left": False, "right": True},
        )

    def test_left_blocked_clear_context_is_explicit_and_side_specific(self) -> None:
        request = self._request("left")
        baseline = request["baseline_context"]
        self.assertTrue(baseline["preferred_lane_permission"])
        self.assertEqual(
            baseline["lane_change_permission"],
            {"left": True, "right": False},
        )

    def test_live_request_carries_bounded_bevfusion_context(self) -> None:
        request = self._request("right")

        self.assertEqual(len(request["tracked_objects"]), 1)
        self.assertEqual(request["tracked_objects"][0]["label_name"], "car")
        self.assertEqual(request["tracked_objects"][0]["distance_m"], 9.014)
        self.assertEqual(request["tracked_objects"][0]["dx"], 2.1)
        self.assertEqual(request["ego_state"]["risk_summary"]["minimum_ttc_seconds"], 10.0)
        self.assertEqual(request["ego_state"]["perception"]["lidar_point_count"], 28000)
        self.assertEqual(request["ego_state"]["perception"]["radar_point_count"], 612)
        self.assertEqual(request["stop_context"]["source"], "stage10_bevfusion_world_state")
        self.assertTrue(request["lane_context"]["target_lane_risk_available"])
        self.assertTrue(request["lane_context"]["target_lane_corridor_clear"])
        self.assertEqual(request["lane_context"]["target_lane_forward_ttc_s"], 12.5)
        self.assertEqual(
            request["lane_context"]["target_lane_risk_source"],
            "bevfusion_target_corridor_v1",
        )


if __name__ == "__main__":
    unittest.main()
