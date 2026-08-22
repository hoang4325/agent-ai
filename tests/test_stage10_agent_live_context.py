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
            agent_preferred_lane=preferred_lane,
            agent_active_maneuver="",
            lane_change_permission=True,
            route_conflict_flags=["blocked_clear_adjacent_lane"],
            drivable_envelope=SimpleNamespace(forward_clear_m=10.0),
        )
        return adapter.build_intent_request(world, baseline_intent="keep_lane")

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


if __name__ == "__main__":
    unittest.main()
