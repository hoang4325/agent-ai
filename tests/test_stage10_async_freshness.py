from __future__ import annotations

import argparse
import time
import unittest
from types import SimpleNamespace

from benchmark.async_agent_worker import AsyncAgentRequest, AsyncAgentResult
from scripts.run_stage10_stage1_live_bridge import (
    _apply_agent_episode_limit,
    _apply_agent_retry_cooldown,
    _active_assist_stop_reason,
    _agent_response_freshness,
    _assist_commit_promotion_frames,
    _assist_completion_metadata,
    _assist_hold_frames,
    _assist_lane_transition_completed,
    _assist_lifecycle_action,
    _can_continue_active_assist,
    _lane_change_ttc_safety,
    _lane_center_longitudinal_control,
    _summarize_assist_log,
    _target_lane_corridor_risk,
    _update_lane_transition_stability,
)


def _result(*, timestamp_s: float, intent: str = "prepare_lane_change_right"):
    request = AsyncAgentRequest(
        request_id=1,
        frame_id=10,
        frame_idx=10,
        sim_timestamp_s=timestamp_s,
        submitted_wall_s=time.monotonic(),
        payload={},
        context={"ego_lane_id": "-1", "preferred_lane": "right"},
    )
    intent_record = SimpleNamespace(tactical_intent=intent)
    return AsyncAgentResult(
        request=request,
        intent_record=intent_record,
        completed_wall_s=time.monotonic(),
        latency_ms=1000.0,
    )


class Stage10AsyncFreshnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = argparse.Namespace(
            agent_response_max_age_s=3.0,
            agent_risk_ttc_threshold=2.0,
        )
        self.world = SimpleNamespace(
            ego_lane_id="-1",
            agent_preferred_lane="right",
            lane_change_permission=True,
        )

    def test_accepts_fresh_compatible_response(self) -> None:
        accepted, reason, age = _agent_response_freshness(
            args=self.args,
            result=_result(timestamp_s=10.0),
            current_timestamp_s=11.0,
            current_world_state=self.world,
            current_min_ttc_s=5.0,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "fresh")
        self.assertEqual(age, 1.0)

    def test_rejects_stale_response(self) -> None:
        accepted, reason, _ = _agent_response_freshness(
            args=self.args,
            result=_result(timestamp_s=5.0),
            current_timestamp_s=11.0,
            current_world_state=self.world,
            current_min_ttc_s=5.0,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "stale_response_age")

    def test_rejects_lane_change_when_current_ttc_is_low(self) -> None:
        accepted, reason, _ = _agent_response_freshness(
            args=self.args,
            result=_result(timestamp_s=10.0),
            current_timestamp_s=11.0,
            current_world_state=self.world,
            current_min_ttc_s=1.5,
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "stale_response_low_ttc_target_corridor_unavailable")

    def test_accepts_low_global_ttc_when_target_corridor_is_clear(self) -> None:
        self.world.target_lane_risk_available = True
        self.world.target_lane_corridor_clear = True
        accepted, reason, _ = _agent_response_freshness(
            args=self.args,
            result=_result(timestamp_s=10.0),
            current_timestamp_s=11.0,
            current_world_state=self.world,
            current_min_ttc_s=1.5,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "fresh")

    def test_target_corridor_ignores_origin_lane_blocker(self) -> None:
        risk = _target_lane_corridor_risk(
            detections=[SimpleNamespace(x=4.0, y=0.0, dx=4.0, dy=1.8)],
            ego_v_mps=2.0,
            lateral_center_m=-3.5,
            corridor_half_width_m=1.6,
            rear_clearance_m=8.0,
            ttc_threshold_s=2.0,
        )
        self.assertTrue(risk["available"])
        self.assertTrue(risk["clear"])
        self.assertEqual(risk["object_count"], 0)

    def test_target_corridor_rejects_close_forward_object(self) -> None:
        risk = _target_lane_corridor_risk(
            detections=[SimpleNamespace(x=4.0, y=-3.5, dx=4.0, dy=1.8)],
            ego_v_mps=2.0,
            lateral_center_m=-3.5,
            corridor_half_width_m=1.6,
            rear_clearance_m=8.0,
            ttc_threshold_s=2.0,
        )
        self.assertFalse(risk["clear"])
        self.assertLess(risk["forward_ttc_s"], 2.0)

    def test_target_corridor_rejects_close_rear_object(self) -> None:
        risk = _target_lane_corridor_risk(
            detections=[SimpleNamespace(x=-5.0, y=3.5, dx=4.0, dy=1.8)],
            ego_v_mps=2.0,
            lateral_center_m=3.5,
            corridor_half_width_m=1.6,
            rear_clearance_m=8.0,
            ttc_threshold_s=2.0,
        )
        self.assertFalse(risk["clear"])
        self.assertLess(risk["rear_clearance_m"], 8.0)

    def test_active_maneuver_continues_when_only_origin_lane_ttc_is_low(self) -> None:
        world = SimpleNamespace(
            lane_change_permission=True,
            target_lane_risk_available=True,
            target_lane_corridor_clear=True,
        )
        request = SimpleNamespace(tactical_intent="commit_lane_change_right")
        self.assertTrue(
            _can_continue_active_assist(
                args=self.args,
                active_request=request,
                active_metadata={},
                baseline_intent="stop_before_obstacle",
                world_state=world,
                min_ttc_s=1.0,
            )
        )
        self.assertIsNone(
            _active_assist_stop_reason(
                args=self.args,
                active_request=request,
                baseline_intent="stop_before_obstacle",
                world_state=world,
                min_ttc_s=1.0,
                lane_change_completed=False,
            )
        )

    def test_target_corridor_evidence_is_fail_safe_when_missing(self) -> None:
        safe, reason = _lane_change_ttc_safety(
            world_state=SimpleNamespace(),
            global_min_ttc_s=1.0,
            threshold_s=2.0,
        )
        self.assertFalse(safe)
        self.assertEqual(reason, "low_ttc_target_corridor_unavailable")

    def test_emergency_ttc_cannot_be_overridden_by_clear_target_lane(self) -> None:
        safe, reason = _lane_change_ttc_safety(
            world_state=SimpleNamespace(
                target_lane_risk_available=True,
                target_lane_corridor_clear=True,
            ),
            global_min_ttc_s=0.5,
            threshold_s=2.0,
            emergency_floor_s=0.75,
        )
        self.assertFalse(safe)
        self.assertEqual(reason, "emergency_global_ttc")

    def test_async_summary_does_not_count_pending_query_as_rejection(self) -> None:
        rows = [
            {
                "frame_id": 1,
                "timestamp_s": 0.1,
                "agent_queried": True,
                "agent_response_received": False,
                "assist_applied": False,
            },
            {
                "frame_id": 2,
                "timestamp_s": 0.2,
                "agent_queried": False,
                "agent_response_received": True,
                "agent_response_fresh": False,
                "agent_response_age_s": 4.0,
                "agent_call_latency_ms": 1200.0,
                "assist_applied": False,
                "assist_reject_reason": "stale_response_age",
            },
        ]
        args = argparse.Namespace(delta_t=0.1, seed=0)
        summary = _summarize_assist_log(
            rows,
            {"frames": 2, "tick_latency_samples_ms": [20.0, 30.0]},
            args,
            worker_stats={"submitted": 1, "completed": 1},
        )
        self.assertEqual(summary["agent_query_frames"], 1)
        self.assertEqual(summary["agent_response_frames"], 1)
        self.assertEqual(summary["assist_rejected_frames"], 1)
        self.assertEqual(summary["agent_query_rejection_rate"], 1.0)
        self.assertEqual(summary["stale_response_discard_rate"], 1.0)
        self.assertEqual(summary["safety_arbitration_rejection_rate"], 1.0)

    def test_timeout_fallback_is_not_counted_as_safety_rejection(self) -> None:
        rows = [
            {
                "frame_id": 1,
                "timestamp_s": 0.1,
                "agent_queried": True,
                "agent_response_received": True,
                "agent_worker_error_type": "timeout",
                "agent_fallback_to_baseline": True,
                "agent_fallback_reason": "timeout",
                "assist_applied": False,
                "assist_reject_reason": "same_as_baseline",
            },
            {
                "frame_id": 2,
                "timestamp_s": 0.2,
                "agent_queried": True,
                "agent_response_received": True,
                "assist_applied": False,
                "assist_reject_reason": "lane_change_not_permitted",
            },
        ]
        args = argparse.Namespace(delta_t=0.1, seed=0)
        summary = _summarize_assist_log(
            rows,
            {"frames": 2, "tick_latency_samples_ms": []},
            args,
        )
        self.assertEqual(summary["agent_api_failure_rate"], 0.5)
        self.assertEqual(summary["agent_timeout_rate"], 0.5)
        self.assertEqual(summary["safety_arbitration_evaluated_frames"], 1)
        self.assertEqual(summary["safety_arbitration_rejected_frames"], 1)
        self.assertEqual(summary["safety_arbitration_rejection_rate"], 1.0)
        self.assertEqual(
            summary["safety_arbitration_rejection_reason_counts"],
            {"lane_change_not_permitted": 1},
        )

    def test_completion_tracking_survives_active_hold_cleanup(self) -> None:
        tracked = {"origin_lane_id": "-1", "target_lane_id": "-2"}
        metadata = _assist_completion_metadata({}, tracked)
        world = SimpleNamespace(
            ego_lane_id="-2",
            ego_lateral_error_m=0.2,
            heading_error_rad=0.05,
        )
        self.assertTrue(
            _assist_lane_transition_completed(
                world_state=world,
                active_metadata=metadata,
            )
        )

    def test_summary_uses_physical_lane_transition_without_post_cruise(self) -> None:
        rows = [
            {
                "frame_id": 1,
                "timestamp_s": 0.0,
                "agent_queried": True,
                "agent_response_received": True,
                "agent_intent": "prepare_lane_change_right",
                "assist_applied": True,
                "agent_call_latency_ms": 1200.0,
                "post_lane_change_cruise": False,
            },
            {
                "frame_id": 2,
                "timestamp_s": 1.2,
                "agent_queried": False,
                "agent_response_received": False,
                "assist_applied": True,
                "assist_continued": True,
                "lane_change_completed": True,
                "lane_change_completion_timestamp_s": 1.2,
                "post_lane_change_cruise": False,
            },
        ]
        args = argparse.Namespace(delta_t=0.1, seed=0)
        summary = _summarize_assist_log(
            rows,
            {"frames": 2, "tick_latency_samples_ms": [20.0, 30.0]},
            args,
        )
        maneuver = summary["lane_change_maneuver"]
        self.assertEqual(maneuver["completed_timestamp_s"], 1.2)
        self.assertEqual(maneuver["completion_time_s"], 1.2)
        self.assertEqual(maneuver["completion_source"], "lane_transition")
        self.assertIsNone(maneuver["post_lane_change_cruise_timestamp_s"])

    def test_summary_records_terminal_maneuver_failure(self) -> None:
        rows = [
            {
                "frame_id": 1,
                "timestamp_s": 0.0,
                "agent_queried": True,
                "agent_response_received": True,
                "agent_intent": "prepare_lane_change_right",
                "assist_applied": True,
                "agent_call_latency_ms": 800.0,
            },
            {
                "frame_id": 2,
                "timestamp_s": 15.0,
                "agent_queried": False,
                "agent_response_received": False,
                "assist_applied": False,
                "maneuver_failure_reason": "maneuver_timeout",
                "maneuver_failure_timestamp_s": 15.0,
            },
        ]
        summary = _summarize_assist_log(
            rows,
            {"frames": 2, "tick_latency_samples_ms": []},
            argparse.Namespace(delta_t=0.1, seed=0),
        )
        maneuver = summary["lane_change_maneuver"]
        self.assertFalse(maneuver["completed"])
        self.assertEqual(maneuver["failure_reason"], "maneuver_timeout")
        self.assertEqual(summary["maneuver_failure_reason_counts"], {"maneuver_timeout": 1})

    def test_episode_request_cap_allows_one_high_level_decision(self) -> None:
        allowed, reason = _apply_agent_episode_limit(
            requested=True,
            trigger_reason="scenario_lane_change_right",
            submitted_count=0,
            max_requests=1,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "scenario_lane_change_right")

        allowed, reason = _apply_agent_episode_limit(
            requested=True,
            trigger_reason="periodic_stride_10",
            submitted_count=1,
            max_requests=1,
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "episode_request_cap_1_reached")

    def test_lane_change_hold_outlives_commit_promotion(self) -> None:
        args = argparse.Namespace(delta_t=0.1)
        self.assertEqual(_assist_hold_frames(args), 150)
        self.assertEqual(_assist_commit_promotion_frames(args), 10)
        self.assertLess(_assist_commit_promotion_frames(args), _assist_hold_frames(args))

    def test_lane_change_completion_requires_consecutive_stable_frames(self) -> None:
        stable = 0
        completed = False
        for _ in range(4):
            stable, completed = _update_lane_transition_stability(
                candidate=True,
                previous_frames=stable,
                required_frames=5,
            )
            self.assertFalse(completed)
        stable, completed = _update_lane_transition_stability(
            candidate=True,
            previous_frames=stable,
            required_frames=5,
        )
        self.assertEqual(stable, 5)
        self.assertTrue(completed)

        stable, completed = _update_lane_transition_stability(
            candidate=False,
            previous_frames=stable,
            required_frames=5,
        )
        self.assertEqual(stable, 0)
        self.assertFalse(completed)

    def test_failed_agent_retry_obeys_cooldown(self) -> None:
        allowed, reason = _apply_agent_retry_cooldown(
            requested=True,
            trigger_reason="scenario_lane_change_right",
            now_s=11.0,
            last_failure_s=10.0,
            cooldown_s=2.0,
        )
        self.assertFalse(allowed)
        self.assertIn("retry_cooldown_wait", reason)

        allowed, reason = _apply_agent_retry_cooldown(
            requested=True,
            trigger_reason="scenario_lane_change_right",
            now_s=12.0,
            last_failure_s=10.0,
            cooldown_s=2.0,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "scenario_lane_change_right")

    def test_newly_accepted_assist_is_not_cleared_on_response_frame(self) -> None:
        action = _assist_lifecycle_action(
            accepted_new_assist=True,
            preserve_active_after_fallback=False,
            response_received=True,
            can_continue_active=True,
        )
        self.assertEqual(action, "accepted")

        action = _assist_lifecycle_action(
            accepted_new_assist=False,
            preserve_active_after_fallback=False,
            response_received=False,
            can_continue_active=True,
        )
        self.assertEqual(action, "continue")

    def test_adjacent_lane_control_never_commands_throttle_and_brake_together(self) -> None:
        throttle, brake = _lane_center_longitudinal_control(
            current_speed_mps=0.0,
            requested_speed_mps=1.5,
            lateral_distance_m=3.5,
        )
        self.assertGreater(throttle, 0.0)
        self.assertEqual(brake, 0.0)

        throttle, brake = _lane_center_longitudinal_control(
            current_speed_mps=3.0,
            requested_speed_mps=1.5,
            lateral_distance_m=3.5,
        )
        self.assertEqual(throttle, 0.0)
        self.assertGreater(brake, 0.0)


if __name__ == "__main__":
    unittest.main()
