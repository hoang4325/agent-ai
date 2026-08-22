from __future__ import annotations

import argparse
import time
import unittest
from types import SimpleNamespace

from benchmark.async_agent_worker import AsyncAgentRequest, AsyncAgentResult
from scripts.run_stage10_stage1_live_bridge import (
    _agent_response_freshness,
    _summarize_assist_log,
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
        self.assertEqual(reason, "stale_response_low_ttc")

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


if __name__ == "__main__":
    unittest.main()
