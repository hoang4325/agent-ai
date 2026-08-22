from __future__ import annotations

import json
import os
import socket
import unittest
from unittest import mock

from benchmark.agent_shadow_adapter import (
    AgentShadowAdapter,
    AgentShadowAdapterConfig,
    _build_bevfusion_prompt_context,
)


def _call(
    adapter: AgentShadowAdapter,
    *,
    baseline_intent: str = "stop_before_obstacle",
    preferred_lane: str = "right",
    blocked_clear_facts: bool = False,
):
    lane_change_permission = {
        "left": preferred_lane == "left",
        "right": preferred_lane == "right",
    }
    return adapter.call(
        case_id="unit",
        frame_id=1,
        ego_state={
            "risk_summary": {"highest_risk_level": "medium"},
            "scene": {"front_free_space_m": 10.0},
        },
        tracked_objects=[],
        lane_context={},
        route_context={
            "route_option": "straight",
            "preferred_lane": preferred_lane,
            "route_conflict_flags": ["blocked_clear_adjacent_lane"],
        },
        stop_context={},
        baseline_context={
            "requested_behavior": baseline_intent,
            "target_lane": preferred_lane,
            "lane_change_permission": lane_change_permission,
            "current_lane_blocked": blocked_clear_facts,
            "adjacent_preferred_lane_clear": blocked_clear_facts,
            "preferred_lane_permission": blocked_clear_facts,
        },
    )


class _Response:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


class AgentAPIRuntimePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ, {"TEST_AGENT_KEY": "not-a-real-key"})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()

    def _adapter(self, *, retries: int = 0) -> AgentShadowAdapter:
        return AgentShadowAdapter(
            AgentShadowAdapterConfig(
                mode="api",
                model_id="fixed-model",
                api_endpoint="http://127.0.0.1:9999/v1/chat/completions",
                api_key_env_var="TEST_AGENT_KEY",
                api_timeout_s=0.25,
                api_max_retries=retries,
            )
        )

    @mock.patch("urllib.request.urlopen", side_effect=socket.timeout("slow provider"))
    def test_zero_retries_makes_one_network_attempt(self, urlopen) -> None:
        result = _call(self._adapter(retries=0))
        self.assertEqual(urlopen.call_count, 1)
        self.assertTrue(result.fallback_to_baseline)
        self.assertEqual(result.provenance["fallback_reason"], "timeout")

    @mock.patch("urllib.request.urlopen")
    def test_records_backend_model_and_usage(self, urlopen) -> None:
        urlopen.return_value = _Response(
            {
                "model": "provider/backend-fixed",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "tactical_intent": "prepare_lane_change_right",
                                    "target_lane": "right",
                                    "confidence": 0.8,
                                    "reason_tags": ["clear_adjacent_lane"],
                                }
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            }
        )
        result = _call(self._adapter())
        self.assertFalse(result.fallback_to_baseline)
        self.assertEqual(result.provenance["backend_model_id"], "provider/backend-fixed")
        self.assertEqual(result.provenance["total_token_count"], 30)
        self.assertEqual(result.provenance["api_attempt_count"], 1)
        self.assertEqual(result.provenance["api_payload_variant"], "rich_json")

    @mock.patch("urllib.request.urlopen")
    def test_blocked_clear_prompt_requests_progress_even_when_baseline_keeps_lane(self, urlopen) -> None:
        captured_payloads: list[dict] = []

        def respond(request, timeout):
            del timeout
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return _Response(
                {
                    "model": "provider/backend-fixed",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "tactical_intent": "prepare_lane_change_right",
                                        "target_lane": "right",
                                        "confidence": 0.85,
                                        "reason_tags": ["blocked_clear_recovery"],
                                    }
                                )
                            }
                        }
                    ],
                }
            )

        urlopen.side_effect = respond
        result = _call(
            self._adapter(),
            baseline_intent="keep_lane",
            blocked_clear_facts=True,
        )

        self.assertFalse(result.fallback_to_baseline)
        self.assertEqual(result.tactical_intent, "prepare_lane_change_right")
        prompt = captured_payloads[0]["messages"][1]["content"]
        self.assertIn("blocked_clear_state=eligible", prompt)
        self.assertIn("prepare_lane_change_right", prompt)
        self.assertIn("BEVFUSION_CONTEXT_JSON=", prompt)
        self.assertIn("bevfusion_tactical_context_v1", prompt)
        self.assertIn("even when baseline=keep_lane", prompt)
        self.assertEqual(captured_payloads[0].get("response_format"), {"type": "json_object"})
        self.assertIn("Never return an ellipsis", captured_payloads[0]["messages"][0]["content"])

    @mock.patch("urllib.request.urlopen")
    def test_wrong_lane_change_direction_falls_back(self, urlopen) -> None:
        urlopen.return_value = _Response(
            {
                "model": "provider/backend-fixed",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "tactical_intent": "prepare_lane_change_left",
                                    "target_lane": "left",
                                    "confidence": 0.9,
                                    "reason_tags": ["wrong_side"],
                                }
                            )
                        }
                    }
                ],
            }
        )
        result = _call(
            self._adapter(),
            baseline_intent="keep_lane",
            blocked_clear_facts=True,
        )
        self.assertTrue(result.fallback_to_baseline)
        self.assertEqual(result.validation_status, "invalid_intent")

    def test_bevfusion_prompt_context_is_tactical_and_bounded(self) -> None:
        tracked_objects = [
            {
                "track_id": index,
                "class_name": "car",
                "position_ego": [float(20 - index), float(index % 2), 0.0],
                "distance_m": float(20 - index),
                "ttc_seconds": float(10 - index) if index < 9 else 0.5,
                "source_confidence": 0.9,
                "box_half_extents_m": [2.1, 0.9, 0.8],
                "yaw_rad": 0.05,
                "source_track": {"raw_tensor": [1] * 1000},
            }
            for index in range(10)
        ]
        context = _build_bevfusion_prompt_context(
            ego_state={
                "speed_mps": 5.0,
                "risk_summary": {
                    "highest_risk_level": "medium",
                    "minimum_ttc_seconds": 2.0,
                },
                "scene": {"front_free_space_m": 9.0},
                "perception": {
                    "sensor_health": "OK",
                    "sync_ok": True,
                    "lidar_point_count": 28000,
                    "radar_point_count": 612,
                },
            },
            tracked_objects=tracked_objects,
            lane_context={
                "current_lane_id": "-1",
                "corridor_clear": True,
                "drivable_envelope": {
                    "left_bound_m": 3.5,
                    "right_bound_m": 3.5,
                    "forward_clear_m": 9.0,
                },
            },
            route_context={"route_option": "straight", "preferred_lane": "right"},
            stop_context={"binding_status": "derived_active", "distance_to_stop_m": 8.0},
            baseline_context={
                "requested_behavior": "stop_before_obstacle",
                "target_lane": "right",
                "lane_change_permission": {"left": False, "right": True},
            },
        )

        self.assertEqual(context["schema"], "bevfusion_tactical_context_v1")
        self.assertEqual(len(context["objects"]), 6)
        self.assertEqual(context["objects"][0]["id"], "9")
        self.assertEqual(context["objects"][0]["box_half_extents_m"], [2.1, 0.9, 0.8])
        self.assertTrue(context["lanes"]["corridor_clear"])
        self.assertNotIn("source_track", json.dumps(context))
        self.assertLess(len(json.dumps(context)), 6000)


if __name__ == "__main__":
    unittest.main()
