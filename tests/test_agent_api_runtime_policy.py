from __future__ import annotations

import json
import os
import socket
import unittest
from unittest import mock

from benchmark.agent_shadow_adapter import AgentShadowAdapter, AgentShadowAdapterConfig


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


if __name__ == "__main__":
    unittest.main()
