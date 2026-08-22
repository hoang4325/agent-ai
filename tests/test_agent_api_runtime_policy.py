from __future__ import annotations

import json
import os
import socket
import unittest
from unittest import mock

from benchmark.agent_shadow_adapter import AgentShadowAdapter, AgentShadowAdapterConfig


def _call(adapter: AgentShadowAdapter):
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
            "preferred_lane": "right",
            "route_conflict_flags": ["blocked_clear_adjacent_lane"],
        },
        stop_context={},
        baseline_context={
            "requested_behavior": "stop_before_obstacle",
            "target_lane": "right",
            "lane_change_permission": {"left": True, "right": True},
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


if __name__ == "__main__":
    unittest.main()
