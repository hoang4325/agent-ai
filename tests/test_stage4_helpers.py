from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_ai.runtime.control_helpers import build_shadow_candidate_control
from agent_ai.runtime.session_utils import normalize_town_name, resolve_attach_actor_id


class _FakeCarla:
    def VehicleControl(self, **kwargs):
        return kwargs


class Stage4HelperTests(unittest.TestCase):
    def test_normalize_town_name(self) -> None:
        self.assertEqual(normalize_town_name(r"C:\CARLA\Town10HD"), "Town10HD")
        self.assertEqual(normalize_town_name("Town03"), "Town03")

    def test_resolve_attach_actor_id_from_cli(self) -> None:
        args = SimpleNamespace(attach_to_actor_id=42, scenario_manifest=None)
        actor_id, manifest = resolve_attach_actor_id(args)
        self.assertEqual(actor_id, 42)
        self.assertIsNone(manifest)

    def test_resolve_attach_actor_id_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "scenario.json"
            manifest_path.write_text('{"ego_actor_id": 7, "town": "Town10HD"}', encoding="utf-8")
            args = SimpleNamespace(attach_to_actor_id=None, scenario_manifest=str(manifest_path))
            actor_id, manifest = resolve_attach_actor_id(args)
            self.assertEqual(actor_id, 7)
            assert manifest is not None
            self.assertEqual(manifest["town"], "Town10HD")

    def test_build_shadow_candidate_control_stop(self) -> None:
        control = build_shadow_candidate_control(
            carla_module=_FakeCarla(),
            proposal={
                "shadow_requested_behavior": "stop_before_obstacle",
                "shadow_target_speed_mps": 0.0,
                "proposed_trajectory": {"sampled_path": []},
            },
            current_speed_mps=1.0,
        )
        self.assertGreater(control["brake"], 0)
        self.assertEqual(control["throttle"], 0.0)

    def test_build_shadow_candidate_control_lane_change_left(self) -> None:
        control = build_shadow_candidate_control(
            carla_module=_FakeCarla(),
            proposal={
                "shadow_requested_behavior": "lane_change_left",
                "shadow_target_speed_mps": 2.0,
                "proposed_trajectory": {
                    "sampled_path": [
                        {"lateral_offset_m": 0.0},
                        {"lateral_offset_m": 0.0},
                        {"lateral_offset_m": 0.0},
                    ]
                },
            },
            current_speed_mps=0.5,
        )
        self.assertGreater(control["steer"], 0)


if __name__ == "__main__":
    unittest.main()
