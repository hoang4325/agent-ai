from __future__ import annotations

import importlib
import unittest
from pathlib import Path

from agent_ai.paths import REPO_ROOT


PACKAGE_MAP = {
    "common": "agent_ai.shared",
    "carla_bevfusion_stage1": "agent_ai.perception",
    "stage2": "agent_ai.world_state",
    "stage3": "agent_ai.behavior.lane",
    "stage3b": "agent_ai.behavior.route",
    "stage3c": "agent_ai.behavior.execution",
    "stage3c_coverage": "agent_ai.behavior.coverage",
    "stage4": "agent_ai.runtime",
    "stage9": "agent_ai.authority",
    "benchmark": "agent_ai.benchmark",
}


class PackageLayoutTests(unittest.TestCase):
    def test_repo_root_contains_agent_ai(self) -> None:
        self.assertTrue((REPO_ROOT / "agent_ai").is_dir())
        self.assertTrue((REPO_ROOT / "agent_ai" / "benchmark" / "benchmark_v1.yaml").is_file())

    def test_canonical_packages_importable(self) -> None:
        for name in PACKAGE_MAP.values():
            module = importlib.import_module(name)
            self.assertIsNotNone(module)

    def test_legacy_shims_importable(self) -> None:
        for legacy in PACKAGE_MAP:
            module = importlib.import_module(legacy)
            self.assertIsNotNone(module)

    def test_shared_helpers_via_legacy_common(self) -> None:
        from common.numeric import clamp
        from agent_ai.shared.numeric import clamp as clamp2

        self.assertEqual(clamp(5, 0, 1), clamp2(5, 0, 1))

    def test_runtime_helpers_via_legacy_stage4(self) -> None:
        from stage4.session_utils import normalize_town_name
        from agent_ai.runtime.session_utils import normalize_town_name as normalize2

        self.assertEqual(normalize_town_name("Town10HD"), normalize2("Town10HD"))


if __name__ == "__main__":
    unittest.main()
