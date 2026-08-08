from __future__ import annotations

import importlib
import unittest
from pathlib import Path

from agent_ai.paths import REPO_ROOT


CANONICAL_PACKAGES = (
    "agent_ai.shared",
    "agent_ai.perception",
    "agent_ai.world_state",
    "agent_ai.behavior.lane",
    "agent_ai.behavior.route",
    "agent_ai.behavior.execution",
    "agent_ai.behavior.coverage",
    "agent_ai.runtime",
    "agent_ai.authority",
    "agent_ai.benchmark",
    "agent_ai.cli",
)

REMOVED_ROOT_PACKAGES = (
    "stage2",
    "stage3",
    "stage3b",
    "stage3c",
    "stage3c_coverage",
    "stage4",
    "stage9",
    "carla_bevfusion_stage1",
    "common",
    # root ``benchmark`` was only a shim; real package is agent_ai.benchmark
)


class PackageLayoutTests(unittest.TestCase):
    def test_repo_root_contains_agent_ai(self) -> None:
        self.assertTrue((REPO_ROOT / "agent_ai").is_dir())
        self.assertTrue((REPO_ROOT / "agent_ai" / "benchmark" / "benchmark_v1.yaml").is_file())

    def test_canonical_packages_importable(self) -> None:
        for name in CANONICAL_PACKAGES:
            module = importlib.import_module(name)
            self.assertIsNotNone(module)

    def test_legacy_root_packages_removed(self) -> None:
        for name in REMOVED_ROOT_PACKAGES:
            path = REPO_ROOT / name
            self.assertFalse(path.exists(), msg=f"legacy package dir still present: {path}")

    def test_shared_helpers(self) -> None:
        from agent_ai.shared.numeric import clamp

        self.assertEqual(clamp(5, 0, 1), 1.0)

    def test_runtime_helpers(self) -> None:
        from agent_ai.runtime.session_utils import normalize_town_name

        self.assertEqual(normalize_town_name("Town10HD"), "Town10HD")

    def test_canonical_module_files_exist(self) -> None:
        from agent_ai.module_map import CANONICAL_MODULES

        for canonical in CANONICAL_MODULES.values():
            rel = Path(*canonical.split("."))
            path = REPO_ROOT / f"{rel}.py"
            # some historical map entries may point at renamed modules that still exist
            self.assertTrue(path.is_file(), msg=f"missing {path}")

    def test_renamed_authority_modules(self) -> None:
        from agent_ai.authority.arbiter import AuthorityArbiter
        from agent_ai.authority.evaluator import Stage9Evaluator
        from agent_ai.authority.state_machine import AuthorityStateMachine

        self.assertTrue(callable(AuthorityArbiter))
        self.assertTrue(callable(AuthorityStateMachine))
        self.assertTrue(callable(Stage9Evaluator))

    def test_cli_command_modules_use_domain_names(self) -> None:
        from agent_ai.cli.registry_data import COMMANDS

        self.assertEqual(COMMANDS["world_replay"], "agent_ai.cli.commands.world_replay")
        self.assertEqual(COMMANDS["stage2_replay"], "agent_ai.cli.commands.world_replay")
        self.assertEqual(COMMANDS["run_stage2_replay"], "agent_ai.cli.commands.world_replay")
        self.assertTrue((REPO_ROOT / "agent_ai/cli/commands/world_replay.py").is_file())
        self.assertFalse((REPO_ROOT / "agent_ai/cli/commands/stage2_replay.py").exists())


if __name__ == "__main__":
    unittest.main()
