from __future__ import annotations

import importlib
import unittest

from agent_ai.cli.dispatch import list_commands, run_command, run_module_main
from agent_ai.cli.registry_data import COMMANDS


class CliRegistryTests(unittest.TestCase):
    def test_registry_not_empty(self) -> None:
        self.assertGreaterEqual(len(COMMANDS), 80)
        self.assertIn("world_replay", COMMANDS)
        self.assertIn("stage2_replay", COMMANDS)  # legacy alias
        self.assertIn("run_stage2_replay", COMMANDS)
        self.assertIn("system_benchmark", COMMANDS)

    def test_list_commands_prefers_short_names(self) -> None:
        names = list_commands()
        self.assertIn("world_replay", names)
        self.assertNotIn("run_world_replay", names)

    def test_command_modules_importable(self) -> None:
        # Spot-check a diverse subset (not every module — some need CARLA/deps).
        targets = {
            # Avoid modules that import CARLA at module load time.
            "agent_ai.cli.commands.world_replay",
            "agent_ai.cli.commands.takeover_canary",
            "agent_ai.cli.commands.system_benchmark",
            "agent_ai.cli.commands.baseline_repin",
        }
        for name in targets:
            module = importlib.import_module(name)
            self.assertTrue(callable(getattr(module, "main", None)) or hasattr(module, "__file__"))

    def test_unknown_command_exits(self) -> None:
        with self.assertRaises(SystemExit):
            run_command("this_command_does_not_exist")

    def test_world_replay_help_returns_zero(self) -> None:
        code = run_module_main(
            "agent_ai.cli.commands.world_replay",
            argv=["--help"],
        )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
