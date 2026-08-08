"""
Agent-AI CLI package.

Canonical command implementations live in ``agent_ai.cli.commands``.

Usage:
  python -m agent_ai.cli list
  python -m agent_ai.cli world_replay --help
  python -m agent_ai.cli online_runtime --output-dir ...

There is no top-level ``scripts/`` directory; use this module entrypoint only.
"""
from __future__ import annotations

from agent_ai.cli.bootstrap import PROJECT_ROOT, REPO_ROOT, ensure_repo_on_path
from agent_ai.cli.dispatch import list_commands, run_command, run_module_main

__all__ = [
    "PROJECT_ROOT",
    "REPO_ROOT",
    "ensure_repo_on_path",
    "list_commands",
    "run_command",
    "run_module_main",
]
