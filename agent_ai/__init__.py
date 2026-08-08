"""
Agent-AI — domain-oriented package layout (no stage* package names).

Public packages
---------------
- agent_ai.perception
- agent_ai.world_state
- agent_ai.behavior.lane | .route | .execution | .coverage
- agent_ai.runtime
- agent_ai.authority
- agent_ai.benchmark
- agent_ai.shared
- agent_ai.cli

Prefer ``agent_ai.*`` imports and ``python -m agent_ai.cli <command>``.
Legacy root packages (``stage2``, ``stage9``, ``common``, …) have been removed.
See ``agent_ai.module_map`` for rename history.
"""

from __future__ import annotations

from .paths import REPO_ROOT

__all__ = ["REPO_ROOT"]
