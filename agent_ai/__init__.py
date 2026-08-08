"""
Agent-AI — standardized package layout for the autonomous-driving research stack.

Package map (legacy → canonical)
--------------------------------
carla_bevfusion_stage1 → agent_ai.perception
stage2                 → agent_ai.world_state
stage3                 → agent_ai.behavior.lane
stage3b                → agent_ai.behavior.route
stage3c                → agent_ai.behavior.execution
stage3c_coverage       → agent_ai.behavior.coverage
stage4                 → agent_ai.runtime
stage9                 → agent_ai.authority
benchmark              → agent_ai.benchmark
common                 → agent_ai.shared

Legacy top-level names remain as thin shims so existing scripts keep working.
Prefer canonical ``agent_ai.*`` imports in new code.
"""
from __future__ import annotations

from .paths import REPO_ROOT

__all__ = ["REPO_ROOT"]
