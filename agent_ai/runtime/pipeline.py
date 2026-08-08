"""
Runtime (Stage 4) pipeline composition notes.

Canonical packages live under ``agent_ai.*``. Legacy stage* names remain as
shims for older scripts.

Online composition order (see ``Stage4OnlineOrchestrator.run``):

  1. PerceptionOnlineAdapter   (agent_ai.perception bridge)
  2. WorldBehaviorRuntime      (world_state → behavior.lane → behavior.route)
  3. ExecutionRuntime          (behavior.execution local planner + arbitration)
  4. Stage4ShadowRuntime       (optional shadow proposals)
  5. Stage4Monitor             (tick / latency / fallback artifacts)

Authority plugs in later via ``agent_ai.shared.ports.AuthorityPort``.
"""
from __future__ import annotations

from agent_ai.shared.ports import (
    AgentPort,
    AuthorityPort,
    BehaviorPort,
    ExecutionPort,
    PerceptionPort,
    SafetyGatePort,
    WorldBuildPort,
)

ONLINE_PIPELINE_ORDER: tuple[str, ...] = (
    "perception",
    "world_behavior",
    "execution",
    "shadow",
    "monitor",
)

PIPELINE_COMPONENTS: dict[str, str] = {
    "perception": "agent_ai.runtime.perception_online_adapter.PerceptionOnlineAdapter",
    "world_behavior": "agent_ai.runtime.world_behavior_runtime.WorldBehaviorRuntime",
    "execution": "agent_ai.runtime.execution_runtime.ExecutionRuntime",
    "shadow": "agent_ai.runtime.shadow_runtime.Stage4ShadowRuntime",
    "monitor": "agent_ai.runtime.monitoring.Stage4Monitor",
    "orchestrator": "agent_ai.runtime.online_orchestrator.Stage4OnlineOrchestrator",
    "session_utils": "agent_ai.runtime.session_utils",
    "control_helpers": "agent_ai.runtime.control_helpers",
    "authority": "agent_ai.authority.authority_arbiter.AuthorityArbiter",
}

__all__ = [
    "ONLINE_PIPELINE_ORDER",
    "PIPELINE_COMPONENTS",
    "AgentPort",
    "AuthorityPort",
    "BehaviorPort",
    "ExecutionPort",
    "PerceptionPort",
    "SafetyGatePort",
    "WorldBuildPort",
]
