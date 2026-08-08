"""
Canonical module naming map for Agent-AI (domain names, no stage prefixes).

Naming rules for new code
-------------------------
1. Package encodes domain: ``authority``, ``runtime``, ``behavior.lane``,
   ``benchmark``, ``cli.commands``, …
2. File names are short snake_case role nouns — no ``stageN_`` prefixes.
3. CLI commands use domain verbs: ``world_replay``, ``online_runtime``,
   ``shadow_gate``, ``authority_campaign``.
4. Legacy ``stage*`` / ``run_stage*`` names remain only as aliases in
   ``cli.registry_data`` and temporary root package shims.
5. Pipeline step IDs inside benchmark YAML (``stage1``…``stage4``) are data
   contracts for corpus/orchestration and are intentionally unchanged.

This table documents renames already applied for maintainers.
"""
from __future__ import annotations

# Selected historical module renames (not exhaustive for every benchmark gate).
CANONICAL_MODULES: dict[str, str] = {
    # core packages (file renames)
    "agent_ai.authority.authority_arbiter": "agent_ai.authority.arbiter",
    "agent_ai.authority.authority_logger": "agent_ai.authority.logger",
    "agent_ai.authority.authority_state_machine": "agent_ai.authority.state_machine",
    "agent_ai.authority.stage9_evaluator": "agent_ai.authority.evaluator",
    "agent_ai.world_state.object_schema": "agent_ai.world_state.schema",
    "agent_ai.world_state.world_state_builder": "agent_ai.world_state.builder",
    "agent_ai.runtime.online_orchestrator": "agent_ai.runtime.orchestrator",
    "agent_ai.runtime.world_behavior_runtime": "agent_ai.runtime.behavior_runtime",
    "agent_ai.runtime.perception_online_adapter": "agent_ai.runtime.perception_adapter",
    "agent_ai.runtime.execution_runtime": "agent_ai.runtime.execution",
    "agent_ai.perception.adapter": "agent_ai.perception.sample_adapter",
    "agent_ai.perception.bevfusion_live_adapter": "agent_ai.perception.live_adapter",
    "agent_ai.perception.bevfusion_runtime": "agent_ai.perception.model_runtime",
    "agent_ai.perception.carla_sensor_sync": "agent_ai.perception.sensor_sync",
    "agent_ai.perception.stage9_adapters": "agent_ai.perception.authority_adapters",
    # CLI examples
    "agent_ai.cli.commands.stage2_replay": "agent_ai.cli.commands.world_replay",
    "agent_ai.cli.commands.stage4_online": "agent_ai.cli.commands.online_runtime",
    "agent_ai.cli.commands.stage9_campaign": "agent_ai.cli.commands.authority_campaign",
    "agent_ai.cli.commands.stage6_shadow_gate": "agent_ai.cli.commands.shadow_gate",
    # benchmark examples
    "agent_ai.benchmark.stage6_shadow_gate": "agent_ai.benchmark.shadow_contract_audit",
    "agent_ai.benchmark.stage6_takeover_canary": "agent_ai.benchmark.takeover_canary",
    "agent_ai.benchmark.stage5a_contract_audit": "agent_ai.benchmark.contract_audit",
    "agent_ai.benchmark.stage7_agent_shadow_audit": "agent_ai.benchmark.agent_shadow_audit",
    "agent_ai.benchmark.stage8_assist_adapter": "agent_ai.benchmark.assist_adapter",
}

# Public packages (no stage* names)
PUBLIC_PACKAGES: tuple[str, ...] = (
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

# Root packages that used to exist as shims and have been removed.
REMOVED_ROOT_PACKAGES: tuple[str, ...] = (
    "stage2",
    "stage3",
    "stage3b",
    "stage3c",
    "stage3c_coverage",
    "stage4",
    "stage9",
    "carla_bevfusion_stage1",
    "common",
    # root ``benchmark`` shim removed; use agent_ai.benchmark
)

__all__ = ["CANONICAL_MODULES", "PUBLIC_PACKAGES", "REMOVED_ROOT_PACKAGES"]
