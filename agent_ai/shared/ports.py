"""
Extension ports (Protocol interfaces) for the Agent-AI pipeline.

These ports document stable seams where implementations can be swapped
without rearranging stage packages:

  PerceptionPort  → stage1 / PerceptionOnlineAdapter
  WorldBuildPort  → stage2 WorldStateBuilder + tracker/risk
  BehaviorPort    → stage3 / 3b behavior builders
  ExecutionPort   → stage3c LocalPlannerBridge / ExecutionRuntime
  AgentPort       → stage9 Stage9AgentAdapter (L1 contracts only)
  AuthorityPort   → stage9 AuthorityArbiter

New components should implement the relevant Protocol and be injected at
composition roots (e.g. OnlineOrchestrator, AuthorityArbiter) rather
than hard-wiring deeper into stage internals.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class PerceptionPort(Protocol):
    """Produces a perception frame / prediction from a capture sample."""

    def process_sample(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class WorldBuildPort(Protocol):
    """Builds world state / behavior frame from a perception prediction."""

    def process_prediction(self, perception_frame: Any) -> Any: ...


@runtime_checkable
class BehaviorPort(Protocol):
    """Selects or builds a behavior request given world / route context."""

    def select_or_build(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ExecutionPort(Protocol):
    """Turns an execution request into vehicle control / planner steps."""

    def prepare(self, *args: Any, **kwargs: Any) -> Any: ...

    def step(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class AgentPort(Protocol):
    """
    Agentic decision seam (L1 only).

    Implementations may propose a bounded ManeuverContract but must never
    emit raw actuator commands.
    """

    def propose_contract(self, world: Any) -> Optional[Any]: ...

    def get_intent(self, world: Any, contract: Any) -> str: ...


@runtime_checkable
class AuthorityPort(Protocol):
    """Authority arbitration tick — returns an actuator-level command."""

    def step(self, world: Any, sim_time_s: float = 0.0) -> Any: ...


@runtime_checkable
class SafetyGatePort(Protocol):
    """Safety verification / continuous watch for agent contracts."""

    def verify_contract(self, contract: Any, world: Any) -> Any: ...

    def watch_frame(self, world: Any, contract: Any, sim_time_s: float = 0.0) -> Any: ...
