"""
Stage 9 — Package init (Phase 1 + Phase 2)

Extension tip: new agents/planners should implement
``agent_ai.shared.ports.AgentPort`` / ``AuthorityPort`` and plug into
``AuthorityArbiter`` rather than forking stage packages.
"""
from agent_ai.shared.ports import AgentPort, AuthorityPort, SafetyGatePort

# ── Schemas ───────────────────────────────────────────────────────────────────
from .schemas import (
    AuthorityState,
    ActiveAuthority,
    ODDStatus,
    SensorHealth,
    VetoSeverity,
    MRMStrategy,
    WorldState,
    ManeuverContract,
    TakeoverRequest,
    AuthorityContext,
    TrajectoryRequest,
    TORSignal,
    VetoSignal,
    SafetyVerdict,
    MRCPlan,
    ActuatorCommand,
    DrivableEnvelope,
    Pose2D,
)

# ── Phase 1 modules ───────────────────────────────────────────────────────────
from .state_machine import AuthorityStateMachine
from .baseline_detector import BaselineDetector
from .maneuver_contract import build_contract, validate_contract
from .safety_supervisor import SafetySupervisor
from .contract_resolver import ContractResolver
from .handoff_planner import HandoffPlanner
from .logger import AuthorityLogger
from .arbiter import AuthorityArbiter

# ── Phase 2 modules ───────────────────────────────────────────────────────────
from .tor_manager import TORManager
from .minimal_risk_maneuver import MRMExecutor
from .human_override_monitor import HumanOverrideMonitor
from .evaluator import Stage9Evaluator, EvaluationReport

# ── Phase 3 modules ───────────────────────────────────────────────────────────
from .scenario_runner import ScenarioResult, ALL_SCENARIOS

__all__ = [
    # extension ports
    "AgentPort", "AuthorityPort", "SafetyGatePort",
    # schemas
    "AuthorityState", "ActiveAuthority", "ODDStatus", "SensorHealth",
    "VetoSeverity", "MRMStrategy",
    "WorldState", "ManeuverContract", "TakeoverRequest", "AuthorityContext",
    "TrajectoryRequest", "TORSignal", "VetoSignal", "SafetyVerdict",
    "MRCPlan", "ActuatorCommand", "DrivableEnvelope", "Pose2D",
    # Phase 1
    "AuthorityStateMachine", "BaselineDetector",
    "build_contract", "validate_contract",
    "SafetySupervisor", "ContractResolver", "HandoffPlanner",
    "AuthorityLogger", "AuthorityArbiter",
    # Phase 2
    "TORManager", "MRMExecutor", "HumanOverrideMonitor",
    "Stage9Evaluator", "EvaluationReport",
]

