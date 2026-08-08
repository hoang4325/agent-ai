"""
Authority package — state machine, safety supervisor, arbiter, TOR/MRM.

Extension tip: new agents/planners should implement
``agent_ai.shared.ports.AgentPort`` / ``AuthorityPort`` and plug into
``AuthorityArbiter`` rather than forking packages.
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
from .soft_contract import (
    SoftContractBundle,
    SoftVeto,
    build_soft_contract_from_behavior,
    derive_soft_bounds,
    evaluate_soft_vetoes,
    soft_cost_profile,
)
from .safety_supervisor import SafetySupervisor
from .contract_resolver import ContractResolver
from .handoff_planner import HandoffPlanner
from .logger import AuthorityLogger
from .arbiter import AuthorityArbiter
try:
    from .osqp_mpc_adapter import OSQPMpcAdapter, trajectory_request_to_bounds
except Exception:  # pragma: no cover - optional heavy deps (osqp)
    OSQPMpcAdapter = None  # type: ignore
    trajectory_request_to_bounds = None  # type: ignore

# ── Phase 2 modules ───────────────────────────────────────────────────────────
from .tor_manager import TORManager
from .minimal_risk_maneuver import MRMExecutor
from .human_override_monitor import HumanOverrideMonitor
from .evaluator import AuthorityEvaluator, EvaluationReport
Stage9Evaluator = AuthorityEvaluator

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
    "SoftContractBundle", "SoftVeto",
    "build_soft_contract_from_behavior", "derive_soft_bounds",
    "evaluate_soft_vetoes", "soft_cost_profile",
    "SafetySupervisor", "ContractResolver", "HandoffPlanner",
    "AuthorityLogger", "AuthorityArbiter",
    "OSQPMpcAdapter", "trajectory_request_to_bounds",
    # Phase 2
    "TORManager", "MRMExecutor", "HumanOverrideMonitor",
    "AuthorityEvaluator", "Stage9Evaluator", "EvaluationReport",
]

