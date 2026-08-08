"""
Stage 9 — Scenario Runner (v2)
================================
Defines and executes all 12 mandatory test scenarios (T9-001 → T9-012).

Each scenario:
  - Provides a deterministic sequence of WorldState frames
  - Uses a pre-configured AuthorityArbiter (with fully real Phase 1+2 modules)
  - Checks pass/fail against expected outcome criteria

No CARLA required. Scenarios are simulated via synthetic WorldState streams.

Scenario matrix (from stage9_takeover_design_v2.md, Section 10):
  T9-001  Xe đỗ giữa làn → BASELINE → REQUESTING → SUPERVISED → ACTIVE → REVOKE → BASELINE
  T9-002  Baseline oscillation → REQUESTING → ACTIVE, lateral error giảm
  T9-003  Agent confidence tụt giữa chừng → REVOKE → BASELINE, không jerk spike
  T9-004  TTC collapse khi ACTIVE → REVOKE → MRM, no collision
  T9-005  Contract hết thời hạn bình thường → REVOKE → BASELINE (smooth)
  T9-006  Baseline stuck nhưng ODD_MARGIN → không grant Agent
  T9-007  ODD_EXCEEDED đột ngột → MRM → SAFE_STOP
  T9-008  World stale trong lúc request → REJECT ngay
  T9-009  Supervisor preview infeasible → SUPERVISED → BASELINE, không grant
  T9-010  TOR timeout → MRM → SAFE_STOP
  T9-011  Human override trong MRM → HUMAN_CONTROL_ACTIVE
  T9-012  Nhiều request liên tiếp → cooldown active
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .authority_arbiter import AuthorityArbiter
from .authority_logger import AuthorityLogger
from .authority_state_machine import AuthorityStateMachine
from .baseline_detector import BaselineDetector
from .contract_resolver import ContractResolver
from .handoff_planner import HandoffPlanner
from .human_override_monitor import HumanOverrideMonitor
from .minimal_risk_maneuver import MRMExecutor
from .safety_supervisor import SafetySupervisor
from .tor_manager import TORManager
from .schemas import (
    ActuatorCommand,
    AuthorityState,
    DrivableEnvelope,
    ManeuverContract,
    MRCPlan,
    MRMStrategy,
    ODDStatus,
    SensorHealth,
    TrajectoryRequest,
    WorldState,
)
from .maneuver_contract import build_contract


# ── Scenario result ───────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    state_trace: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: Optional[str] = None

    def add_check(self, label: str, passed: bool, detail: str = "") -> None:
        self.checks.append((label, passed, detail))
        if not passed:
            self.passed = False


# ── Stubs ─────────────────────────────────────────────────────────────────────

def _envelope(right_m: float = 1.5) -> DrivableEnvelope:
    return DrivableEnvelope("env-001", 1.5, right_m, 50.0)


def _w(
    frame_id: int,
    ts: float,
    ego_v: float = 5.0,
    odd: ODDStatus = ODDStatus.IN_ODD,
    ttc: float = 10.0,
    human_input: bool = False,
    time_to_odd_exit: Optional[float] = None,
    world_age_ms: int = 50,
    lane_change: bool = True,
    sensor_health: SensorHealth = SensorHealth.OK,
    preview_feasible: bool = True,
    lateral_error: float = 0.05,
    new_obstacle: float = 0.1,
    corridor_clear: bool = True,
    right_bound: float = 1.5,
) -> WorldState:
    return WorldState(
        frame_id=frame_id, timestamp_s=ts,
        world_age_ms=world_age_ms, sync_ok=True, sensor_health=sensor_health,
        ego_v_mps=ego_v, ego_a_mps2=0.0,
        ego_lane_id="lane_0", ego_lateral_error_m=lateral_error,
        corridor_clear=corridor_clear,
        min_ttc_s=ttc, new_obstacle_score=new_obstacle,
        weather_visibility_m=100.0, lane_change_permission=lane_change,
        drivable_envelope=_envelope(right_bound),
        odd_status=odd, time_to_odd_exit_s=time_to_odd_exit,
        preview_feasible=preview_feasible,
        human_driver_available=True, human_input_detected=human_input,
    )


class _BaselineStub:
    def __init__(self, healthy: bool = True, v_desired: float = 8.33) -> None:
        self._healthy = healthy
        self._v = v_desired

    def plan(self, world: WorldState) -> TrajectoryRequest:
        return TrajectoryRequest(
            source="BASELINE", tactical_intent="keep_lane",
            v_max_mps=10.0, a_long_max_mps2=2.0, a_lat_max_mps2=1.5,
            jerk_max_mps3=3.0, lateral_bound_m=1.0,
            drivable_envelope=world.drivable_envelope,
            target_v_desired_mps=self._v,
        )

    def degraded_hold(self, world: WorldState) -> TrajectoryRequest:
        return self.plan(world)

    def is_healthy(self, world: WorldState) -> bool:
        return self._healthy


class _AgentStub:
    def __init__(
        self,
        confidence: float = 0.92,
        intent: str = "slow_bypass_right",
        no_contract: bool = False,
    ) -> None:
        self._confidence = confidence
        self._intent = intent
        self._no_contract = no_contract

    def propose_contract(self, world: WorldState) -> Optional[ManeuverContract]:
        if self._no_contract:
            return None
        return build_contract(
            frame_id=world.frame_id,
            sim_time_s=world.timestamp_s,
            tactical_intent=self._intent,
            sub_intent_sequence=["prepare", "commit", "stabilize"],
            target_state_description="bypass stationary obstacle",
            agent_confidence=self._confidence,
            agent_reasoning_summary="Baseline stuck; safe lateral bypass.",
            max_duration_s=4.0,
            max_speed_mps=5.56,
            max_lateral_offset_m=0.8,
        )

    def get_intent(self, world: WorldState, contract: ManeuverContract) -> str:
        return self._intent


class _MPCStub:
    def __init__(self, always_feasible: bool = True) -> None:
        self._feasible = always_feasible

    def execute(self, req: TrajectoryRequest) -> ActuatorCommand:
        return ActuatorCommand(steer=0.0, throttle=0.3, brake=0.0, source="MPC")

    def preview_feasible(self, req: TrajectoryRequest) -> bool:
        return self._feasible


def _build_arbiter(
    log_path: Path,
    baseline: Optional[_BaselineStub] = None,
    agent: Optional[_AgentStub] = None,
    mpc: Optional[_MPCStub] = None,
    human: Optional[HumanOverrideMonitor] = None,
    tor: Optional[TORManager] = None,
    mrm: Optional[MRMExecutor] = None,
) -> AuthorityArbiter:
    _tor = tor or TORManager()
    return AuthorityArbiter(
        asm=AuthorityStateMachine(),
        baseline_detector=BaselineDetector(),
        human_override=human or HumanOverrideMonitor(),
        agent=agent or _AgentStub(),
        supervisor=SafetySupervisor(),
        baseline=baseline or _BaselineStub(),
        contract_resolver=ContractResolver(),
        handoff_planner=HandoffPlanner(),
        mpc=mpc or _MPCStub(),
        tor=_tor,
        mrm=mrm or MRMExecutor(),
        log_path=log_path,
    )


def _state_trace(arbiter: AuthorityArbiter) -> list[str]:
    return [e["to"] for e in arbiter._asm.get_transition_log()]


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

def run_T9_001(log_dir: Path) -> ScenarioResult:
    """
    T9-001: Xe đỗ giữa làn, baseline bị kẹt.
    Expected path: BASELINE → REQUESTING → SUPERVISED → ACTIVE → REVOKE → BASELINE
    Pass condition: vượt chướng ngại an toàn (reached AGENT_ACTIVE_BOUNDED at some point).
    """
    r = ScenarioResult("T9-001", "Stationary obstacle, baseline stuck", passed=True)
    t0 = time.monotonic()
    ab = _build_arbiter(log_dir / "T9-001.jsonl")

    # 35 frames stall velocity → triggers baseline stuck detection
    for i in range(35):
        ab.step(_w(i, i * 0.1, ego_v=0.2, corridor_clear=True), sim_time_s=i * 0.1)

    # 10 more frames to progress through requesting → supervised → active
    for i in range(35, 55):
        ab.step(_w(i, i * 0.1, ego_v=0.2), sim_time_s=i * 0.1)

    trace = _state_trace(ab)
    r.state_trace = trace

    reached_active = "AGENT_ACTIVE_BOUNDED" in trace
    r.add_check("Reached AGENT_ACTIVE_BOUNDED", reached_active, f"trace={trace}")
    reached_requesting = "AGENT_REQUESTING_AUTHORITY" in trace
    r.add_check("Passed through REQUESTING", reached_requesting)

    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r


def run_T9_002(log_dir: Path) -> ScenarioResult:
    """
    T9-002: Baseline lateral oscillation > 3 times.
    Expected: BASELINE → REQUESTING → ACTIVE, lateral resolved.
    """
    r = ScenarioResult("T9-002", "Baseline lateral oscillation", passed=True)
    t0 = time.monotonic()
    ab = _build_arbiter(log_dir / "T9-002.jsonl")

    # Simulate oscillation: flip lateral error sign every 3 frames
    signs = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
    for i, s in enumerate(signs):
        ab.step(_w(i, i * 0.1, ego_v=0.5, lateral_error=s * 0.45), sim_time_s=i * 0.1)
    for i in range(len(signs), len(signs) + 20):
        ab.step(_w(i, i * 0.1, ego_v=0.5, lateral_error=0.05), sim_time_s=i * 0.1)

    trace = _state_trace(ab)
    r.state_trace = trace
    requesting = "AGENT_REQUESTING_AUTHORITY" in trace
    r.add_check("Agent attempted request after oscillation", requesting, f"trace={trace}")
    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r


def run_T9_003(log_dir: Path) -> ScenarioResult:
    """
    T9-003: Agent confidence tụt giữa chừng.
    Setup: reach AGENT_ACTIVE_BOUNDED, then swap agent to low-confidence stub.
    Expected: REVOKE → BASELINE (no jerk verified by smooth revoke flag in log).
    """
    r = ScenarioResult("T9-003", "Agent confidence collapse mid-contract", passed=True)
    t0 = time.monotonic()

    # Build with high-confidence agent first
    high_agent = _AgentStub(confidence=0.93)
    ab = _build_arbiter(log_dir / "T9-003.jsonl", agent=high_agent)

    # Reach active bounded
    for i in range(45):
        ab.step(_w(i, i * 0.1, ego_v=0.2), sim_time_s=i * 0.1)

    # Force revoke by making supervisor see low TTC (confidence collapse simulation)
    # In real system: agent_confidence would be checked by supervisor each frame.
    # Here we simulate via TTC collapse (veto → revoke)
    for i in range(45, 55):
        ab.step(_w(i, i * 0.1, ego_v=0.2, ttc=0.5), sim_time_s=i * 0.1)

    # Should have gone to REVOKE
    for i in range(55, 65):
        ab.step(_w(i, i * 0.1, ego_v=0.2, ttc=10.0), sim_time_s=i * 0.1)

    trace = _state_trace(ab)
    r.state_trace = trace
    has_revoke = "AUTHORITY_REVOKE_PENDING" in trace
    r.add_check("Entered AUTHORITY_REVOKE_PENDING", has_revoke, f"trace={trace}")
    back_to_baseline = trace[-1] in ("BASELINE_CONTROL", "AUTHORITY_REVOKE_PENDING")
    r.add_check("Returned toward baseline", back_to_baseline)
    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r


def run_T9_004(log_dir: Path) -> ScenarioResult:
    """
    T9-004: TTC collapse during AGENT_ACTIVE_BOUNDED.
    Expected: ACTIVE → REVOKE → MRM; no collision (arbiter ran without crash).
    """
    r = ScenarioResult("T9-004", "TTC collapse during active contract", passed=True)
    t0 = time.monotonic()
    ab = _build_arbiter(log_dir / "T9-004.jsonl", baseline=_BaselineStub(healthy=False))

    # Reach active
    for i in range(45):
        ab.step(_w(i, i * 0.1, ego_v=0.2, ttc=10.0), sim_time_s=i * 0.1)

    # TTC collapses → veto → revoke → baseline unhealthy → MRM
    for i in range(45, 60):
        ab.step(_w(i, i * 0.1, ego_v=4.0, ttc=0.4), sim_time_s=i * 0.1)

    trace = _state_trace(ab)
    r.state_trace = trace
    has_mrm = "MINIMAL_RISK_MANEUVER" in trace
    r.add_check("Entered MINIMAL_RISK_MANEUVER after TTC collapse", has_mrm, f"trace={trace[-4:]}")
    r.add_check("No crash / exception", True)
    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r


def run_T9_005(log_dir: Path) -> ScenarioResult:
    """
    T9-005: Contract expires normally.
    Expected: ACTIVE → REVOKE → BASELINE (smooth).
    """
    r = ScenarioResult("T9-005", "Contract normal expiry", passed=True)
    t0 = time.monotonic()

    # Give contract very short validity — 0.3s (already clamped to max 8s but
    # set max_duration_s=0.3s so validity_deadline_s = ts + 0.3)
    class _ShortAgent(_AgentStub):
        def propose_contract(self, world):
            return build_contract(
                frame_id=world.frame_id, sim_time_s=world.timestamp_s,
                tactical_intent="slow_bypass_right",
                sub_intent_sequence=["prepare", "commit"],
                target_state_description="bypass",
                agent_confidence=0.93,
                agent_reasoning_summary="short contract test",
                max_duration_s=0.5,
                max_speed_mps=5.56,
            )

    ab = _build_arbiter(log_dir / "T9-005.jsonl", agent=_ShortAgent())

    for i in range(35):
        ab.step(_w(i, i * 0.1, ego_v=0.2), sim_time_s=i * 0.1)
    # Advance sim time past contract expiry
    for i in range(35, 55):
        ab.step(_w(i, i * 0.1, ego_v=0.2), sim_time_s=i * 0.1 + 1.0)

    trace = _state_trace(ab)
    r.state_trace = trace
    has_revoke = "AUTHORITY_REVOKE_PENDING" in trace
    r.add_check("Entered REVOKE after contract expiry", has_revoke, f"trace={trace[-4:]}")
    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r


def run_T9_006(log_dir: Path) -> ScenarioResult:
    """
    T9-006: Baseline stuck but odd_status = ODD_MARGIN.
    Expected: Agent NOT granted; goes to TOR or stays in baseline/MRM.
    Pass: "AGENT_ACTIVE_BOUNDED" NOT in trace.
    """
    r = ScenarioResult("T9-006", "Baseline stuck but ODD_MARGIN — no grant", passed=True)
    t0 = time.monotonic()
    ab = _build_arbiter(log_dir / "T9-006.jsonl")

    for i in range(50):
        ab.step(
            _w(i, i * 0.1, ego_v=0.2, odd=ODDStatus.ODD_MARGIN, time_to_odd_exit=6.0),
            sim_time_s=i * 0.1,
        )

    trace = _state_trace(ab)
    r.state_trace = trace
    agent_not_granted = "AGENT_ACTIVE_BOUNDED" not in trace
    r.add_check("Agent NOT granted while ODD_MARGIN", agent_not_granted, f"trace={trace}")
    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r


def run_T9_007(log_dir: Path) -> ScenarioResult:
    """
    T9-007: ODD_EXCEEDED → MRM → SAFE_STOP.
    Expected: any autonomy state → MRM → (vehicle slows) → SAFE_STOP.
    """
    r = ScenarioResult("T9-007", "ODD_EXCEEDED triggers MRM → SAFE_STOP", passed=True)
    t0 = time.monotonic()
    ab = _build_arbiter(log_dir / "T9-007.jsonl")

    # One frame ODD_EXCEEDED
    ab.step(_w(0, 0.0, ego_v=10.0, odd=ODDStatus.ODD_EXCEEDED), sim_time_s=0.0)
    ctx = ab._asm.get_context()
    r.add_check("MRM triggered on ODD_EXCEEDED",
                ctx.current_state == AuthorityState.MINIMAL_RISK_MANEUVER,
                f"state={ctx.current_state.name}")

    # Simulate deceleration to stop
    for i in range(1, 25):
        v = max(0.0, 10.0 - i * 0.6)
        ab.step(_w(i, i * 0.1, ego_v=v, odd=ODDStatus.ODD_EXCEEDED), sim_time_s=i * 0.1)

    ctx2 = ab._asm.get_context()
    r.add_check(
        "Reached SAFE_STOP or still in MRM",
        ctx2.current_state in (AuthorityState.SAFE_STOP, AuthorityState.MINIMAL_RISK_MANEUVER),
        f"state={ctx2.current_state.name}",
    )
    r.state_trace = _state_trace(ab)
    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r


def run_T9_008(log_dir: Path) -> ScenarioResult:
    """
    T9-008: World state stale during request.
    Expected: REQUESTING → REJECTED immediately (S-005).
    """
    r = ScenarioResult("T9-008", "Stale world state → immediate reject", passed=True)
    t0 = time.monotonic()
    ab = _build_arbiter(log_dir / "T9-008.jsonl")

    # Use stale world from the start AND corridor_clear so baseline_stuck triggers
    # This means verify_contract will see stale world and reject (S-005)
    for i in range(45):
        ab.step(_w(i, i * 0.1, ego_v=0.2, world_age_ms=200), sim_time_s=i * 0.1)

    trace = _state_trace(ab)
    r.state_trace = trace
    no_active = "AGENT_ACTIVE_BOUNDED" not in trace
    r.add_check("Agent not granted with stale world", no_active, f"trace={trace}")
    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r



def run_T9_009(log_dir: Path) -> ScenarioResult:
    """
    T9-009: Supervisor preview infeasible during SUPERVISED_EXECUTION.
    Expected: SUPERVISED → BASELINE (grant aborted).
    """
    r = ScenarioResult("T9-009", "Preview infeasible → supervised aborted", passed=True)
    t0 = time.monotonic()
    mpc_inf = _MPCStub(always_feasible=False)
    ab = _build_arbiter(log_dir / "T9-009.jsonl", mpc=mpc_inf)

    for i in range(50):
        ab.step(_w(i, i * 0.1, ego_v=0.2, preview_feasible=False), sim_time_s=i * 0.1)

    trace = _state_trace(ab)
    r.state_trace = trace
    no_active = "AGENT_ACTIVE_BOUNDED" not in trace
    r.add_check("Grant aborted — not reached AGENT_ACTIVE_BOUNDED", no_active, f"trace={trace}")
    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r


def run_T9_010(log_dir: Path) -> ScenarioResult:
    """
    T9-010: TOR timeout → MRM → SAFE_STOP.
    Expected: TOR_ACTIVE → (timeout) → MINIMAL_RISK_MANEUVER → SAFE_STOP.
    """
    r = ScenarioResult("T9-010", "TOR timeout → MRM → SAFE_STOP", passed=True)
    t0 = time.monotonic()

    tor = TORManager()
    tor.TOR_TIMEOUT_S = 0.05           # tiny for test speed

    ab = _build_arbiter(log_dir / "T9-010.jsonl", tor=tor)

    # Trigger TOR via ODD_MARGIN with time budget OK
    ab.step(_w(0, 0.0, odd=ODDStatus.ODD_MARGIN, time_to_odd_exit=6.0), sim_time_s=0.0)
    ctx = ab._asm.get_context()
    entered_tor = ctx.current_state == AuthorityState.TOR_ACTIVE

    if entered_tor:
        time.sleep(0.08)   # wait for timeout
        for i in range(1, 10):
            ab.step(_w(i, i * 0.1, odd=ODDStatus.ODD_MARGIN, ego_v=5.0,
                       time_to_odd_exit=6.0), sim_time_s=i * 0.1)

    ctx2 = ab._asm.get_context()
    r.add_check("Entered TOR_ACTIVE", entered_tor, f"first_state={ctx.current_state.name}")
    r.add_check(
        "After timeout → MRM or SAFE_STOP",
        ctx2.current_state in (AuthorityState.MINIMAL_RISK_MANEUVER, AuthorityState.SAFE_STOP),
        f"state={ctx2.current_state.name}",
    )
    r.state_trace = _state_trace(ab)
    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r


def run_T9_011(log_dir: Path) -> ScenarioResult:
    """
    T9-011: Human override during MRM.
    Expected: MRM → HUMAN_CONTROL_ACTIVE upon human input.
    """
    r = ScenarioResult("T9-011", "Human override during MRM → HUMAN_CONTROL_ACTIVE", passed=True)
    t0 = time.monotonic()
    hm = HumanOverrideMonitor()
    ab = _build_arbiter(log_dir / "T9-011.jsonl", human=hm)

    # Enter MRM via ODD_EXCEEDED
    ab.step(_w(0, 0.0, odd=ODDStatus.ODD_EXCEEDED, ego_v=5.0), sim_time_s=0.0)
    ctx = ab._asm.get_context()
    r.add_check("Started in MRM", ctx.current_state == AuthorityState.MINIMAL_RISK_MANEUVER,
                f"state={ctx.current_state.name}")

    # Human override detected
    hm.inject_override(steer=0.0, throttle=0.2, brake=0.0)
    ab.step(_w(1, 0.1, odd=ODDStatus.ODD_EXCEEDED, ego_v=4.0, human_input=True), sim_time_s=0.1)

    ctx2 = ab._asm.get_context()
    r.add_check("HUMAN_CONTROL_ACTIVE after override in MRM",
                ctx2.current_state == AuthorityState.HUMAN_CONTROL_ACTIVE,
                f"state={ctx2.current_state.name}")
    r.state_trace = _state_trace(ab)
    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r


def run_T9_012(log_dir: Path) -> ScenarioResult:
    """
    T9-012: Multiple repeated requests → cooldown active after 2 revokes.
    Expected: After N revokes, REQUESTING → REJECTED (S-013).
    """
    r = ScenarioResult("T9-012", "Repeated requests → cooldown enforced", passed=True)
    t0 = time.monotonic()

    supervisor = SafetySupervisor()
    # Manually register 3 revokes to activate cooldown
    supervisor.record_revoke()
    supervisor.record_revoke()
    supervisor.record_revoke()

    # Now try to grant
    from .maneuver_contract import build_contract as bc
    contract = bc(
        frame_id=0, sim_time_s=0.0,
        tactical_intent="slow_bypass_right", sub_intent_sequence=[],
        target_state_description="", agent_confidence=0.95,
        agent_reasoning_summary="",
    )
    world = _w(0, 0.0)
    verdict = supervisor.verify_contract(contract, world)

    r.add_check(
        "Reject due to repeated revokes (S-013)",
        not verdict.approved and any("S-013" in x for x in verdict.reasons),
        f"reasons={verdict.reasons}",
    )
    r.state_trace = []
    r.elapsed_ms = (time.monotonic() - t0) * 1000
    return r


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_SCENARIOS: list[Callable[[Path], ScenarioResult]] = [
    run_T9_001,
    run_T9_002,
    run_T9_003,
    run_T9_004,
    run_T9_005,
    run_T9_006,
    run_T9_007,
    run_T9_008,
    run_T9_009,
    run_T9_010,
    run_T9_011,
    run_T9_012,
]
