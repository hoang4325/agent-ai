"""
Stage 9 — Authority Arbiter (v2)
===================================
Main arbitration loop — runs at 10 Hz.

Implements the full pseudocode from stage9_takeover_design_v2.md Section 8.

Design invariants enforced here:
  - Human override always wins, regardless of autonomy state.
  - Agent never touches L3 (actuator commands).
  - HandoffPlanner blends at reference level only.
  - Defensive fallback: any unexpected state → MRM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from .authority_logger import AuthorityLogger
from .authority_state_machine import AuthorityStateMachine
from .baseline_detector import BaselineDetector
from .contract_resolver import ContractResolver
from .handoff_planner import HandoffPlanner
from .safety_supervisor import SafetySupervisor
from .schemas import (
    ActuatorCommand,
    AuthorityState,
    ManeuverContract,
    MRCPlan,
    MRMStrategy,
    ODDStatus,
    TrajectoryRequest,
    WorldState,
)


# ── Protocols (interfaces for MPC, Baseline, Agent, TOR, MRM stubs) ──────────

@runtime_checkable
class BaselinePlannerProto(Protocol):
    def plan(self, world: WorldState) -> TrajectoryRequest: ...
    def degraded_hold(self, world: WorldState) -> TrajectoryRequest: ...
    def is_healthy(self, world: WorldState) -> bool: ...


@runtime_checkable
class Stage9AgentAdapterProto(Protocol):
    def propose_contract(self, world: WorldState) -> Optional[ManeuverContract]: ...
    def get_intent(self, world: WorldState, contract: ManeuverContract) -> str: ...


@runtime_checkable
class MPCOptimizerProto(Protocol):
    def execute(self, req: TrajectoryRequest) -> ActuatorCommand: ...
    def preview_feasible(self, req: TrajectoryRequest) -> bool: ...


@runtime_checkable
class TORManagerProto(Protocol):
    MIN_WAIT_BUDGET_S: float
    def start(self, world: WorldState) -> None: ...
    def tick(self, world: WorldState) -> None: ...
    def timed_out(self) -> bool: ...


@runtime_checkable
class MRMExecutorProto(Protocol):
    def compute(self, world: WorldState) -> MRCPlan: ...
    def is_full_stop_reached(self, world: WorldState) -> bool: ...
    def get_standstill_request(self, world: WorldState) -> TrajectoryRequest: ...


@runtime_checkable
class HumanOverrideMonitorProto(Protocol):
    def detect(self, world: WorldState) -> bool: ...
    def released(self) -> bool: ...
    def current_command(self) -> ActuatorCommand: ...


# ── AuthorityArbiter ──────────────────────────────────────────────────────────

class AuthorityArbiter:
    """
    Stage 9 Authority Arbiter — 10 Hz main loop.

    Call `step(world, sim_time_s)` each tick. Returns an ActuatorCommand
    produced exclusively by MPC (or human override monitor in HUMAN_CONTROL_ACTIVE).
    """

    def __init__(
        self,
        asm: AuthorityStateMachine,
        baseline_detector: BaselineDetector,
        human_override: HumanOverrideMonitorProto,
        agent: Stage9AgentAdapterProto,
        supervisor: SafetySupervisor,
        baseline: BaselinePlannerProto,
        contract_resolver: ContractResolver,
        handoff_planner: HandoffPlanner,
        mpc: MPCOptimizerProto,
        tor: TORManagerProto,
        mrm: MRMExecutorProto,
        log_path: Optional[Path] = None,
    ) -> None:
        self._asm = asm
        self._bd = baseline_detector
        self._human = human_override
        self._agent = agent
        self._supervisor = supervisor
        self._baseline = baseline
        self._resolver = contract_resolver
        self._handoff = handoff_planner
        self._mpc = mpc
        self._tor = tor
        self._mrm = mrm
        self._logger = AuthorityLogger(log_path)
        self._frame_id: int = 0

    def step(self, world: WorldState, sim_time_s: float = 0.0) -> ActuatorCommand:
        """Execute one arbitration tick. Returns ActuatorCommand from MPC."""
        self._frame_id = world.frame_id
        self._logger.set_frame(world.frame_id)
        self._bd.update(world)

        ctx = self._asm.get_context(sim_time_s)
        state = ctx.current_state

        # ── HIGHEST PRIORITY: explicit human input ────────────────────────────
        if self._human.detect(world):
            if state != AuthorityState.HUMAN_CONTROL_ACTIVE:
                self._logger.log_human_override()
                self._asm.transition_to(
                    AuthorityState.HUMAN_CONTROL_ACTIVE,
                    reason="manual_override",
                    sim_time_s=sim_time_s,
                )
                self._logger.log_state_transition(state.name, "HUMAN_CONTROL_ACTIVE", "manual_override")
            return self._human.current_command()

        # Refresh context after possible human-override transition
        ctx = self._asm.get_context(sim_time_s)
        state = ctx.current_state

        # ── SAFE_STOP ─────────────────────────────────────────────────────────
        if state == AuthorityState.SAFE_STOP:
            return self._mpc.execute(self._mrm.get_standstill_request(world))

        # ── HUMAN_CONTROL_ACTIVE ──────────────────────────────────────────────
        if state == AuthorityState.HUMAN_CONTROL_ACTIVE:
            if self._human.released() and self._supervisor.autonomy_reengage_allowed(world):
                self._asm.transition_to(AuthorityState.BASELINE_CONTROL, reason="manual_release", sim_time_s=sim_time_s)
                self._logger.log_state_transition("HUMAN_CONTROL_ACTIVE", "BASELINE_CONTROL", "manual_release")
                return self._mpc.execute(self._baseline.plan(world))
            if self._supervisor.must_enter_mrm_from_manual(world):
                mrm_plan = self._mrm.compute(world)
                self._asm.transition_to(AuthorityState.MINIMAL_RISK_MANEUVER, reason="manual_fault_fallback", sim_time_s=sim_time_s)
                self._logger.log_mrm_start("manual_fault_fallback", mrm_plan.strategy.value)
                return self._mpc.execute(mrm_plan.to_trajectory_request())
            return self._human.current_command()

        # ── MINIMAL_RISK_MANEUVER ─────────────────────────────────────────────
        if state == AuthorityState.MINIMAL_RISK_MANEUVER:
            mrm_plan = self._mrm.compute(world)
            cmd = self._mpc.execute(mrm_plan.to_trajectory_request())
            if self._mrm.is_full_stop_reached(world):
                self._asm.transition_to(AuthorityState.SAFE_STOP, reason="mrm_complete", sim_time_s=sim_time_s)
                self._logger.log_safe_stop()
            return cmd

        # ── TOR_ACTIVE ────────────────────────────────────────────────────────
        if state == AuthorityState.TOR_ACTIVE:
            self._tor.tick(world)
            low_budget = (
                world.time_to_odd_exit_s is not None
                and world.time_to_odd_exit_s < self._tor.MIN_WAIT_BUDGET_S
            )
            if self._tor.timed_out() or low_budget:
                mrm_plan = self._mrm.compute(world)
                self._asm.transition_to(AuthorityState.MINIMAL_RISK_MANEUVER, reason="tor_timeout_or_low_budget", sim_time_s=sim_time_s)
                self._logger.log_tor_timeout()
                self._logger.log_mrm_start("tor_timeout", mrm_plan.strategy.value)
                return self._mpc.execute(mrm_plan.to_trajectory_request())
            return self._mpc.execute(self._baseline.degraded_hold(world))

        # ── BASELINE_CONTROL ──────────────────────────────────────────────────
        if state == AuthorityState.BASELINE_CONTROL:
            # Critical fault or ODD exceeded → MRM immediately
            if self._supervisor.critical_fault(world):
                mrm_plan = self._mrm.compute(world)
                self._asm.transition_to(AuthorityState.MINIMAL_RISK_MANEUVER, reason="odd_exceeded_or_critical_fault", sim_time_s=sim_time_s)
                self._logger.log_mrm_start("critical_fault", mrm_plan.strategy.value)
                return self._mpc.execute(mrm_plan.to_trajectory_request())

            # ODD margin → TOR if human available and sufficient time budget
            if (
                world.odd_status == ODDStatus.ODD_MARGIN
                and world.human_driver_available
                and (world.time_to_odd_exit_s or 0.0) >= 5.0
            ):
                self._asm.transition_to(AuthorityState.TOR_ACTIVE, reason="odd_margin_tor", sim_time_s=sim_time_s)
                self._tor.start(world)
                self._logger.log_tor_start("odd_margin", world.time_to_odd_exit_s)
                return self._mpc.execute(self._baseline.degraded_hold(world))

            baseline_req = self._baseline.plan(world)

            # Forward takeover gate
            if (
                self._bd.is_stuck(world)
                and self._supervisor.forward_takeover_allowed(world)
                and not self._asm.cooldown_active()
            ):
                contract = self._agent.propose_contract(world)
                if contract is not None:
                    self._asm.transition_to(
                        AuthorityState.AGENT_REQUESTING_AUTHORITY,
                        contract=contract,
                        sim_time_s=sim_time_s,
                    )
                    self._logger.log_state_transition("BASELINE_CONTROL", "AGENT_REQUESTING_AUTHORITY", "baseline_stuck")

            return self._mpc.execute(baseline_req)

        # ── AGENT_REQUESTING_AUTHORITY ────────────────────────────────────────
        if state == AuthorityState.AGENT_REQUESTING_AUTHORITY:
            contract = ctx.active_contract
            if contract is None:
                self._asm.transition_to(AuthorityState.BASELINE_CONTROL, reason="nil_contract", sim_time_s=sim_time_s)
                return self._mpc.execute(self._baseline.plan(world))

            verdict = self._supervisor.verify_contract(contract, world)
            if not verdict.approved:
                self._asm.transition_to(AuthorityState.BASELINE_CONTROL, reason=verdict.primary_reason(), sim_time_s=sim_time_s)
                self._logger.log_reject(verdict.primary_reason(), verdict.reasons, contract.contract_id)
                return self._mpc.execute(self._baseline.plan(world))

            if self._asm.request_timed_out():
                self._asm.transition_to(AuthorityState.BASELINE_CONTROL, reason="request_timeout", sim_time_s=sim_time_s)
                self._logger.log_reject("request_timeout", [], contract.contract_id)
                return self._mpc.execute(self._baseline.plan(world))

            # Approved → begin supervised handoff
            self._asm.transition_to(AuthorityState.SUPERVISED_EXECUTION, contract=contract, sim_time_s=sim_time_s)
            self._logger.log_state_transition("AGENT_REQUESTING_AUTHORITY", "SUPERVISED_EXECUTION", "approved")
            return self._mpc.execute(self._baseline.plan(world))

        # ── SUPERVISED_EXECUTION ──────────────────────────────────────────────
        if state == AuthorityState.SUPERVISED_EXECUTION:
            contract = ctx.active_contract
            if contract is None:
                self._asm.transition_to(AuthorityState.BASELINE_CONTROL, reason="nil_contract_supervised", sim_time_s=sim_time_s)
                return self._mpc.execute(self._baseline.plan(world))

            baseline_req = self._baseline.plan(world)
            agent_intent = self._agent.get_intent(world, contract)
            agent_req = self._resolver.resolve(agent_intent, contract)

            preview = self._handoff.preview(baseline_req, agent_req, alpha=0.5)

            veto = self._supervisor.watch_frame(world, contract, sim_time_s=sim_time_s)
            if veto is not None or not self._mpc.preview_feasible(preview):
                reason = veto.reason if veto else "preview_infeasible"
                self._asm.transition_to(AuthorityState.BASELINE_CONTROL, reason=reason, sim_time_s=sim_time_s)
                if veto:
                    self._logger.log_veto(veto.reason, veto.severity.value, contract.contract_id)
                return self._mpc.execute(baseline_req)

            self._asm.increment_supervised_frames()
            if self._asm.supervised_frames >= 3:
                self._asm.transition_to(AuthorityState.AGENT_ACTIVE_BOUNDED, sim_time_s=sim_time_s)
                self._logger.log_grant(
                    contract.contract_id,
                    contract.tactical_intent,
                    contract.agent_confidence,
                    contract.max_duration_s,
                    "SUPERVISED_EXECUTION",
                )

            return self._mpc.execute(preview)

        # ── AGENT_ACTIVE_BOUNDED ──────────────────────────────────────────────
        if state == AuthorityState.AGENT_ACTIVE_BOUNDED:
            contract = ctx.active_contract
            if contract is None:
                self._asm.transition_to(AuthorityState.BASELINE_CONTROL, reason="nil_contract_active", sim_time_s=sim_time_s)
                return self._mpc.execute(self._baseline.plan(world))

            veto = self._supervisor.watch_frame(world, contract, sim_time_s=sim_time_s)
            if veto is not None:
                self._asm.transition_to(AuthorityState.AUTHORITY_REVOKE_PENDING, reason=veto.reason, sim_time_s=sim_time_s)
                self._asm.start_revoke_window()
                self._supervisor.record_revoke()
                self._logger.log_veto(veto.reason, veto.severity.value, contract.contract_id)
                self._logger.log_revoke(veto.reason, contract.contract_id, revoke_alpha_at_trigger=1.0)
                return self._mpc.execute(self._baseline.plan(world))

            agent_intent = self._agent.get_intent(world, contract)
            agent_req = self._resolver.resolve(agent_intent, contract)
            self._asm.cache_agent_request(agent_req)
            return self._mpc.execute(agent_req)

        # ── AUTHORITY_REVOKE_PENDING ──────────────────────────────────────────
        if state == AuthorityState.AUTHORITY_REVOKE_PENDING:
            baseline_req = self._baseline.plan(world)
            last_agent_req = self._asm.last_agent_request()
            revoke_alpha = self._asm.current_revoke_alpha()

            transition_req = self._handoff.preview(
                baseline_req=baseline_req,
                candidate_req=last_agent_req or baseline_req,
                alpha=revoke_alpha,
            )

            self._asm.increment_revoke_frame()

            if self._asm.revoke_complete():
                if self._baseline.is_healthy(world):
                    self._asm.transition_to(AuthorityState.BASELINE_CONTROL, reason="revoke_complete_to_baseline", sim_time_s=sim_time_s)
                elif (
                    world.human_driver_available
                    and (world.time_to_odd_exit_s or 0.0) >= 5.0
                ):
                    self._asm.transition_to(AuthorityState.TOR_ACTIVE, reason="baseline_unhealthy_tor", sim_time_s=sim_time_s)
                    self._tor.start(world)
                    self._logger.log_tor_start("baseline_unhealthy_after_revoke", world.time_to_odd_exit_s)
                else:
                    mrm_plan = self._mrm.compute(world)
                    self._asm.transition_to(AuthorityState.MINIMAL_RISK_MANEUVER, reason="baseline_unhealthy_mrm", sim_time_s=sim_time_s)
                    self._logger.log_mrm_start("baseline_unhealthy_after_revoke", mrm_plan.strategy.value)

            return self._mpc.execute(transition_req)

        # ── Defensive fallback for unexpected states ───────────────────────────
        mrm_plan = self._mrm.compute(world)
        self._asm.transition_to(AuthorityState.MINIMAL_RISK_MANEUVER, reason="unexpected_state_fallback", sim_time_s=sim_time_s)
        self._logger.log_mrm_start("unexpected_state_fallback", mrm_plan.strategy.value)
        return self._mpc.execute(mrm_plan.to_trajectory_request())
