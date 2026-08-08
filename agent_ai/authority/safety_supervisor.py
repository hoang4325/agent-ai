"""
Stage 9 — Safety Supervisor (v2)
==================================
4-layer independent safety verification and per-frame veto.

Layers:
  1. Physical hard constraints
  2. Freshness / ODD / sensor validity
  3. Situational safety (TTC, lateral error, preview)
  4. Authority hygiene (confidence floor, repeated-revoke cooldown)

This module is INDEPENDENT of the agent. It has veto authority over
every grant (verify_contract) and every frame (watch_frame).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .schemas import (
    ManeuverContract,
    ODDStatus,
    SafetyVerdict,
    SensorHealth,
    VetoSignal,
    VetoSeverity,
    WorldState,
)


@dataclass
class _SupervisorMemory:
    recent_revoke_timestamps: deque = field(default_factory=deque)
    contract_start_sim_time_s: float = 0.0
    active_contract: ManeuverContract | None = None


class SafetySupervisor:
    """
    Safety Supervisor — Stage 9.

    Usage:
      verdict = supervisor.verify_contract(contract, world)   # before grant
      veto    = supervisor.watch_frame(world, contract)       # every frame during ACTIVE
      allowed = supervisor.forward_takeover_allowed(world)    # quick gate
    """

    # absolute hard limits (mirror of maneuver_contract._ABS_*)
    _ABS_MAX_LATERAL_M   = 1.0
    _ABS_MAX_SPEED_MPS   = 13.89
    _ABS_MAX_DURATION_S  = 8.0
    _ABS_MAX_JERK        = 4.0
    _REVOKE_WINDOW_S     = 60.0
    _MAX_REVOKES_IN_WIN  = 2

    def __init__(self) -> None:
        self._mem = _SupervisorMemory()

    # ── Grant-time verification (4 layers) ───────────────────────────────────

    def verify_contract(
        self,
        contract: ManeuverContract,
        world: WorldState,
    ) -> SafetyVerdict:
        verdict = SafetyVerdict(approved=True)

        # Layer 1: physical hard constraints
        if contract.max_lateral_offset_m > self._ABS_MAX_LATERAL_M:
            verdict.reject("S-001: lateral offset exceeds hard limit 1.0m")
        if contract.max_speed_mps > self._ABS_MAX_SPEED_MPS:
            verdict.reject("S-002: speed cap exceeded (> 13.89 m/s)")
        if contract.max_duration_s > self._ABS_MAX_DURATION_S:
            verdict.reject("S-003: contract duration > 8.0s hard limit")
        if contract.max_jerk_mps3 > self._ABS_MAX_JERK:
            verdict.reject("S-004: jerk bound exceeds 4.0 m/s³")

        # Layer 2: freshness / ODD / sensor validity
        if world.world_age_ms > contract.freshness_required_ms:
            verdict.reject(f"S-005: world state stale ({world.world_age_ms} ms > {contract.freshness_required_ms} ms)")
        if not world.sync_ok:
            verdict.reject("S-006: sensor sync not OK")
        if world.odd_status != ODDStatus.IN_ODD:
            verdict.reject(f"S-007: ODD status is {world.odd_status.value}, grant only allowed IN_ODD")
        if world.sensor_health == SensorHealth.FAULT:
            verdict.reject("S-008: sensor health FAULT")

        # Layer 3: situational safety
        ttc_floor = max(2.0, contract.min_ttc_threshold_s * 1.5)
        if world.min_ttc_s < ttc_floor:
            verdict.reject(f"S-009: TTC {world.min_ttc_s:.2f}s < floor {ttc_floor:.2f}s at grant")
        if world.ego_lateral_error_m > 0.6:
            verdict.reject(f"S-010: ego lateral error {world.ego_lateral_error_m:.2f}m > 0.6m; not stable for handoff")
        if not world.preview_feasible:
            verdict.reject("S-011: preview trajectory infeasible at grant time")

        # Layer 4: authority hygiene
        if contract.agent_confidence < 0.85:
            verdict.reject(f"S-012: agent confidence {contract.agent_confidence:.2f} < 0.85 floor")
        recent_revokes = self._count_recent_revokes()
        if recent_revokes > self._MAX_REVOKES_IN_WIN:
            verdict.reject(f"S-013: {recent_revokes} revokes in last 60s — cooldown active")

        return verdict

    # ── Per-frame watch (called every frame during AGENT_ACTIVE_BOUNDED) ─────

    def watch_frame(
        self,
        world: WorldState,
        contract: ManeuverContract,
        sim_time_s: float = 0.0,
    ) -> VetoSignal | None:
        if world.world_age_ms > contract.freshness_required_ms:
            return VetoSignal(
                f"S-V001: stale world ({world.world_age_ms} ms > {contract.freshness_required_ms} ms)",
                severity=VetoSeverity.HARD,
            )
        if world.odd_status != ODDStatus.IN_ODD:
            return VetoSignal(
                f"S-V002: ODD boundary breached ({world.odd_status.value}) during active contract",
                severity=VetoSeverity.HARD,
            )
        if world.min_ttc_s < contract.min_ttc_threshold_s:
            return VetoSignal(
                f"S-V003: TTC collapsed to {world.min_ttc_s:.2f}s < {contract.min_ttc_threshold_s}s",
                severity=VetoSeverity.HARD,
            )
        if world.new_obstacle_score > contract.max_new_obstacle_score:
            return VetoSignal(
                f"S-V004: new obstacle score {world.new_obstacle_score:.2f} > {contract.max_new_obstacle_score}",
                severity=VetoSeverity.HARD,
            )
        if contract.planner_feasibility_required and not world.preview_feasible:
            return VetoSignal(
                "S-V005: MPC reports trajectory infeasible during active contract",
                severity=VetoSeverity.HARD,
            )
        if self.contract_timer_expired(contract, sim_time_s):
            return VetoSignal(
                f"S-V006: contract expired (sim_time={sim_time_s:.2f}s >= deadline={contract.validity_deadline_s:.2f}s)",
                severity=VetoSeverity.SOFT,
            )
        return None

    # ── Helper gates for arbiter ──────────────────────────────────────────────

    def forward_takeover_allowed(self, world: WorldState) -> bool:
        """Quick gate: can the arbiter even attempt a forward takeover?"""
        return (
            world.odd_status == ODDStatus.IN_ODD
            and world.world_age_ms <= 100
            and world.sync_ok
            and world.sensor_health != SensorHealth.FAULT
            and self._count_recent_revokes() <= self._MAX_REVOKES_IN_WIN
        )

    def autonomy_reengage_allowed(self, world: WorldState) -> bool:
        """After human control: can we return to baseline autonomy?"""
        return (
            world.odd_status == ODDStatus.IN_ODD
            and world.sync_ok
            and world.sensor_health in (SensorHealth.OK, SensorHealth.DEGRADED_BUT_VALID)
            and world.preview_feasible
        )

    def must_enter_mrm_from_manual(self, world: WorldState) -> bool:
        """During HUMAN_CONTROL_ACTIVE: should we force MRM?"""
        return (
            world.sensor_health == SensorHealth.FAULT
            or world.odd_status == ODDStatus.ODD_EXCEEDED
        )

    def critical_fault(self, world: WorldState) -> bool:
        return (
            world.sensor_health == SensorHealth.FAULT
            or (world.odd_status == ODDStatus.ODD_EXCEEDED)
        )

    def contract_timer_expired(self, contract: ManeuverContract, sim_time_s: float) -> bool:
        return sim_time_s >= contract.validity_deadline_s

    # ── Audit ─────────────────────────────────────────────────────────────────

    def record_revoke(self) -> None:
        self._mem.recent_revoke_timestamps.append(time.monotonic())
        self._prune_revoke_window()

    def _count_recent_revokes(self) -> int:
        self._prune_revoke_window()
        return len(self._mem.recent_revoke_timestamps)

    def _prune_revoke_window(self) -> None:
        now = time.monotonic()
        cutoff = now - self._REVOKE_WINDOW_S
        while (
            self._mem.recent_revoke_timestamps
            and self._mem.recent_revoke_timestamps[0] < cutoff
        ):
            self._mem.recent_revoke_timestamps.popleft()
