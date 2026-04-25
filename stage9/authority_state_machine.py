"""
Stage 9 — Authority State Machine (v2)
=======================================
9-state FSM governing authority transitions between:
  BASELINE_CONTROL, AGENT_REQUESTING_AUTHORITY, SUPERVISED_EXECUTION,
  AGENT_ACTIVE_BOUNDED, AUTHORITY_REVOKE_PENDING,
  TOR_ACTIVE, HUMAN_CONTROL_ACTIVE, MINIMAL_RISK_MANEUVER, SAFE_STOP

Features:
  - Cooldown / hysteresis anti-thrashing
  - Supervised frames counter
  - Revoke frame ramp tracker
  - Last-agent-request cache for revoke blending
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .schemas import (
    AuthorityContext,
    AuthorityState,
    ActiveAuthority,
    ManeuverContract,
    TrajectoryRequest,
)


@dataclass
class _ASMMemory:
    """Internal mutable state tracked by the ASM."""
    current_state: AuthorityState = AuthorityState.BASELINE_CONTROL
    active_contract: Optional[ManeuverContract] = None
    contract_start_sim_time_s: Optional[float] = None

    # supervised handoff
    supervised_frames: int = 0

    # revoke ramp
    revoke_frame_count: int = 0
    revoke_total_frames: int = 5     # 5 frames @ 10 Hz = 0.5 s
    last_agent_request: Optional[TrajectoryRequest] = None

    # authority hygiene
    revoke_count_last_60s: int = 0
    last_revoke_real_time_s: Optional[float] = None
    cooldown_start_real_time_s: Optional[float] = None
    cooldown_duration_s: float = 20.0   # will be overridden by contract.cooldown_after_revoke_s

    # request gate
    request_start_real_time_s: Optional[float] = None
    request_timeout_s: float = 1.0

    # transition log
    transition_log: list = field(default_factory=list)
    state_entered_real_time_s: float = field(default_factory=time.monotonic)


class AuthorityStateMachine:
    """
    Authority State Machine — Stage 9.

    Responsibilities:
      - Maintain current state
      - Gate transitions with pre-conditions
      - Track cooldown / hysteresis
      - Expose AuthorityContext to arbiter
    """

    def __init__(self) -> None:
        self._m = _ASMMemory()

    # ── Read interface ────────────────────────────────────────────────────────

    def get_context(self, sim_time_s: float = 0.0) -> AuthorityContext:
        s = self._m
        contract_remaining = None
        if s.active_contract and s.contract_start_sim_time_s is not None:
            elapsed = sim_time_s - s.contract_start_sim_time_s
            contract_remaining = max(0.0, s.active_contract.max_duration_s - elapsed)

        state = s.current_state
        if state in (AuthorityState.AGENT_ACTIVE_BOUNDED, AuthorityState.SUPERVISED_EXECUTION):
            active_auth = ActiveAuthority.AGENT
            agent_w = 1.0 if state == AuthorityState.AGENT_ACTIVE_BOUNDED else 0.5
        elif state == AuthorityState.HUMAN_CONTROL_ACTIVE:
            active_auth = ActiveAuthority.HUMAN
            agent_w = 0.0
        elif state in (AuthorityState.MINIMAL_RISK_MANEUVER, AuthorityState.SAFE_STOP):
            active_auth = ActiveAuthority.MRM
            agent_w = 0.0
        else:
            active_auth = ActiveAuthority.BASELINE
            agent_w = 0.5 if state == AuthorityState.AUTHORITY_REVOKE_PENDING else 0.0

        cooldown_remaining = self._cooldown_remaining()

        return AuthorityContext(
            current_state=state,
            active_authority=active_auth,
            active_contract=s.active_contract,
            contract_remaining_s=contract_remaining,
            baseline_weight=1.0 - agent_w,
            agent_weight=agent_w,
            cooldown_remaining_s=cooldown_remaining,
            tor_active=(state == AuthorityState.TOR_ACTIVE),
        )

    @property
    def current_state(self) -> AuthorityState:
        return self._m.current_state

    @property
    def supervised_frames(self) -> int:
        return self._m.supervised_frames

    @property
    def revoke_frame_count(self) -> int:
        return self._m.revoke_frame_count

    def last_agent_request(self) -> Optional[TrajectoryRequest]:
        return self._m.last_agent_request

    def current_revoke_alpha(self) -> float:
        """alpha: 1.0 = full agent, 0.0 = full baseline, ramps down over revoke window."""
        n = self._m.revoke_total_frames
        k = self._m.revoke_frame_count
        return max(0.0, 1.0 - k / n)

    def revoke_complete(self) -> bool:
        return self._m.revoke_frame_count >= self._m.revoke_total_frames

    def request_timed_out(self) -> bool:
        if self._m.request_start_real_time_s is None:
            return False
        return (time.monotonic() - self._m.request_start_real_time_s) > self._m.request_timeout_s

    def cooldown_active(self) -> bool:
        return self._cooldown_remaining() > 0.0

    # ── Write interface ───────────────────────────────────────────────────────

    def transition_to(
        self,
        new_state: AuthorityState,
        *,
        contract: Optional[ManeuverContract] = None,
        reason: str = "",
        sim_time_s: float = 0.0,
    ) -> None:
        old_state = self._m.current_state
        if old_state == new_state and new_state != AuthorityState.BASELINE_CONTROL:
            return  # no-op for same-state (except baseline re-entry)

        self._log_transition(old_state, new_state, reason)

        # state-exit housekeeping
        if old_state in (AuthorityState.AGENT_ACTIVE_BOUNDED, AuthorityState.AUTHORITY_REVOKE_PENDING):
            self._handle_revoke_accounting(contract)

        # state-enter housekeeping
        self._m.current_state = new_state
        self._m.state_entered_real_time_s = time.monotonic()

        if new_state == AuthorityState.AGENT_REQUESTING_AUTHORITY:
            self._m.active_contract = contract
            self._m.request_start_real_time_s = time.monotonic()

        elif new_state == AuthorityState.SUPERVISED_EXECUTION:
            if contract:
                self._m.active_contract = contract
            self._m.supervised_frames = 0

        elif new_state == AuthorityState.AGENT_ACTIVE_BOUNDED:
            self._m.contract_start_sim_time_s = sim_time_s
            self._m.revoke_frame_count = 0

        elif new_state == AuthorityState.AUTHORITY_REVOKE_PENDING:
            self._m.revoke_frame_count = 0

        elif new_state == AuthorityState.BASELINE_CONTROL:
            self._m.active_contract = None
            self._m.supervised_frames = 0

        elif new_state == AuthorityState.SAFE_STOP:
            self._m.active_contract = None

    def increment_supervised_frames(self) -> None:
        self._m.supervised_frames += 1

    def increment_revoke_frame(self) -> None:
        self._m.revoke_frame_count += 1

    def start_revoke_window(self) -> None:
        self._m.revoke_frame_count = 0

    def cache_agent_request(self, req: TrajectoryRequest) -> None:
        """Cache the most recent agent-derived request for revoke blending."""
        self._m.last_agent_request = req

    # ── Internal ─────────────────────────────────────────────────────────────

    def _cooldown_remaining(self) -> float:
        if self._m.cooldown_start_real_time_s is None:
            return 0.0
        elapsed = time.monotonic() - self._m.cooldown_start_real_time_s
        remaining = self._m.cooldown_duration_s - elapsed
        if remaining <= 0:
            self._m.cooldown_start_real_time_s = None
            return 0.0
        return remaining

    def _handle_revoke_accounting(self, contract: Optional[ManeuverContract]) -> None:
        now = time.monotonic()
        self._m.revoke_count_last_60s += 1
        self._m.last_revoke_real_time_s = now

        # prune old revoke count
        # (simplified: decrement after 60s — proper impl uses deque timestamps)
        cd = contract.cooldown_after_revoke_s if contract else 20.0
        self._m.cooldown_duration_s = cd
        self._m.cooldown_start_real_time_s = now

    def _log_transition(
        self,
        old_state: AuthorityState,
        new_state: AuthorityState,
        reason: str,
    ) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "from": old_state.name,
            "to": new_state.name,
            "reason": reason,
        }
        self._m.transition_log.append(entry)

    def get_transition_log(self) -> list:
        return list(self._m.transition_log)
