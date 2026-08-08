"""
Stage 9 — TOR Manager (v2)
============================
Takeover Request Manager — 3-level escalation with timeout policy.

Escalation timeline:
  Level 1 (t=0.0s):  VISUAL_AMBER  — amber flash
  Level 2 (t=2.5s):  VISUAL_RED + AUDIO — red flash + beep + haptic
  Level 3 (t=5.0s):  CRITICAL — continuous + MRM prepare

Policy:
  - If time_budget_s (time_to_odd_exit_s) drops below MIN_WAIT_BUDGET_S
    while TOR is active → arbiter must escalate to MRM immediately.
  - Timeout after TOR_TIMEOUT_S (8s) with no human response → SAFE_STOP path.
  - Once TOR is resolved (human responded OR timeout), it must be reset before
    it can be started again.

Design: stateful, reset-on-use. Not a singleton.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .schemas import TORSignal, WorldState


@dataclass
class _TORState:
    active: bool = False
    start_real_time_s: float = 0.0
    start_sim_time_s: float = 0.0
    current_level: int = 0
    reason: str = ""
    resolved: bool = False
    resolution: str = ""   # "HUMAN_RESPONDED" | "TIMEOUT" | "RISK_ESCALATED"


class TORManager:
    """
    Takeover Request Manager — Stage 9.

    Usage:
      tor.start(world, reason="odd_margin")
      tor.tick(world)          ← call every frame while TOR_ACTIVE
      if tor.timed_out(): ...
      tor.reset()              ← after resolution
    """

    # ── Configuration ─────────────────────────────────────────────────────────
    ESCALATION_LEVELS: dict[int, dict] = {
        1: {
            "t_start_s":    0.0,
            "alert":        "VISUAL_AMBER",
            "action":       "hmi_amber_flash",
            "description":  "Amber visual alert — driver attention requested",
        },
        2: {
            "t_start_s":    2.5,
            "alert":        "VISUAL_RED_AUDIO",
            "action":       "hmi_red_flash + audio_beep + haptic_pulse",
            "description":  "Red visual + audio + haptic — urgent takeover request",
        },
        3: {
            "t_start_s":    5.0,
            "alert":        "CRITICAL",
            "action":       "continuous_alert + mrm_prepare",
            "description":  "Critical alert — MRM preparation initiated",
        },
    }

    TOR_TIMEOUT_S:    float = 8.0
    MIN_WAIT_BUDGET_S: float = 3.0   # if time_to_odd_exit_s < this → immediate MRM

    def __init__(self) -> None:
        self._s = _TORState()

    # ── Start / tick / stop ───────────────────────────────────────────────────

    def start(self, world: WorldState, reason: str = "") -> None:
        """Activate TOR. Must call reset() after resolution before calling start() again."""
        self._s = _TORState(
            active=True,
            start_real_time_s=time.monotonic(),
            start_sim_time_s=world.timestamp_s,
            current_level=1,
            reason=reason or "unspecified",
        )
        self._emit_alert(1)

    def tick(self, world: WorldState) -> TORSignal:
        """
        Advance TOR state one frame. Returns current TORSignal.
        Must be called every frame while in TOR_ACTIVE state.
        """
        if not self._s.active:
            return self._make_signal(elapsed=0.0, human_detected=False)

        elapsed = self._elapsed_s()

        # Level escalation
        new_level = self._compute_level(elapsed)
        if new_level > self._s.current_level:
            self._s.current_level = new_level
            self._emit_alert(new_level)

        # Human input check (from world state)
        human_detected = world.human_input_detected

        return self._make_signal(elapsed, human_detected)

    def reset(self) -> None:
        """Reset TOR state after resolution. Must be called before next start()."""
        self._s = _TORState()

    # ── Query interface ───────────────────────────────────────────────────────

    def timed_out(self) -> bool:
        """Returns True if TOR_TIMEOUT_S exceeded with no human response."""
        return self._s.active and self._elapsed_s() >= self.TOR_TIMEOUT_S

    def is_active(self) -> bool:
        return self._s.active

    def current_level(self) -> int:
        return self._s.current_level

    def elapsed_s(self) -> float:
        return self._elapsed_s() if self._s.active else 0.0

    def current_signal(self, world: WorldState) -> TORSignal:
        return self._make_signal(self._elapsed_s(), world.human_input_detected)

    def time_budget_critical(self, world: WorldState) -> bool:
        """True if remaining ODD time is too short to keep waiting for human."""
        if world.time_to_odd_exit_s is None:
            return False
        return world.time_to_odd_exit_s < self.MIN_WAIT_BUDGET_S

    # ── Internal ──────────────────────────────────────────────────────────────

    def _elapsed_s(self) -> float:
        if not self._s.active:
            return 0.0
        return time.monotonic() - self._s.start_real_time_s

    def _compute_level(self, elapsed: float) -> int:
        level = 1
        for lvl, cfg in sorted(self.ESCALATION_LEVELS.items()):
            if elapsed >= cfg["t_start_s"]:
                level = lvl
        return level

    def _emit_alert(self, level: int) -> None:
        cfg = self.ESCALATION_LEVELS.get(level, {})
        # In a real system this would trigger HMI events.
        # In simulation / CARLA it logs to console.
        print(
            f"[TOR] Level {level} — {cfg.get('alert', '?')} | "
            f"action: {cfg.get('action', '?')} | "
            f"elapsed: {self._elapsed_s():.2f}s"
        )

    def _make_signal(self, elapsed: float, human_detected: bool) -> TORSignal:
        return TORSignal(
            level=self._s.current_level,
            started_at_s=self._s.start_sim_time_s,
            elapsed_s=elapsed,
            timeout_s=self.TOR_TIMEOUT_S,
            human_input_detected=human_detected,
        )
