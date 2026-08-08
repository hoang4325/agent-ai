"""
Stage 9 — Human Override Monitor (v2)
=======================================
Detects when a human driver physically intervenes in the vehicle.

In a real system, this watches:
  - Steering torque sensor (override threshold)
  - Brake pedal position sensor
  - Throttle pedal position sensor
  - Manual disengage button / lever

In CARLA / simulation (stub mode):
  - Reads world.human_input_detected (set by test harness)
  - Supports programmatic injection via inject_override() for scenario testing

Design invariants:
  - Human override detection ALWAYS has higher priority than any autonomy state.
  - This module does NOT command the vehicle — it only reports state.
  - current_command() returns the last known human input as an ActuatorCommand.
    In sim, this is a conservative safe hold (throttle=0, brake=0.3).
  - "Released" = human has not applied input for >= RELEASE_HYSTERESIS_S.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .schemas import ActuatorCommand, WorldState


# Time without human input before considering it "released"
_RELEASE_HYSTERESIS_S = 1.5


@dataclass
class _HumanMonitorState:
    override_active: bool = False
    override_start_real_time_s: float = 0.0
    last_input_real_time_s: float = 0.0
    injected: bool = False                 # for test harness injection
    injected_command: ActuatorCommand = field(
        default_factory=lambda: ActuatorCommand(steer=0.0, throttle=0.0, brake=0.3)
    )
    last_command: ActuatorCommand = field(
        default_factory=lambda: ActuatorCommand(steer=0.0, throttle=0.0, brake=0.3)
    )


class HumanOverrideMonitor:
    """
    Human Override Monitor — Stage 9.

    Usage in arbiter:
      if human_override.detect(world):
          asm.transition_to(HUMAN_CONTROL_ACTIVE)
      if human_override.released():
          asm.transition_to(BASELINE_CONTROL)
      cmd = human_override.current_command()

    Scenario testing:
      human_override.inject_override(steer=0.3, throttle=0.0, brake=0.0)
      human_override.clear_override()
    """

    def __init__(self, release_hysteresis_s: float = _RELEASE_HYSTERESIS_S) -> None:
        self._state = _HumanMonitorState()
        self._hysteresis_s = release_hysteresis_s

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, world: WorldState) -> bool:
        """
        Returns True if human driver is actively applying input.
        Priority: injected override > world.human_input_detected.
        """
        if self._state.injected:
            self._set_active(world)
            return True

        if world.human_input_detected:
            self._set_active(world)
            return True

        return False

    def released(self) -> bool:
        """
        Returns True if the driver has NOT applied input for >= hysteresis duration.
        Used to determine when to offer autonomy re-engagement.
        """
        if not self._state.override_active:
            return True
        elapsed_since_last = time.monotonic() - self._state.last_input_real_time_s
        if elapsed_since_last >= self._hysteresis_s:
            self._state.override_active = False
            return True
        return False

    def current_command(self) -> ActuatorCommand:
        """
        Returns the current human-derived actuation command.
        In real system: from steering/pedal sensors.
        In CARLA sim stub: safe hold (throttle=0, moderate brake).
        """
        if self._state.injected:
            return self._state.injected_command
        return self._state.last_command

    # ── Scenario / test injection ─────────────────────────────────────────────

    def inject_override(
        self,
        steer: float = 0.0,
        throttle: float = 0.0,
        brake: float = 0.5,
    ) -> None:
        """Programmatically inject a human override (for scenario testing in CARLA)."""
        self._state.injected = True
        self._state.injected_command = ActuatorCommand(
            steer=steer,
            throttle=throttle,
            brake=brake,
            source="HUMAN",
        )
        self._state.override_active = True
        self._state.last_input_real_time_s = time.monotonic()

    def clear_override(self) -> None:
        """Remove injected override (simulate driver releasing controls)."""
        self._state.injected = False
        self._state.injected_command = ActuatorCommand(steer=0.0, throttle=0.0, brake=0.3)
        self._state.last_input_real_time_s = time.monotonic()
        # override_active stays True until hysteresis clears

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_active(self, world: WorldState) -> None:
        now = time.monotonic()
        if not self._state.override_active:
            self._state.override_start_real_time_s = now
        self._state.override_active = True
        self._state.last_input_real_time_s = now
