"""
Stage 9 — Baseline Detector (v2)
==================================
Detects when the baseline planner is stuck, oscillating, or degenerate.
Uses a sliding window of world state observations to avoid false positives.

Three detection signals:
  1. velocity_stall  : ego < 1.0 m/s for > 3.0 s while corridor is clear
  2. lateral_osc     : > 3 lateral oscillations within 5.0 s window
  3. planner_degeneracy : MPC cost > 5x nominal OR convergence failed
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .schemas import WorldState


@dataclass
class _ObsFrame:
    timestamp_s: float
    ego_v_mps: float
    ego_lateral_error_m: float
    corridor_clear: bool
    mpc_cost_normalized: float    # cost / nominal_cost
    mpc_converged: bool


class BaselineDetector:
    """
    Sliding-window stuck / oscillation / degeneracy detector.

    is_stuck() returns True only when confirmed over a sustained window.
    raw_score() returns [0, 1] for use in precondition gate.
    """

    STALL_V_THRESHOLD_MPS = 1.0
    STALL_MIN_DURATION_S  = 3.0
    OSC_COUNT_THRESHOLD   = 3
    OSC_WINDOW_S          = 5.0
    DEGEN_COST_RATIO      = 5.0

    def __init__(self, window_s: float = 5.0) -> None:
        self._window_s = window_s
        self._buffer: deque[_ObsFrame] = deque()
        self._last_lateral_sign: int = 0
        self._osc_events: deque[float] = deque()  # timestamps of direction flips

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        world: WorldState,
        mpc_cost_normalized: float = 1.0,
        mpc_converged: bool = True,
    ) -> None:
        """Call once per frame (10 Hz)."""
        frame = _ObsFrame(
            timestamp_s=world.timestamp_s,
            ego_v_mps=world.ego_v_mps,
            ego_lateral_error_m=world.ego_lateral_error_m,
            corridor_clear=world.corridor_clear,
            mpc_cost_normalized=mpc_cost_normalized,
            mpc_converged=mpc_converged,
        )
        self._buffer.append(frame)
        self._prune(world.timestamp_s)
        self._track_oscillation(world)

    def is_stuck(self, world: WorldState) -> bool:
        """Returns True if ANY sustained baseline failure is detected."""
        return (
            self._detect_velocity_stall(world.timestamp_s)
            or self._detect_lateral_oscillation(world.timestamp_s)
            or self._detect_planner_degeneracy()
        )

    def stuck_score(self, world: WorldState) -> float:
        """Continuous score in [0, 1]. Threshold at 0.7 in forward takeover gate."""
        signals = [
            1.0 if self._detect_velocity_stall(world.timestamp_s) else 0.0,
            1.0 if self._detect_lateral_oscillation(world.timestamp_s) else 0.0,
            1.0 if self._detect_planner_degeneracy() else 0.0,
        ]
        return sum(signals) / len(signals)

    def active_signals(self, world: WorldState) -> list[str]:
        out = []
        if self._detect_velocity_stall(world.timestamp_s):
            out.append("velocity_stall")
        if self._detect_lateral_oscillation(world.timestamp_s):
            out.append("lateral_oscillation")
        if self._detect_planner_degeneracy():
            out.append("planner_degeneracy")
        return out

    # ── Detection methods ─────────────────────────────────────────────────────

    def _detect_velocity_stall(self, now_s: float) -> bool:
        stall_start = None
        for obs in self._buffer:
            if not obs.corridor_clear:
                stall_start = None
                continue
            if obs.ego_v_mps < self.STALL_V_THRESHOLD_MPS:
                if stall_start is None:
                    stall_start = obs.timestamp_s
                elif (now_s - stall_start) >= self.STALL_MIN_DURATION_S:
                    return True
            else:
                stall_start = None
        return False

    def _detect_lateral_oscillation(self, now_s: float) -> bool:
        # Count oscillation events in the last OSC_WINDOW_S
        recent = [t for t in self._osc_events if (now_s - t) <= self.OSC_WINDOW_S]
        return len(recent) >= self.OSC_COUNT_THRESHOLD

    def _detect_planner_degeneracy(self) -> bool:
        if not self._buffer:
            return False
        latest = self._buffer[-1]
        return (
            latest.mpc_cost_normalized > self.DEGEN_COST_RATIO
            or not latest.mpc_converged
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prune(self, now_s: float) -> None:
        cutoff = now_s - self._window_s
        while self._buffer and self._buffer[0].timestamp_s < cutoff:
            self._buffer.popleft()
        while self._osc_events and self._osc_events[0] < cutoff:
            self._osc_events.popleft()

    def _track_oscillation(self, world: WorldState) -> None:
        sign = 1 if world.ego_lateral_error_m >= 0 else -1
        if sign != self._last_lateral_sign and self._last_lateral_sign != 0:
            self._osc_events.append(world.timestamp_s)
        self._last_lateral_sign = sign
