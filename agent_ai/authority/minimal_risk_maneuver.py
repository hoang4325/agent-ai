"""
Stage 9 — Minimal Risk Maneuver (MRM) Executor (v2)
=====================================================
Plans and executes a safe, controlled stop when autonomy can no longer continue.

Key design principles:
  - MRM ≠ Emergency Brake. MRM is a controlled deceleration, preferably
    into a shoulder / safe corridor.
  - MRM uses TrajectoryRequest passed to MPC. It never issues raw actuator
    commands directly.
  - Three strategies: COAST_TO_STOP, PULL_OVER_RIGHT, EMERGENCY_STOP
  - Strategy is selected based on world state (corridor, speed, TTC).
  - Once SAFE_STOP is reached, MRM holds the standstill request.

SAFE_STOP_SEQUENCE (implemented as strategy hints to MPC):
  1. ACTIVATE_HAZARDS
  2. MAINTAIN_STABLE_YAW
  3. SIGNAL_RIGHT_IF_SHOULDER_CLEAR
  4. DECELERATE_WITH_JERK_LIMIT (max_decel preferred 2.5 m/s², emergency 6.0 m/s²)
  5. STOP_FULLY
  6. SET_SAFE_STOP_FLAG
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import (
    DrivableEnvelope,
    MRCPlan,
    MRMStrategy,
    TrajectoryRequest,
    WorldState,
)


# ── Configuration ─────────────────────────────────────────────────────────────
_COMFORT_DECEL_MPS2   = 2.5     # preferred deceleration
_EMERGENCY_DECEL_MPS2 = 6.0     # only if TTC < 1.0s
_COMFORT_JERK_MPS3    = 3.0
_EMERGENCY_JERK_MPS3  = 6.0

_PULL_OVER_SPEED_MPS  = 3.0     # speed limit during pull-over lateral movement
_STANDSTILL_V_MPS     = 0.15    # below this = "full stop reached"


@dataclass
class _MRMState:
    active: bool = False
    plan: Optional[MRCPlan] = None
    full_stop_reached: bool = False
    total_mrm_frames: int = 0
    strategy_selected: Optional[MRMStrategy] = None


class MRMExecutor:
    """
    Minimal Risk Maneuver (MRM) Executor — Stage 9.

    compute(world) → MRCPlan
    to_trajectory_request() is called by the plan itself via MRCPlan.to_trajectory_request()

    Additional methods:
      is_full_stop_reached(world)   → bool
      get_standstill_request(world) → TrajectoryRequest  (for SAFE_STOP state)
    """

    def __init__(self) -> None:
        self._state = _MRMState()

    # ── Main API ──────────────────────────────────────────────────────────────

    def compute(self, world: WorldState) -> MRCPlan:
        """
        Compute the appropriate MRC plan for current world state.
        Called every frame while in MINIMAL_RISK_MANEUVER.
        """
        self._state.active = True
        self._state.total_mrm_frames += 1

        strategy = self._select_strategy(world)
        self._state.strategy_selected = strategy

        decel, jerk = self._select_decel(world)
        stop_dist, stop_time = self._estimate_stop(world.ego_v_mps, decel)

        corridor = self._find_safe_corridor(world, strategy)

        plan = MRCPlan(
            strategy=strategy,
            target_v_final=0.0,
            max_decel_mps2=decel,
            max_jerk_mps3=jerk,
            prefer_shoulder=(strategy == MRMStrategy.PULL_OVER_RIGHT),
            hazard_lights=True,
            keep_lane_centering=(strategy == MRMStrategy.COAST_TO_STOP),
            estimated_stop_distance_m=stop_dist,
            estimated_stop_time_s=stop_time,
            corridor_for_stop=corridor,
        )
        self._state.plan = plan
        return plan

    def is_full_stop_reached(self, world: WorldState) -> bool:
        """Returns True once the vehicle has come to a full stop."""
        if world.ego_v_mps < _STANDSTILL_V_MPS:
            self._state.full_stop_reached = True
        return self._state.full_stop_reached

    def get_standstill_request(self, world: WorldState) -> TrajectoryRequest:
        """
        TrajectoryRequest to hold the vehicle stationary (for SAFE_STOP state).
        MPC receives this and outputs zero throttle + sufficient brake.
        """
        return TrajectoryRequest(
            source="MRM",
            tactical_intent="safe_stop",
            v_max_mps=0.0,
            a_long_max_mps2=_COMFORT_DECEL_MPS2,
            a_lat_max_mps2=0.5,
            jerk_max_mps3=_COMFORT_JERK_MPS3,
            lateral_bound_m=0.3,
            drivable_envelope=world.drivable_envelope,
            target_lane_id=world.ego_lane_id,
            target_v_desired_mps=0.0,
            horizon_s=2.0,
            cost_profile="MRM",
        )

    def reset(self) -> None:
        self._state = _MRMState()

    # ── Strategy selection ────────────────────────────────────────────────────

    def _select_strategy(self, world: WorldState) -> MRMStrategy:
        """
        Hierarchy:
          1. EMERGENCY_STOP if TTC is critically low (< 1.0s)
          2. PULL_OVER_RIGHT if lane_change_permission available and shoulder exists
          3. COAST_TO_STOP (default — stay in lane, comfort decel)
        """
        if world.min_ttc_s < 1.0:
            return MRMStrategy.EMERGENCY_STOP

        shoulder_possible = (
            world.lane_change_permission
            and world.drivable_envelope.right_bound_m >= 1.5
        )
        if shoulder_possible:
            return MRMStrategy.PULL_OVER_RIGHT

        return MRMStrategy.COAST_TO_STOP

    def _select_decel(self, world: WorldState) -> tuple[float, float]:
        if world.min_ttc_s < 1.0:
            return _EMERGENCY_DECEL_MPS2, _EMERGENCY_JERK_MPS3
        return _COMFORT_DECEL_MPS2, _COMFORT_JERK_MPS3

    def _estimate_stop(self, v_mps: float, decel_mps2: float) -> tuple[float, float]:
        """d = v² / (2a),  t = v / a"""
        if decel_mps2 <= 0:
            return 999.9, 999.9
        stop_dist = (v_mps ** 2) / (2 * decel_mps2)
        stop_time = v_mps / decel_mps2
        return round(stop_dist, 2), round(stop_time, 2)

    def _find_safe_corridor(
        self,
        world: WorldState,
        strategy: MRMStrategy,
    ) -> Optional[str]:
        if strategy == MRMStrategy.PULL_OVER_RIGHT:
            return f"shoulder_right_of_{world.ego_lane_id}"
        return world.ego_lane_id

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @property
    def total_mrm_frames(self) -> int:
        return self._state.total_mrm_frames

    @property
    def last_strategy(self) -> Optional[MRMStrategy]:
        return self._state.strategy_selected
