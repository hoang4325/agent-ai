"""
Stage 10 — Real Stage 9 Component Adapters
============================================
Bridges the real implementation classes from previous stages to the
Protocol interfaces required by the Stage 9 AuthorityArbiter.

Adapters defined here:
  - RealMPCAdapter       : wraps kinematic_mpc_shadow → MPCOptimizerProto
  - RealBaselineAdapter  : wraps stage9 safety-aware baseline → BaselinePlannerProto
  - RealAgentAdapter     : wraps benchmark.agent_shadow_adapter → Stage9AgentAdapterProto
"""
from __future__ import annotations

import math
import logging
import sys
import os
from typing import Optional, Any

LOGGER = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _speed_mps_from_world(world) -> float:
    """Extract ego speed from WorldState."""
    try:
        return float(getattr(world, "ego_v_mps", 0.0))
    except Exception:
        return 0.0


# ── 1. RealMPCAdapter (kinematic_mpc_shadow → MPCOptimizerProto) ──────────────

class RealMPCAdapter:
    """
    Wraps `kinematic_mpc_shadow.run_shadow_mpc_step()` to expose the
    MPCOptimizerProto interface required by AuthorityArbiter.

    The shadow MPC uses an OSQP solver (real QP) for longitudinal and
    lateral control. We translate a TrajectoryRequest → solver inputs,
    run the MPC, and translate the output → ActuatorCommand.
    """

    def __init__(self, *, wheelbase_m: float = 2.7, dt_s: float = 0.1) -> None:
        self._wheelbase = float(wheelbase_m)
        self._dt_s = float(dt_s)
        self._stop_context: Any = None  # will be lazily imported

        # Lazy import — only available in the Docker/Linux environment
        try:
            sys.path.insert(0, str(os.environ.get("BEV_REPO", "/workspace/bevfusion") + "/../.."))
            from benchmark.kinematic_mpc_shadow import (
                _solve_longitudinal_mpc_qp,
                _solve_lateral_mpc_qp,
                _StopContext,
            )
            self._lon_qp = _solve_longitudinal_mpc_qp
            self._lat_qp = _solve_lateral_mpc_qp
            self._stop_context = _StopContext()
            LOGGER.info("RealMPCAdapter: OSQP-backed kinematic MPC loaded.")
        except ImportError as exc:
            LOGGER.warning("RealMPCAdapter: kinematic_mpc_shadow unavailable (%s). Falling back to P-controller.", exc)
            self._lon_qp = None
            self._lat_qp = None

    # ── MPCOptimizerProto ─────────────────────────────────────────────────────

    def execute(self, req) -> Any:
        """Convert TrajectoryRequest → ActuatorCommand via real or fallback MPC."""
        from stage9.schemas import ActuatorCommand

        target_v = float(getattr(req, "target_v_desired_mps", 8.0))
        tactical_intent = str(getattr(req, "tactical_intent", "keep_lane"))
        stop_mode = target_v < 0.5 or tactical_intent in {"safe_stop", "stop", "stop_before_obstacle"}
        lateral_bound = float(getattr(req, "lateral_bound_m", 1.5))

        if self._lon_qp is not None:
            throttle, brake = self._run_real_lon_mpc(target_v, stop_mode)
            steer = self._run_real_lat_mpc(lateral_bound, tactical_intent)
        else:
            throttle, brake, steer = self._p_controller_fallback(target_v, stop_mode)

        return ActuatorCommand(
            steer=_clamp(steer, -1.0, 1.0),
            throttle=_clamp(throttle, 0.0, 1.0),
            brake=_clamp(brake, 0.0, 1.0),
            source="MPC",
        )

    def preview_feasible(self, req) -> bool:
        """Quick sanity check — returns True unless the request is physically impossible."""
        try:
            v_max = float(getattr(req, "v_max_mps", 30.0))
            a_max = float(getattr(req, "a_long_max_mps2", 4.0))
            return v_max >= 0.0 and a_max >= 0.0
        except Exception:
            return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_real_lon_mpc(self, target_v_mps: float, stop_mode: bool) -> tuple[float, float]:
        """Run the OSQP longitudinal MPC and translate first acceleration to throttle/brake."""
        try:
            current_v = 0.0  # will be updated when integrated with sensor live state
            result = self._lon_qp(
                current_speed_mps=current_v,
                target_speed_mps=target_v_mps,
                dt_s=self._dt_s,
                horizon_steps=10,
                stop_distance_m=None,
            )
            accels = result.get("accelerations") or []
            a0 = float(accels[0]) if accels else 0.0
            if a0 >= 0.0:
                return _clamp(a0 / 3.0, 0.0, 1.0), 0.0
            else:
                return 0.0, _clamp(-a0 / 6.0, 0.0, 1.0)
        except Exception as exc:
            LOGGER.debug("lon MPC failed: %s", exc)
            return self._p_controller_fallback(target_v_mps, stop_mode)[:2]

    def _run_real_lat_mpc(self, lateral_bound_m: float, intent: str) -> float:
        """Run the OSQP lateral MPC and return normalised steer for the first step."""
        try:
            # For keep_lane intents, target offset = 0 (track lane centre).
            offset = 0.0
            if "right" in intent:
                offset = -min(lateral_bound_m, 3.5)
            elif "left" in intent:
                offset = min(lateral_bound_m, 3.5)

            result = self._lat_qp(
                target_offset_m=offset,
                dt_s=self._dt_s,
                horizon_steps=10,
            )
            controls = result.get("controls") or []
            u0 = float(controls[0]) if controls else 0.0
            # u0 is lateral jerk; normalise to [-1, 1] steer using wheelbase
            steer = _clamp(u0 / (self._wheelbase * 4.0), -1.0, 1.0)
            return steer
        except Exception as exc:
            LOGGER.debug("lat MPC failed: %s", exc)
            return 0.0

    def _p_controller_fallback(
        self, target_v: float, stop_mode: bool
    ) -> tuple[float, float, float]:
        """Simple proportional fallback when OSQP is unavailable."""
        current_v = 0.0
        err = target_v - current_v
        if stop_mode or err < -0.2:
            throttle = 0.0
            brake = _clamp(-err / max(target_v + 0.1, 0.1), 0.0, 1.0)
        else:
            throttle = _clamp(err / max(target_v + 0.1, 4.0), 0.0, 0.6)
            brake = 0.0
        return throttle, brake, 0.0


# ── 2. RealBaselineAdapter (WorldState → BaselinePlannerProto) ────────────────

class RealBaselineAdapter:
    """
    Translates WorldState → TrajectoryRequest using simple safety-aware rules.

    In Live Mode (CARLA active), the LocalPlannerBridge handles the actual
    waypoint tracking via carla.agents. Here we produce a TrajectoryRequest
    that MPC uses, consistent with what the Arbiter expects.
    """

    DEFAULT_CRUISE_SPEED_MPS = 8.0   # 30 km/h

    def plan(self, world) -> Any:
        from stage9.schemas import TrajectoryRequest, DrivableEnvelope
        intent = self._derive_intent(world)
        target_v = self._derive_speed(world, intent)
        return TrajectoryRequest(
            source="BASELINE",
            tactical_intent=intent,
            target_v_desired_mps=target_v,
            v_max_mps=min(target_v * 1.3, 15.0),
            a_long_max_mps2=2.5,
            a_lat_max_mps2=1.5,
            jerk_max_mps3=3.0,
            lateral_bound_m=0.75,
            horizon_s=3.0,
            cost_profile="BASELINE",
        )

    def degraded_hold(self, world) -> Any:
        from stage9.schemas import TrajectoryRequest
        return TrajectoryRequest(
            source="BASELINE",
            tactical_intent="keep_lane",
            target_v_desired_mps=max(float(getattr(world, "ego_v_mps", 0.0)) * 0.7, 0.0),
            v_max_mps=5.0,
            a_long_max_mps2=1.5,
            a_lat_max_mps2=1.0,
            jerk_max_mps3=2.0,
            lateral_bound_m=0.5,
            horizon_s=3.0,
            cost_profile="BASELINE",
        )

    def is_healthy(self, world) -> bool:
        try:
            from stage9.schemas import SensorHealth
            return getattr(world, "sensor_health", SensorHealth.OK) != SensorHealth.FAULT
        except Exception:
            return True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _derive_intent(self, world) -> str:
        """Derive baseline tactical intent from WorldState fields."""
        try:
            min_ttc = float(getattr(world, "min_ttc_s", 10.0))
            corridor_clear = bool(getattr(world, "corridor_clear", True))
            ego_v = float(getattr(world, "ego_v_mps", 0.0))

            if min_ttc < 1.5 or not corridor_clear:
                return "stop_before_obstacle"
            if min_ttc < 4.0:
                return "follow"
            return "keep_lane"
        except Exception:
            return "keep_lane"

    def _derive_speed(self, world, intent: str) -> float:
        """Choose target speed based on intent and TTC."""
        try:
            min_ttc = float(getattr(world, "min_ttc_s", 10.0))
            ego_v = float(getattr(world, "ego_v_mps", 0.0))
            if intent == "stop_before_obstacle":
                return 0.0
            if intent == "follow":
                # Adaptive follow: slow proportionally to TTC
                follow_v = _clamp(ego_v * (min_ttc / 5.0), 0.0, self.DEFAULT_CRUISE_SPEED_MPS)
                return follow_v
            return self.DEFAULT_CRUISE_SPEED_MPS
        except Exception:
            return self.DEFAULT_CRUISE_SPEED_MPS


# ── 3. RealAgentAdapter (agent_shadow_adapter → Stage9AgentAdapterProto) ──────

class RealAgentAdapter:
    """
    Wraps AgentShadowAdapter to implement Stage9AgentAdapterProto.

    The AgentShadowAdapter works in shadow mode — it proposes a ManeuverContract
    based on current WorldState without directly issuing actuator commands.
    """

    def __init__(self, mode: str = "stub") -> None:
        try:
            # Add Agent-AI to path so benchmark package is importable
            _root = str(os.environ.get("AGENT_AI_ROOT", "/workspace/Agent-AI"))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from benchmark.agent_shadow_adapter import AgentShadowAdapter, AgentShadowAdapterConfig
            self._adapter = AgentShadowAdapter(
                config=AgentShadowAdapterConfig(mode=mode, model_id=f"stage10_{mode}")
            )
            LOGGER.info("RealAgentAdapter: AgentShadowAdapter loaded in mode=%s", mode)
        except ImportError as exc:
            LOGGER.warning("RealAgentAdapter: AgentShadowAdapter unavailable (%s). Using fallback.", exc)
            self._adapter = None

    # ── Stage9AgentAdapterProto ───────────────────────────────────────────────

    def propose_contract(self, world) -> Optional[Any]:
        """
        Observe world state, ask Agent for an intent, and pack it into
        a ManeuverContract if the Agent proposes something different from baseline.
        """
        if self._adapter is None:
            return None

        try:
            intent_record = self._adapter.call(
                case_id="stage10_live",
                frame_id=int(getattr(world, "frame_id", 0)),
                ego_state=self._world_to_ego_state(world),
                tracked_objects=[],
                lane_context=self._world_to_lane_context(world),
                route_context={"route_option": "straight", "preferred_lane": "current"},
                stop_context={},
                baseline_context=self._world_to_baseline_context(world),
            )

            # Only emit a contract if Agent disagrees with baseline in a useful way
            if intent_record.fallback_to_baseline or not intent_record.disagreement_useful:
                return None

            return self._pack_contract(intent_record, world)

        except Exception as exc:
            LOGGER.debug("propose_contract error: %s", exc)
            return None

    def get_intent(self, world, contract) -> str:
        """Return the tactical intent string from the active contract."""
        try:
            return str(getattr(contract, "tactical_intent", "keep_lane"))
        except Exception:
            return "keep_lane"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _world_to_ego_state(self, world) -> dict:
        return {
            "speed_mps": float(getattr(world, "ego_v_mps", 0.0)),
            "accel_mps2": float(getattr(world, "ego_a_mps2", 0.0)),
            "lateral_error_m": float(getattr(world, "ego_lateral_error_m", 0.0)),
            "scene": {
                "front_free_space_m": float(getattr(world, "drivable_envelope", None) and
                                            getattr(world.drivable_envelope, "forward_clear_m", 50.0) or 50.0),
            },
            "risk_summary": {
                "highest_risk_level": "high" if float(getattr(world, "min_ttc_s", 10.0)) < 3.0 else "none",
            },
        }

    def _world_to_lane_context(self, world) -> dict:
        return {
            "current_lane_id": str(getattr(world, "ego_lane_id", "unknown")),
            "lane_change_permission": {
                "left": bool(getattr(world, "lane_change_permission", False)),
                "right": bool(getattr(world, "lane_change_permission", False)),
            },
        }

    def _world_to_baseline_context(self, world) -> dict:
        min_ttc = float(getattr(world, "min_ttc_s", 10.0))
        if min_ttc < 1.5:
            baseline = "stop_before_obstacle"
        elif min_ttc < 4.0:
            baseline = "follow"
        else:
            baseline = "keep_lane"
        return {
            "requested_behavior": baseline,
            "target_lane": "current",
            "lane_change_permission": {
                "left": bool(getattr(world, "lane_change_permission", False)),
                "right": bool(getattr(world, "lane_change_permission", False)),
            },
        }

    def _pack_contract(self, intent_record, world) -> Any:
        """Turn an AgentShadowIntent into a ManeuverContract."""
        try:
            from stage9.schemas import ManeuverContract
            return ManeuverContract(
                issued_at_frame=int(getattr(world, "frame_id", 0)),
                source=f"agent_shadow_{intent_record.model_id}",
                tactical_intent=intent_record.tactical_intent,
                max_duration_s=5.0,
                max_speed_mps=8.33,
                agent_confidence=float(intent_record.confidence),
                agent_reasoning_summary=", ".join(intent_record.reason_tags or []),
            )
        except Exception as exc:
            LOGGER.debug("pack_contract failed: %s", exc)
            return None
