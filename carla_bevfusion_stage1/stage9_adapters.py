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
        current_v = float(getattr(req, "current_speed_mps", 0.0))
        current_lateral_error = float(getattr(req, "current_lateral_error_m", 0.0))

        if self._lon_qp is not None:
            throttle, brake = self._run_real_lon_mpc(target_v, stop_mode, current_v)
            steer = self._run_real_lat_mpc(lateral_bound, tactical_intent, current_lateral_error)
            source = "MPC_OSQP"
        else:
            throttle, brake, steer = self._p_controller_fallback(target_v, stop_mode, current_v)
            source = "MPC_FALLBACK"

        steer = self._enforce_lane_change_steer_floor(
            steer=steer,
            intent=tactical_intent,
            current_lateral_error_m=current_lateral_error,
        )
        throttle, brake = self._cap_lane_change_longitudinal(
            throttle=throttle,
            brake=brake,
            intent=tactical_intent,
        )

        return ActuatorCommand(
            steer=_clamp(steer, -1.0, 1.0),
            throttle=_clamp(throttle, 0.0, 1.0),
            brake=_clamp(brake, 0.0, 1.0),
            source=source,
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

    def _run_real_lon_mpc(
        self,
        target_v_mps: float,
        stop_mode: bool,
        current_v_mps: float,
    ) -> tuple[float, float]:
        """Run the OSQP longitudinal MPC and translate first acceleration to throttle/brake."""
        try:
            result = self._lon_qp(
                current_speed_mps=max(0.0, float(current_v_mps)),
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
            return self._p_controller_fallback(target_v_mps, stop_mode, current_v_mps)[:2]

    def _run_real_lat_mpc(
        self,
        lateral_bound_m: float,
        intent: str,
        current_lateral_error_m: float = 0.0,
    ) -> float:
        """Run the OSQP lateral MPC and return normalised steer for the first step."""
        try:
            # For keep_lane intents, target offset = 0 (track lane centre).
            offset = 0.0
            if "right" in intent:
                offset = min(lateral_bound_m, 3.5)
            elif "left" in intent:
                offset = -min(lateral_bound_m, 3.5)

            result = self._lat_qp(
                target_offset_m=offset,
                dt_s=self._dt_s,
                horizon_steps=10,
            )
            controls = result.get("controls") or []
            u0 = float(controls[0]) if controls else 0.0
            # u0 is lateral jerk; normalise to [-1, 1] steer using wheelbase
            steer = _clamp(u0 / (self._wheelbase * 4.0), -1.0, 1.0)
            steer = self._enforce_lane_change_steer_floor(
                steer=steer,
                intent=intent,
                current_lateral_error_m=current_lateral_error_m,
            )
            return steer
        except Exception as exc:
            LOGGER.debug("lat MPC failed: %s", exc)
            return self._enforce_lane_change_steer_floor(
                steer=0.0,
                intent=intent,
                current_lateral_error_m=current_lateral_error_m,
            )

    def _p_controller_fallback(
        self,
        target_v: float,
        stop_mode: bool,
        current_v: float = 0.0,
    ) -> tuple[float, float, float]:
        """Simple proportional fallback when OSQP is unavailable."""
        err = target_v - current_v
        if stop_mode or err < -0.2:
            throttle = 0.0
            brake = _clamp(-err / max(target_v + 0.1, 0.1), 0.0, 1.0)
        else:
            throttle = _clamp(err / max(target_v + 0.1, 4.0), 0.0, 0.6)
            brake = 0.0
        return throttle, brake, 0.0

    def _enforce_lane_change_steer_floor(
        self,
        *,
        steer: float,
        intent: str,
        current_lateral_error_m: float,
    ) -> float:
        if "right" in intent:
            direction = 1.0
        elif "left" in intent:
            direction = -1.0
        else:
            return _clamp(steer, -1.0, 1.0)

        error_mag = abs(float(current_lateral_error_m))
        if intent.startswith("prepare_lane_change_"):
            floor_mag = 0.18 if error_mag < 0.35 else 0.14 if error_mag < 0.9 else 0.10
        else:
            floor_mag = 0.30 if error_mag < 0.35 else 0.24 if error_mag < 1.1 else 0.16

        signed_floor = direction * floor_mag
        if abs(steer) < floor_mag:
            steer = signed_floor
        return _clamp(steer, -1.0, 1.0)

    def _cap_lane_change_longitudinal(
        self,
        *,
        throttle: float,
        brake: float,
        intent: str,
    ) -> tuple[float, float]:
        if intent.startswith("prepare_lane_change_"):
            return min(float(throttle), 0.22), min(float(brake), 0.05)
        if intent.startswith("commit_lane_change_"):
            return min(float(throttle), 0.32), min(float(brake), 0.05)
        return float(throttle), float(brake)


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

    def __init__(
        self,
        mode: str = "stub",
        *,
        api_timeout_s: float | None = None,
        api_max_retries: int | None = None,
    ) -> None:
        try:
            # Add Agent-AI to path so benchmark package is importable
            _root = str(os.environ.get("AGENT_AI_ROOT", "/workspace/Agent-AI"))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from benchmark.agent_shadow_adapter import AgentShadowAdapter, AgentShadowAdapterConfig
            resolved_model_id = str(os.environ.get("AGENT_MODEL_ID", f"stage10_{mode}"))
            resolved_timeout_s = float(
                api_timeout_s
                if api_timeout_s is not None
                else os.environ.get("AGENT_API_TIMEOUT_S", "3.0")
            )
            resolved_max_retries = int(
                api_max_retries
                if api_max_retries is not None
                else os.environ.get("AGENT_API_MAX_RETRIES", "0")
            )
            if resolved_timeout_s <= 0.0:
                raise ValueError("api_timeout_s must be positive")
            if resolved_max_retries < 0:
                raise ValueError("api_max_retries must be non-negative")
            self._adapter = AgentShadowAdapter(
                config=AgentShadowAdapterConfig(
                    mode=mode,
                    model_id=resolved_model_id,
                    api_timeout_s=resolved_timeout_s,
                    api_max_retries=resolved_max_retries,
                )
            )
            LOGGER.info(
                "RealAgentAdapter: AgentShadowAdapter loaded in mode=%s model_id=%s "
                "timeout_s=%.1f max_retries=%d",
                mode,
                resolved_model_id,
                resolved_timeout_s,
                resolved_max_retries,
            )
        except ImportError as exc:
            LOGGER.warning("RealAgentAdapter: AgentShadowAdapter unavailable (%s). Using fallback.", exc)
            self._adapter = None

    # ── Stage9AgentAdapterProto ───────────────────────────────────────────────

    def observe_intent(self, world) -> Optional[Any]:
        """Return the full shadow intent record for evaluation/logging."""
        return self.observe_intent_request(self.build_intent_request(world))

    def build_intent_request(
        self,
        world,
        *,
        baseline_intent: str | None = None,
        detections: Optional[list[Any]] = None,
        sensor_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Freeze WorldState into primitive values safe for a worker thread."""
        resolved_baseline = self._world_to_baseline_context(
            world,
            baseline_intent=baseline_intent,
        )
        return {
            "case_id": "stage10_live",
            "frame_id": int(getattr(world, "frame_id", 0)),
            "ego_state": self._world_to_ego_state(world, sensor_input=sensor_input),
            "tracked_objects": self._detections_to_tracked_objects(
                detections or [],
                ego_v_mps=float(getattr(world, "ego_v_mps", 0.0)),
            ),
            "lane_context": self._world_to_lane_context(world),
            "route_context": self._world_to_route_context(world),
            "stop_context": self._world_to_stop_context(
                world,
                baseline_intent=str(resolved_baseline["requested_behavior"]),
            ),
            "baseline_context": resolved_baseline,
        }

    def observe_intent_request(self, request: dict[str, Any]) -> Optional[Any]:
        """Execute a previously frozen request (normally on the async worker)."""
        if self._adapter is None:
            return None

        try:
            return self._adapter.call(**request)
        except Exception as exc:
            LOGGER.debug("observe_intent_request error: %s", exc)
            return None

    def propose_contract(self, world) -> Optional[Any]:
        """
        Observe world state, ask Agent for an intent, and pack it into
        a ManeuverContract if the Agent proposes something different from baseline.
        """
        intent_record = self.observe_intent(world)
        if intent_record is None:
            return None

        # Only emit a contract if Agent disagrees with baseline in a useful way.
        if intent_record.fallback_to_baseline or not intent_record.disagreement_useful:
            return None

        return self._pack_contract(intent_record, world)

    def get_intent(self, world, contract) -> str:
        """Return the tactical intent string from the active contract."""
        try:
            return str(getattr(contract, "tactical_intent", "keep_lane"))
        except Exception:
            return "keep_lane"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _enum_text(value: Any) -> str:
        return str(getattr(value, "value", value) or "unknown")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return float(default)
        return number if math.isfinite(number) else float(default)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return int(default)

    def _world_to_ego_state(
        self,
        world,
        *,
        sensor_input: Optional[dict[str, Any]] = None,
    ) -> dict:
        sensor = dict(sensor_input or {})
        min_ttc_s = self._safe_float(getattr(world, "min_ttc_s", 99.0), 99.0)
        perception = {
            "sensor_health": self._enum_text(getattr(world, "sensor_health", "unknown")),
            "sync_ok": bool(getattr(world, "sync_ok", False)),
            "world_age_ms": max(0, self._safe_int(getattr(world, "world_age_ms", 0))),
            "new_obstacle_score": self._safe_float(
                getattr(world, "new_obstacle_score", 0.0),
                0.0,
            ),
            "odd_status": self._enum_text(getattr(world, "odd_status", "unknown")),
            "time_to_odd_exit_s": getattr(world, "time_to_odd_exit_s", None),
            "preview_feasible": bool(getattr(world, "preview_feasible", False)),
            "weather_visibility_m": self._safe_float(
                getattr(world, "weather_visibility_m", 0.0),
                0.0,
            ),
            "inference_time_ms": self._safe_float(sensor.get("inference_time_ms"), 0.0),
            "num_detections": max(0, self._safe_int(sensor.get("num_detections", 0))),
            "num_raw_boxes": max(0, self._safe_int(sensor.get("num_raw_boxes", 0))),
            "lidar_point_count": max(0, self._safe_int(sensor.get("lidar_point_count", 0))),
            "radar_point_count": max(0, self._safe_int(sensor.get("radar_point_count", 0))),
        }
        return {
            "speed_mps": float(getattr(world, "ego_v_mps", 0.0)),
            "accel_mps2": float(getattr(world, "ego_a_mps2", 0.0)),
            "lateral_error_m": float(getattr(world, "ego_lateral_error_m", 0.0)),
            "ego_lane_id": str(getattr(world, "ego_lane_id", "unknown")),
            "min_ttc_s": min_ttc_s,
            "new_obstacle_score": perception["new_obstacle_score"],
            "scene": {
                "front_free_space_m": float(getattr(world, "drivable_envelope", None) and
                                            getattr(world.drivable_envelope, "forward_clear_m", 50.0) or 50.0),
            },
            "risk_summary": {
                "highest_risk_level": (
                    "critical" if min_ttc_s < 1.5
                    else "high" if min_ttc_s < 3.0
                    else "medium" if min_ttc_s < 5.0
                    else "none"
                ),
                "minimum_ttc_seconds": min_ttc_s,
            },
            "perception": perception,
        }

    def _world_to_lane_context(self, world) -> dict:
        lane_change_permission = self._world_lane_change_permissions(world)
        envelope = getattr(world, "drivable_envelope", None)
        return {
            "current_lane_id": str(getattr(world, "ego_lane_id", "unknown")),
            "lane_change_permission": lane_change_permission,
            "lane_change_rule": str(getattr(world, "lane_change_rule", "unknown")),
            "origin_lane_id": str(getattr(world, "agent_origin_lane_id", "") or ""),
            "target_lane_id": str(getattr(world, "agent_target_lane_id", "") or ""),
            "corridor_clear": bool(getattr(world, "corridor_clear", False)),
            "target_lane_risk_available": bool(
                getattr(world, "target_lane_risk_available", False)
            ),
            "target_lane_corridor_clear": bool(
                getattr(world, "target_lane_corridor_clear", False)
            ),
            "target_lane_forward_ttc_s": self._safe_float(
                getattr(world, "target_lane_forward_ttc_s", None),
                99.0,
            ),
            "target_lane_rear_clearance_m": self._safe_float(
                getattr(world, "target_lane_rear_clearance_m", None),
                0.0,
            ),
            "target_lane_corridor_object_count": self._safe_int(
                getattr(world, "target_lane_corridor_object_count", 0),
                0,
            ),
            "target_lane_lateral_offset_m": self._safe_float(
                getattr(world, "target_lane_lateral_offset_m", None),
                0.0,
            ),
            "target_lane_risk_source": str(
                getattr(world, "target_lane_risk_source", "unavailable")
            ),
            "drivable_envelope": {
                "left_bound_m": self._safe_float(getattr(envelope, "left_bound_m", None), 0.0),
                "right_bound_m": self._safe_float(getattr(envelope, "right_bound_m", None), 0.0),
                "forward_clear_m": self._safe_float(getattr(envelope, "forward_clear_m", None), 0.0),
            },
        }

    def _detections_to_tracked_objects(
        self,
        detections: list[Any],
        *,
        ego_v_mps: float,
        max_objects: int = 12,
    ) -> list[dict[str, Any]]:
        """Freeze nearest BEVFusion boxes into a bounded primitive snapshot."""
        objects: list[dict[str, Any]] = []
        ego_speed_mps = max(0.0, self._safe_float(ego_v_mps, 0.0))
        for index, detection in enumerate(detections):
            x = self._safe_float(getattr(detection, "x", None), 0.0)
            y = self._safe_float(getattr(detection, "y", None), 0.0)
            z = self._safe_float(getattr(detection, "z", None), 0.0)
            distance_m = math.hypot(x, y)
            ttc_s = (
                max(0.1, distance_m - 1.0) / max(0.5, ego_speed_mps)
                if x >= 0.5
                else None
            )
            objects.append(
                {
                    "detection_id": f"bev_{index}",
                    "label_idx": self._safe_int(getattr(detection, "label_idx", -1), -1),
                    "label_name": str(getattr(detection, "label_name", "") or "unknown")[:64],
                    "score": round(
                        _clamp(self._safe_float(getattr(detection, "score", 0.0), 0.0), 0.0, 1.0),
                        4,
                    ),
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "z": round(z, 3),
                    "dx": round(self._safe_float(getattr(detection, "dx", None), 0.0), 3),
                    "dy": round(self._safe_float(getattr(detection, "dy", None), 0.0), 3),
                    "dz": round(self._safe_float(getattr(detection, "dz", None), 0.0), 3),
                    "yaw_rad": round(
                        self._safe_float(getattr(detection, "yaw_rad", None), 0.0),
                        3,
                    ),
                    "distance_m": round(distance_m, 3),
                    "ttc_s": round(ttc_s, 3) if ttc_s is not None else None,
                    "relative_sector": (
                        "rear" if x < 0.0
                        else "front_left" if y > 1.75
                        else "front_right" if y < -1.75
                        else "front"
                    ),
                }
            )
        objects.sort(
            key=lambda obj: (
                obj["ttc_s"] if obj["ttc_s"] is not None else 1e9,
                obj["distance_m"],
            )
        )
        return objects[:max(0, int(max_objects))]

    def _world_to_stop_context(self, world, *, baseline_intent: str) -> dict[str, Any]:
        envelope = getattr(world, "drivable_envelope", None)
        obstacle_distance_m = self._safe_float(
            getattr(envelope, "forward_clear_m", None),
            50.0,
        )
        stop_active = baseline_intent in {"stop", "stop_before_obstacle", "follow"}
        return {
            "binding_status": "derived_active" if stop_active else "inactive",
            "distance_to_stop_m": obstacle_distance_m if stop_active else None,
            "distance_to_obstacle_m": obstacle_distance_m,
            "minimum_ttc_s": self._safe_float(getattr(world, "min_ttc_s", 99.0), 99.0),
            "reason": "bevfusion_forward_obstacle" if stop_active else "no_active_stop_target",
            "source": "stage10_bevfusion_world_state",
        }

    def _world_to_route_context(self, world) -> dict:
        preferred_lane = str(getattr(world, "agent_preferred_lane", "current") or "current")
        return {
            "route_option": "straight",
            "preferred_lane": preferred_lane,
            "route_mode": "stage10_live",
            "route_conflict_flags": list(getattr(world, "route_conflict_flags", []) or []),
        }

    def _world_lane_change_permissions(self, world) -> dict[str, bool]:
        preferred_lane = str(getattr(world, "agent_preferred_lane", "current") or "current")
        permitted = bool(getattr(world, "lane_change_permission", False))
        return {
            "left": permitted and preferred_lane == "left",
            "right": permitted and preferred_lane == "right",
        }

    def _world_to_baseline_context(
        self,
        world,
        *,
        baseline_intent: str | None = None,
    ) -> dict:
        min_ttc = float(getattr(world, "min_ttc_s", 10.0))
        corridor_clear = bool(getattr(world, "corridor_clear", True))
        if baseline_intent is not None:
            baseline = str(baseline_intent)
        elif min_ttc < 1.5 or not corridor_clear:
            baseline = "stop_before_obstacle"
        elif min_ttc < 4.0:
            baseline = "follow"
        else:
            baseline = "keep_lane"
        preferred_lane = str(getattr(world, "agent_preferred_lane", "current") or "current")
        route_conflicts = list(getattr(world, "route_conflict_flags", []) or [])
        lane_change_permission = self._world_lane_change_permissions(world)
        preferred_lane_permitted = bool(lane_change_permission.get(preferred_lane, False))
        blocked_clear_event = bool(
            "blocked_clear_adjacent_lane" in route_conflicts
            and preferred_lane in {"left", "right"}
            and preferred_lane_permitted
        )
        return {
            "requested_behavior": baseline,
            "target_lane": preferred_lane if preferred_lane in {"left", "right"} else "current",
            "active_maneuver": str(getattr(world, "agent_active_maneuver", "") or ""),
            "lane_change_permission": lane_change_permission,
            "current_lane_blocked": blocked_clear_event,
            "adjacent_preferred_lane_clear": blocked_clear_event,
            "preferred_lane_permission": preferred_lane_permitted,
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
