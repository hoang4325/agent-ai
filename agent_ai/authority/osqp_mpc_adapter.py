"""
OSQP kinematic MPC adapter for Stage 9 authority (P0-MPC).

Wraps ``agent_ai.benchmark.shadow.kinematic_mpc_shadow`` so the arbiter's
``MPCOptimizerProto`` is a real QP planner, not a throttle stub.

Invariants:
  - Produces only ActuatorCommand (L3); never invents tactical intent.
  - Bounds come exclusively from TrajectoryRequest (+ optional world context).
  - On infeasible solve, degrades to a safe brake-hold command.
"""
from __future__ import annotations

from typing import Any, Optional

from agent_ai.benchmark.shadow.kinematic_mpc_shadow import (
    _build_follow_solution,
    _build_lane_change_solution,
    _build_stop_solution,
    apply_bounds_to_target_speed,
    extract_mpc_bounds,
)

from .schemas import ActuatorCommand, TrajectoryRequest, WorldState


def _intent_to_behavior(intent: str) -> str:
    intent = str(intent or "keep_lane")
    mapping = {
        "keep_lane": "keep_lane",
        "safe_stop": "stop_before_obstacle",
        "prepare_lane_change_left": "prepare_lane_change_left",
        "prepare_lane_change_right": "prepare_lane_change_right",
        "commit_lane_change_left": "commit_lane_change_left",
        "commit_lane_change_right": "commit_lane_change_right",
        "abort_lane_change": "keep_lane",
        "pull_over_right": "prepare_lane_change_right",
        "slow_bypass_left": "prepare_lane_change_left",
        "slow_bypass_right": "prepare_lane_change_right",
    }
    return mapping.get(intent, "keep_lane")


def trajectory_request_to_bounds(req: TrajectoryRequest) -> dict[str, Any]:
    clear = None
    if req.drivable_envelope is not None:
        clear = float(req.drivable_envelope.forward_clear_m)
    return {
        "v_max_mps": float(req.v_max_mps),
        "a_long_max_mps2": float(req.a_long_max_mps2),
        # Symmetric lower accel bound from max (allow full brake up to -a_long_max*1.5)
        "a_long_min_mps2": -abs(float(req.a_long_max_mps2)) * 1.5,
        "a_lat_max_mps2": float(req.a_lat_max_mps2),
        "lateral_bound_m": float(req.lateral_bound_m),
        "forward_clear_m": clear,
        "soft_speed_scale": 1.0,
        "target_v_desired_mps": float(req.target_v_desired_mps),
        "jerk_max_mps3": float(req.jerk_max_mps3),
        "cost_profile": req.cost_profile,
    }


def world_context_bounds(world: WorldState | None) -> dict[str, Any]:
    if world is None:
        return {}
    return {
        "min_ttc_s": float(world.min_ttc_s) if world.min_ttc_s is not None else None,
        # Use corridor clear as soft clear when envelope missing.
        "front_free_space_m": (
            float(world.drivable_envelope.forward_clear_m)
            if world.drivable_envelope is not None
            else None
        ),
    }


def _accel_to_throttle_brake(accel: float, *, a_brake_full: float = -4.0, a_throttle_full: float = 2.0) -> tuple[float, float]:
    if accel >= 0.0:
        throttle = max(0.0, min(1.0, float(accel) / max(a_throttle_full, 1e-3)))
        return throttle, 0.0
    brake = max(0.0, min(1.0, abs(float(accel)) / max(abs(a_brake_full), 1e-3)))
    return 0.0, brake


def _lateral_to_steer(control_u: float | None, offset_error: float, *, u_scale: float = 4.0) -> float:
    if control_u is not None:
        return max(-1.0, min(1.0, float(control_u) / max(u_scale, 1e-3)))
    # Proportional fallback on remaining lateral error.
    return max(-1.0, min(1.0, float(offset_error) * 0.35))


class OSQPMpcAdapter:
    """
    Real OSQP-backed MPC for AuthorityArbiter.

    Usage:
        mpc = OSQPMpcAdapter()
        cmd = mpc.execute(req)                 # uses last ego_v or 0
        cmd = mpc.execute(req, world=world)    # inject TTC / clear from world
    """

    def __init__(self, *, default_dt_s: float = 0.1) -> None:
        self.default_dt_s = float(default_dt_s)
        self._last_plan: dict[str, Any] | None = None
        self._last_ego_v: float = 0.0

    def plan(
        self,
        req: TrajectoryRequest,
        *,
        ego_v_mps: float | None = None,
        world: WorldState | None = None,
        dt_s: float | None = None,
        lateral_error_m: float = 0.0,
        lane_change_progress: float = 0.0,
    ) -> dict[str, Any]:
        ego_v = float(self._last_ego_v if ego_v_mps is None else ego_v_mps)
        if world is not None:
            ego_v = float(world.ego_v_mps)
        self._last_ego_v = ego_v
        dt = float(self.default_dt_s if dt_s is None else dt_s)

        bounds = trajectory_request_to_bounds(req)
        wbounds = world_context_bounds(world)
        # Merge: explicit request clear wins; else world envelope.
        if bounds.get("forward_clear_m") is None and wbounds.get("front_free_space_m") is not None:
            bounds["forward_clear_m"] = wbounds["front_free_space_m"]
        if wbounds.get("min_ttc_s") is not None:
            bounds["min_ttc_s"] = wbounds["min_ttc_s"]

        behavior = _intent_to_behavior(req.tactical_intent)
        target_v = float(req.target_v_desired_mps)
        target_v, _ = apply_bounds_to_target_speed(target_v, bounds)

        request = {
            "requested_behavior": behavior,
            "target_speed_mps": target_v,
            "target_lane": req.target_lane_id,
            "mpc_bounds": bounds,
            "cost_profile": req.cost_profile,
            "v_max_mps": req.v_max_mps,
            "forward_clear_m": bounds.get("forward_clear_m"),
            "min_ttc_s": bounds.get("min_ttc_s"),
        }
        state = {
            "speed_mps": ego_v,
            "target_lane_id": int(req.target_lane_id) if str(req.target_lane_id or "").isdigit() else 1,
            "current_lane_id": 0,
            "distance_to_target_lane_center": float(lateral_error_m),
            "lane_change_progress": float(lane_change_progress),
        }

        if behavior == "stop_before_obstacle" or float(req.v_max_mps) <= 0.05 or float(req.target_v_desired_mps) <= 0.05:
            clear = bounds.get("forward_clear_m")
            stop_d = float(clear) if clear is not None else max(ego_v * 1.2, 4.0)
            solution = _build_stop_solution(
                current_speed_mps=ego_v,
                dt_s=dt,
                stop_distance_m=stop_d,
                bounds=bounds,
            )
            solution["recommended_behavior"] = "stop_before_obstacle"
        elif "lane_change" in behavior:
            solution = _build_lane_change_solution(
                request=request,
                state=state,
                current_speed_mps=ego_v,
                dt_s=dt,
                bounds=bounds,
            )
        else:
            solution = _build_follow_solution(
                request=request,
                current_speed_mps=ego_v,
                dt_s=dt,
                bounds=bounds,
            )

        accels = list(solution.get("accelerations") or [])
        first_accel = float(accels[0]) if accels else (
            -2.5 if not solution.get("feasible") else max(-1.0, min(1.0, target_v - ego_v))
        )
        if not solution.get("feasible"):
            # Safe degrade: firm brake, zero throttle, hold steer.
            first_accel = min(first_accel, -2.5)
            solution = dict(solution)
            solution["fallback_safe_brake"] = True
            notes = list(solution.get("notes") or [])
            notes.append("adapter_safe_brake_on_infeasible")
            solution["notes"] = notes

        lat_u = None
        controls = solution.get("lateral_controls") or solution.get("controls")
        if isinstance(controls, list) and controls:
            lat_u = float(controls[0])
        throttle, brake = _accel_to_throttle_brake(first_accel)
        steer = _lateral_to_steer(lat_u, lateral_error_m)

        command = ActuatorCommand(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake),
            source="MPC",
        )
        plan = {
            "feasible": bool(solution.get("feasible")),
            "degraded": bool(solution.get("degraded") or solution.get("fallback_safe_brake")),
            "solver_status": solution.get("solver_status"),
            "target_speed_mps": solution.get("target_speed_mps"),
            "recommended_behavior": solution.get("recommended_behavior") or behavior,
            "trajectory_points": solution.get("trajectory_points") or [],
            "accelerations": accels,
            "notes": list(solution.get("notes") or []),
            "applied_bounds": bounds,
            "command": command,
            "first_accel_mps2": first_accel,
        }
        self._last_plan = plan
        return plan

    def execute(
        self,
        req: TrajectoryRequest,
        world: Optional[WorldState] = None,
        *,
        ego_v_mps: float | None = None,
    ) -> ActuatorCommand:
        plan = self.plan(req, ego_v_mps=ego_v_mps, world=world)
        return plan["command"]

    def preview_feasible(
        self,
        req: TrajectoryRequest,
        world: Optional[WorldState] = None,
        *,
        ego_v_mps: float | None = None,
    ) -> bool:
        plan = self.plan(req, ego_v_mps=ego_v_mps, world=world)
        # Degraded safe brake still counts as "executable"; only hard solver failure
        # without recovery is infeasible for grant preview.
        if plan.get("feasible"):
            return True
        return bool(plan.get("degraded"))

    @property
    def last_plan(self) -> dict[str, Any] | None:
        return self._last_plan
