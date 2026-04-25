from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from math import copysign
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
import osqp
import scipy.sparse as sp


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean_or_none(values: list[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(mean(cleaned))


DIRECT_PROPOSAL_QUALITY_METRICS = {
    "shadow_lateral_oscillation_rate",
    "shadow_trajectory_smoothness_proxy",
    "shadow_control_jerk_proxy",
}

COUNTERFACTUAL_PROPOSAL_QUALITY_METRICS = {
    "shadow_lane_change_completion_time_s",
    "shadow_heading_settle_time_s",
}

HEURISTIC_STOP_TARGET_SUPPORT_LEVELS = {
    "runtime_distance",
    "runtime_obstacle_distance",
    "stop_event_first_distance",
    "stop_event_initial_distance",
    "stop_event_target_distance",
    "active_stop_context",
}

EXACT_STOP_BINDING_STATUSES = {"exact"}
HEURISTIC_STOP_BINDING_STATUSES = {"heuristic", "approximate", "bound"}


DEFAULT_SHADOW_MPC_CONFIG: dict[str, Any] = {
    "config_id": "baseline_v1",
    "longitudinal": {
        "follow": {
            "q_speed": 0.9,
            "q_speed_terminal": 2.2,
            "r_accel": 0.18,
            "r_accel_delta": 0.3,
            "horizon_steps": 10,
            "a_min": -3.5,
            "a_max": 2.0,
            "v_max_floor_mps": 8.0,
            "v_max_buffer_mps": 4.0,
            "p_upper_scale": 1.8,
            "p_upper_bias_m": 6.0,
        },
        "stop": {
            "q_speed": 0.45,
            "q_position": 0.12,
            "q_speed_terminal": 5.5,
            "q_position_terminal": 18.0,
            "r_accel": 0.18,
            "r_accel_delta": 0.3,
            "horizon_steps": 12,
            "a_min": -6.0,
            "a_max": 1.5,
            "hold_speed_threshold_mps": 0.35,
            "near_zero_distance_threshold_m": 0.05,
        },
    },
    "lateral": {
        "q_offset": 1.1,
        "q_offset_terminal": 8.0,
        "q_velocity": 0.15,
        "r_control": 0.12,
        "r_control_delta": 0.25,
        "y_limit_scale": 1.5,
        "y_limit_floor_m": 3.5,
        "vy_limit_mps": 3.0,
        "u_limit": 4.0,
    },
    "lane_change": {
        "prepare_commit_progress_threshold": 0.05,
        "prepare_commit_lateral_error_threshold_m": 2.5,
        "target_speed_cap_high_error_mps": 4.5,
        "target_speed_cap_low_error_mps": 6.0,
        "high_error_threshold_m": 2.0,
        "nominal_time_base_s": 0.8,
        "nominal_time_speed_factor": 0.35,
        "nominal_time_speed_bias": 0.8,
        "nominal_time_min_s": 1.2,
        "nominal_time_max_s": 4.5,
        "min_horizon_steps": 8,
        "max_horizon_steps": 14,
    },
    "osqp": {
        "warm_start": True,
        "polish": False,
        "adaptive_rho": True,
        "max_iter": 4000,
        "eps_abs": 1e-4,
        "eps_rel": 1e-4,
    },
}


_ACTIVE_SHADOW_MPC_CONFIG: dict[str, Any] | None = None
_ACTIVE_SHADOW_MPC_CONFIG_SOURCE = "builtin_default"
DEFAULT_SHADOW_MPC_CONFIG_OVERRIDE_PATH = Path(__file__).with_name("stage6_mpc_shadow_default_config.json")


def _merge_nested_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _normalized_shadow_mpc_config(override: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = _merge_nested_dict(DEFAULT_SHADOW_MPC_CONFIG, _load_default_shadow_config_override())
    merged = _merge_nested_dict(merged, override or {})
    merged["config_id"] = str(merged.get("config_id") or DEFAULT_SHADOW_MPC_CONFIG["config_id"])
    return merged


def _load_default_shadow_config_override() -> dict[str, Any]:
    path = DEFAULT_SHADOW_MPC_CONFIG_OVERRIDE_PATH
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def reset_shadow_solver_caches() -> None:
    _longitudinal_template.cache_clear()
    _lateral_template.cache_clear()


def set_shadow_tuning_config(config: dict[str, Any] | None, *, source: str = "runtime_override") -> dict[str, Any]:
    global _ACTIVE_SHADOW_MPC_CONFIG
    global _ACTIVE_SHADOW_MPC_CONFIG_SOURCE
    _ACTIVE_SHADOW_MPC_CONFIG = _normalized_shadow_mpc_config(config)
    _ACTIVE_SHADOW_MPC_CONFIG_SOURCE = str(source or "runtime_override")
    reset_shadow_solver_caches()
    return deepcopy(_ACTIVE_SHADOW_MPC_CONFIG)


def reset_shadow_tuning_config() -> dict[str, Any]:
    return set_shadow_tuning_config(None, source="builtin_default")


def get_shadow_tuning_config() -> dict[str, Any]:
    global _ACTIVE_SHADOW_MPC_CONFIG
    if _ACTIVE_SHADOW_MPC_CONFIG is None:
        _ACTIVE_SHADOW_MPC_CONFIG = _normalized_shadow_mpc_config()
    return deepcopy(_ACTIVE_SHADOW_MPC_CONFIG)


def get_shadow_tuning_config_source() -> str:
    global _ACTIVE_SHADOW_MPC_CONFIG_SOURCE
    if _ACTIVE_SHADOW_MPC_CONFIG is None:
        reset_shadow_tuning_config()
    return str(_ACTIVE_SHADOW_MPC_CONFIG_SOURCE)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _smoothstep(alpha: float) -> float:
    alpha = _clamp(alpha, 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _derive_dt(timestamp: float | None, previous_timestamp: float | None) -> float:
    if timestamp is None or previous_timestamp is None:
        return 0.1
    dt = float(timestamp - previous_timestamp)
    if dt <= 0.0:
        return 0.1
    return _clamp(dt, 0.05, 0.25)


def _osqp_status(status: str | None) -> tuple[bool, bool, str]:
    normalized = str(status or "unknown").strip().lower()
    feasible = normalized in {"solved", "solved inaccurate"}
    timed_out = "time limit" in normalized or "maximum iterations" in normalized
    return feasible, timed_out, normalized.replace(" ", "_")


@dataclass
class _LongitudinalTemplate:
    N: int
    dt_s: float
    stop_mode: bool
    v_idx: np.ndarray
    p_idx: np.ndarray
    a_idx: np.ndarray
    v_weights: np.ndarray
    p_weights: np.ndarray | None
    P: sp.csc_matrix
    A: sp.csc_matrix
    solver: osqp.OSQP | None = None


@dataclass
class _LateralTemplate:
    N: int
    dt_s: float
    y_idx: np.ndarray
    vy_idx: np.ndarray
    u_idx: np.ndarray
    y_weights: np.ndarray
    P: sp.csc_matrix
    A: sp.csc_matrix
    solver: osqp.OSQP | None = None


@lru_cache(maxsize=32)
def _longitudinal_template(
    horizon_steps: int,
    dt_s: float,
    stop_mode: bool,
    q_v: float,
    q_p: float,
    q_v_terminal: float,
    q_p_terminal: float,
    r_a: float,
    r_da: float,
) -> _LongitudinalTemplate:
    N = int(max(int(horizon_steps), 6))
    dt_s = round(float(dt_s), 4)
    v_idx = np.arange(N, dtype=int)
    p_idx = np.arange(N, 2 * N, dtype=int)
    a_idx = np.arange(2 * N, 3 * N, dtype=int)

    v_weights = np.full(N, q_v, dtype=float)
    v_weights[-1] = q_v_terminal
    p_weights = None
    if stop_mode:
        p_weights = np.full(N, q_p, dtype=float)
        p_weights[-1] = q_p_terminal

    accel_block = 2.0 * r_a * sp.eye(N, format="csc")
    if N > 1:
        D = sp.diags(
            diagonals=[-np.ones(N - 1), np.ones(N - 1)],
            offsets=[0, 1],
            shape=(N - 1, N),
            format="csc",
        )
        accel_block = accel_block + 2.0 * r_da * (D.T @ D)
    v_block = sp.diags(2.0 * v_weights, format="csc")
    p_block = sp.diags(2.0 * p_weights, format="csc") if p_weights is not None else sp.csc_matrix((N, N))
    P = sp.triu(sp.block_diag([v_block, p_block, accel_block], format="csc")).tocsc()

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row_cursor = 0
    for k in range(N):
        rows.extend([row_cursor, row_cursor])
        cols.extend([int(v_idx[k]), int(a_idx[k])])
        data.extend([1.0, -dt_s])
        if k > 0:
            rows.append(row_cursor)
            cols.append(int(v_idx[k - 1]))
            data.append(-1.0)
        row_cursor += 1
    for k in range(N):
        rows.append(row_cursor)
        cols.append(int(p_idx[k]))
        data.append(1.0)
        if k > 0:
            rows.extend([row_cursor, row_cursor])
            cols.extend([int(p_idx[k - 1]), int(v_idx[k - 1])])
            data.extend([-1.0, -dt_s])
        row_cursor += 1
    for k in range(N):
        rows.append(row_cursor)
        cols.append(int(v_idx[k]))
        data.append(1.0)
        row_cursor += 1
    for k in range(N):
        rows.append(row_cursor)
        cols.append(int(p_idx[k]))
        data.append(1.0)
        row_cursor += 1
    for k in range(N):
        rows.append(row_cursor)
        cols.append(int(a_idx[k]))
        data.append(1.0)
        row_cursor += 1
    A = sp.csc_matrix((data, (rows, cols)), shape=(row_cursor, 3 * N))
    return _LongitudinalTemplate(
        N=N,
        dt_s=dt_s,
        stop_mode=bool(stop_mode),
        v_idx=v_idx,
        p_idx=p_idx,
        a_idx=a_idx,
        v_weights=v_weights,
        p_weights=p_weights,
        P=P,
        A=A,
    )


@lru_cache(maxsize=32)
def _lateral_template(
    horizon_steps: int,
    dt_s: float,
    q_y: float,
    q_y_terminal: float,
    q_vy: float,
    r_u: float,
    r_du: float,
) -> _LateralTemplate:
    N = int(max(int(horizon_steps), 6))
    dt_s = round(float(dt_s), 4)
    y_idx = np.arange(N, dtype=int)
    vy_idx = np.arange(N, 2 * N, dtype=int)
    u_idx = np.arange(2 * N, 3 * N, dtype=int)

    y_weights = np.full(N, q_y, dtype=float)
    y_weights[-1] = q_y_terminal
    y_block = sp.diags(2.0 * y_weights, format="csc")
    vy_block = sp.diags(np.full(N, 2.0 * q_vy, dtype=float), format="csc")
    u_block = 2.0 * r_u * sp.eye(N, format="csc")
    if N > 1:
        D = sp.diags(
            diagonals=[-np.ones(N - 1), np.ones(N - 1)],
            offsets=[0, 1],
            shape=(N - 1, N),
            format="csc",
        )
        u_block = u_block + 2.0 * r_du * (D.T @ D)
    P = sp.triu(sp.block_diag([y_block, vy_block, u_block], format="csc")).tocsc()

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row_cursor = 0
    for k in range(N):
        rows.append(row_cursor)
        cols.append(int(vy_idx[k]))
        data.append(1.0)
        if k > 0:
            rows.extend([row_cursor, row_cursor])
            cols.extend([int(vy_idx[k - 1]), int(u_idx[k])])
            data.extend([-1.0, -dt_s])
        else:
            rows.append(row_cursor)
            cols.append(int(u_idx[k]))
            data.append(-dt_s)
        row_cursor += 1
    for k in range(N):
        rows.append(row_cursor)
        cols.append(int(y_idx[k]))
        data.append(1.0)
        if k > 0:
            rows.extend([row_cursor, row_cursor])
            cols.extend([int(y_idx[k - 1]), int(vy_idx[k])])
            data.extend([-1.0, -dt_s])
        else:
            rows.append(row_cursor)
            cols.append(int(vy_idx[k]))
            data.append(-dt_s)
        row_cursor += 1
    for k in range(N):
        rows.append(row_cursor)
        cols.append(int(y_idx[k]))
        data.append(1.0)
        row_cursor += 1
    for k in range(N):
        rows.append(row_cursor)
        cols.append(int(vy_idx[k]))
        data.append(1.0)
        row_cursor += 1
    for k in range(N):
        rows.append(row_cursor)
        cols.append(int(u_idx[k]))
        data.append(1.0)
        row_cursor += 1
    A = sp.csc_matrix((data, (rows, cols)), shape=(row_cursor, 3 * N))
    return _LateralTemplate(
        N=N,
        dt_s=dt_s,
        y_idx=y_idx,
        vy_idx=vy_idx,
        u_idx=u_idx,
        y_weights=y_weights,
        P=P,
        A=A,
    )


def _solve_osqp_template(
    *,
    template: _LongitudinalTemplate | _LateralTemplate,
    q: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    osqp_cfg = (get_shadow_tuning_config().get("osqp") or {})
    setup_started = perf_counter()
    if template.solver is None:
        solver = osqp.OSQP()
        solver.setup(
            P=template.P,
            q=q,
            A=template.A,
            l=l,
            u=u,
            verbose=False,
            warm_start=bool(osqp_cfg.get("warm_start", True)),
            polish=bool(osqp_cfg.get("polish", False)),
            adaptive_rho=bool(osqp_cfg.get("adaptive_rho", True)),
            max_iter=int(osqp_cfg.get("max_iter", 4000)),
            eps_abs=float(osqp_cfg.get("eps_abs", 1e-4)),
            eps_rel=float(osqp_cfg.get("eps_rel", 1e-4)),
        )
        template.solver = solver
    else:
        template.solver.update(q=q, l=l, u=u)
    setup_latency_ms = float((perf_counter() - setup_started) * 1000.0)
    solve_started = perf_counter()
    result = template.solver.solve()
    solve_latency_ms = float((perf_counter() - solve_started) * 1000.0)
    feasible, timed_out, normalized_status = _osqp_status(getattr(result.info, "status", None))
    diagnostics = {
        "status": normalized_status,
        "feasible": feasible,
        "timed_out": timed_out,
        "iterations": int(getattr(result.info, "iter", 0) or 0),
        "objective": None if getattr(result.info, "obj_val", None) is None else float(result.info.obj_val),
        "setup_latency_ms": setup_latency_ms,
        "solve_latency_ms": solve_latency_ms,
        "total_latency_ms": float(setup_latency_ms + solve_latency_ms),
    }
    if not feasible or result.x is None:
        return None, diagnostics
    return np.asarray(result.x, dtype=float), diagnostics


def _solve_osqp_problem(
    *,
    P: sp.csc_matrix,
    q: np.ndarray,
    A: sp.csc_matrix,
    l: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    osqp_cfg = (get_shadow_tuning_config().get("osqp") or {})
    solver = osqp.OSQP()
    setup_started = perf_counter()
    solver.setup(
        P=P,
        q=q,
        A=A,
        l=l,
        u=u,
        verbose=False,
        warm_start=bool(osqp_cfg.get("warm_start", False)),
        polish=bool(osqp_cfg.get("polish", False)),
        adaptive_rho=bool(osqp_cfg.get("adaptive_rho", True)),
        max_iter=int(osqp_cfg.get("max_iter", 4000)),
        eps_abs=float(osqp_cfg.get("eps_abs", 1e-4)),
        eps_rel=float(osqp_cfg.get("eps_rel", 1e-4)),
    )
    setup_latency_ms = float((perf_counter() - setup_started) * 1000.0)
    solve_started = perf_counter()
    result = solver.solve()
    solve_latency_ms = float((perf_counter() - solve_started) * 1000.0)
    feasible, timed_out, normalized_status = _osqp_status(getattr(result.info, "status", None))
    diagnostics = {
        "status": normalized_status,
        "feasible": feasible,
        "timed_out": timed_out,
        "iterations": int(getattr(result.info, "iter", 0) or 0),
        "objective": None if getattr(result.info, "obj_val", None) is None else float(result.info.obj_val),
        "setup_latency_ms": setup_latency_ms,
        "solve_latency_ms": solve_latency_ms,
        "total_latency_ms": float(setup_latency_ms + solve_latency_ms),
    }
    if not feasible or result.x is None:
        return None, diagnostics
    return np.asarray(result.x, dtype=float), diagnostics


def _solve_longitudinal_mpc_qp(
    *,
    current_speed_mps: float,
    target_speed_mps: float,
    dt_s: float,
    horizon_steps: int,
    stop_distance_m: float | None = None,
) -> dict[str, Any]:
    stop_mode = stop_distance_m is not None
    shadow_cfg = get_shadow_tuning_config()
    lon_cfg = ((shadow_cfg.get("longitudinal") or {}).get("stop" if stop_mode else "follow") or {})
    template_build_started = perf_counter()
    template = _longitudinal_template(
        int(max(horizon_steps, 6)),
        round(float(dt_s), 4),
        stop_mode,
        float(lon_cfg.get("q_speed", 0.45 if stop_mode else 0.9)),
        float(lon_cfg.get("q_position", 0.12 if stop_mode else 0.02)),
        float(lon_cfg.get("q_speed_terminal", 5.5 if stop_mode else 2.2)),
        float(lon_cfg.get("q_position_terminal", 18.0 if stop_mode else 0.0)),
        float(lon_cfg.get("r_accel", 0.18)),
        float(lon_cfg.get("r_accel_delta", 0.3)),
    )
    N = template.N
    q = np.zeros(3 * N, dtype=float)

    if not stop_mode:
        v_reference = np.full(N, float(target_speed_mps), dtype=float)
        q[template.v_idx] = -2.0 * template.v_weights * v_reference
    else:
        p_reference = np.array(
            [float(stop_distance_m) * ((k + 1) / N) for k in range(N)],
            dtype=float,
        )
        v_reference = np.array(
            [max(float(current_speed_mps) * (1.0 - ((k + 1) / N)), 0.0) for k in range(N)],
            dtype=float,
        )
        q[template.v_idx] = -2.0 * template.v_weights * v_reference
        if template.p_weights is not None:
            q[template.p_idx] = -2.0 * template.p_weights * p_reference

    rhs_speed = np.zeros(N, dtype=float)
    rhs_speed[0] = float(current_speed_mps)
    rhs_pos = np.zeros(N, dtype=float)
    rhs_pos[0] = float(template.dt_s * float(current_speed_mps))
    v_max = max(
        float(current_speed_mps),
        float(target_speed_mps),
        float(lon_cfg.get("v_max_floor_mps", 8.0)),
    ) + float(lon_cfg.get("v_max_buffer_mps", 4.0))
    p_upper = (
        max(float(stop_distance_m), 0.5)
        if stop_mode
        else max(float(target_speed_mps), float(current_speed_mps), 4.0) * template.dt_s * N * float(lon_cfg.get("p_upper_scale", 1.8))
        + float(lon_cfg.get("p_upper_bias_m", 6.0))
    )
    a_min = float(lon_cfg.get("a_min", -6.0 if stop_mode else -3.5))
    a_max = float(lon_cfg.get("a_max", 1.5 if stop_mode else 2.0))
    l_bounds = np.concatenate(
        [
            rhs_speed,
            rhs_pos,
            np.zeros(N, dtype=float),
            np.zeros(N, dtype=float),
            np.full(N, a_min, dtype=float),
        ]
    )
    u_bounds = np.concatenate(
        [
            rhs_speed,
            rhs_pos,
            np.full(N, v_max, dtype=float),
            np.full(N, p_upper, dtype=float),
            np.full(N, a_max, dtype=float),
        ]
    )
    build_latency_ms = float((perf_counter() - template_build_started) * 1000.0)
    solution, diagnostics = _solve_osqp_template(
        template=template,
        q=q,
        l=l_bounds,
        u=u_bounds,
    )
    if solution is None:
        return {
            "feasible": False,
            "solver_status": diagnostics["status"],
            "solver_timed_out": diagnostics["timed_out"],
            "objective": diagnostics["objective"],
            "iterations": diagnostics["iterations"],
            "solver_problem_build_latency_ms": build_latency_ms,
            "solver_setup_latency_ms": diagnostics["setup_latency_ms"],
            "solver_solve_latency_ms": diagnostics["solve_latency_ms"],
            "solver_total_latency_ms": float(build_latency_ms + diagnostics["total_latency_ms"]),
            "speeds": None,
            "distances": None,
            "accelerations": None,
        }

    v_solution = [float(current_speed_mps)] + [float(solution[int(index)]) for index in template.v_idx]
    p_solution = [0.0] + [float(solution[int(index)]) for index in template.p_idx]
    a_solution = [float(solution[int(index)]) for index in template.a_idx]
    return {
        "feasible": True,
        "solver_status": diagnostics["status"],
        "solver_timed_out": diagnostics["timed_out"],
        "objective": diagnostics["objective"],
        "iterations": diagnostics["iterations"],
        "solver_problem_build_latency_ms": build_latency_ms,
        "solver_setup_latency_ms": diagnostics["setup_latency_ms"],
        "solver_solve_latency_ms": diagnostics["solve_latency_ms"],
        "solver_total_latency_ms": float(build_latency_ms + diagnostics["total_latency_ms"]),
        "speeds": v_solution,
        "distances": p_solution,
        "accelerations": a_solution,
    }


def _solve_lateral_mpc_qp(
    *,
    target_offset_m: float,
    dt_s: float,
    horizon_steps: int,
) -> dict[str, Any]:
    lat_cfg = get_shadow_tuning_config().get("lateral") or {}
    template_build_started = perf_counter()
    template = _lateral_template(
        int(max(horizon_steps, 6)),
        round(float(dt_s), 4),
        float(lat_cfg.get("q_offset", 1.1)),
        float(lat_cfg.get("q_offset_terminal", 8.0)),
        float(lat_cfg.get("q_velocity", 0.15)),
        float(lat_cfg.get("r_control", 0.12)),
        float(lat_cfg.get("r_control_delta", 0.25)),
    )
    N = template.N
    q = np.zeros(3 * N, dtype=float)
    y_reference = np.array(
        [float(target_offset_m) * _smoothstep((k + 1) / N) for k in range(N)],
        dtype=float,
    )
    q[template.y_idx] = -2.0 * template.y_weights * y_reference
    y_limit = max(
        abs(float(target_offset_m)) * float(lat_cfg.get("y_limit_scale", 1.5)),
        float(lat_cfg.get("y_limit_floor_m", 3.5)),
    )
    vy_limit = float(lat_cfg.get("vy_limit_mps", 3.0))
    u_limit = float(lat_cfg.get("u_limit", 4.0))
    zeros = np.zeros(N, dtype=float)
    l_bounds = np.concatenate(
        [
            zeros,
            zeros,
            np.full(N, -y_limit, dtype=float),
            np.full(N, -vy_limit, dtype=float),
            np.full(N, -u_limit, dtype=float),
        ]
    )
    u_bounds = np.concatenate(
        [
            zeros,
            zeros,
            np.full(N, y_limit, dtype=float),
            np.full(N, vy_limit, dtype=float),
            np.full(N, u_limit, dtype=float),
        ]
    )
    build_latency_ms = float((perf_counter() - template_build_started) * 1000.0)
    solution, diagnostics = _solve_osqp_template(
        template=template,
        q=q,
        l=l_bounds,
        u=u_bounds,
    )
    if solution is None:
        return {
            "feasible": False,
            "solver_status": diagnostics["status"],
            "solver_timed_out": diagnostics["timed_out"],
            "objective": diagnostics["objective"],
            "iterations": diagnostics["iterations"],
            "solver_problem_build_latency_ms": build_latency_ms,
            "solver_setup_latency_ms": diagnostics["setup_latency_ms"],
            "solver_solve_latency_ms": diagnostics["solve_latency_ms"],
            "solver_total_latency_ms": float(build_latency_ms + diagnostics["total_latency_ms"]),
            "offsets": None,
            "lateral_velocities": None,
            "controls": None,
        }

    offsets = [0.0] + [float(solution[int(index)]) for index in template.y_idx]
    lateral_velocities = [0.0] + [float(solution[int(index)]) for index in template.vy_idx]
    controls = [float(solution[int(index)]) for index in template.u_idx]
    return {
        "feasible": True,
        "solver_status": diagnostics["status"],
        "solver_timed_out": diagnostics["timed_out"],
        "objective": diagnostics["objective"],
        "iterations": diagnostics["iterations"],
        "solver_problem_build_latency_ms": build_latency_ms,
        "solver_setup_latency_ms": diagnostics["setup_latency_ms"],
        "solver_solve_latency_ms": diagnostics["solve_latency_ms"],
        "solver_total_latency_ms": float(build_latency_ms + diagnostics["total_latency_ms"]),
        "offsets": offsets,
        "lateral_velocities": lateral_velocities,
        "controls": controls,
    }


def _speed_profile(
    *,
    current_speed_mps: float,
    target_speed_mps: float,
    horizon_steps: int,
    dt_s: float,
    max_accel_mps2: float,
    max_decel_mps2: float,
) -> tuple[list[float], list[float]]:
    speeds = [float(max(current_speed_mps, 0.0))]
    distances = [0.0]
    for _ in range(horizon_steps):
        current = speeds[-1]
        speed_error = float(target_speed_mps) - current
        if speed_error >= 0.0:
            accel = min(speed_error / max(dt_s, 1e-3), max_accel_mps2)
        else:
            accel = max(speed_error / max(dt_s, 1e-3), -max_decel_mps2)
        next_speed = max(current + accel * dt_s, 0.0)
        distances.append(distances[-1] + 0.5 * (current + next_speed) * dt_s)
        speeds.append(next_speed)
    return speeds, distances


def _jerk_proxy_from_profile(speeds: list[float], dt_s: float) -> float | None:
    if len(speeds) < 4 or dt_s <= 0.0:
        return None
    accels = [
        float((speeds[i + 1] - speeds[i]) / dt_s)
        for i in range(len(speeds) - 1)
    ]
    jerks = [
        abs(float((accels[i + 1] - accels[i]) / dt_s))
        for i in range(len(accels) - 1)
    ]
    return _mean_or_none(jerks)


def _smoothness_proxy(distances: list[float], lateral_offsets: list[float]) -> float | None:
    if len(distances) < 3 or len(lateral_offsets) != len(distances):
        return None
    curvature_like: list[float] = []
    for i in range(1, len(distances) - 1):
        ds1 = float(distances[i] - distances[i - 1])
        ds2 = float(distances[i + 1] - distances[i])
        if abs(ds1) < 1e-6 or abs(ds2) < 1e-6:
            continue
        dy1 = float(lateral_offsets[i] - lateral_offsets[i - 1]) / ds1
        dy2 = float(lateral_offsets[i + 1] - lateral_offsets[i]) / ds2
        curvature_like.append(abs(dy2 - dy1))
    return _mean_or_none(curvature_like)


@dataclass
class _StopContext:
    active: bool = False
    remaining_distance_m: float | None = None
    source_type: str | None = None
    binding_status: str | None = None
    target_lane_id: int | None = None
    confidence: float | None = None
    stop_reason: str | None = None
    source_actor_id: int | None = None
    source_role: str | None = None
    support_level: str = "missing"

    def clear(self) -> None:
        self.active = False
        self.remaining_distance_m = None
        self.source_type = None
        self.binding_status = None
        self.target_lane_id = None
        self.confidence = None
        self.stop_reason = None
        self.source_actor_id = None
        self.source_role = None
        self.support_level = "missing"


def _resolve_stop_distance(
    *,
    stop_event: dict[str, Any] | None,
    request: dict[str, Any],
    state: dict[str, Any],
    stop_context: _StopContext,
    dt_s: float,
) -> tuple[float | None, dict[str, Any], bool]:
    source_type = None
    binding_status = None
    confidence = None
    target_lane_id = None
    stop_reason = request.get("stop_reason")
    source_actor_id = None
    source_role = None
    support_level = "missing"

    distance_candidates = [
        (_safe_float(state.get("distance_to_stop_target_m")), "runtime_distance"),
        (_safe_float(state.get("distance_to_obstacle_m")), "runtime_obstacle_distance"),
    ]
    if stop_event:
        distance_candidates.extend(
            [
                (_safe_float(stop_event.get("first_distance_to_target_m")), "stop_event_first_distance"),
                (_safe_float(stop_event.get("initial_distance_to_target_m")), "stop_event_initial_distance"),
                (_safe_float(stop_event.get("target_distance_m")), "stop_event_target_distance"),
            ]
        )
        source_type = stop_event.get("target_source_type")
        binding_status = stop_event.get("binding_status")
        confidence = _safe_float(stop_event.get("target_confidence"))
        target_lane_id = _safe_int(stop_event.get("target_lane_id"))
        stop_reason = stop_reason or stop_event.get("stop_reason")
        source_actor_id = _safe_int(stop_event.get("source_actor_id"))
        source_role = stop_event.get("source_role")

    resolved_distance = None
    for candidate, candidate_support in distance_candidates:
        if candidate is not None and candidate >= 0.0:
            resolved_distance = float(candidate)
            support_level = candidate_support
            break

    used_active_context = False
    if resolved_distance is None and stop_context.active and stop_context.remaining_distance_m is not None:
        resolved_distance = max(float(stop_context.remaining_distance_m), 0.0)
        source_type = source_type or stop_context.source_type
        binding_status = binding_status or stop_context.binding_status
        confidence = confidence if confidence is not None else stop_context.confidence
        target_lane_id = target_lane_id if target_lane_id is not None else stop_context.target_lane_id
        stop_reason = stop_reason or stop_context.stop_reason
        source_actor_id = source_actor_id if source_actor_id is not None else stop_context.source_actor_id
        source_role = source_role or stop_context.source_role
        support_level = "active_stop_context"
        used_active_context = True

    if resolved_distance is not None:
        stop_context.active = True
        stop_context.remaining_distance_m = max(
            float(resolved_distance) - max(_safe_float(state.get("speed_mps")) or 0.0, 0.0) * dt_s,
            0.0,
        )
        stop_context.source_type = str(source_type or stop_context.source_type or "obstacle")
        stop_context.binding_status = str(binding_status or stop_context.binding_status or "heuristic")
        stop_context.target_lane_id = target_lane_id if target_lane_id is not None else stop_context.target_lane_id
        stop_context.confidence = confidence if confidence is not None else stop_context.confidence
        stop_context.stop_reason = str(stop_reason or stop_context.stop_reason or "front_obstacle_or_high_risk")
        stop_context.source_actor_id = source_actor_id if source_actor_id is not None else stop_context.source_actor_id
        stop_context.source_role = source_role or stop_context.source_role
        stop_context.support_level = support_level
    else:
        stop_context.clear()

    info = {
        "target_source_type": source_type or stop_context.source_type,
        "binding_status": binding_status or stop_context.binding_status,
        "target_distance_m": resolved_distance,
        "target_lane_id": target_lane_id if target_lane_id is not None else stop_context.target_lane_id,
        "target_confidence": confidence if confidence is not None else stop_context.confidence,
        "stop_reason": stop_reason or stop_context.stop_reason,
        "source_actor_id": source_actor_id if source_actor_id is not None else stop_context.source_actor_id,
        "source_role": source_role or stop_context.source_role,
        "support_level": support_level,
    }
    return resolved_distance, info, used_active_context


def _build_stop_solution(
    *,
    current_speed_mps: float,
    dt_s: float,
    stop_distance_m: float | None,
) -> dict[str, Any]:
    stop_cfg = ((get_shadow_tuning_config().get("longitudinal") or {}).get("stop") or {})
    horizon_steps = int(stop_cfg.get("horizon_steps", 12))
    hold_speed_threshold_mps = float(stop_cfg.get("hold_speed_threshold_mps", 0.35))
    if stop_distance_m is None:
        if current_speed_mps <= hold_speed_threshold_mps:
            speeds = [0.0] * (horizon_steps + 1)
            distances = [0.0] * (horizon_steps + 1)
            return {
                "feasible": True,
                "solver_status": "optimal_stop_hold_without_explicit_target",
                "solver_timed_out": False,
                "target_speed_mps": 0.0,
                "speeds": speeds,
                "distances": distances,
                "stop_final_error_m": None,
                "stop_overshoot_distance_m": None,
                "fallback_recommendation": None,
                "notes": ["stop_target_missing_but_vehicle_already_in_hold_window"],
                "objective": None,
                "iterations": 0,
                "solver_problem_build_latency_ms": 0.0,
                "solver_setup_latency_ms": 0.0,
                "solver_solve_latency_ms": 0.0,
                "solver_total_latency_ms": 0.0,
            }
        return {
            "feasible": False,
            "solver_status": "infeasible_missing_stop_distance",
            "solver_timed_out": False,
            "target_speed_mps": 0.0,
            "speeds": [float(current_speed_mps)] + [0.0] * horizon_steps,
            "distances": [0.0] * (horizon_steps + 1),
            "stop_final_error_m": None,
            "stop_overshoot_distance_m": None,
            "fallback_recommendation": "maintain_baseline_stop_controller",
            "notes": ["stop_distance_missing_for_shadow_solver"],
            "objective": None,
            "iterations": 0,
            "solver_problem_build_latency_ms": 0.0,
            "solver_setup_latency_ms": 0.0,
            "solver_solve_latency_ms": 0.0,
            "solver_total_latency_ms": 0.0,
        }
    near_zero_distance_threshold_m = float(stop_cfg.get("near_zero_distance_threshold_m", 0.05))
    if stop_distance_m <= near_zero_distance_threshold_m and current_speed_mps > hold_speed_threshold_mps:
        fallback_started = perf_counter()
        speeds, distances = _speed_profile(
            current_speed_mps=float(current_speed_mps),
            target_speed_mps=0.0,
            horizon_steps=horizon_steps,
            dt_s=float(dt_s),
            max_accel_mps2=0.0,
            max_decel_mps2=abs(float(stop_cfg.get("a_min", -6.0))),
        )
        total_latency_ms = float((perf_counter() - fallback_started) * 1000.0)
        predicted_distance = float(distances[-1]) if distances else 0.0
        distance_error = float(predicted_distance - float(stop_distance_m))
        return {
            "feasible": True,
            "solver_status": "fallback_near_zero_stop_profile",
            "solver_timed_out": False,
            "target_speed_mps": 0.0,
            "speeds": speeds,
            "distances": distances,
            "stop_final_error_m": abs(distance_error),
            "stop_overshoot_distance_m": max(distance_error, 0.0),
            "fallback_recommendation": None,
            "notes": [
                f"near_zero_stop_distance_bypassed_qp:{float(stop_distance_m):.6f}m",
                "fallback_profile=deterministic_max_decel",
            ],
            "objective": None,
            "iterations": 0,
            "solver_problem_build_latency_ms": 0.0,
            "solver_setup_latency_ms": 0.0,
            "solver_solve_latency_ms": total_latency_ms,
            "solver_total_latency_ms": total_latency_ms,
        }
    qp = _solve_longitudinal_mpc_qp(
        current_speed_mps=current_speed_mps,
        target_speed_mps=0.0,
        dt_s=dt_s,
        horizon_steps=horizon_steps,
        stop_distance_m=stop_distance_m,
    )
    speeds = list(qp.get("speeds") or ([float(current_speed_mps)] + [0.0] * horizon_steps))
    distances = list(qp.get("distances") or ([0.0] * (horizon_steps + 1)))
    predicted_distance = float(distances[-1]) if distances else 0.0
    distance_error = float(predicted_distance - float(stop_distance_m))
    solver_status = str(qp.get("solver_status") or "unknown")
    feasible = bool(qp.get("feasible"))
    return {
        "feasible": feasible,
        "solver_status": solver_status,
        "solver_timed_out": bool(qp.get("solver_timed_out", False)),
        "target_speed_mps": 0.0,
        "speeds": speeds,
        "distances": distances,
        "stop_final_error_m": abs(distance_error),
        "stop_overshoot_distance_m": max(distance_error, 0.0),
        "fallback_recommendation": None if feasible else "maintain_baseline_stop_controller",
        "notes": [f"mpc_objective={qp.get('objective')}"],
        "objective": qp.get("objective"),
        "iterations": qp.get("iterations", 0),
        "solver_problem_build_latency_ms": float(qp.get("solver_problem_build_latency_ms") or 0.0),
        "solver_setup_latency_ms": float(qp.get("solver_setup_latency_ms") or 0.0),
        "solver_solve_latency_ms": float(qp.get("solver_solve_latency_ms") or 0.0),
        "solver_total_latency_ms": float(qp.get("solver_total_latency_ms") or 0.0),
    }


def _build_lane_change_solution(
    *,
    request: dict[str, Any],
    state: dict[str, Any],
    current_speed_mps: float,
    dt_s: float,
) -> dict[str, Any]:
    shadow_cfg = get_shadow_tuning_config()
    lane_cfg = shadow_cfg.get("lane_change") or {}
    requested_behavior = str(request.get("requested_behavior") or "")
    progress = _safe_float(state.get("lane_change_progress")) or 0.0
    target_lane_id = _safe_int(state.get("target_lane_id"))
    current_lane_id = _safe_int(state.get("current_lane_id"))
    lateral_error_m = abs(_safe_float(state.get("distance_to_target_lane_center")) or 0.0)
    direction = "left" if requested_behavior.endswith("left") else "right"
    recommended_behavior = requested_behavior
    notes: list[str] = []

    if target_lane_id is None:
        return {
            "feasible": False,
            "solver_status": "infeasible_missing_target_lane",
            "solver_timed_out": False,
            "target_speed_mps": max(_safe_float(request.get("target_speed_mps")) or current_speed_mps, 0.0),
            "recommended_behavior": requested_behavior,
            "fallback_recommendation": "maintain_baseline_lane_follow",
            "completion_time_s": None,
            "heading_settle_time_s": None,
            "lateral_oscillation_rate": None,
            "trajectory_smoothness_proxy": None,
            "control_jerk_proxy": None,
            "trajectory_points": [],
            "notes": ["missing_target_lane_id"],
            "objective": None,
            "iterations": 0,
            "solver_problem_build_latency_ms": 0.0,
            "solver_setup_latency_ms": 0.0,
            "solver_solve_latency_ms": 0.0,
            "solver_total_latency_ms": 0.0,
        }

    if (
        requested_behavior.startswith("commit_lane_change")
        and progress < float(lane_cfg.get("prepare_commit_progress_threshold", 0.05))
        and lateral_error_m > float(lane_cfg.get("prepare_commit_lateral_error_threshold_m", 2.5))
    ):
        recommended_behavior = requested_behavior.replace("commit", "prepare", 1)
        notes.append("shadow_defers_commit_until_lateral_alignment_progress")

    target_speed_mps = _safe_float(request.get("target_speed_mps")) or current_speed_mps
    capped_target_speed_mps = min(
        target_speed_mps,
        float(lane_cfg.get("target_speed_cap_high_error_mps", 4.5))
        if lateral_error_m > float(lane_cfg.get("high_error_threshold_m", 2.0))
        else float(lane_cfg.get("target_speed_cap_low_error_mps", 6.0)),
    )
    nominal_completion_time_s = _clamp(
        float(lane_cfg.get("nominal_time_base_s", 0.8))
        + lateral_error_m
        / max(
            current_speed_mps * float(lane_cfg.get("nominal_time_speed_factor", 0.35))
            + float(lane_cfg.get("nominal_time_speed_bias", 0.8)),
            float(lane_cfg.get("nominal_time_speed_bias", 0.8)),
        ),
        float(lane_cfg.get("nominal_time_min_s", 1.2)),
        float(lane_cfg.get("nominal_time_max_s", 4.5)),
    )
    horizon_steps = min(
        max(int(lane_cfg.get("min_horizon_steps", 8)), int(round(nominal_completion_time_s / max(dt_s, 0.05)))),
        int(lane_cfg.get("max_horizon_steps", 14)),
    )
    lon_qp = _solve_longitudinal_mpc_qp(
        current_speed_mps=current_speed_mps,
        target_speed_mps=capped_target_speed_mps,
        dt_s=dt_s,
        horizon_steps=horizon_steps,
        stop_distance_m=None,
    )
    signed_total_offset = copysign(lateral_error_m, -1.0 if direction == "right" else 1.0)
    lat_qp = _solve_lateral_mpc_qp(
        target_offset_m=signed_total_offset,
        dt_s=dt_s,
        horizon_steps=horizon_steps,
    )
    speeds = list(lon_qp.get("speeds") or ([float(current_speed_mps)] + [capped_target_speed_mps] * horizon_steps))
    distances = list(lon_qp.get("distances") or [float(index) * dt_s * capped_target_speed_mps for index in range(horizon_steps + 1)])
    lateral_offsets = list(lat_qp.get("offsets") or ([0.0] * (horizon_steps + 1)))
    feasible = bool(lon_qp.get("feasible")) and bool(lat_qp.get("feasible"))
    solver_timed_out = bool(lon_qp.get("solver_timed_out", False) or lat_qp.get("solver_timed_out", False))
    solver_status = "optimal_lane_change_mpc"
    if not feasible:
        solver_status = "infeasible_lane_change_mpc"
    elif str(recommended_behavior) != str(requested_behavior):
        solver_status = "constraint_limited_lane_change_mpc"
    points: list[dict[str, float]] = []
    for index in range(len(distances)):
        offset = float(lateral_offsets[min(index, len(lateral_offsets) - 1)])
        points.append(
            {
                "t_s": round(index * dt_s, 4),
                "s_m": round(distances[index], 4),
                "lateral_offset_m": round(offset, 4),
                "speed_mps": round(speeds[min(index, len(speeds) - 1)], 4),
            }
        )

    completion_time_s = None
    heading_settle_time_s = None
    for index in range(1, len(lateral_offsets)):
        if abs(lateral_offsets[index] - signed_total_offset) <= 0.25:
            completion_time_s = float(index * dt_s)
            heading_settle_time_s = float(completion_time_s + 0.45)
            break
    if completion_time_s is None:
        completion_time_s = float(horizon_steps * dt_s)
        heading_settle_time_s = float(completion_time_s + 0.45)
    sign_changes = 0
    previous_error = None
    for offset in lateral_offsets[1:]:
        error = float(signed_total_offset - offset)
        if previous_error is not None and error != 0.0 and previous_error != 0.0 and error * previous_error < 0.0:
            sign_changes += 1
        previous_error = error

    return {
        "feasible": feasible,
        "solver_status": solver_status,
        "solver_timed_out": solver_timed_out,
        "target_speed_mps": capped_target_speed_mps,
        "recommended_behavior": recommended_behavior,
        "fallback_recommendation": None if feasible else "maintain_baseline_lane_follow",
        "completion_time_s": completion_time_s,
        "heading_settle_time_s": heading_settle_time_s,
        "lateral_oscillation_rate": float(sign_changes / max(len(lateral_offsets) - 1, 1)),
        "trajectory_smoothness_proxy": _smoothness_proxy(distances, lateral_offsets),
        "control_jerk_proxy": _jerk_proxy_from_profile(speeds, dt_s),
        "trajectory_points": points,
        "notes": notes,
        "objective": {
            "longitudinal": lon_qp.get("objective"),
            "lateral": lat_qp.get("objective"),
        },
        "iterations": {
            "longitudinal": lon_qp.get("iterations", 0),
            "lateral": lat_qp.get("iterations", 0),
        },
        "solver_problem_build_latency_ms": float(lon_qp.get("solver_problem_build_latency_ms") or 0.0) + float(lat_qp.get("solver_problem_build_latency_ms") or 0.0),
        "solver_setup_latency_ms": float(lon_qp.get("solver_setup_latency_ms") or 0.0) + float(lat_qp.get("solver_setup_latency_ms") or 0.0),
        "solver_solve_latency_ms": float(lon_qp.get("solver_solve_latency_ms") or 0.0) + float(lat_qp.get("solver_solve_latency_ms") or 0.0),
        "solver_total_latency_ms": float(lon_qp.get("solver_total_latency_ms") or 0.0) + float(lat_qp.get("solver_total_latency_ms") or 0.0),
    }


def _build_follow_solution(
    *,
    request: dict[str, Any],
    current_speed_mps: float,
    dt_s: float,
) -> dict[str, Any]:
    target_speed_mps = _safe_float(request.get("target_speed_mps")) or current_speed_mps
    follow_cfg = ((get_shadow_tuning_config().get("longitudinal") or {}).get("follow") or {})
    qp = _solve_longitudinal_mpc_qp(
        current_speed_mps=current_speed_mps,
        target_speed_mps=target_speed_mps,
        dt_s=dt_s,
        horizon_steps=int(follow_cfg.get("horizon_steps", 10)),
        stop_distance_m=None,
    )
    horizon_steps = int(follow_cfg.get("horizon_steps", 10))
    speeds = list(qp.get("speeds") or ([float(current_speed_mps)] + [target_speed_mps] * horizon_steps))
    distances = list(qp.get("distances") or [float(index) * dt_s * target_speed_mps for index in range(horizon_steps + 1)])
    points = [
        {
            "t_s": round(index * dt_s, 4),
            "s_m": round(distances[index], 4),
            "lateral_offset_m": 0.0,
            "speed_mps": round(speeds[index], 4),
        }
        for index in range(len(distances))
    ]
    return {
        "feasible": bool(qp.get("feasible")),
        "solver_status": "optimal_lane_follow_mpc" if bool(qp.get("feasible")) else str(qp.get("solver_status") or "infeasible_lane_follow_mpc"),
        "solver_timed_out": bool(qp.get("solver_timed_out", False)),
        "target_speed_mps": target_speed_mps,
        "recommended_behavior": str(request.get("requested_behavior") or "keep_lane"),
        "fallback_recommendation": None if bool(qp.get("feasible")) else "maintain_baseline_lane_follow",
        "completion_time_s": None,
        "heading_settle_time_s": None,
        "lateral_oscillation_rate": 0.0,
        "trajectory_smoothness_proxy": 0.0,
        "control_jerk_proxy": _jerk_proxy_from_profile(speeds, dt_s),
        "trajectory_points": points,
        "notes": [],
        "objective": qp.get("objective"),
        "iterations": qp.get("iterations", 0),
        "solver_problem_build_latency_ms": float(qp.get("solver_problem_build_latency_ms") or 0.0),
        "solver_setup_latency_ms": float(qp.get("solver_setup_latency_ms") or 0.0),
        "solver_solve_latency_ms": float(qp.get("solver_solve_latency_ms") or 0.0),
        "solver_total_latency_ms": float(qp.get("solver_total_latency_ms") or 0.0),
    }


@dataclass
class ShadowRuntimeState:
    stop_context: _StopContext
    previous_timestamp: float | None = None


def new_shadow_runtime_state() -> ShadowRuntimeState:
    return ShadowRuntimeState(stop_context=_StopContext(), previous_timestamp=None)


def _shadow_metric_support_map(
    proposals: list[dict[str, Any]],
    shadow_metrics: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    stop_binding_statuses = {
        str(((row.get("stop_target_info") or {}).get("binding_status")) or "").strip().lower()
        for row in proposals
        if ((row.get("stop_target_info") or {}).get("binding_status")) is not None
    }
    stop_support_levels = {
        str(((row.get("stop_target_info") or {}).get("support_level")) or "").strip().lower()
        for row in proposals
        if ((row.get("stop_target_info") or {}).get("support_level")) is not None
    }

    def _value_available(metric_name: str) -> bool:
        return shadow_metrics.get(metric_name) is not None

    def _support(metric_name: str) -> str:
        if not _value_available(metric_name):
            return "unavailable"
        if metric_name == "shadow_planner_execution_latency_ms":
            return "proposal_direct_solver_timing"
        if metric_name in DIRECT_PROPOSAL_QUALITY_METRICS:
            return "proposal_direct"
        if metric_name in COUNTERFACTUAL_PROPOSAL_QUALITY_METRICS:
            return "counterfactual_from_shadow_path"
        if metric_name in {"shadow_stop_final_error_m", "shadow_stop_overshoot_distance_m"}:
            if stop_binding_statuses & EXACT_STOP_BINDING_STATUSES:
                return "proposal_direct_with_exact_stop_target"
            if stop_binding_statuses & HEURISTIC_STOP_BINDING_STATUSES:
                return "proposal_direct_with_heuristic_stop_target"
            if stop_support_levels & HEURISTIC_STOP_TARGET_SUPPORT_LEVELS:
                return "proposal_direct_with_runtime_stop_target"
            return "counterfactual_with_missing_stop_target_binding"
        return "proposal_direct"

    def _recommended_gate_usage(metric_name: str, support_level: str) -> str:
        if support_level == "unavailable":
            return "diagnostic_only"
        if metric_name == "shadow_planner_execution_latency_ms":
            return "soft_guardrail"
        if metric_name in DIRECT_PROPOSAL_QUALITY_METRICS:
            return "soft_guardrail"
        if metric_name in {"shadow_stop_final_error_m", "shadow_stop_overshoot_distance_m"} and support_level in {
            "proposal_direct_with_exact_stop_target",
            "proposal_direct_with_heuristic_stop_target",
            "proposal_direct_with_runtime_stop_target",
        }:
            return "soft_guardrail"
        return "diagnostic_only"

    support_map: dict[str, dict[str, Any]] = {}
    for metric_name in shadow_metrics:
        support_level = _support(metric_name)
        support_map[metric_name] = {
            "support_level": support_level,
            "value_available": _value_available(metric_name),
            "authority": "proposal_only_non_executed",
            "recommended_gate_usage": _recommended_gate_usage(metric_name, support_level),
        }
    return support_map


def _build_shadow_summary_from_proposals(
    *,
    case_id: str,
    proposals: list[dict[str, Any]],
    disagreement_events: list[dict[str, Any]],
    proposal_source: str,
    quality_proxy_mode: str,
) -> dict[str, Any]:
    tuning_config_ids = sorted(
        {
            str(((row.get("provenance") or {}).get("mpc_tuning_config_id")) or "").strip()
            for row in proposals
            if ((row.get("provenance") or {}).get("mpc_tuning_config_id")) is not None
        }
    )
    tuning_config_sources = sorted(
        {
            str(((row.get("provenance") or {}).get("mpc_tuning_config_source")) or "").strip()
            for row in proposals
            if ((row.get("provenance") or {}).get("mpc_tuning_config_source")) is not None
        }
    )
    stop_support_modes = {
        str(((row.get("stop_target_info") or {}).get("support_level")) or "missing")
        for row in proposals
        if (row.get("stop_target_info") or {}).get("support_level")
    }
    shadow_latencies = [_safe_float(row.get("shadow_planner_execution_latency_ms")) for row in proposals]
    shadow_stop_final_errors = [_safe_float(row.get("predicted_stop_final_error_m")) for row in proposals]
    shadow_stop_overshoots = [_safe_float(row.get("predicted_stop_overshoot_distance_m")) for row in proposals]
    shadow_lane_change_times = [_safe_float(row.get("predicted_lane_change_completion_time_s")) for row in proposals]
    shadow_heading_settles = [_safe_float(row.get("predicted_heading_settle_time_s")) for row in proposals]
    shadow_lateral_oscillations = [_safe_float(row.get("predicted_lateral_oscillation_rate")) for row in proposals]
    shadow_smoothness = [_safe_float(row.get("predicted_trajectory_smoothness_proxy")) for row in proposals]
    shadow_jerks = [_safe_float(row.get("predicted_control_jerk_proxy")) for row in proposals]
    feasible_count = sum(1 for row in proposals if bool(row.get("feasible")))
    solver_timeout_count = sum(1 for row in proposals if bool(row.get("solver_timed_out")))
    disagreement_count = sum(
        1
        for row in proposals
        if bool(
            row.get("behavior_disagreement_with_baseline")
            if row.get("behavior_disagreement_with_baseline") is not None
            else row.get("disagreement_with_baseline")
        )
    )
    fallback_rec_count = sum(1 for row in proposals if row.get("fallback_recommendation"))

    shadow_metrics = {
        "shadow_stop_final_error_m": _mean_or_none(shadow_stop_final_errors),
        "shadow_stop_overshoot_distance_m": _mean_or_none(shadow_stop_overshoots),
        "shadow_lane_change_completion_time_s": _mean_or_none(shadow_lane_change_times),
        "shadow_heading_settle_time_s": _mean_or_none(shadow_heading_settles),
        "shadow_lateral_oscillation_rate": _mean_or_none(shadow_lateral_oscillations),
        "shadow_trajectory_smoothness_proxy": _mean_or_none(shadow_smoothness),
        "shadow_control_jerk_proxy": _mean_or_none(shadow_jerks),
        "shadow_planner_execution_latency_ms": _mean_or_none(shadow_latencies),
    }
    metric_support = _shadow_metric_support_map(proposals, shadow_metrics)
    proposal_direct_metrics = sorted(
        metric_name
        for metric_name, support in metric_support.items()
        if str(support.get("support_level", "")).startswith("proposal_direct")
    )
    soft_guardrail_candidate_metrics = sorted(
        metric_name
        for metric_name, support in metric_support.items()
        if support.get("recommended_gate_usage") == "soft_guardrail"
    )
    diagnostic_only_metrics = sorted(
        metric_name
        for metric_name, support in metric_support.items()
        if support.get("recommended_gate_usage") == "diagnostic_only"
    )

    return {
        "schema_version": "stage6.mpc_shadow_summary.v1",
        "case_id": case_id,
        "proposal_mode": "shadow_only_no_takeover",
        "proposal_source": proposal_source,
        "mpc_tuning_config_ids": tuning_config_ids,
        "mpc_tuning_config_sources": tuning_config_sources,
        "comparison_reference_mode": "current_run_stage3c_actual",
        "measurement_support": {
            "shadow_target_speed_profile": "present" if proposals else "missing",
            "shadow_stop_target_handling": (
                "present"
                if stop_support_modes and stop_support_modes != {"missing"}
                else "approximate"
            ),
            "shadow_solver_status": "present" if proposals else "missing",
            "shadow_latency": "direct_osqp_solver_timing" if any(item is not None for item in shadow_latencies) else "missing",
            "shadow_disagreement": "present" if proposals else "missing",
        },
        "shadow_output_contract_completeness": 1.0 if proposals else 0.0,
        "shadow_feasibility_rate": None if not proposals else float(feasible_count / len(proposals)),
        "shadow_solver_timeout_rate": None if not proposals else float(solver_timeout_count / len(proposals)),
        "shadow_planner_execution_latency_ms": shadow_metrics["shadow_planner_execution_latency_ms"],
        "shadow_baseline_disagreement_rate": None if not proposals else float(disagreement_count / len(proposals)),
        "shadow_fallback_recommendation_rate": None if not proposals else float(fallback_rec_count / len(proposals)),
        "shadow_stop_final_error_m": shadow_metrics["shadow_stop_final_error_m"],
        "shadow_stop_overshoot_distance_m": shadow_metrics["shadow_stop_overshoot_distance_m"],
        "shadow_lane_change_completion_time_s": shadow_metrics["shadow_lane_change_completion_time_s"],
        "shadow_heading_settle_time_s": shadow_metrics["shadow_heading_settle_time_s"],
        "shadow_lateral_oscillation_rate": shadow_metrics["shadow_lateral_oscillation_rate"],
        "shadow_trajectory_smoothness_proxy": shadow_metrics["shadow_trajectory_smoothness_proxy"],
        "shadow_control_jerk_proxy": shadow_metrics["shadow_control_jerk_proxy"],
        "shadow_metric_support": metric_support,
        "proposal_direct_metrics": proposal_direct_metrics,
        "soft_guardrail_candidate_metrics": soft_guardrail_candidate_metrics,
        "diagnostic_only_metrics": diagnostic_only_metrics,
        "warnings": [
            "shadow_quality_metrics_are_proposal_only_not_executed_control",
            "shadow_metric_support_is_mixed_check_shadow_metric_support",
            "shadow_planner_execution_latency_ms_reflects_osqp_solver_runtime_only",
        ],
        "disagreement_events_count": int(len(disagreement_events)),
    }


def solve_kinematic_shadow_step(
    *,
    case_id: str,
    row: dict[str, Any],
    control_row: dict[int | str | Any, Any] | dict[str, Any] | None = None,
    latency_row: dict[int | str | Any, Any] | dict[str, Any] | None = None,
    stop_event: dict[str, Any] | None = None,
    runtime_state: ShadowRuntimeState | None = None,
    source_stage: str = "stage3c_current_run",
    proposal_source: str = "osqp_mpc_shadow_backend_v1",
    quality_proxy_mode: str = "predicted_from_osqp_mpc_shadow_backend",
) -> dict[str, Any]:
    tuning_config = get_shadow_tuning_config()
    runtime_state = runtime_state or new_shadow_runtime_state()
    request = dict(row.get("execution_request") or {})
    state = dict(row.get("execution_state") or {})
    request_frame = _safe_int(request.get("frame"))
    carla_frame = _safe_int(row.get("carla_frame"))
    timestamp = _safe_float(row.get("timestamp"))
    requested_behavior = str(request.get("requested_behavior") or "")
    current_speed_mps = max(_safe_float(state.get("speed_mps")) or 0.0, 0.0)
    target_speed_mps = _safe_float(request.get("target_speed_mps")) or current_speed_mps
    dt_s = _derive_dt(timestamp, runtime_state.previous_timestamp)
    runtime_state.previous_timestamp = timestamp
    control_row = dict(control_row or {})
    latency_row = dict(latency_row or {})

    if requested_behavior == "stop_before_obstacle" or request.get("stop_reason"):
        stop_distance_m, stop_info, used_active_context = _resolve_stop_distance(
            stop_event=stop_event,
            request=request,
            state=state,
            stop_context=runtime_state.stop_context,
            dt_s=dt_s,
        )
        solution = _build_stop_solution(
            current_speed_mps=current_speed_mps,
            dt_s=dt_s,
            stop_distance_m=stop_distance_m,
        )
        recommended_behavior = "stop_before_obstacle"
        trajectory_points = [
            {
                "t_s": round(index * dt_s, 4),
                "s_m": round(solution["distances"][index], 4),
                "lateral_offset_m": 0.0,
                "speed_mps": round(solution["speeds"][index], 4),
            }
            for index in range(len(solution["distances"]))
        ]
        notes = list(solution["notes"])
        if used_active_context:
            notes.append("stop_distance_reused_from_active_stop_context")
    elif "lane_change" in requested_behavior:
        runtime_state.stop_context.clear()
        solution = _build_lane_change_solution(
            request=request,
            state=state,
            current_speed_mps=current_speed_mps,
            dt_s=dt_s,
        )
        recommended_behavior = str(solution["recommended_behavior"])
        trajectory_points = list(solution["trajectory_points"])
        notes = list(solution["notes"])
        stop_info = {
            "target_source_type": None,
            "binding_status": None,
            "target_distance_m": None,
            "target_lane_id": None,
            "target_confidence": None,
            "stop_reason": None,
            "source_actor_id": None,
            "source_role": None,
            "support_level": "not_applicable",
        }
    else:
        runtime_state.stop_context.clear()
        solution = _build_follow_solution(
            request=request,
            current_speed_mps=current_speed_mps,
            dt_s=dt_s,
        )
        recommended_behavior = str(solution["recommended_behavior"])
        trajectory_points = list(solution["trajectory_points"])
        notes = list(solution["notes"])
        stop_info = {
            "target_source_type": None,
            "binding_status": None,
            "target_distance_m": None,
            "target_lane_id": None,
            "target_confidence": None,
            "stop_reason": None,
            "source_actor_id": None,
            "source_role": None,
            "support_level": "not_applicable",
        }

    solver_latency_ms = float(solution.get("solver_solve_latency_ms") or 0.0)
    behavior_disagreement = recommended_behavior != requested_behavior
    divergence_reasons: list[str] = []
    if behavior_disagreement:
        divergence_reasons.append("behavior_mismatch")
    if bool(solution.get("fallback_recommendation")):
        divergence_reasons.append("fallback_recommendation")
    if abs(float(solution.get("target_speed_mps") or 0.0) - float(target_speed_mps or 0.0)) > 0.75:
        divergence_reasons.append("target_speed_delta")
    solver_status_text = str(solution.get("solver_status", ""))
    if solver_status_text.startswith("constraint_limited"):
        divergence_reasons.append("constraint_limited")
    if solver_status_text.startswith("infeasible"):
        divergence_reasons.append("solver_infeasible")
    proposal_divergence = bool(divergence_reasons)
    proposal = {
        "case_id": case_id,
        "request_frame": request_frame,
        "sample_name": str(request.get("sample_name") or row.get("sample_name") or ""),
        "carla_frame": carla_frame,
        "timestamp": timestamp,
        "shadow_mode": True,
        "baseline_requested_behavior": requested_behavior,
        "baseline_target_lane": request.get("target_lane"),
        "baseline_target_speed_mps": target_speed_mps,
        "shadow_requested_behavior": recommended_behavior,
        "shadow_target_lane": request.get("target_lane"),
        "shadow_target_lane_id": state.get("target_lane_id"),
        "shadow_target_speed_mps": float(solution.get("target_speed_mps") or 0.0),
        "target_corridor": {
            "route_option": request.get("route_option"),
            "priority": request.get("priority"),
            "current_lane_id": state.get("current_lane_id"),
            "target_lane_id": state.get("target_lane_id"),
        },
        "proposed_trajectory": {
            "path_mode": (
                "stop_profile"
                if recommended_behavior == "stop_before_obstacle"
                else ("lane_change_profile" if "lane_change" in recommended_behavior else "lane_follow_profile")
            ),
            "horizon_distance_m": round(float(trajectory_points[-1]["s_m"]) if trajectory_points else 0.0, 4),
            "horizon_steps": len(trajectory_points),
            "estimated_completion_time_s": solution.get("completion_time_s"),
            "distance_to_target_lane_center_m": _safe_float(state.get("distance_to_target_lane_center")),
            "lane_change_progress_hint": float(_safe_float(state.get("lane_change_progress")) or 0.0),
            "sampled_path": trajectory_points[:12],
        },
        "proposed_target_speed_profile": {
            "current_speed_mps": current_speed_mps,
            "target_speed_mps": float(solution.get("target_speed_mps") or 0.0),
            "speed_error_mps": float((solution.get("target_speed_mps") or 0.0) - current_speed_mps),
            "profile": [
                {
                    "t_s": point["t_s"],
                    "speed_mps": point["speed_mps"],
                }
                for point in trajectory_points[:12]
            ],
        },
        "stop_target_info": stop_info,
        "feasible": bool(solution.get("feasible")),
        "solver_status": str(solution.get("solver_status") or "unknown"),
        "solver_timed_out": bool(solution.get("solver_timed_out", False)),
        "predicted_stop_final_error_m": _safe_float(solution.get("stop_final_error_m")),
        "predicted_stop_overshoot_distance_m": _safe_float(solution.get("stop_overshoot_distance_m")),
        "predicted_lane_change_completion_time_s": _safe_float(solution.get("completion_time_s")),
        "predicted_heading_settle_time_s": _safe_float(solution.get("heading_settle_time_s")),
        "predicted_lateral_oscillation_rate": _safe_float(solution.get("lateral_oscillation_rate")),
        "predicted_trajectory_smoothness_proxy": _safe_float(solution.get("trajectory_smoothness_proxy")),
        "predicted_control_jerk_proxy": _safe_float(solution.get("control_jerk_proxy")),
        "objective_summary": {
            "speed_tracking_cost": round(abs(float(solution.get("target_speed_mps") or 0.0) - current_speed_mps), 6),
            "lane_center_cost": round(abs(_safe_float(state.get("distance_to_current_lane_center")) or 0.0) + abs(_safe_float(state.get("distance_to_target_lane_center")) or 0.0), 6),
            "control_effort_cost": round(abs(_safe_float(control_row.get("steer")) or 0.0) + abs(_safe_float(control_row.get("throttle")) or 0.0) + abs(_safe_float(control_row.get("brake")) or 0.0), 6),
            "stop_target_cost": None if stop_info.get("target_distance_m") is None else round(abs(float(stop_info["target_distance_m"]) - float(trajectory_points[-1]["s_m"]) if trajectory_points else float(stop_info["target_distance_m"])), 6),
            "solver_objective": solution.get("objective"),
            "solver_iterations": solution.get("iterations"),
            "solver_problem_build_latency_ms": solution.get("solver_problem_build_latency_ms"),
            "solver_setup_latency_ms": solution.get("solver_setup_latency_ms"),
            "solver_total_latency_ms": solution.get("solver_total_latency_ms"),
        },
        "shadow_planner_problem_build_latency_ms": float(solution.get("solver_problem_build_latency_ms") or 0.0),
        "shadow_planner_execution_latency_ms": solver_latency_ms,
        "shadow_planner_setup_latency_ms": float(solution.get("solver_setup_latency_ms") or 0.0),
        "shadow_planner_total_compute_latency_ms": float(solution.get("solver_total_latency_ms") or solver_latency_ms),
        "baseline_latency_reference_ms": _safe_float(latency_row.get("planner_execution_latency_ms")),
        "fallback_recommendation": solution.get("fallback_recommendation"),
        "behavior_disagreement_with_baseline": behavior_disagreement,
        "proposal_divergence_from_baseline": proposal_divergence,
        "disagreement_with_baseline": behavior_disagreement,
        "divergence_reasons": divergence_reasons,
        "notes": notes,
        "provenance": {
            "source_stage": source_stage,
            "proposal_mode": "shadow_only_no_takeover",
            "proposal_source": proposal_source,
            "quality_proxy_mode": quality_proxy_mode,
            "mpc_tuning_config_id": str(tuning_config.get("config_id") or "baseline_v1"),
            "mpc_tuning_config_source": get_shadow_tuning_config_source(),
        },
    }
    disagreement_event = None
    if behavior_disagreement:
        disagreement_event = {
            "case_id": case_id,
            "request_frame": request_frame,
            "sample_name": proposal["sample_name"],
            "baseline_requested_behavior": requested_behavior,
            "shadow_requested_behavior": recommended_behavior,
            "fallback_recommendation": solution.get("fallback_recommendation"),
            "solver_status": proposal["solver_status"],
            "behavior_disagreement_with_baseline": behavior_disagreement,
            "proposal_divergence_from_baseline": proposal_divergence,
            "divergence_reasons": divergence_reasons,
            "notes": notes,
        }
    return {
        "proposal": proposal,
        "disagreement_event": disagreement_event,
    }


def solve_kinematic_shadow_rollout(
    *,
    case_id: str,
    execution_rows: list[dict[str, Any]],
    control_by_frame: dict[int, dict[str, Any]],
    latency_by_frame: dict[int, dict[str, Any]],
    stop_events_by_frame: dict[int, dict[str, Any]],
    stop_event_frames: list[int],
) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    disagreement_events: list[dict[str, Any]] = []
    runtime_state = new_shadow_runtime_state()

    for row in execution_rows:
        request = dict(row.get("execution_request") or {})
        request_frame = _safe_int(request.get("frame"))
        carla_frame = _safe_int(row.get("carla_frame"))
        stop_event = stop_events_by_frame.get(request_frame or -1)
        control_row = control_by_frame.get(carla_frame or -1) or {}
        latency_row = latency_by_frame.get(carla_frame or -1) or latency_by_frame.get(request_frame or -1) or {}
        step = solve_kinematic_shadow_step(
            case_id=case_id,
            row=row,
            control_row=control_row,
            latency_row=latency_row,
            stop_event=stop_event,
            runtime_state=runtime_state,
            source_stage="stage3c_current_run",
            proposal_source="osqp_mpc_shadow_backend_v1",
            quality_proxy_mode="predicted_from_osqp_mpc_shadow_backend",
        )
        proposals.append(dict(step["proposal"]))
        disagreement_event = step.get("disagreement_event")
        if disagreement_event:
            disagreement_events.append(dict(disagreement_event))

    shadow_summary = _build_shadow_summary_from_proposals(
        case_id=case_id,
        proposals=proposals,
        disagreement_events=disagreement_events,
        proposal_source="osqp_mpc_shadow_backend_v1",
        quality_proxy_mode="predicted_from_osqp_mpc_shadow_backend",
    )
    return {
        "proposals": proposals,
        "disagreement_events": disagreement_events,
        "shadow_summary": shadow_summary,
    }
