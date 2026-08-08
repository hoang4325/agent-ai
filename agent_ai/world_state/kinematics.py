"""Shared kinematic helpers for tracking, risk, and maneuver validation."""
from __future__ import annotations

import math
from typing import Sequence


def hypot2(x: float, y: float) -> float:
    return float(math.hypot(x, y))


def speed_xy(velocity_xy: Sequence[float]) -> float:
    return hypot2(float(velocity_xy[0]), float(velocity_xy[1]))


def ttc_longitudinal_s(
    position_ego: Sequence[float],
    velocity_ego: Sequence[float],
    *,
    min_closing_mps: float = 0.1,
) -> float | None:
    """Classic 1-D TTC along ego x (forward)."""
    x = float(position_ego[0])
    if x <= 0.0:
        return None
    closing = max(0.0, -float(velocity_ego[0]))
    if closing <= min_closing_mps:
        return None
    return float(x / closing)


def ttc_2d_s(
    position_ego: Sequence[float],
    velocity_ego: Sequence[float],
    *,
    min_closing_mps: float = 0.15,
    max_ttc_s: float = 20.0,
) -> float | None:
    """
    2-D time-to-closest-approach for a point object in ego frame.

    position/velocity are relative to ego (object - ego) in ego coordinates.
    Returns TTC only when currently closing (range rate < 0).
    """
    px = float(position_ego[0])
    py = float(position_ego[1])
    vx = float(velocity_ego[0])
    vy = float(velocity_ego[1])
    range_m = hypot2(px, py)
    if range_m < 1e-3:
        return 0.0
    # range rate: d/dt ||p|| = (p·v)/||p||
    range_rate = (px * vx + py * vy) / range_m
    if range_rate >= -min_closing_mps:
        return None
    ttc = -range_m / range_rate
    if ttc < 0.0 or ttc > max_ttc_s:
        return None
    return float(ttc)


def time_headway_gap_m(
    ego_speed_mps: float,
    *,
    t_gap_s: float,
    d0_m: float,
    v_min_for_scale: float = 0.5,
) -> float:
    """Minimum geometric gap using constant time headway + standstill buffer."""
    speed = max(float(v_min_for_scale), float(ego_speed_mps))
    return float(max(d0_m, d0_m + speed * float(t_gap_s)))


def idm_desired_speed_mps(
    *,
    ego_speed_mps: float,
    leader_distance_m: float | None,
    leader_speed_mps: float | None,
    v0_mps: float,
    t_gap_s: float = 1.4,
    d0_m: float = 3.0,
    a_mps2: float = 1.2,
    b_mps2: float = 2.0,
    delta: float = 4.0,
    dt_s: float = 0.1,
) -> float:
    """
    One-step IDM-style free/follow speed target (not full acceleration integration).

    Returns a desired speed clamped to [0, v0].
    """
    v = max(0.0, float(ego_speed_mps))
    v0 = max(0.5, float(v0_mps))
    if leader_distance_m is None or leader_distance_m <= 0.05:
        # free road or invalid
        free = v + a_mps2 * float(dt_s) * (1.0 - (v / v0) ** delta)
        return float(max(0.0, min(v0, free)))

    s = max(0.1, float(leader_distance_m))
    v_l = 0.0 if leader_speed_mps is None else max(0.0, float(leader_speed_mps))
    delta_v = v - v_l
    s_star = d0_m + v * t_gap_s + (v * delta_v) / (2.0 * math.sqrt(max(1e-3, a_mps2 * b_mps2)))
    s_star = max(d0_m, s_star)
    acc = a_mps2 * (1.0 - (v / v0) ** delta - (s_star / s) ** 2)
    v_next = v + acc * float(dt_s)
    return float(max(0.0, min(v0, v_next)))
