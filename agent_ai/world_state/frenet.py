"""
Lightweight Frenet (s, d) helpers on a 2-D polyline (P2+/A1).

Coordinates are typically ego-frame xy. Used for map-aware prediction and
lane-relative gap reasoning without requiring a full HD-map client.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class FrenetPose:
    s_m: float
    d_m: float
    segment_index: int
    heading_rad: float
    closest_xy: Point


def _as_points(polyline: Sequence[Sequence[float]]) -> List[Point]:
    pts: List[Point] = []
    for p in polyline:
        if len(p) < 2:
            continue
        pts.append((float(p[0]), float(p[1])))
    return pts


def polyline_length(polyline: Sequence[Sequence[float]]) -> float:
    pts = _as_points(polyline)
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return float(total)


def cumulative_s(polyline: Sequence[Sequence[float]]) -> List[float]:
    pts = _as_points(polyline)
    if not pts:
        return []
    out = [0.0]
    for i in range(1, len(pts)):
        out.append(out[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    return out


def project_to_frenet(xy: Sequence[float], polyline: Sequence[Sequence[float]]) -> FrenetPose | None:
    """Project point onto polyline; return (s, d, heading)."""
    pts = _as_points(polyline)
    if len(pts) < 2:
        return None
    x, y = float(xy[0]), float(xy[1])
    best_dist2 = 1e18
    best: FrenetPose | None = None
    s_prefix = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            s_prefix += 0.0
            continue
        t = ((x - x0) * dx + (y - y0) * dy) / seg_len2
        t_clamped = max(0.0, min(1.0, t))
        cx = x0 + t_clamped * dx
        cy = y0 + t_clamped * dy
        ddx, ddy = x - cx, y - cy
        dist2 = ddx * ddx + ddy * ddy
        if dist2 < best_dist2:
            best_dist2 = dist2
            seg_len = math.sqrt(seg_len2)
            heading = math.atan2(dy, dx)
            # Signed lateral: left of heading is positive d.
            cross = dx * (y - y0) - dy * (x - x0)
            # Use unclamped projection for sign stability near segment.
            d_sign = 1.0 if cross >= 0.0 else -1.0
            d_abs = math.sqrt(dist2)
            best = FrenetPose(
                s_m=float(s_prefix + t_clamped * seg_len),
                d_m=float(d_sign * d_abs),
                segment_index=i,
                heading_rad=float(heading),
                closest_xy=(float(cx), float(cy)),
            )
        s_prefix += math.sqrt(seg_len2)
    return best


def sample_polyline_at_s(polyline: Sequence[Sequence[float]], s_m: float) -> Point | None:
    pts = _as_points(polyline)
    if len(pts) < 2:
        return None
    if s_m <= 0.0:
        return pts[0]
    remaining = float(s_m)
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-9:
            continue
        if remaining <= seg:
            r = remaining / seg
            return (x0 + r * (x1 - x0), y0 + r * (y1 - y0))
        remaining -= seg
    return pts[-1]


def heading_at_s(polyline: Sequence[Sequence[float]], s_m: float) -> float:
    pts = _as_points(polyline)
    if len(pts) < 2:
        return 0.0
    remaining = max(0.0, float(s_m))
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-9:
            continue
        if remaining <= seg + 1e-6:
            return math.atan2(y1 - y0, x1 - x0)
        remaining -= seg
    x0, y0 = pts[-2]
    x1, y1 = pts[-1]
    return math.atan2(y1 - y0, x1 - x0)


def frenet_to_xy(polyline: Sequence[Sequence[float]], s_m: float, d_m: float) -> Point | None:
    base = sample_polyline_at_s(polyline, s_m)
    if base is None:
        return None
    heading = heading_at_s(polyline, s_m)
    # Left normal.
    nx, ny = -math.sin(heading), math.cos(heading)
    return (base[0] + d_m * nx, base[1] + d_m * ny)


def default_straight_corridor(
    *,
    horizon_m: float = 60.0,
    step_m: float = 5.0,
    lateral_offset_m: float = 0.0,
) -> List[List[float]]:
    """Synthetic current-lane centerline along +x in ego frame."""
    pts: List[List[float]] = []
    s = 0.0
    while s <= horizon_m + 1e-6:
        pts.append([float(s), float(lateral_offset_m)])
        s += step_m
    if len(pts) < 2:
        pts = [[0.0, lateral_offset_m], [horizon_m, lateral_offset_m]]
    return pts


def corridor_polyline_from_forward_corridor(
    forward_corridor: dict | None,
    *,
    ego_frame_fallback: bool = True,
    horizon_m: float = 60.0,
) -> List[List[float]] | None:
    """
    Extract a 2-D polyline from LaneContext.forward_corridor.

    Prefer ego-relative samples if present; else build a straight fallback so
    map-aware modes still work offline without CARLA.
    """
    if forward_corridor:
        chain = forward_corridor.get("chain") or []
        pts: List[List[float]] = []
        for sample in chain:
            # Prefer explicit ego-frame positions if producers attach them.
            if "position_ego" in sample:
                pe = sample["position_ego"]
                pts.append([float(pe[0]), float(pe[1])])
                continue
            # distance_m along corridor with optional lateral_m.
            if "distance_m" in sample:
                lat = float(sample.get("lateral_m") or 0.0)
                pts.append([float(sample["distance_m"]), lat])
        if len(pts) >= 2:
            return pts
    if ego_frame_fallback:
        return default_straight_corridor(horizon_m=horizon_m)
    return None


def target_lane_polyline(
    *,
    direction: str,
    lane_width_m: float = 3.5,
    horizon_m: float = 60.0,
) -> List[List[float]]:
    """Synthetic target-lane centerline: left = +y, right = -y in ego frame."""
    lat = lane_width_m if direction == "left" else -lane_width_m
    return default_straight_corridor(horizon_m=horizon_m, lateral_offset_m=lat)


def longitudinal_gap_on_polyline(
    ego_xy: Sequence[float],
    object_xy: Sequence[float],
    polyline: Sequence[Sequence[float]],
) -> float | None:
    """Signed gap object_s - ego_s along polyline (positive = object ahead)."""
    ego_f = project_to_frenet(ego_xy, polyline)
    obj_f = project_to_frenet(object_xy, polyline)
    if ego_f is None or obj_f is None:
        return None
    return float(obj_f.s_m - ego_f.s_m)
