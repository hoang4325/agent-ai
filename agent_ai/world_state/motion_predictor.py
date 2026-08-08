"""
Multi-mode short-horizon motion prediction for tracked objects (P1 + A1).

Modes (ego frame):
  - cv: constant velocity
  - brake: decelerate along heading
  - lat_left / lat_right: CV + mild lateral acceleration (vehicles)
  - lane_follow: constant speed along map/reference polyline (Frenet)
  - lane_change_left / lane_change_right: blend toward target-lane offset
  - static: stay put (static / very slow objects)

Each mode produces waypoints and a mode probability. Consumers use either the
most-likely mode or a probability-weighted risk envelope (min TTC / range).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

from .frenet import frenet_to_xy, project_to_frenet
from .kinematics import hypot2, ttc_2d_s
from .schema import TrackedObject


@dataclass
class PredictedMode:
    mode_id: str
    probability: float
    waypoints_ego: List[List[float]]  # [x, y, t]
    terminal_position_ego: List[float]
    min_range_m: float
    ttc_to_ego_s: float | None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _unit(vx: float, vy: float) -> tuple[float, float]:
    n = hypot2(vx, vy)
    if n < 1e-3:
        return 1.0, 0.0  # default forward in ego
    return vx / n, vy / n


def _rollout(
    *,
    px: float,
    py: float,
    vx: float,
    vy: float,
    ax: float,
    ay: float,
    horizon_s: float,
    dt: float,
) -> tuple[List[List[float]], float, float | None]:
    """
    Integrate constant acceleration.

    Returns:
      waypoints, min_range_along_horizon, ttc_at_t0 (instantaneous, not min-along-path).
    TTC is evaluated at the initial state of the mode so a future near-miss does not
    collapse TTC to 0 after the object has already arrived.
    """
    t = 0.0
    x, y = px, py
    speed_x, speed_y = vx, vy
    waypoints: List[List[float]] = [[x, y, 0.0]]
    min_range = hypot2(x, y)
    ttc_at_t0 = ttc_2d_s([px, py], [vx, vy])

    steps = max(1, int(round(horizon_s / dt)))
    for _ in range(steps):
        speed_x = speed_x + ax * dt
        speed_y = speed_y + ay * dt
        x = x + speed_x * dt
        y = y + speed_y * dt
        t += dt
        waypoints.append([float(x), float(y), float(t)])
        r = hypot2(x, y)
        if r < min_range:
            min_range = r

    # If initial TTC is missing but the rollout closes within the horizon, estimate
    # time-to-closest-approach as the first time range decreases below start * 0.5
    # or below 4 m — a soft proxy, not a post-collision zero.
    if ttc_at_t0 is None and min_range + 0.5 < hypot2(px, py):
        start_r = hypot2(px, py)
        for wp in waypoints[1:]:
            if hypot2(wp[0], wp[1]) <= max(4.0, 0.5 * start_r):
                ttc_at_t0 = float(wp[2])
                break

    return waypoints, float(min_range), ttc_at_t0


def _normalize_probs(modes: List[PredictedMode]) -> List[PredictedMode]:
    total = sum(max(0.0, m.probability) for m in modes)
    if total <= 1e-9:
        n = len(modes) or 1
        for m in modes:
            m.probability = 1.0 / n
        return modes
    for m in modes:
        m.probability = float(max(0.0, m.probability) / total)
    return modes


def _rollout_lane_follow(
    *,
    px: float,
    py: float,
    speed_mps: float,
    polyline: Sequence[Sequence[float]],
    horizon_s: float,
    dt: float,
    lateral_blend: float = 0.35,
    target_d_m: float | None = None,
) -> tuple[List[List[float]], float, float | None] | None:
    """
    Integrate motion along polyline: advance s at |speed|, blend d toward target_d.
    """
    pose = project_to_frenet([px, py], polyline)
    if pose is None:
        return None
    s = pose.s_m
    d = pose.d_m if target_d_m is None else float(target_d_m) * 0.0 + pose.d_m
    d_goal = 0.0 if target_d_m is None else float(target_d_m)
    speed = max(0.3, abs(float(speed_mps)))
    # Longitudinal speed along path: use projection of velocity if available via speed.
    waypoints: List[List[float]] = [[px, py, 0.0]]
    min_range = hypot2(px, py)
    t = 0.0
    steps = max(1, int(round(horizon_s / dt)))
    x, y = px, py
    for _ in range(steps):
        s = s + speed * dt
        d = d + lateral_blend * (d_goal - d)
        xy = frenet_to_xy(polyline, s, d)
        if xy is None:
            break
        x, y = float(xy[0]), float(xy[1])
        t += dt
        waypoints.append([x, y, float(t)])
        r = hypot2(x, y)
        if r < min_range:
            min_range = r
    # Approx velocity along path at t0 for TTC.
    h = pose.heading_rad
    vx0 = speed * math.cos(h)
    vy0 = speed * math.sin(h)
    # If object is roughly ahead of ego and moving with path, relative vel toward ego
    # is better captured as object motion in ego frame; use position-delta velocity.
    if len(waypoints) >= 2:
        vx0 = (waypoints[1][0] - waypoints[0][0]) / max(dt, 1e-3)
        vy0 = (waypoints[1][1] - waypoints[0][1]) / max(dt, 1e-3)
    ttc = ttc_2d_s([px, py], [vx0, vy0])
    return waypoints, float(min_range), ttc


def predict_modes_for_track(
    track: TrackedObject,
    *,
    horizon_s: float = 3.0,
    dt_s: float = 0.2,
    brake_accel_mps2: float = 2.5,
    lateral_accel_mps2: float = 1.2,
    reference_polyline: Sequence[Sequence[float]] | None = None,
    enable_lane_change_modes: bool = True,
    lane_width_m: float = 3.5,
) -> List[PredictedMode]:
    """Generate multi-mode predictions for a single track in ego coordinates."""
    px = float(track.position_ego[0])
    py = float(track.position_ego[1])
    vx = float(track.velocity_ego[0]) if track.velocity_ego else 0.0
    vy = float(track.velocity_ego[1]) if len(track.velocity_ego) > 1 else 0.0
    speed = hypot2(vx, vy)
    ux, uy = _unit(vx, vy)
    # Lateral unit (left of velocity direction in 2D).
    lx, ly = -uy, ux

    modes: List[PredictedMode] = []
    map_aware = reference_polyline is not None and len(reference_polyline) >= 2

    # --- CV ---
    wp, rmin, ttc = _rollout(
        px=px, py=py, vx=vx, vy=vy, ax=0.0, ay=0.0, horizon_s=horizon_s, dt=dt_s
    )
    p_cv = 0.40 if (map_aware and track.class_group == "vehicle") else (
        0.55 if track.class_group == "vehicle" else 0.45
    )
    if speed < 0.3:
        p_cv = 0.25
    modes.append(
        PredictedMode(
            mode_id="cv",
            probability=p_cv,
            waypoints_ego=wp,
            terminal_position_ego=wp[-1][:2],
            min_range_m=rmin,
            ttc_to_ego_s=ttc,
            tags=["constant_velocity"],
        )
    )

    # --- Brake (along heading) ---
    if speed > 0.4:
        ax = -brake_accel_mps2 * ux
        ay = -brake_accel_mps2 * uy
        wp, rmin, ttc = _rollout(
            px=px, py=py, vx=vx, vy=vy, ax=ax, ay=ay, horizon_s=horizon_s, dt=dt_s
        )
        modes.append(
            PredictedMode(
                mode_id="brake",
                probability=0.18 if track.class_group == "vehicle" else 0.15,
                waypoints_ego=wp,
                terminal_position_ego=wp[-1][:2],
                min_range_m=rmin,
                ttc_to_ego_s=ttc,
                tags=["decelerate"],
            )
        )

    # --- Lateral bias (vehicles / VRU slightly) — free-space when no map ---
    if track.class_group in {"vehicle", "vru"} and speed > 0.5 and not map_aware:
        lat_scale = lateral_accel_mps2 if track.class_group == "vehicle" else 0.6 * lateral_accel_mps2
        for mode_id, sign, base_p in (
            ("lat_left", 1.0, 0.12),
            ("lat_right", -1.0, 0.12),
        ):
            ax = sign * lat_scale * lx
            ay = sign * lat_scale * ly
            wp, rmin, ttc = _rollout(
                px=px, py=py, vx=vx, vy=vy, ax=ax, ay=ay, horizon_s=horizon_s, dt=dt_s
            )
            modes.append(
                PredictedMode(
                    mode_id=mode_id,
                    probability=base_p if track.class_group == "vehicle" else base_p * 0.7,
                    waypoints_ego=wp,
                    terminal_position_ego=wp[-1][:2],
                    min_range_m=rmin,
                    ttc_to_ego_s=ttc,
                    tags=["lateral_bias", mode_id],
                )
            )

    # --- Map-aware lane_follow (+ optional LC modes) ---
    if map_aware and track.class_group in {"vehicle", "vru"} and speed > 0.2:
        path_speed = max(speed, 0.5)
        # Prefer signed longitudinal speed along path from velocity projection.
        pose0 = project_to_frenet([px, py], reference_polyline)  # type: ignore[arg-type]
        if pose0 is not None:
            # Component of velocity along heading.
            path_speed = max(
                0.3,
                abs(vx * math.cos(pose0.heading_rad) + vy * math.sin(pose0.heading_rad)),
            )
            if path_speed < 0.3:
                path_speed = max(speed, 0.5)
        lf = _rollout_lane_follow(
            px=px,
            py=py,
            speed_mps=path_speed,
            polyline=reference_polyline,  # type: ignore[arg-type]
            horizon_s=horizon_s,
            dt=dt_s,
            target_d_m=0.0,
        )
        if lf is not None:
            wp, rmin, ttc = lf
            modes.append(
                PredictedMode(
                    mode_id="lane_follow",
                    probability=0.35 if track.class_group == "vehicle" else 0.25,
                    waypoints_ego=wp,
                    terminal_position_ego=wp[-1][:2],
                    min_range_m=rmin,
                    ttc_to_ego_s=ttc,
                    tags=["map_aware", "lane_follow"],
                )
            )

        if enable_lane_change_modes and track.class_group == "vehicle" and speed > 1.0:
            for direction, mode_id, base_p in (
                ("left", "lane_change_left", 0.08),
                ("right", "lane_change_right", 0.08),
            ):
                # Use current corridor with target lateral offset (±lane width).
                d_goal = lane_width_m if direction == "left" else -lane_width_m
                lc = _rollout_lane_follow(
                    px=px,
                    py=py,
                    speed_mps=path_speed * 0.95,
                    polyline=reference_polyline,  # type: ignore[arg-type]
                    horizon_s=horizon_s,
                    dt=dt_s,
                    lateral_blend=0.45,
                    target_d_m=d_goal,
                )
                if lc is None:
                    continue
                wp, rmin, ttc = lc
                modes.append(
                    PredictedMode(
                        mode_id=mode_id,
                        probability=base_p,
                        waypoints_ego=wp,
                        terminal_position_ego=wp[-1][:2],
                        min_range_m=rmin,
                        ttc_to_ego_s=ttc,
                        tags=["map_aware", "lane_change", direction],
                    )
                )

    # --- Static hold ---
    if track.class_group == "static" or speed < 0.25:
        wp = [[px, py, float(i) * dt_s] for i in range(int(round(horizon_s / dt_s)) + 1)]
        modes.append(
            PredictedMode(
                mode_id="static",
                probability=0.55 if track.class_group == "static" else 0.15,
                waypoints_ego=wp,
                terminal_position_ego=[px, py],
                min_range_m=hypot2(px, py),
                ttc_to_ego_s=None,
                tags=["static_hold"],
            )
        )

    return _normalize_probs(modes)


def annotate_tracks_with_prediction(
    tracks: Sequence[TrackedObject],
    *,
    horizon_s: float = 3.0,
    dt_s: float = 0.2,
    reference_polyline: Sequence[Sequence[float]] | None = None,
    enable_lane_change_modes: bool = True,
    lane_width_m: float = 3.5,
) -> List[TrackedObject]:
    """
    In-place annotate tracks with predicted_modes / envelope metrics.
    When reference_polyline is provided (ego-frame centerline), adds map-aware modes.
    Returns the same list for chaining.
    """
    for track in tracks:
        modes = predict_modes_for_track(
            track,
            horizon_s=horizon_s,
            dt_s=dt_s,
            reference_polyline=reference_polyline,
            enable_lane_change_modes=enable_lane_change_modes,
            lane_width_m=lane_width_m,
        )
        track.predicted_modes = [m.to_dict() for m in modes]
        min_ttc = None
        min_range = None
        for m in modes:
            if m.ttc_to_ego_s is not None:
                min_ttc = m.ttc_to_ego_s if min_ttc is None else min(min_ttc, m.ttc_to_ego_s)
            min_range = m.min_range_m if min_range is None else min(min_range, m.min_range_m)
        # Probability-weighted expected min range (for soft risk).
        if modes:
            exp_range = sum(m.probability * m.min_range_m for m in modes)
            track.predicted_min_range_m = float(exp_range)
        else:
            track.predicted_min_range_m = min_range
        track.predicted_min_ttc_s = min_ttc
    return list(tracks)


def worst_case_ttc_s(tracks: Sequence[TrackedObject]) -> float | None:
    values = [t.predicted_min_ttc_s for t in tracks if t.predicted_min_ttc_s is not None]
    if not values:
        # Fall back to instantaneous tracker TTC.
        values = [t.ttc_seconds for t in tracks if t.ttc_seconds is not None]
    if not values:
        return None
    return float(min(values))


def mode_collision_likelihood(
    track: TrackedObject,
    *,
    range_threshold_m: float = 4.0,
    ttc_threshold_s: float = 2.5,
) -> float:
    """Soft probability mass on modes that get uncomfortably close."""
    if not track.predicted_modes:
        if track.ttc_seconds is not None and track.ttc_seconds < ttc_threshold_s:
            return 0.6
        if track.distance_m < range_threshold_m:
            return 0.5
        return 0.0
    mass = 0.0
    for mode in track.predicted_modes:
        p = float(mode.get("probability", 0.0))
        r = float(mode.get("min_range_m", 1e9))
        ttc = mode.get("ttc_to_ego_s")
        risky = r < range_threshold_m or (ttc is not None and float(ttc) < ttc_threshold_s)
        if risky:
            mass += p
    return float(min(1.0, mass))
