"""
Interaction-aware multi-agent prediction (P2).

Builds on independent multi-mode rollouts (P1) by:
  1. Pairwise conflict scoring between tracks in ego frame
  2. Reweighting mode probabilities (yield/brake vs aggressive lateral)
  3. Optional ego-reactive brake boost when ego is closing on a leader
  4. Scene-level interaction summary for risk / contracts

This is a lightweight social-forces style layer — not a full game-theoretic planner.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

from .kinematics import hypot2, ttc_2d_s
from .motion_predictor import PredictedMode, annotate_tracks_with_prediction, predict_modes_for_track
from .schema import TrackedObject


@dataclass
class PairConflict:
    track_a: int
    track_b: int
    min_sep_m: float
    time_of_closest_s: float | None
    severity: float  # [0, 1]
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InteractionSummary:
    pair_conflicts: List[PairConflict]
    ego_leader_track_id: int | None
    ego_leader_ttc_s: float | None
    max_interaction_severity: float
    reweighted_track_ids: List[int]
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_conflicts": [c.to_dict() for c in self.pair_conflicts],
            "ego_leader_track_id": self.ego_leader_track_id,
            "ego_leader_ttc_s": self.ego_leader_ttc_s,
            "max_interaction_severity": self.max_interaction_severity,
            "reweighted_track_ids": list(self.reweighted_track_ids),
            "tags": list(self.tags),
        }


def _mode_waypoints(track: TrackedObject) -> List[List[float]]:
    """Most-likely mode waypoints, or CV fallback."""
    modes = track.predicted_modes or []
    if not modes:
        return []
    best = max(modes, key=lambda m: float(m.get("probability", 0.0)))
    return list(best.get("waypoints_ego") or [])


def _sep_along_modes(a: TrackedObject, b: TrackedObject) -> tuple[float, float | None]:
    """Min separation using most-likely mode waypoint sequences (aligned by time index)."""
    wa = _mode_waypoints(a)
    wb = _mode_waypoints(b)
    if not wa or not wb:
        # instantaneous
        dx = float(a.position_ego[0]) - float(b.position_ego[0])
        dy = float(a.position_ego[1]) - float(b.position_ego[1])
        return hypot2(dx, dy), None
    n = min(len(wa), len(wb))
    min_sep = 1e9
    t_star: float | None = None
    for i in range(n):
        dx = float(wa[i][0]) - float(wb[i][0])
        dy = float(wa[i][1]) - float(wb[i][1])
        sep = hypot2(dx, dy)
        if sep < min_sep:
            min_sep = sep
            t_star = float(wa[i][2]) if len(wa[i]) > 2 else float(i) * 0.2
    return float(min_sep), t_star


def _severity_from_sep(sep_m: float, *, critical_m: float = 3.0, safe_m: float = 12.0) -> float:
    if sep_m <= critical_m:
        return 1.0
    if sep_m >= safe_m:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - (sep_m - critical_m) / (safe_m - critical_m))))


def find_pair_conflicts(
    tracks: Sequence[TrackedObject],
    *,
    max_pairs: int = 24,
    sep_alert_m: float = 10.0,
) -> List[PairConflict]:
    """O(n²) but n is small (≤ ~20 tracked objects after filtering)."""
    items = [t for t in tracks if t.distance_m < 60.0]
    conflicts: List[PairConflict] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            # Skip far pairs in y if both far longitudinally.
            if abs(float(a.position_ego[1]) - float(b.position_ego[1])) > 8.0:
                if abs(float(a.position_ego[0]) - float(b.position_ego[0])) > 25.0:
                    continue
            sep, t_star = _sep_along_modes(a, b)
            if sep > sep_alert_m:
                continue
            sev = _severity_from_sep(sep)
            if sev < 0.15:
                continue
            tags = ["pair_conflict"]
            if sev > 0.7:
                tags.append("severe_pair_conflict")
            conflicts.append(
                PairConflict(
                    track_a=int(a.track_id),
                    track_b=int(b.track_id),
                    min_sep_m=float(sep),
                    time_of_closest_s=t_star,
                    severity=float(sev),
                    tags=tags,
                )
            )
    conflicts.sort(key=lambda c: -c.severity)
    return conflicts[:max_pairs]


def _reweight_modes_for_interaction(
    track: TrackedObject,
    *,
    conflict_severity: float,
    is_ego_leader: bool,
    ego_closing: bool,
) -> None:
    """In-place reweight predicted_modes probabilities on a track."""
    modes = track.predicted_modes
    if not modes:
        return

    # Boost cooperative / yielding modes under conflict.
    boost = {
        "brake": 1.0 + 1.8 * conflict_severity + (0.6 if is_ego_leader and ego_closing else 0.0),
        "static": 1.0 + 0.4 * conflict_severity,
        "cv": 1.0 - 0.35 * conflict_severity,
        "lat_left": 1.0 + 0.25 * conflict_severity,  # mild escape
        "lat_right": 1.0 + 0.25 * conflict_severity,
    }
    # If ego leader and closing hard, suppress lateral aggressiveness slightly
    # (prefer brake over cut-across).
    if is_ego_leader and ego_closing:
        boost["lat_left"] *= 0.85
        boost["lat_right"] *= 0.85
        boost["brake"] *= 1.15

    total = 0.0
    for mode in modes:
        mid = str(mode.get("mode_id", "cv"))
        p = float(mode.get("probability", 0.0)) * float(boost.get(mid, 1.0))
        mode["probability"] = p
        tags = list(mode.get("tags") or [])
        if conflict_severity > 0.2 and "interaction_reweight" not in tags:
            tags.append("interaction_reweight")
        mode["tags"] = tags
        total += p

    if total <= 1e-9:
        n = len(modes)
        for mode in modes:
            mode["probability"] = 1.0 / n
    else:
        for mode in modes:
            mode["probability"] = float(mode["probability"] / total)

    # Refresh envelope metrics after reweight.
    min_ttc = None
    exp_range = 0.0
    for mode in modes:
        ttc = mode.get("ttc_to_ego_s")
        if ttc is not None:
            min_ttc = float(ttc) if min_ttc is None else min(min_ttc, float(ttc))
        exp_range += float(mode.get("probability", 0.0)) * float(mode.get("min_range_m", track.distance_m))
    track.predicted_min_ttc_s = min_ttc
    track.predicted_min_range_m = float(exp_range)


def _find_ego_leader(tracks: Sequence[TrackedObject], *, half_width_m: float = 2.5) -> TrackedObject | None:
    front = [
        t
        for t in tracks
        if float(t.position_ego[0]) > 0.5 and abs(float(t.position_ego[1])) <= half_width_m
    ]
    if not front:
        return None
    front.sort(key=lambda t: float(t.position_ego[0]))
    return front[0]


def apply_interaction_prediction(
    tracks: Sequence[TrackedObject],
    *,
    ego_speed_mps: float = 0.0,
    horizon_s: float = 3.0,
    dt_s: float = 0.2,
    ensure_independent: bool = True,
) -> InteractionSummary:
    """
    Run (optional) independent prediction then interaction reweighting.

    Mutates tracks in place. Returns InteractionSummary for world/decision context.
    """
    track_list = list(tracks)
    if ensure_independent:
        # Only recompute if modes missing.
        need = [t for t in track_list if not t.predicted_modes]
        if need:
            annotate_tracks_with_prediction(need, horizon_s=horizon_s, dt_s=dt_s)
        elif not track_list:
            pass
        else:
            # Ensure all have modes.
            for t in track_list:
                if not t.predicted_modes:
                    modes = predict_modes_for_track(t, horizon_s=horizon_s, dt_s=dt_s)
                    t.predicted_modes = [m.to_dict() for m in modes]

    conflicts = find_pair_conflicts(track_list)
    severity_by_id: Dict[int, float] = {}
    for c in conflicts:
        severity_by_id[c.track_a] = max(severity_by_id.get(c.track_a, 0.0), c.severity)
        severity_by_id[c.track_b] = max(severity_by_id.get(c.track_b, 0.0), c.severity)

    leader = _find_ego_leader(track_list)
    leader_id = int(leader.track_id) if leader is not None else None
    leader_ttc = None
    ego_closing = False
    if leader is not None:
        # Relative velocity toward ego origin along x (closing if vx_rel < 0 and x > 0)
        # Object velocity in ego frame already relative if perception is ego-relative.
        vx = float(leader.velocity_ego[0]) if leader.velocity_ego else 0.0
        # Closing to ego: object ahead with negative x-velocity (approaching origin).
        if float(leader.position_ego[0]) > 0 and vx < -0.3:
            ego_closing = True
        # Also treat high ego speed + short gap as closing pressure.
        if ego_speed_mps > 4.0 and leader.distance_m < max(12.0, 1.6 * ego_speed_mps):
            ego_closing = True
        leader_ttc = leader.predicted_min_ttc_s
        if leader_ttc is None:
            leader_ttc = ttc_2d_s(leader.position_ego, leader.velocity_ego)
        # Leader always gets some ego-interaction severity.
        base = 0.25 if ego_closing else 0.1
        if leader.distance_m < 15.0:
            base = max(base, min(0.9, 1.0 - leader.distance_m / 20.0))
        severity_by_id[leader_id] = max(severity_by_id.get(leader_id or -1, 0.0), base)

    reweighted: List[int] = []
    for track in track_list:
        sev = float(severity_by_id.get(int(track.track_id), 0.0))
        if sev < 0.12:
            continue
        _reweight_modes_for_interaction(
            track,
            conflict_severity=sev,
            is_ego_leader=(leader_id is not None and int(track.track_id) == leader_id),
            ego_closing=ego_closing,
        )
        reweighted.append(int(track.track_id))
        # Store interaction severity for downstream soft costs.
        if not hasattr(track, "reasoning_tags"):
            continue
        tags = list(track.reasoning_tags or [])
        if "interaction_aware" not in tags:
            tags.append("interaction_aware")
        track.reasoning_tags = tags

    max_sev = max([c.severity for c in conflicts] + [severity_by_id.get(leader_id or -1, 0.0)], default=0.0)
    tags = ["interaction_prediction"]
    if max_sev > 0.5:
        tags.append("high_interaction_scene")
    if ego_closing and leader_id is not None:
        tags.append("ego_leader_closing")

    return InteractionSummary(
        pair_conflicts=conflicts,
        ego_leader_track_id=leader_id,
        ego_leader_ttc_s=float(leader_ttc) if leader_ttc is not None else None,
        max_interaction_severity=float(max_sev),
        reweighted_track_ids=reweighted,
        tags=tags,
    )


def interaction_risk_boost(summary: InteractionSummary | None) -> float:
    """Scalar [0, 1] additive risk boost for risk engine / contracts."""
    if summary is None:
        return 0.0
    return float(max(0.0, min(1.0, 0.65 * summary.max_interaction_severity)))
