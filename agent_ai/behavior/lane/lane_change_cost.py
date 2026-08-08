"""
Cost-based lane-change evaluation (P1).

Scores keep_lane / lane_change_left / lane_change_right using gap, risk,
occupancy, route preference, and multi-mode prediction envelopes when available.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from agent_ai.world_state.kinematics import time_headway_gap_m
from agent_ai.world_state.soft_constraint_arbiter import (
    ManeuverCandidate,
    SoftConstraintArbiter,
    build_soft_components,
    inverse_gap_cost,
    ttc_cost,
)

from .schema import LaneAwareObject, LaneContext


def _nearest(
    lane_objects: Iterable[LaneAwareObject],
    *,
    relation: str,
    ahead: bool,
) -> LaneAwareObject | None:
    candidates: List[LaneAwareObject] = []
    for item in lane_objects:
        if item.lane_relation != relation:
            continue
        if item.longitudinal_m is None:
            continue
        long_m = float(item.longitudinal_m)
        if ahead and long_m <= 0.0:
            continue
        if not ahead and long_m >= 0.0:
            continue
        candidates.append(item)
    if not candidates:
        return None
    candidates.sort(key=lambda o: abs(float(o.longitudinal_m or 0.0)))
    return candidates[0]


def _obj_speed(obj: LaneAwareObject | None) -> float:
    if obj is None:
        return 0.0
    src = obj.source_track or {}
    if src.get("speed_mps") is not None:
        return max(0.0, float(src["speed_mps"]))
    return 0.0


def _prediction_risk(obj: LaneAwareObject | None) -> float:
    if obj is None:
        return 0.0
    src = obj.source_track or {}
    # Prefer multi-mode envelope if present.
    modes = src.get("predicted_modes") or []
    if modes:
        mass = 0.0
        for mode in modes:
            p = float(mode.get("probability", 0.0))
            r = float(mode.get("min_range_m", 1e9))
            ttc = mode.get("ttc_to_ego_s")
            if r < 6.0 or (ttc is not None and float(ttc) < 2.5):
                mass += p
        return min(1.0, mass)
    ttc = src.get("ttc_seconds")
    if ttc is not None:
        return ttc_cost(float(ttc))
    return 0.0


def evaluate_lane_change_candidates(
    *,
    lane_context: LaneContext,
    lane_objects: Sequence[LaneAwareObject],
    ego_speed_mps: float,
    left_ok: bool,
    left_reason: str | None,
    right_ok: bool,
    right_reason: str | None,
    route_prefer: str | None = None,
    current_front_gap_m: float | None = None,
    left_occupancy: float = 0.5,
    right_occupancy: float = 0.5,
    highest_risk: str = "low",
    prev_maneuver: str | None = None,
    arbiter: SoftConstraintArbiter | None = None,
) -> Dict[str, Any]:
    """
    Rank keep / LC-left / LC-right.

    Returns dict with ranked candidates, selected maneuver, and stage hint
    (prepare vs commit) based on cost margin vs keep_lane.
    """
    arb = arbiter or SoftConstraintArbiter()
    ego_v = max(0.0, float(ego_speed_mps))
    min_front = time_headway_gap_m(ego_v, t_gap_s=1.5, d0_m=8.0)
    min_rear = time_headway_gap_m(ego_v, t_gap_s=1.2, d0_m=6.0)
    comfort_front = min_front * 1.6
    comfort_rear = min_rear * 1.5

    risk_bias = {"low": 0.0, "medium": 0.15, "high": 0.35, "critical": 0.7}.get(
        str(highest_risk), 0.1
    )

    def target_costs(direction: str) -> tuple[Dict[str, float], List[str], Dict[str, Any]]:
        relation = f"{direction}_lane"
        front = _nearest(lane_objects, relation=relation, ahead=True)
        rear = _nearest(lane_objects, relation=relation, ahead=False)
        front_gap = None if front is None else float(front.longitudinal_m or 0.0)
        rear_gap = None if rear is None else abs(float(rear.longitudinal_m or 0.0))
        tags: List[str] = [f"eval_{direction}"]
        gap_c = 0.5 * inverse_gap_cost(front_gap, comfortable_m=comfort_front, critical_m=min_front)
        gap_c += 0.5 * inverse_gap_cost(rear_gap, comfortable_m=comfort_rear, critical_m=min_rear)
        if front_gap is None and rear_gap is None:
            gap_c = 0.05  # empty target lane slight uncertainty
            tags.append("target_lane_empty")
        pred = 0.5 * _prediction_risk(front) + 0.5 * _prediction_risk(rear)
        occ = float(left_occupancy if direction == "left" else right_occupancy)
        comfort = 0.35 + 0.4 * occ  # LC always has comfort cost
        # Progress: reward escape when current front is tight.
        progress = 0.4
        if current_front_gap_m is not None and current_front_gap_m < comfort_front:
            progress = max(0.0, 0.4 - 0.5 * inverse_gap_cost(
                current_front_gap_m, comfortable_m=comfort_front, critical_m=min_front * 0.7
            ))
            tags.append("escape_motive")
        route = 0.0
        if route_prefer == direction:
            route = -0.35  # bonus (negative cost)
            tags.append("route_align")
        elif route_prefer and route_prefer != direction:
            route = 0.25
            tags.append("route_misalign")
        hyst = 0.0
        if prev_maneuver == f"lane_change_{direction}":
            hyst = -0.15
            tags.append("hysteresis_bonus")
        elif prev_maneuver and prev_maneuver.startswith("lane_change_"):
            hyst = 0.15
        safety = risk_bias * 0.5 + pred
        components = build_soft_components(
            safety=max(0.0, safety),
            gap=gap_c,
            progress=max(0.0, progress),
            comfort=comfort,
            route=route,
            risk=pred + risk_bias * 0.3,
            hysteresis=hyst,
            preference=0.0,
        )
        meta = {
            "front_gap_m": front_gap,
            "rear_gap_m": rear_gap,
            "min_front_gap_m": min_front,
            "min_rear_gap_m": min_rear,
            "occupancy": occ,
        }
        return components, tags, meta

    candidates: List[ManeuverCandidate] = []

    # keep_lane
    keep_gap = inverse_gap_cost(
        current_front_gap_m,
        comfortable_m=time_headway_gap_m(ego_v, t_gap_s=2.5, d0_m=18.0),
        critical_m=time_headway_gap_m(ego_v, t_gap_s=0.8, d0_m=6.0),
    )
    keep_components = build_soft_components(
        safety=risk_bias * 0.4,
        gap=keep_gap,
        progress=0.15 if current_front_gap_m is not None and current_front_gap_m < 20.0 else 0.0,
        comfort=0.05,
        route=0.1 if route_prefer in {"left", "right"} else 0.0,
        risk=risk_bias,
        hysteresis=-0.1 if prev_maneuver in {None, "keep_lane", "follow", "slow_down"} else 0.05,
        preference=0.0,
    )
    candidates.append(
        ManeuverCandidate(
            maneuver="keep_lane",
            hard_ok=True,
            soft_cost=0.0,
            components=keep_components,
            tags=["baseline_keep"],
            metadata={"front_gap_m": current_front_gap_m},
        )
    )

    # left
    left_comp, left_tags, left_meta = target_costs("left")
    candidates.append(
        ManeuverCandidate(
            maneuver="lane_change_left",
            hard_ok=bool(left_ok) and bool(lane_context.left_lane.exists),
            hard_reason=None if left_ok else (left_reason or "left_unavailable"),
            components=left_comp,
            tags=left_tags,
            metadata=left_meta,
        )
    )

    # right
    right_comp, right_tags, right_meta = target_costs("right")
    candidates.append(
        ManeuverCandidate(
            maneuver="lane_change_right",
            hard_ok=bool(right_ok) and bool(lane_context.right_lane.exists),
            hard_reason=None if right_ok else (right_reason or "right_unavailable"),
            components=right_comp,
            tags=right_tags,
            metadata=right_meta,
        )
    )

    ranked = arb.rank(candidates)
    selected = arb.select(ranked)
    keep = next(c for c in ranked if c.maneuver == "keep_lane")
    margin = float(keep.soft_cost - selected.soft_cost) if selected.hard_ok else -1.0

    # Stage: commit if selected LC and clearly better; prepare if modest gain.
    stage = "none"
    behavior = None
    if selected.maneuver.startswith("lane_change_") and selected.hard_ok:
        direction = "left" if selected.maneuver.endswith("left") else "right"
        if margin >= 0.45:
            stage = "commit"
            behavior = f"commit_lane_change_{direction}"
        elif margin >= 0.12:
            stage = "prepare"
            behavior = f"prepare_lane_change_{direction}"
        else:
            # Selected LC only barely better — still emit prepare if margin positive.
            if margin > 0.0:
                stage = "prepare"
                behavior = f"prepare_lane_change_{direction}"
            else:
                selected = keep
                stage = "none"
                behavior = None

    return {
        "selected_maneuver": selected.maneuver if behavior is None else selected.maneuver,
        "selected_behavior": behavior or ("keep_lane" if selected.maneuver == "keep_lane" else selected.maneuver),
        "stage": stage,
        "cost_margin_vs_keep": margin,
        "ranked": [c.to_dict() for c in ranked],
        "selected": selected.to_dict(),
        "reasoning_tags": arb.explain(selected) + [f"cost_margin={margin:.2f}"],
    }
