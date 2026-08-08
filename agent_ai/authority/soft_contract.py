"""
Soft ManeuverContract enrichment (P2).

Maps behavior-stack signals (decision intent, risk, interaction, IDM speed)
into L1 ManeuverContract bounds that stay within hard absolute limits.

Also provides soft (advisory) veto signals that do not by themselves revoke
authority — SafetySupervisor remains the hard gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .maneuver_contract import build_contract, validate_contract
from .schemas import ManeuverContract, WorldState


# Map stack maneuvers → contract tactical intents (authority vocabulary).
_INTENT_MAP: Dict[str, str] = {
    "keep_lane": "keep_lane",
    "follow": "keep_lane",
    "slow_down": "keep_lane",
    "stop": "safe_stop",
    "emergency_stop": "safe_stop",
    "yield": "safe_stop",
    "lane_change_left": "prepare_lane_change_left",
    "lane_change_right": "prepare_lane_change_right",
    "prepare_lane_change_left": "prepare_lane_change_left",
    "prepare_lane_change_right": "prepare_lane_change_right",
    "commit_lane_change_left": "commit_lane_change_left",
    "commit_lane_change_right": "commit_lane_change_right",
    "stop_before_obstacle": "safe_stop",
}


@dataclass
class SoftVeto:
    code: str
    message: str
    severity: str = "SOFT"  # SOFT | HARD advisory label only
    metric: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SoftContractBundle:
    contract: ManeuverContract
    soft_vetoes: List[SoftVeto] = field(default_factory=list)
    hard_validation_errors: List[str] = field(default_factory=list)
    bound_notes: List[str] = field(default_factory=list)
    source_signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": {
                "contract_id": self.contract.contract_id,
                "tactical_intent": self.contract.tactical_intent,
                "max_lateral_offset_m": self.contract.max_lateral_offset_m,
                "max_speed_mps": self.contract.max_speed_mps,
                "max_duration_s": self.contract.max_duration_s,
                "max_longitudinal_accel_mps2": self.contract.max_longitudinal_accel_mps2,
                "max_lateral_accel_mps2": self.contract.max_lateral_accel_mps2,
                "max_jerk_mps3": self.contract.max_jerk_mps3,
                "min_ttc_threshold_s": self.contract.min_ttc_threshold_s,
                "agent_confidence": self.contract.agent_confidence,
                "agent_reasoning_summary": self.contract.agent_reasoning_summary,
                "target_lane_id": self.contract.target_lane_id,
                "validity_deadline_s": self.contract.validity_deadline_s,
                "sub_intent_sequence": list(self.contract.sub_intent_sequence),
            },
            "soft_vetoes": [v.to_dict() for v in self.soft_vetoes],
            "hard_validation_errors": list(self.hard_validation_errors),
            "bound_notes": list(self.bound_notes),
            "source_signals": dict(self.source_signals),
        }


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def derive_soft_bounds(
    *,
    maneuver: str,
    ego_speed_mps: float,
    target_speed_mps: float | None,
    min_ttc_s: float | None,
    highest_risk: str,
    interaction_severity: float = 0.0,
    confidence: float = 0.7,
    front_gap_m: float | None = None,
) -> Dict[str, float | str | list]:
    """Pure function: stack signals → suggested contract numeric bounds."""
    intent = _INTENT_MAP.get(maneuver, "keep_lane")
    notes: List[str] = []

    # Speed: prefer IDM/target, demote under risk.
    speed_cap = 8.33
    if target_speed_mps is not None:
        speed_cap = float(target_speed_mps)
    else:
        speed_cap = max(2.0, float(ego_speed_mps))
    if highest_risk == "critical":
        speed_cap = min(speed_cap, 1.0)
        notes.append("risk_critical_speed_cap")
    elif highest_risk == "high":
        speed_cap = min(speed_cap, max(2.0, 0.6 * max(ego_speed_mps, 1.0)))
        notes.append("risk_high_speed_cap")
    if interaction_severity > 0.5:
        speed_cap *= 0.85
        notes.append("interaction_speed_derate")
    speed_cap = _clamp(speed_cap, 0.0, 13.89)

    # Lateral: LC needs room; keep_lane tight.
    if "lane_change" in intent or "lane_change" in maneuver:
        lat = 0.95 if "commit" in intent or "commit" in maneuver else 0.75
        notes.append("lc_lateral_bound")
    elif intent == "safe_stop":
        lat = 0.4
    else:
        lat = 0.55
    if highest_risk in {"high", "critical"}:
        lat = min(lat, 0.6)
    lat = _clamp(lat, 0.2, 1.0)

    # Accel / jerk: tighter when risk or short TTC.
    a_long = 2.0
    a_lat = 1.5
    jerk = 3.0
    if min_ttc_s is not None and min_ttc_s < 2.5:
        a_long = 1.4
        a_lat = 1.0
        jerk = 2.2
        notes.append("short_ttc_dynamics_tight")
    if intent == "safe_stop":
        a_long = 2.5  # allow firmer brake within contract long accel as bound
        notes.append("stop_dynamics")

    # Duration: short for stop, longer for LC commit.
    if intent == "safe_stop":
        duration = 3.0
    elif "commit_lane_change" in intent:
        duration = 5.5
    elif "lane_change" in intent:
        duration = 4.5
    else:
        duration = 4.0
    if interaction_severity > 0.55:
        duration = min(duration, 3.5)
        notes.append("interaction_short_horizon")

    # TTC abort threshold: more conservative when scene is busy.
    ttc_thr = 1.5
    if min_ttc_s is not None:
        # Keep abort threshold below observed TTC so we don't instant-abort.
        ttc_thr = _clamp(min(1.8, 0.7 * float(min_ttc_s)), 1.0, 2.5)
    if highest_risk in {"high", "critical"}:
        ttc_thr = max(ttc_thr, 1.8)
        notes.append("elevated_ttc_abort")

    # Confidence floor for contract agent_confidence field.
    conf = _clamp(float(confidence), 0.0, 1.0)
    if highest_risk == "critical":
        conf = min(conf, 0.5)
    # Soft contracts used for shadow may have lower conf; hard grant still needs ≥0.85.

    sub = [intent]
    if front_gap_m is not None and front_gap_m < 12.0 and intent == "keep_lane":
        sub.append("gap_constrained")

    target_lane = None
    if "left" in maneuver or "left" in intent:
        target_lane = "left"
    elif "right" in maneuver or "right" in intent:
        target_lane = "right"

    return {
        "tactical_intent": intent,
        "sub_intent_sequence": sub,
        "max_lateral_offset_m": lat,
        "max_longitudinal_accel_mps2": a_long,
        "max_lateral_accel_mps2": a_lat,
        "max_jerk_mps3": jerk,
        "max_duration_s": duration,
        "max_speed_mps": speed_cap,
        "min_ttc_threshold_s": ttc_thr,
        "agent_confidence": conf,
        "target_lane_id": target_lane,
        "notes": notes,
    }


def evaluate_soft_vetoes(
    *,
    min_ttc_s: float | None,
    highest_risk: str,
    interaction_severity: float,
    ego_speed_mps: float,
    max_speed_mps: float,
    world_age_ms: int | None = None,
) -> List[SoftVeto]:
    vetoes: List[SoftVeto] = []
    if min_ttc_s is not None and min_ttc_s < 2.0:
        vetoes.append(
            SoftVeto(
                code="SOFT-TTC",
                message=f"short TTC {min_ttc_s:.2f}s — prefer baseline or reduced authority",
                severity="SOFT",
                metric=float(min_ttc_s),
            )
        )
    if highest_risk == "critical":
        vetoes.append(
            SoftVeto(
                code="SOFT-RISK-CRIT",
                message="critical risk — soft veto agent expansion",
                severity="SOFT",
                metric=1.0,
            )
        )
    if interaction_severity > 0.65:
        vetoes.append(
            SoftVeto(
                code="SOFT-INTERACT",
                message=f"high interaction severity {interaction_severity:.2f}",
                severity="SOFT",
                metric=float(interaction_severity),
            )
        )
    if ego_speed_mps > max_speed_mps + 1.5:
        vetoes.append(
            SoftVeto(
                code="SOFT-SPEED",
                message=f"ego {ego_speed_mps:.1f} > contract cap {max_speed_mps:.1f}",
                severity="SOFT",
                metric=float(ego_speed_mps - max_speed_mps),
            )
        )
    if world_age_ms is not None and world_age_ms > 80:
        vetoes.append(
            SoftVeto(
                code="SOFT-STALE",
                message=f"world age {world_age_ms}ms approaching freshness limit",
                severity="SOFT",
                metric=float(world_age_ms),
            )
        )
    return vetoes


def build_soft_contract_from_behavior(
    *,
    frame_id: int,
    sim_time_s: float,
    maneuver: str,
    ego_speed_mps: float,
    target_speed_mps: float | None = None,
    min_ttc_s: float | None = None,
    highest_risk: str = "low",
    interaction_severity: float = 0.0,
    confidence: float = 0.7,
    front_gap_m: float | None = None,
    reasoning_tags: Sequence[str] | None = None,
    world_for_validate: WorldState | None = None,
    agent_reasoning_summary: str | None = None,
) -> SoftContractBundle:
    """
    Build a ManeuverContract enriched from behavior/risk/interaction signals.

    Always clamps via build_contract. Soft vetoes are advisory.
    If world_for_validate is provided, hard validate_contract errors are collected
    (does not raise).
    """
    bounds = derive_soft_bounds(
        maneuver=maneuver,
        ego_speed_mps=ego_speed_mps,
        target_speed_mps=target_speed_mps,
        min_ttc_s=min_ttc_s,
        highest_risk=highest_risk,
        interaction_severity=interaction_severity,
        confidence=confidence,
        front_gap_m=front_gap_m,
    )
    notes = list(bounds.pop("notes"))  # type: ignore[arg-type]
    tags = list(reasoning_tags or [])
    summary = agent_reasoning_summary or (
        f"soft_contract maneuver={maneuver} risk={highest_risk} "
        f"interact={interaction_severity:.2f} tags={','.join(tags[:6])}"
    )

    # For shadow/soft path we may want confidence as-is; for hard grant path
    # callers can bump confidence. build_contract does not enforce 0.85 itself.
    contract = build_contract(
        frame_id=frame_id,
        sim_time_s=sim_time_s,
        tactical_intent=str(bounds["tactical_intent"]),
        sub_intent_sequence=list(bounds["sub_intent_sequence"]),  # type: ignore[arg-type]
        target_state_description=f"soft_enriched:{maneuver}",
        agent_confidence=float(bounds["agent_confidence"]),  # type: ignore[arg-type]
        agent_reasoning_summary=summary,
        target_lane_id=bounds["target_lane_id"],  # type: ignore[arg-type]
        max_lateral_offset_m=float(bounds["max_lateral_offset_m"]),  # type: ignore[arg-type]
        max_longitudinal_accel_mps2=float(bounds["max_longitudinal_accel_mps2"]),  # type: ignore[arg-type]
        max_lateral_accel_mps2=float(bounds["max_lateral_accel_mps2"]),  # type: ignore[arg-type]
        max_jerk_mps3=float(bounds["max_jerk_mps3"]),  # type: ignore[arg-type]
        max_duration_s=float(bounds["max_duration_s"]),  # type: ignore[arg-type]
        max_speed_mps=float(bounds["max_speed_mps"]),  # type: ignore[arg-type]
        min_ttc_threshold_s=float(bounds["min_ttc_threshold_s"]),  # type: ignore[arg-type]
    )

    soft_vetoes = evaluate_soft_vetoes(
        min_ttc_s=min_ttc_s,
        highest_risk=highest_risk,
        interaction_severity=interaction_severity,
        ego_speed_mps=ego_speed_mps,
        max_speed_mps=float(contract.max_speed_mps),
        world_age_ms=None if world_for_validate is None else int(world_for_validate.world_age_ms),
    )

    hard_errors: List[str] = []
    if world_for_validate is not None:
        for err in validate_contract(contract, world_for_validate):
            hard_errors.append(f"{err.code}: {err.message}")

    return SoftContractBundle(
        contract=contract,
        soft_vetoes=soft_vetoes,
        hard_validation_errors=hard_errors,
        bound_notes=notes,
        source_signals={
            "maneuver": maneuver,
            "ego_speed_mps": ego_speed_mps,
            "target_speed_mps": target_speed_mps,
            "min_ttc_s": min_ttc_s,
            "highest_risk": highest_risk,
            "interaction_severity": interaction_severity,
            "confidence": confidence,
            "front_gap_m": front_gap_m,
        },
    )


def soft_cost_profile(
    *,
    highest_risk: str,
    interaction_severity: float,
    soft_veto_count: int,
) -> str:
    """Label for TrajectoryRequest.cost_profile enrichment."""
    if highest_risk == "critical" or soft_veto_count >= 3:
        return "AGENT_BOUNDED_DEFENSIVE"
    if interaction_severity > 0.5 or highest_risk == "high":
        return "AGENT_BOUNDED_CAUTIOUS"
    return "AGENT_BOUNDED"
