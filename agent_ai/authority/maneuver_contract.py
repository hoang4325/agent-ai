"""
Stage 9 — Maneuver Contract (v2)
==================================
ManeuverContract builder + validator.

Responsibilities:
  - Provide default-safe bounds
  - Validate that no contract violates hard-coded physical limits before submission
  - Stamp contract with issued_at_frame and validity_deadline_s
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import ManeuverContract, WorldState


# ── Absolute hard limits (cannot be overridden by agent or contract) ──────────
_ABS_MAX_LATERAL_OFFSET_M      = 1.0
_ABS_MAX_SPEED_MPS             = 13.89   # 50 km/h
_ABS_MAX_DURATION_S            = 8.0
_ABS_MAX_JERK_MPS3             = 4.0
_ABS_MIN_CONFIDENCE            = 0.85
_ABS_FRESHNESS_MS              = 100


@dataclass
class ContractValidationError:
    code: str
    message: str


def validate_contract(
    contract: ManeuverContract,
    world: WorldState,
) -> list[ContractValidationError]:
    """
    Lightweight pre-submission validation (before Safety Supervisor).
    Returns list of errors; empty = valid for submission.
    """
    errors: list[ContractValidationError] = []

    if contract.max_lateral_offset_m > _ABS_MAX_LATERAL_OFFSET_M:
        errors.append(ContractValidationError(
            "C-001", f"lateral_offset {contract.max_lateral_offset_m} > hard limit {_ABS_MAX_LATERAL_OFFSET_M}m"
        ))
    if contract.max_speed_mps > _ABS_MAX_SPEED_MPS:
        errors.append(ContractValidationError(
            "C-002", f"speed {contract.max_speed_mps} m/s > hard limit {_ABS_MAX_SPEED_MPS} m/s"
        ))
    if contract.max_duration_s > _ABS_MAX_DURATION_S:
        errors.append(ContractValidationError(
            "C-003", f"duration {contract.max_duration_s}s > hard limit {_ABS_MAX_DURATION_S}s"
        ))
    if contract.max_jerk_mps3 > _ABS_MAX_JERK_MPS3:
        errors.append(ContractValidationError(
            "C-004", f"jerk {contract.max_jerk_mps3} > hard limit {_ABS_MAX_JERK_MPS3}"
        ))
    if contract.agent_confidence < _ABS_MIN_CONFIDENCE:
        errors.append(ContractValidationError(
            "C-005", f"confidence {contract.agent_confidence} < floor {_ABS_MIN_CONFIDENCE}"
        ))
    if contract.freshness_required_ms > _ABS_FRESHNESS_MS:
        errors.append(ContractValidationError(
            "C-006", f"freshness_required_ms {contract.freshness_required_ms} > allowed {_ABS_FRESHNESS_MS}"
        ))
    if world.world_age_ms > _ABS_FRESHNESS_MS:
        errors.append(ContractValidationError(
            "C-007", f"world state stale: {world.world_age_ms} ms > {_ABS_FRESHNESS_MS} ms at submission"
        ))

    return errors


def build_contract(
    *,
    frame_id: int,
    sim_time_s: float,
    tactical_intent: str,
    sub_intent_sequence: list,
    target_state_description: str,
    agent_confidence: float,
    agent_reasoning_summary: str,
    target_lane_id: Optional[str] = None,
    max_lateral_offset_m: float = 0.8,
    max_longitudinal_accel_mps2: float = 2.0,
    max_lateral_accel_mps2: float = 1.5,
    max_jerk_mps3: float = 3.0,
    max_duration_s: float = 5.0,
    max_speed_mps: float = 8.33,
    min_ttc_threshold_s: float = 1.5,
    max_new_obstacle_score: float = 0.70,
    min_confidence_floor: float = 0.60,
    revoke_strategy: str = "RETURN_TO_BASELINE",
    cooldown_after_revoke_s: float = 20.0,
    drivable_envelope_uuid: str = "",
    freshness_required_ms: int = 100,
) -> ManeuverContract:
    """
    Convenience builder. Clamps all values to hard limits before creating the contract.
    Use this instead of constructing ManeuverContract directly.
    """
    return ManeuverContract(
        issued_at_frame=frame_id,
        source="stage9_agent",
        tactical_intent=tactical_intent,
        sub_intent_sequence=list(sub_intent_sequence),
        target_state_description=target_state_description,
        max_lateral_offset_m=min(max_lateral_offset_m, _ABS_MAX_LATERAL_OFFSET_M),
        max_longitudinal_accel_mps2=max_longitudinal_accel_mps2,
        max_lateral_accel_mps2=max_lateral_accel_mps2,
        max_jerk_mps3=min(max_jerk_mps3, _ABS_MAX_JERK_MPS3),
        drivable_envelope_uuid=drivable_envelope_uuid,
        target_lane_id=target_lane_id,
        max_duration_s=min(max_duration_s, _ABS_MAX_DURATION_S),
        max_speed_mps=min(max_speed_mps, _ABS_MAX_SPEED_MPS),
        validity_deadline_s=sim_time_s + min(max_duration_s, _ABS_MAX_DURATION_S),
        min_ttc_threshold_s=min_ttc_threshold_s,
        max_new_obstacle_score=max_new_obstacle_score,
        min_confidence_floor=min_confidence_floor,
        planner_feasibility_required=True,
        freshness_required_ms=freshness_required_ms,
        revoke_strategy=revoke_strategy,
        cooldown_after_revoke_s=cooldown_after_revoke_s,
        agent_confidence=agent_confidence,
        agent_reasoning_summary=agent_reasoning_summary,
    )
