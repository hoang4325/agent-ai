"""
Stage 9 — Schemas (v2)
========================
All shared dataclasses, enums, and type definitions used across Stage 9 modules.

Design principle:
  Agent may gain bounded maneuver authority (L1), but never unconditional actuator authority (L3).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────────

class AuthorityState(Enum):
    BASELINE_CONTROL             = auto()
    AGENT_REQUESTING_AUTHORITY   = auto()
    SUPERVISED_EXECUTION         = auto()
    AGENT_ACTIVE_BOUNDED         = auto()
    AUTHORITY_REVOKE_PENDING     = auto()
    TOR_ACTIVE                   = auto()
    HUMAN_CONTROL_ACTIVE         = auto()
    MINIMAL_RISK_MANEUVER        = auto()
    SAFE_STOP                    = auto()


class ActiveAuthority(str, Enum):
    BASELINE = "BASELINE"
    AGENT    = "AGENT"
    HUMAN    = "HUMAN"
    MRM      = "MRM"


class ODDStatus(str, Enum):
    IN_ODD       = "IN_ODD"
    ODD_MARGIN   = "ODD_MARGIN"
    ODD_EXCEEDED = "ODD_EXCEEDED"


class SensorHealth(str, Enum):
    OK                  = "OK"
    DEGRADED_BUT_VALID  = "DEGRADED_BUT_VALID"
    FAULT               = "FAULT"


class VetoSeverity(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


class MRMStrategy(str, Enum):
    COAST_TO_STOP   = "COAST_TO_STOP"
    PULL_OVER_RIGHT = "PULL_OVER_RIGHT"
    EMERGENCY_STOP  = "EMERGENCY_STOP"


# ── Supporting primitives ─────────────────────────────────────────────────────

@dataclass
class Pose2D:
    x: float
    y: float
    yaw_rad: float


@dataclass
class DrivableEnvelope:
    envelope_uuid: str
    left_bound_m: float
    right_bound_m: float
    forward_clear_m: float


# ── WorldState (BEVFusion output — read-only for Stage 9) ────────────────────

@dataclass
class WorldState:
    frame_id: int
    timestamp_s: float
    world_age_ms: int          # age of the world-state data (staleness)
    sync_ok: bool              # all sensors in sync
    sensor_health: SensorHealth

    ego_v_mps: float
    ego_a_mps2: float
    ego_lane_id: str
    ego_lateral_error_m: float
    corridor_clear: bool

    min_ttc_s: float
    new_obstacle_score: float  # [0, 1]
    weather_visibility_m: float
    lane_change_permission: bool
    drivable_envelope: DrivableEnvelope

    odd_status: ODDStatus
    time_to_odd_exit_s: Optional[float]

    preview_feasible: bool
    human_driver_available: bool
    human_input_detected: bool


# ── ManeuverContract (L1 bounded authority, emitted by Stage9AgentAdapter) ───

@dataclass
class ManeuverContract:
    # identity
    contract_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    issued_at_frame: int = 0
    source: str = "stage9_agent"

    # maneuver scope (L1 only — no raw actuator fields)
    tactical_intent: str = "keep_lane"
    sub_intent_sequence: list = field(default_factory=list)
    target_state_description: str = ""

    # spatial / dynamic bounds
    max_lateral_offset_m: float = 0.8
    max_longitudinal_accel_mps2: float = 2.0
    max_lateral_accel_mps2: float = 1.5
    max_jerk_mps3: float = 3.0
    drivable_envelope_uuid: str = ""
    target_lane_id: Optional[str] = None

    # temporal bounds
    max_duration_s: float = 5.0
    max_speed_mps: float = 8.33       # 30 km/h default
    validity_deadline_s: float = 0.0  # absolute sim time

    # safety abort conditions
    min_ttc_threshold_s: float = 1.5
    max_new_obstacle_score: float = 0.70
    min_confidence_floor: float = 0.60
    planner_feasibility_required: bool = True
    freshness_required_ms: int = 100

    # revoke policy
    revoke_strategy: str = "RETURN_TO_BASELINE"   # or "ENTER_MRM"
    cooldown_after_revoke_s: float = 20.0

    # audit
    agent_confidence: float = 0.0
    agent_reasoning_summary: str = ""


# ── TakeoverRequest ───────────────────────────────────────────────────────────

@dataclass
class TakeoverRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "AGENT"          # "AGENT" | "BASELINE" | "SUPERVISOR"
    reason: str = ""
    requested_maneuver: Optional[str] = None
    confidence: Optional[float] = None
    issued_at_frame: int = 0
    validity_s: float = 5.0


# ── AuthorityContext (output of AuthorityStateMachine, read by all planners) ─

@dataclass
class AuthorityContext:
    current_state: AuthorityState = AuthorityState.BASELINE_CONTROL
    active_authority: ActiveAuthority = ActiveAuthority.BASELINE
    active_contract: Optional[ManeuverContract] = None
    contract_remaining_s: Optional[float] = None
    baseline_weight: float = 1.0
    agent_weight: float = 0.0
    revoke_reason: Optional[str] = None
    cooldown_remaining_s: float = 0.0
    tor_active: bool = False
    tor_elapsed_s: Optional[float] = None


# ── TrajectoryRequest (input to MPC — constraint injection from contract) ────

@dataclass
class TrajectoryRequest:
    source: str = "BASELINE"      # "BASELINE" | "CONTRACT_RESOLVER" | "MRM" | "HANDOFF"
    tactical_intent: str = "keep_lane"

    v_max_mps: float = 30.0
    a_long_max_mps2: float = 3.0
    a_lat_max_mps2: float = 2.0
    jerk_max_mps3: float = 4.0
    lateral_bound_m: float = 1.5
    drivable_envelope: Optional[DrivableEnvelope] = None

    target_lane_id: Optional[str] = None
    target_v_desired_mps: float = 10.0
    horizon_s: float = 3.0

    reference_path: Optional[list] = None
    cost_profile: Optional[str] = None  # "BASELINE" | "HANDOFF" | "MRM"

    @staticmethod
    def blend(
        baseline: "TrajectoryRequest",
        candidate: "TrajectoryRequest",
        agent_weight: float,
    ) -> "TrajectoryRequest":
        """
        Blend two TrajectoryRequests at the reference/objective level only.
        Does NOT blend raw actuator values.
        """
        assert 0.0 <= agent_weight <= 1.0
        bw = 1.0 - agent_weight
        return TrajectoryRequest(
            source="HANDOFF",
            tactical_intent=candidate.tactical_intent if agent_weight >= 0.5 else baseline.tactical_intent,
            v_max_mps=bw * baseline.v_max_mps + agent_weight * candidate.v_max_mps,
            a_long_max_mps2=min(baseline.a_long_max_mps2, candidate.a_long_max_mps2),
            a_lat_max_mps2=min(baseline.a_lat_max_mps2, candidate.a_lat_max_mps2),
            jerk_max_mps3=min(baseline.jerk_max_mps3, candidate.jerk_max_mps3),
            lateral_bound_m=bw * baseline.lateral_bound_m + agent_weight * candidate.lateral_bound_m,
            drivable_envelope=baseline.drivable_envelope,
            target_lane_id=candidate.target_lane_id if agent_weight >= 0.5 else baseline.target_lane_id,
            target_v_desired_mps=bw * baseline.target_v_desired_mps + agent_weight * candidate.target_v_desired_mps,
            horizon_s=baseline.horizon_s,
            cost_profile="HANDOFF",
        )


# ── TORSignal ─────────────────────────────────────────────────────────────────

@dataclass
class TORSignal:
    level: int
    started_at_s: float
    elapsed_s: float
    timeout_s: float
    human_input_detected: bool


# ── VetoSignal ────────────────────────────────────────────────────────────────

@dataclass
class VetoSignal:
    reason: str
    severity: VetoSeverity = VetoSeverity.HARD


# ── SafetyVerdict ─────────────────────────────────────────────────────────────

@dataclass
class SafetyVerdict:
    approved: bool = True
    reasons: list = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.approved = False
        self.reasons.append(reason)

    def primary_reason(self) -> str:
        return self.reasons[0] if self.reasons else "unknown"


# ── MRCPlan ───────────────────────────────────────────────────────────────────

@dataclass
class MRCPlan:
    strategy: MRMStrategy = MRMStrategy.COAST_TO_STOP
    target_v_final: float = 0.0
    max_decel_mps2: float = 2.5
    max_jerk_mps3: float = 3.0
    prefer_shoulder: bool = True
    hazard_lights: bool = True
    keep_lane_centering: bool = True
    estimated_stop_distance_m: float = 0.0
    estimated_stop_time_s: float = 0.0
    corridor_for_stop: Optional[str] = None

    def to_trajectory_request(self) -> TrajectoryRequest:
        return TrajectoryRequest(
            source="MRM",
            tactical_intent="safe_stop",
            v_max_mps=0.0,
            a_long_max_mps2=self.max_decel_mps2,
            a_lat_max_mps2=1.0,
            jerk_max_mps3=self.max_jerk_mps3,
            lateral_bound_m=0.5,
            target_v_desired_mps=0.0,
            horizon_s=self.estimated_stop_time_s or 5.0,
            cost_profile="MRM",
        )


# ── ActuatorCommand (output of MPC to CARLA) — Agent never produces this ────

@dataclass
class ActuatorCommand:
    steer: float    # [-1, 1]
    throttle: float # [0, 1]
    brake: float    # [0, 1]
    source: str = "MPC"   # always "MPC" — never "AGENT"
