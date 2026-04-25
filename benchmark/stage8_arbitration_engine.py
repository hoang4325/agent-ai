"""
Stage 8 — Arbitration Engine (AE)
===================================
The Arbitration Engine is the SOLE authority that decides whether an agent
Behavioral_Recommendation_Packet (BRP) is approved for binding advisory
or vetoed and rolled back to baseline.

VETO CODES (from stage8_assist_veto_taxonomy.json):
  V-001  VETO_FORBIDDEN_FIELD          Hard  — contract breach, disable assist
  V-002  VETO_INVALID_INTENT           Hard
  V-003  VETO_NO_LC_PERMISSION         Hard
  V-004  VETO_COLLISION_RISK           Hard
  V-005  VETO_LOW_CONFIDENCE           Auto  — no AE review needed
  V-006  VETO_TEMPORAL_CONFLICT        Soft  — escalate to MPC
  V-007  VETO_STOP_OVERRIDE_ATTEMPT    Hard
  V-008  VETO_STALE_BRP                Silent discard
  V-009  VETO_AUTHORITY_SCOPE_EXCEEDED Hard  — contract breach, disable assist
  V-010  VETO_RATE_LIMIT               Silent discard

INVARIANTS:
  - AE has unconditional veto authority.
  - AE veto overrides agent confidence.
  - MPC guardian authority is never weakened by AE approval.
  - 3 consecutive hard vetoes → assist disabled for 10 frames.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from benchmark.agent_shadow_adapter import ALLOWED_AGENT_INTENTS, FORBIDDEN_CONTROL_FIELDS
from benchmark.stage8_temporal_memory import TemporalMemoryStore

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

AE_MIN_CONFIDENCE = 0.50          # Below this → V-005 auto-veto
AE_SOFT_AUDIT_CONFIDENCE = 0.70   # Below this → soft audit flag
AE_CONSECUTIVE_VETO_LIMIT = 3     # Disable assist after N consecutive hard vetoes
AE_DISABLE_DURATION_FRAMES = 10   # Frames to disable assist after streak
AE_COLLISION_TTC_THRESHOLD_S = 2.5  # TTC threshold for V-004

LC_INTENTS = frozenset({
    "prepare_lane_change_left", "prepare_lane_change_right",
    "commit_lane_change_left",  "commit_lane_change_right",
})


# ── BRP Dataclass ─────────────────────────────────────────────────────────────

@dataclass
class BehavioralRecommendationPacket:
    """
    The ONLY communication channel from the agent to the baseline layer.
    All fields are validated by AE before any binding advisory is emitted.
    """
    packet_id: str
    timestamp_utc: str
    case_id: str
    frame_id: int
    shadow_mode: bool                    # MUST be True
    agent_tactical_intent: str
    agent_target_lane: str | None
    agent_confidence: float
    reason_tags: list[str]
    temporal_context_frames: int
    memory_snapshot_id: str | None
    veto_eligible: bool                  # MUST be True
    binding_request: bool                # MUST be True
    assist_scope: str                    # MUST be "tactical_only"
    authority_assertion: str             # informational
    # Source shadow intent record for traceability
    source_frame_record: dict[str, Any] = field(default_factory=dict)


# ── AE Decision Record ────────────────────────────────────────────────────────

@dataclass
class AEDecision:
    packet_id: str
    frame_id: int
    case_id: str
    decision: str                        # "approved" | "vetoed" | "silent_discard"
    veto_code: str | None
    veto_name: str | None
    veto_severity: str | None           # "hard" | "soft" | "auto" | "silent"
    approved_intent: str | None
    baseline_fallback_intent: str | None
    consecutive_hard_vetoes: int
    assist_disabled: bool
    assist_disabled_remaining_frames: int
    soft_audit_flag: bool
    ae_notes: list[str]
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Arbitration Engine ────────────────────────────────────────────────────────

class ArbitrationEngine:
    """
    Single-instance AE per assist run.
    Call `arbitrate(brp, context)` once per frame.
    """

    def __init__(self, memory: TemporalMemoryStore):
        self._memory = memory
        self._consecutive_hard_vetoes = 0
        self._assist_disabled_frames_remaining = 0
        self._total_brps = 0
        self._total_approved = 0
        self._total_vetoed = 0
        self._total_discarded = 0
        self._contract_breach_count = 0
        self._veto_counts: dict[str, int] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def arbitrate(
        self,
        brp: BehavioralRecommendationPacket,
        safety_context: dict[str, Any],
        current_frame_id: int,
    ) -> AEDecision:
        """
        Main entry point. Returns AEDecision with full audit fields.
        safety_context must include:
          - lane_change_permission: bool
          - stop_binding_active: bool
          - estimated_ttc_s: float | None
        """
        self._total_brps += 1
        notes: list[str] = []

        # ── Assist disabled check ──────────────────────────────────────────────
        if self._assist_disabled_frames_remaining > 0:
            self._assist_disabled_frames_remaining -= 1
            self._total_discarded += 1
            return self._make_decision(
                brp, "silent_discard", None, None, "silent",
                f"assist_disabled ({self._assist_disabled_frames_remaining + 1} frames remaining)",
                notes, disabled=True,
            )

        # ── V-008: Stale BRP ──────────────────────────────────────────────────
        if brp.frame_id < current_frame_id:
            self._total_discarded += 1
            return self._make_decision(
                brp, "silent_discard", "V-008", "VETO_STALE_BRP", "silent",
                f"stale BRP frame_id={brp.frame_id} < current={current_frame_id}",
                notes,
            )

        # ── V-010: Rate limit (duplicate) ─────────────────────────────────────
        # (Caller is responsible for single BRP per frame; AE records silently)

        # ── V-001: Forbidden fields ───────────────────────────────────────────
        forbidden_found = [
            f for f in FORBIDDEN_CONTROL_FIELDS
            if f in brp.source_frame_record
        ]
        if forbidden_found:
            return self._hard_veto(
                brp, "V-001", "VETO_FORBIDDEN_FIELD",
                f"forbidden fields in BRP: {forbidden_found}",
                notes, contract_breach=True,
            )

        # ── V-009: Authority scope exceeded ──────────────────────────────────
        if brp.assist_scope != "tactical_only":
            return self._hard_veto(
                brp, "V-009", "VETO_AUTHORITY_SCOPE_EXCEEDED",
                f"assist_scope={brp.assist_scope!r} must be 'tactical_only'",
                notes, contract_breach=True,
            )

        # ── shadow_mode integrity check ───────────────────────────────────────
        if not brp.shadow_mode:
            return self._hard_veto(
                brp, "V-009", "VETO_AUTHORITY_SCOPE_EXCEEDED",
                "shadow_mode must be True in BRP",
                notes, contract_breach=True,
            )

        # ── V-002: Invalid intent ─────────────────────────────────────────────
        if brp.agent_tactical_intent not in ALLOWED_AGENT_INTENTS:
            return self._hard_veto(
                brp, "V-002", "VETO_INVALID_INTENT",
                f"intent {brp.agent_tactical_intent!r} not in whitelist",
                notes,
            )

        # ── V-005: Low confidence (auto-reject, no further AE review) ─────────
        if brp.agent_confidence < AE_MIN_CONFIDENCE:
            self._consecutive_hard_vetoes += 1
            self._total_vetoed += 1
            self._veto_counts["V-005"] = self._veto_counts.get("V-005", 0) + 1
            return self._make_decision(
                brp, "vetoed", "V-005", "VETO_LOW_CONFIDENCE", "auto",
                f"confidence {brp.agent_confidence:.2f} < {AE_MIN_CONFIDENCE}",
                notes,
            )

        # ── V-007: Stop override attempt ──────────────────────────────────────
        stop_binding = safety_context.get("stop_binding_active", False)
        if stop_binding and brp.agent_tactical_intent not in {"stop", "yield"}:
            return self._hard_veto(
                brp, "V-007", "VETO_STOP_OVERRIDE_ATTEMPT",
                f"stop spine binding but agent recommends {brp.agent_tactical_intent!r}",
                notes,
            )

        # ── V-003: No LC permission ───────────────────────────────────────────
        lc_perm = safety_context.get("lane_change_permission", True)
        if brp.agent_tactical_intent in LC_INTENTS and not lc_perm:
            return self._hard_veto(
                brp, "V-003", "VETO_NO_LC_PERMISSION",
                f"LC intent {brp.agent_tactical_intent!r} but lane_change_permission=False",
                notes,
            )

        # ── V-004: Collision risk ─────────────────────────────────────────────
        ttc = safety_context.get("estimated_ttc_s")
        if ttc is not None and ttc < AE_COLLISION_TTC_THRESHOLD_S:
            if brp.agent_tactical_intent in LC_INTENTS:
                return self._hard_veto(
                    brp, "V-004", "VETO_COLLISION_RISK",
                    f"TTC={ttc:.2f}s < {AE_COLLISION_TTC_THRESHOLD_S}s for LC intent",
                    notes,
                )

        # ── V-006: Temporal conflict ──────────────────────────────────────────
        if self._memory.detect_temporal_conflict(brp.agent_tactical_intent):
            self._total_vetoed += 1
            self._veto_counts["V-006"] = self._veto_counts.get("V-006", 0) + 1
            # V-006 is a SOFT veto. It does NOT increment consecutive_hard_vetoes.
            return self._make_decision(
                brp, "vetoed", "V-006", "VETO_TEMPORAL_CONFLICT", "soft",
                f"intent {brp.agent_tactical_intent!r} conflicts with committed maneuver "
                f"{self._memory.active_maneuver!r}",
                notes,
            )

        # ── Soft audit flag ───────────────────────────────────────────────────
        soft_audit = brp.agent_confidence < AE_SOFT_AUDIT_CONFIDENCE
        if soft_audit:
            notes.append(f"soft_audit: confidence {brp.agent_confidence:.2f} < {AE_SOFT_AUDIT_CONFIDENCE}")

        # ── APPROVED ──────────────────────────────────────────────────────────
        self._consecutive_hard_vetoes = 0
        self._total_approved += 1

        return AEDecision(
            packet_id=brp.packet_id,
            frame_id=brp.frame_id,
            case_id=brp.case_id,
            decision="approved",
            veto_code=None,
            veto_name=None,
            veto_severity=None,
            approved_intent=brp.agent_tactical_intent,
            baseline_fallback_intent=None,
            consecutive_hard_vetoes=0,
            assist_disabled=False,
            assist_disabled_remaining_frames=0,
            soft_audit_flag=soft_audit,
            ae_notes=notes,
        )

    def stats(self) -> dict[str, Any]:
        total = max(self._total_brps, 1)
        return {
            "total_brps": self._total_brps,
            "total_approved": self._total_approved,
            "total_vetoed": self._total_vetoed,
            "total_discarded": self._total_discarded,
            "ae_approval_rate": round(self._total_approved / total, 4),
            "ae_veto_rate": round(self._total_vetoed / total, 4),
            "contract_breach_count": self._contract_breach_count,
            "veto_counts_by_code": dict(self._veto_counts),
            "assist_disabled_frames_remaining": self._assist_disabled_frames_remaining,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _hard_veto(
        self, brp: BehavioralRecommendationPacket,
        code: str, name: str, reason: str,
        notes: list[str], contract_breach: bool = False,
    ) -> AEDecision:
        self._consecutive_hard_vetoes += 1
        self._total_vetoed += 1
        self._veto_counts[code] = self._veto_counts.get(code, 0) + 1
        if contract_breach:
            self._contract_breach_count += 1
            self._assist_disabled_frames_remaining = AE_DISABLE_DURATION_FRAMES
            logger.error(
                "[AE] CONTRACT BREACH %s frame=%s: %s", code, brp.frame_id, reason
            )
        else:
            logger.warning("[AE] VETO %s frame=%s: %s", code, brp.frame_id, reason)
        self._check_streak()
        return self._make_decision(brp, "vetoed", code, name, "hard", reason, notes,
                                   disabled=contract_breach)

    def _check_streak(self) -> None:
        if self._consecutive_hard_vetoes >= AE_CONSECUTIVE_VETO_LIMIT:
            self._assist_disabled_frames_remaining = AE_DISABLE_DURATION_FRAMES
            logger.warning(
                "[AE] %d consecutive vetoes — assist disabled for %d frames",
                self._consecutive_hard_vetoes, AE_DISABLE_DURATION_FRAMES,
            )

    def _make_decision(
        self,
        brp: BehavioralRecommendationPacket,
        decision: str,
        code: str | None,
        name: str | None,
        severity: str | None,
        reason: str,
        notes: list[str],
        disabled: bool = False,
    ) -> AEDecision:
        if code:
            notes.append(f"{code}({name}): {reason}")
        return AEDecision(
            packet_id=brp.packet_id,
            frame_id=brp.frame_id,
            case_id=brp.case_id,
            decision=decision,
            veto_code=code,
            veto_name=name,
            veto_severity=severity,
            approved_intent=None,
            baseline_fallback_intent=brp.source_frame_record.get("baseline_intent"),
            consecutive_hard_vetoes=self._consecutive_hard_vetoes,
            assist_disabled=self._assist_disabled_frames_remaining > 0 or disabled,
            assist_disabled_remaining_frames=self._assist_disabled_frames_remaining,
            soft_audit_flag=False,
            ae_notes=notes,
        )
