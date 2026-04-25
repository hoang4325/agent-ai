"""
Stage 8 — Assist Mode Adapter
================================
Wraps Stage 7 AgentShadowAdapter with full assist-mode pipeline:
  1. Calls LLM via Stage 7 adapter to get shadow intent.
  2. Packages intent into a BRP (Behavioral_Recommendation_Packet).
  3. Submits BRP to the ArbitrationEngine (AE).
  4. AE decision → approved intent OR fallback to baseline.
  5. Records BRP audit + veto events + binding advisory.

INVARIANTS (identical to Stage 7, extended):
  - MPC and baseline tactical remain guardian authority.
  - No assist mode changes permitted without new contract review.
  - No takeover. No authority transfer.
  - Every BRP logged with full provenance.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.agent_shadow_adapter import (
    AgentShadowAdapter,
    AgentShadowAdapterConfig,
    AgentShadowIntent,
    ALLOWED_AGENT_INTENTS,
    FORBIDDEN_CONTROL_FIELDS,
)
from benchmark.stage8_arbitration_engine import (
    AEDecision,
    ArbitrationEngine,
    BehavioralRecommendationPacket,
)
from benchmark.stage8_temporal_memory import TemporalMemoryStore
from benchmark.io import dump_json, dump_jsonl


# ── Per-frame assist result ────────────────────────────────────────────────────

@dataclass
class AssistFrameRecord:
    case_id: str
    frame_id: int
    shadow_mode: bool                   # Always True
    # Agent shadow intent (from Stage 7 adapter)
    shadow_tactical_intent: str
    shadow_confidence: float
    shadow_agrees_with_baseline: bool
    shadow_fallback: bool
    shadow_timeout: bool
    shadow_validation_status: str
    # BRP
    brp_packet_id: str
    # AE Decision
    ae_decision: str                    # approved | vetoed | silent_discard
    ae_veto_code: str | None
    ae_veto_name: str | None
    ae_approved_intent: str | None      # Only set when approved
    # Binding advisory
    binding_advisory_intent: str        # Final intent used by baseline
    binding_advisory_source: str        # "agent" | "baseline_fallback"
    # Misc
    baseline_intent: str
    reason_tags: list[str]
    call_latency_ms: float | None
    ae_notes: list[str]
    memory_active_maneuver: str | None
    memory_committed: bool
    consecutive_hard_vetoes: int
    assist_disabled: bool


# ── Assist Adapter ─────────────────────────────────────────────────────────────

class Stage8AssistAdapter:
    """
    Full Stage 8 assist pipeline for one case.
    Instantiate fresh per case. Call `step()` per frame.
    """

    def __init__(
        self,
        shadow_adapter: AgentShadowAdapter,
        case_id: str,
        memory_window: int = 5,
    ):
        self._shadow = shadow_adapter
        self._case_id = case_id
        self._memory = TemporalMemoryStore(case_id=case_id, window_size=memory_window)
        self._ae = ArbitrationEngine(self._memory)

        self._brp_audit: list[dict[str, Any]] = []
        self._veto_events: list[dict[str, Any]] = []
        self._binding_advisory: list[dict[str, Any]] = []
        self._frame_records: list[dict[str, Any]] = []

        # 10Hz sim -> 10 frames = 1 call per second (1Hz)
        self._llm_downsample_ratio: int = 10
        self._frames_since_call: int = 0
        self._cached_shadow_intent: AgentShadowIntent | None = None

    def step(
        self,
        frame_id: int,
        ego_state: dict,
        tracked_objects: list,
        lane_context: dict,
        route_context: dict,
        stop_context: dict,
        baseline_context: dict,
        safety_context: dict | None = None,
    ) -> AssistFrameRecord:
        """
        Execute one frame of the assist pipeline.

        safety_context: {
            lane_change_permission: bool,
            stop_binding_active: bool,
            estimated_ttc_s: float | None,
        }
        """
        safety_context = safety_context or {}

        # Inject memory context into baseline context for LLM prompt
        baseline_context_injected = dict(baseline_context)
        baseline_context_injected["active_maneuver"] = self._memory.active_maneuver

        # ── Fast Cache Invalidation ───────────────────────────────────────────
        # If the cached intent was a lane change, but permission was just revoked mid-cycle,
        # we MUST invalidate the cache and force the LLM to recalculate to avoid a stale V-003 hard veto.
        if self._cached_shadow_intent and self._cached_shadow_intent.tactical_intent.endswith(("_left", "_right")):
            if not safety_context.get("lane_change_permission", True):
                self._cached_shadow_intent = None

        # ── Step 1: Get shadow intent via Stage 7 adapter ─────────────────────
        # Temporal Downsampling (1Hz)
        call_llm = (self._frames_since_call % self._llm_downsample_ratio == 0) or (self._cached_shadow_intent is None)

        if call_llm:
            shadow_intent: AgentShadowIntent = self._shadow.call(
                case_id=self._case_id,
                frame_id=frame_id,
                ego_state=ego_state,
                tracked_objects=tracked_objects,
                lane_context=lane_context,
                route_context=route_context,
                stop_context=stop_context,
                baseline_context=baseline_context_injected,
            )
            self._cached_shadow_intent = shadow_intent
        else:
            import copy
            shadow_intent = copy.deepcopy(self._cached_shadow_intent)
            shadow_intent.frame_id = frame_id
            if shadow_intent.provenance:
                shadow_intent.provenance["call_latency_ms"] = 0.0
                shadow_intent.provenance["cached_frame"] = True

        self._frames_since_call += 1
        shadow_dict = asdict(shadow_intent)

        # Derive safety context from shadow+lane context if not provided externally
        if not safety_context:
            lc_perm = (lane_context.get("lane_change_permission") or {})
            if isinstance(lc_perm, dict):
                perm_left = lc_perm.get("left", False)
                perm_right = lc_perm.get("right", False)
                safety_context["lane_change_permission"] = perm_left or perm_right
            else:
                safety_context["lane_change_permission"] = bool(lc_perm)
            safety_context["stop_binding_active"] = (
                (stop_context.get("binding_status") or "") == "active"
            )
            safety_context["estimated_ttc_s"] = None

        # ── Step 2: Build BRP ─────────────────────────────────────────────────
        mem_snap = self._memory.snapshot()
        brp = BehavioralRecommendationPacket(
            packet_id=str(uuid.uuid4()),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            case_id=self._case_id,
            frame_id=frame_id,
            shadow_mode=True,
            agent_tactical_intent=shadow_intent.tactical_intent,
            agent_target_lane=shadow_intent.target_lane,
            agent_confidence=shadow_intent.confidence,
            reason_tags=list(shadow_intent.reason_tags or []),
            temporal_context_frames=len(mem_snap.frames),
            memory_snapshot_id=mem_snap.memory_id,
            veto_eligible=True,
            binding_request=True,
            assist_scope="tactical_only",
            authority_assertion="mpc_and_baseline_remain_guardian",
            source_frame_record=shadow_dict,
        )

        # ── Step 2.5: Physical Maneuver Completion Check ──────────────────────
        # If we were committed to a lane change, but the physical environment report 
        # that lane change permission is now False (e.g. ego entered the target lane),
        # we consider the maneuver organically completed and MUST clear the memory.
        if self._memory.active_maneuver and self._memory.maneuver_committed:
            man_dir = "right" if "right" in self._memory.active_maneuver else "left"
            if not safety_context.get("lane_change_permission", True):
                self._memory.reset(reason="maneuver_physically_completed")

        # ── Step 3: Arbitrate ─────────────────────────────────────────────────
        ae_dec: AEDecision = self._ae.arbitrate(
            brp=brp,
            safety_context=safety_context,
            current_frame_id=frame_id,
        )

        # ── Step 4: Determine binding advisory ───────────────────────────────
        if ae_dec.decision == "approved" and ae_dec.approved_intent:
            binding_intent = ae_dec.approved_intent
            binding_source = "agent"
        else:
            binding_intent = shadow_intent.baseline_intent or "keep_lane"
            binding_source = "baseline_fallback"

        # ── Step 5: Update temporal memory ────────────────────────────────────
        self._memory.record(
            frame_id=frame_id,
            agent_intent=shadow_intent.tactical_intent,
            approved=(ae_dec.decision == "approved"),
            veto_code=ae_dec.veto_code,
            confidence=shadow_intent.confidence,
            agrees_with_baseline=shadow_intent.agrees_with_baseline,
        )

        # ── Step 6: Build frame record ────────────────────────────────────────
        prov = shadow_intent.provenance or {}
        record = AssistFrameRecord(
            case_id=self._case_id,
            frame_id=frame_id,
            shadow_mode=True,
            shadow_tactical_intent=shadow_intent.tactical_intent,
            shadow_confidence=shadow_intent.confidence,
            shadow_agrees_with_baseline=shadow_intent.agrees_with_baseline,
            shadow_fallback=shadow_intent.fallback_to_baseline,
            shadow_timeout=shadow_intent.timeout_flag,
            shadow_validation_status=shadow_intent.validation_status,
            brp_packet_id=brp.packet_id,
            ae_decision=ae_dec.decision,
            ae_veto_code=ae_dec.veto_code,
            ae_veto_name=ae_dec.veto_name,
            ae_approved_intent=ae_dec.approved_intent,
            binding_advisory_intent=binding_intent,
            binding_advisory_source=binding_source,
            baseline_intent=shadow_intent.baseline_intent,
            reason_tags=list(shadow_intent.reason_tags or []),
            call_latency_ms=prov.get("call_latency_ms"),
            ae_notes=list(ae_dec.ae_notes),
            memory_active_maneuver=self._memory.active_maneuver,
            memory_committed=self._memory.maneuver_committed,
            consecutive_hard_vetoes=ae_dec.consecutive_hard_vetoes,
            assist_disabled=ae_dec.assist_disabled,
        )
        rec_dict = asdict(record)
        self._frame_records.append(rec_dict)

        # ── Step 7: Audit logs ────────────────────────────────────────────────
        self._brp_audit.append({
            "packet_id": brp.packet_id,
            "frame_id": frame_id,
            "case_id": self._case_id,
            "agent_intent": brp.agent_tactical_intent,
            "confidence": brp.agent_confidence,
            "assist_scope": brp.assist_scope,
            "shadow_mode": brp.shadow_mode,
            "ae_decision": ae_dec.decision,
            "veto_code": ae_dec.veto_code,
            "timestamp_utc": brp.timestamp_utc,
        })

        if ae_dec.veto_code:
            self._veto_events.append({
                "frame_id": frame_id,
                "case_id": self._case_id,
                "packet_id": brp.packet_id,
                "veto_code": ae_dec.veto_code,
                "veto_name": ae_dec.veto_name,
                "veto_severity": ae_dec.veto_severity,
                "ae_notes": ae_dec.ae_notes,
                "agent_intent": brp.agent_tactical_intent,
                "fallback_intent": binding_intent,
                "timestamp_utc": ae_dec.timestamp_utc,
            })

        if ae_dec.decision == "approved":
            self._binding_advisory.append({
                "frame_id": frame_id,
                "case_id": self._case_id,
                "packet_id": brp.packet_id,
                "binding_intent": binding_intent,
                "agent_confidence": brp.agent_confidence,
                "memory_active_maneuver": self._memory.active_maneuver,
                "timestamp_utc": ae_dec.timestamp_utc,
            })

        return record

    def ae_stats(self) -> dict[str, Any]:
        return self._ae.stats()

    def materialize_artifacts(self, output_dir: Path) -> dict[str, Path]:
        """Write all 4 Stage 8 mandatory artifact files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        brp_path = output_dir / "stage8_brp_audit.jsonl"
        veto_path = output_dir / "stage8_veto_events.jsonl"
        bind_path = output_dir / "stage8_binding_advisory.jsonl"
        mem_path  = output_dir / "agent_temporal_memory.json"

        dump_jsonl(brp_path, self._brp_audit)
        dump_jsonl(veto_path, self._veto_events)
        dump_jsonl(bind_path, self._binding_advisory)
        self._memory.save(mem_path)

        return {
            "stage8_brp_audit": brp_path,
            "stage8_veto_events": veto_path,
            "stage8_binding_advisory": bind_path,
            "agent_temporal_memory": mem_path,
        }

    @property
    def frame_records(self) -> list[dict[str, Any]]:
        return self._frame_records
