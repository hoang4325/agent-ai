# Stage 8 — Agentic Tactical Assist: Sandbox Design
**Version:** 0.1.0 — 2026-04-18
**Author:** Antigravity Engineering
**Status:** DESIGN LOCKED — pending canary run

---

## Invariants (Non-negotiable — identical to Stage 7)

- Agent binds ONLY at tactical layer. No trajectory, no MPC, no control surface.
- Baseline tactical + MPC remain guardian authority at all times.
- Every agent recommendation MUST pass veto/arbitration before binding.
- No takeover. No assist-initiated controller handoff. No authority transfer.
- Shadow mode remains available as fallback at any moment.

---

## Part 1 — Behavioral_Recommendation_Packet (BRP)

The BRP is the ONLY communication channel from the agent to the baseline layer.
No other field, side-effect, or memory reference is permitted.

### Schema

```json
{
  "schema_version": "brp_v1",
  "packet_id": "<uuid4>",
  "timestamp_utc": "<ISO-8601>",
  "case_id": "<str>",
  "frame_id": "<int>",
  "shadow_mode": true,
  "agent_tactical_intent": "<AllowedIntent>",
  "agent_target_lane": "<str | null>",
  "agent_confidence": "<float 0.0–1.0>",
  "reason_tags": ["<str>", "..."],
  "temporal_context_frames": "<int>",
  "memory_snapshot_id": "<str | null>",
  "veto_eligible": true,
  "binding_request": true,
  "assist_scope": "tactical_only",
  "forbidden_fields": [],
  "authority_assertion": "mpc_and_baseline_remain_guardian"
}
```

### Rules
1. `binding_request=true` triggers the Arbitration Engine (AE). If AE vetoes, BRP is downgraded to advisory-only (logged but not executed).
2. Fields `target_speed_mps`, `steering`, `throttle`, `brake`, `trajectory` are **schema-banned** in BRP.
3. `agent_confidence < 0.70` → BRP is flagged for soft audit; AE may still approve.
4. `agent_confidence < 0.50` → BRP is automatically vetoed (VETO_LOW_CONFIDENCE).
5. BRP TTL (time-to-live) is 1 frame. Stale BRPs are discarded by AE.

---

## Part 2 — Veto Taxonomy

### Veto Categories

| Code | Name | Trigger Condition | Handler |
|------|------|-------------------|---------|
| `V-001` | `VETO_FORBIDDEN_FIELD` | BRP contains any of `target_speed_mps`, `steering`, `throttle`, `brake`, `trajectory` | Hard reject + audit entry |
| `V-002` | `VETO_INVALID_INTENT` | `agent_tactical_intent` not in AllowedIntents | Hard reject + audit entry |
| `V-003` | `VETO_NO_LC_PERMISSION` | LC intent but `lane_change_permission=False` from Safety Spine | Hard reject |
| `V-004` | `VETO_COLLISION_RISK` | MPC geometry check detects unsafe gap (<2.5s TTC) in target lane | Hard reject |
| `V-005` | `VETO_LOW_CONFIDENCE` | `agent_confidence < 0.50` | Auto-reject, no AE review |
| `V-006` | `VETO_TEMPORAL_CONFLICT` | Intent contradicts memory window (VD: cancel committed LC mid-execution) | Soft reject, escalate to MPC |
| `V-007` | `VETO_STOP_OVERRIDE_ATTEMPT` | Agent recommends non-stop intent when Stop Spine is binding | Hard reject |
| `V-008` | `VETO_STALE_BRP` | BRP `frame_id` older than current frame | Silent discard |
| `V-009` | `VETO_AUTHORITY_SCOPE_EXCEEDED` | BRP `assist_scope` ≠ `tactical_only` | Hard reject + RED alert |
| `V-010` | `VETO_RATE_LIMIT` | More than 1 BRP per frame submitted | Silent discard of extras |

### Arbitration Engine (AE) Logic
```
RECEIVE BRP
│
├── Check forbidden fields → V-001 (hard reject)
├── Validate intent whitelist → V-002 (hard reject)
├── Check confidence threshold → V-005 (auto-reject if < 0.50)
├── Consult Safety Spine:
│   ├── LC permission check → V-003
│   ├── Stop binding check → V-007
│   └── TTC geometry check → V-004
├── Consult Temporal Memory:
│   └── Conflict with history window → V-006
├── Validate assist_scope → V-009
│
├── ALL CHECKS PASS → APPROVE BRP → emit binding advisory to MPC
│                     log: BindingAdvisoryEvent
│
└── ANY CHECK FAILS → VETO BRP → fallback to baseline intent
                      log: VetoEvent(code, reason, frame_id)
```

---

## Part 3 — Temporal Memory (3–5 frames)

### Purpose
Prevent intent thrashing (e.g., `prepare_lc_right` → `keep_lane` → `prepare_lc_right` every other frame) and give AE contextual awareness of ongoing maneuver phases.

### Memory Structure

```json
{
  "memory_id": "<str>",
  "case_id": "<str>",
  "window_size": 5,
  "frames": [
    {
      "frame_id": 60010,
      "agent_intent": "prepare_lane_change_right",
      "approved": true,
      "veto_code": null
    },
    {
      "frame_id": 60011,
      "agent_intent": "commit_lane_change_right",
      "approved": true,
      "veto_code": null
    }
  ],
  "active_maneuver": "lane_change_right",
  "maneuver_frame_start": 60010,
  "maneuver_committed": true
}
```

### Rules
1. Window = last N frames (N configurable: `stage8_memory_window_size = 5`).
2. AE reads `active_maneuver` before arbitration. If `maneuver_committed=True`, a contradictory intent triggers `V-006`.
3. A `VETO`ed intent still enters memory (for audit fidelity) with `approved=false`.
4. Memory resets on case boundary, on explicit `stop` completion, and on emergency brake event.
5. Memory is NOT exposed to the LLM prompt — agent must re-infer context from raw state each frame (no context injection bias).

---

## Part 4 — Rollback / Authority Hygiene

### Rollback Triggers
Any of the following causes an immediate rollback to baseline authority, memory reset, and BRP pipeline flush:

| Trigger | Code | Action |
|---------|------|--------|
| `V-001` to `V-010` veto (hard reject) | `ROLLBACK_VETO` | Revert to baseline intent; log |
| LLM timeout (>10s) | `ROLLBACK_TIMEOUT` | Revert to baseline; increment timeout counter |
| 3 consecutive vetoes (any code) | `ROLLBACK_STREAK` | Disable assist mode for 10 frames; alert |
| Forbidden field detected | `ROLLBACK_CONTRACT_BREACH` | HARD STOP assist mode; audit RED FLAG |
| AE crash / exception | `ROLLBACK_AE_FAULT` | Emergency baseline; alert ops |

### Authority Hygiene Protocol
1. **No authority bleed:** Baseline MPC never reads agent memory directly. It receives only approved intents via the BRP binding advisory channel.
2. **Audit trail:** Every BRP (approved or vetoed) is logged to `stage8_brp_audit.jsonl` with full provenance.
3. **Canary kill switch:** Any run can be aborted mid-frame via `--kill-assist` flag, reverting all frames to baseline.
4. **No persistent state leak:** At end of each case, memory is serialized to `agent_temporal_memory.json` and wiped from runtime.

---

## Canary Case Selection

**Chosen:** `ml_right_positive_core`

### Rationale
- Stage 7 real API run: Disagreement = 3.3% (2/60 frames). Low disagreement means AE will rarely be stressed — ideal for first assist binding.
- Case involves a clean, well-lit lane change scenario with unambiguous route context. No stop spine binding. No ambiguous yielding.
- Safety profile: No tight TTC constraints. Gap geometry is predictable.
- All 60 frames already have real LLM intents recorded. Allows replaying assist mode against known ground truth.

**Rejection list:**
- `stop_follow_ambiguity_core`: Disagreement = 26.6%. Too many veto triggers in first assist sandbox.
- `adjacent_right_unsafe_hold`: Risk window for adjacent-unsafe detection — AE geometry checks not yet stress-tested.

---

## Final Verdict

| Question | Answer |
|---|---|
| **Đã đủ mở Stage 8 assist sandbox chưa?** | **CÓ.** Stage 7 real API đã Pass 2/2 case, 60 frames, 0 timeout, 0 fallback, contract validity 100%. Đủ điều kiện binding advisory. |
| **Trên case nào trước?** | `ml_right_positive_core`. Disagreement thấp (3.3%), scenario sạch, không có stop binding. |
| **Blocker còn lại là gì?** | (1) Chưa có `ArbitrationEngine` implementation. (2) Chưa có `TemporalMemoryStore`. (3) BRP channel chưa được nối dây vào MPC input pipeline. (4) Stage 8 gate metrics chưa được định nghĩa đầy đủ. |
| **Vì sao chưa phải takeover?** | Takeover nghĩa là LLM **thay thế** MPC làm bộ điều khiển chính. Assist nghĩa là LLM **gợi ý** còn MPC **quyết định**. Khi AE còn toàn quyền veto, khi MPC còn là guardian, khi BRP TTL chỉ 1 frame — đó là Assist, không phải Takeover. Takeover đòi hỏi Stage 9 với: stability campaign ≥ 500 frames, formal safety verification, và AE approval rate ≥ 95% qua nhiều run độc lập. |
