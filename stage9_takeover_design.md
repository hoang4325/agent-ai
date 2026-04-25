# Stage 9 — Takeover Design
## Autonomous Driving Agentic Authority Protocol
**Role:** Principal Autonomous Driving Architect
**Stack:** BEVFusion → Agentic AI → MPC → CARLA
**Core Invariant:** _"Agent may gain bounded maneuver authority, but never unconditional actuator authority."_

---

## 1. Kiến Trúc Tổng Thể

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SENSOR FUSION LAYER                              │
│   Camera + LiDAR + Radar ──► BEVFusion ──► World State (OGM, V, A)  │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ world_state_t
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    AUTHORITY ORCHESTRATION LAYER                     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Authority State Machine (ASM)                   │    │
│  │  State: BASELINE_CONTROL | AGENT_REQUESTING | AGENT_ACTIVE   │    │
│  │         SUPERVISED_EXEC  | REVOKE_PENDING   | MRM | TOR     │    │
│  └──────┬──────────────────────────────────────────────────────┘    │
│         │ authority_context_t                                        │
│         ├──────────────────┬──────────────────┐                     │
│         ▼                  ▼                  ▼                     │
│  ┌────────────┐  ┌──────────────────┐  ┌──────────────┐            │
│  │  Baseline  │  │   Agentic AI     │  │   Safety     │            │
│  │  Planner   │  │  (LLM + Stage8)  │  │  Supervisor  │            │
│  │  (default) │  │  Stage 9 Agent   │  │  (veto/ok)   │            │
│  └────────────┘  └──────────────────┘  └──────────────┘            │
│         │ maneuver_intent_t  │ ManeuverContract  │ SafetyVerdict    │
└─────────┼────────────────────┼───────────────────┼──────────────────┘
          ▼ winning intent     ▼ bounded contract   ▼ gate decision
┌──────────────────────────────────────────────────────────────────────┐
│                   TRAJECTORY SYNTHESIS LAYER                         │
│                                                                      │
│   Contract Resolver → TrajectoryRequest → MPC Optimizer              │
│               (hard constraints injected from approved contract)     │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ u(t): steer, throttle, brake
                                 ▼
                            CARLA Actuators
```

### 3 tầng quyền lực — định nghĩa tuyệt đối

| Tầng | Tên | Ai giữ | Giải thích |
|------|-----|--------|-----------|
| L1 | **Maneuver Authority** | ASM cấp có thời hạn cho Agent | Quyền chọn hành vi chiến thuật: `keep_lane`, `commit_lc_right`, [abort](file:///d:/Agent-AI/benchmark/metrics.py#221-228) |
| L2 | **Trajectory Authority** | MPC luôn giữ | Quyền tính quỹ đạo tối ưu từ intent L1 |
| L3 | **Actuator Authority** | MPC → CARLA, không ai khác | Quyền xuất `steer`, `throttle`, `brake` thực tế |

> Agent chỉ được L1, không bao giờ được L2 hoặc L3 vô điều kiện.

---

## 2. Các Module Cần Có

### Module bắt buộc (triển khai Phase 1)

```
stage9/
├── authority_state_machine.py   # ASM: 8 states, transitions, timers
├── maneuver_contract.py         # Contract dataclass + validator
├── safety_supervisor.py         # Hard-constraint checker, veto engine
├── authority_arbiter.py         # Main loop: 10Hz orchestration
├── baseline_detector.py         # Phát hiện khi Baseline đang stuck/lúng túng
├── contract_resolver.py         # Translate ManeuverContract → TrajectoryRequest
└── schemas.py                   # Tất cả dataclass/TypedDict interface
```

### Module supporting (Phase 2)

```
stage9/
├── tor_manager.py               # TOR timer, alert escalation, human response check
├── minimal_risk_maneuver.py     # MRM executor (đường tấp lề, dừng an toàn)
├── authority_logger.py          # Log mọi event: grant/revoke/abort + lý do
└── stage9_evaluator.py          # Metrics: takeover success rate, stuck rate, etc.
```

---

## 3. State Machine Chi Tiết

### 8 trạng thái

```
 ┌─────────────────────────────────────────────────────────┐
 │          AUTHORITY STATE MACHINE — Stage 9               │
 │                                                          │
 │  [BASELINE_CONTROL] ◄──────────────────────────────┐   │
 │         │                                           │   │
 │   agent detects baseline stuck                      │   │
 │   + ODD check OK                                    │   │
 │         ▼                                           │   │
 │  [AGENT_REQUESTING_AUTHORITY]                       │   │
 │         │                                           │   │
 │   Safety Supervisor: contract_valid?                │   │
 │      YES │                  NO │                   │   │
 │          ▼                     ▼                   │   │
 │  [SUPERVISED_EXECUTION]    [BASELINE_CONTROL]      │   │
 │  (warm handoff 0.5s)                               │   │
 │         │                                          │   │
 │  supervisor confirms stable                        │   │
 │         ▼                                          │   │
 │  [AGENT_ACTIVE_BOUNDED] ─── contract expired ──► [AUTHORITY_REVOKE_PENDING]
 │         │                                     ▲   │   │
 │  abort condition                contract brake│   │   │
 │         ▼                   or safety veto    │   │   │
 │  [AUTHORITY_REVOKE_PENDING]──────────────────┘   │   │
 │         │                                         │   │
 │   graceful return to baseline (≤0.5s)             │   │
 │         ▼                                         │   │
 │  [BASELINE_CONTROL] ─  or if baseline fails ──► [MINIMAL_RISK_MANEUVER]
 │                                                    │   │
 │                                    no human TOR ───┤   │
 │                                    response        │   │
 │                                                    ▼   │
 │                                              [HUMAN_TOR]│
 │                                                    │   │
 │                                human takes over ───┘   │
 │                                timeout → [SAFE_STOP]   │
 └─────────────────────────────────────────────────────────┘
```

### Bảng chuyển trạng thái

| Từ | Đến | Điều kiện trigger | Action |
|----|-----|-------------------|--------|
| BASELINE_CONTROL | AGENT_REQUESTING | `baseline_stuck_score > 0.7` AND `ODD_valid` AND `agent_confidence > 0.85` | Agent submit `ManeuverContract` |
| AGENT_REQUESTING | SUPERVISED_EXECUTION | `safety_supervisor.verify(contract) == APPROVED` | Warm handoff begin, baseline stays hot |
| AGENT_REQUESTING | BASELINE_CONTROL | `verify == REJECTED` hoặc timeout 1s | Log reject reason, reset |
| SUPERVISED_EXECUTION | AGENT_ACTIVE_BOUNDED | `supervisor confirms intent 3 frames stable` | Full L1 authority granted |
| AGENT_ACTIVE_BOUNDED | AUTHORITY_REVOKE_PENDING | contract expired OR abort_condition OR supervisor veto | Begin graceful revoke |
| AGENT_ACTIVE_BOUNDED | AUTHORITY_REVOKE_PENDING | `agent_confidence < 0.6` liên tiếp 5 frames | Confidence collapse |
| AUTHORITY_REVOKE_PENDING | BASELINE_CONTROL | baseline planner has stable output | Smooth handoff |
| AUTHORITY_REVOKE_PENDING | MINIMAL_RISK_MANEUVER | baseline cũng fail, no safe output | Enter MRM |
| MINIMAL_RISK_MANEUVER | HUMAN_TOR | MRM active > 2s, situation unresolved | Issue TOR |
| HUMAN_TOR | BASELINE_CONTROL | human input detected, override | Human takes over |
| HUMAN_TOR | SAFE_STOP | TOR timeout (8s no response) | Emergency deceleration to stop |

---

## 4. Forward Takeover Protocol

### Điều kiện để Agent được REQUESTING AUTHORITY

Agent chỉ được kích hoạt Request khi ít nhất **một** điều kiện sau được thoả:

```python
BASELINE_STUCK_CONDITIONS = [
    # Baseline velocity stall: xe dưới 1 m/s trong 3s liên tiếp trong corridor rõ ràng
    "v_ego < 1.0 m/s AND t_stall > 3.0s AND corridor_clear",
    # Baseline oscillation: lateral deviation > 0.5m lặp lại 3 lần trong 5s
    "lateral_deviation_oscillating AND count > 3 AND t_window < 5s",
    # Planner degeneracy: MPC cost > 5x nominal AND no convergence
    "mpc_cost > nominal_cost * 5 AND convergence_failed",
    # ODD exceedance pending: tình huống sắp ra khỏi ODD
    "odd_exceedance_margin < 2s",
]
```

### Cấu trúc ManeuverContract (triển khai thành code)

```python
@dataclass
class ManeuverContract:
    # Định danh
    contract_id: str               # UUID
    issued_at_frame: int
    source: str                    # "stage9_agent"

    # Mục tiêu hành vi (L1 - Maneuver Authority only)
    tactical_intent: str           # e.g. "avoid_obstacle_right", "lane_change_left"
    sub_intent_sequence: list[str] # ["prepare_lc_left", "commit_lc_left"]
    target_state_description: str  # human-readable mục tiêu

    # Ràng buộc không gian
    max_lateral_offset_m: float    # <= 0.8 m mặc định
    max_longitudinal_accel_mps2: float  # <= 2.0 m/s² mặc định
    max_lateral_accel_mps2: float  # <= 1.5 m/s² mặc định
    drivable_envelope_uuid: str    # reference đến envelope đã validated

    # Ràng buộc thời gian
    max_duration_s: float          # <= 5.0s mặc định, tuyệt đối không vượt
    max_speed_mps: float           # <= 8.33 m/s (30 km/h) mặc định

    # Điều kiện huỷ tự động (abort conditions)
    min_ttc_threshold_s: float     # TTC < 1.5s → auto abort
    max_new_obstacle_score: float  # score > 0.7 nếu có vật cản mới
    min_confidence_floor: float    # confidence < 0.6 → abort
    planner_feasibility_required: bool  # True = MPC must confirm feasible

    # Thông tin bổ sung
    agent_confidence: float        # [0.0, 1.0]
    agent_reasoning_summary: str   # Rút gọn từ LLM CoT
    fallback_intent: str           # e.g. "keep_lane" - dùng khi contract bị abort
```

### Flow chi tiết: Warm Handoff (không giật cục)

```
Frame N:   [AGENT_REQUESTING] — Agent submit contract
Frame N+1: [SUPERVISED_EXECUTION] — Safety Supervisor verify (max 3 frames)
           - Baseline ở chế độ "hot standby" (vẫn tính trajectory, không phát ra)
           - Nếu Supervisor REJECT → back to BASELINE, log reason
Frame N+3: Agent intent stable 3 frames liên tiếp
           → [AGENT_ACTIVE_BOUNDED]
           → MPC nhận maneuver_context = contract (không nhận raw steering)
Frame N+3 đến N+53 (5s @10Hz):
           - Mỗi frame: supervisor re-checks abort conditions
           - Mỗi frame: contract timer decrements
           - MPC vẫn là người duy nhất xuất lệnh điều khiển

Frame N+53 (hết thời hạn):
           → [AUTHORITY_REVOKE_PENDING]
           → Agent nhận signal: "contract_expired"
           → 5 frame transition: Agent output fallback_intent
           → MPC blend: 0% agent weight → 100% baseline weight over 5 frames
           → [BASELINE_CONTROL]
```

---

## 5. Backward Takeover / TOR / MRC Logic

### TOR Manager — 3 cấp cảnh báo

```python
class TORManager:
    """
    Takeover Request Manager — Stage 9
    human-in-the-loop simulation stub (full-sim mode)
    """
    ESCALATION_LEVELS = {
        1: {"t_start": 0.0, "alert": "VISUAL_AMBER",   "action": "hmi_amber_flash"},
        2: {"t_start": 3.0, "alert": "VISUAL_RED+AUDIO","action": "hmi_red_continuous + haptic"},
        3: {"t_start": 6.0, "alert": "CRITICAL",       "action": "emergency_brake_prep"},
    }
    TOR_TIMEOUT_S = 8.0   # Nếu không có response sau 8s → SAFE_STOP
```

### Minimal Risk Condition (MRC) — không phải Emergency Stop

> **MRC ≠ Emergency brake**: MRC là dừng có kiểm soát, tấp vào lề nếu có thể, không phanh cứng gây lật/đâm đuôi.

```python
@dataclass
class MRCPlan:
    strategy: str          # "COAST_TO_STOP" | "LANE_TIP_RIGHT" | "EMERGENCY_STOP"
    target_v_final: float  # 0.0 m/s
    max_decel_mps2: float  # <= 3.0 m/s² bình thường, <= 6.0 m/s² emergency
    prefer_shoulder: bool  # True = cố tấp vào lề phải trước khi dừng
    hazard_lights: bool    # True = bật đèn hazard ngay khi bắt đầu MRC

    # Tính từ world state hiện tại:
    estimated_stop_distance_m: float
    estimated_stop_time_s: float
    corridor_for_stop: Optional[str]   # lane_id nếu có

SAFE_STOP_SEQUENCE = [
    "ACTIVATE_HAZARDS",
    "SIGNAL_RIGHT_IF_SHOULDER_CLEAR",
    "DECELERATE_AT MAX_2.5_MPS2",
    "MAINTAIN_LANE_CENTERING",
    "STOP_FULLY",
    "BROADCAST_V2X_STOPPED" if v2x_available else "SET_FLAG_PARKED",
]
```

---

## 6. Safety Contract — 3 lớp kiểm tra

```python
class SafetySupervisor:
    """
    Safety Supervisor — KHÔNG phụ thuộc Agent, độc lập hoàn toàn.
    Luôn chạy song song. Có quyền VETO bất kỳ lúc nào.
    """

    def verify_contract(self, contract: ManeuverContract, world: WorldState) -> SafetyVerdict:
        verdict = SafetyVerdict(approved=True, reasons=[])

        # Lớp 1: Physical Hard Constraints
        if contract.max_lateral_offset_m > 1.0:
            verdict.reject("S-001: lateral offset exceeds 1.0m hard limit")
        if contract.max_speed_mps > 13.89:   # 50 km/h absolute cap
            verdict.reject("S-002: speed cap exceeded")
        if contract.max_duration_s > 8.0:
            verdict.reject("S-003: contract duration > 8s not allowed")

        # Lớp 2: Situational Safety (real-time world state)
        if world.min_ttc < contract.min_ttc_threshold_s * 1.5:
            verdict.reject("S-004: TTC dangerously low at contract start")
        if world.weather_visibility < 30.0:    # metres
            verdict.reject("S-005: visibility < 30m, ODD exceeded")
        if world.ego_lateral_error > 0.6:
            verdict.reject("S-006: ego already deviating, baseline not stable")

        # Lớp 3: Agent Confidence & Temporal Consistency
        if contract.agent_confidence < 0.80:
            verdict.reject("S-007: agent confidence below 0.80 floor")
        if self._memory.consecutive_revokes_last_60s > 2:
            verdict.reject("S-008: too many recent revocations, cooldown active")

        return verdict

    def watch_frame(self, world: WorldState, contract: ManeuverContract) -> Optional[VetoSignal]:
        """Gọi mỗi frame khi đang AGENT_ACTIVE_BOUNDED."""
        if world.min_ttc < contract.min_ttc_threshold_s:
            return VetoSignal("S-V001: TTC collapsed", severity="HARD")
        if world.new_obstacle_score > contract.max_new_obstacle_score:
            return VetoSignal("S-V002: unexpected obstacle", severity="HARD")
        if self._contract_timer_expired(contract):
            return VetoSignal("S-V003: contract time expired", severity="SOFT")
        if self._mpc_reports_infeasible():
            return VetoSignal("S-V004: MPC trajectory infeasible", severity="HARD")
        return None
```

---

## 7. Interface / Message Schema

### `world_state_t` (đầu ra BEVFusion, đầu vào mọi module)

```python
@dataclass
class WorldState:
    frame_id: int
    timestamp_s: float
    ego_v_mps: float
    ego_a_mps2: float
    ego_lane_id: str
    ego_lateral_error_m: float
    corridor_clear: bool
    min_ttc_s: float                  # Time-to-collision với điểm nguy hiểm nhất
    new_obstacle_score: float         # [0,1] khả năng vật cản bất ngờ
    weather_visibility_m: float
    lane_change_permission: bool
    drivable_envelope: DrivableEnvelope
    odd_status: ODDStatus             # IN_ODD | ODD_MARGIN | ODD_EXCEEDED
```

### `authority_context_t` (đầu ra ASM, đầu vào tất cả planner)

```python
@dataclass
class AuthorityContext:
    current_state: AuthorityState     # enum 8 states
    active_authority: str             # "BASELINE" | "AGENT" | "MRM" | "HUMAN"
    active_contract: Optional[ManeuverContract]
    contract_remaining_s: Optional[float]
    baseline_weight: float            # [0.0, 1.0] — blend weight cho trajectory
    agent_weight: float               # 1.0 - baseline_weight
    revoke_reason: Optional[str]
    tor_active: bool
    tor_elapsed_s: Optional[float]
```

### `trajectory_request_t` (đầu vào MPC)

```python
@dataclass
class TrajectoryRequest:
    source: str                       # "BASELINE" | "CONTRACT_RESOLVER" | "MRM"
    tactical_intent: str
    # Hard constraints từ contract (MPC KHÔNG được vi phạm)
    v_max_mps: float
    a_long_max_mps2: float
    a_lat_max_mps2: float
    lateral_bound_m: float
    drivable_envelope: DrivableEnvelope
    # Objective hints (MPC tối ưu trong phạm vi trên)
    target_lane_id: Optional[str]
    target_v_desired_mps: float
    horizon_s: float                  # prediction horizon
```

---

## 8. Pseudocode — Authority Arbitration Loop (10 Hz)

```python
def authority_arbitration_loop(
    world: WorldState,
    asm: AuthorityStateMachine,
    agent: Stage9AgentAdapter,
    supervisor: SafetySupervisor,
    baseline: BaselinePlanner,
    mpc: MPCOptimizer,
    contract_resolver: ContractResolver,
    tor: TORManager,
    mrm: MRCExecutor,
) -> ActuatorCommand:

    ctx: AuthorityContext = asm.get_context()

    # ── SAFE_STOP: Tuyệt đối ưu tiên ──────────────────────────────────
    if ctx.current_state == SAFE_STOP:
        return mpc.execute(mrm.get_emergency_stop_request(world))

    # ── MINIMAL_RISK_MANEUVER ──────────────────────────────────────────
    if ctx.current_state == MINIMAL_RISK_MANEUVER:
        mrm_plan = mrm.compute(world)
        cmd = mpc.execute(mrm_plan.to_trajectory_request())
        if mrm.is_complete():
            asm.transition_to(HUMAN_TOR if not human_present else BASELINE_CONTROL)
        return cmd

    # ── HUMAN_TOR ─────────────────────────────────────────────────────
    if ctx.current_state == HUMAN_TOR:
        tor.tick(world)
        if tor.human_responded():
            asm.transition_to(BASELINE_CONTROL)
        elif tor.timed_out():
            asm.transition_to(SAFE_STOP)
        return mpc.execute(mrm.get_hold_trajectory(world))   # giữ nguyên, không làm gì thêm

    # ── BASELINE_CONTROL ──────────────────────────────────────────────
    if ctx.current_state == BASELINE_CONTROL:
        baseline_stuck = baseline_detector.is_stuck(world)
        odd_ok = world.odd_status != ODD_EXCEEDED

        if baseline_stuck and odd_ok:
            contract = agent.propose_contract(world)
            if contract is not None:
                asm.transition_to(AGENT_REQUESTING_AUTHORITY, contract=contract)

        baseline_req = baseline.plan(world)
        return mpc.execute(baseline_req)

    # ── AGENT_REQUESTING_AUTHORITY ────────────────────────────────────
    if ctx.current_state == AGENT_REQUESTING_AUTHORITY:
        contract = ctx.active_contract
        verdict = supervisor.verify_contract(contract, world)

        if verdict.approved:
            asm.transition_to(SUPERVISED_EXECUTION, contract=contract)
        elif asm.request_timed_out():
            asm.transition_to(BASELINE_CONTROL, reason="request_timeout")

        # Trong lúc chờ: Baseline vẫn chạy nền
        return mpc.execute(baseline.plan(world))

    # ── SUPERVISED_EXECUTION (warm handoff, max 3 frames) ─────────────
    if ctx.current_state == SUPERVISED_EXECUTION:
        agent_intent = agent.get_intent(world, ctx.active_contract)
        supervisor_ok = supervisor.watch_frame(world, ctx.active_contract)

        if supervisor_ok is None and asm.supervised_frames >= 3:
            asm.transition_to(AGENT_ACTIVE_BOUNDED)
        elif supervisor_ok is not None and supervisor_ok.severity == "HARD":
            asm.transition_to(BASELINE_CONTROL, reason=supervisor_ok.reason)

        # Blend: 50% baseline, 50% agent trong supervised phase
        baseline_req = baseline.plan(world)
        agent_req = contract_resolver.resolve(agent_intent, ctx.active_contract)
        blended = TrajectoryRequest.blend(baseline_req, agent_req, agent_weight=0.5)
        return mpc.execute(blended)

    # ── AGENT_ACTIVE_BOUNDED ──────────────────────────────────────────
    if ctx.current_state == AGENT_ACTIVE_BOUNDED:
        # Supervisor luôn watch
        veto = supervisor.watch_frame(world, ctx.active_contract)
        if veto is not None:
            asm.transition_to(AUTHORITY_REVOKE_PENDING, reason=veto.reason)
            return mpc.execute(baseline.plan(world))

        # Agent xuất intent (L1 only, không phải raw control)
        agent_intent = agent.get_intent(world, ctx.active_contract)
        agent_req = contract_resolver.resolve(agent_intent, ctx.active_contract)

        # Contract constraints được inject vào MPC — không phải bypass MPC
        return mpc.execute(agent_req)   # MPC vẫn tối ưu, chỉ trong bounds của contract

    # ── AUTHORITY_REVOKE_PENDING (graceful, 5 frames = 0.5s) ────────
    if ctx.current_state == AUTHORITY_REVOKE_PENDING:
        revoke_frame = asm.revoke_frame_count
        agent_weight = max(0.0, 1.0 - (revoke_frame / 5.0))
        baseline_weight = 1.0 - agent_weight

        agent_req = contract_resolver.resolve(ctx.active_contract.fallback_intent, ctx.active_contract)
        baseline_req = baseline.plan(world)
        blended = TrajectoryRequest.blend(baseline_req, agent_req, agent_weight=agent_weight)

        asm.increment_revoke_frame()
        if revoke_frame >= 5:
            if baseline.is_healthy(world):
                asm.transition_to(BASELINE_CONTROL)
            else:
                asm.transition_to(MINIMAL_RISK_MANEUVER)

        return mpc.execute(blended)
```

---

## 9. Tích Hợp Vào Stack Hiện Tại

### Điểm chèn vào pipeline BEVFusion → Agentic AI → MPC

```
Stage 8 stack (hiện tại):
  world_state → Stage8AssistAdapter → BRP → ArbitrationEngine → (advisory only) → MPC (baseline)

Stage 9 stack (thêm vào):
  world_state
      │
      ├─► BaselineDetector (mới — phase 1)
      │
      ├─► AuthorityStateMachine (mới — phase 1)
      │       │
      │   authority_context_t
      │       │
      ├─► Stage9AgentAdapter (mới — dùng lại Stage8AssistAdapter)
      │       │ ManeuverContract
      │       │
      ├─► SafetySupervisor (mở rộng từ Stage8 ArbitrationEngine)
      │       │ SafetyVerdict / VetoSignal
      │       │
      ├─► ContractResolver (mới — phase 1)
      │       │ TrajectoryRequest
      │       │
      └─► MPC (giữ nguyên, chỉ thêm constraint injection interface)
              │
              └─► CARLA (không thay đổi)
```

### Các thay đổi **không làm** (để đảm bảo an toàn khi tích hợp):

- **Không** sửa MPC core — chỉ thêm constraint injection vào `TrajectoryRequest`.
- **Không** sửa BEVFusion output — `WorldState` là read-only với tất cả module Stage 9.
- **Không** xoá Stage 8 code — Stage9AgentAdapter kế thừa và bọc lại.

---

## 10. Evaluation Metrics & Scenario Tests

### Metrics (đo trong CARLA)

| Metric | Công thức | Target |
|--------|-----------|--------|
| **Takeover Success Rate (TSR)** | `successful_takeovers / total_takeovers` | ≥ 90% |
| **Stuck Resolution Rate (SRR)** | `resolved_stucks / total_stucks` | ≥ 85% |
| **False Takeover Rate (FTR)** | `unnecessary_takeovers / total_takeovers` | ≤ 5% |
| **Contract Abort Rate (CAR)** | `mid-contract aborts / total_contracts` | ≤ 10% |
| **Graceful Revoke Rate (GRR)** | `smooth_revokes / total_revokes` | ≥ 95% |
| **MRM Trigger Rate** | `mrm_events / total_km` | ≤ 0.1 /km |
| **Authority Oscillation Index (AOI)** | `state_transitions / 60s` | ≤ 2 /min |
| **Contract Duration Utilization** | `avg(actual_duration / max_duration)` | 60–80% |

### Scenario Test Matrix (CARLA)

| Mã | Tình huống | Expected State Path | Pass Condition |
|----|-----------|--------------------|----|
| T9-001 | Xe giữa làn đang đỗ | BASELINE → REQUESTING → ACTIVE → REVOKE → BASELINE | xe vượt, TSR=1 |
| T9-002 | Baseline oscillation > 3 lần | BASELINE → REQUESTING → ACTIVE | lat error giảm < 0.3m |
| T9-003 | Agent confidence tụt giữa chừng | ACTIVE → REVOKE | không jerk, AOI < 2 |
| T9-004 | TTC collapse khi đang ACTIVE | ACTIVE → REVOKE → MRM | dừng an toàn, no collision |
| T9-005 | Contract hết giờ bình thường | ACTIVE → REVOKE → BASELINE | GRR = 1.0 |
| T9-006 | Baseline stuck + ODD exceeded | BASELINE → MRM | không grant takeover |
| T9-007 | TOR simulation stub (CARLA) | MRM → TOR → SAFE_STOP | xe dừng trong 25m |
| T9-008 | Nhiều request liên tiếp | REQUESTING → REJECTED × 3 | cooldown active (S-008) |

---

## 11. Risk Analysis + Mitigation

| Rủi Ro | Mức độ | Mitigation |
|--------|--------|-----------|
| **Authority Oscillation**: Agent liên tục xin-bị-thu quyền | CAO | AOI gate (≤ 2/min), cooldown 30s sau 2 revokes |
| **Latency: LLM chậm giữa ACTIVE** | TRUNG BÌNH | 1Hz downsampling (kế thừa Stage 8), cached intent trong contract window |
| **Contract Infeasibility**: MPC không optimize được trong bounds | TRUNG BÌNH | MPC feasibility check trước khi APPROVED, infeasibility → immediate revoke (S-V004) |
| **Perception Noise**: BEVFusion nhiễu → fake stuck signal | TRUNG BÌNH | BaselineDetector dùng sliding window 3s, không phản ứng spike đơn |
| **Graceful Revoke Fails**: Baseline cũng lỗi khi thu quyền | THẤP-CAO | Fallback chain: REVOKE → check baseline_healthy → MRM nếu không |
| **TOR không có người nhận** | THẤP (sim) | TOR timeout → SAFE_STOP tự động, không bao giờ bị kẹt |
| **Maneuver Contract quá bảo thủ** | THẤP | Tune bounds qua scenario test, không hard-code tất cả vào code |

---

## 12. Implementation Roadmap

### Phase 1 — Core Authority Infrastructure (Ưu tiên cao, ~2 tuần)

```
1. schemas.py:              WorldState, ManeuverContract, AuthorityContext, TrajectoryRequest
2. authority_state_machine.py: 8 states, transitions, timers
3. maneuver_contract.py:    dataclass + validator
4. safety_supervisor.py:    verify_contract() + watch_frame() — mở rộng từ Stage8 AE
5. baseline_detector.py:    sliding window stuck detector (3 conditions)
6. contract_resolver.py:    ManeuverContract → TrajectoryRequest + constraint injection
7. authority_arbiter.py:    main loop 10Hz, full pseudocode từ Section 8
```

> **Gate Phase 1**: Chạy scenarios T9-001, T9-005, T9-008. TSR ≥ 90%, GRR ≥ 95%.

### Phase 2 — Revoke, MRM, TOR Stub (~1 tuần)

```
8. tor_manager.py:           TOR timer + 3-level escalation
9. minimal_risk_maneuver.py: MRM executor (COAST + LANE_TIP)
10. authority_logger.py:     structured log: event, state, reason, frame_id
```

> **Gate Phase 2**: Scenarios T9-004, T9-006, T9-007. MRM Trigger Rate ≤ 0.1/km.

### Phase 3 — Evaluation & Integration Validation (~1 tuần)

```
11. stage9_evaluator.py:     metrics: TSR, SRR, FTR, AOI
12. Run full T9-001 → T9-008 + regression on Stage 8 canary cases
13. Stage 9 campaign report generation
```

> **Gate Phase 3 (Promotion)**: All 8 scenarios pass, TSR ≥ 90%, AOI ≤ 2/min, MRM Rate ≤ 0.1/km.

---

## Tóm Tắt Nguyên Tắc Thiết Kế

```
┌─────────────────────────────────────────────────────────────────────┐
│            STAGE 9 DESIGN PRINCIPLES — SUMMARY                      │
│                                                                      │
│  1. Agent nhận L1 (Maneuver), không bao giờ L3 (Actuator)           │
│  2. Mọi takeover đều có Contract: mục tiêu + bounds + thời hạn      │
│  3. Safety Supervisor độc lập, chạy mọi frame, veto > authority      │
│  4. Warm handoff: không giật cục khi cấp / thu quyền (blend 5 frames)│
│  5. Graceful fallback chain: AGENT → BASELINE → MRM → TOR → STOP    │
│  6. Authority Oscillation là kẻ thù: cooldown + AOI gate bắt buộc   │
│  7. MPC là bức tường cuối cùng: LUÔN tối ưu, LUÔN trong bounds      │
│  8. Log mọi thứ: ai cấp quyền, khi nào, tại sao, ai thu lại         │
└─────────────────────────────────────────────────────────────────────┘
```
