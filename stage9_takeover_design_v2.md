# Stage 9 — Takeover Design (v2 Corrected)
## Autonomous Driving Agentic Authority Protocol
**Role:** Principal Autonomous Driving Architect  
**Stack:** BEVFusion → Agentic AI → MPC → CARLA  
**Core Invariant:** _"Agent may gain bounded maneuver authority, but never unconditional actuator authority."_

---

## 1. Kiến Trúc Tổng Thể

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SENSOR FUSION LAYER                               │
│  Camera + LiDAR + Radar ──► BEVFusion ──► WorldState / TacticalContext     │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ world_state_t
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     AUTHORITY ORCHESTRATION LAYER                           │
│                                                                              │
│  ┌──────────────────────────┐   ┌──────────────────────────┐                │
│  │ Authority State Machine  │   │    Authority Arbiter     │                │
│  │ (ASM, timers, cooldown)  │──►│ (10 Hz, transition gate) │                │
│  └──────────────┬───────────┘   └──────────────┬───────────┘                │
│                 │ authority_context_t          │                             │
│        ┌────────┼─────────┬──────────┬─────────┼─────────────┐              │
│        ▼        ▼         ▼          ▼         ▼             ▼              │
│  Baseline   Stage9Agent   Safety   Handoff   TOR Manager   Human Override   │
│  Planner    Adapter       Supervisor Planner   + MRM        Monitor          │
│ (default)   (bounded L1)  (veto)   (ref blend)(fallback)   (manual detect)  │
└───────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────────┘
        │          │          │          │          │          │
        └──────────┴──────────┴──────────┴──────────┴──────────┘
                                     │ winning TrajectoryRequest
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TRAJECTORY & CONTROL LAYER                           │
│                                                                              │
│  Contract Resolver / Handoff Planner ──► TrajectoryRequest ──► MPC          │
│             (constraint injection + reference transition)                    │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ u(t): steer, throttle, brake
                                     ▼
                                CARLA Actuators
```

### 3 tầng quyền lực — định nghĩa tuyệt đối

| Tầng | Tên | Ai giữ | Ý nghĩa |
|------|-----|--------|---------|
| L1 | **Maneuver Authority** | ASM cấp có thời hạn cho Agent | Quyền chọn hành vi chiến thuật như `keep_lane`, `prepare_lane_change`, `bypass_obstacle_slow`, `pull_over` |
| L2 | **Trajectory Authority** | Planner/MPC | Quyền biến maneuver hợp lệ thành quỹ đạo tham chiếu, có kiểm tra feasibility |
| L3 | **Actuator Authority** | MPC → CARLA | Quyền xuất `steer`, `throttle`, `brake` thực tế |

> Agent chỉ được L1. Agent không bao giờ được phát lệnh L2/L3 trực tiếp, kể cả khi đang takeover.

### Sửa triết lý Stage 9 cho rõ ràng

- **Forward Takeover** = Agent được quyền chọn maneuver trong một hợp đồng hữu hạn.
- **Backward Takeover / TOR** = hệ thống yêu cầu con người tiếp quản khi sắp ra khỏi ODD hoặc khi autonomy không còn đáng tin.
- **MRM** = fallback an toàn nếu không thể chờ con người hoặc con người không phản hồi kịp.
- **Human takeover** là một state riêng, không được gộp nhầm vào `BASELINE_CONTROL`.

---

## 2. Các Module Cần Có

### Module bắt buộc (Phase 1)

```
stage9/
├── authority_state_machine.py    # 9 states, timers, cooldown, hysteresis
├── authority_arbiter.py          # Main loop 10 Hz
├── schemas.py                    # Dataclass / enums / interfaces
├── maneuver_contract.py          # Contract dataclass + contract validation
├── safety_supervisor.py          # Hard constraints, ODD gate, freshness gate, veto
├── baseline_detector.py          # Detect stuck / oscillation / degeneracy bằng sliding window
├── contract_resolver.py          # ManeuverContract -> TrajectoryRequest
├── handoff_planner.py            # Reference-level handoff, không blend raw actuator
└── authority_logger.py           # grant / reject / revoke / TOR / MRM audit log
```

### Module supporting (Phase 2)

```
stage9/
├── tor_manager.py                # TOR timing + escalation + timeout policy
├── minimal_risk_maneuver.py      # MRM planner / executor
├── human_override_monitor.py     # phát hiện thao tác người lái trong CARLA / HIL stub
└── stage9_evaluator.py           # TSR, SRR, AOI, comfort, latency, stale-grant metrics
```

---

## 3. State Machine Chi Tiết

## 9 trạng thái chuẩn

1. `BASELINE_CONTROL`  
2. `AGENT_REQUESTING_AUTHORITY`  
3. `SUPERVISED_EXECUTION`  
4. `AGENT_ACTIVE_BOUNDED`  
5. `AUTHORITY_REVOKE_PENDING`  
6. `TOR_ACTIVE`  
7. `HUMAN_CONTROL_ACTIVE`  
8. `MINIMAL_RISK_MANEUVER`  
9. `SAFE_STOP`

### Sơ đồ logic

```
[BASELINE_CONTROL]
    │ baseline stuck + IN_ODD + fresh world + agent eligible
    ▼
[AGENT_REQUESTING_AUTHORITY]
    │ approved
    ▼
[SUPERVISED_EXECUTION]
    │ stable + feasible
    ▼
[AGENT_ACTIVE_BOUNDED]
    │ expiry / veto / conf collapse / ODD margin / stale world
    ▼
[AUTHORITY_REVOKE_PENDING]
    ├──────── baseline healthy ───────► [BASELINE_CONTROL]
    ├──────── human available + time budget đủ ─► [TOR_ACTIVE]
    └──────── no safe wait / no human / risk cao ─► [MINIMAL_RISK_MANEUVER]

[TOR_ACTIVE]
    ├──────── human input detected ───► [HUMAN_CONTROL_ACTIVE]
    └──────── timeout / risk escalates ─► [MINIMAL_RISK_MANEUVER]

[HUMAN_CONTROL_ACTIVE]
    ├──────── explicit disengage + autonomy healthy ─► [BASELINE_CONTROL]
    └──────── driver abandons / severe fault ─► [MINIMAL_RISK_MANEUVER]

[MINIMAL_RISK_MANEUVER]
    ├──────── human input detected ───► [HUMAN_CONTROL_ACTIVE]
    └──────── full stop reached ──────► [SAFE_STOP]
```

### Bảng chuyển trạng thái

| Từ | Đến | Trigger chính | Action |
|----|-----|---------------|--------|
| BASELINE_CONTROL | AGENT_REQUESTING_AUTHORITY | `baseline_stuck_score > 0.7` AND `odd_status == IN_ODD` AND `world_age_ms <= 100` AND `sync_ok == True` AND `agent_confidence >= 0.85` | Agent submit `ManeuverContract` |
| BASELINE_CONTROL | TOR_ACTIVE | `odd_status == ODD_MARGIN` AND `human_driver_available` AND `time_budget_s >= 5.0` | Start TOR sequence |
| BASELINE_CONTROL | MINIMAL_RISK_MANEUVER | `odd_status == ODD_EXCEEDED` OR critical degradation OR `time_budget_s < 5.0` | Enter MRM immediately |
| AGENT_REQUESTING_AUTHORITY | SUPERVISED_EXECUTION | `safety_supervisor.verify_contract(...) == APPROVED` | Start warm handoff preview |
| AGENT_REQUESTING_AUTHORITY | BASELINE_CONTROL | `verify == REJECTED` OR timeout 1s | Log reject reason, reset |
| SUPERVISED_EXECUTION | AGENT_ACTIVE_BOUNDED | stable 3 frames AND preview trajectory feasible | Grant bounded L1 authority |
| SUPERVISED_EXECUTION | BASELINE_CONTROL | supervisor veto OR preview infeasible | Abort request |
| AGENT_ACTIVE_BOUNDED | AUTHORITY_REVOKE_PENDING | contract expired OR abort condition OR supervisor veto OR stale world OR `odd_status != IN_ODD` | Begin graceful revoke |
| AUTHORITY_REVOKE_PENDING | BASELINE_CONTROL | baseline healthy and feasible | Smooth return |
| AUTHORITY_REVOKE_PENDING | TOR_ACTIVE | baseline unhealthy AND human available AND time budget đủ | Request human takeover |
| AUTHORITY_REVOKE_PENDING | MINIMAL_RISK_MANEUVER | baseline unhealthy AND waiting unsafe | Enter MRM |
| TOR_ACTIVE | HUMAN_CONTROL_ACTIVE | manual override detected | Human takes authority |
| TOR_ACTIVE | MINIMAL_RISK_MANEUVER | timeout OR risk escalation OR `time_budget_s < min_wait_budget` | Safe fallback |
| HUMAN_CONTROL_ACTIVE | BASELINE_CONTROL | human releases control AND autonomy healthy AND explicit re-engage | Return to autonomy |
| HUMAN_CONTROL_ACTIVE | MINIMAL_RISK_MANEUVER | driver abandons OR severe fault persists | MRM |
| MINIMAL_RISK_MANEUVER | HUMAN_CONTROL_ACTIVE | manual override detected | Human intervenes |
| MINIMAL_RISK_MANEUVER | SAFE_STOP | vehicle fully stopped in safe corridor | Terminal safe state |

### Hysteresis / anti-thrashing bắt buộc

- Sau mỗi `REVOKE` hoặc `REJECT`, bật **cooldown** tối thiểu 15–30 giây trước khi Agent được xin quyền lại.
- Chỉ cho phép grant mới nếu `revoke_count_last_60s <= 2`.
- `baseline_stuck_score` phải vượt ngưỡng liên tục trong cửa sổ 2–3 giây, không phản ứng spike đơn.
- Khi đã vào `TOR_ACTIVE` hoặc `MINIMAL_RISK_MANEUVER`, không quay về Agent ngay trong cùng sự kiện.

---

## 4. Forward Takeover Protocol

### Điều kiện để Agent được REQUESTING AUTHORITY

Agent chỉ được quyền request nếu đồng thời thoả các nhóm sau:

```python
FORWARD_TAKEOVER_PRECONDITIONS = {
    "baseline_need": [
        "velocity_stall: ego_v < 1.0 m/s for > 3.0s while corridor_clear",
        "lateral_oscillation: > 3 oscillations within 5.0s",
        "planner_degeneracy: mpc_cost > 5x nominal OR convergence_failed",
    ],
    "safety_gate": [
        "odd_status == IN_ODD",
        "world_age_ms <= 100",
        "sync_ok == True",
        "sensor_health in {OK, DEGRADED_BUT_VALID}",
        "time_to_odd_exit_s >= contract.max_duration_s + 1.0",
    ],
    "agent_gate": [
        "agent_confidence >= 0.85",
        "cooldown_inactive == True",
        "recent_revokes_last_60s <= 2",
    ],
    "control_gate": [
        "preview_feasible == True",
        "ttc_start >= max(2.0, contract.min_ttc_threshold_s * 1.5)",
    ],
}
```

> **Loại bỏ grant khi gần hết ODD.** `ODD_MARGIN` và `ODD_EXCEEDED` không phải vùng để tăng quyền cho Agent.

### Cấu trúc `ManeuverContract`

```python
@dataclass
class ManeuverContract:
    # identity
    contract_id: str
    issued_at_frame: int
    source: str                      # "stage9_agent"

    # maneuver scope (L1 only)
    tactical_intent: str             # e.g. "slow_bypass_right"
    sub_intent_sequence: list[str]   # e.g. ["prepare", "commit", "stabilize"]
    target_state_description: str

    # spatial / dynamic bounds
    max_lateral_offset_m: float
    max_longitudinal_accel_mps2: float
    max_lateral_accel_mps2: float
    max_jerk_mps3: float
    drivable_envelope_uuid: str
    target_lane_id: str | None

    # temporal bounds
    max_duration_s: float
    max_speed_mps: float
    validity_deadline_s: float

    # safety abort conditions
    min_ttc_threshold_s: float
    max_new_obstacle_score: float
    min_confidence_floor: float
    planner_feasibility_required: bool
    freshness_required_ms: int       # e.g. 100 ms max world staleness

    # revoke policy
    revoke_strategy: str             # "RETURN_TO_BASELINE" | "ENTER_MRM"
    cooldown_after_revoke_s: float   # e.g. 20 s

    # audit
    agent_confidence: float
    agent_reasoning_summary: str
```

### Warm handoff đúng cách

Không blend raw actuator. Chỉ được blend ở tầng **reference / objective / cost weighting**:

1. `AGENT_REQUESTING_AUTHORITY`: Agent gửi contract. Baseline vẫn phát trajectory bình thường.
2. `SUPERVISED_EXECUTION`: 3 frame preview. HandoffPlanner nội suy reference giữa baseline preview và contract-resolved preview.
3. Nếu preview feasible, ổn định, và supervisor không veto → vào `AGENT_ACTIVE_BOUNDED`.
4. Trong `AGENT_ACTIVE_BOUNDED`, MPC nhận `TrajectoryRequest` có hard bounds từ contract.
5. Khi revoke, HandoffPlanner ramp reference trở về baseline trong 3–5 frames.

---

## 5. Backward Takeover / TOR / MRM Logic

### Chính sách ưu tiên

- Nếu còn **đủ thời gian** và có khả năng người lái tiếp quản: vào `TOR_ACTIVE` trước.
- Nếu **không đủ thời gian**, autonomy đã mất ổn định nghiêm trọng, hoặc không có người lái khả dụng: vào `MINIMAL_RISK_MANEUVER` ngay.
- Nếu đang `TOR_ACTIVE` mà timeout hoặc risk tăng nhanh: leo thang sang `MINIMAL_RISK_MANEUVER`.
- Nếu đang MRM mà người lái can thiệp: chuyển sang `HUMAN_CONTROL_ACTIVE`.

### TOR Manager — 3 cấp cảnh báo

```python
class TORManager:
    ESCALATION_LEVELS = {
        1: {"t_start": 0.0, "alert": "VISUAL_AMBER",     "action": "amber_flash"},
        2: {"t_start": 2.5, "alert": "VISUAL_RED_AUDIO", "action": "red_flash + audio + haptic"},
        3: {"t_start": 5.0, "alert": "CRITICAL",         "action": "continuous_alert + mrm_prepare"},
    }
    TOR_TIMEOUT_S = 8.0
    MIN_WAIT_BUDGET_S = 3.0   # nếu time budget thấp hơn mức này thì không nên chờ tiếp
```

### Minimal Risk Condition (MRC)

> **MRM/MRC không đồng nghĩa emergency stop.** Ưu tiên dừng có kiểm soát, giữ ổn định, tấp lề nếu hợp lệ.

```python
@dataclass
class MRCPlan:
    strategy: str                  # "COAST_TO_STOP" | "PULL_OVER_RIGHT" | "EMERGENCY_STOP"
    target_v_final: float          # 0.0 m/s
    max_decel_mps2: float          # 2.5~3.0 bình thường, <= 6.0 emergency
    max_jerk_mps3: float           # comfort + stability bound
    prefer_shoulder: bool
    hazard_lights: bool
    keep_lane_centering: bool
    estimated_stop_distance_m: float
    estimated_stop_time_s: float
    corridor_for_stop: str | None
```

```python
SAFE_STOP_SEQUENCE = [
    "ACTIVATE_HAZARDS",
    "MAINTAIN_STABLE_YAW",
    "SIGNAL_RIGHT_IF_SHOULDER_CLEAR",
    "DECELERATE_WITH_JERK_LIMIT",
    "STOP_FULLY",
    "SET_SAFE_STOP_FLAG",
]
```

---

## 6. Safety Contract — 4 lớp kiểm tra

```python
class SafetySupervisor:
    """
    Độc lập với Agent. Có quyền reject grant và revoke bất kỳ lúc nào.
    """

    def verify_contract(self, contract: ManeuverContract, world: WorldState) -> SafetyVerdict:
        verdict = SafetyVerdict(approved=True, reasons=[])

        # Lớp 1: physical hard constraints
        if contract.max_lateral_offset_m > 1.0:
            verdict.reject("S-001: lateral offset exceeds hard limit")
        if contract.max_speed_mps > 13.89:
            verdict.reject("S-002: speed cap exceeded")
        if contract.max_duration_s > 8.0:
            verdict.reject("S-003: contract duration exceeded")
        if contract.max_jerk_mps3 > 4.0:
            verdict.reject("S-004: jerk bound too high")

        # Lớp 2: freshness / ODD / sensor validity
        if world.world_age_ms > contract.freshness_required_ms:
            verdict.reject("S-005: world state stale")
        if not world.sync_ok:
            verdict.reject("S-006: sensor sync invalid")
        if world.odd_status != ODDStatus.IN_ODD:
            verdict.reject("S-007: not inside ODD for grant")
        if world.sensor_health == SensorHealth.FAULT:
            verdict.reject("S-008: sensor health fault")

        # Lớp 3: situational safety
        if world.min_ttc_s < max(2.0, contract.min_ttc_threshold_s * 1.5):
            verdict.reject("S-009: TTC too low at grant")
        if world.ego_lateral_error_m > 0.6:
            verdict.reject("S-010: ego not stable enough for handoff")
        if not world.preview_feasible:
            verdict.reject("S-011: preview trajectory infeasible")

        # Lớp 4: authority hygiene
        if contract.agent_confidence < 0.85:
            verdict.reject("S-012: agent confidence below floor")
        if self.memory.recent_revokes_last_60s > 2:
            verdict.reject("S-013: cooldown due to repeated revokes")

        return verdict

    def watch_frame(self, world: WorldState, contract: ManeuverContract) -> VetoSignal | None:
        if world.world_age_ms > contract.freshness_required_ms:
            return VetoSignal("S-V001: stale world state", severity="HARD")
        if world.odd_status != ODDStatus.IN_ODD:
            return VetoSignal("S-V002: ODD margin/exceeded during active contract", severity="HARD")
        if world.min_ttc_s < contract.min_ttc_threshold_s:
            return VetoSignal("S-V003: TTC collapsed", severity="HARD")
        if world.new_obstacle_score > contract.max_new_obstacle_score:
            return VetoSignal("S-V004: unexpected obstacle", severity="HARD")
        if not world.preview_feasible:
            return VetoSignal("S-V005: trajectory infeasible", severity="HARD")
        if self.contract_timer_expired(contract):
            return VetoSignal("S-V006: contract expired", severity="SOFT")
        return None
```

---

## 7. Interface / Message Schema

### `world_state_t`

```python
@dataclass
class WorldState:
    frame_id: int
    timestamp_s: float
    world_age_ms: int
    sync_ok: bool
    sensor_health: SensorHealth      # OK | DEGRADED_BUT_VALID | FAULT

    ego_v_mps: float
    ego_a_mps2: float
    ego_lane_id: str
    ego_lateral_error_m: float
    corridor_clear: bool

    min_ttc_s: float
    new_obstacle_score: float
    weather_visibility_m: float
    lane_change_permission: bool
    drivable_envelope: DrivableEnvelope

    odd_status: ODDStatus            # IN_ODD | ODD_MARGIN | ODD_EXCEEDED
    time_to_odd_exit_s: float | None

    preview_feasible: bool
    human_driver_available: bool
    human_input_detected: bool
```

### `takeover_request_t`

```python
@dataclass
class TakeoverRequest:
    request_id: str
    source: str                      # "AGENT" | "BASELINE" | "SUPERVISOR"
    reason: str
    requested_maneuver: str | None
    confidence: float | None
    issued_at_frame: int
    validity_s: float
```

### `authority_context_t`

```python
@dataclass
class AuthorityContext:
    current_state: AuthorityState
    active_authority: str            # "BASELINE" | "AGENT" | "HUMAN" | "MRM"
    active_contract: ManeuverContract | None
    contract_remaining_s: float | None
    baseline_weight: float
    agent_weight: float
    revoke_reason: str | None
    cooldown_remaining_s: float
    tor_active: bool
    tor_elapsed_s: float | None
```

### `trajectory_request_t`

```python
@dataclass
class TrajectoryRequest:
    source: str                      # "BASELINE" | "CONTRACT_RESOLVER" | "MRM" | "HANDOFF"
    tactical_intent: str

    v_max_mps: float
    a_long_max_mps2: float
    a_lat_max_mps2: float
    jerk_max_mps3: float
    lateral_bound_m: float
    drivable_envelope: DrivableEnvelope

    target_lane_id: str | None
    target_v_desired_mps: float
    horizon_s: float

    reference_path: list[Pose2D] | None
    cost_profile: str | None         # "BASELINE" | "HANDOFF" | "MRM"
```

### `tor_signal_t`

```python
@dataclass
class TORSignal:
    level: int
    started_at_s: float
    elapsed_s: float
    timeout_s: float
    human_input_detected: bool
```

---

## 8. Pseudocode — Authority Arbitration Loop (10 Hz)

```python
def authority_arbitration_loop(
    world: WorldState,
    asm: AuthorityStateMachine,
    baseline_detector: BaselineDetector,
    human_override: HumanOverrideMonitor,
    agent: Stage9AgentAdapter,
    supervisor: SafetySupervisor,
    baseline: BaselinePlanner,
    contract_resolver: ContractResolver,
    handoff_planner: HandoffPlanner,
    mpc: MPCOptimizer,
    tor: TORManager,
    mrm: MRMExecutor,
) -> ActuatorCommand:

    ctx = asm.get_context()

    # Highest priority: explicit human input
    if human_override.detect(world):
        if ctx.current_state != HUMAN_CONTROL_ACTIVE:
            asm.transition_to(HUMAN_CONTROL_ACTIVE, reason="manual_override")

    if ctx.current_state == SAFE_STOP:
        return mpc.execute(mrm.get_standstill_request(world))

    if ctx.current_state == HUMAN_CONTROL_ACTIVE:
        if human_override.released() and supervisor.autonomy_reengage_allowed(world):
            asm.transition_to(BASELINE_CONTROL, reason="manual_release")
            return mpc.execute(baseline.plan(world))
        if supervisor.must_enter_mrm_from_manual(world):
            asm.transition_to(MINIMAL_RISK_MANEUVER, reason="manual_fault_fallback")
            return mpc.execute(mrm.compute(world).to_trajectory_request())
        return human_override.current_command()

    if ctx.current_state == MINIMAL_RISK_MANEUVER:
        mrm_plan = mrm.compute(world)
        cmd = mpc.execute(mrm_plan.to_trajectory_request())
        if mrm.is_full_stop_reached(world):
            asm.transition_to(SAFE_STOP, reason="mrm_complete")
        return cmd

    if ctx.current_state == TOR_ACTIVE:
        tor.tick(world)
        if human_override.detect(world):
            asm.transition_to(HUMAN_CONTROL_ACTIVE, reason="driver_took_over")
            return human_override.current_command()
        if tor.timed_out() or world.time_to_odd_exit_s is not None and world.time_to_odd_exit_s < tor.MIN_WAIT_BUDGET_S:
            asm.transition_to(MINIMAL_RISK_MANEUVER, reason="tor_timeout_or_low_budget")
            return mpc.execute(mrm.compute(world).to_trajectory_request())
        return mpc.execute(baseline.degraded_hold(world))

    if ctx.current_state == BASELINE_CONTROL:
        if world.odd_status == ODDStatus.ODD_EXCEEDED or supervisor.critical_fault(world):
            asm.transition_to(MINIMAL_RISK_MANEUVER, reason="odd_exceeded_or_critical_fault")
            return mpc.execute(mrm.compute(world).to_trajectory_request())

        if world.odd_status == ODDStatus.ODD_MARGIN and world.human_driver_available and (world.time_to_odd_exit_s or 0.0) >= 5.0:
            asm.transition_to(TOR_ACTIVE, reason="odd_margin_tor")
            tor.start(world)
            return mpc.execute(baseline.degraded_hold(world))

        baseline_req = baseline.plan(world)

        if baseline_detector.is_stuck(world) and supervisor.forward_takeover_allowed(world):
            contract = agent.propose_contract(world)
            if contract is not None:
                asm.transition_to(AGENT_REQUESTING_AUTHORITY, contract=contract)

        return mpc.execute(baseline_req)

    if ctx.current_state == AGENT_REQUESTING_AUTHORITY:
        contract = ctx.active_contract
        verdict = supervisor.verify_contract(contract, world)

        if not verdict.approved:
            asm.transition_to(BASELINE_CONTROL, reason=verdict.primary_reason())
            return mpc.execute(baseline.plan(world))

        if asm.request_timed_out():
            asm.transition_to(BASELINE_CONTROL, reason="request_timeout")
            return mpc.execute(baseline.plan(world))

        asm.transition_to(SUPERVISED_EXECUTION, contract=contract)
        return mpc.execute(baseline.plan(world))

    if ctx.current_state == SUPERVISED_EXECUTION:
        baseline_req = baseline.plan(world)
        agent_intent = agent.get_intent(world, ctx.active_contract)
        agent_req = contract_resolver.resolve(agent_intent, ctx.active_contract)

        preview = handoff_planner.preview(
            baseline_req=baseline_req,
            candidate_req=agent_req,
            alpha=0.5,
        )

        veto = supervisor.watch_frame(world, ctx.active_contract)
        if veto is not None or not mpc.preview_feasible(preview):
            asm.transition_to(BASELINE_CONTROL, reason=(veto.reason if veto else "preview_infeasible"))
            return mpc.execute(baseline_req)

        asm.increment_supervised_frames()
        if asm.supervised_frames >= 3:
            asm.transition_to(AGENT_ACTIVE_BOUNDED)

        return mpc.execute(preview)

    if ctx.current_state == AGENT_ACTIVE_BOUNDED:
        veto = supervisor.watch_frame(world, ctx.active_contract)
        if veto is not None:
            asm.transition_to(AUTHORITY_REVOKE_PENDING, reason=veto.reason)
            asm.start_revoke_window()
            return mpc.execute(baseline.plan(world))

        agent_intent = agent.get_intent(world, ctx.active_contract)
        agent_req = contract_resolver.resolve(agent_intent, ctx.active_contract)
        return mpc.execute(agent_req)

    if ctx.current_state == AUTHORITY_REVOKE_PENDING:
        baseline_req = baseline.plan(world)
        last_agent_req = asm.last_agent_request()
        revoke_alpha = asm.current_revoke_alpha()    # 1.0 -> 0.0 over 3~5 frames

        transition_req = handoff_planner.preview(
            baseline_req=baseline_req,
            candidate_req=last_agent_req,
            alpha=revoke_alpha,
        )

        asm.increment_revoke_frame()
        if asm.revoke_complete():
            if baseline.is_healthy(world):
                asm.transition_to(BASELINE_CONTROL, reason="revoke_complete_to_baseline")
            elif world.human_driver_available and (world.time_to_odd_exit_s or 0.0) >= 5.0:
                asm.transition_to(TOR_ACTIVE, reason="baseline_unhealthy_tor")
                tor.start(world)
            else:
                asm.transition_to(MINIMAL_RISK_MANEUVER, reason="baseline_unhealthy_mrm")

        return mpc.execute(transition_req)

    # defensive fallback
    asm.transition_to(MINIMAL_RISK_MANEUVER, reason="unexpected_state_fallback")
    return mpc.execute(mrm.compute(world).to_trajectory_request())
```

### Ghi chú pseudocode

- `baseline_detector` và `human_override` được truyền tường minh, không dùng biến ẩn.
- Nhánh reject trong `AGENT_REQUESTING_AUTHORITY` quay về baseline ngay lập tức.
- Revoke không dùng `fallback_intent` do Agent tự định nghĩa; chỉ dùng **baseline request + last safe agent-derived request** để ramp về baseline.
- Warm handoff và revoke đều ở tầng `HandoffPlanner`, không blend actuator trực tiếp.

---

## 9. Tích Hợp Vào Stack Hiện Tại

### Điểm chèn vào pipeline

```
Current stack:
  world_state -> Stage8AssistAdapter -> advisory behavior -> baseline MPC

Stage 9 stack:
  world_state
      │
      ├─► BaselineDetector
      ├─► AuthorityStateMachine
      ├─► Stage9AgentAdapter (reuse Stage8 logic, but bounded)
      ├─► SafetySupervisor
      ├─► HumanOverrideMonitor
      ├─► TORManager / MRMExecutor
      ├─► ContractResolver
      ├─► HandoffPlanner
      └─► MPC
              └─► CARLA
```

### Boundaries rõ ràng

- **BEVFusion / World State**: read-only source of truth cho Stage 9.
- **Stage9AgentAdapter**: chỉ sinh `ManeuverContract` và `maneuver intent`.
- **SafetySupervisor**: gate cuối cùng trước grant và watcher mọi frame.
- **HandoffPlanner**: thực hiện chuyển tham chiếu mượt.
- **MPC**: lớp cuối cùng tạo trajectory và actuator.
- **HumanOverrideMonitor**: thắng tất cả autonomy states khi phát hiện người lái can thiệp.

### Những gì không làm

- Không cho Agent đụng trực tiếp `steer/throttle/brake`.
- Không bypass MPC.
- Không grant mới khi `odd_status != IN_ODD`.
- Không dùng world state stale để cấp quyền.

---

## 10. Evaluation Metrics & Scenario Tests

### Metrics bắt buộc

| Metric | Công thức | Target |
|--------|-----------|--------|
| **Takeover Success Rate (TSR)** | `successful_takeovers / total_takeovers` | ≥ 90% |
| **Stuck Resolution Rate (SRR)** | `resolved_stucks / total_stucks` | ≥ 85% |
| **False Takeover Rate (FTR)** | `unnecessary_takeovers / total_takeovers` | ≤ 5% |
| **Graceful Revoke Rate (GRR)** | `smooth_revokes / total_revokes` | ≥ 95% |
| **Authority Oscillation Index (AOI)** | `state_transitions / minute` | ≤ 2 /min |
| **Stale Grant Count** | số grant dùng world stale | = 0 |
| **ODD-Grant Violation Count** | grant khi `odd_status != IN_ODD` | = 0 |
| **Handoff Jerk Peak** | max jerk trong cửa sổ cấp/thu quyền | theo comfort bound |
| **Steering Rate Spike** | max d(steer)/dt khi handoff | thấp hơn baseline threshold |
| **Time-to-Revoke** | từ veto đến revoke effective | càng thấp càng tốt |
| **MRM Success Rate** | MRM đến stop an toàn / tổng MRM | ≥ 99% |
| **TOR Response Latency** | thời gian người lái phản hồi | log + phân bố |

### Scenario Test Matrix

| Mã | Tình huống | Expected State Path | Pass Condition |
|----|-----------|--------------------|----------------|
| T9-001 | Xe đỗ giữa làn, baseline bị kẹt | BASELINE → REQUESTING → SUPERVISED → ACTIVE → REVOKE → BASELINE | vượt chướng ngại an toàn |
| T9-002 | Baseline oscillation > 3 lần | BASELINE → REQUESTING → ACTIVE | lateral error giảm < 0.3 m |
| T9-003 | Agent confidence tụt giữa chừng | ACTIVE → REVOKE → BASELINE | không jerk spike |
| T9-004 | TTC collapse khi đang ACTIVE | ACTIVE → REVOKE → MRM | no collision |
| T9-005 | Contract hết thời hạn bình thường | ACTIVE → REVOKE → BASELINE | revoke mượt |
| T9-006 | Baseline stuck nhưng `odd_status = ODD_MARGIN` | BASELINE → TOR hoặc MRM | không grant Agent |
| T9-007 | `ODD_EXCEEDED` đột ngột | bất kỳ autonomy state → MRM → SAFE_STOP | dừng an toàn |
| T9-008 | World state stale trong lúc request | REQUESTING → BASELINE | reject ngay |
| T9-009 | Supervisor preview infeasible | SUPERVISED → BASELINE | không grant |
| T9-010 | TOR timeout, không phản hồi | TOR → MRM → SAFE_STOP | stop trong corridor hợp lệ |
| T9-011 | Human override trong MRM | MRM → HUMAN_CONTROL_ACTIVE | human authority thắng |
| T9-012 | Nhiều request liên tiếp | REQUESTING → REJECT × N | cooldown active |

---

## 11. Risk Analysis + Mitigation

| Rủi ro | Mức độ | Mitigation |
|--------|--------|-----------|
| Authority oscillation | CAO | cooldown 15–30s, AOI gate, hysteresis window |
| Grant gần biên ODD | CAO | cấm grant nếu `odd_status != IN_ODD` |
| World state stale / unsynced | CAO | freshness gate, `sync_ok` gate, zero stale grants |
| Preview feasible nhưng runtime infeasible | CAO | supervisor watch every frame + immediate revoke |
| Handoff gây discontinuity | TRUNG BÌNH | HandoffPlanner ở tầng reference, jerk/steering-rate bounds |
| Revoke vẫn phụ thuộc Agent | CAO | revoke chỉ ramp về baseline hoặc vào MRM |
| Human takeover semantics bị lẫn với baseline | CAO | state riêng `HUMAN_CONTROL_ACTIVE` |
| LLM latency khi ACTIVE | TRUNG BÌNH | cached intent trong contract window, bounded refresh rate |
| BaselineDetector false positive | TRUNG BÌNH | sliding window + multi-signal confirmation |
| TOR chờ quá lâu | CAO | `MIN_WAIT_BUDGET_S`, timeout -> MRM |
| MRM dừng sai vị trí | TRUNG BÌNH | drivable envelope + shoulder validation |

---

## 12. Implementation Roadmap

### Phase 1 — Core authority skeleton

```
1. schemas.py
2. authority_state_machine.py (9 states)
3. baseline_detector.py
4. maneuver_contract.py
5. safety_supervisor.py
6. contract_resolver.py
7. handoff_planner.py
8. authority_arbiter.py
9. authority_logger.py
```

**Gate:** T9-001, T9-003, T9-005, T9-008, T9-009 pass.

### Phase 2 — Fallback / human path

```
10. tor_manager.py
11. minimal_risk_maneuver.py
12. human_override_monitor.py
13. integrate SAFE_STOP + TOR + MRM path
```

**Gate:** T9-006, T9-007, T9-010, T9-011 pass.

### Phase 3 — Evaluation and tuning

```
14. stage9_evaluator.py
15. metrics logging: jerk, steering-rate, stale-grant, ODD-grant violations
16. scenario campaign T9-001 -> T9-012
17. tune thresholds: stuck_score, confidence floor, cooldown, handoff frames
```

**Promotion gate:**
- `TSR >= 90%`
- `SRR >= 85%`
- `AOI <= 2/min`
- `Stale Grant Count = 0`
- `ODD-Grant Violation Count = 0`
- `MRM Success Rate >= 99%`

---

## Tóm Tắt Nguyên Tắc Thiết Kế

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 9 DESIGN PRINCIPLES — v2                         │
│                                                                            │
│  1. Agent chỉ nhận bounded maneuver authority (L1)                        │
│  2. Grant chỉ hợp lệ khi đang IN_ODD + world fresh + preview feasible     │
│  3. Safety Supervisor là lớp veto độc lập, chạy mọi frame                 │
│  4. Warm handoff / revoke chỉ blend ở tầng reference, không blend actuator│
│  5. Human takeover là state riêng: HUMAN_CONTROL_ACTIVE                   │
│  6. TOR được ưu tiên khi còn đủ thời gian; không đủ thì vào MRM ngay      │
│  7. Revoke không dùng fallback intent do Agent tự định nghĩa              │
│  8. Baseline -> TOR/MRM là đường chính khi sát biên ODD                   │
│  9. MPC luôn là lớp cuối cùng tạo trajectory và actuator                  │
│ 10. Log đầy đủ mọi grant / reject / revoke / TOR / MRM                    │
└────────────────────────────────────────────────────────────────────────────┘
```
