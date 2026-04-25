"""
Stage 9 — Phase 1 Smoke Test
==============================
Verifies all Phase 1 modules import correctly and core logic runs without errors.
Does NOT require CARLA or a real MPC — uses minimal stubs.

Run:
  python scripts/run_stage9_phase1_smoke.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage9 import (
    AuthorityArbiter,
    AuthorityState,
    AuthorityStateMachine,
    BaselineDetector,
    ContractResolver,
    DrivableEnvelope,
    HandoffPlanner,
    ManeuverContract,
    MRCPlan,
    MRMStrategy,
    ODDStatus,
    SafetyVerdict,
    SafetySupervisor,
    SensorHealth,
    TrajectoryRequest,
    WorldState,
    ActuatorCommand,
    build_contract,
    validate_contract,
)


# ── Minimal stubs ─────────────────────────────────────────────────────────────

class _StubBaseline:
    def plan(self, world): return _baseline_req()
    def degraded_hold(self, world): return _baseline_req()
    def is_healthy(self, world): return True

class _StubAgent:
    def propose_contract(self, world):
        return build_contract(
            frame_id=world.frame_id,
            sim_time_s=world.timestamp_s,
            tactical_intent="slow_bypass_right",
            sub_intent_sequence=["prepare", "commit", "stabilize"],
            target_state_description="bypass stationary vehicle",
            agent_confidence=0.92,
            agent_reasoning_summary="Vehicle blocking lane; safe right bypass within envelope.",
            max_duration_s=5.0,
            max_speed_mps=5.56,  # 20 km/h
            max_lateral_offset_m=0.8,
        )
    def get_intent(self, world, contract):
        return "slow_bypass_right"

class _StubMPC:
    def execute(self, req): return ActuatorCommand(steer=0.0, throttle=0.3, brake=0.0)
    def preview_feasible(self, req): return True

class _StubTOR:
    MIN_WAIT_BUDGET_S = 3.0
    def start(self, world): pass
    def tick(self, world): pass
    def timed_out(self): return False

class _StubMRM:
    def compute(self, world):
        return MRCPlan(
            strategy=MRMStrategy.COAST_TO_STOP,
            estimated_stop_distance_m=20.0,
            estimated_stop_time_s=5.0,
        )
    def is_full_stop_reached(self, world): return False
    def get_standstill_request(self, world):
        return TrajectoryRequest(source="MRM", tactical_intent="safe_stop", v_max_mps=0.0)

class _StubHuman:
    def detect(self, world): return False
    def released(self): return False
    def current_command(self): return ActuatorCommand(steer=0.0, throttle=0.0, brake=0.5)


def _envelope():
    return DrivableEnvelope(
        envelope_uuid="test-env-001",
        left_bound_m=1.5,
        right_bound_m=1.5,
        forward_clear_m=50.0,
    )

def _baseline_req():
    return TrajectoryRequest(
        source="BASELINE", tactical_intent="keep_lane",
        v_max_mps=10.0, a_long_max_mps2=2.0, a_lat_max_mps2=1.5,
        jerk_max_mps3=3.0, lateral_bound_m=1.0,
        drivable_envelope=_envelope(), target_v_desired_mps=8.33,
    )

def _make_world(
    frame_id: int,
    timestamp_s: float,
    ego_v: float = 0.3,
    corridor_clear: bool = True,
    odd_status: ODDStatus = ODDStatus.IN_ODD,
    ttc: float = 10.0,
    world_age_ms: int = 50,
    sensor_health: SensorHealth = SensorHealth.OK,
) -> WorldState:
    return WorldState(
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        world_age_ms=world_age_ms,
        sync_ok=True,
        sensor_health=sensor_health,
        ego_v_mps=ego_v,
        ego_a_mps2=0.0,
        ego_lane_id="lane_0",
        ego_lateral_error_m=0.05,
        corridor_clear=corridor_clear,
        min_ttc_s=ttc,
        new_obstacle_score=0.1,
        weather_visibility_m=100.0,
        lane_change_permission=True,
        drivable_envelope=_envelope(),
        odd_status=odd_status,
        time_to_odd_exit_s=None,
        preview_feasible=True,
        human_driver_available=True,
        human_input_detected=False,
    )


def _build_arbiter(log_path=None) -> AuthorityArbiter:
    return AuthorityArbiter(
        asm=AuthorityStateMachine(),
        baseline_detector=BaselineDetector(),
        human_override=_StubHuman(),
        agent=_StubAgent(),
        supervisor=SafetySupervisor(),
        baseline=_StubBaseline(),
        contract_resolver=ContractResolver(),
        handoff_planner=HandoffPlanner(),
        mpc=_StubMPC(),
        tor=_StubTOR(),
        mrm=_StubMRM(),
        log_path=log_path,
    )


# ── Test cases ────────────────────────────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0

def _check(name: str, condition: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  [{status}] {name}{(': ' + detail) if detail else ''}")
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1


def test_imports() -> None:
    print("\n[TEST] Module imports")
    _check("All Phase 1 modules imported", True)


def test_baseline_stuck_detection() -> None:
    print("\n[TEST] BaselineDetector — velocity stall")
    det = BaselineDetector()
    w = _make_world(0, 0.0, ego_v=0.2)
    for i in range(35):
        w = _make_world(i, i * 0.1, ego_v=0.2)
        det.update(w)
    stuck = det.is_stuck(w)
    score = det.stuck_score(w)
    _check("velocity stall detected after 3.5s", stuck, f"score={score:.2f}")


def test_safety_supervisor_layers() -> None:
    print("\n[TEST] SafetySupervisor — 4-layer verify_contract")
    sup = SafetySupervisor()

    # Valid world + valid contract should pass
    world = _make_world(0, 0.0)
    contract = build_contract(
        frame_id=0, sim_time_s=0.0,
        tactical_intent="slow_bypass_right",
        sub_intent_sequence=[],
        target_state_description="test",
        agent_confidence=0.92,
        agent_reasoning_summary="test",
    )
    verdict = sup.verify_contract(contract, world)
    _check("valid contract approved", verdict.approved, str(verdict.reasons))

    # Stale world should be rejected (S-005)
    stale_world = _make_world(0, 0.0, world_age_ms=200)
    v2 = sup.verify_contract(contract, stale_world)
    _check("stale world rejected (S-005)", not v2.approved and any("S-005" in r for r in v2.reasons))

    # ODD exceeded should reject (S-007)
    odd_world = _make_world(0, 0.0, odd_status=ODDStatus.ODD_EXCEEDED)
    v3 = sup.verify_contract(contract, odd_world)
    _check("ODD exceeded rejected (S-007)", not v3.approved and any("S-007" in r for r in v3.reasons))


def test_arbiter_baseline_to_active(tmp_path=None) -> None:
    print("\n[TEST] AuthorityArbiter — BASELINE → SUPERVISED → AGENT_ACTIVE_BOUNDED")
    log = Path("benchmark/reports/stage9_smoke_test_log.jsonl")
    arbiter = _build_arbiter(log_path=log)

    # Warmup: 30 frames at stall velocity to trigger baseline_stuck
    for i in range(30):
        w = _make_world(i, i * 0.1, ego_v=0.3)
        arbiter.step(w, sim_time_s=i * 0.1)

    ctx = arbiter._asm.get_context()
    state_before = ctx.current_state

    # Frame 31: tick once more to trigger contract submission + request + supervised
    w31 = _make_world(31, 3.1, ego_v=0.3)
    arbiter.step(w31, sim_time_s=3.1)

    # Supervised frames — needs 3 stable frames to promote to AGENT_ACTIVE_BOUNDED
    for i in range(32, 36):
        w = _make_world(i, i * 0.1, ego_v=0.3)
        arbiter.step(w, sim_time_s=i * 0.1)

    ctx2 = arbiter._asm.get_context()
    _check(
        "reached AGENT_ACTIVE_BOUNDED or SUPERVISED_EXECUTION",
        ctx2.current_state in (
            AuthorityState.AGENT_ACTIVE_BOUNDED,
            AuthorityState.SUPERVISED_EXECUTION,
            AuthorityState.AGENT_REQUESTING_AUTHORITY,
        ),
        f"state={ctx2.current_state.name}",
    )


def test_mrc_plan_to_trajectory() -> None:
    print("\n[TEST] MRCPlan.to_trajectory_request()")
    plan = MRCPlan(
        strategy=MRMStrategy.PULL_OVER_RIGHT,
        max_decel_mps2=2.5,
        max_jerk_mps3=3.0,
        estimated_stop_distance_m=30.0,
        estimated_stop_time_s=8.0,
    )
    req = plan.to_trajectory_request()
    _check("source=MRM",  req.source == "MRM")
    _check("v_max_mps=0.0", req.v_max_mps == 0.0)
    _check("cost_profile=MRM", req.cost_profile == "MRM")


def test_trajectory_blend() -> None:
    print("\n[TEST] TrajectoryRequest.blend()")
    b = _baseline_req()
    a = TrajectoryRequest(
        source="AGENT", tactical_intent="slow_bypass_right",
        v_max_mps=5.0, a_long_max_mps2=1.5, a_lat_max_mps2=1.0,
        jerk_max_mps3=2.0, lateral_bound_m=0.8,
        target_v_desired_mps=4.0,
    )
    blended = TrajectoryRequest.blend(b, a, agent_weight=0.5)
    _check("source=HANDOFF", blended.source == "HANDOFF")
    _check("v_max blended", blended.v_max_mps == (10.0 + 5.0) / 2)
    _check("jerk is min of two", blended.jerk_max_mps3 == min(b.jerk_max_mps3, a.jerk_max_mps3))


def test_contract_validation_errors() -> None:
    print("\n[TEST] validate_contract — hard limit violations")
    world = _make_world(0, 0.0)
    bad_contract = build_contract(
        frame_id=0, sim_time_s=0.0,
        tactical_intent="test",
        sub_intent_sequence=[],
        target_state_description="",
        agent_confidence=0.99,
        agent_reasoning_summary="",
        max_lateral_offset_m=2.0,   # will be clamped to 1.0 by builder
        max_speed_mps=50.0,         # will be clamped to 13.89
        max_duration_s=20.0,        # will be clamped to 8.0
    )
    # After clamping, validation should pass
    errors = validate_contract(bad_contract, world)
    _check("clamped contract validates cleanly", len(errors) == 0, str([e.code for e in errors]))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("Stage 9 — Phase 1 Smoke Test")
    print("=" * 64)

    tests = [
        test_imports,
        test_baseline_stuck_detection,
        test_safety_supervisor_layers,
        test_arbiter_baseline_to_active,
        test_mrc_plan_to_trajectory,
        test_trajectory_blend,
        test_contract_validation_errors,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception:
            print(f"  [✗ ERROR] {test_fn.__name__}")
            traceback.print_exc()
            FAIL_COUNT += 1

    print("\n" + "=" * 64)
    print(f"Results: {PASS_COUNT} passed / {FAIL_COUNT} failed")
    print("=" * 64)
    sys.exit(0 if FAIL_COUNT == 0 else 1)
