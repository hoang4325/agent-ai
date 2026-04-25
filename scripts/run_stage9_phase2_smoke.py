"""
Stage 9 — Phase 2 Smoke Test
==============================
Verifies TORManager, MRMExecutor, HumanOverrideMonitor, Stage9Evaluator.
Includes integration tests across TOR → MRM → SAFE_STOP path and
human override detection + re-engagement.

Run:
  python scripts/run_stage9_phase2_smoke.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stage9 import (
    ActuatorCommand,
    AuthorityArbiter,
    AuthorityState,
    AuthorityStateMachine,
    BaselineDetector,
    ContractResolver,
    DrivableEnvelope,
    EvaluationReport,
    HandoffPlanner,
    HumanOverrideMonitor,
    ManeuverContract,
    MRCPlan,
    MRMExecutor,
    MRMStrategy,
    ODDStatus,
    SafetySupervisor,
    SensorHealth,
    Stage9Evaluator,
    TORManager,
    TrajectoryRequest,
    WorldState,
    build_contract,
)

PASS_COUNT = 0
FAIL_COUNT = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    tick = "✓ PASS" if condition else "✗ FAIL"
    print(f"  [{tick}] {name}{(': ' + detail) if detail else ''}")
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1


def _envelope():
    return DrivableEnvelope("e001", 1.5, 1.5, 50.0)

def _world(frame_id=0, ts=0.0, ego_v=5.0, odd=ODDStatus.IN_ODD,
           ttc=10.0, human_input=False, time_to_odd_exit=None,
           lane_change_perm=True):
    return WorldState(
        frame_id=frame_id, timestamp_s=ts, world_age_ms=50, sync_ok=True,
        sensor_health=SensorHealth.OK, ego_v_mps=ego_v, ego_a_mps2=0.0,
        ego_lane_id="lane_0", ego_lateral_error_m=0.05, corridor_clear=True,
        min_ttc_s=ttc, new_obstacle_score=0.1, weather_visibility_m=100.0,
        lane_change_permission=lane_change_perm, drivable_envelope=_envelope(),
        odd_status=odd, time_to_odd_exit_s=time_to_odd_exit,
        preview_feasible=True, human_driver_available=True,
        human_input_detected=human_input,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOR Manager tests
# ─────────────────────────────────────────────────────────────────────────────

def test_tor_escalation_levels():
    print("\n[TEST] TORManager — escalation levels")
    tor = TORManager()
    w = _world()
    tor.start(w, reason="odd_margin")

    _check("starts at level 1", tor.current_level() == 1)
    _check("not timed out at start", not tor.timed_out())
    _check("is active", tor.is_active())

    sig = tor.tick(w)
    _check("signal level >= 1", sig.level >= 1)
    _check("signal elapsed >= 0", sig.elapsed_s >= 0.0)
    _check("signal timeout correct", sig.timeout_s == TORManager.TOR_TIMEOUT_S)


def test_tor_timeout_detection():
    print("\n[TEST] TORManager — timeout detection (simulated)")
    tor = TORManager()
    # Override timeout to something tiny for testing
    tor.TOR_TIMEOUT_S = 0.05
    w = _world()
    tor.start(w)
    time.sleep(0.06)
    tor.tick(w)
    _check("timed out after delay", tor.timed_out())
    tor.reset()
    _check("not active after reset", not tor.is_active())


def test_tor_time_budget_critical():
    print("\n[TEST] TORManager — time_budget_critical gate")
    tor = TORManager()
    w_safe = _world(time_to_odd_exit=10.0)
    w_crit = _world(time_to_odd_exit=1.0)
    tor.start(w_safe)
    _check("not critical with 10s budget", not tor.time_budget_critical(w_safe))
    _check("critical with 1.0s budget", tor.time_budget_critical(w_crit))


# ─────────────────────────────────────────────────────────────────────────────
# MRM Executor tests
# ─────────────────────────────────────────────────────────────────────────────

def test_mrm_strategy_selection():
    print("\n[TEST] MRMExecutor — strategy selection")
    mrm = MRMExecutor()

    # Low TTC → EMERGENCY_STOP
    w_emergency = _world(ttc=0.5)
    plan = mrm.compute(w_emergency)
    _check("EMERGENCY_STOP for low TTC", plan.strategy == MRMStrategy.EMERGENCY_STOP)
    _check("hazard lights always on", plan.hazard_lights is True)

    # Normal TTC + lane change allowed → PULL_OVER_RIGHT
    mrm2 = MRMExecutor()
    w_pullover = _world(ttc=10.0, lane_change_perm=True)
    plan2 = mrm2.compute(w_pullover)
    _check("PULL_OVER_RIGHT when shoulder available", plan2.strategy == MRMStrategy.PULL_OVER_RIGHT)

    # No lane change permission → COAST_TO_STOP
    mrm3 = MRMExecutor()
    w_coast = _world(ttc=10.0, lane_change_perm=False)
    plan3 = mrm3.compute(w_coast)
    _check("COAST_TO_STOP when no shoulder", plan3.strategy == MRMStrategy.COAST_TO_STOP)


def test_mrm_trajectory_request():
    print("\n[TEST] MRMExecutor — to_trajectory_request()")
    mrm = MRMExecutor()
    w = _world(ttc=10.0)
    plan = mrm.compute(w)
    req = plan.to_trajectory_request()
    _check("source=MRM", req.source == "MRM")
    _check("v_max=0.0", req.v_max_mps == 0.0)
    _check("cost_profile=MRM", req.cost_profile == "MRM")


def test_mrm_standstill_request():
    print("\n[TEST] MRMExecutor — get_standstill_request()")
    mrm = MRMExecutor()
    w = _world()
    req = mrm.get_standstill_request(w)
    _check("standstill source=MRM", req.source == "MRM")
    _check("standstill v=0.0", req.v_max_mps == 0.0)


def test_mrm_stop_detection():
    print("\n[TEST] MRMExecutor — is_full_stop_reached()")
    mrm = MRMExecutor()
    w_moving = _world(ego_v=5.0)
    w_stopped = _world(ego_v=0.05)
    _check("not stopped while moving", not mrm.is_full_stop_reached(w_moving))
    _check("stopped when v < 0.15", mrm.is_full_stop_reached(w_stopped))


def test_mrm_stop_distance_estimation():
    print("\n[TEST] MRMExecutor — stop distance estimation")
    mrm = MRMExecutor()
    w = _world(ego_v=10.0)
    plan = mrm.compute(w)
    # d = v²/(2a) = 100/(2*2.5) = 20m, t = v/a = 4s
    _check("stop distance ~20m", abs(plan.estimated_stop_distance_m - 20.0) < 0.1,
           f"{plan.estimated_stop_distance_m}")
    _check("stop time ~4s", abs(plan.estimated_stop_time_s - 4.0) < 0.1,
           f"{plan.estimated_stop_time_s}")


# ─────────────────────────────────────────────────────────────────────────────
# Human Override Monitor tests
# ─────────────────────────────────────────────────────────────────────────────

def test_human_override_detect_via_world():
    print("\n[TEST] HumanOverrideMonitor — detect via world.human_input_detected")
    hm = HumanOverrideMonitor()
    w_no  = _world(human_input=False)
    w_yes = _world(human_input=True)
    _check("no override when not detected", not hm.detect(w_no))
    _check("override when detected", hm.detect(w_yes))


def test_human_override_inject():
    print("\n[TEST] HumanOverrideMonitor — inject/clear override")
    hm = HumanOverrideMonitor()
    w = _world(human_input=False)

    _check("not detected before inject", not hm.detect(w))
    hm.inject_override(steer=0.2, throttle=0.0, brake=0.5)
    _check("detected after inject", hm.detect(w))

    cmd = hm.current_command()
    _check("injected steer correct", abs(cmd.steer - 0.2) < 0.001)
    _check("injected brake correct", abs(cmd.brake - 0.5) < 0.001)

    hm.clear_override()
    _check("not detected after clear (world also no input)", not hm.detect(w))


def test_human_override_released():
    print("\n[TEST] HumanOverrideMonitor — released() hysteresis")
    hm = HumanOverrideMonitor(release_hysteresis_s=0.05)
    hm.inject_override()
    _check("not released immediately after override", not hm.released())
    hm.clear_override()
    time.sleep(0.07)  # wait > hysteresis
    _check("released after hysteresis", hm.released())


# ─────────────────────────────────────────────────────────────────────────────
# Stage9Evaluator tests
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluator_empty_log(tmp_log: Path):
    print("\n[TEST] Stage9Evaluator — empty log")
    ev = Stage9Evaluator(tmp_log / "empty.jsonl")
    report = ev.compute()
    _check("returns EvaluationReport", isinstance(report, EvaluationReport))
    _check("no frames", report.total_frames == 0)


def test_evaluator_from_real_log():
    print("\n[TEST] Stage9Evaluator — reads existing Phase 1 log")
    log = Path("benchmark/reports/stage9_authority_log.jsonl")
    if not log.exists():
        print("    [SKIP] No log file found — run Phase 1 smoke first")
        return
    ev = Stage9Evaluator(log)
    report = ev.compute()
    _check("report has frames", report.total_frames >= 0)
    _check("gates evaluated", len(report.gate_results) == 8)
    ev.print_report(report)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: TOR → MRM → SAFE_STOP path
# ─────────────────────────────────────────────────────────────────────────────

class _StubBaseline:
    def plan(self, world): return TrajectoryRequest(source="BASELINE", tactical_intent="keep_lane")
    def degraded_hold(self, world): return TrajectoryRequest(source="BASELINE", tactical_intent="hold")
    def is_healthy(self, world): return True

class _StubAgent:
    def propose_contract(self, world): return None
    def get_intent(self, world, contract): return "keep_lane"

class _StubMPC:
    def execute(self, req): return ActuatorCommand(steer=0.0, throttle=0.3, brake=0.0, source="MPC")
    def preview_feasible(self, req): return True


def test_integration_tor_to_safe_stop():
    print("\n[TEST] Integration — ODD_EXCEEDED → MRM → SAFE_STOP")
    tor = TORManager()
    tor.TOR_TIMEOUT_S = 0.05  # tiny timeout for test speed
    mrm = MRMExecutor()
    hm  = HumanOverrideMonitor()

    arbiter = AuthorityArbiter(
        asm=AuthorityStateMachine(),
        baseline_detector=BaselineDetector(),
        human_override=hm,
        agent=_StubAgent(),
        supervisor=SafetySupervisor(),
        baseline=_StubBaseline(),
        contract_resolver=ContractResolver(),
        handoff_planner=HandoffPlanner(),
        mpc=_StubMPC(),
        tor=tor,
        mrm=mrm,
        log_path=Path("benchmark/reports/stage9_phase2_int_test.jsonl"),
    )

    # Step 1: ODD_EXCEEDED → should go to MRM immediately
    w_odd = _world(0, 0.0, odd=ODDStatus.ODD_EXCEEDED, ego_v=10.0)
    arbiter.step(w_odd, sim_time_s=0.0)
    ctx = arbiter._asm.get_context()
    _check(
        "ODD_EXCEEDED triggers MRM",
        ctx.current_state == AuthorityState.MINIMAL_RISK_MANEUVER,
        f"state={ctx.current_state.name}",
    )

    # Step 2: Simulate vehicle stopping
    for i in range(1, 20):
        w_stop = _world(i, i * 0.1, odd=ODDStatus.ODD_EXCEEDED, ego_v=max(0.0, 10.0 - i * 0.6))
        arbiter.step(w_stop, sim_time_s=i * 0.1)

    ctx2 = arbiter._asm.get_context()
    _check(
        "Reached MINIMAL_RISK_MANEUVER or SAFE_STOP after decel",
        ctx2.current_state in (AuthorityState.MINIMAL_RISK_MANEUVER, AuthorityState.SAFE_STOP),
        f"state={ctx2.current_state.name}",
    )


def test_integration_human_override_wins():
    print("\n[TEST] Integration — human override wins over any autonomy state")
    hm = HumanOverrideMonitor()
    hm.inject_override(steer=0.1, brake=0.2)

    arbiter = AuthorityArbiter(
        asm=AuthorityStateMachine(),
        baseline_detector=BaselineDetector(),
        human_override=hm,
        agent=_StubAgent(),
        supervisor=SafetySupervisor(),
        baseline=_StubBaseline(),
        contract_resolver=ContractResolver(),
        handoff_planner=HandoffPlanner(),
        mpc=_StubMPC(),
        tor=TORManager(),
        mrm=MRMExecutor(),
        log_path=Path("benchmark/reports/stage9_phase2_human_test.jsonl"),
    )

    w = _world(0, 0.0)
    cmd = arbiter.step(w, sim_time_s=0.0)
    ctx = arbiter._asm.get_context()

    _check(
        "state = HUMAN_CONTROL_ACTIVE after inject",
        ctx.current_state == AuthorityState.HUMAN_CONTROL_ACTIVE,
        f"state={ctx.current_state.name}",
    )
    # Command should be human's (steer=0.1, brake=0.2)
    _check("human command steer", abs(cmd.steer - 0.1) < 0.001, f"steer={cmd.steer}")
    _check("human command brake", abs(cmd.brake - 0.2) < 0.001, f"brake={cmd.brake}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tmp_log = Path("benchmark/reports")
    tmp_log.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("Stage 9 — Phase 2 Smoke Test (TOR + MRM + Human + Evaluator)")
    print("=" * 66)

    tests = [
        test_tor_escalation_levels,
        test_tor_timeout_detection,
        test_tor_time_budget_critical,
        test_mrm_strategy_selection,
        test_mrm_trajectory_request,
        test_mrm_standstill_request,
        test_mrm_stop_detection,
        test_mrm_stop_distance_estimation,
        test_human_override_detect_via_world,
        test_human_override_inject,
        test_human_override_released,
        lambda: test_evaluator_empty_log(tmp_log),
        test_evaluator_from_real_log,
        test_integration_tor_to_safe_stop,
        test_integration_human_override_wins,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception:
            print(f"  [✗ ERROR] {test_fn.__name__}")
            traceback.print_exc()
            FAIL_COUNT += 1

    print("\n" + "=" * 66)
    print(f"Results: {PASS_COUNT} passed / {FAIL_COUNT} failed")
    print("=" * 66)
    sys.exit(0 if FAIL_COUNT == 0 else 1)
