# Stage 9 Incremental Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor stage9 module for type safety, testability, and maintainability while preserving all existing behavior.

**Architecture:** Add pytest test infrastructure, write comprehensive tests for each stage9 component against current behavior, add strict type hints, extract shared constants, then reorganize into agent_ai/authority package with backward-compatible re-exports.

**Tech Stack:** Python 3.10+, pytest, mypy, ruff, dataclasses (existing)

## Global Constraints

- No behavior changes — only structural improvements
- All existing public APIs must remain importable from `stage9` package during transition
- Every commit must pass `pytest tests/stage9/ -v` and `mypy stage9/`
- Use existing dataclass patterns; do not introduce pydantic unless already present
- Hard limit constants must be defined once in a single location and imported everywhere
- Test files mirror source structure: `tests/stage9/test_<module>.py`

---

### Task 1: Set Up Test Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/stage9/__init__.py`
- Create: `tests/stage9/conftest.py`
- Create: `pyproject.toml` (if not exists, append pytest/mypy/ruff config)

**Interfaces:**
- Produces: pytest configuration, shared fixtures for WorldState, ManeuverContract, ActuatorCommand

- [ ] **Step 1: Create test directory structure**

```bash
mkdir -p tests/stage9
touch tests/__init__.py tests/stage9/__init__.py
```

- [ ] **Step 2: Create conftest with shared fixtures**

Create `tests/stage9/conftest.py`:

```python
import pytest
from stage9.schemas import (
    WorldState, DrivableEnvelope, SensorHealth, ODDStatus,
    ManeuverContract, ActuatorCommand, TrajectoryRequest,
)


@pytest.fixture
def default_world() -> WorldState:
    return WorldState(
        frame_id=100,
        timestamp_s=10.0,
        world_age_ms=50,
        sync_ok=True,
        sensor_health=SensorHealth.OK,
        ego_v_mps=8.0,
        ego_a_mps2=0.0,
        ego_lane_id="lane_1",
        ego_lateral_error_m=0.1,
        corridor_clear=True,
        min_ttc_s=5.0,
        new_obstacle_score=0.1,
        weather_visibility_m=200.0,
        lane_change_permission=True,
        drivable_envelope=DrivableEnvelope(
            envelope_uuid="env_1",
            left_bound_m=1.5,
            right_bound_m=1.5,
            forward_clear_m=100.0,
        ),
        odd_status=ODDStatus.IN_ODD,
        time_to_odd_exit_s=None,
        preview_feasible=True,
        human_driver_available=True,
        human_input_detected=False,
    )


@pytest.fixture
def default_contract() -> ManeuverContract:
    return ManeuverContract(
        issued_at_frame=100,
        tactical_intent="keep_lane",
        max_lateral_offset_m=0.8,
        max_speed_mps=8.33,
        max_duration_s=5.0,
        max_jerk_mps3=3.0,
        agent_confidence=0.9,
        freshness_required_ms=100,
        validity_deadline_s=15.0,
    )


@pytest.fixture
def default_actuator_command() -> ActuatorCommand:
    return ActuatorCommand(steer=0.0, throttle=0.3, brake=0.0, source="MPC")
```

- [ ] **Step 3: Add pytest/mypy/ruff config to pyproject.toml**

If `pyproject.toml` exists, append; otherwise create:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[tool.ruff]
target-version = "py310"
line-length = 120
```

- [ ] **Step 4: Verify test infrastructure works**

Run: `pytest tests/stage9/ -v`
Expected: "no tests ran" or empty collection, exit code 0 or 5 (no tests collected)

- [ ] **Step 5: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "chore: set up pytest infrastructure for stage9 refactor"
```

---

### Task 2: Extract Shared Constants

**Files:**
- Create: `stage9/constants.py`
- Modify: `stage9/safety_supervisor.py:49-55`
- Modify: `stage9/maneuver_contract.py:20-26`

**Interfaces:**
- Consumes: nothing
- Produces: `stage9.constants` module with all hard-limit constants used by SafetySupervisor and maneuver_contract

- [ ] **Step 1: Write failing test for constants module**

Create `tests/stage9/test_constants.py`:

```python
from stage9.constants import (
    ABS_MAX_LATERAL_OFFSET_M,
    ABS_MAX_SPEED_MPS,
    ABS_MAX_DURATION_S,
    ABS_MAX_JERK_MPS3,
    ABS_MIN_CONFIDENCE,
    ABS_FRESHNESS_MS,
    REVOKE_WINDOW_S,
    MAX_REVOKES_IN_WINDOW,
)


def test_constants_are_positive() -> None:
    assert ABS_MAX_LATERAL_OFFSET_M > 0
    assert ABS_MAX_SPEED_MPS > 0
    assert ABS_MAX_DURATION_S > 0
    assert ABS_MAX_JERK_MPS3 > 0
    assert ABS_MIN_CONFIDENCE > 0
    assert ABS_FRESHNESS_MS > 0
    assert REVOKE_WINDOW_S > 0
    assert MAX_REVOKES_IN_WINDOW >= 0


def test_constants_match_expected_values() -> None:
    assert ABS_MAX_LATERAL_OFFSET_M == 1.0
    assert ABS_MAX_SPEED_MPS == 13.89
    assert ABS_MAX_DURATION_S == 8.0
    assert ABS_MAX_JERK_MPS3 == 4.0
    assert ABS_MIN_CONFIDENCE == 0.85
    assert ABS_FRESHNESS_MS == 100
    assert REVOKE_WINDOW_S == 60.0
    assert MAX_REVOKES_IN_WINDOW == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/stage9/test_constants.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Create constants module**

Create `stage9/constants.py`:

```python
ABS_MAX_LATERAL_OFFSET_M: float = 1.0
ABS_MAX_SPEED_MPS: float = 13.89
ABS_MAX_DURATION_S: float = 8.0
ABS_MAX_JERK_MPS3: float = 4.0
ABS_MIN_CONFIDENCE: float = 0.85
ABS_FRESHNESS_MS: int = 100
REVOKE_WINDOW_S: float = 60.0
MAX_REVOKES_IN_WINDOW: int = 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/stage9/test_constants.py -v`
Expected: PASS

- [ ] **Step 5: Update safety_supervisor.py to import constants**

In `stage9/safety_supervisor.py`, replace lines 49-55:

```python
    _ABS_MAX_LATERAL_M   = 1.0
    _ABS_MAX_SPEED_MPS   = 13.89
    _ABS_MAX_DURATION_S  = 8.0
    _ABS_MAX_JERK        = 4.0
    _REVOKE_WINDOW_S     = 60.0
    _MAX_REVOKES_IN_WIN  = 2
```

With:

```python
from .constants import (
    ABS_MAX_LATERAL_OFFSET_M as _ABS_MAX_LATERAL_M,
    ABS_MAX_SPEED_MPS as _ABS_MAX_SPEED_MPS,
    ABS_MAX_DURATION_S as _ABS_MAX_DURATION_S,
    ABS_MAX_JERK_MPS3 as _ABS_MAX_JERK,
    REVOKE_WINDOW_S as _REVOKE_WINDOW_S,
    MAX_REVOKES_IN_WINDOW as _MAX_REVOKES_IN_WIN,
)
```

Remove the class-level constant definitions (lines 49-55) since they are now imported at module level. Place the import after the existing imports block (after line 29).

- [ ] **Step 6: Update maneuver_contract.py to import constants**

In `stage9/maneuver_contract.py`, replace lines 20-26:

```python
_ABS_MAX_LATERAL_OFFSET_M      = 1.0
_ABS_MAX_SPEED_MPS             = 13.89
_ABS_MAX_DURATION_S            = 8.0
_ABS_MAX_JERK_MPS3             = 4.0
_ABS_MIN_CONFIDENCE            = 0.85
_ABS_FRESHNESS_MS              = 100
```

With:

```python
from .constants import (
    ABS_MAX_LATERAL_OFFSET_M as _ABS_MAX_LATERAL_OFFSET_M,
    ABS_MAX_SPEED_MPS as _ABS_MAX_SPEED_MPS,
    ABS_MAX_DURATION_S as _ABS_MAX_DURATION_S,
    ABS_MAX_JERK_MPS3 as _ABS_MAX_JERK_MPS3,
    ABS_MIN_CONFIDENCE as _ABS_MIN_CONFIDENCE,
    ABS_FRESHNESS_MS as _ABS_FRESHNESS_MS,
)
```

Place the import after the existing imports block (after line 16).

- [ ] **Step 7: Run all stage9 tests to verify no regression**

Run: `pytest tests/stage9/ -v`
Expected: PASS (only test_constants tests so far)

- [ ] **Step 8: Commit**

```bash
git add stage9/constants.py stage9/safety_supervisor.py stage9/maneuver_contract.py tests/stage9/test_constants.py
git commit -m "refactor: extract shared hard-limit constants to stage9/constants.py"
```

---

### Task 3: Add Type Hints to schemas.py

**Files:**
- Modify: `stage9/schemas.py`

**Interfaces:**
- Consumes: nothing
- Produces: fully typed schemas module; all downstream modules continue to work unchanged

- [ ] **Step 1: Write test verifying schema construction and types**

Create `tests/stage9/test_schemas.py`:

```python
from stage9.schemas import (
    AuthorityState, ActiveAuthority, ODDStatus, SensorHealth,
    VetoSeverity, MRMStrategy, WorldState, ManeuverContract,
    TakeoverRequest, AuthorityContext, TrajectoryRequest,
    TORSignal, VetoSignal, SafetyVerdict, MRCPlan, ActuatorCommand,
    DrivableEnvelope, Pose2D,
)


def test_authority_state_enum_members() -> None:
    assert AuthorityState.BASELINE_CONTROL is not None
    assert AuthorityState.AGENT_ACTIVE_BOUNDED is not None
    assert AuthorityState.SAFE_STOP is not None
    assert len(AuthorityState) == 9


def test_active_authority_values() -> None:
    assert ActiveAuthority.BASELINE.value == "BASELINE"
    assert ActiveAuthority.AGENT.value == "AGENT"
    assert ActiveAuthority.HUMAN.value == "HUMAN"
    assert ActiveAuthority.MRM.value == "MRM"


def test_world_state_construction(default_world: WorldState) -> None:
    assert default_world.frame_id == 100
    assert default_world.sync_ok is True
    assert default_world.sensor_health == SensorHealth.OK
    assert default_world.odd_status == ODDStatus.IN_ODD


def test_maneuver_contract_defaults(default_contract: ManeuverContract) -> None:
    assert default_contract.tactical_intent == "keep_lane"
    assert default_contract.max_lateral_offset_m == 0.8
    assert default_contract.revoke_strategy == "RETURN_TO_BASELINE"
    assert default_contract.contract_id != ""


def test_trajectory_request_blend() -> None:
    baseline = TrajectoryRequest(source="BASELINE", v_max_mps=10.0, target_v_desired_mps=8.0)
    candidate = TrajectoryRequest(source="CONTRACT_RESOLVER", v_max_mps=5.0, target_v_desired_mps=4.0)
    blended = TrajectoryRequest.blend(baseline, candidate, agent_weight=0.5)
    assert blended.source == "HANDOFF"
    assert abs(blended.v_max_mps - 7.5) < 0.01
    assert abs(blended.target_v_desired_mps - 6.0) < 0.01
    assert blended.a_long_max_mps2 == min(baseline.a_long_max_mps2, candidate.a_long_max_mps2)


def test_safety_verdict_reject() -> None:
    v = SafetyVerdict()
    assert v.approved is True
    v.reject("test reason")
    assert v.approved is False
    assert v.primary_reason() == "test reason"


def test_mrc_plan_to_trajectory_request() -> None:
    plan = MRCPlan(strategy=MRMStrategy.COAST_TO_STOP, max_decel_mps2=2.5)
    req = plan.to_trajectory_request()
    assert req.source == "MRM"
    assert req.tactical_intent == "safe_stop"
    assert req.v_max_mps == 0.0


def test_actuator_command_source_default() -> None:
    cmd = ActuatorCommand(steer=0.0, throttle=0.0, brake=0.0)
    assert cmd.source == "MPC"
```

- [ ] **Step 2: Run test to verify it passes against current code**

Run: `pytest tests/stage9/test_schemas.py -v`
Expected: PASS (schemas already have dataclass types; this establishes baseline)

- [ ] **Step 3: Add explicit type annotations to list fields in schemas.py**

In `stage9/schemas.py`, change line 119:

```python
    sub_intent_sequence: list = field(default_factory=list)
```

To:

```python
    sub_intent_sequence: list[str] = field(default_factory=list)
```

Change line 198:

```python
    reference_path: Optional[list] = None
```

To:

```python
    reference_path: Optional[list[tuple[float, float]]] = None
```

Change line 253:

```python
    reasons: list = field(default_factory=list)
```

To:

```python
    reasons: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Run tests and mypy to verify**

Run: `pytest tests/stage9/test_schemas.py -v && mypy stage9/schemas.py`
Expected: PASS, no mypy errors

- [ ] **Step 5: Commit**

```bash
git add stage9/schemas.py tests/stage9/test_schemas.py
git commit -m "refactor: add strict type hints to schemas.py list fields"
```

---

### Task 4: Write Tests for AuthorityStateMachine

**Files:**
- Create: `tests/stage9/test_authority_state_machine.py`

**Interfaces:**
- Consumes: `stage9.AuthorityStateMachine`, `stage9.schemas.*`
- Produces: comprehensive test coverage for state transitions, cooldown, revoke ramp, context generation

- [ ] **Step 1: Write tests for AuthorityStateMachine**

Create `tests/stage9/test_authority_state_machine.py`:

```python
from stage9.authority_state_machine import AuthorityStateMachine
from stage9.schemas import (
    AuthorityState, ActiveAuthority, ManeuverContract, TrajectoryRequest,
)


def test_initial_state_is_baseline_control() -> None:
    asm = AuthorityStateMachine()
    assert asm.current_state == AuthorityState.BASELINE_CONTROL


def test_get_context_baseline(default_contract: ManeuverContract) -> None:
    asm = AuthorityStateMachine()
    ctx = asm.get_context(sim_time_s=0.0)
    assert ctx.current_state == AuthorityState.BASELINE_CONTROL
    assert ctx.active_authority == ActiveAuthority.BASELINE
    assert ctx.agent_weight == 0.0
    assert ctx.baseline_weight == 1.0


def test_transition_to_requesting(default_contract: ManeuverContract) -> None:
    asm = AuthorityStateMachine()
    asm.transition_to(
        AuthorityState.AGENT_REQUESTING_AUTHORITY,
        contract=default_contract,
        reason="test",
    )
    assert asm.current_state == AuthorityState.AGENT_REQUESTING_AUTHORITY
    ctx = asm.get_context()
    assert ctx.active_contract is not None
    assert ctx.active_contract.contract_id == default_contract.contract_id


def test_transition_to_agent_active_sets_agent_weight(default_contract: ManeuverContract) -> None:
    asm = AuthorityStateMachine()
    asm.transition_to(AuthorityState.AGENT_ACTIVE_BOUNDED, contract=default_contract, sim_time_s=10.0)
    ctx = asm.get_context(sim_time_s=10.0)
    assert ctx.active_authority == ActiveAuthority.AGENT
    assert ctx.agent_weight == 1.0
    assert ctx.baseline_weight == 0.0


def test_supervised_execution_has_half_weight(default_contract: ManeuverContract) -> None:
    asm = AuthorityStateMachine()
    asm.transition_to(AuthorityState.SUPERVISED_EXECUTION, contract=default_contract)
    ctx = asm.get_context()
    assert ctx.agent_weight == 0.5


def test_revoke_ramp_alpha_decreases() -> None:
    asm = AuthorityStateMachine()
    asm.start_revoke_window()
    assert asm.current_revoke_alpha() == 1.0
    asm.increment_revoke_frame()
    assert asm.current_revoke_alpha() == 0.8
    asm.increment_revoke_frame()
    assert asm.current_revoke_alpha() == 0.6


def test_revoke_complete_after_total_frames() -> None:
    asm = AuthorityStateMachine()
    asm.start_revoke_window()
    assert not asm.revoke_complete()
    for _ in range(5):
        asm.increment_revoke_frame()
    assert asm.revoke_complete()


def test_cooldown_active_after_revoke(default_contract: ManeuverContract) -> None:
    asm = AuthorityStateMachine()
    asm.transition_to(AuthorityState.AGENT_ACTIVE_BOUNDED, contract=default_contract, sim_time_s=0.0)
    asm.transition_to(AuthorityState.AUTHORITY_REVOKE_PENDING, reason="test_revoke")
    assert asm.cooldown_active()


def test_same_state_transition_is_noop(default_contract: ManeuverContract) -> None:
    asm = AuthorityStateMachine()
    asm.transition_to(AuthorityState.AGENT_ACTIVE_BOUNDED, contract=default_contract, sim_time_s=0.0)
    log_before = len(asm.get_transition_log())
    asm.transition_to(AuthorityState.AGENT_ACTIVE_BOUNDED, sim_time_s=1.0)
    log_after = len(asm.get_transition_log())
    assert log_after == log_before


def test_cache_agent_request() -> None:
    asm = AuthorityStateMachine()
    req = TrajectoryRequest(source="CONTRACT_RESOLVER", tactical_intent="keep_lane")
    asm.cache_agent_request(req)
    assert asm.last_agent_request() is req


def test_transition_log_records_entries() -> None:
    asm = AuthorityStateMachine()
    asm.transition_to(AuthorityState.TOR_ACTIVE, reason="odd_margin")
    log = asm.get_transition_log()
    assert len(log) == 1
    assert log[0]["from"] == "BASELINE_CONTROL"
    assert log[0]["to"] == "TOR_ACTIVE"
    assert log[0]["reason"] == "odd_margin"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/stage9/test_authority_state_machine.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/stage9/test_authority_state_machine.py
git commit -m "test: add comprehensive tests for AuthorityStateMachine"
```

---

### Task 5: Write Tests for SafetySupervisor

**Files:**
- Create: `tests/stage9/test_safety_supervisor.py`

**Interfaces:**
- Consumes: `stage9.SafetySupervisor`, `stage9.schemas.*`, fixtures from conftest
- Produces: test coverage for verify_contract (4 layers), watch_frame, helper gates

- [ ] **Step 1: Write tests for SafetySupervisor**

Create `tests/stage9/test_safety_supervisor.py`:

```python
import copy
from stage9.safety_supervisor import SafetySupervisor
from stage9.schemas import (
    ManeuverContract, WorldState, ODDStatus, SensorHealth, VetoSeverity,
)


def test_verify_contract_approves_valid(default_contract: ManeuverContract, default_world: WorldState) -> None:
    sup = SafetySupervisor()
    verdict = sup.verify_contract(default_contract, default_world)
    assert verdict.approved is True


def test_verify_rejects_lateral_offset_exceeded(default_contract: ManeuverContract, default_world: WorldState) -> None:
    sup = SafetySupervisor()
    bad = copy.replace(default_contract, max_lateral_offset_m=1.5)
    verdict = sup.verify_contract(bad, default_world)
    assert not verdict.approved
    assert any("S-001" in r for r in verdict.reasons)


def test_verify_rejects_speed_exceeded(default_contract: ManeuverContract, default_world: WorldState) -> None:
    sup = SafetySupervisor()
    bad = copy.replace(default_contract, max_speed_mps=20.0)
    verdict = sup.verify_contract(bad, default_world)
    assert not verdict.approved
    assert any("S-002" in r for r in verdict.reasons)


def test_verify_rejects_stale_world(default_contract: ManeuverContract, default_world: WorldState) -> None:
    sup = SafetySupervisor()
    stale = copy.replace(default_world, world_age_ms=200)
    verdict = sup.verify_contract(default_contract, stale)
    assert not verdict.approved
    assert any("S-005" in r for r in verdict.reasons)


def test_verify_rejects_odd_not_in_odd(default_contract: ManeuverContract, default_world: WorldState) -> None:
    sup = SafetySupervisor()
    bad_world = copy.replace(default_world, odd_status=ODDStatus.ODD_EXCEEDED)
    verdict = sup.verify_contract(default_contract, bad_world)
    assert not verdict.approved
    assert any("S-007" in r for r in verdict.reasons)


def test_verify_rejects_sensor_fault(default_contract: ManeuverContract, default_world: WorldState) -> None:
    sup = SafetySupervisor()
    bad_world = copy.replace(default_world, sensor_health=SensorHealth.FAULT)
    verdict = sup.verify_contract(default_contract, bad_world)
    assert not verdict.approved
    assert any("S-008" in r for r in verdict.reasons)


def test_verify_rejects_low_confidence(default_contract: ManeuverContract, default_world: WorldState) -> None:
    sup = SafetySupervisor()
    bad = copy.replace(default_contract, agent_confidence=0.5)
    verdict = sup.verify_contract(bad, default_world)
    assert not verdict.approved
    assert any("S-012" in r for r in verdict.reasons)


def test_watch_frame_returns_none_when_safe(default_contract: ManeuverContract, default_world: WorldState) -> None:
    sup = SafetySupervisor()
    veto = sup.watch_frame(default_world, default_contract, sim_time_s=10.0)
    assert veto is None


def test_watch_frame_vetos_on_ttc_collapse(default_contract: ManeuverContract, default_world: WorldState) -> None:
    sup = SafetySupervisor()
    dangerous = copy.replace(default_world, min_ttc_s=0.5)
    veto = sup.watch_frame(dangerous, default_contract, sim_time_s=10.0)
    assert veto is not None
    assert "S-V003" in veto.reason
    assert veto.severity == VetoSeverity.HARD


def test_forward_takeover_allowed_when_healthy(default_world: WorldState) -> None:
    sup = SafetySupervisor()
    assert sup.forward_takeover_allowed(default_world) is True


def test_forward_takeover_blocked_by_sensor_fault(default_world: WorldState) -> None:
    sup = SafetySupervisor()
    bad = copy.replace(default_world, sensor_health=SensorHealth.FAULT)
    assert sup.forward_takeover_allowed(bad) is False


def test_critical_fault_on_sensor_fault(default_world: WorldState) -> None:
    sup = SafetySupervisor()
    bad = copy.replace(default_world, sensor_health=SensorHealth.FAULT)
    assert sup.critical_fault(bad) is True


def test_critical_fault_on_odd_exceeded(default_world: WorldState) -> None:
    sup = SafetySupervisor()
    bad = copy.replace(default_world, odd_status=ODDStatus.ODD_EXCEEDED)
    assert sup.critical_fault(bad) is True


def test_record_revoke_and_count() -> None:
    sup = SafetySupervisor()
    sup.record_revoke()
    sup.record_revoke()
    assert sup._count_recent_revokes() == 2
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/stage9/test_safety_supervisor.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/stage9/test_safety_supervisor.py
git commit -m "test: add comprehensive tests for SafetySupervisor"
```

---

### Task 6: Write Tests for ContractResolver and HandoffPlanner

**Files:**
- Create: `tests/stage9/test_contract_resolver.py`
- Create: `tests/stage9/test_handoff_planner.py`

**Interfaces:**
- Consumes: `stage9.ContractResolver`, `stage9.HandoffPlanner`, `stage9.schemas.*`
- Produces: test coverage for intent-to-trajectory mapping and blend logic

- [ ] **Step 1: Write tests for ContractResolver**

Create `tests/stage9/test_contract_resolver.py`:

```python
from stage9.contract_resolver import ContractResolver
from stage9.schemas import ManeuverContract


def test_resolve_keep_lane(default_contract: ManeuverContract) -> None:
    resolver = ContractResolver()
    req = resolver.resolve("keep_lane", default_contract)
    assert req.source == "CONTRACT_RESOLVER"
    assert req.tactical_intent == "keep_lane"
    assert req.v_max_mps == default_contract.max_speed_mps
    assert abs(req.target_v_desired_mps - default_contract.max_speed_mps) < 0.01


def test_resolve_slow_bypass_reduces_speed(default_contract: ManeuverContract) -> None:
    resolver = ContractResolver()
    req = resolver.resolve("slow_bypass_right", default_contract)
    expected_v = default_contract.max_speed_mps * 0.70
    assert abs(req.target_v_desired_mps - expected_v) < 0.01


def test_resolve_safe_stop_zero_speed(default_contract: ManeuverContract) -> None:
    resolver = ContractResolver()
    req = resolver.resolve("safe_stop", default_contract)
    assert req.target_v_desired_mps == 0.0


def test_resolve_unknown_intent_uses_full_speed(default_contract: ManeuverContract) -> None:
    resolver = ContractResolver()
    req = resolver.resolve("unknown_intent", default_contract)
    assert req.target_v_desired_mps == default_contract.max_speed_mps


def test_resolve_horizon_capped_at_3s(default_contract: ManeuverContract) -> None:
    import copy
    long_contract = copy.replace(default_contract, max_duration_s=10.0)
    resolver = ContractResolver()
    req = resolver.resolve("keep_lane", long_contract)
    assert req.horizon_s == 3.0
```

- [ ] **Step 2: Run ContractResolver tests**

Run: `pytest tests/stage9/test_contract_resolver.py -v`
Expected: ALL PASS

- [ ] **Step 3: Write tests for HandoffPlanner**

Create `tests/stage9/test_handoff_planner.py`:

```python
from stage9.handoff_planner import HandoffPlanner
from stage9.schemas import TrajectoryRequest


def test_preview_full_baseline() -> None:
    hp = HandoffPlanner()
    baseline = TrajectoryRequest(source="BASELINE", v_max_mps=10.0)
    candidate = TrajectoryRequest(source="CONTRACT_RESOLVER", v_max_mps=5.0)
    result = hp.preview(baseline, candidate, alpha=0.0)
    assert abs(result.v_max_mps - 10.0) < 0.01


def test_preview_full_agent() -> None:
    hp = HandoffPlanner()
    baseline = TrajectoryRequest(source="BASELINE", v_max_mps=10.0)
    candidate = TrajectoryRequest(source="CONTRACT_RESOLVER", v_max_mps=5.0)
    result = hp.preview(baseline, candidate, alpha=1.0)
    assert abs(result.v_max_mps - 5.0) < 0.01


def test_preview_half_blend() -> None:
    hp = HandoffPlanner()
    baseline = TrajectoryRequest(source="BASELINE", v_max_mps=10.0)
    candidate = TrajectoryRequest(source="CONTRACT_RESOLVER", v_max_mps=5.0)
    result = hp.preview(baseline, candidate, alpha=0.5)
    assert abs(result.v_max_mps - 7.5) < 0.01


def test_ramp_to_baseline_with_no_last_agent() -> None:
    hp = HandoffPlanner()
    baseline = TrajectoryRequest(source="BASELINE", v_max_mps=10.0)
    result = hp.ramp_to_baseline(baseline, None, revoke_frame=0)
    assert result is baseline


def test_ramp_to_baseline_frame_zero_is_full_agent() -> None:
    hp = HandoffPlanner()
    baseline = TrajectoryRequest(source="BASELINE", v_max_mps=10.0)
    agent = TrajectoryRequest(source="CONTRACT_RESOLVER", v_max_mps=5.0)
    result = hp.ramp_to_baseline(baseline, agent, revoke_frame=0, total_revoke_frames=5)
    assert abs(result.v_max_mps - 5.0) < 0.01


def test_ramp_to_baseline_final_frame_is_full_baseline() -> None:
    hp = HandoffPlanner()
    baseline = TrajectoryRequest(source="BASELINE", v_max_mps=10.0)
    agent = TrajectoryRequest(source="CONTRACT_RESOLVER", v_max_mps=5.0)
    result = hp.ramp_to_baseline(baseline, agent, revoke_frame=5, total_revoke_frames=5)
    assert abs(result.v_max_mps - 10.0) < 0.01
```

- [ ] **Step 4: Run HandoffPlanner tests**

Run: `pytest tests/stage9/test_handoff_planner.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit both test files**

```bash
git add tests/stage9/test_contract_resolver.py tests/stage9/test_handoff_planner.py
git commit -m "test: add tests for ContractResolver and HandoffPlanner"
```

---

### Task 7: Write Tests for BaselineDetector, TORManager, MRMExecutor, HumanOverrideMonitor

**Files:**
- Create: `tests/stage9/test_baseline_detector.py`
- Create: `tests/stage9/test_tor_manager.py`
- Create: `tests/stage9/test_mrm_executor.py`
- Create: `tests/stage9/test_human_override_monitor.py`

**Interfaces:**
- Consumes: respective stage9 modules, fixtures from conftest
- Produces: test coverage for all remaining stage9 components

- [ ] **Step 1: Write BaselineDetector tests**

Create `tests/stage9/test_baseline_detector.py`:

```python
import copy
from stage9.baseline_detector import BaselineDetector
from stage9.schemas import WorldState


def test_not_stuck_initially(default_world: WorldState) -> None:
    bd = BaselineDetector()
    bd.update(default_world)
    assert bd.is_stuck(default_world) is False


def test_velocity_stall_detected(default_world: WorldState) -> None:
    bd = BaselineDetector(window_s=5.0)
    stalled = copy.replace(default_world, ego_v_mps=0.5, corridor_clear=True)
    for i in range(40):
        frame = copy.replace(stalled, timestamp_s=stalled.timestamp_s + i * 0.1)
        bd.update(frame)
    last_frame = copy.replace(stalled, timestamp_s=stalled.timestamp_s + 3.9)
    assert bd.is_stuck(last_frame) is True


def test_stall_not_detected_when_corridor_blocked(default_world: WorldState) -> None:
    bd = BaselineDetector(window_s=5.0)
    stalled = copy.replace(default_world, ego_v_mps=0.5, corridor_clear=False)
    for i in range(40):
        frame = copy.replace(stalled, timestamp_s=stalled.timestamp_s + i * 0.1)
        bd.update(frame)
    last_frame = copy.replace(stalled, timestamp_s=stalled.timestamp_s + 3.9)
    assert bd.is_stuck(last_frame) is False


def test_planner_degeneracy_detected(default_world: WorldState) -> None:
    bd = BaselineDetector()
    bd.update(default_world, mpc_cost_normalized=10.0, mpc_converged=True)
    assert bd.is_stuck(default_world) is True


def test_active_signals_empty_when_healthy(default_world: WorldState) -> None:
    bd = BaselineDetector()
    bd.update(default_world)
    assert bd.active_signals(default_world) == []
```

- [ ] **Step 2: Write TORManager tests**

Create `tests/stage9/test_tor_manager.py`:

```python
from stage9.tor_manager import TORManager
from stage9.schemas import WorldState


def test_tor_starts_at_level_1(default_world: WorldState) -> None:
    tor = TORManager()
    tor.start(default_world, reason="test")
    assert tor.is_active() is True
    assert tor.current_level() == 1


def test_tor_not_timed_out_immediately(default_world: WorldState) -> None:
    tor = TORManager()
    tor.start(default_world)
    assert tor.timed_out() is False


def test_tor_reset_clears_state(default_world: WorldState) -> None:
    tor = TORManager()
    tor.start(default_world)
    tor.reset()
    assert tor.is_active() is False
    assert tor.elapsed_s() == 0.0


def test_time_budget_critical(default_world: WorldState) -> None:
    import copy
    tor = TORManager()
    tight = copy.replace(default_world, time_to_odd_exit_s=2.0)
    assert tor.time_budget_critical(tight) is True
    safe = copy.replace(default_world, time_to_odd_exit_s=5.0)
    assert tor.time_budget_critical(safe) is False


def test_tick_returns_signal(default_world: WorldState) -> None:
    tor = TORManager()
    tor.start(default_world)
    signal = tor.tick(default_world)
    assert signal.level == 1
    assert signal.timeout_s == TORManager.TOR_TIMEOUT_S
```

- [ ] **Step 3: Write MRMExecutor tests**

Create `tests/stage9/test_mrm_executor.py`:

```python
import copy
from stage9.minimal_risk_maneuver import MRMExecutor
from stage9.schemas import MRMStrategy


def test_compute_selects_coast_to_stop_by_default(default_world) -> None:
    mrm = MRMExecutor()
    plan = mrm.compute(default_world)
    assert plan.strategy == MRMStrategy.COAST_TO_STOP


def test_compute_selects_emergency_stop_on_low_ttc(default_world) -> None:
    mrm = MRMExecutor()
    dangerous = copy.replace(default_world, min_ttc_s=0.5)
    plan = mrm.compute(dangerous)
    assert plan.strategy == MRMStrategy.EMERGENCY_STOP


def test_compute_selects_pull_over_when_shoulder_available(default_world) -> None:
    mrm = MRMExecutor()
    shoulder = copy.replace(default_world, lane_change_permission=True)
    from stage9.schemas import DrivableEnvelope
    wide_env = DrivableEnvelope(envelope_uuid="wide", left_bound_m=1.5, right_bound_m=2.0, forward_clear_m=100.0)
    shoulder = copy.replace(shoulder, drivable_envelope=wide_env)
    plan = mrm.compute(shoulder)
    assert plan.strategy == MRMStrategy.PULL_OVER_RIGHT


def test_full_stop_reached_at_low_speed(default_world) -> None:
    mrm = MRMExecutor()
    stopped = copy.replace(default_world, ego_v_mps=0.1)
    assert mrm.is_full_stop_reached(stopped) is True


def test_standstill_request_has_zero_speed(default_world) -> None:
    mrm = MRMExecutor()
    req = mrm.get_standstill_request(default_world)
    assert req.v_max_mps == 0.0
    assert req.source == "MRM"


def test_reset_clears_state(default_world) -> None:
    mrm = MRMExecutor()
    mrm.compute(default_world)
    mrm.reset()
    assert mrm.total_mrm_frames == 0
```

- [ ] **Step 4: Write HumanOverrideMonitor tests**

Create `tests/stage9/test_human_override_monitor.py`:

```python
from stage9.human_override_monitor import HumanOverrideMonitor


def test_no_override_initially(default_world) -> None:
    mon = HumanOverrideMonitor()
    assert mon.detect(default_world) is False


def test_inject_override_detected(default_world) -> None:
    mon = HumanOverrideMonitor()
    mon.inject_override(steer=0.3)
    assert mon.detect(default_world) is True


def test_current_command_returns_injected(default_world) -> None:
    mon = HumanOverrideMonitor()
    mon.inject_override(steer=0.5, throttle=0.1, brake=0.0)
    cmd = mon.current_command()
    assert cmd.steer == 0.5
    assert cmd.throttle == 0.1
    assert cmd.source == "HUMAN"


def test_clear_override_removes_injection(default_world) -> None:
    mon = HumanOverrideMonitor(release_hysteresis_s=0.0)
    mon.inject_override()
    mon.clear_override()
    import time
    time.sleep(0.01)
    assert mon.released() is True


def test_released_true_when_no_override() -> None:
    mon = HumanOverrideMonitor()
    assert mon.released() is True
```

- [ ] **Step 5: Run all new tests**

Run: `pytest tests/stage9/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add tests/stage9/test_baseline_detector.py tests/stage9/test_tor_manager.py tests/stage9/test_mrm_executor.py tests/stage9/test_human_override_monitor.py
git commit -m "test: add tests for BaselineDetector, TORManager, MRMExecutor, HumanOverrideMonitor"
```

---

### Task 8: Add Type Hints to All Stage9 Modules

**Files:**
- Modify: `stage9/authority_state_machine.py`
- Modify: `stage9/safety_supervisor.py`
- Modify: `stage9/contract_resolver.py`
- Modify: `stage9/handoff_planner.py`
- Modify: `stage9/baseline_detector.py`
- Modify: `stage9/tor_manager.py`
- Modify: `stage9/minimal_risk_maneuver.py`
- Modify: `stage9/human_override_monitor.py`
- Modify: `stage9/authority_logger.py`
- Modify: `stage9/stage9_evaluator.py`
- Modify: `stage9/maneuver_contract.py`

**Interfaces:**
- Consumes: all stage9 modules
- Produces: all modules pass `mypy --strict stage9/`

- [ ] **Step 1: Run mypy to establish baseline**

Run: `mypy stage9/ --ignore-missing-imports`
Expected: note current error count

- [ ] **Step 2: Fix type annotations across all modules**

For each file, ensure:
- All function parameters and return types are annotated
- All class attributes are annotated
- Replace bare `list` with `list[T]`, bare `dict` with `dict[K, V]`
- Replace `X | None` with `Optional[X]` where needed for consistency
- Add `-> None` to void methods

Key fixes needed per file:
- `authority_state_machine.py`: annotate `_ASMMemory` fields, all method returns
- `safety_supervisor.py`: annotate `_SupervisorMemory`, return types on all methods
- `baseline_detector.py`: annotate `_ObsFrame`, return types
- `tor_manager.py`: annotate `_TORState`, return types
- `minimal_risk_maneuver.py`: annotate `_MRMState`, return types
- `human_override_monitor.py`: annotate `_HumanMonitorState`, return types
- `authority_logger.py`: annotate `_write` parameter as `dict[str, Any]`
- `stage9_evaluator.py`: annotate `_load_events` return, `_analyse` parameter
- `maneuver_contract.py`: annotate `validate_contract` return, `build_contract` params

- [ ] **Step 3: Run mypy to verify improvement**

Run: `mypy stage9/ --ignore-missing-imports`
Expected: fewer errors than baseline

- [ ] **Step 4: Run all tests to verify no regression**

Run: `pytest tests/stage9/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add stage9/
git commit -m "refactor: add strict type hints to all stage9 modules"
```

---

### Task 9: Create agent_ai/authority Package with Re-exports

**Files:**
- Create: `agent_ai/__init__.py`
- Create: `agent_ai/authority/__init__.py`
- Create: `agent_ai/authority/schemas.py` (copy of stage9/schemas.py)
- Create: `agent_ai/authority/constants.py` (copy of stage9/constants.py)
- Create: `agent_ai/authority/state_machine.py` (copy of stage9/authority_state_machine.py)
- Create: `agent_ai/authority/safety_supervisor.py` (copy)
- Create: `agent_ai/authority/arbiter.py` (copy of stage9/authority_arbiter.py)
- Create: `agent_ai/authority/baseline_detector.py` (copy)
- Create: `agent_ai/authority/contract_resolver.py` (copy)
- Create: `agent_ai/authority/handoff_planner.py` (copy)
- Create: `agent_ai/authority/tor_manager.py` (copy)
- Create: `agent_ai/authority/minimal_risk_maneuver.py` (copy)
- Create: `agent_ai/authority/human_override_monitor.py` (copy)
- Create: `agent_ai/authority/logger.py` (copy of stage9/authority_logger.py)
- Create: `agent_ai/authority/evaluator.py` (copy of stage9/stage9_evaluator.py)
- Create: `agent_ai/authority/maneuver_contract.py` (copy)
- Modify: `stage9/__init__.py` (add re-exports from agent_ai.authority)

**Interfaces:**
- Consumes: all stage9 modules (as source for copy)
- Produces: `agent_ai.authority` package with clean naming; `stage9` continues to work via re-exports

- [ ] **Step 1: Create package structure**

```bash
mkdir -p agent_ai/authority
touch agent_ai/__init__.py
```

- [ ] **Step 2: Copy and adapt all modules**

Copy each stage9 module into agent_ai/authority/ with cleaned names. Update internal imports from `.schemas` to `.schemas` (same relative structure). Example for `agent_ai/authority/__init__.py`:

```python
from .schemas import (
    AuthorityState, ActiveAuthority, ODDStatus, SensorHealth,
    VetoSeverity, MRMStrategy, WorldState, ManeuverContract,
    TakeoverRequest, AuthorityContext, TrajectoryRequest,
    TORSignal, VetoSignal, SafetyVerdict, MRCPlan, ActuatorCommand,
    DrivableEnvelope, Pose2D,
)
from .state_machine import AuthorityStateMachine
from .baseline_detector import BaselineDetector
from .maneuver_contract import build_contract, validate_contract
from .safety_supervisor import SafetySupervisor
from .contract_resolver import ContractResolver
from .handoff_planner import HandoffPlanner
from .logger import AuthorityLogger
from .arbiter import AuthorityArbiter
from .tor_manager import TORManager
from .minimal_risk_maneuver import MRMExecutor
from .human_override_monitor import HumanOverrideMonitor
from .evaluator import Stage9Evaluator, EvaluationReport

__all__ = [
    "AuthorityState", "ActiveAuthority", "ODDStatus", "SensorHealth",
    "VetoSeverity", "MRMStrategy", "WorldState", "ManeuverContract",
    "TakeoverRequest", "AuthorityContext", "TrajectoryRequest",
    "TORSignal", "VetoSignal", "SafetyVerdict", "MRCPlan",
    "ActuatorCommand", "DrivableEnvelope", "Pose2D",
    "AuthorityStateMachine", "BaselineDetector",
    "build_contract", "validate_contract",
    "SafetySupervisor", "ContractResolver", "HandoffPlanner",
    "AuthorityLogger", "AuthorityArbiter",
    "TORManager", "MRMExecutor", "HumanOverrideMonitor",
    "Stage9Evaluator", "EvaluationReport",
]
```

- [ ] **Step 3: Add backward-compatible re-exports to stage9/__init__.py**

At the top of `stage9/__init__.py`, add:

```python
import warnings
warnings.warn(
    "Importing from 'stage9' is deprecated. Use 'agent_ai.authority' instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

Keep all existing imports intact so nothing breaks.

- [ ] **Step 4: Run all tests against both import paths**

Run: `pytest tests/stage9/ -v`
Expected: ALL PASS

Add a quick smoke test for new import path in `tests/stage9/test_new_package.py`:

```python
def test_import_from_new_package() -> None:
    from agent_ai.authority import (
        AuthorityStateMachine, SafetySupervisor, AuthorityArbiter,
        WorldState, ManeuverContract, AuthorityState,
    )
    assert AuthorityStateMachine is not None
    assert SafetySupervisor is not None
    assert AuthorityState.BASELINE_CONTROL is not None
```

Run: `pytest tests/stage9/test_new_package.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent_ai/ stage9/__init__.py tests/stage9/test_new_package.py
git commit -m "refactor: create agent_ai.authority package with backward-compatible stage9 re-exports"
```

---

### Task 10: Final Verification and Cleanup

**Files:**
- Modify: `stage9/__init__.py` (verify deprecation warning)
- Verify: all tests, mypy, lint

**Interfaces:**
- Consumes: entire stage9 and agent_ai.authority packages
- Produces: green CI-equivalent check

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/stage9/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: Run mypy on both packages**

Run: `mypy stage9/ agent_ai/ --ignore-missing-imports`
Expected: minimal or zero errors

- [ ] **Step 3: Run ruff lint**

Run: `ruff check stage9/ agent_ai/`
Expected: clean or only pre-existing warnings

- [ ] **Step 4: Verify backward compatibility**

Run: `python -c "from stage9 import AuthorityArbiter, WorldState, AuthorityState; print('stage9 imports OK')"`
Expected: prints "stage9 imports OK" with DeprecationWarning

Run: `python -c "from agent_ai.authority import AuthorityArbiter, WorldState, AuthorityState; print('new imports OK')"`
Expected: prints "new imports OK" with no warning

- [ ] **Step 5: Commit final cleanup if any changes**

```bash
git add -A
git commit -m "refactor: stage9 incremental refactor complete"
```

</parameter>