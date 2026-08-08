"""Unit tests for P0-MPC bound injection, cost profiles, and OSQP adapter."""
from __future__ import annotations

import unittest

try:
    import osqp  # noqa: F401
    import scipy  # noqa: F401

    HAS_OSQP = True
except Exception:
    HAS_OSQP = False

from agent_ai.authority.schemas import DrivableEnvelope, TrajectoryRequest

if HAS_OSQP:
    from agent_ai.authority.osqp_mpc_adapter import (
        OSQPMpcAdapter,
        trajectory_request_to_bounds,
    )
    from agent_ai.benchmark.shadow.kinematic_mpc_shadow import (
        _build_follow_solution,
        _solve_longitudinal_mpc_qp,
        apply_bounds_to_target_speed,
        extract_mpc_bounds,
    )


@unittest.skipUnless(HAS_OSQP, "osqp/scipy not installed")
class ExtractBoundsTests(unittest.TestCase):
    def test_extract_from_nested_soft_contract(self) -> None:
        req = {
            "target_speed_mps": 8.0,
            "soft_contract": {
                "contract": {
                    "max_speed_mps": 5.0,
                    "max_longitudinal_accel_mps2": 1.5,
                    "max_lateral_offset_m": 0.6,
                },
                "cost_profile": "AGENT_BOUNDED_CAUTIOUS",
            },
            "front_free_space_m": 12.0,
            "min_ttc_s": 1.8,
        }
        b = extract_mpc_bounds(req)
        self.assertAlmostEqual(float(b["v_max_mps"]), 5.0)
        self.assertAlmostEqual(float(b["forward_clear_m"]), 12.0)
        self.assertAlmostEqual(float(b["min_ttc_s"]), 1.8)
        self.assertEqual(b["cost_profile"], "AGENT_BOUNDED_CAUTIOUS")

    def test_apply_bounds_caps_speed(self) -> None:
        speed, notes = apply_bounds_to_target_speed(
            10.0,
            {"v_max_mps": 4.0, "soft_speed_scale": 1.0, "cost_profile": "AGENT_BOUNDED"},
        )
        self.assertLessEqual(speed, 4.0 + 1e-6)
        self.assertTrue(any("v_max" in n for n in notes))

    def test_defensive_profile_reduces_speed(self) -> None:
        base, _ = apply_bounds_to_target_speed(8.0, {"cost_profile": "AGENT_BOUNDED"})
        defn, notes = apply_bounds_to_target_speed(8.0, {"cost_profile": "AGENT_BOUNDED_DEFENSIVE"})
        self.assertLess(defn, base)
        self.assertTrue(notes)


@unittest.skipUnless(HAS_OSQP, "osqp/scipy not installed")
class LongitudinalBoundsTests(unittest.TestCase):
    def test_clear_distance_limits_progress(self) -> None:
        free = _solve_longitudinal_mpc_qp(
            current_speed_mps=6.0,
            target_speed_mps=8.0,
            dt_s=0.1,
            horizon_steps=10,
            stop_distance_m=None,
            bounds=None,
            allow_degrade=False,
        )
        tight = _solve_longitudinal_mpc_qp(
            current_speed_mps=6.0,
            target_speed_mps=8.0,
            dt_s=0.1,
            horizon_steps=10,
            stop_distance_m=None,
            bounds={"forward_clear_m": 8.0, "cost_profile": "AGENT_BOUNDED_CAUTIOUS"},
            allow_degrade=False,
        )
        self.assertTrue(free.get("feasible"))
        self.assertTrue(tight.get("feasible"))
        free_p = float(free["distances"][-1])
        tight_p = float(tight["distances"][-1])
        self.assertLessEqual(tight_p, free_p + 1e-3)
        self.assertLessEqual(tight_p, 8.0 + 0.5)

    def test_v_max_caps_solution_speeds(self) -> None:
        qp = _solve_longitudinal_mpc_qp(
            current_speed_mps=3.0,
            target_speed_mps=12.0,
            dt_s=0.1,
            horizon_steps=10,
            bounds={"v_max_mps": 4.0},
            allow_degrade=False,
        )
        self.assertTrue(qp.get("feasible"))
        for v in qp["speeds"]:
            self.assertLessEqual(float(v), 4.5)  # small numeric buffer

    def test_follow_solution_notes_bounds(self) -> None:
        sol = _build_follow_solution(
            request={
                "requested_behavior": "keep_lane",
                "target_speed_mps": 9.0,
                "v_max_mps": 5.0,
                "forward_clear_m": 15.0,
                "cost_profile": "AGENT_BOUNDED_CAUTIOUS",
            },
            current_speed_mps=4.0,
            dt_s=0.1,
        )
        self.assertTrue(sol.get("feasible"))
        self.assertLessEqual(float(sol["target_speed_mps"]), 5.0 + 0.1)
        self.assertTrue(sol.get("notes"))


@unittest.skipUnless(HAS_OSQP, "osqp/scipy not installed")
class OSQPAdapterTests(unittest.TestCase):
    def test_trajectory_request_to_bounds(self) -> None:
        req = TrajectoryRequest(
            tactical_intent="keep_lane",
            v_max_mps=6.0,
            a_long_max_mps2=1.8,
            a_lat_max_mps2=1.2,
            lateral_bound_m=0.7,
            target_v_desired_mps=5.0,
            cost_profile="AGENT_BOUNDED_CAUTIOUS",
            drivable_envelope=DrivableEnvelope("e", 0.7, 0.7, 20.0),
        )
        b = trajectory_request_to_bounds(req)
        self.assertAlmostEqual(b["v_max_mps"], 6.0)
        self.assertAlmostEqual(b["forward_clear_m"], 20.0)
        self.assertEqual(b["cost_profile"], "AGENT_BOUNDED_CAUTIOUS")

    def test_execute_follow_command_in_range(self) -> None:
        mpc = OSQPMpcAdapter()
        req = TrajectoryRequest(
            tactical_intent="keep_lane",
            v_max_mps=8.0,
            a_long_max_mps2=2.0,
            a_lat_max_mps2=1.5,
            lateral_bound_m=0.8,
            target_v_desired_mps=5.0,
            cost_profile="AGENT_BOUNDED",
            drivable_envelope=DrivableEnvelope("e", 0.8, 0.8, 40.0),
        )
        cmd = mpc.execute(req, ego_v_mps=3.0)
        self.assertEqual(cmd.source, "MPC")
        self.assertGreaterEqual(cmd.throttle, 0.0)
        self.assertLessEqual(cmd.throttle, 1.0)
        self.assertGreaterEqual(cmd.brake, 0.0)
        self.assertLessEqual(cmd.brake, 1.0)
        self.assertTrue(mpc.preview_feasible(req, ego_v_mps=3.0))

    def test_stop_intent_applies_brake(self) -> None:
        mpc = OSQPMpcAdapter()
        req = TrajectoryRequest(
            tactical_intent="safe_stop",
            v_max_mps=0.0,
            a_long_max_mps2=3.0,
            target_v_desired_mps=0.0,
            cost_profile="MRM",
            drivable_envelope=DrivableEnvelope("e", 0.5, 0.5, 10.0),
        )
        cmd = mpc.execute(req, ego_v_mps=5.0)
        self.assertGreater(cmd.brake, 0.0)
        self.assertEqual(cmd.throttle, 0.0)


if __name__ == "__main__":
    unittest.main()
