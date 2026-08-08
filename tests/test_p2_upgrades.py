"""Unit tests for P2: interaction prediction, cost memory, soft contracts."""
from __future__ import annotations

import math
import unittest

from agent_ai.authority.contract_resolver import ContractResolver
from agent_ai.authority.soft_contract import (
    build_soft_contract_from_behavior,
    derive_soft_bounds,
    evaluate_soft_vetoes,
    soft_cost_profile,
)
from agent_ai.world_state.cost_memory import CostMemory, classify_scene
from agent_ai.world_state.interaction_predictor import (
    apply_interaction_prediction,
    find_pair_conflicts,
    interaction_risk_boost,
)
from agent_ai.world_state.motion_predictor import annotate_tracks_with_prediction
from agent_ai.world_state.schema import TrackedObject
from agent_ai.world_state.soft_constraint_arbiter import SoftConstraintArbiter


def _track(
    track_id: int,
    x: float,
    y: float,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    class_group: str = "vehicle",
) -> TrackedObject:
    dist = math.hypot(x, y)
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name="car",
        class_group=class_group,
        latest_detection_id=f"d{track_id}",
        age_frames=5,
        hits=5,
        missed_frames=0,
        is_occluded_est=False,
        score=0.9,
        mean_score=0.9,
        position_ego=[x, y, 0.0],
        velocity_ego=[vx, vy],
        speed_mps=math.hypot(vx, vy),
        bbox=[x, y, 0.0, 4.0, 2.0, 1.5, 0.0, vx, vy],
        size_xyz=[4.0, 2.0, 1.5],
        yaw_rad=0.0,
        distance_m=dist,
        bearing_deg=0.0,
        ttc_seconds=None,
        relative_sector="front",
        source_confidence=0.9,
    )


class InteractionPredictorTests(unittest.TestCase):
    def test_pair_conflict_when_closing(self) -> None:
        a = _track(1, 12.0, 0.5, vx=-3.0)
        b = _track(2, 14.0, -0.5, vx=-2.5)
        annotate_tracks_with_prediction([a, b])
        conflicts = find_pair_conflicts([a, b], sep_alert_m=15.0)
        self.assertGreaterEqual(len(conflicts), 1)
        self.assertGreater(conflicts[0].severity, 0.0)

    def test_interaction_reweights_leader_brake(self) -> None:
        leader = _track(1, 10.0, 0.0, vx=-4.0)
        other = _track(2, 25.0, 6.0, vx=0.0)
        annotate_tracks_with_prediction([leader, other])
        # Capture brake prob before
        def brake_p(t: TrackedObject) -> float:
            for m in t.predicted_modes:
                if m.get("mode_id") == "brake":
                    return float(m["probability"])
            return 0.0

        before = brake_p(leader)
        summary = apply_interaction_prediction(
            [leader, other],
            ego_speed_mps=8.0,
            ensure_independent=False,
        )
        after = brake_p(leader)
        self.assertIsNotNone(summary.ego_leader_track_id)
        self.assertEqual(summary.ego_leader_track_id, 1)
        self.assertGreaterEqual(after, before - 1e-6)
        self.assertIn(1, summary.reweighted_track_ids)
        self.assertGreater(interaction_risk_boost(summary), 0.0)

    def test_no_tracks_ok(self) -> None:
        summary = apply_interaction_prediction([], ego_speed_mps=0.0)
        self.assertEqual(summary.max_interaction_severity, 0.0)


class CostMemoryTests(unittest.TestCase):
    def test_classify_scene(self) -> None:
        self.assertEqual(
            classify_scene(object_count=2, nearest_front_m=40.0, highest_risk="low"),
            "open",
        )
        self.assertEqual(
            classify_scene(object_count=3, nearest_front_m=10.0, highest_risk="low"),
            "urban",
        )
        self.assertEqual(
            classify_scene(object_count=1, nearest_front_m=30.0, highest_risk="low", junction_near=True),
            "junction",
        )
        self.assertEqual(
            classify_scene(object_count=10, nearest_front_m=20.0, highest_risk="medium"),
            "dense",
        )

    def test_near_miss_increases_safety_weight(self) -> None:
        mem = CostMemory(lr=0.2, decay=1.0)
        base_safety = mem.adapted["safety"]
        for _ in range(5):
            mem.observe(min_ttc_s=1.2, highest_risk="high", maneuver="slow_down")
        self.assertGreater(mem.adapted["safety"], base_safety)
        self.assertGreater(mem.near_miss_count, 0)

    def test_thrash_increases_hysteresis(self) -> None:
        mem = CostMemory(lr=0.25, decay=1.0)
        base_h = mem.adapted["hysteresis"]
        mem.observe(maneuver="follow", min_ttc_s=5.0)
        mem.observe(maneuver="lane_change_left", min_ttc_s=5.0)
        mem.observe(maneuver="follow", min_ttc_s=5.0)
        self.assertGreater(mem.adapted["hysteresis"], base_h)

    def test_arbiter_for_scene_returns_weights(self) -> None:
        mem = CostMemory()
        mem.observe(object_count=12, nearest_front_m=8.0, highest_risk="medium", interaction_severity=0.6)
        arb = mem.arbiter_for_scene()
        self.assertIsInstance(arb, SoftConstraintArbiter)
        self.assertGreater(arb.weights["safety"], 0.0)
        snap = mem.snapshot()
        self.assertIn("effective", snap)


class SoftContractTests(unittest.TestCase):
    def test_derive_bounds_stop_vs_lc(self) -> None:
        stop = derive_soft_bounds(
            maneuver="emergency_stop",
            ego_speed_mps=6.0,
            target_speed_mps=0.0,
            min_ttc_s=1.2,
            highest_risk="critical",
        )
        self.assertEqual(stop["tactical_intent"], "safe_stop")
        self.assertLessEqual(float(stop["max_speed_mps"]), 1.0)

        lc = derive_soft_bounds(
            maneuver="lane_change_left",
            ego_speed_mps=7.0,
            target_speed_mps=6.0,
            min_ttc_s=4.0,
            highest_risk="low",
            confidence=0.8,
        )
        self.assertIn("lane_change", str(lc["tactical_intent"]))
        self.assertGreater(float(lc["max_lateral_offset_m"]), float(stop["max_lateral_offset_m"]))

    def test_soft_vetoes_on_short_ttc(self) -> None:
        vetoes = evaluate_soft_vetoes(
            min_ttc_s=1.0,
            highest_risk="critical",
            interaction_severity=0.8,
            ego_speed_mps=10.0,
            max_speed_mps=5.0,
        )
        codes = {v.code for v in vetoes}
        self.assertIn("SOFT-TTC", codes)
        self.assertIn("SOFT-RISK-CRIT", codes)
        self.assertIn("SOFT-INTERACT", codes)

    def test_build_soft_contract_bundle(self) -> None:
        bundle = build_soft_contract_from_behavior(
            frame_id=3,
            sim_time_s=1.5,
            maneuver="follow",
            ego_speed_mps=6.0,
            target_speed_mps=4.0,
            min_ttc_s=3.0,
            highest_risk="medium",
            interaction_severity=0.3,
            confidence=0.75,
            front_gap_m=12.0,
            reasoning_tags=["idm_speed"],
        )
        self.assertEqual(bundle.contract.issued_at_frame, 3)
        self.assertLessEqual(bundle.contract.max_speed_mps, 13.89)
        self.assertLessEqual(bundle.contract.max_lateral_offset_m, 1.0)
        payload = bundle.to_dict()
        self.assertIn("contract", payload)
        self.assertIn("soft_vetoes", payload)

    def test_soft_cost_profile_labels(self) -> None:
        self.assertEqual(soft_cost_profile(highest_risk="low", interaction_severity=0.1, soft_veto_count=0), "AGENT_BOUNDED")
        self.assertEqual(
            soft_cost_profile(highest_risk="high", interaction_severity=0.6, soft_veto_count=1),
            "AGENT_BOUNDED_CAUTIOUS",
        )
        self.assertEqual(
            soft_cost_profile(highest_risk="critical", interaction_severity=0.2, soft_veto_count=0),
            "AGENT_BOUNDED_DEFENSIVE",
        )

    def test_contract_resolver_soft_kwargs(self) -> None:
        bundle = build_soft_contract_from_behavior(
            frame_id=1,
            sim_time_s=0.0,
            maneuver="keep_lane",
            ego_speed_mps=5.0,
            target_speed_mps=5.0,
            min_ttc_s=4.0,
            highest_risk="high",
            interaction_severity=0.55,
            confidence=0.9,
        )
        profile = soft_cost_profile(
            highest_risk="high",
            interaction_severity=0.55,
            soft_veto_count=len(bundle.soft_vetoes),
        )
        req = ContractResolver().resolve(
            bundle.contract.tactical_intent,
            bundle.contract,
            cost_profile=profile,
            forward_clear_m=18.0,
            soft_speed_scale=0.8,
        )
        self.assertEqual(req.cost_profile, profile)
        self.assertEqual(req.drivable_envelope.forward_clear_m, 18.0)
        self.assertLess(req.target_v_desired_mps, bundle.contract.max_speed_mps + 1e-6)


if __name__ == "__main__":
    unittest.main()
