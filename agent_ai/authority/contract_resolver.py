"""
Stage 9 — Contract Resolver (v2)
===================================
Converts a ManeuverContract (L1 bounded intent) into a TrajectoryRequest (L2).

Key invariant:
  - All MPC constraints are derived ONLY from the contract bounds.
  - No raw actuator fields (steer/throttle/brake) are ever set here.
  - The resolved TrajectoryRequest is then passed to MPC for trajectory optimization.
"""
from __future__ import annotations

from .schemas import DrivableEnvelope, ManeuverContract, TrajectoryRequest


# Map tactical intents to desired-speed hints (relative to contract max_speed_mps)
_INTENT_SPEED_RATIO: dict[str, float] = {
    "keep_lane":              1.00,
    "slow_bypass_right":      0.70,
    "slow_bypass_left":       0.70,
    "prepare_lane_change_right": 0.90,
    "prepare_lane_change_left":  0.90,
    "commit_lane_change_right":  0.95,
    "commit_lane_change_left":   0.95,
    "abort_lane_change":      0.85,
    "pull_over_right":        0.40,
    "safe_stop":              0.00,
}


class ContractResolver:
    """
    Resolve ManeuverContract + current maneuver intent -> TrajectoryRequest.

    Usage:
        req = resolver.resolve(agent_intent, contract)
        cmd = mpc.execute(req)
    """

    def resolve(
        self,
        agent_intent: str,
        contract: ManeuverContract,
    ) -> TrajectoryRequest:
        """
        Translate L1 maneuver intent + contract bounds into L2 TrajectoryRequest.
        MPC is the only entity that will produce L3 actuation from this.
        """
        speed_ratio = _INTENT_SPEED_RATIO.get(agent_intent, 1.0)
        desired_v = contract.max_speed_mps * speed_ratio

        envelope = DrivableEnvelope(
            envelope_uuid=contract.drivable_envelope_uuid or "default",
            left_bound_m=contract.max_lateral_offset_m,
            right_bound_m=contract.max_lateral_offset_m,
            forward_clear_m=100.0,   # placeholder; real value comes from BEVFusion
        )

        return TrajectoryRequest(
            source="CONTRACT_RESOLVER",
            tactical_intent=agent_intent,
            v_max_mps=contract.max_speed_mps,
            a_long_max_mps2=contract.max_longitudinal_accel_mps2,
            a_lat_max_mps2=contract.max_lateral_accel_mps2,
            jerk_max_mps3=contract.max_jerk_mps3,
            lateral_bound_m=contract.max_lateral_offset_m,
            drivable_envelope=envelope,
            target_lane_id=contract.target_lane_id,
            target_v_desired_mps=desired_v,
            horizon_s=min(contract.max_duration_s, 3.0),
            cost_profile="AGENT_BOUNDED",
        )
