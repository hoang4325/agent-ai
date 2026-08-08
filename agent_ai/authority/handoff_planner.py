"""
Stage 9 — Handoff Planner (v2)
================================
Reference-level blending for warm handoff and graceful revoke.

CRITICAL: blending is performed at the TrajectoryRequest (reference/objective) level ONLY.
Raw actuator values (steer, throttle, brake) are NEVER blended here.
MPC remains the sole source of all L3 commands.

Usage:
  # Warm handoff (SUPERVISED_EXECUTION):
  preview = handoff_planner.preview(baseline_req, candidate_req, alpha=0.5)

  # Revoke ramp (AUTHORITY_REVOKE_PENDING):
  transition = handoff_planner.preview(baseline_req, last_agent_req, alpha=revoke_alpha)
"""
from __future__ import annotations

from .schemas import TrajectoryRequest


class HandoffPlanner:
    """
    Smooth reference transition between baseline and agent-bounded trajectory requests.

    alpha = 1.0 → full agent request
    alpha = 0.0 → full baseline request
    """

    def preview(
        self,
        baseline_req: TrajectoryRequest,
        candidate_req: TrajectoryRequest,
        alpha: float,
    ) -> TrajectoryRequest:
        """
        Blend two TrajectoryRequests at the objective/bound level.
        alpha is clamped to [0.0, 1.0].
        The tightest safety constraints (min of accel, jerk) are always taken.
        """
        alpha = max(0.0, min(1.0, alpha))
        return TrajectoryRequest.blend(
            baseline=baseline_req,
            candidate=candidate_req,
            agent_weight=alpha,
        )

    def ramp_to_baseline(
        self,
        baseline_req: TrajectoryRequest,
        last_agent_req: TrajectoryRequest | None,
        revoke_frame: int,
        total_revoke_frames: int = 5,
    ) -> TrajectoryRequest:
        """
        Compute the blended request for the current revoke frame.
        Frame 0 => alpha=1.0 (agent), Frame total => alpha=0.0 (baseline).
        """
        if last_agent_req is None or total_revoke_frames <= 0:
            return baseline_req

        alpha = max(0.0, 1.0 - revoke_frame / total_revoke_frames)
        return self.preview(baseline_req, last_agent_req, alpha=alpha)
