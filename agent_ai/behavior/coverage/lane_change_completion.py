from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class LaneChangeCompletionCriteria:
    success_hold_ticks: int = 4
    max_duration_ticks: int = 45
    min_lateral_shift_m: float = 0.50
    min_lateral_shift_fraction: float = 0.15

    def required_lateral_shift_m(self, lane_width_m: float) -> float:
        lane_width = max(float(lane_width_m), 0.1)
        return max(float(self.min_lateral_shift_m), lane_width * float(self.min_lateral_shift_fraction))


def evaluate_lane_change_step(
    *,
    current_lane_id: int | None,
    target_lane_id: int | None,
    lateral_shift_m: float,
    lane_width_m: float,
    stable_ticks: int,
    criteria: LaneChangeCompletionCriteria,
) -> Dict[str, Any]:
    required_shift_m = criteria.required_lateral_shift_m(float(lane_width_m))
    max_progress_denom = max(required_shift_m, 0.1)
    target_lane_reached = target_lane_id is not None and current_lane_id == target_lane_id
    lateral_shift_ready = abs(float(lateral_shift_m)) >= required_shift_m
    next_stable_ticks = stable_ticks + 1 if target_lane_reached and lateral_shift_ready else 0
    success = bool(target_lane_reached and lateral_shift_ready and next_stable_ticks >= criteria.success_hold_ticks)

    reason = None
    if target_lane_reached and not lateral_shift_ready:
        reason = "target_lane_reached_before_lateral_shift_threshold"
    elif target_lane_reached and lateral_shift_ready:
        reason = "target_lane_stabilizing"

    return {
        "required_lateral_shift_m": float(required_shift_m),
        "progress": float(min(1.0, abs(float(lateral_shift_m)) / max_progress_denom)),
        "target_lane_reached": bool(target_lane_reached),
        "lateral_shift_ready": bool(lateral_shift_ready),
        "stable_ticks": int(next_stable_ticks),
        "success": bool(success),
        "reason": reason,
    }
