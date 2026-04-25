from __future__ import annotations

from typing import Any, Dict

from .map_adapter import CarlaMapAdapter
from .schema import LaneContext


def build_lane_context(
    *,
    world_state: Dict[str, Any],
    map_adapter: CarlaMapAdapter,
    forward_horizon_m: float = 60.0,
    waypoint_step_m: float = 5.0,
) -> LaneContext:
    return map_adapter.build_lane_context(
        frame_id=int(world_state["frame"]),
        sample_name=str(world_state["sample_name"]),
        ego_position_world_bevfusion=world_state["ego"]["position_world"],
        horizon_m=float(forward_horizon_m),
        waypoint_step_m=float(waypoint_step_m),
    )
