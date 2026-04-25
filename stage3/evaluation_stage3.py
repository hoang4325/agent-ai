from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


def summarize_stage3_session(frame_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(frame_records)
    if not records:
        return {
            "num_frames": 0,
            "behavior_counts": {},
            "stage2_maneuver_counts": {},
        }

    behavior_counter = Counter(record["behavior_request"]["requested_behavior"] for record in records)
    stage2_counter = Counter(record["stage2_decision"]["maneuver"] for record in records)
    override_count = sum(1 for record in records if bool(record["comparison"]["stage3_overrode_stage2"]))
    current_lane_blocked_frames = sum(
        1 for record in records if bool(record["lane_scene"].get("current_lane_blocked"))
    )
    junction_frames = sum(
        1 for record in records if bool(record["lane_context"]["junction_context"]["is_in_junction"])
    )
    left_allowed = sum(
        1 for record in records if bool(record["maneuver_validation"]["lane_change_permission"]["left"])
    )
    right_allowed = sum(
        1 for record in records if bool(record["maneuver_validation"]["lane_change_permission"]["right"])
    )
    lane_objects_per_frame = [
        len(record["lane_relative_objects"])
        for record in records
    ]
    current_lane_objects_per_frame = [
        sum(1 for item in record["lane_relative_objects"] if item["lane_relation"] == "current_lane")
        for record in records
    ]
    branch_counts = [
        int(record["lane_context"]["junction_context"]["branch_count_ahead"])
        for record in records
    ]
    return {
        "num_frames": int(len(records)),
        "behavior_counts": dict(behavior_counter),
        "stage2_maneuver_counts": dict(stage2_counter),
        "override_count": int(override_count),
        "override_rate": float(override_count / len(records)),
        "current_lane_blocked_frames": int(current_lane_blocked_frames),
        "junction_frames": int(junction_frames),
        "left_lane_change_permitted_frames": int(left_allowed),
        "right_lane_change_permitted_frames": int(right_allowed),
        "average_lane_objects_per_frame": float(sum(lane_objects_per_frame) / len(records)),
        "average_current_lane_objects_per_frame": float(
            sum(current_lane_objects_per_frame) / len(records)
        ),
        "average_branch_count_ahead": float(sum(branch_counts) / len(records)),
        "max_route_progress_m": float(max(record["world_state"]["ego"]["route_progress_m"] for record in records)),
    }
