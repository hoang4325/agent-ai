from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


def summarize_stage3b_session(frame_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(frame_records)
    if not records:
        return {
            "num_frames": 0,
            "behavior_counts": {},
            "route_option_counts": {},
            "lane_change_stage_counts": {},
        }

    behavior_counter = Counter(record["behavior_request_v2"]["requested_behavior"] for record in records)
    stage3a_counter = Counter(record["stage3a_behavior_request"]["requested_behavior"] for record in records)
    stage2_counter = Counter(record["stage2_decision"]["maneuver"] for record in records)
    route_option_counter = Counter(record["route_context"]["route_option"] for record in records)
    route_mode_counter = Counter(record["route_context"]["route_mode"] for record in records)
    preferred_lane_counter = Counter(record["route_context"]["preferred_lane"] for record in records)
    lane_change_stage_counter = Counter(record["behavior_selection"]["lane_change_stage"] for record in records)

    route_conditioned_override_count = sum(
        1
        for record in records
        if bool(record["comparison"]["stage3b_overrode_stage3a"])
    )
    route_influenced_frames = sum(
        1
        for record in records
        if bool(record["behavior_selection"]["route_influence"])
    )
    prepare_frames = sum(
        1
        for record in records
        if str(record["behavior_selection"]["lane_change_stage"]) == "prepare"
    )
    commit_frames = sum(
        1
        for record in records
        if str(record["behavior_selection"]["lane_change_stage"]) == "commit"
    )
    keep_route_frames = sum(
        1
        for record in records
        if str(record["behavior_request_v2"]["requested_behavior"]) == "keep_route_through_junction"
    )
    left_prepare_frames = sum(
        1
        for record in records
        if str(record["behavior_request_v2"]["requested_behavior"]) == "prepare_lane_change_left"
    )
    right_prepare_frames = sum(
        1
        for record in records
        if str(record["behavior_request_v2"]["requested_behavior"]) == "prepare_lane_change_right"
    )
    left_commit_frames = sum(
        1
        for record in records
        if str(record["behavior_request_v2"]["requested_behavior"]) == "commit_lane_change_left"
    )
    right_commit_frames = sum(
        1
        for record in records
        if str(record["behavior_request_v2"]["requested_behavior"]) == "commit_lane_change_right"
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

    return {
        "num_frames": int(len(records)),
        "behavior_counts": dict(behavior_counter),
        "stage3a_behavior_counts": dict(stage3a_counter),
        "stage2_maneuver_counts": dict(stage2_counter),
        "route_option_counts": dict(route_option_counter),
        "route_mode_counts": dict(route_mode_counter),
        "preferred_lane_counts": dict(preferred_lane_counter),
        "lane_change_stage_counts": dict(lane_change_stage_counter),
        "route_conditioned_override_count": int(route_conditioned_override_count),
        "route_conditioned_override_rate": float(route_conditioned_override_count / len(records)),
        "route_influenced_frames": int(route_influenced_frames),
        "prepare_lane_change_frames": int(prepare_frames),
        "commit_lane_change_frames": int(commit_frames),
        "keep_route_through_junction_frames": int(keep_route_frames),
        "left_prepare_frames": int(left_prepare_frames),
        "right_prepare_frames": int(right_prepare_frames),
        "left_commit_frames": int(left_commit_frames),
        "right_commit_frames": int(right_commit_frames),
        "junction_frames": int(junction_frames),
        "left_lane_change_permitted_frames": int(left_allowed),
        "right_lane_change_permitted_frames": int(right_allowed),
        "max_route_progress_m": float(max(record["world_state"]["ego"]["route_progress_m"] for record in records)),
    }
