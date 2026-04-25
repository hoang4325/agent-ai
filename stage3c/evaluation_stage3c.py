from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List

from stage3c_coverage.evaluation_stage3c_coverage import derive_lane_change_events, derive_stop_events


def summarize_stage3c_session(
    execution_records: Iterable[Dict[str, Any]],
    lane_change_events: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    records = list(execution_records)
    raw_events = list(lane_change_events)
    if not records:
        return {
            "num_frames": 0,
            "lane_change_attempts": 0,
            "lane_change_successes": 0,
            "lane_change_failures": 0,
            "stop_attempts": 0,
            "stop_successes": 0,
            "behavior_overrides": 0,
            "mean_lane_change_time_s": None,
        }

    stop_events = derive_stop_events(records)
    events = derive_lane_change_events(
        execution_records=records,
        raw_lane_change_events=raw_events,
    )

    requested_counter = Counter(record["execution_request"]["requested_behavior"] for record in records)
    executed_counter = Counter(record["execution_state"]["current_behavior"] for record in records)
    execution_status_counter = Counter(record["execution_state"]["execution_status"] for record in records)

    if events and "event_type" not in events[0]:
        lane_change_attempts = len(events)
    else:
        lane_change_attempts = sum(1 for event in events if event.get("event_type") == "start")
    lane_change_successes = sum(
        1
        for event in events
        if event.get("event_type") == "success" or event.get("result") == "success"
    )
    lane_change_failures = sum(
        1
        for event in events
        if event.get("event_type") in {"fail", "aborted"} or event.get("result") in {"fail", "abort"}
    )

    stop_attempts = len(stop_events)
    stop_successes = sum(1 for event in stop_events if event.get("result") == "success")
    slow_down_compliance_frames = sum(
        1
        for record in records
        if record["execution_request"]["requested_behavior"] in {"slow_down", "prepare_lane_change_left", "prepare_lane_change_right"}
        and record["execution_state"]["speed_mps"] <= record["execution_request"]["target_speed_mps"] + 0.75
    )
    slow_down_total_frames = sum(
        1
        for record in records
        if record["execution_request"]["requested_behavior"] in {"slow_down", "prepare_lane_change_left", "prepare_lane_change_right"}
    )
    behavior_overrides = sum(
        1
        for record in records
        if record["execution_request"]["requested_behavior"] != record["execution_state"]["current_behavior"]
    )
    lane_change_times = [
        float(event["completion_time_s"])
        for event in events
        if (event.get("event_type") == "success" or event.get("result") == "success") and event.get("completion_time_s") is not None
    ]

    return {
        "num_frames": int(len(records)),
        "requested_behavior_counts": dict(requested_counter),
        "executed_behavior_counts": dict(executed_counter),
        "execution_status_counts": dict(execution_status_counter),
        "lane_change_attempts": int(lane_change_attempts),
        "lane_change_successes": int(lane_change_successes),
        "lane_change_failures": int(lane_change_failures),
        "stop_attempts": int(stop_attempts),
        "stop_successes": int(stop_successes),
        "behavior_overrides": int(behavior_overrides),
        "behavior_override_rate": float(behavior_overrides / len(records)),
        "slow_down_compliance_frames": int(slow_down_compliance_frames),
        "slow_down_compliance_rate": (
            None if slow_down_total_frames == 0 else float(slow_down_compliance_frames / slow_down_total_frames)
        ),
        "mean_lane_change_time_s": (None if not lane_change_times else float(mean(lane_change_times))),
        "max_speed_mps": float(max(record["execution_state"]["speed_mps"] for record in records)),
    }
