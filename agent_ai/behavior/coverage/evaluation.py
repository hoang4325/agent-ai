from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from .stop_controller_tuner import StopControllerConfig, classify_stop_event


TERMINAL_LANE_CHANGE_EVENT_TYPES = {"success", "fail", "aborted"}
LANE_CHANGE_REQUEST_BEHAVIORS = {
    "prepare_lane_change_left",
    "commit_lane_change_left",
    "prepare_lane_change_right",
    "commit_lane_change_right",
}


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _contiguous_request_windows(
    records: List[Dict[str, Any]],
    *,
    predicate,
) -> List[List[Dict[str, Any]]]:
    windows: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for record in records:
        requested_behavior = str((record.get("execution_request") or {}).get("requested_behavior"))
        if predicate(requested_behavior):
            current.append(record)
            continue
        if current:
            windows.append(current)
            current = []
    if current:
        windows.append(current)
    return windows


def _lane_change_direction(requested_behavior: str | None) -> str | None:
    if not requested_behavior:
        return None
    if requested_behavior.endswith("_left"):
        return "left"
    if requested_behavior.endswith("_right"):
        return "right"
    return None


def derive_stop_events(
    execution_records: Iterable[Dict[str, Any]],
    *,
    config: StopControllerConfig | None = None,
) -> List[Dict[str, Any]]:
    config = config or StopControllerConfig()
    records = list(execution_records)
    events: List[Dict[str, Any]] = []
    stop_windows = _contiguous_request_windows(
        records,
        predicate=lambda behavior: behavior == "stop_before_obstacle",
    )
    for window in stop_windows:
        if not window:
            continue

        min_speed_mps = min(float(record["execution_state"]["speed_mps"]) for record in window)
        obstacle_distances = [
            float(record["execution_state"]["distance_to_obstacle_m"])
            for record in window
            if record["execution_state"].get("distance_to_obstacle_m") is not None
        ]
        min_distance_to_obstacle_m = None if not obstacle_distances else min(obstacle_distances)

        hold_ticks = 0
        longest_hold_ticks = 0
        stop_frame = None
        for record in window:
            speed_mps = float(record["execution_state"]["speed_mps"])
            if speed_mps <= config.full_stop_speed_threshold_mps:
                hold_ticks += 1
                longest_hold_ticks = max(longest_hold_ticks, hold_ticks)
                if stop_frame is None:
                    stop_frame = int(record["carla_frame"])
            else:
                hold_ticks = 0

        classification = classify_stop_event(
            min_speed_mps=min_speed_mps,
            min_distance_to_obstacle_m=min_distance_to_obstacle_m,
            hold_ticks=longest_hold_ticks,
            config=config,
        )
        start_record = window[0]
        end_record = window[-1]
        events.append(
            {
                "start_frame": int(start_record["carla_frame"]),
                "stop_frame": stop_frame,
                "request_frame": int(start_record["request_frame"]),
                "sample_name": str(start_record["sample_name"]),
                "result": str(classification["result"]),
                "distance_to_target_at_stop_m": min_distance_to_obstacle_m,
                "hold_ticks": int(longest_hold_ticks),
                "min_speed_mps": float(min_speed_mps),
                "reason": str(classification["reason"]),
                "end_frame": int(end_record["carla_frame"]),
            }
        )
    return events


def derive_enriched_lane_change_events(raw_events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    active: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    enriched: List[Dict[str, Any]] = []
    for event in raw_events:
        key = (
            event.get("sample_name"),
            event.get("request_frame"),
            event.get("direction"),
            event.get("start_lane_id"),
            event.get("target_lane_id"),
        )
        event_type = str(event.get("event_type"))
        if event_type == "start":
            active[key] = dict(event)
            continue
        if event_type not in TERMINAL_LANE_CHANGE_EVENT_TYPES:
            continue
        start_event = active.pop(key, None)
        if start_event is None:
            start_event = dict(event)
        enriched.append(
            {
                "start_frame": int(start_event.get("carla_frame", start_event.get("request_frame", 0))),
                "end_frame": int(event.get("carla_frame", event.get("request_frame", 0))),
                "request_frame": int(event.get("request_frame", start_event.get("request_frame", 0))),
                "sample_name": str(event.get("sample_name", start_event.get("sample_name", ""))),
                "direction": str(event.get("direction", start_event.get("direction", "unknown"))),
                "start_lane_id": start_event.get("start_lane_id"),
                "target_lane_id": event.get("target_lane_id", start_event.get("target_lane_id")),
                "result": "abort" if event_type == "aborted" else event_type,
                "completion_time_s": event.get("completion_time_s"),
                "max_lateral_shift_m": float(
                    event.get(
                        "max_lateral_shift_m",
                        abs(float(event.get("lateral_shift_m", 0.0))),
                    )
                ),
                "required_lateral_shift_m": event.get("required_lateral_shift_m"),
                "reason": event.get("completion_reason") or ";".join(event.get("notes", [])),
            }
        )
    for start_event in active.values():
        enriched.append(
            {
                "start_frame": int(start_event.get("carla_frame", start_event.get("request_frame", 0))),
                "end_frame": None,
                "request_frame": int(start_event.get("request_frame", 0)),
                "sample_name": str(start_event.get("sample_name", "")),
                "direction": str(start_event.get("direction", "unknown")),
                "start_lane_id": start_event.get("start_lane_id"),
                "target_lane_id": start_event.get("target_lane_id"),
                "result": "fail",
                "completion_time_s": None,
                "max_lateral_shift_m": float(abs(float(start_event.get("lateral_shift_m", 0.0)))),
                "required_lateral_shift_m": start_event.get("required_lateral_shift_m"),
                "reason": "lane_change_terminal_event_missing",
            }
        )
    return enriched


def derive_fallback_lane_change_events(
    execution_records: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records = list(execution_records)
    if not records:
        return []

    episodes: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_direction: str | None = None
    for record in records:
        requested_behavior = str((record.get("execution_request") or {}).get("requested_behavior"))
        direction = _lane_change_direction(requested_behavior) if requested_behavior in LANE_CHANGE_REQUEST_BEHAVIORS else None
        if direction is None:
            if current:
                episodes.append(current)
                current = []
                current_direction = None
            continue
        if current and direction != current_direction:
            episodes.append(current)
            current = []
        current.append(record)
        current_direction = direction
    if current:
        episodes.append(current)

    events: List[Dict[str, Any]] = []
    for window in episodes:
        if not window:
            continue
        requested_behaviors = [
            str((record.get("execution_request") or {}).get("requested_behavior"))
            for record in window
        ]
        commit_rows = [
            record
            for record in window
            if str((record.get("execution_request") or {}).get("requested_behavior", "")).startswith("commit_lane_change")
        ]
        start_record = commit_rows[0] if commit_rows else window[0]
        end_record = window[-1]
        first_request = start_record.get("execution_request") or {}
        direction = _lane_change_direction(str(first_request.get("requested_behavior"))) or "unknown"
        initial_lane_id = (window[0].get("execution_state") or {}).get("current_lane_id")
        final_lane_id = (end_record.get("execution_state") or {}).get("current_lane_id")
        max_lateral_shift_m = max(
            abs(float((record.get("execution_state") or {}).get("lateral_shift_m", 0.0)))
            for record in window
        )
        completion_time_s = None
        if start_record.get("timestamp") is not None and end_record.get("timestamp") is not None:
            completion_time_s = float(end_record["timestamp"]) - float(start_record["timestamp"])

        success = any(
            str((record.get("execution_state") or {}).get("execution_status")) == "success"
            and str((record.get("execution_state") or {}).get("current_behavior", "")).startswith("commit_lane_change")
            for record in window
        )
        if not success and initial_lane_id is not None and final_lane_id is not None and final_lane_id != initial_lane_id:
            success = True
        if not success and any(
            float((record.get("execution_state") or {}).get("lane_change_progress", 0.0)) >= 1.0
            for record in window
        ):
            success = True

        notes = []
        for record in window:
            notes.extend(list((record.get("execution_state") or {}).get("notes") or []))
        reason = "derived_from_execution_without_explicit_lane_change_events"
        if not success:
            unique_notes = sorted(set(str(note) for note in notes if note))
            if unique_notes:
                reason = ";".join(unique_notes)
            else:
                reason = "lane_change_terminal_event_missing"

        events.append(
            {
                "start_frame": int(start_record.get("carla_frame", first_request.get("frame", 0))),
                "end_frame": int(end_record.get("carla_frame", first_request.get("frame", 0))),
                "request_frame": int(first_request.get("frame", 0)),
                "sample_name": str(start_record.get("sample_name", "")),
                "direction": direction,
                "start_lane_id": initial_lane_id,
                "target_lane_id": (end_record.get("execution_state") or {}).get("target_lane_id"),
                "result": "success" if success else "fail",
                "completion_time_s": completion_time_s,
                "max_lateral_shift_m": float(max_lateral_shift_m),
                "required_lateral_shift_m": None,
                "reason": reason,
                "requested_behaviors": requested_behaviors,
                "derived_from_execution_timeline": True,
            }
        )
    return events


def derive_lane_change_events(
    *,
    execution_records: Iterable[Dict[str, Any]],
    raw_lane_change_events: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched = derive_enriched_lane_change_events(raw_lane_change_events)
    if enriched:
        return enriched
    return derive_fallback_lane_change_events(execution_records)


def build_stage3c_run_metrics(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    execution_records = load_jsonl(run_dir / "execution_timeline.jsonl")
    raw_lane_change_events = load_jsonl(run_dir / "lane_change_events.jsonl")
    summary = load_json(run_dir / "execution_summary.json") if (run_dir / "execution_summary.json").exists() else {}
    stage3c_manifest = load_json(run_dir / "stage3c_manifest.json") if (run_dir / "stage3c_manifest.json").exists() else {}
    from .planner_quality import build_planner_quality_bundle, load_planner_quality_bundle

    planner_bundle = load_planner_quality_bundle(run_dir)
    planner_quality_summary = planner_bundle.get("planner_quality_summary") or {}
    if not planner_quality_summary:
        planner_bundle = build_planner_quality_bundle(execution_records, raw_lane_change_events)
        planner_quality_summary = planner_bundle.get("planner_quality_summary") or {}

    stop_events = derive_stop_events(execution_records)
    lane_change_events = derive_lane_change_events(
        execution_records=execution_records,
        raw_lane_change_events=raw_lane_change_events,
    )

    requested_counter = Counter(record["execution_request"]["requested_behavior"] for record in execution_records)
    executed_counter = Counter(record["execution_state"]["current_behavior"] for record in execution_records)
    initial_lane_id = None if not execution_records else execution_records[0]["execution_state"]["current_lane_id"]
    final_lane_id = None if not execution_records else execution_records[-1]["execution_state"]["current_lane_id"]
    junction_frames = sum(1 for record in execution_records if bool(record["execution_state"].get("is_in_junction")))
    right_successes = sum(1 for event in lane_change_events if event["direction"] == "right" and event["result"] == "success")
    left_successes = sum(1 for event in lane_change_events if event["direction"] == "left" and event["result"] == "success")

    lane_change_success_times = [
        float(event["completion_time_s"])
        for event in lane_change_events
        if event["result"] == "success" and event.get("completion_time_s") is not None
    ]
    lane_change_max_shifts = [
        float(event["max_lateral_shift_m"])
        for event in lane_change_events
        if event.get("max_lateral_shift_m") is not None
    ]

    return {
        "run_name": str(run_dir.name),
        "run_dir": str(run_dir),
        "summary": summary,
        "stage3c_manifest": stage3c_manifest,
        "num_frames": int(summary.get("num_frames", len(execution_records))),
        "requested_behavior_counts": dict(requested_counter),
        "executed_behavior_counts": dict(executed_counter),
        "initial_lane_id": initial_lane_id,
        "final_lane_id": final_lane_id,
        "junction_frames": int(junction_frames),
        "lane_change_events": lane_change_events,
        "stop_events": stop_events,
        "lane_change_attempts": int(len(lane_change_events)),
        "lane_change_successes": int(sum(1 for event in lane_change_events if event["result"] == "success")),
        "lane_change_failures": int(sum(1 for event in lane_change_events if event["result"] in {"fail", "abort"})),
        "right_lane_change_successes": int(right_successes),
        "left_lane_change_successes": int(left_successes),
        "stop_attempts": int(len(stop_events)),
        "stop_successes": int(sum(1 for event in stop_events if event["result"] == "success")),
        "stop_overshoots": int(sum(1 for event in stop_events if event["result"] == "overshoot")),
        "stop_aborts": int(sum(1 for event in stop_events if event["result"] == "abort")),
        "mean_lane_change_time_s": (None if not lane_change_success_times else float(mean(lane_change_success_times))),
        "max_lane_change_lateral_shift_m": (None if not lane_change_max_shifts else float(max(lane_change_max_shifts))),
        "planner_quality_summary": planner_quality_summary,
        "lane_change_events_enriched": planner_bundle.get("lane_change_events_enriched") or lane_change_events,
        "stop_events_enriched": planner_bundle.get("stop_events") or stop_events,
        "trajectory_trace": planner_bundle.get("trajectory_trace") or [],
        "control_trace": planner_bundle.get("control_trace") or [],
        "lane_change_phase_trace": planner_bundle.get("lane_change_phase_trace") or [],
        "stop_target": planner_bundle.get("stop_target") or {},
        "planner_control_latency": planner_bundle.get("planner_control_latency") or [],
        "fallback_events": planner_bundle.get("fallback_events") or [],
        "stop_target_defined_rate": planner_quality_summary.get("stop_target_defined_rate"),
        "lane_change_completion_rate": planner_quality_summary.get("lane_change_completion_rate"),
        "lane_change_completion_time_s": planner_quality_summary.get("mean_lane_change_completion_time_s"),
        "lane_change_abort_rate": planner_quality_summary.get("lane_change_abort_rate"),
        "lane_change_completion_validity": planner_quality_summary.get("lane_change_completion_validity"),
        "stop_hold_stability": planner_quality_summary.get("stop_hold_stability"),
        "stop_final_error_m": planner_quality_summary.get("stop_final_error_m"),
        "stop_overshoot_distance_m": planner_quality_summary.get("stop_overshoot_distance_m"),
        "heading_settle_time_s": planner_quality_summary.get("heading_settle_time_s"),
        "lateral_oscillation_rate": planner_quality_summary.get("lateral_oscillation_rate"),
        "trajectory_smoothness_proxy": planner_quality_summary.get("trajectory_smoothness_proxy"),
        "control_jerk_proxy": planner_quality_summary.get("control_jerk_proxy"),
        "planner_execution_latency_ms": planner_quality_summary.get("planner_execution_latency_ms"),
        "decision_to_execution_latency_ms": planner_quality_summary.get("decision_to_execution_latency_ms"),
        "over_budget_rate": planner_quality_summary.get("over_budget_rate"),
        "fallback_activation_rate": planner_quality_summary.get("fallback_activation_rate"),
        "fallback_reason_coverage": planner_quality_summary.get("fallback_reason_coverage"),
        "fallback_takeover_latency_ms": planner_quality_summary.get("fallback_takeover_latency_ms"),
        "fallback_missing_when_required_count": planner_quality_summary.get("fallback_missing_when_required_count"),
        "suspected_no_reset_rerun": bool(
            requested_counter.get("commit_lane_change_right", 0) > 0 and len(lane_change_events) == 0
        ),
    }


def select_best_run(
    run_metrics: Iterable[Dict[str, Any]],
    *,
    direction: str | None = None,
    require_success: bool = False,
) -> Dict[str, Any] | None:
    candidates = []
    for metrics in run_metrics:
        if direction == "right" and int(metrics.get("requested_behavior_counts", {}).get("commit_lane_change_right", 0)) <= 0:
            continue
        if direction == "left" and int(metrics.get("requested_behavior_counts", {}).get("commit_lane_change_left", 0)) <= 0:
            continue
        if require_success:
            key_name = f"{direction}_lane_change_successes" if direction in {"left", "right"} else "lane_change_successes"
            if int(metrics.get(key_name, metrics.get("lane_change_successes", 0))) <= 0:
                continue
        candidates.append(metrics)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            int(item.get("lane_change_successes", 0)),
            int(item.get("stop_successes", 0)),
            -int(item.get("lane_change_failures", 0)),
            int(item.get("num_frames", 0)),
            str(item.get("run_name", "")),
        ),
        reverse=True,
    )
    return candidates[0]
