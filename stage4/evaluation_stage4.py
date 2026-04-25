from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List

from stage3c.evaluation_stage3c import summarize_stage3c_session


def _mean_or_none(values: List[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(mean(filtered))


def summarize_stage4_session(
    *,
    tick_records: Iterable[Dict[str, Any]],
    behavior_updates: Iterable[Dict[str, Any]],
    execution_records: Iterable[Dict[str, Any]],
    lane_change_events: Iterable[Dict[str, Any]],
    dropped_frames: int,
    skipped_updates: int,
    state_resets: int,
    stage1_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    tick_records = list(tick_records)
    behavior_updates = list(behavior_updates)
    execution_records = list(execution_records)
    lane_change_events = list(lane_change_events)
    stage3c_summary = summarize_stage3c_session(execution_records, lane_change_events)

    behavior_counter = Counter(
        record["selected_behavior"] for record in tick_records if record.get("selected_behavior")
    )
    arbitration_counter = Counter(
        record["arbitration_action"] for record in tick_records if record.get("arbitration_action")
    )
    execution_status_counter = Counter(
        record["execution_status"] for record in tick_records if record.get("execution_status")
    )

    return {
        "num_ticks": int(len(tick_records)),
        "num_behavior_updates": int(len(behavior_updates)),
        "behavior_counts": dict(behavior_counter),
        "execution_status_counts": dict(execution_status_counter),
        "lane_change_attempts": int(stage3c_summary["lane_change_attempts"]),
        "lane_change_successes": int(stage3c_summary["lane_change_successes"]),
        "lane_change_failures": int(stage3c_summary["lane_change_failures"]),
        "stop_successes": int(stage3c_summary["stop_successes"]),
        "behavior_overrides": int(
            arbitration_counter.get("cancel", 0)
            + arbitration_counter.get("downgrade", 0)
            + arbitration_counter.get("emergency_override", 0)
        ),
        "arbitration_counts": dict(arbitration_counter),
        "mean_perception_latency_ms": _mean_or_none([record.get("perception_latency_ms") for record in tick_records]),
        "mean_world_update_latency_ms": _mean_or_none([record.get("world_update_latency_ms") for record in tick_records]),
        "mean_behavior_latency_ms": _mean_or_none([record.get("behavior_latency_ms") for record in tick_records]),
        "mean_execution_latency_ms": _mean_or_none([record.get("execution_latency_ms") for record in tick_records]),
        "mean_planner_prepare_latency_ms": _mean_or_none([record.get("planner_prepare_latency_ms") for record in tick_records]),
        "mean_execution_finalize_latency_ms": _mean_or_none([record.get("execution_finalize_latency_ms") for record in tick_records]),
        "mean_planner_execution_latency_ms": _mean_or_none([record.get("planner_execution_latency_ms") for record in tick_records]),
        "mean_decision_to_execution_latency_ms": _mean_or_none([record.get("decision_to_execution_latency_ms") for record in tick_records]),
        "mean_total_loop_latency_ms": _mean_or_none(
            [
                (
                    record.get("total_loop_latency_ms")
                    if record.get("total_loop_latency_ms") is not None
                    else (
                        (record.get("perception_latency_ms") or 0.0)
                        + (record.get("world_update_latency_ms") or 0.0)
                        + (record.get("behavior_latency_ms") or 0.0)
                        + (record.get("execution_latency_ms") or 0.0)
                    )
                )
                for record in tick_records
            ]
        ),
        "over_budget_rate": _mean_or_none(
            [1.0 if record.get("over_budget") else 0.0 for record in tick_records if record.get("over_budget") is not None]
        ),
        "fallback_activation_rate": _mean_or_none(
            [
                1.0 if record.get("fallback_activated") else 0.0
                for record in tick_records
                if record.get("fallback_activated") is not None
            ]
        ),
        "fallback_reason_coverage": _mean_or_none(
            [
                1.0 if str(record.get("fallback_reason") or "").strip() else 0.0
                for record in tick_records
                if record.get("fallback_activated") is True
            ]
        ),
        "mean_fallback_takeover_latency_ms": _mean_or_none(
            [record.get("fallback_takeover_latency_ms") for record in tick_records]
        ),
        "fallback_missing_when_required_count": int(
            sum(
                1
                for record in tick_records
                if record.get("fallback_required") is True and not bool(record.get("fallback_activated"))
            )
        ),
        "dropped_frames": int(dropped_frames),
        "skipped_updates": int(skipped_updates),
        "state_resets": int(state_resets),
        "stage1_summary": dict(stage1_summary or {}),
        "stage3c_execution_summary": stage3c_summary,
    }
