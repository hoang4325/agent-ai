from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from stage3c_coverage.planner_quality import write_planner_quality_artifacts

from .io import dump_json, dump_jsonl, load_json, load_jsonl
from .kinematic_mpc_shadow import solve_kinematic_shadow_rollout


LOWER_BETTER_METRICS = {
    "shadow_stop_final_error_m",
    "shadow_stop_overshoot_distance_m",
    "shadow_lane_change_completion_time_s",
    "shadow_heading_settle_time_s",
    "shadow_lateral_oscillation_rate",
    "shadow_trajectory_smoothness_proxy",
    "shadow_control_jerk_proxy",
    "shadow_planner_execution_latency_ms",
    "shadow_realized_speed_mae_mps",
    "shadow_realized_progress_mae_m",
    "shadow_realized_stop_distance_mae_m",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean_or_none(values: list[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(mean(cleaned))


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _case_id_from_stage3c_dir(stage3c_dir: Path) -> str:
    parent = stage3c_dir.parent
    if parent.name:
        return parent.name
    return stage3c_dir.name


def _load_or_build_planner_quality(stage3c_dir: Path) -> dict[str, Any]:
    planner_quality_path = stage3c_dir / "planner_quality_summary.json"
    if not planner_quality_path.exists():
        write_planner_quality_artifacts(stage3c_dir)
    return load_json(planner_quality_path, default={}) or {}


def _stop_event_indexes(stop_target_payload: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], list[int]]:
    by_frame: dict[int, dict[str, Any]] = {}
    request_frames: list[int] = []
    for event in stop_target_payload.get("stop_events") or []:
        request_frame = _safe_int(event.get("request_frame"))
        if request_frame is None:
            continue
        by_frame[request_frame] = dict(event)
        request_frames.append(request_frame)
    request_frames.sort()
    return by_frame, request_frames


def _control_row_by_frame(control_trace: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in control_trace:
        frame = _safe_int(row.get("carla_frame"))
        if frame is None:
            continue
        result[frame] = dict(row)
    return result


def _execution_row_by_frame(execution_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in execution_rows:
        frame = _safe_int(row.get("carla_frame"))
        if frame is None:
            continue
        result[frame] = dict(row)
    return result


def _latency_row_by_frame(latency_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in latency_rows:
        frame = _safe_int(row.get("carla_frame"))
        if frame is None:
            frame = _safe_int(row.get("request_frame"))
        if frame is None:
            continue
        result[frame] = dict(row)
    return result


def _proposal_latency_ms(*, requested_behavior: str, speed_error_mps: float, in_junction: bool) -> float:
    complexity = 1.0
    if "lane_change" in requested_behavior:
        complexity += 0.7
    if requested_behavior == "stop_before_obstacle":
        complexity += 0.5
    if in_junction:
        complexity += 0.3
    return round(0.95 + 0.55 * complexity + 0.06 * min(abs(speed_error_mps), 10.0), 6)


def _shadow_behavior_for_row(
    *,
    request_frame: int | None,
    requested_behavior: str,
    lane_change_progress: float,
    distance_to_target_lane_center: float | None,
    stop_event_frames: list[int],
) -> tuple[str, list[str]]:
    notes: list[str] = []
    recommended = requested_behavior
    if requested_behavior.startswith("commit_lane_change") and lane_change_progress < 0.05:
        if distance_to_target_lane_center is not None and distance_to_target_lane_center > 2.5:
            recommended = requested_behavior.replace("commit", "prepare", 1)
            notes.append("shadow_prefers_prepare_before_commit")
    if request_frame is not None:
        next_stop = next((frame for frame in stop_event_frames if frame >= request_frame), None)
        if next_stop is not None:
            delta = int(next_stop - request_frame)
            if 0 < delta <= 1 and requested_behavior != "stop_before_obstacle":
                recommended = "stop_before_obstacle"
                notes.append("shadow_prefers_immediate_stop_for_upcoming_stop_window")
            elif 1 < delta <= 3 and requested_behavior in {"keep_lane", "follow"}:
                recommended = "slow_down"
                notes.append("shadow_prefers_pre_stop_slow_down")
    return recommended, notes


def _proposal_target_speed(
    *,
    requested_behavior: str,
    recommended_behavior: str,
    baseline_target_speed_mps: float | None,
) -> float | None:
    baseline_target_speed_mps = _safe_float(baseline_target_speed_mps)
    if baseline_target_speed_mps is None:
        return None
    if recommended_behavior == "stop_before_obstacle":
        return 0.0
    if recommended_behavior == "slow_down":
        return float(min(baseline_target_speed_mps, 3.0))
    if requested_behavior.startswith("commit_lane_change") and recommended_behavior.startswith("prepare_lane_change"):
        return float(min(baseline_target_speed_mps, 4.0))
    return float(baseline_target_speed_mps)


def _comparison_row(*, metric: str, shadow_value: Any, baseline_value: Any) -> dict[str, Any]:
    if shadow_value is None or baseline_value is None:
        return {
            "metric": metric,
            "baseline_value": baseline_value,
            "shadow_value": shadow_value,
            "delta": None,
            "trend": "lower_better" if metric in LOWER_BETTER_METRICS else "higher_better",
            "classification": "unavailable",
        }
    delta = float(shadow_value) - float(baseline_value)
    if metric in LOWER_BETTER_METRICS:
        if delta < -1e-6:
            classification = "better"
        elif delta > 1e-6:
            classification = "worse"
        else:
            classification = "equal"
    else:
        if delta > 1e-6:
            classification = "better"
        elif delta < -1e-6:
            classification = "worse"
        else:
            classification = "equal"
    return {
        "metric": metric,
        "baseline_value": baseline_value,
        "shadow_value": shadow_value,
        "delta": delta,
        "trend": "lower_better" if metric in LOWER_BETTER_METRICS else "higher_better",
        "classification": classification,
    }


def _shadow_metric_support(shadow_summary: dict[str, Any], metric_name: str) -> dict[str, Any]:
    support = (shadow_summary.get("shadow_metric_support") or {}).get(metric_name) or {}
    return {
        "support_level": support.get("support_level", "unavailable"),
        "authority": support.get("authority", "proposal_only_non_executed"),
        "recommended_gate_usage": support.get("recommended_gate_usage", "diagnostic_only"),
    }


def _shadow_baseline_metrics(planner_quality_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "shadow_stop_final_error_m": planner_quality_summary.get("stop_final_error_m"),
        "shadow_stop_overshoot_distance_m": planner_quality_summary.get("stop_overshoot_distance_m"),
        "shadow_lane_change_completion_time_s": planner_quality_summary.get("mean_lane_change_completion_time_s"),
        "shadow_heading_settle_time_s": planner_quality_summary.get("heading_settle_time_s"),
        "shadow_lateral_oscillation_rate": planner_quality_summary.get("lateral_oscillation_rate"),
        "shadow_trajectory_smoothness_proxy": planner_quality_summary.get("trajectory_smoothness_proxy"),
        "shadow_control_jerk_proxy": planner_quality_summary.get("control_jerk_proxy"),
        "shadow_planner_execution_latency_ms": planner_quality_summary.get("planner_execution_latency_ms"),
    }


def _shadow_summary_metrics(shadow_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "shadow_stop_final_error_m": shadow_summary.get("shadow_stop_final_error_m"),
        "shadow_stop_overshoot_distance_m": shadow_summary.get("shadow_stop_overshoot_distance_m"),
        "shadow_lane_change_completion_time_s": shadow_summary.get("shadow_lane_change_completion_time_s"),
        "shadow_heading_settle_time_s": shadow_summary.get("shadow_heading_settle_time_s"),
        "shadow_lateral_oscillation_rate": shadow_summary.get("shadow_lateral_oscillation_rate"),
        "shadow_trajectory_smoothness_proxy": shadow_summary.get("shadow_trajectory_smoothness_proxy"),
        "shadow_control_jerk_proxy": shadow_summary.get("shadow_control_jerk_proxy"),
        "shadow_planner_execution_latency_ms": shadow_summary.get("shadow_planner_execution_latency_ms"),
    }


def _pose_xyz(row: dict[str, Any]) -> tuple[float, float, float] | None:
    pose = row.get("vehicle_pose") or {}
    x = _safe_float(pose.get("x_m"))
    y = _safe_float(pose.get("y_m"))
    z = _safe_float(pose.get("z_m"))
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def _distance_xyz(origin: tuple[float, float, float], target: tuple[float, float, float]) -> float:
    dx = float(target[0] - origin[0])
    dy = float(target[1] - origin[1])
    dz = float(target[2] - origin[2])
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _actual_speed_mps(row: dict[str, Any]) -> float | None:
    state = row.get("execution_state") or {}
    speed = _safe_float(state.get("speed_mps"))
    if speed is not None:
        return speed
    kinematics = row.get("vehicle_kinematics") or {}
    return _safe_float(kinematics.get("speed_mps"))


def _actual_behavior(row: dict[str, Any]) -> str | None:
    state = row.get("execution_state") or {}
    return _safe_str(state.get("current_behavior")) or _safe_str(state.get("requested_behavior"))


def _actual_stop_distance_m(row: dict[str, Any]) -> float | None:
    state = row.get("execution_state") or {}
    return _safe_float(state.get("distance_to_stop_target_m"))


def _future_execution_match_rows(
    *,
    origin_row: dict[str, Any],
    future_rows: list[dict[str, Any]],
    sampled_path: list[dict[str, Any]],
    horizon_s: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    origin_ts = _safe_float(origin_row.get("timestamp"))
    if origin_ts is None:
        return []
    target_points = [
        point
        for point in sampled_path
        if (_safe_float(point.get("t_s")) or 0.0) > 0.0 and (_safe_float(point.get("t_s")) or 0.0) <= horizon_s
    ]
    if not target_points:
        return []
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    cursor = 0
    for point in target_points:
        point_ts = origin_ts + float(_safe_float(point.get("t_s")) or 0.0)
        best_index: int | None = None
        best_delta: float | None = None
        for index in range(cursor, len(future_rows)):
            row = future_rows[index]
            row_ts = _safe_float(row.get("timestamp"))
            if row_ts is None:
                continue
            delta = abs(float(row_ts - point_ts))
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_index = index
            if row_ts >= point_ts and best_delta is not None and delta > best_delta:
                break
        if best_index is None or best_delta is None or best_delta > 0.16:
            continue
        matches.append((point, future_rows[best_index]))
        cursor = best_index
    return matches


def _realized_eligible_target_points(
    *,
    origin_row: dict[str, Any],
    sampled_path: list[dict[str, Any]],
    horizon_s: float,
    last_timestamp: float | None,
) -> list[dict[str, Any]]:
    origin_ts = _safe_float(origin_row.get("timestamp"))
    if origin_ts is None:
        return []
    available_future_s = None if last_timestamp is None else max(float(last_timestamp - origin_ts), 0.0)
    effective_horizon_s = float(horizon_s)
    if available_future_s is not None:
        effective_horizon_s = min(effective_horizon_s, available_future_s + 1e-6)
    return [
        point
        for point in sampled_path
        if (_safe_float(point.get("t_s")) or 0.0) > 0.0 and (_safe_float(point.get("t_s")) or 0.0) <= effective_horizon_s
    ]


def build_shadow_realized_outcome_payload(
    *,
    case_id: str,
    proposals: list[dict[str, Any]],
    execution_rows: list[dict[str, Any]],
    horizon_s: float = 0.6,
) -> dict[str, Any]:
    rows_sorted = sorted(execution_rows, key=lambda row: float(_safe_float(row.get("timestamp")) or 0.0))
    rows_by_frame = _execution_row_by_frame(rows_sorted)
    last_timestamp = None if not rows_sorted else _safe_float(rows_sorted[-1].get("timestamp"))
    trace_rows: list[dict[str, Any]] = []
    support_count = 0
    eligible_count = 0
    behavior_matches = 0
    behavior_total = 0
    speed_maes: list[float] = []
    progress_maes: list[float] = []
    stop_distance_maes: list[float] = []
    exact_stop_support = False
    heuristic_stop_support = False

    for proposal in proposals:
        carla_frame = _safe_int(proposal.get("carla_frame"))
        origin_row = rows_by_frame.get(carla_frame or -1)
        if origin_row is None:
            continue
        origin_pose = _pose_xyz(origin_row)
        sampled_path = list(((proposal.get("proposed_trajectory") or {}).get("sampled_path")) or [])
        eligible_target_points = _realized_eligible_target_points(
            origin_row=origin_row,
            sampled_path=sampled_path,
            horizon_s=horizon_s,
            last_timestamp=last_timestamp,
        )
        comparison_eligible = len(eligible_target_points) >= 2
        origin_index = rows_sorted.index(origin_row)
        future_rows = rows_sorted[origin_index:]
        matches = _future_execution_match_rows(
            origin_row=origin_row,
            future_rows=future_rows,
            sampled_path=sampled_path,
            horizon_s=horizon_s,
        )
        speed_errors: list[float] = []
        progress_errors: list[float] = []
        stop_distance_errors: list[float] = []
        match_behavior_count = 0
        for point, actual_row in matches:
            predicted_speed = _safe_float(point.get("speed_mps"))
            actual_speed = _actual_speed_mps(actual_row)
            if predicted_speed is not None and actual_speed is not None:
                speed_errors.append(abs(float(predicted_speed - actual_speed)))
            if origin_pose is not None:
                actual_pose = _pose_xyz(actual_row)
                predicted_progress = _safe_float(point.get("s_m"))
                if actual_pose is not None and predicted_progress is not None:
                    progress_errors.append(abs(float(predicted_progress - _distance_xyz(origin_pose, actual_pose))))
            actual_behavior = _actual_behavior(actual_row)
            if actual_behavior is not None:
                behavior_total += 1
                if actual_behavior == _safe_str(proposal.get("shadow_requested_behavior")):
                    match_behavior_count += 1
                    behavior_matches += 1
            predicted_stop_distance = None
            stop_target_info = proposal.get("stop_target_info") or {}
            target_distance = _safe_float(stop_target_info.get("target_distance_m"))
            point_progress = _safe_float(point.get("s_m"))
            if target_distance is not None and point_progress is not None:
                predicted_stop_distance = max(float(target_distance - point_progress), 0.0)
            actual_stop_distance = _actual_stop_distance_m(actual_row)
            if predicted_stop_distance is not None and actual_stop_distance is not None:
                stop_distance_errors.append(abs(float(predicted_stop_distance - actual_stop_distance)))
                binding_status = _safe_str(stop_target_info.get("binding_status")) or "missing"
                if binding_status == "exact":
                    exact_stop_support = True
                elif binding_status in {"heuristic", "approximate", "bound"}:
                    heuristic_stop_support = True

        comparison_ready = len(matches) >= 2
        if comparison_eligible:
            eligible_count += 1
        if comparison_eligible and comparison_ready:
            support_count += 1
            speed_mae = _mean_or_none(speed_errors)
            progress_mae = _mean_or_none(progress_errors)
            stop_distance_mae = _mean_or_none(stop_distance_errors)
            if speed_mae is not None:
                speed_maes.append(float(speed_mae))
            if progress_mae is not None:
                progress_maes.append(float(progress_mae))
            if stop_distance_mae is not None:
                stop_distance_maes.append(float(stop_distance_mae))
        else:
            speed_mae = None
            progress_mae = None
            stop_distance_mae = None

        trace_rows.append(
            {
                "case_id": case_id,
                "carla_frame": carla_frame,
                "request_frame": _safe_int(proposal.get("request_frame")),
                "sample_name": proposal.get("sample_name"),
                "shadow_requested_behavior": proposal.get("shadow_requested_behavior"),
                "comparison_eligible": comparison_eligible,
                "comparison_ready": comparison_ready,
                "eligible_points": len(eligible_target_points),
                "matched_points": len(matches),
                "horizon_s": float(horizon_s),
                "available_future_horizon_s": (
                    None
                    if last_timestamp is None
                    else max(float(last_timestamp - (_safe_float(origin_row.get("timestamp")) or last_timestamp)), 0.0)
                ),
                "speed_mae_mps": speed_mae,
                "progress_mae_m": progress_mae,
                "stop_distance_mae_m": stop_distance_mae,
                "behavior_match_rate": None if not matches else float(match_behavior_count / len(matches)),
                "support_level": (
                    "realized_short_horizon"
                    if comparison_eligible and comparison_ready
                    else ("terminal_window_excluded" if not comparison_eligible else "unavailable")
                ),
            }
        )

    if exact_stop_support:
        stop_support_level = "realized_short_horizon_with_exact_stop_target"
    elif heuristic_stop_support:
        stop_support_level = "realized_short_horizon_with_heuristic_stop_target"
    else:
        stop_support_level = "unavailable"

    summary = {
        "schema_version": "stage6.shadow_realized_outcome_summary.v1",
        "generated_at_utc": _utc_now_iso(),
        "case_id": case_id,
        "comparison_mode": "proposal_vs_realized_baseline_short_horizon",
        "horizon_s": float(horizon_s),
        "shadow_realized_support_rate": None if eligible_count <= 0 else float(support_count / eligible_count),
        "shadow_realized_behavior_match_rate": None if behavior_total <= 0 else float(behavior_matches / behavior_total),
        "shadow_realized_speed_mae_mps": _mean_or_none(speed_maes),
        "shadow_realized_progress_mae_m": _mean_or_none(progress_maes),
        "shadow_realized_stop_distance_mae_m": _mean_or_none(stop_distance_maes),
        "matched_proposals": int(support_count),
        "eligible_proposals": int(eligible_count),
        "total_proposals": int(len(proposals)),
        "shadow_metric_support_updates": {
            "shadow_realized_support_rate": {
                "support_level": "realized_short_horizon_direct" if eligible_count > 0 else "unavailable",
                "value_available": eligible_count > 0,
                "authority": "proposal_vs_realized_baseline_short_horizon",
                "recommended_gate_usage": "soft_guardrail" if eligible_count > 0 else "diagnostic_only",
            },
            "shadow_realized_behavior_match_rate": {
                "support_level": "realized_short_horizon_direct" if behavior_total > 0 else "unavailable",
                "value_available": behavior_total > 0,
                "authority": "proposal_vs_realized_baseline_short_horizon",
                "recommended_gate_usage": "soft_guardrail" if behavior_total > 0 else "diagnostic_only",
            },
            "shadow_realized_speed_mae_mps": {
                "support_level": "realized_short_horizon_direct" if speed_maes else "unavailable",
                "value_available": bool(speed_maes),
                "authority": "proposal_vs_realized_baseline_short_horizon",
                "recommended_gate_usage": "diagnostic_only",
            },
            "shadow_realized_progress_mae_m": {
                "support_level": "realized_short_horizon_direct" if progress_maes else "unavailable",
                "value_available": bool(progress_maes),
                "authority": "proposal_vs_realized_baseline_short_horizon",
                "recommended_gate_usage": "diagnostic_only",
            },
            "shadow_realized_stop_distance_mae_m": {
                "support_level": stop_support_level,
                "value_available": bool(stop_distance_maes),
                "authority": "proposal_vs_realized_baseline_short_horizon",
                "recommended_gate_usage": "diagnostic_only",
            },
        },
        "warnings": [
            "realized_outcome_comparison_uses_baseline_future_execution_not_shadow_takeover",
            "realized_support_rate_uses_eligible_proposals_only_excludes_tail_truncated_windows",
        ],
    }
    return {
        "trace_rows": trace_rows,
        "summary": summary,
    }


def _merge_realized_outcome_into_shadow_summary(
    shadow_summary: dict[str, Any],
    realized_summary: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(shadow_summary)
    support_map = dict(updated.get("shadow_metric_support") or {})
    support_map.update(realized_summary.get("shadow_metric_support_updates") or {})
    updated["shadow_metric_support"] = support_map
    for metric_name in [
        "shadow_realized_support_rate",
        "shadow_realized_behavior_match_rate",
        "shadow_realized_speed_mae_mps",
        "shadow_realized_progress_mae_m",
        "shadow_realized_stop_distance_mae_m",
    ]:
        updated[metric_name] = realized_summary.get(metric_name)
    updated["shadow_realized_outcome_comparison_mode"] = realized_summary.get("comparison_mode")
    updated["shadow_realized_horizon_s"] = realized_summary.get("horizon_s")
    updated["warnings"] = list(dict.fromkeys([
        *(updated.get("warnings") or []),
        *(realized_summary.get("warnings") or []),
    ]))
    updated["proposal_direct_metrics"] = sorted(
        metric_name
        for metric_name, support in support_map.items()
        if str(support.get("support_level", "")).startswith("proposal_direct")
    )
    updated["soft_guardrail_candidate_metrics"] = sorted(
        metric_name
        for metric_name, support in support_map.items()
        if support.get("recommended_gate_usage") == "soft_guardrail"
    )
    updated["diagnostic_only_metrics"] = sorted(
        metric_name
        for metric_name, support in support_map.items()
        if support.get("recommended_gate_usage") == "diagnostic_only"
    )
    return updated


def build_shadow_baseline_comparison_payload(
    *,
    case_id: str,
    shadow_summary: dict[str, Any],
    planner_quality_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = _shadow_baseline_metrics(planner_quality_summary)
    shadow_metrics = _shadow_summary_metrics(shadow_summary)
    comparison_metrics: dict[str, Any] = {}
    for metric in shadow_metrics:
        row = _comparison_row(
            metric=metric,
            shadow_value=shadow_metrics.get(metric),
            baseline_value=baseline_metrics.get(metric),
        )
        row.update(_shadow_metric_support(shadow_summary, metric))
        comparison_metrics[metric] = row
    return {
        "schema_version": "stage6.shadow_baseline_comparison.v1",
        "generated_at_utc": _utc_now_iso(),
        "case_id": case_id,
        "comparison_metrics": comparison_metrics,
        "proposal_direct_metrics": sorted(
            metric
            for metric, row in comparison_metrics.items()
            if str(row.get("support_level", "")).startswith("proposal_direct")
        ),
        "soft_guardrail_candidate_metrics": sorted(
            metric
            for metric, row in comparison_metrics.items()
            if row.get("recommended_gate_usage") == "soft_guardrail"
        ),
        "better_metrics": sorted(
            metric for metric, row in comparison_metrics.items() if row.get("classification") == "better"
        ),
        "worse_metrics": sorted(
            metric for metric, row in comparison_metrics.items() if row.get("classification") == "worse"
        ),
        "equal_metrics": sorted(
            metric for metric, row in comparison_metrics.items() if row.get("classification") == "equal"
        ),
        "unavailable_metrics": sorted(
            metric for metric, row in comparison_metrics.items() if row.get("classification") == "unavailable"
        ),
        "notes": [
            (
                "comparison rows now distinguish proposal-direct metrics from counterfactual/unavailable ones; "
                "all shadow metrics remain proposal-only because baseline is still the only execution authority"
            ),
        ],
    }


def write_mpc_shadow_artifacts(
    stage3c_dir: str | Path,
    *,
    case_id: str | None = None,
    proposals: list[dict[str, Any]],
    disagreement_events: list[dict[str, Any]],
    shadow_summary: dict[str, Any],
    realized_outcome_trace: list[dict[str, Any]] | None = None,
    realized_outcome_summary: dict[str, Any] | None = None,
    planner_quality_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage3c_dir = Path(stage3c_dir)
    case_id = case_id or _case_id_from_stage3c_dir(stage3c_dir)
    planner_quality_summary = dict(planner_quality_summary or _load_or_build_planner_quality(stage3c_dir))
    summary_payload = dict(shadow_summary or {})
    summary_payload["generated_at_utc"] = _utc_now_iso()
    comparison_payload = build_shadow_baseline_comparison_payload(
        case_id=case_id,
        shadow_summary=summary_payload,
        planner_quality_summary=planner_quality_summary,
    )
    dump_jsonl(stage3c_dir / "mpc_shadow_proposal.jsonl", proposals)
    dump_json(stage3c_dir / "mpc_shadow_summary.json", summary_payload)
    dump_json(stage3c_dir / "shadow_baseline_comparison.json", comparison_payload)
    dump_jsonl(stage3c_dir / "shadow_disagreement_events.jsonl", disagreement_events)
    if realized_outcome_trace is not None:
        dump_jsonl(stage3c_dir / "shadow_realized_outcome_trace.jsonl", realized_outcome_trace)
    if realized_outcome_summary is not None:
        dump_json(stage3c_dir / "shadow_realized_outcome_summary.json", realized_outcome_summary)
    return {
        "proposal_rows": proposals,
        "summary": summary_payload,
        "comparison": comparison_payload,
        "disagreement_events": disagreement_events,
        "realized_outcome_trace": list(realized_outcome_trace or []),
        "realized_outcome_summary": dict(realized_outcome_summary or {}),
    }


def generate_mpc_shadow_artifacts(
    stage3c_dir: str | Path,
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    stage3c_dir = Path(stage3c_dir)
    case_id = case_id or _case_id_from_stage3c_dir(stage3c_dir)

    execution_rows = load_jsonl(stage3c_dir / "execution_timeline.jsonl")
    control_trace = load_jsonl(stage3c_dir / "control_trace.jsonl")
    latency_rows = load_jsonl(stage3c_dir / "planner_control_latency.jsonl")
    stop_target_payload = load_json(stage3c_dir / "stop_target.json", default={}) or {}
    planner_quality_summary = _load_or_build_planner_quality(stage3c_dir)

    stop_events_by_frame, stop_event_frames = _stop_event_indexes(stop_target_payload)
    control_by_frame = _control_row_by_frame(control_trace)
    latency_by_frame = _latency_row_by_frame(latency_rows)
    rollout = solve_kinematic_shadow_rollout(
        case_id=case_id,
        execution_rows=execution_rows,
        control_by_frame=control_by_frame,
        latency_by_frame=latency_by_frame,
        stop_events_by_frame=stop_events_by_frame,
        stop_event_frames=stop_event_frames,
    )
    proposals = list(rollout.get("proposals") or [])
    disagreement_events = list(rollout.get("disagreement_events") or [])
    shadow_summary = dict(rollout.get("shadow_summary") or {})
    realized_outcome_payload = build_shadow_realized_outcome_payload(
        case_id=case_id,
        proposals=proposals,
        execution_rows=execution_rows,
    )
    realized_outcome_summary = dict(realized_outcome_payload.get("summary") or {})
    shadow_summary = _merge_realized_outcome_into_shadow_summary(shadow_summary, realized_outcome_summary)
    return write_mpc_shadow_artifacts(
        stage3c_dir,
        case_id=case_id,
        proposals=proposals,
        disagreement_events=disagreement_events,
        shadow_summary=shadow_summary,
        realized_outcome_trace=list(realized_outcome_payload.get("trace_rows") or []),
        realized_outcome_summary=realized_outcome_summary,
        planner_quality_summary=planner_quality_summary,
    )
