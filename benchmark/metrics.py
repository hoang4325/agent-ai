from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from stage3c_coverage.evaluation_stage3c_coverage import build_stage3c_run_metrics

from .io import load_json, load_jsonl, resolve_repo_path
from .semantic_artifacts import ensure_semantic_artifacts


LANE_CHANGE_LEFT_BEHAVIORS = {"prepare_lane_change_left", "commit_lane_change_left"}
LANE_CHANGE_RIGHT_BEHAVIORS = {"prepare_lane_change_right", "commit_lane_change_right"}
DETECTED_BINDING_STATUSES = {"bound", "ambiguous", "stale"}
NON_CONFLICT_ARBITRATION_ACTIONS = {"continue"}


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _load_stage3c_metrics(case_spec: dict[str, Any], repo_root: Path) -> dict[str, Any] | None:
    stage3c_dir = case_spec.get("source", {}).get("stage3c_dir")
    resolved = resolve_repo_path(repo_root, stage3c_dir)
    if resolved is None or not resolved.exists():
        return None
    return build_stage3c_run_metrics(resolved)


def _behavior_counts(context: dict[str, Any]) -> dict[str, int]:
    stage3b_summary = context.get("stage3b_summary") or {}
    counts = stage3b_summary.get("behavior_counts")
    if counts:
        return {str(key): int(value) for key, value in counts.items()}
    stage3c_metrics = context.get("stage3c_metrics") or {}
    counts = stage3c_metrics.get("requested_behavior_counts") or {}
    return {str(key): int(value) for key, value in counts.items()}


def _load_context(case_spec: dict[str, Any], repo_root: Path, generated_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    artifacts = case_spec.get("artifacts", {})
    resolved = {
        name: resolve_repo_path(repo_root, path)
        for name, path in artifacts.items()
    }
    semantic_paths, semantic_metadata = ensure_semantic_artifacts(
        case_spec=case_spec,
        repo_root=repo_root,
        generated_root=generated_root,
    )
    resolved.update(semantic_paths)
    availability = {
        artifact_name: {
            "available": bool(path and Path(path).exists()),
            "path": (str(path) if path else None),
            "source": "case_spec",
            "notes": [],
        }
        for artifact_name, path in resolved.items()
    }
    for name, metadata in semantic_metadata.items():
        availability[name] = {
            "available": metadata.available,
            "path": metadata.path,
            "source": metadata.source,
            "notes": metadata.notes,
        }

    context = {
        "artifacts": resolved,
        "stage1_summary": load_json(resolved.get("stage1_session_summary")),
        "stage2_summary": load_json(resolved.get("stage2_evaluation_summary")),
        "stage2_world_state_sample": load_json(resolved.get("stage2_world_state_sample")),
        "stage2_tracked_objects_sample": load_json(resolved.get("stage2_tracked_objects_sample")),
        "stage3a_summary": load_json(resolved.get("stage3a_evaluation_summary")),
        "stage3a_lane_world_state_sample": load_json(resolved.get("stage3a_lane_world_state_sample")),
        "stage3b_summary": load_json(resolved.get("stage3b_evaluation_summary")),
        "stage3b_timeline": load_jsonl(resolved.get("stage3b_behavior_timeline")),
        "stage3b_behavior_request_sample": load_json(resolved.get("stage3b_behavior_request_sample")),
        "stage4_summary": load_json(resolved.get("stage4_online_run_summary")),
        "stage4_stack_state": load_json(resolved.get("stage4_online_stack_state")),
        "stage4_tick_records": load_jsonl(resolved.get("stage4_online_tick_record")),
        "stage3c_metrics": _load_stage3c_metrics(case_spec, repo_root),
        "stage3c_stop_scene_alignment": load_json(resolved.get("stage3c_stop_scene_alignment")),
        "stage3c_mpc_shadow_proposal": load_jsonl(resolved.get("stage3c_mpc_shadow_proposal")),
        "stage3c_mpc_shadow_summary": load_json(resolved.get("stage3c_mpc_shadow_summary")),
        "stage3c_shadow_baseline_comparison": load_json(resolved.get("stage3c_shadow_baseline_comparison")),
        "stage3c_shadow_disagreement_events": load_jsonl(resolved.get("stage3c_shadow_disagreement_events")),
        "stage3c_shadow_realized_outcome_trace": load_jsonl(resolved.get("stage3c_shadow_realized_outcome_trace")),
        "stage3c_shadow_realized_outcome_summary": load_json(resolved.get("stage3c_shadow_realized_outcome_summary")),
        "ground_truth_actor_roles": load_json(resolved.get("ground_truth_actor_roles")),
        "critical_actor_binding": load_jsonl(resolved.get("critical_actor_binding")),
        "behavior_arbitration_events": load_jsonl(resolved.get("behavior_arbitration_events")),
        "route_session_binding": load_json(resolved.get("route_session_binding")),
    }
    return context, availability


def _route_option_match(case_spec: dict[str, Any], context: dict[str, Any]) -> float | None:
    expected = case_spec.get("route_option")
    if not expected:
        return None
    route_counts = (context.get("stage3b_summary") or {}).get("route_option_counts") or {}
    return 1.0 if route_counts.get(expected, 0) > 0 else 0.0


def _expected_behavior_presence(case_spec: dict[str, Any], context: dict[str, Any]) -> int | None:
    expected_behaviors = case_spec.get("expected_behaviors") or []
    if not expected_behaviors:
        return None
    behavior_counts = _behavior_counts(context)
    return int(sum(int(behavior_counts.get(name, 0)) for name in expected_behaviors))


def _forbidden_behavior_count(case_spec: dict[str, Any], context: dict[str, Any]) -> int | None:
    forbidden_behaviors = case_spec.get("forbidden_behaviors") or []
    if not forbidden_behaviors:
        return 0
    behavior_counts = _behavior_counts(context)
    return int(sum(int(behavior_counts.get(name, 0)) for name in forbidden_behaviors))


def _unsafe_lane_change_intent_count(context: dict[str, Any]) -> int | None:
    records = context.get("stage3b_timeline") or []
    if not records:
        return None
    count = 0
    for record in records:
        request = record.get("behavior_request_v2") or {}
        behavior = str(request.get("requested_behavior"))
        permission = request.get("lane_change_permission") or {}
        if behavior in LANE_CHANGE_LEFT_BEHAVIORS and not bool(permission.get("left")):
            count += 1
        if behavior in LANE_CHANGE_RIGHT_BEHAVIORS and not bool(permission.get("right")):
            count += 1
    return int(count)


def _front_hazard_signal_present(context: dict[str, Any]) -> float | None:
    world_state = context.get("stage2_world_state_sample") or {}
    risk_summary = world_state.get("risk_summary") or {}
    if not risk_summary:
        return None
    front_hazard_track_id = risk_summary.get("front_hazard_track_id")
    nearest_front_vehicle_distance_m = risk_summary.get("nearest_front_vehicle_distance_m")
    return 1.0 if (front_hazard_track_id is not None or nearest_front_vehicle_distance_m is not None) else 0.0


def _junction_keep_route_presence(context: dict[str, Any]) -> int | None:
    stage3b_summary = context.get("stage3b_summary") or {}
    if not stage3b_summary:
        return None
    return int(stage3b_summary.get("keep_route_through_junction_frames", 0))


def _route_conditioned_override_rate(context: dict[str, Any]) -> float | None:
    stage3b_summary = context.get("stage3b_summary") or {}
    value = stage3b_summary.get("route_conditioned_override_rate")
    return None if value is None else float(value)


def _lane_change_success_count(context: dict[str, Any]) -> int | None:
    stage3c_metrics = context.get("stage3c_metrics") or {}
    if not stage3c_metrics:
        return None
    return int(stage3c_metrics.get("lane_change_successes", 0))


def _lane_change_success_rate(context: dict[str, Any]) -> float | None:
    stage3c_metrics = context.get("stage3c_metrics") or {}
    if not stage3c_metrics:
        return None
    lane_change_attempts = float(stage3c_metrics.get("lane_change_attempts", 0))
    if lane_change_attempts <= 0:
        requested = stage3c_metrics.get("requested_behavior_counts") or {}
        requested_attempts = (
            float(requested.get("commit_lane_change_left", 0))
            + float(requested.get("commit_lane_change_right", 0))
        )
        if requested_attempts > 0:
            return 0.0
    return _safe_div(
        float(stage3c_metrics.get("lane_change_successes", 0)),
        lane_change_attempts,
    )


def _stage3c_planner_metric(context: dict[str, Any], metric_name: str) -> float | None:
    stage3c_metrics = context.get("stage3c_metrics") or {}
    if not stage3c_metrics:
        return None
    value = stage3c_metrics.get(metric_name)
    if value is None:
        value = (stage3c_metrics.get("planner_quality_summary") or {}).get(metric_name)
    return None if value is None else float(value)


def _stop_success_rate(context: dict[str, Any]) -> float | None:
    stage3c_metrics = context.get("stage3c_metrics") or {}
    if not stage3c_metrics:
        return None
    return _safe_div(
        float(stage3c_metrics.get("stop_successes", 0)),
        float(stage3c_metrics.get("stop_attempts", 0)),
    )


def _stop_overshoot_rate(context: dict[str, Any]) -> float | None:
    stage3c_metrics = context.get("stage3c_metrics") or {}
    if not stage3c_metrics:
        return None
    return _safe_div(
        float(stage3c_metrics.get("stop_overshoots", 0)),
        float(stage3c_metrics.get("stop_attempts", 0)),
    )


def _abort_rate(context: dict[str, Any]) -> float | None:
    stage3c_metrics = context.get("stage3c_metrics") or {}
    if not stage3c_metrics:
        return None
    numerator = float(stage3c_metrics.get("lane_change_failures", 0)) + float(stage3c_metrics.get("stop_aborts", 0))
    denominator = float(stage3c_metrics.get("lane_change_attempts", 0)) + float(stage3c_metrics.get("stop_attempts", 0))
    return _safe_div(numerator, denominator)


def _legacy_behavior_execution_mismatch_rate(context: dict[str, Any]) -> float | None:
    stage3c_metrics = context.get("stage3c_metrics") or {}
    summary = stage3c_metrics.get("summary") or {}
    value = summary.get("behavior_override_rate")
    return None if value is None else float(value)


def _behavior_execution_mismatch_rate(context: dict[str, Any]) -> float | None:
    arbitration_events = context.get("behavior_arbitration_events") or []
    stage3c_metrics = context.get("stage3c_metrics") or {}
    num_frames = float((stage3c_metrics.get("summary") or {}).get("num_frames", 0))
    if arbitration_events and num_frames > 0:
        mismatches = 0
        for event in arbitration_events:
            requested = event.get("requested_behavior")
            executing = event.get("executing_behavior", event.get("current_execution_behavior"))
            classification = str(event.get("classification"))
            action = str(event.get("arbitration_action"))
            if requested != executing:
                if classification == "diagnostic":
                    continue
                if action in NON_CONFLICT_ARBITRATION_ACTIONS:
                    continue
                mismatches += 1
        return _safe_div(float(mismatches), num_frames)
    return _legacy_behavior_execution_mismatch_rate(context)


def _mean_lane_change_time_s(context: dict[str, Any]) -> float | None:
    stage3c_metrics = context.get("stage3c_metrics") or {}
    value = stage3c_metrics.get("mean_lane_change_time_s")
    return None if value is None else float(value)


def _lane_change_completion_time_s(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "lane_change_completion_time_s")


def _lane_change_completion_rate(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "lane_change_completion_rate")


def _lane_change_abort_rate(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "lane_change_abort_rate")


def _lane_change_completion_validity(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "lane_change_completion_validity")


def _stop_final_error_m(context: dict[str, Any]) -> float | None:
    alignment = context.get("stage3c_stop_scene_alignment") or {}
    if alignment and not bool(alignment.get("alignment_ready")):
        return None
    return _stage3c_planner_metric(context, "stop_final_error_m")


def _stop_overshoot_distance_m(context: dict[str, Any]) -> float | None:
    alignment = context.get("stage3c_stop_scene_alignment") or {}
    if alignment and not bool(alignment.get("alignment_ready")):
        return None
    return _stage3c_planner_metric(context, "stop_overshoot_distance_m")


def _stop_hold_stability(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "stop_hold_stability")


def _stop_target_defined_rate(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "stop_target_defined_rate")


def _heading_settle_time_s(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "heading_settle_time_s")


def _lateral_oscillation_rate(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "lateral_oscillation_rate")


def _trajectory_smoothness_proxy(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "trajectory_smoothness_proxy")


def _control_jerk_proxy(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "control_jerk_proxy")


def _planner_execution_latency_ms(context: dict[str, Any]) -> float | None:
    planner_metric = _stage3c_planner_metric(context, "planner_execution_latency_ms")
    if planner_metric is not None:
        return planner_metric
    stage4_summary = context.get("stage4_summary") or {}
    value = stage4_summary.get("mean_total_loop_latency_ms")
    return None if value is None else float(value)


def _decision_to_execution_latency_ms(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "decision_to_execution_latency_ms")


def _over_budget_rate(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "over_budget_rate")


def _critical_binding_records_by_role(context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in context.get("critical_actor_binding") or []:
        grouped[str(record.get("role"))].append(record)
    return grouped


def _critical_actor_detected_rate(context: dict[str, Any]) -> float | None:
    ground_truth = context.get("ground_truth_actor_roles") or {}
    if ground_truth.get("pending_ground_truth_binding"):
        return None
    grouped = _critical_binding_records_by_role(context)
    eligible_records: list[dict[str, Any]] = []
    detected = 0
    for actor in ground_truth.get("actors") or []:
        if "critical_actor_detected_rate" not in (actor.get("evaluation_scope") or []):
            continue
        for record in grouped.get(str(actor.get("role")), []):
            eligible_records.append(record)
            if str(record.get("status")) in DETECTED_BINDING_STATUSES:
                detected += 1
    return _safe_div(float(detected), float(len(eligible_records)))


def _blocker_binding_accuracy(context: dict[str, Any]) -> float | None:
    ground_truth = context.get("ground_truth_actor_roles") or {}
    if ground_truth.get("pending_ground_truth_binding"):
        return None
    grouped = _critical_binding_records_by_role(context)
    eligible = 0
    score = 0.0
    for actor in ground_truth.get("actors") or []:
        if "blocker_binding_accuracy" not in (actor.get("evaluation_scope") or []):
            continue
        expected_lane = actor.get("expected_lane_relation")
        for record in grouped.get(str(actor.get("role")), []):
            eligible += 1
            status = str(record.get("status"))
            observed_lane = record.get("observed_lane_relation")
            if status == "bound" and observed_lane == expected_lane:
                score += 1.0
            elif status == "ambiguous" and observed_lane == expected_lane:
                score += 0.5
            elif status == "stale" and observed_lane == expected_lane:
                score += 0.5
    return _safe_div(score, float(eligible))


def _lane_assignment_consistency(context: dict[str, Any]) -> float | None:
    ground_truth = context.get("ground_truth_actor_roles") or {}
    if ground_truth.get("pending_ground_truth_binding"):
        return None
    grouped = _critical_binding_records_by_role(context)
    eligible = 0
    correct = 0
    for actor in ground_truth.get("actors") or []:
        if "lane_assignment_consistency" not in (actor.get("evaluation_scope") or []):
            continue
        expected_lane = actor.get("expected_lane_relation")
        for record in grouped.get(str(actor.get("role")), []):
            if str(record.get("status")) not in DETECTED_BINDING_STATUSES:
                continue
            eligible += 1
            if record.get("observed_lane_relation") == expected_lane:
                correct += 1
    return _safe_div(float(correct), float(eligible))


def _critical_actor_track_continuity_rate(context: dict[str, Any]) -> float | None:
    ground_truth = context.get("ground_truth_actor_roles") or {}
    if ground_truth.get("pending_ground_truth_binding"):
        return None
    grouped = _critical_binding_records_by_role(context)
    transitions = 0
    stable = 0
    for actor in ground_truth.get("actors") or []:
        if "critical_actor_track_continuity_rate" not in (actor.get("evaluation_scope") or []):
            continue
        previous_track_id = None
        for record in grouped.get(str(actor.get("role")), []):
            if str(record.get("status")) not in DETECTED_BINDING_STATUSES:
                continue
            current_track_id = record.get("bound_track_id")
            if previous_track_id is not None:
                transitions += 1
                if current_track_id == previous_track_id:
                    stable += 1
            previous_track_id = current_track_id
    return _safe_div(float(stable), float(transitions))


def _arbitration_conflict_count(case_spec: dict[str, Any], context: dict[str, Any]) -> int | None:
    arbitration_events = context.get("behavior_arbitration_events") or []
    if not arbitration_events:
        return None
    if not case_spec.get("expected_arbitration_actions") and not case_spec.get("forbidden_arbitration_actions"):
        return None
    conflicts = 0
    for event in arbitration_events:
        if str(event.get("classification")) != "unexpected":
            continue
        action = str(event.get("arbitration_action"))
        reason = str(event.get("arbitration_reason"))
        requested_behavior = str(event.get("requested_behavior"))
        notes = {str(note) for note in (event.get("notes") or [])}
        if action in NON_CONFLICT_ARBITRATION_ACTIONS:
            continue
        if (
            action == "cancel"
            and reason == "active_lane_change_aborted_for_stop"
            and requested_behavior == "stop_before_obstacle"
            and "superseded_by_stop_request" in notes
        ):
            continue
        conflicts += 1
    return int(conflicts)


def _stale_tick_rate(context: dict[str, Any]) -> float | None:
    stage4_summary = context.get("stage4_summary") or {}
    num_ticks = float(stage4_summary.get("num_ticks", 0))
    skipped_updates = float(stage4_summary.get("skipped_updates", 0))
    return _safe_div(skipped_updates, num_ticks)


def _mean_total_loop_latency_ms(context: dict[str, Any]) -> float | None:
    stage4_summary = context.get("stage4_summary") or {}
    value = stage4_summary.get("mean_total_loop_latency_ms")
    return None if value is None else float(value)


def _state_reset_count(context: dict[str, Any]) -> int | None:
    stage4_summary = context.get("stage4_summary") or {}
    if not stage4_summary:
        return None
    return int(stage4_summary.get("state_resets", 0))


def _route_context_consistency(case_spec: dict[str, Any], context: dict[str, Any]) -> float | None:
    route_session = context.get("route_session_binding") or {}
    route_binding = (route_session.get("route_binding")) or {}
    expected = (route_session.get("expected_binding")) or {}
    if not route_binding:
        return None
    expected_route_option = expected.get("expected_route_option") or case_spec.get("route_option")
    route_option_ok = route_binding.get("route_option") == expected_route_option
    preferred_expected = expected.get("expected_preferred_lane")
    preferred_ok = True if not preferred_expected else route_binding.get("preferred_lane") == preferred_expected
    conflicts_ok = not bool(route_binding.get("route_conflict_flags"))
    source_ok = bool(route_binding.get("source_of_route_binding"))
    return 1.0 if route_option_ok and preferred_ok and conflicts_ok and source_ok else 0.0


def _route_influence_correctness(case_spec: dict[str, Any], context: dict[str, Any]) -> float | None:
    timeline = context.get("stage3b_timeline") or []
    if not timeline:
        return None
    expected_behaviors = set(case_spec.get("expected_behaviors") or [])
    forbidden_behaviors = set(case_spec.get("forbidden_behaviors") or [])
    route_influenced = []
    correct = 0
    for record in timeline:
        comparison = record.get("comparison") or {}
        if not bool(comparison.get("route_influenced")):
            continue
        route_influenced.append(record)
        behavior = ((record.get("behavior_request_v2") or {}).get("requested_behavior"))
        if behavior in expected_behaviors and behavior not in forbidden_behaviors:
            correct += 1
    return _safe_div(float(correct), float(len(route_influenced)))


def _fallback_activation_rate(context: dict[str, Any]) -> float | None:
    stage3c_metric = _stage3c_planner_metric(context, "fallback_activation_rate")
    if stage3c_metric is not None:
        return stage3c_metric
    arbitration_events = context.get("behavior_arbitration_events") or []
    if not arbitration_events:
        return None
    emergency_count = sum(
        1 for e in arbitration_events
        if str(e.get("arbitration_action")) == "emergency_override"
    )
    return _safe_div(float(emergency_count), float(len(arbitration_events)))


def _fallback_reason_coverage(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "fallback_reason_coverage")


def _fallback_takeover_latency_ms(context: dict[str, Any]) -> float | None:
    return _stage3c_planner_metric(context, "fallback_takeover_latency_ms")


def _fallback_missing_when_required_count(context: dict[str, Any]) -> int | None:
    value = _stage3c_planner_metric(context, "fallback_missing_when_required_count")
    return None if value is None else int(value)


def _shadow_summary(context: dict[str, Any]) -> dict[str, Any]:
    return context.get("stage3c_mpc_shadow_summary") or {}


def _shadow_realized_summary(context: dict[str, Any]) -> dict[str, Any]:
    return context.get("stage3c_shadow_realized_outcome_summary") or {}


def _shadow_metric_support(context: dict[str, Any], metric_name: str) -> dict[str, Any]:
    summary = _shadow_summary(context)
    support = (summary.get("shadow_metric_support") or {}).get(metric_name) or {}
    return {
        "support_level": support.get("support_level"),
        "authority": support.get("authority"),
        "recommended_gate_usage": support.get("recommended_gate_usage"),
    }


def _shadow_metric(context: dict[str, Any], metric_name: str) -> float | None:
    summary = _shadow_summary(context)
    value = summary.get(metric_name)
    return None if value is None else float(value)


def _shadow_realized_metric(context: dict[str, Any], metric_name: str) -> float | None:
    summary = _shadow_realized_summary(context)
    value = summary.get(metric_name)
    return None if value is None else float(value)


def _shadow_output_contract_completeness(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_output_contract_completeness")


def _shadow_feasibility_rate(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_feasibility_rate")


def _shadow_solver_timeout_rate(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_solver_timeout_rate")


def _shadow_planner_execution_latency_ms(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_planner_execution_latency_ms")


def _shadow_baseline_disagreement_rate(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_baseline_disagreement_rate")


def _shadow_fallback_recommendation_rate(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_fallback_recommendation_rate")


def _shadow_stop_final_error_m(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_stop_final_error_m")


def _shadow_stop_overshoot_distance_m(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_stop_overshoot_distance_m")


def _shadow_lane_change_completion_time_s(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_lane_change_completion_time_s")


def _shadow_heading_settle_time_s(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_heading_settle_time_s")


def _shadow_lateral_oscillation_rate(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_lateral_oscillation_rate")


def _shadow_trajectory_smoothness_proxy(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_trajectory_smoothness_proxy")


def _shadow_control_jerk_proxy(context: dict[str, Any]) -> float | None:
    return _shadow_metric(context, "shadow_control_jerk_proxy")


def _shadow_realized_support_rate(context: dict[str, Any]) -> float | None:
    return _shadow_realized_metric(context, "shadow_realized_support_rate")


def _shadow_realized_behavior_match_rate(context: dict[str, Any]) -> float | None:
    return _shadow_realized_metric(context, "shadow_realized_behavior_match_rate")


def _shadow_realized_speed_mae_mps(context: dict[str, Any]) -> float | None:
    return _shadow_realized_metric(context, "shadow_realized_speed_mae_mps")


def _shadow_realized_progress_mae_m(context: dict[str, Any]) -> float | None:
    return _shadow_realized_metric(context, "shadow_realized_progress_mae_m")


def _shadow_realized_stop_distance_mae_m(context: dict[str, Any]) -> float | None:
    return _shadow_realized_metric(context, "shadow_realized_stop_distance_mae_m")


def _unexpected_override_count(context: dict[str, Any]) -> int | None:
    arbitration_events = context.get("behavior_arbitration_events") or []
    if not arbitration_events:
        return None
    return int(sum(
        1 for e in arbitration_events
        if str(e.get("arbitration_action")) == "emergency_override"
        and str(e.get("classification")) == "unexpected"
    ))


def _completion_criterion_validity(context: dict[str, Any]) -> float | None:
    stage3c_metrics = context.get("stage3c_metrics") or {}
    if not stage3c_metrics:
        return None
    # Count all completions (stop + lane change)
    stop_successes = float(stage3c_metrics.get("stop_successes", 0))
    stop_failures = float(stage3c_metrics.get("stop_aborts", 0))
    lc_successes = float(stage3c_metrics.get("lane_change_successes", 0))
    lc_failures = float(stage3c_metrics.get("lane_change_failures", 0))
    total_completions = stop_successes + stop_failures + lc_successes + lc_failures
    if total_completions <= 0:
        return None
    # Valid completions = successes (where completion reason is justified)
    valid = stop_successes + lc_successes
    return _safe_div(valid, total_completions)


def evaluate_case_metrics(
    *,
    case_spec: dict[str, Any],
    metric_registry: dict[str, Any],
    repo_root: Path,
    generated_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    context, availability = _load_context(case_spec, repo_root, generated_root)
    warnings: list[str] = []

    mandatory = case_spec.get("mandatory_artifacts") or []
    total_required = len(mandatory)
    present_required = sum(1 for artifact_name in mandatory if (availability.get(artifact_name) or {}).get("available"))
    artifact_completeness = None if total_required == 0 else float(present_required / total_required)

    registry_items = metric_registry.get("metrics", {})
    calculators = {
        "artifact_completeness": lambda: artifact_completeness,
        "route_option_match": lambda: _route_option_match(case_spec, context),
        "expected_behavior_presence": lambda: _expected_behavior_presence(case_spec, context),
        "forbidden_behavior_count": lambda: _forbidden_behavior_count(case_spec, context),
        "unsafe_lane_change_intent_count": lambda: _unsafe_lane_change_intent_count(context),
        "route_conditioned_override_rate": lambda: _route_conditioned_override_rate(context),
        "junction_keep_route_presence": lambda: _junction_keep_route_presence(context),
        "front_hazard_signal_present": lambda: _front_hazard_signal_present(context),
        "blocker_binding_accuracy": lambda: _blocker_binding_accuracy(context),
        "critical_actor_detected_rate": lambda: _critical_actor_detected_rate(context),
        "critical_actor_track_continuity_rate": lambda: _critical_actor_track_continuity_rate(context),
        "lane_assignment_consistency": lambda: _lane_assignment_consistency(context),
        "lane_change_success_count": lambda: _lane_change_success_count(context),
        "lane_change_success_rate": lambda: _lane_change_success_rate(context),
        "stop_success_rate": lambda: _stop_success_rate(context),
        "stop_overshoot_rate": lambda: _stop_overshoot_rate(context),
        "abort_rate": lambda: _abort_rate(context),
        "behavior_execution_mismatch_rate": lambda: _behavior_execution_mismatch_rate(context),
        "mean_lane_change_time_s": lambda: _mean_lane_change_time_s(context),
        "lane_change_completion_time_s": lambda: _lane_change_completion_time_s(context),
        "lane_change_completion_rate": lambda: _lane_change_completion_rate(context),
        "lane_change_abort_rate": lambda: _lane_change_abort_rate(context),
        "lane_change_completion_validity": lambda: _lane_change_completion_validity(context),
        "stop_final_error_m": lambda: _stop_final_error_m(context),
        "stop_overshoot_distance_m": lambda: _stop_overshoot_distance_m(context),
        "stop_hold_stability": lambda: _stop_hold_stability(context),
        "stop_target_defined_rate": lambda: _stop_target_defined_rate(context),
        "heading_settle_time_s": lambda: _heading_settle_time_s(context),
        "lateral_oscillation_rate": lambda: _lateral_oscillation_rate(context),
        "trajectory_smoothness_proxy": lambda: _trajectory_smoothness_proxy(context),
        "control_jerk_proxy": lambda: _control_jerk_proxy(context),
        "planner_execution_latency_ms": lambda: _planner_execution_latency_ms(context),
        "decision_to_execution_latency_ms": lambda: _decision_to_execution_latency_ms(context),
        "over_budget_rate": lambda: _over_budget_rate(context),
        "arbitration_conflict_count": lambda: _arbitration_conflict_count(case_spec, context),
        "stale_tick_rate": lambda: _stale_tick_rate(context),
        "mean_total_loop_latency_ms": lambda: _mean_total_loop_latency_ms(context),
        "state_reset_count": lambda: _state_reset_count(context),
        "route_context_consistency": lambda: _route_context_consistency(case_spec, context),
        "route_influence_correctness": lambda: _route_influence_correctness(case_spec, context),
        "fallback_activation_rate": lambda: _fallback_activation_rate(context),
        "fallback_reason_coverage": lambda: _fallback_reason_coverage(context),
        "fallback_takeover_latency_ms": lambda: _fallback_takeover_latency_ms(context),
        "fallback_missing_when_required_count": lambda: _fallback_missing_when_required_count(context),
        "shadow_output_contract_completeness": lambda: _shadow_output_contract_completeness(context),
        "shadow_feasibility_rate": lambda: _shadow_feasibility_rate(context),
        "shadow_solver_timeout_rate": lambda: _shadow_solver_timeout_rate(context),
        "shadow_planner_execution_latency_ms": lambda: _shadow_planner_execution_latency_ms(context),
        "shadow_baseline_disagreement_rate": lambda: _shadow_baseline_disagreement_rate(context),
        "shadow_fallback_recommendation_rate": lambda: _shadow_fallback_recommendation_rate(context),
        "shadow_stop_final_error_m": lambda: _shadow_stop_final_error_m(context),
        "shadow_stop_overshoot_distance_m": lambda: _shadow_stop_overshoot_distance_m(context),
        "shadow_lane_change_completion_time_s": lambda: _shadow_lane_change_completion_time_s(context),
        "shadow_heading_settle_time_s": lambda: _shadow_heading_settle_time_s(context),
        "shadow_lateral_oscillation_rate": lambda: _shadow_lateral_oscillation_rate(context),
        "shadow_trajectory_smoothness_proxy": lambda: _shadow_trajectory_smoothness_proxy(context),
        "shadow_control_jerk_proxy": lambda: _shadow_control_jerk_proxy(context),
        "shadow_realized_support_rate": lambda: _shadow_realized_support_rate(context),
        "shadow_realized_behavior_match_rate": lambda: _shadow_realized_behavior_match_rate(context),
        "shadow_realized_speed_mae_mps": lambda: _shadow_realized_speed_mae_mps(context),
        "shadow_realized_progress_mae_m": lambda: _shadow_realized_progress_mae_m(context),
        "shadow_realized_stop_distance_mae_m": lambda: _shadow_realized_stop_distance_mae_m(context),
        "unexpected_override_count": lambda: _unexpected_override_count(context),
        "completion_criterion_validity": lambda: _completion_criterion_validity(context),
    }

    results: dict[str, Any] = {}
    for metric_name, metadata in registry_items.items():
        required_artifacts = list(metadata.get("required_artifacts") or [])
        missing_artifacts = [
            artifact_name
            for artifact_name in required_artifacts
            if not (availability.get(artifact_name) or {}).get("available")
        ]
        evaluation_state = "ok"
        degraded_reason = None

        if missing_artifacts:
            fallback_behavior = metadata.get("fallback_behavior", "skip_metric")
            if fallback_behavior == "skip_metric":
                value = None
                evaluation_state = "metric_skipped"
                degraded_reason = f"missing_required_artifacts:{','.join(missing_artifacts)}"
            else:
                value = calculators.get(metric_name, lambda: None)()
                evaluation_state = "metric_degraded"
                degraded_reason = f"fallback_behavior={fallback_behavior};missing={','.join(missing_artifacts)}"
        else:
            value = calculators.get(metric_name, lambda: None)()
            if value is None and required_artifacts:
                evaluation_state = "metric_skipped"
                degraded_reason = "required_artifacts_present_but_metric_unavailable"

        if degraded_reason:
            warnings.append(f"{metric_name}:{degraded_reason}")

        results[metric_name] = {
            "value": value,
            "available": value is not None,
            "maturity": metadata.get("maturity"),
            "gate_usage": metadata.get("gate_usage"),
            "stage_layer": metadata.get("stage_layer"),
            "required_artifacts": required_artifacts,
            "missing_artifacts": missing_artifacts,
            "evaluation_state": evaluation_state,
            "degraded_reason": degraded_reason,
        }
        if metric_name.startswith("shadow_"):
            results[metric_name].update(_shadow_metric_support(context, metric_name))
    return results, availability, context.get("artifacts"), warnings
