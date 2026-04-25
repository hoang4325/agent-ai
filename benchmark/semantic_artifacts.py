from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import dump_json, dump_jsonl, load_json, load_jsonl, resolve_repo_path


@dataclass
class SemanticArtifact:
    name: str
    path: str | None
    available: bool
    source: str
    notes: list[str]


# ── Stage 5A constants ────────────────────────────────────────────────────────
MAX_STALE_HOLD_FRAMES = 5  # Max frames a stale binding can be held before forced missing


ROLE_HINT_TEMPLATES: dict[str, dict[str, Any]] = {
    "blocker": {
        "expected_lane_relation": "current_lane",
        "expected_behavioral_effect": "slow_down_or_stop",
        "binding_hints": {
            "class_groups": ["vehicle"],
            "allowed_lane_relations": ["current_lane"],
            "prefer_flags": ["is_blocking_current_lane", "is_front_in_current_lane"],
            "risk_hint": "front_hazard_track_id",
            "require_same_direction": True,
        },
    },
    "adjacent_left": {
        "expected_lane_relation": "left_lane",
        "expected_behavioral_effect": "lane_change_left_context",
        "binding_hints": {
            "class_groups": ["vehicle"],
            "allowed_lane_relations": ["left_lane"],
            "prefer_flags": ["same_direction_as_ego_lane"],
            "require_same_direction": True,
        },
    },
    "adjacent_right": {
        "expected_lane_relation": "right_lane",
        "expected_behavioral_effect": "lane_change_right_context",
        "binding_hints": {
            "class_groups": ["vehicle"],
            "allowed_lane_relations": ["right_lane"],
            "prefer_flags": ["same_direction_as_ego_lane"],
            "require_same_direction": True,
        },
    },
    "cross_traffic": {
        "expected_lane_relation": "cross_traffic",
        "expected_behavioral_effect": "yield_or_stop",
        "binding_hints": {
            "class_groups": ["vehicle"],
            "allowed_lane_relations": ["cross_lane", "opposite_lane"],
            "prefer_flags": ["is_front_in_route_lane"],
        },
    },
    "pedestrian": {
        "expected_lane_relation": "near_route",
        "expected_behavioral_effect": "slow_down_or_stop",
        "binding_hints": {
            "class_groups": ["pedestrian"],
            "prefer_flags": ["is_front_in_current_lane"],
        },
    },
    "lead_vehicle": {
        "expected_lane_relation": "current_lane",
        "expected_behavioral_effect": "follow_or_slow_down",
        "binding_hints": {
            "class_groups": ["vehicle"],
            "allowed_lane_relations": ["current_lane"],
            "prefer_flags": ["is_front_in_current_lane"],
            "require_same_direction": True,
        },
    },
    "route_critical": {
        "expected_lane_relation": "route_lane",
        "expected_behavioral_effect": "route_conditioned_arbitration",
        "binding_hints": {
            "class_groups": ["vehicle"],
            "allowed_route_relations": ["route_lane", "upcoming_route_lane"],
            "prefer_flags": ["is_in_route_lane", "is_route_blocker_candidate"],
        },
    },
}


def _canonical_role_key(role: str) -> str:
    role_l = role.lower()
    if "blocker" in role_l:
        return "blocker"
    if "adjacent" in role_l and "left" in role_l:
        return "adjacent_left"
    if "adjacent" in role_l and "right" in role_l:
        return "adjacent_right"
    if "cross" in role_l:
        return "cross_traffic"
    if "pedestrian" in role_l:
        return "pedestrian"
    if "lead" in role_l or "follow_target" in role_l:
        return "lead_vehicle"
    if "route" in role_l:
        return "route_critical"
    return "lead_vehicle"


def _derive_ground_truth_actor_roles(*, scenario_id: str, case_spec: dict[str, Any]) -> dict[str, Any]:
    actor_roles = list(case_spec.get("critical_actors") or [])
    actors: list[dict[str, Any]] = []
    pending = bool(case_spec.get("pending_ground_truth_binding", True))

    for role in actor_roles:
        template = ROLE_HINT_TEMPLATES.get(_canonical_role_key(str(role)), ROLE_HINT_TEMPLATES["lead_vehicle"])
        actors.append(
            {
                "role": str(role),
                "entity_type": "actor",
                "actor_source": "track_binding_hint",
                "expected_actor_ref": f"role:{role}",
                "expected_lane_relation": template.get("expected_lane_relation"),
                "expected_behavioral_effect": template.get("expected_behavioral_effect"),
                "binding_hints": template.get("binding_hints") or {},
                "evaluation_scope": [
                    "critical_actor_detected_rate",
                    "blocker_binding_accuracy",
                    "lane_assignment_consistency",
                    "critical_actor_track_continuity_rate",
                ],
                "notes": "auto_generated_from_case_spec",
            }
        )

    return {
        "schema_version": "system_benchmark_semantics_v1_1",
        "scenario_id": scenario_id,
        "pending_ground_truth_binding": pending,
        "determinism_level": case_spec.get("determinism_level", "unknown"),
        "case_mode": case_spec.get("mode"),
        "actors": actors,
        "route_binding_expectations": {
            "route_option": case_spec.get("route_option"),
            "preferred_lane": ((case_spec.get("route_binding_expectations") or {}).get("preferred_lane"))
            or _infer_expected_preferred_lane(case_spec.get("route_option")),
        },
    }


def _resolve_ground_truth_path(repo_root: Path, artifacts: dict[str, Any], scenario_id: str) -> Path | None:
    explicit = resolve_repo_path(repo_root, artifacts.get("ground_truth_actor_roles"))
    if explicit and explicit.exists():
        return explicit
    reference = repo_root / "benchmark" / "reference_artifacts" / scenario_id / "ground_truth_actor_roles.json"
    if reference.exists():
        return reference
    return explicit


def _infer_expected_preferred_lane(route_option: str | None) -> str | None:
    if route_option == "keep_right":
        return "right"
    if route_option == "keep_left":
        return "left"
    if route_option in {"straight", "keep_lane"}:
        return "current"
    return None


def _merge_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    merged: dict[Any, dict[str, Any]] = {}
    lane_objects = record.get("lane_relative_objects") or []
    route_objects = ((record.get("route_conditioned_scene") or {}).get("route_relative_objects")) or []

    for obj in lane_objects:
        track_id = obj.get("track_id")
        merged[track_id] = {
            "track_id": track_id,
            "class_name": obj.get("class_name"),
            "class_group": obj.get("class_group"),
            "lane_relation": obj.get("lane_relation"),
            "route_relation": None,
            "lane_tag": obj.get("lane_tag"),
            "route_relevance": None,
            "longitudinal_m": obj.get("longitudinal_m"),
            "lateral_m": obj.get("lateral_m"),
            "flags": {
                "is_front_in_current_lane": bool(obj.get("is_front_in_current_lane")),
                "is_rear_in_current_lane": bool(obj.get("is_rear_in_current_lane")),
                "is_blocking_current_lane": bool(obj.get("is_blocking_current_lane")),
                "same_direction_as_ego_lane": bool(obj.get("same_direction_as_ego_lane")),
                "is_in_route_lane": False,
                "is_front_in_route_lane": False,
                "is_route_blocker_candidate": False,
            },
            "source_track": obj.get("source_track") or {},
        }

    for obj in route_objects:
        track_id = obj.get("track_id")
        existing = merged.setdefault(
            track_id,
            {
                "track_id": track_id,
                "class_name": obj.get("class_name"),
                "class_group": obj.get("class_group"),
                "lane_relation": None,
                "route_relation": obj.get("route_relation"),
                "lane_tag": None,
                "route_relevance": obj.get("route_relevance"),
                "longitudinal_m": obj.get("longitudinal_m"),
                "lateral_m": obj.get("lateral_m"),
                "flags": {
                    "is_front_in_current_lane": False,
                    "is_rear_in_current_lane": False,
                    "is_blocking_current_lane": False,
                    "same_direction_as_ego_lane": False,
                    "is_in_route_lane": bool(obj.get("is_in_route_lane")),
                    "is_front_in_route_lane": bool(obj.get("is_front_in_route_lane")),
                    "is_route_blocker_candidate": bool(obj.get("is_route_blocker_candidate")),
                },
                "source_track": (obj.get("source_object") or {}).get("source_track") or {},
            },
        )
        existing["route_relation"] = obj.get("route_relation")
        existing["route_relevance"] = obj.get("route_relevance")
        existing["flags"]["is_in_route_lane"] = bool(obj.get("is_in_route_lane"))
        existing["flags"]["is_front_in_route_lane"] = bool(obj.get("is_front_in_route_lane"))
        existing["flags"]["is_route_blocker_candidate"] = bool(obj.get("is_route_blocker_candidate"))
        existing["longitudinal_m"] = obj.get("longitudinal_m", existing.get("longitudinal_m"))
        existing["lateral_m"] = obj.get("lateral_m", existing.get("lateral_m"))
    return list(merged.values())


def _score_candidate(role_spec: dict[str, Any], candidate: dict[str, Any], record: dict[str, Any]) -> tuple[float, list[str]]:
    hints = role_spec.get("binding_hints") or {}
    score = 0.0
    reasons: list[str] = []

    class_groups = hints.get("class_groups") or []
    if class_groups and candidate.get("class_group") not in class_groups:
        return -1.0, ["class_group_mismatch"]
    if class_groups:
        score += 0.1
        reasons.append("class_group_match")

    allowed_lane_relations = hints.get("allowed_lane_relations") or []
    if allowed_lane_relations and candidate.get("lane_relation") in allowed_lane_relations:
        score += 0.35
        reasons.append(f"lane_relation={candidate.get('lane_relation')}")

    allowed_route_relations = hints.get("allowed_route_relations") or []
    if allowed_route_relations and candidate.get("route_relation") in allowed_route_relations:
        score += 0.25
        reasons.append(f"route_relation={candidate.get('route_relation')}")

    for flag_name in hints.get("prefer_flags") or []:
        if bool((candidate.get("flags") or {}).get(flag_name)):
            score += 0.18
            reasons.append(f"flag:{flag_name}")

    risk_summary = ((record.get("world_state") or {}).get("risk_summary")) or {}
    if hints.get("risk_hint") == "front_hazard_track_id" and risk_summary.get("front_hazard_track_id") == candidate.get("track_id"):
        score += 0.2
        reasons.append("risk_front_hazard_match")

    track_score = ((candidate.get("source_track") or {}).get("risk_score"))
    if track_score is not None:
        score += min(float(track_score), 0.2)
        reasons.append("risk_score_bias")

    if bool(hints.get("require_same_direction")) and not bool((candidate.get("flags") or {}).get("same_direction_as_ego_lane")):
        score -= 0.3
        reasons.append("same_direction_penalty")

    return score, reasons


def _derive_critical_actor_binding_records(
    *,
    scenario_id: str,
    ground_truth: dict[str, Any] | None,
    timeline_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not ground_truth or not timeline_records:
        return []

    roles = [actor for actor in (ground_truth.get("actors") or []) if str(actor.get("entity_type", "actor")) == "actor"]
    previous_by_role: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for record in timeline_records:
        merged_candidates = _merge_candidates(record)
        sample_name = str(record.get("sample_name", ""))
        frame_id = int(record.get("frame_id", 0))
        for role_spec in roles:
            role = str(role_spec.get("role"))
            scored: list[tuple[float, dict[str, Any], list[str]]] = []
            for candidate in merged_candidates:
                score, reasons = _score_candidate(role_spec, candidate, record)
                if score < 0:
                    continue
                scored.append((score, candidate, reasons))
            scored.sort(key=lambda item: item[0], reverse=True)

            status = "missing"
            confidence = 0.0
            bound_track_id = None
            bound_object_id = None
            bound_class_name = None
            observed_lane_relation = None
            observed_route_relation = None
            binding_reason = "no_candidate"
            notes: list[str] = []

            previous = previous_by_role.get(role)
            if scored:
                best_score, best_candidate, best_reasons = scored[0]
                second_score = scored[1][0] if len(scored) > 1 else -1.0
                confidence = max(0.0, min(1.0, best_score))
                bound_track_id = best_candidate.get("track_id")
                bound_object_id = ((best_candidate.get("source_track") or {}).get("latest_detection_id"))
                bound_class_name = best_candidate.get("class_name")
                observed_lane_relation = best_candidate.get("lane_relation")
                observed_route_relation = best_candidate.get("route_relation")
                if best_score < 0.25:
                    status = "missing"
                    binding_reason = "candidate_score_too_low"
                elif second_score >= 0 and abs(best_score - second_score) < 0.08:
                    status = "ambiguous"
                    binding_reason = ";".join(best_reasons + ["ambiguous_top_candidates"])
                else:
                    status = "bound"
                    binding_reason = ";".join(best_reasons) or "heuristic_match"
            elif previous and previous.get("status") in {"bound", "ambiguous"}:
                status = "stale"
                bound_track_id = previous.get("bound_track_id")
                bound_object_id = previous.get("bound_object_id")
                bound_class_name = previous.get("bound_class_name")
                observed_lane_relation = previous.get("observed_lane_relation")
                observed_route_relation = previous.get("observed_route_relation")
                confidence = max(0.0, float(previous.get("binding_confidence", 0.0)) * 0.5)
                binding_reason = "previous_binding_stale_hold"
                notes.append("no_candidate_reused_previous_binding")
                # Stage 5A: enforce stale hold limit
                stale_count = int(previous.get("_stale_hold_count", 0)) + 1
                if stale_count > MAX_STALE_HOLD_FRAMES:
                    status = "missing"
                    confidence = 0.0
                    binding_reason = "stale_hold_exceeded_max_frames"
                    notes.append(f"stale_hold_count={stale_count}_exceeded_{MAX_STALE_HOLD_FRAMES}")

            role_record = {
                "schema_version": "system_benchmark_semantics_v1_1",
                "scenario_id": scenario_id,
                "frame_id": frame_id,
                "sample_name": sample_name,
                "role": role,
                "entity_type": role_spec.get("entity_type", "actor"),
                "expected_actor_ref": role_spec.get("expected_actor_ref", f"role:{role}"),
                "bound_track_id": bound_track_id,
                "bound_object_id": bound_object_id,
                "bound_class_name": bound_class_name,
                "binding_confidence": confidence,
                "binding_reason": binding_reason,
                "status": status,
                "observed_lane_relation": observed_lane_relation,
                "observed_route_relation": observed_route_relation,
                "expected_lane_relation": role_spec.get("expected_lane_relation"),
                "notes": notes,
            }
            previous_by_role[role] = role_record
            records.append(role_record)

    return records


def _classify_arbitration_action(
    *,
    requested_behavior: str | None,
    current_behavior: str | None,
    execution_status: str | None,
    notes: list[str],
) -> tuple[str, str]:
    notes = [str(note) for note in notes]
    if "continuing_active_lane_change" in notes:
        return "continue", "continuing_active_lane_change"
    if execution_status in {"fail", "aborted"}:
        return "cancel", notes[0] if notes else execution_status
    # Stage 5A: explicit lane_change_abort when commit is in progress
    if current_behavior and current_behavior.startswith("commit_lane_change") and execution_status == "aborted":
        return "cancel", "lane_change_abort_during_commit"
    if current_behavior == "stop_before_obstacle" and requested_behavior != current_behavior:
        return "emergency_override", notes[0] if notes else "stop_preempted_requested_behavior"
    # Stage 5A: stop_hold pattern — stop remains active when requested is also stop variant
    if current_behavior == "stop_before_obstacle" and requested_behavior in {
        "stop_before_obstacle", "follow_to_stop", "stop_hold", "emergency_stop"
    }:
        return "continue", "stop_hold_compatible"
    if current_behavior in {"slow_down", "follow", "keep_lane"} and requested_behavior and requested_behavior.startswith(("prepare_lane_change", "commit_lane_change")):
        return "downgrade", notes[0] if notes else "lane_change_not_executed_as_requested"
    if requested_behavior != current_behavior:
        return "downgrade", notes[0] if notes else "requested_behavior_differs_from_execution"
    return "continue", notes[0] if notes else "no_arbitration_needed"


def _classify_arbitration_event_against_case(
    *,
    action: str,
    expected_actions: set[str],
    forbidden_actions: set[str],
) -> str:
    if action in forbidden_actions:
        return "unexpected"
    if action == "continue":
        return "diagnostic"
    if expected_actions:
        return "expected" if action in expected_actions else "unexpected"
    return "diagnostic"


def _derive_arbitration_events(
    *,
    scenario_id: str,
    case_spec: dict[str, Any],
    execution_records: list[dict[str, Any]],
    online_tick_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    expected_actions = set(case_spec.get("expected_arbitration_actions") or [])
    forbidden_actions = set(case_spec.get("forbidden_arbitration_actions") or [])
    events: list[dict[str, Any]] = []

    if online_tick_records:
        for tick in online_tick_records:
            action = tick.get("arbitration_action")
            if not action:
                continue
            classification = _classify_arbitration_event_against_case(
                action=str(action),
                expected_actions=expected_actions,
                forbidden_actions=forbidden_actions,
            )
            events.append(
                {
                    "schema_version": "system_benchmark_semantics_v1_1",
                    "scenario_id": scenario_id,
                    "frame_id": tick.get("frame_id"),
                    "sample_name": None,
                    "requested_behavior": tick.get("requested_behavior"),
                    "previous_execution_behavior": tick.get("previous_execution_behavior"),
                    "current_execution_behavior": tick.get("selected_behavior"),
                    "executing_behavior": tick.get("selected_behavior"),
                    "arbitration_action": action,
                    "arbitration_reason": ";".join(tick.get("health_flags") or []),
                    "selected_behavior": tick.get("selected_behavior"),
                    "selected_execution_mode": "stage4_online",
                    "result_status": tick.get("execution_status"),
                    "classification": classification,
                    "notes": tick.get("notes") or [],
                }
            )
        return events

    previous_behavior: str | None = None
    for record in execution_records:
        request = record.get("execution_request") or {}
        state = record.get("execution_state") or {}
        requested_behavior = request.get("requested_behavior")
        current_behavior = state.get("current_behavior")
        execution_status = state.get("execution_status")
        notes = state.get("notes") or []
        if requested_behavior == current_behavior and execution_status not in {"fail", "aborted"}:
            previous_behavior = current_behavior
            continue
        action, reason = _classify_arbitration_action(
            requested_behavior=requested_behavior,
            current_behavior=current_behavior,
            execution_status=execution_status,
            notes=notes,
        )
        classification = _classify_arbitration_event_against_case(
            action=action,
            expected_actions=expected_actions,
            forbidden_actions=forbidden_actions,
        )
        events.append(
            {
                "schema_version": "system_benchmark_semantics_v1_1",
                "scenario_id": scenario_id,
                "frame_id": request.get("frame"),
                "sample_name": record.get("sample_name"),
                "requested_behavior": requested_behavior,
                "previous_execution_behavior": previous_behavior,
                "current_execution_behavior": current_behavior,
                "executing_behavior": current_behavior,
                "arbitration_action": action,
                "arbitration_reason": reason,
                "selected_behavior": current_behavior,
                "selected_execution_mode": request.get("execution_mode"),
                "result_status": execution_status,
                "classification": classification,
                "notes": notes,
            }
        )
        previous_behavior = current_behavior
    return events


def _derive_route_session_binding(
    *,
    scenario_id: str,
    case_spec: dict[str, Any],
    stage3b_summary: dict[str, Any] | None,
    stage3b_timeline: list[dict[str, Any]],
    stage4_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not stage3b_timeline and not stage4_summary:
        return None
    first_record = stage3b_timeline[0] if stage3b_timeline else {}
    route_context = first_record.get("route_context") or {}
    route_mode = route_context.get("route_mode") or "corridor_only"
    route_option = route_context.get("route_option") or case_spec.get("route_option")
    preferred_lane = route_context.get("preferred_lane") or _infer_expected_preferred_lane(route_option)
    route_priority = route_context.get("route_priority")
    hint_source = route_context.get("hint_source")
    stage3b_summary = stage3b_summary or {}
    num_frames = int(stage3b_summary.get("num_frames", len(stage3b_timeline)))
    route_override_count = int(stage3b_summary.get("route_conditioned_override_count", 0))
    route_influenced_frames = int(stage3b_summary.get("route_influenced_frames", 0))
    route_confidence = float(route_influenced_frames / num_frames) if num_frames > 0 else None
    route_conflict_flags = list(route_context.get("route_conflict_flags") or [])
    if case_spec.get("route_option") and route_option and route_option != case_spec.get("route_option"):
        route_conflict_flags.append("route_option_mismatch_case_expectation")

    source_of_route_binding = hint_source
    if not source_of_route_binding and route_context.get("route_option") is not None:
        source_of_route_binding = "stage3b_route_context"
    if not source_of_route_binding and case_spec.get("route_option"):
        source_of_route_binding = "case_route_option"
    if not source_of_route_binding:
        source_of_route_binding = "unknown"

    route_override_samples: list[dict[str, Any]] = []
    for record in stage3b_timeline:
        comparison = record.get("comparison") or {}
        if not bool(comparison.get("route_influenced")):
            continue
        selected = (record.get("behavior_output") or {}).get("selected_behavior")
        stage3a_candidate = comparison.get("stage3a_behavior")
        if selected is None or stage3a_candidate is None or selected == stage3a_candidate:
            continue
        route_override_samples.append(
            {
                "frame_id": record.get("frame_id"),
                "sample_name": record.get("sample_name"),
                "stage3a_behavior": stage3a_candidate,
                "selected_behavior": selected,
            }
        )
    return {
        "schema_version": "system_benchmark_semantics_v1_1",
        "scenario_id": scenario_id,
        "route_binding": {
            "route_mode": route_mode,
            "route_option": route_option,
            "preferred_lane": preferred_lane,
            "route_priority": route_priority,
            "source_of_route_binding": source_of_route_binding,
            "route_confidence": route_confidence,
            "route_conflict_flags": route_conflict_flags,
            "route_conditioned_override_count": route_override_count,
            "route_conditioned_override_rate": stage3b_summary.get("route_conditioned_override_rate"),
            "route_influenced_frames": route_influenced_frames,
            "route_overrides": {
                "count": route_override_count,
                "samples": route_override_samples[:20],
            },
        },
        "session_binding": {
            "session_binding_source": case_spec.get("source", {}).get("type"),
            "mode": case_spec.get("mode"),
            "stage1_session_dir": case_spec.get("source", {}).get("stage1_session_dir"),
            "stage2_dir": case_spec.get("source", {}).get("stage2_dir"),
            "stage3a_dir": case_spec.get("source", {}).get("stage3a_dir"),
            "stage3b_dir": case_spec.get("source", {}).get("stage3b_dir"),
            "stage3c_dir": case_spec.get("source", {}).get("stage3c_dir"),
            "stage4_dir": case_spec.get("source", {}).get("stage4_dir"),
        },
        "expected_binding": {
            "expected_route_option": case_spec.get("route_option"),
            "expected_preferred_lane": ((case_spec.get("route_binding_expectations") or {}).get("preferred_lane")) or _infer_expected_preferred_lane(case_spec.get("route_option")),
        },
        "notes": [],
    }


def ensure_semantic_artifacts(
    *,
    case_spec: dict[str, Any],
    repo_root: Path,
    generated_root: Path,
) -> tuple[dict[str, Path | None], dict[str, SemanticArtifact]]:
    scenario_id = case_spec["scenario_id"]
    artifacts = case_spec.get("artifacts", {})
    generated_case_root = generated_root / scenario_id
    generated_case_root.mkdir(parents=True, exist_ok=True)

    resolved_paths: dict[str, Path | None] = {}
    metadata: dict[str, SemanticArtifact] = {}

    ground_truth_path = _resolve_ground_truth_path(repo_root, artifacts, scenario_id)
    ground_truth = load_json(ground_truth_path)
    gt_source = "missing"
    gt_notes: list[str] = []
    if ground_truth_path and ground_truth_path.exists():
        gt_source = "existing"
    else:
        ground_truth = _derive_ground_truth_actor_roles(scenario_id=scenario_id, case_spec=case_spec)
        ground_truth_path = generated_case_root / "ground_truth_actor_roles.json"
        dump_json(ground_truth_path, ground_truth)
        gt_source = "generated"
        gt_notes.append("auto_generated_from_case_spec_with_pending_flag")
    gt_available = bool(ground_truth_path and ground_truth_path.exists())
    resolved_paths["ground_truth_actor_roles"] = ground_truth_path
    metadata["ground_truth_actor_roles"] = SemanticArtifact(
        name="ground_truth_actor_roles",
        path=(str(ground_truth_path) if ground_truth_path else None),
        available=gt_available,
        source=gt_source,
        notes=gt_notes,
    )

    stage3b_timeline_path = resolve_repo_path(repo_root, artifacts.get("stage3b_behavior_timeline"))
    stage3b_timeline = load_jsonl(stage3b_timeline_path)
    stage3b_summary = load_json(resolve_repo_path(repo_root, artifacts.get("stage3b_evaluation_summary")))
    stage3c_execution_timeline_path = resolve_repo_path(repo_root, artifacts.get("stage3c_execution_timeline"))
    stage3c_execution_records = load_jsonl(stage3c_execution_timeline_path)
    stage4_online_run_summary = load_json(resolve_repo_path(repo_root, artifacts.get("stage4_online_run_summary")))
    online_tick_records = load_jsonl(resolve_repo_path(repo_root, artifacts.get("stage4_online_tick_record")))

    binding_path = resolve_repo_path(repo_root, artifacts.get("critical_actor_binding"))
    binding_records: list[dict[str, Any]] = []
    binding_source = "missing"
    binding_notes: list[str] = []
    if binding_path and binding_path.exists():
        binding_records = load_jsonl(binding_path)
        binding_source = "existing"
    elif ground_truth and stage3b_timeline:
        binding_records = _derive_critical_actor_binding_records(
            scenario_id=scenario_id,
            ground_truth=ground_truth,
            timeline_records=stage3b_timeline,
        )
        if binding_records:
            binding_path = generated_case_root / "critical_actor_binding.jsonl"
            dump_jsonl(binding_path, binding_records)
            binding_source = "generated"
        else:
            binding_notes.append("unable_to_derive_binding_records")
    else:
        binding_notes.append("missing_ground_truth_or_behavior_timeline")
    if binding_path is None and ground_truth and not (ground_truth.get("actors") or []):
        binding_path = generated_case_root / "critical_actor_binding.jsonl"
        dump_jsonl(binding_path, [])
        binding_source = "generated_empty_not_applicable"
        binding_notes = ["no_actor_roles_in_ground_truth"]
    resolved_paths["critical_actor_binding"] = binding_path
    metadata["critical_actor_binding"] = SemanticArtifact(
        name="critical_actor_binding",
        path=(str(binding_path) if binding_path else None),
        available=bool(binding_path and Path(binding_path).exists()),
        source=binding_source,
        notes=binding_notes,
    )

    arbitration_path = resolve_repo_path(repo_root, artifacts.get("behavior_arbitration_events"))
    arbitration_source = "missing"
    arbitration_notes: list[str] = []
    if arbitration_path and arbitration_path.exists():
        arbitration_source = "existing"
    else:
        arbitration_events = _derive_arbitration_events(
            scenario_id=scenario_id,
            case_spec=case_spec,
            execution_records=stage3c_execution_records,
            online_tick_records=online_tick_records or None,
        )
        if arbitration_events:
            arbitration_path = generated_case_root / "behavior_arbitration_events.jsonl"
            dump_jsonl(arbitration_path, arbitration_events)
            arbitration_source = "generated"
        else:
            arbitration_notes.append("unable_to_derive_arbitration_events")
    if arbitration_path is None and stage3c_execution_records and not (
        case_spec.get("expected_arbitration_actions") or case_spec.get("forbidden_arbitration_actions")
    ):
        arbitration_path = generated_case_root / "behavior_arbitration_events.jsonl"
        dump_jsonl(arbitration_path, [])
        arbitration_source = "generated_empty_not_applicable"
        arbitration_notes = ["no_arbitration_events_observed_or_expected"]
    resolved_paths["behavior_arbitration_events"] = arbitration_path
    metadata["behavior_arbitration_events"] = SemanticArtifact(
        name="behavior_arbitration_events",
        path=(str(arbitration_path) if arbitration_path else None),
        available=bool(arbitration_path and Path(arbitration_path).exists()),
        source=arbitration_source,
        notes=arbitration_notes,
    )

    route_binding_path = resolve_repo_path(repo_root, artifacts.get("route_session_binding"))
    route_binding_source = "missing"
    route_binding_notes: list[str] = []
    if route_binding_path and route_binding_path.exists():
        route_binding_source = "existing"
    else:
        route_binding = _derive_route_session_binding(
            scenario_id=scenario_id,
            case_spec=case_spec,
            stage3b_summary=stage3b_summary,
            stage3b_timeline=stage3b_timeline,
            stage4_summary=stage4_online_run_summary,
        )
        if route_binding:
            route_binding_path = generated_case_root / "route_session_binding.json"
            dump_json(route_binding_path, route_binding)
            route_binding_source = "generated"
        else:
            route_binding_notes.append("unable_to_derive_route_session_binding")
    resolved_paths["route_session_binding"] = route_binding_path
    metadata["route_session_binding"] = SemanticArtifact(
        name="route_session_binding",
        path=(str(route_binding_path) if route_binding_path else None),
        available=bool(route_binding_path and Path(route_binding_path).exists()),
        source=route_binding_source,
        notes=route_binding_notes,
    )

    return resolved_paths, metadata
