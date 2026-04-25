"""
Stage 5A Failure Replay Case Builder
=======================================
Creates new failure replay case YAML files for Stage 5A-specific regressions:
- Blocker binding wrong lane
- Route/session mismatch
- Arbitration missing cancel
- Stop overshoot false pass
- Lane change commit without permission
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .io import dump_json, load_yaml, resolve_repo_path


# ══════════════════════════════════════════════════════════════════════════════
# Failure replay case templates
# ══════════════════════════════════════════════════════════════════════════════

FAILURE_REPLAY_CASES: list[dict[str, Any]] = [
    {
        "scenario_id": "fr_blocker_binding_wrong_lane",
        "status": "ready",
        "family": "failure_replay",
        "mode": "failure_replay_artifact_audit",
        "source": {
            "type": "existing_artifact_bundle",
            "scenario_manifest": "outputs/scenario_manifests/stage3a_multilane_right_long2_manifest.json",
            "stage1_session_dir": "outputs/live_watch_compare_multilane/lw_s3a_ml_long_v2",
            "stage2_dir": "outputs/stage2/st2_s3a_ml_long_v2",
            "stage3a_dir": "outputs/stage3/st3_s3a_ml_long_v2",
            "stage3b_dir": "outputs/stage3b/st3b_s3a_ml_long_v2",
            "stage3c_dir": "outputs/stage3c/st3c_s3a_ml_long_v2_mp4_topdown",
        },
        "owner": "agentai_system_benchmark",
        "determinism_level": "medium",
        "route_option": "keep_right",
        "critical_actors": ["blocker_ahead_current_lane", "adjacent_right_lane_vehicle"],
        "expected_behaviors": ["stop_before_obstacle", "follow"],
        "forbidden_behaviors": ["commit_lane_change_left", "prepare_lane_change_left"],
        "expected_execution": {"stop_successes_min": 100},
        "pending_ground_truth_binding": False,
        "expected_arbitration_actions": ["continue"],
        "forbidden_arbitration_actions": [],
        "stage5a_failure_class": "blocker_binding_wrong_lane",
        "stage5a_failure_description": (
            "Catches regression where a blocker in an adjacent lane is incorrectly "
            "classified as a current-lane blocker, triggering false stops."
        ),
        "mandatory_artifacts": [
            "ground_truth_actor_roles", "critical_actor_binding",
            "behavior_arbitration_events", "route_session_binding",
            "stage1_session_summary", "stage2_evaluation_summary",
            "stage2_world_state_sample", "stage2_tracked_objects_sample",
            "stage3a_evaluation_summary", "stage3a_lane_world_state_sample",
            "stage3b_evaluation_summary", "stage3b_behavior_timeline",
            "stage3b_behavior_request_sample",
            "stage3c_execution_summary", "stage3c_execution_timeline",
            "stage3c_lane_change_events",
        ],
        "primary_metrics": [
            "artifact_completeness", "blocker_binding_accuracy",
            "lane_assignment_consistency", "critical_actor_detected_rate",
        ],
        "secondary_metrics": [
            "stop_success_rate", "behavior_execution_mismatch_rate",
        ],
        "metric_thresholds": {
            "blocker_binding_accuracy": {"min": 0.6},
            "lane_assignment_consistency": {"min": 0.7},
            "critical_actor_detected_rate": {"min": 0.7},
        },
        "notes": (
            "Stage 5A failure replay: blocker binding wrong lane.\n"
            "This case monitors for regressions where the obstacle binding logic\n"
            "assigns adjacent-lane actors to the current lane blocker role.\n"
        ),
    },
    {
        "scenario_id": "fr_route_session_mismatch",
        "status": "ready",
        "family": "failure_replay",
        "mode": "failure_replay_artifact_audit",
        "source": {
            "type": "existing_artifact_bundle",
            "scenario_manifest": "outputs/scenario_manifests/stage3a_multilane_right_long_manifest.json",
            "stage1_session_dir": "outputs/live_watch_compare_multilane/lw_s3a_ml_long_v1",
            "stage2_dir": "outputs/stage2/st2_s3a_ml_long_v1",
            "stage3a_dir": "outputs/stage3/st3_s3a_ml_long_v1",
            "stage3b_dir": "outputs/stage3b/st3b_s3a_ml_long_v1",
            "stage3c_dir": "outputs/stage3c/st3c_s3a_ml_long_v1_mp4",
        },
        "owner": "agentai_system_benchmark",
        "determinism_level": "high",
        "route_option": "keep_right",
        "critical_actors": ["blocker_ahead_current_lane"],
        "expected_behaviors": ["commit_lane_change_right"],
        "forbidden_behaviors": ["commit_lane_change_left"],
        "expected_execution": {"lane_change_successes_min": 1},
        "pending_ground_truth_binding": False,
        "stage5a_failure_class": "route_session_mismatch",
        "stage5a_failure_description": (
            "Catches regression where route option or preferred lane derived from "
            "session binding does not match the case expectation."
        ),
        "mandatory_artifacts": [
            "ground_truth_actor_roles", "route_session_binding",
            "stage1_session_summary", "stage2_evaluation_summary",
            "stage2_world_state_sample", "stage2_tracked_objects_sample",
            "stage3a_evaluation_summary", "stage3a_lane_world_state_sample",
            "stage3b_evaluation_summary", "stage3b_behavior_timeline",
            "stage3b_behavior_request_sample",
            "stage3c_execution_summary", "stage3c_execution_timeline",
            "stage3c_lane_change_events",
        ],
        "primary_metrics": [
            "artifact_completeness", "route_option_match",
            "route_context_consistency",
        ],
        "secondary_metrics": [
            "route_influence_correctness", "route_conditioned_override_rate",
        ],
        "metric_thresholds": {
            "route_option_match": {"min": 1.0},
            "route_context_consistency": {"min": 1.0},
        },
        "notes": (
            "Stage 5A failure replay: route/session binding mismatch.\n"
            "Monitors for regressions where route option or preferred lane\n"
            "becomes inconsistent with the expected case specification.\n"
        ),
    },
    {
        "scenario_id": "fr_arbitration_missing_cancel",
        "status": "ready",
        "family": "failure_replay",
        "mode": "failure_replay_artifact_audit",
        "source": {
            "type": "existing_artifact_bundle",
            "scenario_manifest": "outputs/scenario_manifests/stage3a_multilane_right_long2_manifest.json",
            "stage1_session_dir": "outputs/live_watch_compare_multilane/lw_s3a_ml_long_v2",
            "stage2_dir": "outputs/stage2/st2_s3a_ml_long_v2",
            "stage3a_dir": "outputs/stage3/st3_s3a_ml_long_v2",
            "stage3b_dir": "outputs/stage3b/st3b_s3a_ml_long_v2",
            "stage3c_dir": "outputs/stage3c/st3c_s3a_ml_long_v2_mp4_topdown",
        },
        "owner": "agentai_system_benchmark",
        "determinism_level": "medium",
        "route_option": "keep_right",
        "critical_actors": ["blocker_ahead_current_lane", "lane_change_target_right"],
        "expected_behaviors": ["prepare_lane_change_right", "stop_before_obstacle"],
        "forbidden_behaviors": ["commit_lane_change_left"],
        "expected_execution": {"stop_successes_min": 100},
        "pending_ground_truth_binding": False,
        "expected_arbitration_actions": ["downgrade", "emergency_override", "cancel"],
        "forbidden_arbitration_actions": [],
        "stage5a_failure_class": "arbitration_missing_cancel",
        "stage5a_failure_description": (
            "Catches regression where a behavior cancellation does not emit an "
            "arbitration cancel event — i.e., a prepare is superseded by stop "
            "without proper cancellation logging."
        ),
        "mandatory_artifacts": [
            "ground_truth_actor_roles", "critical_actor_binding",
            "behavior_arbitration_events", "route_session_binding",
            "stage1_session_summary", "stage2_evaluation_summary",
            "stage2_world_state_sample", "stage2_tracked_objects_sample",
            "stage3a_evaluation_summary", "stage3a_lane_world_state_sample",
            "stage3b_evaluation_summary", "stage3b_behavior_timeline",
            "stage3b_behavior_request_sample",
            "stage3c_execution_summary", "stage3c_execution_timeline",
            "stage3c_lane_change_events",
        ],
        "primary_metrics": [
            "artifact_completeness", "arbitration_conflict_count",
            "behavior_execution_mismatch_rate",
        ],
        "secondary_metrics": [
            "stop_success_rate", "abort_rate",
        ],
        "metric_thresholds": {
            "arbitration_conflict_count": {"max": 0},
            "behavior_execution_mismatch_rate": {"max": 0.35},
        },
        "notes": (
            "Stage 5A failure replay: arbitration missing cancel.\n"
            "Detects when a behavior transition should have emitted a cancel event\n"
            "but did not — a key arbitration contract violation.\n"
        ),
    },
    {
        "scenario_id": "fr_stop_overshoot_false_pass",
        "status": "ready",
        "family": "failure_replay",
        "mode": "failure_replay_artifact_audit",
        "source": {
            "type": "existing_artifact_bundle",
            "scenario_manifest": "outputs/scenario_manifests/stage3a_multilane_right_long2_manifest.json",
            "stage1_session_dir": "outputs/live_watch_compare_multilane/lw_s3a_ml_long_v2",
            "stage2_dir": "outputs/stage2/st2_s3a_ml_long_v2",
            "stage3a_dir": "outputs/stage3/st3_s3a_ml_long_v2",
            "stage3b_dir": "outputs/stage3b/st3b_s3a_ml_long_v2",
            "stage3c_dir": "outputs/stage3c/st3c_s3a_ml_long_v2_mp4_topdown",
        },
        "owner": "agentai_system_benchmark",
        "determinism_level": "medium",
        "route_option": "keep_right",
        "critical_actors": ["blocker_ahead_current_lane"],
        "expected_behaviors": ["stop_before_obstacle"],
        "forbidden_behaviors": ["commit_lane_change_left"],
        "expected_execution": {"stop_successes_min": 100, "stop_overshoots_max": 0},
        "pending_ground_truth_binding": False,
        "stage5a_failure_class": "stop_overshoot_false_pass",
        "stage5a_failure_description": (
            "Catches regression where a stop overshoot event is incorrectly "
            "classified as a success, inflating stop_success_rate."
        ),
        "mandatory_artifacts": [
            "ground_truth_actor_roles",
            "stage1_session_summary", "stage2_evaluation_summary",
            "stage2_world_state_sample", "stage2_tracked_objects_sample",
            "stage3a_evaluation_summary", "stage3a_lane_world_state_sample",
            "stage3b_evaluation_summary", "stage3b_behavior_timeline",
            "stage3b_behavior_request_sample",
            "stage3c_execution_summary", "stage3c_execution_timeline",
            "stage3c_lane_change_events",
        ],
        "primary_metrics": [
            "artifact_completeness", "stop_success_rate",
            "stop_overshoot_rate",
        ],
        "secondary_metrics": [
            "behavior_execution_mismatch_rate",
        ],
        "metric_thresholds": {
            "stop_success_rate": {"min": 0.9},
            "stop_overshoot_rate": {"max": 0.0},
        },
        "notes": (
            "Stage 5A failure replay: stop overshoot false pass.\n"
            "Monitors for the specific regression where overshoot events are\n"
            "counted as success, masking real stop control failures.\n"
        ),
    },
    {
        "scenario_id": "fr_lc_commit_no_permission",
        "status": "ready",
        "family": "failure_replay",
        "mode": "failure_replay_artifact_audit",
        "source": {
            "type": "existing_artifact_bundle",
            "scenario_manifest": "outputs/scenario_manifests/stage3a_multilane_right_long_manifest.json",
            "stage1_session_dir": "outputs/live_watch_compare_multilane/lw_s3a_ml_long_v1",
            "stage2_dir": "outputs/stage2/st2_s3a_ml_long_v1",
            "stage3a_dir": "outputs/stage3/st3_s3a_ml_long_v1",
            "stage3b_dir": "outputs/stage3b/st3b_s3a_ml_long_v1",
            "stage3c_dir": "outputs/stage3c/st3c_s3a_ml_long_v1_mp4",
        },
        "owner": "agentai_system_benchmark",
        "determinism_level": "high",
        "route_option": "keep_right",
        "critical_actors": ["blocker_ahead_current_lane", "adjacent_right_lane_vehicle"],
        "expected_behaviors": ["commit_lane_change_right"],
        "forbidden_behaviors": ["commit_lane_change_left"],
        "expected_execution": {"lane_change_successes_min": 1},
        "pending_ground_truth_binding": False,
        "stage5a_failure_class": "lc_commit_no_permission",
        "stage5a_failure_description": (
            "Catches regression where a lane change commit is issued "
            "without the corresponding lane_change_permission being true."
        ),
        "mandatory_artifacts": [
            "ground_truth_actor_roles",
            "stage1_session_summary", "stage2_evaluation_summary",
            "stage2_world_state_sample", "stage2_tracked_objects_sample",
            "stage3a_evaluation_summary", "stage3a_lane_world_state_sample",
            "stage3b_evaluation_summary", "stage3b_behavior_timeline",
            "stage3b_behavior_request_sample",
            "stage3c_execution_summary", "stage3c_execution_timeline",
            "stage3c_lane_change_events",
        ],
        "primary_metrics": [
            "artifact_completeness", "unsafe_lane_change_intent_count",
            "lane_change_success_rate",
        ],
        "secondary_metrics": [
            "behavior_execution_mismatch_rate", "abort_rate",
        ],
        "metric_thresholds": {
            "unsafe_lane_change_intent_count": {"max": 0},
            "lane_change_success_rate": {"min": 0.5},
        },
        "notes": (
            "Stage 5A failure replay: lane change commit without permission.\n"
            "Detects when the behavior layer issues a commit_lane_change without\n"
            "the safety permission gate being satisfied.\n"
        ),
    },
]


def _case_artifact_paths(case: dict[str, Any]) -> dict[str, str]:
    """Generate standard artifact path mapping from source dirs."""
    source = case.get("source", {})
    sid = case["scenario_id"]
    s1 = source.get("stage1_session_dir", "")
    s2 = source.get("stage2_dir", "")
    s3a = source.get("stage3a_dir", "")
    s3b = source.get("stage3b_dir", "")
    s3c = source.get("stage3c_dir", "")
    sample = "sample_000029"

    return {
        "scenario_manifest": source.get("scenario_manifest", ""),
        "ground_truth_actor_roles": f"benchmark/reference_artifacts/{sid}/ground_truth_actor_roles.json",
        "critical_actor_binding": f"benchmark/reference_artifacts/{sid}/critical_actor_binding.jsonl",
        "behavior_arbitration_events": f"benchmark/reference_artifacts/{sid}/behavior_arbitration_events.jsonl",
        "route_session_binding": f"benchmark/reference_artifacts/{sid}/route_session_binding.json",
        "stage1_session_summary": f"{s1}/session_summary.json",
        "stage2_evaluation_summary": f"{s2}/evaluation_summary.json",
        "stage2_world_state_sample": f"{s2}/frames/{sample}/world_state.json",
        "stage2_tracked_objects_sample": f"{s2}/frames/{sample}/tracked_objects.json",
        "stage3a_evaluation_summary": f"{s3a}/evaluation_summary.json",
        "stage3a_lane_world_state_sample": f"{s3a}/frames/{sample}/lane_aware_world_state.json",
        "stage3b_evaluation_summary": f"{s3b}/evaluation_summary.json",
        "stage3b_behavior_timeline": f"{s3b}/behavior_timeline_v2.jsonl",
        "stage3b_behavior_request_sample": f"{s3b}/frames/{sample}/behavior_request_v2.json",
        "stage3c_execution_summary": f"{s3c}/execution_summary.json",
        "stage3c_execution_timeline": f"{s3c}/execution_timeline.jsonl",
        "stage3c_lane_change_events": f"{s3c}/lane_change_events.jsonl",
    }


def _stage5a_replay_contract(case: dict[str, Any]) -> dict[str, Any]:
    source = dict(case.get("source") or {})
    route_option = case.get("route_option")
    preferred_lane = "current"
    if route_option == "keep_right":
        preferred_lane = "right"
    elif route_option == "keep_left":
        preferred_lane = "left"

    return {
        "benchmark_mode": "replay_regression",
        "supported_benchmark_modes": [
            "replay_regression",
            "scenario_replay",
        ],
        "execution_profile": f"stage5a_failure_replay_{case['scenario_id']}",
        "scenario_recipe": {
            "source_type": "scenario_manifest",
            "scenario_manifest": source.get("scenario_manifest"),
            "attach_strategy": "manifest_ego_actor",
            "stage1_session_dir": source.get("stage1_session_dir"),
        },
        "route_profile": {
            "route_option": route_option,
            "preferred_lane": preferred_lane,
            "route_priority": "normal",
        },
        "stage_profiles": {
            "stage1": "stage1_session_attach_default",
            "stage2": "stage2_replay_default",
            "stage3a": "stage3a_replay_default",
            "stage3b": "stage3b_replay_default",
            "stage3c": "stage3c_closed_loop_default",
        },
        "expected_stage_chain": [
            "spawn",
            "stage1",
            "stage2",
            "stage3a",
            "stage3b",
            "stage3c",
            "evaluate",
        ],
        "trace_requirements": {
            "require_current_run_artifacts": True,
            "required_steps": [
                "spawn",
                "stage1",
                "stage2",
                "stage3a",
                "stage3b",
                "stage3c",
                "evaluate",
            ],
        },
    }


def build_failure_replay_cases(
    repo_root: str | Path,
) -> list[str]:
    """Generate failure replay case YAML files for Stage 5A."""
    repo_root = Path(repo_root)
    cases_dir = repo_root / "benchmark" / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    created_files: list[str] = []
    for case_template in FAILURE_REPLAY_CASES:
        case = dict(case_template)
        case.update(_stage5a_replay_contract(case))
        case["artifacts"] = _case_artifact_paths(case)
        case_path = cases_dir / f"{case['scenario_id']}.yaml"

        with case_path.open("w", encoding="utf-8") as f:
            yaml.dump(case, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

        created_files.append(str(case_path))
        print(f"[Stage5A] Created failure replay case: {case_path.name}")

    return created_files
