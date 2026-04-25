from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage3.evaluation_stage3 import summarize_stage3_session
from stage3b.evaluation_stage3b import summarize_stage3b_session
from stage3c.evaluation_stage3c import summarize_stage3c_session
from stage3c_coverage.planner_quality import write_planner_quality_artifacts

from .io import dump_json, dump_jsonl, load_json, load_jsonl

FULL_STAGE_CHAIN = ["spawn", "stage1", "stage2", "stage3a", "stage3b", "stage3c", "stage4", "evaluate"]
MODE_STAGE_CHAINS = {
    "replay_regression": ["evaluate"],
    "scenario_replay": ["spawn", "stage1", "stage2", "stage3a", "stage3b", "stage3c", "evaluate"],
    "online_e2e": list(FULL_STAGE_CHAIN),
}


def expected_stage_chain(case_spec: dict[str, Any], benchmark_mode: str) -> list[str]:
    mode_chain = MODE_STAGE_CHAINS.get(benchmark_mode, ["evaluate"])
    explicit = case_spec.get("expected_stage_chain") or []
    if not explicit:
        return list(mode_chain)
    explicit_set = {str(item) for item in explicit}
    if benchmark_mode == "online_e2e":
        return list(FULL_STAGE_CHAIN)
    if benchmark_mode == "scenario_replay":
        return [stage for stage in MODE_STAGE_CHAINS["scenario_replay"] if stage in explicit_set or stage in {"spawn", "evaluate"}]
    return list(mode_chain)


def default_trace_requirements(case_spec: dict[str, Any], benchmark_mode: str) -> dict[str, Any]:
    existing = dict(case_spec.get("trace_requirements") or {})
    return {
        **existing,
        "require_current_run_artifacts": bool(benchmark_mode != "replay_regression"),
        "required_steps": list(MODE_STAGE_CHAINS.get(benchmark_mode, ["evaluate"])),
    }


def scenario_replay_artifacts(*, case_run_dir: Path, stage1_session_dir: Path | None) -> tuple[dict[str, str], dict[str, Any]]:
    stage1_dir = stage1_session_dir or (case_run_dir / "stage1")
    sample_name = _select_replay_sample_name(case_run_dir) or "sample_000029"
    artifacts = {
        "stage1_session_summary": str(stage1_dir / "session_summary.json"),
        "stage2_evaluation_summary": str(case_run_dir / "stage2" / "evaluation_summary.json"),
        "stage2_world_state_sample": str(case_run_dir / "stage2" / "frames" / sample_name / "world_state.json"),
        "stage2_tracked_objects_sample": str(case_run_dir / "stage2" / "frames" / sample_name / "tracked_objects.json"),
        "stage3a_evaluation_summary": str(case_run_dir / "stage3a" / "evaluation_summary.json"),
        "stage3a_lane_world_state_sample": str(case_run_dir / "stage3a" / "frames" / sample_name / "lane_aware_world_state.json"),
        "stage3b_evaluation_summary": str(case_run_dir / "stage3b" / "evaluation_summary.json"),
        "stage3b_behavior_timeline": str(case_run_dir / "stage3b" / "behavior_timeline_v2.jsonl"),
        "stage3b_behavior_request_sample": str(case_run_dir / "stage3b" / "frames" / sample_name / "behavior_request_v2.json"),
        "stage3c_execution_summary": str(case_run_dir / "stage3c" / "execution_summary.json"),
        "stage3c_execution_timeline": str(case_run_dir / "stage3c" / "execution_timeline.jsonl"),
        "stage3c_lane_change_events": str(case_run_dir / "stage3c" / "lane_change_events.jsonl"),
        "stage3c_lane_change_events_enriched": str(case_run_dir / "stage3c" / "lane_change_events_enriched.jsonl"),
        "stage3c_stop_events": str(case_run_dir / "stage3c" / "stop_events.jsonl"),
        "stage3c_trajectory_trace": str(case_run_dir / "stage3c" / "trajectory_trace.jsonl"),
        "stage3c_control_trace": str(case_run_dir / "stage3c" / "control_trace.jsonl"),
        "stage3c_lane_change_phase_trace": str(case_run_dir / "stage3c" / "lane_change_phase_trace.jsonl"),
        "stage3c_stop_target": str(case_run_dir / "stage3c" / "stop_target.json"),
        "stage3c_stop_scene_alignment": str(case_run_dir / "stage3c" / "stop_scene_alignment.json"),
        "stage3c_planner_control_latency": str(case_run_dir / "stage3c" / "planner_control_latency.jsonl"),
        "stage3c_fallback_events": str(case_run_dir / "stage3c" / "fallback_events.jsonl"),
        "stage3c_planner_quality_summary": str(case_run_dir / "stage3c" / "planner_quality_summary.json"),
        "stage3c_mpc_shadow_proposal": str(case_run_dir / "stage3c" / "mpc_shadow_proposal.jsonl"),
        "stage3c_mpc_shadow_summary": str(case_run_dir / "stage3c" / "mpc_shadow_summary.json"),
        "stage3c_shadow_baseline_comparison": str(case_run_dir / "stage3c" / "shadow_baseline_comparison.json"),
        "stage3c_shadow_disagreement_events": str(case_run_dir / "stage3c" / "shadow_disagreement_events.jsonl"),
        "stage3c_shadow_realized_outcome_trace": str(case_run_dir / "stage3c" / "shadow_realized_outcome_trace.jsonl"),
        "stage3c_shadow_realized_outcome_summary": str(case_run_dir / "stage3c" / "shadow_realized_outcome_summary.json"),
        "stage3c_shadow_runtime_latency_trace": str(case_run_dir / "stage3c" / "shadow_runtime_latency_trace.jsonl"),
        "stage3c_shadow_runtime_latency_summary": str(case_run_dir / "stage3c" / "shadow_runtime_latency_summary.json"),
    }
    source_updates = {
        "type": "scenario_replay_current_run_bundle",
        "stage1_session_dir": str(stage1_dir),
        "stage2_dir": str(case_run_dir / "stage2"),
        "stage3a_dir": str(case_run_dir / "stage3a"),
        "stage3b_dir": str(case_run_dir / "stage3b"),
        "stage3c_dir": str(case_run_dir / "stage3c"),
    }
    return artifacts, source_updates


def materialize_online_current_run_bundle(
    *,
    case_run_dir: Path,
    case_spec: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    stage4_dir = case_run_dir / "stage4"
    bundle_root = case_run_dir / "_current_run_artifacts"
    stage1_session_dir = _resolve_stage1_session_dir(stage4_dir)
    stage2_dir = bundle_root / "stage2"
    stage3a_dir = bundle_root / "stage3a"
    stage3b_dir = bundle_root / "stage3b"
    stage3c_dir = bundle_root / "stage3c"

    updates = load_jsonl(stage4_dir / "world_behavior_updates.jsonl")
    stage3a_records = [_to_stage3a_record(item) for item in updates]
    stage3b_records = [_to_stage3b_record(item) for item in updates]
    preferred_sample = _preferred_sample_name(case_spec)
    selected_sample = _select_sample_name(
        preferred_sample=preferred_sample,
        stage4_frames_dir=stage4_dir / "frames",
        records=updates,
    )

    _write_stage2_bundle(stage2_dir=stage2_dir, updates=updates, selected_sample=selected_sample, stage4_dir=stage4_dir)
    _write_stage3a_bundle(stage3a_dir=stage3a_dir, stage3a_records=stage3a_records, selected_sample=selected_sample, stage4_dir=stage4_dir)
    _write_stage3b_bundle(stage3b_dir=stage3b_dir, stage3b_records=stage3b_records, selected_sample=selected_sample, stage4_dir=stage4_dir)
    _write_stage3c_bundle(stage3c_dir=stage3c_dir, stage4_dir=stage4_dir)

    sample_name = selected_sample or "sample_000000"
    artifacts = {
        "stage1_session_summary": str((stage1_session_dir or (stage4_dir / "stage1" / "missing_session")) / "session_summary.json"),
        "stage2_evaluation_summary": str(stage2_dir / "evaluation_summary.json"),
        "stage2_world_state_sample": str(stage2_dir / "frames" / sample_name / "world_state.json"),
        "stage2_tracked_objects_sample": str(stage2_dir / "frames" / sample_name / "tracked_objects.json"),
        "stage3a_evaluation_summary": str(stage3a_dir / "evaluation_summary.json"),
        "stage3a_lane_world_state_sample": str(stage3a_dir / "frames" / sample_name / "lane_aware_world_state.json"),
        "stage3b_evaluation_summary": str(stage3b_dir / "evaluation_summary.json"),
        "stage3b_behavior_timeline": str(stage3b_dir / "behavior_timeline_v2.jsonl"),
        "stage3b_behavior_request_sample": str(stage3b_dir / "frames" / sample_name / "behavior_request_v2.json"),
        "stage3c_execution_summary": str(stage3c_dir / "execution_summary.json"),
        "stage3c_execution_timeline": str(stage3c_dir / "execution_timeline.jsonl"),
        "stage3c_lane_change_events": str(stage3c_dir / "lane_change_events.jsonl"),
        "stage3c_lane_change_events_enriched": str(stage3c_dir / "lane_change_events_enriched.jsonl"),
        "stage3c_stop_events": str(stage3c_dir / "stop_events.jsonl"),
        "stage3c_trajectory_trace": str(stage3c_dir / "trajectory_trace.jsonl"),
        "stage3c_control_trace": str(stage3c_dir / "control_trace.jsonl"),
        "stage3c_lane_change_phase_trace": str(stage3c_dir / "lane_change_phase_trace.jsonl"),
        "stage3c_stop_target": str(stage3c_dir / "stop_target.json"),
        "stage3c_stop_scene_alignment": str(stage3c_dir / "stop_scene_alignment.json"),
        "stage3c_planner_control_latency": str(stage3c_dir / "planner_control_latency.jsonl"),
        "stage3c_fallback_events": str(stage3c_dir / "fallback_events.jsonl"),
        "stage3c_planner_quality_summary": str(stage3c_dir / "planner_quality_summary.json"),
        "stage3c_mpc_shadow_proposal": str(stage3c_dir / "mpc_shadow_proposal.jsonl"),
        "stage3c_mpc_shadow_summary": str(stage3c_dir / "mpc_shadow_summary.json"),
        "stage3c_shadow_baseline_comparison": str(stage3c_dir / "shadow_baseline_comparison.json"),
        "stage3c_shadow_disagreement_events": str(stage3c_dir / "shadow_disagreement_events.jsonl"),
        "stage3c_shadow_realized_outcome_trace": str(stage3c_dir / "shadow_realized_outcome_trace.jsonl"),
        "stage3c_shadow_realized_outcome_summary": str(stage3c_dir / "shadow_realized_outcome_summary.json"),
        "stage3c_shadow_runtime_latency_trace": str(stage3c_dir / "shadow_runtime_latency_trace.jsonl"),
        "stage3c_shadow_runtime_latency_summary": str(stage3c_dir / "shadow_runtime_latency_summary.json"),
        "stage4_online_run_summary": str(stage4_dir / "online_run_summary.json"),
        "stage4_online_stack_state": str(stage4_dir / "online_stack_state.json"),
        "stage4_online_tick_record": str(stage4_dir / "online_tick_record.jsonl"),
        "stage4_planner_control_latency": str(stage4_dir / "planner_control_latency.jsonl"),
        "stage4_fallback_events": str(stage4_dir / "fallback_events.jsonl"),
        "stage4_stop_scene_alignment": str(stage4_dir / "stop_scene_alignment.json"),
        "stage4_shadow_runtime_latency_trace": str(stage4_dir / "shadow_runtime_latency_trace.jsonl"),
        "stage4_shadow_runtime_latency_summary": str(stage4_dir / "shadow_runtime_latency_summary.json"),
    }
    source_updates = {
        "type": "current_run_artifact_bundle",
        "stage1_session_dir": str(stage1_session_dir) if stage1_session_dir is not None else str(stage4_dir / "stage1" / "missing_session"),
        "stage2_dir": str(stage2_dir),
        "stage3a_dir": str(stage3a_dir),
        "stage3b_dir": str(stage3b_dir),
        "stage3c_dir": str(stage3c_dir),
        "stage4_dir": str(stage4_dir),
    }
    notes: list[str] = []
    if not updates:
        notes.append("stage4_world_behavior_updates_missing_or_empty")
    if stage1_session_dir is None:
        notes.append("stage1_session_dir_missing_from_stage4_output")
    return artifacts, source_updates, notes


def artifact_stage_origin(artifact_name: str, benchmark_mode: str) -> str:
    if artifact_name.startswith("stage1_"):
        return "stage1"
    if artifact_name.startswith("stage2_"):
        return "stage2"
    if artifact_name.startswith("stage3a_"):
        return "stage3a"
    if artifact_name.startswith("stage3b_"):
        return "stage3b"
    if artifact_name.startswith("stage3c_"):
        return "stage3c"
    if artifact_name.startswith("stage4_"):
        return "stage4"
    if artifact_name in {"route_session_binding", "behavior_arbitration_events", "critical_actor_binding", "ground_truth_actor_roles"}:
        return "benchmark_semantics"
    return "stage4" if benchmark_mode == "online_e2e" else "benchmark"


def current_run_only(case_run_dir: Path, artifact_path: str | Path) -> bool:
    try:
        Path(artifact_path).resolve().relative_to(case_run_dir.resolve())
        return True
    except Exception:
        return False


def representative_sample_name(case_spec: dict[str, Any], case_run_dir: Path) -> str | None:
    stage4_dir = case_run_dir / "stage4"
    updates = load_jsonl(stage4_dir / "world_behavior_updates.jsonl")
    return _select_sample_name(
        preferred_sample=_preferred_sample_name(case_spec),
        stage4_frames_dir=stage4_dir / "frames",
        records=updates,
    )


def _resolve_stage1_session_dir(stage4_dir: Path) -> Path | None:
    manifest = load_json(stage4_dir / "stage4_manifest.json") or {}
    session_root = manifest.get("stage1_session_root")
    if session_root:
        candidate = Path(session_root)
        if candidate.exists():
            return candidate
    summaries = sorted((stage4_dir / "stage1").glob("*/session_summary.json"))
    if summaries:
        return summaries[0].parent
    return None


def _preferred_sample_name(case_spec: dict[str, Any]) -> str | None:
    artifacts = case_spec.get("artifacts") or {}
    for key in ["stage2_world_state_sample", "stage3a_lane_world_state_sample", "stage3b_behavior_request_sample"]:
        candidate = artifacts.get(key)
        if not candidate:
            continue
        name = Path(str(candidate)).parent.name
        if name.startswith("sample_"):
            return name
    return None


def _select_sample_name(
    *,
    preferred_sample: str | None,
    stage4_frames_dir: Path,
    records: list[dict[str, Any]],
) -> str | None:
    if preferred_sample and (stage4_frames_dir / preferred_sample).exists():
        return preferred_sample
    if records:
        middle_index = len(records) // 2
        candidate = str(records[middle_index].get("sample_name") or "")
        if candidate:
            return candidate
    if stage4_frames_dir.exists():
        frame_dirs = sorted(path.name for path in stage4_frames_dir.iterdir() if path.is_dir())
        if frame_dirs:
            return frame_dirs[min(len(frame_dirs) // 2, len(frame_dirs) - 1)]
    return None


def _select_replay_sample_name(case_run_dir: Path) -> str | None:
    candidate_roots = [
        case_run_dir / "stage3b" / "frames",
        case_run_dir / "stage3a" / "frames",
        case_run_dir / "stage2" / "frames",
    ]
    for root in candidate_roots:
        if not root.exists():
            continue
        frame_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
        if frame_dirs:
            return frame_dirs[min(len(frame_dirs) // 2, len(frame_dirs) - 1)]
    return None


def _write_stage2_bundle(*, stage2_dir: Path, updates: list[dict[str, Any]], selected_sample: str | None, stage4_dir: Path) -> None:
    dump_json(stage2_dir / "evaluation_summary.json", _summarize_stage2_records(updates))
    if selected_sample:
        _copy_if_exists(stage4_dir / "frames" / selected_sample / "world_state.json", stage2_dir / "frames" / selected_sample / "world_state.json")
        _copy_if_exists(stage4_dir / "frames" / selected_sample / "tracked_objects.json", stage2_dir / "frames" / selected_sample / "tracked_objects.json")


def _write_stage3a_bundle(*, stage3a_dir: Path, stage3a_records: list[dict[str, Any]], selected_sample: str | None, stage4_dir: Path) -> None:
    dump_jsonl(stage3a_dir / "behavior_timeline.jsonl", stage3a_records)
    dump_json(stage3a_dir / "evaluation_summary.json", summarize_stage3_session(stage3a_records))
    if selected_sample:
        _copy_if_exists(stage4_dir / "frames" / selected_sample / "lane_aware_world_state.json", stage3a_dir / "frames" / selected_sample / "lane_aware_world_state.json")


def _write_stage3b_bundle(*, stage3b_dir: Path, stage3b_records: list[dict[str, Any]], selected_sample: str | None, stage4_dir: Path) -> None:
    dump_jsonl(stage3b_dir / "behavior_timeline_v2.jsonl", stage3b_records)
    dump_json(stage3b_dir / "evaluation_summary.json", summarize_stage3b_session(stage3b_records))
    if selected_sample:
        _copy_if_exists(stage4_dir / "frames" / selected_sample / "behavior_request_v2.json", stage3b_dir / "frames" / selected_sample / "behavior_request_v2.json")


def _write_stage3c_bundle(*, stage3c_dir: Path, stage4_dir: Path) -> None:
    execution_records = load_jsonl(stage4_dir / "execution_timeline.jsonl")
    lane_change_events = load_jsonl(stage4_dir / "lane_change_events.jsonl")
    online_tick_records = load_jsonl(stage4_dir / "online_tick_record.jsonl")
    online_summary = load_json(stage4_dir / "online_run_summary.json") or {}
    stage3c_summary = online_summary.get("stage3c_execution_summary")
    if not isinstance(stage3c_summary, dict):
        stage3c_summary = summarize_stage3c_session(execution_records, lane_change_events)
    dump_json(stage3c_dir / "execution_summary.json", stage3c_summary)
    dump_jsonl(stage3c_dir / "execution_timeline.jsonl", execution_records)
    dump_jsonl(stage3c_dir / "lane_change_events.jsonl", lane_change_events)
    write_planner_quality_artifacts(
        stage3c_dir,
        execution_records=execution_records,
        raw_lane_change_events=lane_change_events,
        online_tick_records=online_tick_records,
        source_stage="stage4_online",
    )
    _copy_if_exists(stage4_dir / "planner_control_latency.jsonl", stage3c_dir / "planner_control_latency.jsonl")
    _copy_if_exists(stage4_dir / "fallback_events.jsonl", stage3c_dir / "fallback_events.jsonl")
    _copy_if_exists(stage4_dir / "stop_scene_alignment.json", stage3c_dir / "stop_scene_alignment.json")
    _copy_if_exists(stage4_dir / "mpc_shadow_proposal.jsonl", stage3c_dir / "mpc_shadow_proposal.jsonl")
    _copy_if_exists(stage4_dir / "mpc_shadow_summary.json", stage3c_dir / "mpc_shadow_summary.json")
    _copy_if_exists(stage4_dir / "shadow_baseline_comparison.json", stage3c_dir / "shadow_baseline_comparison.json")
    _copy_if_exists(stage4_dir / "shadow_disagreement_events.jsonl", stage3c_dir / "shadow_disagreement_events.jsonl")
    _copy_if_exists(stage4_dir / "shadow_realized_outcome_trace.jsonl", stage3c_dir / "shadow_realized_outcome_trace.jsonl")
    _copy_if_exists(stage4_dir / "shadow_realized_outcome_summary.json", stage3c_dir / "shadow_realized_outcome_summary.json")
    _copy_if_exists(stage4_dir / "shadow_runtime_latency_trace.jsonl", stage3c_dir / "shadow_runtime_latency_trace.jsonl")
    _copy_if_exists(stage4_dir / "shadow_runtime_latency_summary.json", stage3c_dir / "shadow_runtime_latency_summary.json")


def _copy_if_exists(source_path: Path, target_path: Path) -> None:
    if not source_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _summarize_stage2_records(updates: list[dict[str, Any]]) -> dict[str, Any]:
    if not updates:
        return {
            "num_frames": 0,
            "num_decisions": 0,
            "unique_tracks": 0,
            "stable_tracks_3plus_frames": 0,
            "mean_track_length_frames": 0.0,
            "average_active_tracks_per_frame": 0.0,
            "average_front_free_space_m": None,
            "maneuver_counts": {},
            "risk_counts": {},
            "max_route_progress_m": 0.0,
        }

    maneuver_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    active_counts: list[int] = []
    front_free_space_values: list[float] = []
    track_lengths: dict[int, int] = defaultdict(int)
    max_route_progress_m = 0.0

    for item in updates:
        decision = item.get("decision_intent") or {}
        world_state = item.get("world_state") or {}
        objects = world_state.get("objects") or []
        risk_summary = world_state.get("risk_summary") or {}
        scene = world_state.get("scene") or {}
        ego = world_state.get("ego") or {}

        maneuver = decision.get("maneuver")
        if maneuver:
            maneuver_counts[str(maneuver)] += 1
        highest_risk = risk_summary.get("highest_risk_level")
        if highest_risk:
            risk_counts[str(highest_risk)] += 1
        active_counts.append(len(objects))
        front_free_space = scene.get("front_free_space_m")
        if front_free_space is not None:
            front_free_space_values.append(float(front_free_space))
        for track in objects:
            track_id = track.get("track_id")
            if track_id is None:
                continue
            track_lengths[int(track_id)] += 1
        route_progress = ego.get("route_progress_m")
        if route_progress is not None:
            max_route_progress_m = max(max_route_progress_m, float(route_progress))

    mean_track_length = float(sum(track_lengths.values()) / len(track_lengths)) if track_lengths else 0.0
    stable_tracks = sum(1 for length in track_lengths.values() if int(length) >= 3)
    return {
        "num_frames": int(len(updates)),
        "num_decisions": int(len(updates)),
        "unique_tracks": int(len(track_lengths)),
        "stable_tracks_3plus_frames": int(stable_tracks),
        "mean_track_length_frames": float(mean_track_length),
        "average_active_tracks_per_frame": float(sum(active_counts) / len(active_counts)) if active_counts else 0.0,
        "average_front_free_space_m": (float(sum(front_free_space_values) / len(front_free_space_values)) if front_free_space_values else None),
        "maneuver_counts": dict(maneuver_counts),
        "risk_counts": dict(risk_counts),
        "max_route_progress_m": float(max_route_progress_m),
    }


def _to_stage3a_record(item: dict[str, Any]) -> dict[str, Any]:
    decision = dict(item.get("decision_intent") or {})
    behavior_request = dict(item.get("behavior_request") or {})
    maneuver_validation = dict(item.get("maneuver_validation") or {})
    lane_context = dict(item.get("lane_context") or {})
    lane_scene = dict(item.get("lane_scene") or {})
    return {
        "frame_id": int(item.get("frame_id", 0)),
        "timestamp": float(item.get("timestamp", 0.0)),
        "sample_name": str(item.get("sample_name", "")),
        "stage2_decision": decision,
        "planner_interface_payload": dict(item.get("planner_interface_payload") or {}),
        "lane_context": lane_context,
        "lane_scene": lane_scene,
        "lane_relative_objects": list(item.get("lane_relative_objects") or []),
        "maneuver_validation": maneuver_validation,
        "behavior_request": behavior_request,
        "comparison": {
            "frame_id": int(item.get("frame_id", 0)),
            "sample_name": str(item.get("sample_name", "")),
            "stage2_maneuver": decision.get("maneuver"),
            "stage3_behavior_request": behavior_request.get("requested_behavior"),
            "stage3_selected_maneuver": maneuver_validation.get("selected_maneuver"),
            "stage3_overrode_stage2": decision.get("maneuver") != maneuver_validation.get("selected_maneuver"),
            "current_lane_blocked": bool(lane_scene.get("current_lane_blocked")),
            "junction_context": dict(lane_context.get("junction_context") or {}),
        },
        "world_state": dict(item.get("world_state") or {}),
    }


def _to_stage3b_record(item: dict[str, Any]) -> dict[str, Any]:
    decision = dict(item.get("decision_intent") or {})
    behavior_request = dict(item.get("behavior_request") or {})
    behavior_request_v2 = dict(item.get("behavior_request_v2") or {})
    route_context = dict(item.get("route_context") or {})
    behavior_selection = dict(item.get("behavior_selection") or {})
    return {
        "frame_id": int(item.get("frame_id", 0)),
        "timestamp": float(item.get("timestamp", 0.0)),
        "sample_name": str(item.get("sample_name", "")),
        "stage2_decision": decision,
        "stage3a_behavior_request": behavior_request,
        "maneuver_validation": dict(item.get("maneuver_validation") or {}),
        "lane_context": dict(item.get("lane_context") or {}),
        "lane_scene": dict(item.get("lane_scene") or {}),
        "lane_relative_objects": list(item.get("lane_relative_objects") or []),
        "route_context": route_context,
        "route_conditioned_scene": dict(item.get("route_conditioned_scene") or {}),
        "behavior_selection": behavior_selection,
        "behavior_request_v2": behavior_request_v2,
        "comparison": {
            "frame_id": int(item.get("frame_id", 0)),
            "sample_name": str(item.get("sample_name", "")),
            "stage2_maneuver": decision.get("maneuver"),
            "stage3a_behavior": behavior_request.get("requested_behavior"),
            "stage3b_behavior": behavior_request_v2.get("requested_behavior"),
            "route_option": route_context.get("route_option"),
            "lane_change_stage": behavior_selection.get("lane_change_stage"),
            "stage3b_overrode_stage3a": behavior_request.get("requested_behavior") != behavior_request_v2.get("requested_behavior"),
            "route_influenced": bool(behavior_selection.get("route_influence")),
            "target_lane_changed": str(behavior_request.get("target_lane", "current")) != str(behavior_request_v2.get("target_lane")),
        },
        "world_state": dict(item.get("world_state") or {}),
    }
