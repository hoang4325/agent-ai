from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from stage3c_coverage.planner_quality import write_planner_quality_artifacts

from .io import dump_json, dump_jsonl, load_json, load_jsonl, resolve_repo_path
from .semantic_artifacts import ensure_semantic_artifacts


BUNDLE_VERSION = "frozen_case_bundle_v1"
SCHEMA_VERSION_PREFIX = f"{BUNDLE_VERSION}.schema"
SEMANTIC_ARTIFACT_NAMES = [
    "ground_truth_actor_roles",
    "critical_actor_binding",
    "behavior_arbitration_events",
    "route_session_binding",
]
WORKSPACE_AGENT_PREFIX = "/workspace/Agent-AI"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel_repo_path(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def resolve_existing_path(repo_root: Path, candidate: str | Path | None) -> Path | None:
    if candidate in {None, ""}:
        return None
    path = Path(str(candidate))
    if path.exists():
        return path
    normalized = str(candidate).replace("\\", "/")
    if normalized.startswith(WORKSPACE_AGENT_PREFIX):
        mapped = repo_root / normalized[len(WORKSPACE_AGENT_PREFIX) :].lstrip("/")
        if mapped.exists():
            return mapped
    resolved = resolve_repo_path(repo_root, candidate)
    if resolved and resolved.exists():
        return resolved
    return None


def _copy_json_tree(
    source_dir: Path | None,
    target_dir: Path,
    *,
    excluded_dir_names: set[str] | None = None,
) -> dict[str, Any]:
    excluded = excluded_dir_names or {"visualization", "video", "camera_overlays"}
    copied_files = 0
    missing = source_dir is None or not source_dir.exists()
    if missing:
        return {"copied_files": copied_files, "missing": True}

    for src_path in source_dir.rglob("*"):
        if any(part in excluded for part in src_path.parts):
            continue
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        relative = src_path.relative_to(source_dir)
        dst_path = target_dir / relative
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied_files += 1
    return {"copied_files": copied_files, "missing": False}


def _copy_single_file(source_path: Path | None, target_path: Path) -> bool:
    if source_path is None or not source_path.exists():
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return True


def _rewrite_session_manifest(session_manifest: dict[str, Any], source_session_dir: Path) -> dict[str, Any]:
    payload = dict(session_manifest or {})
    payload["session_root"] = "."
    payload["source_samples_root"] = "../materialized_samples"
    payload["inference_root"] = "inference"
    payload["source_session_root"] = str(source_session_dir)
    args = dict(payload.get("args") or {})
    if "output_root" in args:
        args["output_root"] = "."
    if "samples_root" in args:
        args["samples_root"] = "../materialized_samples"
    payload["args"] = args
    return payload


def _local_sample_dir(sample_name: str) -> str:
    return PurePosixPath("..", "materialized_samples", sample_name).as_posix()


def _local_prediction_dir(sample_name: str) -> str:
    return PurePosixPath("inference", sample_name).as_posix()


def materialize_stage1_frozen(
    *,
    repo_root: Path,
    source_session_dir: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    session_dir = output_dir / "session"
    materialized_samples_dir = output_dir / "materialized_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_session_dir is None or not source_session_dir.exists():
        return {
            "available": False,
            "session_dir": None,
            "infer_ready_samples": 0,
            "total_live_records": 0,
            "missing_meta_samples": [],
            "copied_json_files": 0,
        }

    copy_stats = _copy_json_tree(source_session_dir, session_dir, excluded_dir_names={"camera_overlays"})

    live_records = load_jsonl(source_session_dir / "live_summary.jsonl")
    rewritten_records: list[dict[str, Any]] = []
    missing_meta_samples: list[str] = []
    infer_ready_samples = 0

    for record in live_records:
        payload = dict(record)
        sample_name = str(payload.get("sample_name", ""))
        source_sample_dir = resolve_existing_path(repo_root, payload.get("source_sample_dir"))
        if sample_name and source_sample_dir is not None:
            copied = _copy_single_file(
                source_sample_dir / "meta.json",
                materialized_samples_dir / sample_name / "meta.json",
            )
            if not copied:
                missing_meta_samples.append(sample_name)
            payload["source_sample_dir"] = _local_sample_dir(sample_name)

        if payload.get("status") in {"inferred", "inferred_compare"} and sample_name:
            infer_ready_samples += 1
        if payload.get("output_dir") and sample_name:
            payload["output_dir"] = _local_prediction_dir(sample_name)
        if payload.get("comparison_output_dir") and sample_name:
            payload["comparison_output_dir"] = _local_prediction_dir(sample_name)

        rewritten_records.append(payload)

    if rewritten_records:
        dump_jsonl(session_dir / "live_summary.jsonl", rewritten_records)

    session_manifest = load_json(source_session_dir / "session_manifest.json", default={})
    if session_manifest:
        dump_json(session_dir / "session_manifest.json", _rewrite_session_manifest(session_manifest, source_session_dir))

    stage1_summary = load_json(source_session_dir / "session_summary.json", default={})
    stage1_provenance = {
        "schema_version": f"{SCHEMA_VERSION_PREFIX}.stage1_provenance",
        "materialized_at": utc_now_iso(),
        "source_session_dir": rel_repo_path(repo_root, source_session_dir),
        "stage1_run_id": source_session_dir.name,
        "session_summary": stage1_summary,
        "session_manifest_args": (session_manifest.get("args") or {}),
        "model_runtime": session_manifest.get("model_runtime") or {},
        "sample_payload_scope": "meta_only",
        "notes": [
            "live_summary_paths_rewritten_relative_to_session_root",
            "raw_sensor_blobs_not_copied_into_frozen_bundle",
        ],
    }
    dump_json(output_dir / "perception_provenance.json", stage1_provenance)

    return {
        "available": True,
        "session_dir": str((output_dir / "session").resolve()),
        "infer_ready_samples": infer_ready_samples,
        "total_live_records": len(rewritten_records),
        "missing_meta_samples": sorted(set(missing_meta_samples)),
        "copied_json_files": int(copy_stats.get("copied_files", 0)),
        "session_summary": stage1_summary,
        "session_manifest": _rewrite_session_manifest(session_manifest, source_session_dir),
        "live_records": rewritten_records,
    }


def _load_manifest_if_exists(base_dir: Path | None, filename: str) -> dict[str, Any]:
    if base_dir is None:
        return {}
    return load_json(base_dir / filename, default={})


def _derive_runner_manifest_payload(
    *,
    repo_root: Path,
    case_spec: dict[str, Any],
    stage1_source_dir: Path | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    source = dict(case_spec.get("source") or {})
    route_profile = dict(case_spec.get("route_profile") or {})
    stage3a_dir = resolve_existing_path(repo_root, source.get("stage3a_dir"))
    stage3b_dir = resolve_existing_path(repo_root, source.get("stage3b_dir"))
    stage3c_dir = resolve_existing_path(repo_root, source.get("stage3c_dir"))

    stage1_manifest = _load_manifest_if_exists(stage1_source_dir, "session_manifest.json")
    stage3a_manifest = _load_manifest_if_exists(stage3a_dir, "stage3_manifest.json")
    stage3b_manifest = _load_manifest_if_exists(stage3b_dir, "stage3b_manifest.json")
    stage3c_manifest = _load_manifest_if_exists(stage3c_dir, "stage3c_manifest.json")

    town = stage3a_manifest.get("town") or ((stage1_manifest.get("args") or {}).get("town"))
    ego_actor_id = stage3c_manifest.get("attach_to_actor_id")
    if town in {None, ""} and ego_actor_id in {None, 0, "0"}:
        return None, ["runner_manifest_derivation_insufficient_inputs"]

    notes = [
        "derived_runner_manifest_from_stage1_stage3_manifests",
        f"stage3a_manifest_available={bool(stage3a_manifest)}",
        f"stage3b_manifest_available={bool(stage3b_manifest)}",
        f"stage3c_manifest_available={bool(stage3c_manifest)}",
    ]
    payload = {
        "town": town or "Carla/Maps/Town10HD_Opt",
        "host": stage3c_manifest.get("carla_host") or (stage1_manifest.get("args") or {}).get("host") or "127.0.0.1",
        "port": stage3c_manifest.get("carla_port") or (stage1_manifest.get("args") or {}).get("port") or 2000,
        "tm_port": (stage1_manifest.get("args") or {}).get("tm_port") or 8000,
        "ego_actor_id": int(ego_actor_id or 0),
        "blocker_actor_id": None,
        "adjacent_actor_id": None,
        "adjacent_side": None,
        "scenario_type": f"derived_{case_spec.get('family', 'scenario')}_replay",
        "corridor": None,
        "placements": None,
        "route_option": route_profile.get("route_option") or case_spec.get("route_option"),
        "preferred_lane": route_profile.get("preferred_lane") or stage3b_manifest.get("preferred_lane"),
        "route_priority": route_profile.get("route_priority") or stage3b_manifest.get("route_priority") or "normal",
        "derivation": {
            "source": "derived_runner_manifest_v1",
            "stage1_session_manifest": rel_repo_path(repo_root, stage1_source_dir / "session_manifest.json") if stage1_source_dir else None,
            "stage3a_manifest": rel_repo_path(repo_root, stage3a_dir / "stage3_manifest.json") if stage3a_dir else None,
            "stage3b_manifest": rel_repo_path(repo_root, stage3b_dir / "stage3b_manifest.json") if stage3b_dir else None,
            "stage3c_manifest": rel_repo_path(repo_root, stage3c_dir / "stage3c_manifest.json") if stage3c_dir else None,
        },
    }
    return payload, notes


def _derive_semantic_reference(
    *,
    repo_root: Path,
    case_spec: dict[str, Any],
    bundle_dir: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    generated_root = bundle_dir / "_derived_semantics"
    semantic_paths, semantic_metadata = ensure_semantic_artifacts(
        case_spec=case_spec,
        repo_root=repo_root,
        generated_root=generated_root,
    )
    semantic_dir = bundle_dir / "reference_outputs" / "semantic"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    provenance_rows: list[dict[str, Any]] = []
    for artifact_name in SEMANTIC_ARTIFACT_NAMES:
        source_path = semantic_paths.get(artifact_name)
        source = Path(source_path) if source_path else None
        default_names = {
            "ground_truth_actor_roles": "ground_truth_actor_roles.json",
            "critical_actor_binding": "critical_actor_binding.jsonl",
            "behavior_arbitration_events": "behavior_arbitration_events.jsonl",
            "route_session_binding": "route_session_binding.json",
        }
        target_name = source.name if source else default_names[artifact_name]
        target_path = semantic_dir / target_name
        copied_ok = _copy_single_file(source, target_path)
        copied[artifact_name] = str(target_path) if copied_ok else ""
        metadata = semantic_metadata.get(artifact_name)
        provenance_rows.append(
            {
                "artifact_name": artifact_name,
                "source_path": source_path,
                "bundled_path": str(target_path) if copied_ok else None,
                "available": copied_ok,
                "source_type": metadata.source if metadata else "missing",
                "notes": list((metadata.notes if metadata else []) or []),
            }
        )
    shutil.rmtree(generated_root, ignore_errors=True)
    return copied, provenance_rows


def _copy_reference_stage(stage_name: str, source_dir: Path | None, reference_root: Path) -> dict[str, Any]:
    stage_root = reference_root / stage_name
    stats = _copy_json_tree(source_dir, stage_root)
    derived_artifacts: list[str] = []
    if stage_name == "stage3c" and not stats["missing"]:
        write_planner_quality_artifacts(stage_root)
        derived_artifacts = sorted(
            [
                "lane_change_events_enriched.jsonl",
                "stop_events.jsonl",
                "trajectory_trace.jsonl",
                "control_trace.jsonl",
                "lane_change_phase_trace.jsonl",
                "stop_target.json",
                "planner_control_latency.jsonl",
                "fallback_events.jsonl",
                "planner_quality_summary.json",
            ]
        )
    return {
        "stage_name": stage_name,
        "available": not stats["missing"],
        "bundled_dir": str(stage_root.resolve()) if not stats["missing"] else None,
        "copied_json_files": int(stats["copied_files"]),
        "source_dir": str(source_dir) if source_dir else None,
        "derived_artifacts": derived_artifacts,
    }


def build_scenario_manifest(
    *,
    case_spec: dict[str, Any],
    runner_manifest_payload: dict[str, Any],
    runner_manifest_path: str | None,
    route_session_binding: dict[str, Any],
    ground_truth_actor_roles: dict[str, Any],
    stage1_info: dict[str, Any],
) -> dict[str, Any]:
    stage1_summary = stage1_info.get("session_summary") or {}
    live_records = stage1_info.get("live_records") or []
    infer_ready_records = [item for item in live_records if item.get("status") in {"inferred", "inferred_compare"}]
    evaluation_window = {
        "observed_samples": int(stage1_summary.get("observed_samples", len(live_records))),
        "history_skipped_samples": int(stage1_summary.get("history_skipped_samples", 0)),
        "infer_ready_samples": int(stage1_info.get("infer_ready_samples", len(infer_ready_records))),
        "first_infer_ready_sample": infer_ready_records[0]["sample_name"] if infer_ready_records else None,
        "last_infer_ready_sample": infer_ready_records[-1]["sample_name"] if infer_ready_records else None,
    }
    route_profile = dict(case_spec.get("route_profile") or {})
    route_profile.setdefault("route_option", case_spec.get("route_option"))

    critical_roles = [
        str(actor.get("role"))
        for actor in (ground_truth_actor_roles.get("actors") or [])
        if actor.get("role") is not None
    ] or [str(item) for item in (case_spec.get("critical_actors") or [])]

    return {
        "schema_version": f"{SCHEMA_VERSION_PREFIX}.scenario_manifest",
        "bundle_version": BUNDLE_VERSION,
        "case_id": str(case_spec.get("scenario_id")),
        "scenario_id": str(case_spec.get("scenario_id")),
        "town": runner_manifest_payload.get("town"),
        "scenario_recipe": {
            "source_type": ((case_spec.get("scenario_recipe") or {}).get("source_type")),
            "attach_strategy": ((case_spec.get("scenario_recipe") or {}).get("attach_strategy")),
            "runner_manifest_path": runner_manifest_path,
            "scenario_type": runner_manifest_payload.get("scenario_type"),
            "spawn_recipe": {
                "ego_actor_id": runner_manifest_payload.get("ego_actor_id"),
                "blocker_actor_id": runner_manifest_payload.get("blocker_actor_id"),
                "adjacent_actor_id": runner_manifest_payload.get("adjacent_actor_id"),
                "adjacent_side": runner_manifest_payload.get("adjacent_side"),
                "corridor": runner_manifest_payload.get("corridor"),
                "placements": runner_manifest_payload.get("placements"),
            },
        },
        "route_profile": route_profile,
        "route_session_binding": route_session_binding,
        "deterministic_seed": runner_manifest_payload.get("deterministic_seed"),
        "evaluation_window": evaluation_window,
        "critical_actor_roles": critical_roles,
        "expected_behaviors": list(case_spec.get("expected_behaviors") or []),
        "forbidden_behaviors": list(case_spec.get("forbidden_behaviors") or []),
        "expected_execution_outcomes": dict(case_spec.get("expected_execution") or {}),
    }


def build_benchmark_reference(
    *,
    case_spec: dict[str, Any],
    artifact_paths: dict[str, str],
    readiness: str,
) -> dict[str, Any]:
    status = str(case_spec.get("status", "planned"))
    replay_gate = "golden" if status == "ready" else ("smoke" if status == "provisional" else "diagnostic_only")
    return {
        "schema_version": f"{SCHEMA_VERSION_PREFIX}.benchmark_reference",
        "bundle_version": BUNDLE_VERSION,
        "case_id": str(case_spec.get("scenario_id")),
        "readiness": readiness,
        "primary_metrics": list(case_spec.get("primary_metrics") or []),
        "secondary_metrics": list(case_spec.get("secondary_metrics") or []),
        "metric_thresholds": dict(case_spec.get("metric_thresholds") or {}),
        "mandatory_artifacts": list(case_spec.get("mandatory_artifacts") or []),
        "required_artifacts": {
            "frozen_input": [
                "scenario_artifacts/runner_manifest.json",
                "stage1_frozen/session/session_summary.json",
                "stage1_frozen/session/live_summary.jsonl",
            ],
            "reference_outputs": sorted(
                name for name in artifact_paths if name.startswith("stage2_") or name.startswith("stage3") or name in SEMANTIC_ARTIFACT_NAMES
            ),
        },
        "gate_profile": {
            "recommended_replay_gate": replay_gate,
            "recommended_scenario_replay_gate": "scenario_replay_smoke"
            if "scenario_replay" in (case_spec.get("supported_benchmark_modes") or [])
            else "diagnostic_only",
            "recommended_online_gate": "online_e2e_smoke"
            if "online_e2e" in (case_spec.get("supported_benchmark_modes") or [])
            else "diagnostic_only",
        },
        "benchmark_tags": sorted(
            {
                str(case_spec.get("family", "unknown")),
                str(case_spec.get("status", "planned")),
                str(case_spec.get("route_option", "unknown")),
                str(case_spec.get("determinism_level", "unknown")),
                readiness,
                "frozen_input_stage1",
                "reference_outputs_stage2_3c",
            }
        ),
        "artifact_paths": artifact_paths,
    }


def _remap_reference_artifact(
    *,
    repo_root: Path,
    source_artifact: str | None,
    source_dir: Path | None,
    bundled_dir: Path,
    fallback_relative: str,
) -> str:
    resolved_artifact = resolve_existing_path(repo_root, source_artifact)
    if resolved_artifact is not None and source_dir is not None:
        try:
            relative = resolved_artifact.relative_to(source_dir)
            return str((bundled_dir / relative).resolve())
        except Exception:
            pass
    return str((bundled_dir / Path(fallback_relative)).resolve())


def determine_case_readiness(
    *,
    case_spec: dict[str, Any],
    runner_manifest_available: bool,
    stage1_info: dict[str, Any],
    stage_reference_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, bool], list[str]]:
    case_status = str(case_spec.get("status", "planned"))
    checks = {
        "case_status_ready": case_status == "ready",
        "runner_scenario_manifest_available": runner_manifest_available,
        "stage1_frozen_replayable": bool(stage1_info.get("available")) and int(stage1_info.get("infer_ready_samples", 0)) > 0,
        "reference_outputs_complete": all(bool(row.get("available")) for row in stage_reference_rows),
        "semantic_reference_complete": all(bool(row.get("available")) for row in semantic_rows),
        "provenance_complete": True,
        "pending_ground_truth_binding": bool(case_spec.get("pending_ground_truth_binding", False)),
    }
    warnings: list[str] = []
    if stage1_info.get("missing_meta_samples"):
        warnings.append(f"stage1_missing_meta_samples:{len(stage1_info['missing_meta_samples'])}")
    if not checks["runner_scenario_manifest_available"]:
        warnings.append("runner_scenario_manifest_missing")
    if not checks["reference_outputs_complete"]:
        missing = [row["stage_name"] for row in stage_reference_rows if not row.get("available")]
        warnings.append("missing_reference_outputs:" + ",".join(missing))
    if not checks["semantic_reference_complete"]:
        missing = [row["artifact_name"] for row in semantic_rows if not row.get("available")]
        warnings.append("missing_semantic_reference:" + ",".join(missing))
    if checks["pending_ground_truth_binding"]:
        warnings.append("pending_ground_truth_binding_true")

    if all(
        [
            checks["case_status_ready"],
            checks["runner_scenario_manifest_available"],
            checks["stage1_frozen_replayable"],
            checks["reference_outputs_complete"],
            checks["semantic_reference_complete"],
            not checks["pending_ground_truth_binding"],
        ]
    ):
        return "corpus_ready", checks, warnings

    if case_status == "planned" and not checks["reference_outputs_complete"] and not checks["stage1_frozen_replayable"]:
        return "corpus_pending", checks, warnings

    if not checks["runner_scenario_manifest_available"] and not checks["stage1_frozen_replayable"] and case_status == "planned":
        return "corpus_pending", checks, warnings

    return "corpus_partial", checks, warnings


def materialize_case_bundle(
    *,
    repo_root: Path,
    benchmark_config: dict[str, Any],
    case_spec: dict[str, Any],
    bundle_dir: Path,
) -> dict[str, Any]:
    case_id = str(case_spec.get("scenario_id"))
    source = dict(case_spec.get("source") or {})
    artifacts = dict(case_spec.get("artifacts") or {})
    scenario_recipe = dict(case_spec.get("scenario_recipe") or {})
    bundle_dir.mkdir(parents=True, exist_ok=True)

    stage1_source_dir = resolve_existing_path(
        repo_root,
        scenario_recipe.get("stage1_session_dir") or source.get("stage1_session_dir"),
    )

    runner_manifest_source = resolve_existing_path(
        repo_root,
        scenario_recipe.get("scenario_manifest") or source.get("scenario_manifest") or artifacts.get("scenario_manifest"),
    )
    runner_manifest_target = bundle_dir / "scenario_artifacts" / "runner_manifest.json"
    runner_manifest_notes: list[str] = []
    if runner_manifest_source is not None and _copy_single_file(runner_manifest_source, runner_manifest_target):
        runner_manifest_available = True
        runner_manifest_payload = load_json(runner_manifest_target, default={})
        runner_manifest_notes.append("copied_existing_runner_manifest")
    else:
        derived_manifest_payload, derived_notes = _derive_runner_manifest_payload(
            repo_root=repo_root,
            case_spec=case_spec,
            stage1_source_dir=stage1_source_dir,
        )
        if derived_manifest_payload:
            dump_json(runner_manifest_target, derived_manifest_payload)
            runner_manifest_available = True
            runner_manifest_payload = derived_manifest_payload
            runner_manifest_notes.extend(derived_notes)
        else:
            runner_manifest_available = False
            runner_manifest_payload = {}
            runner_manifest_notes.extend(derived_notes)
    stage1_info = materialize_stage1_frozen(
        repo_root=repo_root,
        source_session_dir=stage1_source_dir,
        output_dir=bundle_dir / "stage1_frozen",
    )

    reference_root = bundle_dir / "reference_outputs"
    stage_reference_rows = [
        _copy_reference_stage("stage2", resolve_existing_path(repo_root, source.get("stage2_dir")), reference_root),
        _copy_reference_stage("stage3a", resolve_existing_path(repo_root, source.get("stage3a_dir")), reference_root),
        _copy_reference_stage("stage3b", resolve_existing_path(repo_root, source.get("stage3b_dir")), reference_root),
        _copy_reference_stage("stage3c", resolve_existing_path(repo_root, source.get("stage3c_dir")), reference_root),
    ]

    semantic_paths, semantic_rows = _derive_semantic_reference(
        repo_root=repo_root,
        case_spec=case_spec,
        bundle_dir=bundle_dir,
    )
    route_session_binding = load_json(semantic_paths.get("route_session_binding"), default={})
    ground_truth_actor_roles = load_json(semantic_paths.get("ground_truth_actor_roles"), default={})

    readiness, checks, readiness_warnings = determine_case_readiness(
        case_spec=case_spec,
        runner_manifest_available=runner_manifest_available,
        stage1_info=stage1_info,
        stage_reference_rows=stage_reference_rows,
        semantic_rows=semantic_rows,
    )

    scenario_manifest_payload = build_scenario_manifest(
        case_spec=case_spec,
        runner_manifest_payload=runner_manifest_payload,
        runner_manifest_path="scenario_artifacts/runner_manifest.json" if runner_manifest_available else None,
        route_session_binding=route_session_binding,
        ground_truth_actor_roles=ground_truth_actor_roles,
        stage1_info=stage1_info,
    )
    dump_json(bundle_dir / "scenario_manifest.json", scenario_manifest_payload)

    artifact_paths = {
        **{name: path for name, path in semantic_paths.items() if path},
        "stage1_session_summary": str((bundle_dir / "stage1_frozen" / "session" / "session_summary.json").resolve()),
        "stage2_evaluation_summary": str((bundle_dir / "reference_outputs" / "stage2" / "evaluation_summary.json").resolve()),
        "stage2_world_state_sample": _remap_reference_artifact(
            repo_root=repo_root,
            source_artifact=artifacts.get("stage2_world_state_sample"),
            source_dir=resolve_existing_path(repo_root, source.get("stage2_dir")),
            bundled_dir=bundle_dir / "reference_outputs" / "stage2",
            fallback_relative="frames",
        ),
        "stage2_tracked_objects_sample": _remap_reference_artifact(
            repo_root=repo_root,
            source_artifact=artifacts.get("stage2_tracked_objects_sample"),
            source_dir=resolve_existing_path(repo_root, source.get("stage2_dir")),
            bundled_dir=bundle_dir / "reference_outputs" / "stage2",
            fallback_relative="frames",
        ),
        "stage3a_evaluation_summary": str((bundle_dir / "reference_outputs" / "stage3a" / "evaluation_summary.json").resolve()),
        "stage3a_lane_world_state_sample": _remap_reference_artifact(
            repo_root=repo_root,
            source_artifact=artifacts.get("stage3a_lane_world_state_sample"),
            source_dir=resolve_existing_path(repo_root, source.get("stage3a_dir")),
            bundled_dir=bundle_dir / "reference_outputs" / "stage3a",
            fallback_relative="frames",
        ),
        "stage3b_evaluation_summary": str((bundle_dir / "reference_outputs" / "stage3b" / "evaluation_summary.json").resolve()),
        "stage3b_behavior_timeline": str((bundle_dir / "reference_outputs" / "stage3b" / "behavior_timeline_v2.jsonl").resolve()),
        "stage3b_behavior_request_sample": _remap_reference_artifact(
            repo_root=repo_root,
            source_artifact=artifacts.get("stage3b_behavior_request_sample"),
            source_dir=resolve_existing_path(repo_root, source.get("stage3b_dir")),
            bundled_dir=bundle_dir / "reference_outputs" / "stage3b",
            fallback_relative="frames",
        ),
        "stage3c_execution_summary": str((bundle_dir / "reference_outputs" / "stage3c" / "execution_summary.json").resolve()),
        "stage3c_execution_timeline": str((bundle_dir / "reference_outputs" / "stage3c" / "execution_timeline.jsonl").resolve()),
        "stage3c_lane_change_events": str((bundle_dir / "reference_outputs" / "stage3c" / "lane_change_events.jsonl").resolve()),
        "stage3c_lane_change_events_enriched": str((bundle_dir / "reference_outputs" / "stage3c" / "lane_change_events_enriched.jsonl").resolve()),
        "stage3c_stop_events": str((bundle_dir / "reference_outputs" / "stage3c" / "stop_events.jsonl").resolve()),
        "stage3c_trajectory_trace": str((bundle_dir / "reference_outputs" / "stage3c" / "trajectory_trace.jsonl").resolve()),
        "stage3c_control_trace": str((bundle_dir / "reference_outputs" / "stage3c" / "control_trace.jsonl").resolve()),
        "stage3c_lane_change_phase_trace": str((bundle_dir / "reference_outputs" / "stage3c" / "lane_change_phase_trace.jsonl").resolve()),
        "stage3c_stop_target": str((bundle_dir / "reference_outputs" / "stage3c" / "stop_target.json").resolve()),
        "stage3c_planner_control_latency": str((bundle_dir / "reference_outputs" / "stage3c" / "planner_control_latency.jsonl").resolve()),
        "stage3c_fallback_events": str((bundle_dir / "reference_outputs" / "stage3c" / "fallback_events.jsonl").resolve()),
        "stage3c_planner_quality_summary": str((bundle_dir / "reference_outputs" / "stage3c" / "planner_quality_summary.json").resolve()),
    }

    benchmark_reference = build_benchmark_reference(
        case_spec=case_spec,
        artifact_paths={name: str(Path(path)) for name, path in artifact_paths.items() if path},
        readiness=readiness,
    )
    dump_json(bundle_dir / "benchmark_reference" / "benchmark_reference.json", benchmark_reference)

    semantic_hardening_version = (
        route_session_binding.get("schema_version")
        or ground_truth_actor_roles.get("schema_version")
        or "system_benchmark_semantics_v1_1"
    )
    provenance = {
        "schema_version": f"{SCHEMA_VERSION_PREFIX}.provenance",
        "bundle_version": BUNDLE_VERSION,
        "case_id": case_id,
        "build": {
            "built_at": utc_now_iso(),
            "tool": "scripts/build_frozen_corpus.py",
            "bundle_dir": rel_repo_path(repo_root, bundle_dir),
        },
        "source_case_path": case_spec.get("_case_path"),
        "source_artifact_paths": {
            key: value
            for key, value in {
                **source,
                **artifacts,
                "scenario_recipe.stage1_session_dir": scenario_recipe.get("stage1_session_dir"),
                "scenario_recipe.scenario_manifest": scenario_recipe.get("scenario_manifest"),
            }.items()
            if value not in {None, ""}
        },
        "source_run_ids": {
            "stage1": Path(str(source.get("stage1_session_dir", ""))).name if source.get("stage1_session_dir") else None,
            "stage2": Path(str(source.get("stage2_dir", ""))).name if source.get("stage2_dir") else None,
            "stage3a": Path(str(source.get("stage3a_dir", ""))).name if source.get("stage3a_dir") else None,
            "stage3b": Path(str(source.get("stage3b_dir", ""))).name if source.get("stage3b_dir") else None,
            "stage3c": Path(str(source.get("stage3c_dir", ""))).name if source.get("stage3c_dir") else None,
        },
        "source_baseline": dict(benchmark_config.get("baseline") or {}),
        "semantic_hardening_version": semantic_hardening_version,
        "trace_lineage": {
            "primary_input": "scenario_manifest + stage1_frozen_session",
            "reference_outputs": ["stage2", "stage3a", "stage3b", "stage3c", "semantic"],
            "online_smoke_role": "not_primary_input",
        },
        "runner_manifest": {
            "available": runner_manifest_available,
            "origin": "existing" if runner_manifest_source else ("derived" if runner_manifest_available else "missing"),
            "notes": runner_manifest_notes,
        },
        "artifact_lineage": [
            {
                "artifact_name": row["stage_name"],
                "stage_origin": row["stage_name"],
                "source_path": row["source_dir"],
                "bundled_path": row["bundled_dir"],
                "available": row["available"],
                "role": "reference_output",
            }
            for row in stage_reference_rows
        ]
        + [
            {
                "artifact_name": row["artifact_name"],
                "stage_origin": "benchmark_semantics",
                "source_path": row["source_path"],
                "bundled_path": row["bundled_path"],
                "available": row["available"],
                "role": "reference_output",
                "source_type": row["source_type"],
                "notes": row["notes"],
            }
            for row in semantic_rows
        ],
        "warnings": readiness_warnings,
    }
    dump_json(bundle_dir / "provenance" / "provenance.json", provenance)

    case_manifest = {
        "schema_version": f"{SCHEMA_VERSION_PREFIX}.case_manifest",
        "bundle_version": BUNDLE_VERSION,
        "case_id": case_id,
        "scenario_id": case_id,
        "family": case_spec.get("family"),
        "status": case_spec.get("status"),
        "corpus_readiness": readiness,
        "determinism_level": case_spec.get("determinism_level", "unknown"),
        "source": source.get("type", "unknown"),
        "owner": case_spec.get("owner"),
        "case_version": case_spec.get("benchmark_spec_version", "benchmark_case_v1"),
        "build_timestamp": utc_now_iso(),
        "supported_benchmark_modes": list(case_spec.get("supported_benchmark_modes") or [case_spec.get("benchmark_mode", "replay_regression")]),
        "pending_ground_truth_binding": bool(case_spec.get("pending_ground_truth_binding", False)),
        "readiness_checks": checks,
        "bundle_paths": {
            "scenario_manifest": "scenario_manifest.json",
            "runner_scenario_manifest": "scenario_artifacts/runner_manifest.json" if runner_manifest_available else None,
            "stage1_session_dir": "stage1_frozen/session" if stage1_info.get("available") else None,
            "benchmark_reference": "benchmark_reference/benchmark_reference.json",
            "provenance": "provenance/provenance.json",
        },
        "warnings": readiness_warnings,
    }
    dump_json(bundle_dir / "case_manifest.json", case_manifest)

    return {
        "case_id": case_id,
        "bundle_dir": str(bundle_dir.resolve()),
        "readiness": readiness,
        "case_manifest": case_manifest,
        "scenario_manifest_payload": scenario_manifest_payload,
        "benchmark_reference": benchmark_reference,
        "provenance": provenance,
        "bundled_artifacts": artifact_paths,
        "stage1_info": stage1_info,
        "stage_reference_rows": stage_reference_rows,
        "semantic_rows": semantic_rows,
    }
