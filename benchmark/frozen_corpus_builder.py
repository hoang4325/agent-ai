from __future__ import annotations

from pathlib import Path
from typing import Any

from .frozen_case_materializer import materialize_case_bundle, rel_repo_path, resolve_existing_path, utc_now_iso
from .frozen_corpus_validate import validate_corpus
from .io import dump_json, load_yaml, resolve_repo_path


def _load_case_spec(case_path: Path) -> dict[str, Any]:
    payload = load_yaml(case_path)
    payload["_case_path"] = str(case_path)
    return payload


def _benchmark_only_global(repo_root: Path, benchmark_config: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = benchmark_config.get("baseline") or {}
    return [
        {
            "name": "baseline_metrics",
            "path": baseline.get("default_metrics_path"),
            "exists": bool(resolve_existing_path(repo_root, baseline.get("default_metrics_path"))),
        },
        {
            "name": "baseline_commit",
            "path": baseline.get("default_commit_path"),
            "exists": bool(resolve_existing_path(repo_root, baseline.get("default_commit_path"))),
        },
        {
            "name": "benchmark_modes",
            "path": benchmark_config.get("benchmark_modes"),
            "exists": bool(resolve_existing_path(repo_root, benchmark_config.get("benchmark_modes"))),
        },
        {
            "name": "metric_registry",
            "path": benchmark_config.get("metric_registry"),
            "exists": bool(resolve_existing_path(repo_root, benchmark_config.get("metric_registry"))),
        },
    ]


def _scan_online_stage4_candidates(repo_root: Path) -> list[dict[str, Any]]:
    report_root = repo_root / "benchmark" / "reports"
    candidates: list[dict[str, Any]] = []
    if not report_root.exists():
        return candidates
    for stage4_dir in sorted(report_root.glob("e2e_online_*/case_runs/*/stage4")):
        case_id = stage4_dir.parent.name
        candidates.append(
            {
                "case_id": case_id,
                "path": rel_repo_path(repo_root, stage4_dir),
                "available": True,
            }
        )
    return candidates


def audit_case_data(repo_root: Path, case_spec: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case_spec.get("scenario_id"))
    source = dict(case_spec.get("source") or {})
    scenario_recipe = dict(case_spec.get("scenario_recipe") or {})
    artifacts = dict(case_spec.get("artifacts") or {})

    frozen_input = []
    for name, candidate, notes in [
        (
            "scenario_manifest",
            scenario_recipe.get("scenario_manifest") or source.get("scenario_manifest") or artifacts.get("scenario_manifest"),
            ["runner_input", "scenario_spawn_attach_contract"],
        ),
        (
            "stage1_session_dir",
            scenario_recipe.get("stage1_session_dir") or source.get("stage1_session_dir"),
            ["primary_future_input", "stage1_frozen_candidate"],
        ),
    ]:
        path = resolve_existing_path(repo_root, candidate)
        frozen_input.append(
            {
                "name": name,
                "path": rel_repo_path(repo_root, path) if path else candidate,
                "exists": bool(path and path.exists()),
                "notes": notes,
            }
        )

    reference_output = []
    for name, candidate in [
        ("stage2_dir", source.get("stage2_dir")),
        ("stage3a_dir", source.get("stage3a_dir")),
        ("stage3b_dir", source.get("stage3b_dir")),
        ("stage3c_dir", source.get("stage3c_dir")),
    ]:
        path = resolve_existing_path(repo_root, candidate)
        reference_output.append(
            {
                "name": name,
                "path": rel_repo_path(repo_root, path) if path else candidate,
                "exists": bool(path and path.exists()),
                "notes": ["reference_output", "baseline_for_regression"],
            }
        )
    for artifact_name in [
        "ground_truth_actor_roles",
        "critical_actor_binding",
        "behavior_arbitration_events",
        "route_session_binding",
        "stage1_session_summary",
        "stage2_evaluation_summary",
        "stage2_world_state_sample",
        "stage2_tracked_objects_sample",
        "stage3a_evaluation_summary",
        "stage3a_lane_world_state_sample",
        "stage3b_evaluation_summary",
        "stage3b_behavior_timeline",
        "stage3b_behavior_request_sample",
        "stage3c_execution_summary",
        "stage3c_execution_timeline",
        "stage3c_lane_change_events",
    ]:
        candidate = artifacts.get(artifact_name)
        path = resolve_existing_path(repo_root, candidate)
        reference_output.append(
            {
                "name": artifact_name,
                "path": rel_repo_path(repo_root, path) if path else candidate,
                "exists": bool(path and path.exists()),
                "notes": ["reference_output" if artifact_name != "stage1_session_summary" else "summary_proxy"],
            }
        )

    online_smoke_only = [row for row in _scan_online_stage4_candidates(repo_root) if row.get("case_id") == case_id]
    benchmark_only = [
        {
            "name": "case_spec",
            "path": rel_repo_path(repo_root, Path(case_spec["_case_path"])),
            "exists": True,
            "notes": ["benchmark_only"],
        }
    ]
    mixed_role = []
    if source.get("scenario_manifest") or scenario_recipe.get("scenario_manifest"):
        mixed_role.append(
            {
                "name": "scenario_manifest",
                "reason": "currently acts as orchestration input and benchmark artifact hint",
            }
        )
    if artifacts.get("stage1_session_summary") and (source.get("stage1_session_dir") or scenario_recipe.get("stage1_session_dir")):
        mixed_role.append(
            {
                "name": "stage1_session_summary",
                "reason": "summary is benchmark artifact, but current benchmark also uses it as proxy witness for replay input",
            }
        )

    return {
        "case_id": case_id,
        "status": case_spec.get("status"),
        "determinism_level": case_spec.get("determinism_level", "unknown"),
        "frozen_input": frozen_input,
        "reference_output": reference_output,
        "online_smoke_only": online_smoke_only,
        "benchmark_only": benchmark_only,
        "mixed_role": mixed_role,
    }


def _mapping_modes_from_build(repo_root: Path, build_row: dict[str, Any]) -> dict[str, Any]:
    bundle_dir = Path(build_row["bundle_dir"])
    semantic_dir = bundle_dir / "reference_outputs" / "semantic"
    bundled_artifacts = dict(build_row.get("bundled_artifacts") or {})
    runner_manifest_exists = (bundle_dir / "scenario_artifacts" / "runner_manifest.json").exists()
    stage1_session_exists = (bundle_dir / "stage1_frozen" / "session" / "session_summary.json").exists()
    stage_reference_ok = all(bool(row.get("available")) for row in build_row.get("stage_reference_rows", []))
    replay_source = {
        "type": "frozen_case_bundle_v1_reference",
        "scenario_manifest": rel_repo_path(repo_root, bundle_dir / "scenario_artifacts" / "runner_manifest.json"),
        "stage1_session_dir": rel_repo_path(repo_root, bundle_dir / "stage1_frozen" / "session"),
        "stage2_dir": rel_repo_path(repo_root, bundle_dir / "reference_outputs" / "stage2"),
        "stage3a_dir": rel_repo_path(repo_root, bundle_dir / "reference_outputs" / "stage3a"),
        "stage3b_dir": rel_repo_path(repo_root, bundle_dir / "reference_outputs" / "stage3b"),
        "stage3c_dir": rel_repo_path(repo_root, bundle_dir / "reference_outputs" / "stage3c"),
    }
    semantic_artifacts = {
        key: rel_repo_path(repo_root, Path(path))
        for key, path in bundled_artifacts.items()
        if key in {"ground_truth_actor_roles", "critical_actor_binding", "behavior_arbitration_events", "route_session_binding"}
    }
    replay_artifacts = {
        key: rel_repo_path(repo_root, Path(path))
        for key, path in bundled_artifacts.items()
    }
    scenario_mapping = {
        "source": {
            "type": "frozen_case_bundle_v1_input",
            "scenario_manifest": rel_repo_path(repo_root, bundle_dir / "scenario_artifacts" / "runner_manifest.json"),
            "stage1_session_dir": rel_repo_path(repo_root, bundle_dir / "stage1_frozen" / "session"),
            "frozen_bundle_dir": rel_repo_path(repo_root, bundle_dir),
        },
        "scenario_recipe": {
            "source_type": "frozen_case_bundle_v1",
            "scenario_manifest": rel_repo_path(repo_root, bundle_dir / "scenario_artifacts" / "runner_manifest.json"),
            "stage1_session_dir": rel_repo_path(repo_root, bundle_dir / "stage1_frozen" / "session"),
            "stage1_materialization": "frozen_stage1_session",
        },
        "artifacts": {
            **semantic_artifacts,
            "stage1_session_summary": rel_repo_path(repo_root, bundle_dir / "stage1_frozen" / "session" / "session_summary.json"),
        },
    }
    online_mapping = {
        "source": {
            "scenario_manifest": rel_repo_path(repo_root, bundle_dir / "scenario_artifacts" / "runner_manifest.json"),
            "frozen_bundle_dir": rel_repo_path(repo_root, bundle_dir),
        },
        "scenario_recipe": {
            "scenario_manifest": rel_repo_path(repo_root, bundle_dir / "scenario_artifacts" / "runner_manifest.json"),
            "source_type": "frozen_case_bundle_v1_reference_online",
        },
        "artifacts": semantic_artifacts,
    }
    mappings: dict[str, Any] = {}
    if stage_reference_ok:
        mappings["replay_regression"] = {"source": replay_source, "artifacts": replay_artifacts}
    if runner_manifest_exists and stage1_session_exists:
        mappings["scenario_replay"] = scenario_mapping
    if runner_manifest_exists:
        mappings["online_e2e"] = online_mapping
    return mappings


def build_frozen_corpus(
    *,
    repo_root: Path,
    config_path: str | Path,
    output_root: Path,
    selected_cases: list[str] | None = None,
) -> dict[str, Any]:
    benchmark_config = load_yaml(resolve_repo_path(repo_root, config_path))
    cases_dir = resolve_repo_path(repo_root, benchmark_config["cases_dir"])
    if cases_dir is None:
        raise ValueError("cases_dir missing from benchmark config")

    selected = set(selected_cases or [])
    case_paths = sorted(cases_dir.glob("*.yaml"))
    if selected:
        case_paths = [path for path in case_paths if path.stem in selected]

    output_root.mkdir(parents=True, exist_ok=True)
    cases_output_root = output_root / "cases"
    cases_output_root.mkdir(parents=True, exist_ok=True)

    benchmark_only_global = _benchmark_only_global(repo_root, benchmark_config)
    audit_rows: list[dict[str, Any]] = []
    build_rows: list[dict[str, Any]] = []
    readiness_rows: list[dict[str, Any]] = []
    mapping_cases: dict[str, Any] = {}

    for case_path in case_paths:
        case_spec = _load_case_spec(case_path)
        audit_row = audit_case_data(repo_root, case_spec)
        build_row = materialize_case_bundle(
            repo_root=repo_root,
            benchmark_config=benchmark_config,
            case_spec=case_spec,
            bundle_dir=cases_output_root / case_path.stem,
        )
        build_rows.append(build_row)
        readiness_rows.append(
            {
                "case_id": build_row["case_id"],
                "readiness": build_row["readiness"],
                "readiness_checks": build_row["case_manifest"]["readiness_checks"],
                "warnings": build_row["case_manifest"]["warnings"],
            }
        )
        mapping_cases[build_row["case_id"]] = {
            "readiness": build_row["readiness"],
            "bundle_dir": rel_repo_path(repo_root, Path(build_row["bundle_dir"])),
            "case_manifest": rel_repo_path(repo_root, Path(build_row["bundle_dir"]) / "case_manifest.json"),
            "scenario_manifest": rel_repo_path(repo_root, Path(build_row["bundle_dir"]) / "scenario_manifest.json"),
            "runner_scenario_manifest": rel_repo_path(repo_root, Path(build_row["bundle_dir"]) / "scenario_artifacts" / "runner_manifest.json"),
            "stage1_session_dir": rel_repo_path(repo_root, Path(build_row["bundle_dir"]) / "stage1_frozen" / "session"),
            "benchmark_reference_path": rel_repo_path(repo_root, Path(build_row["bundle_dir"]) / "benchmark_reference" / "benchmark_reference.json"),
            "provenance_path": rel_repo_path(repo_root, Path(build_row["bundle_dir"]) / "provenance" / "provenance.json"),
            "mapping_modes": _mapping_modes_from_build(repo_root, build_row),
            "warnings": build_row["case_manifest"]["warnings"],
        }
        audit_row["bundle_readiness"] = build_row["readiness"]
        audit_rows.append(audit_row)

    readiness_counts: dict[str, int] = {}
    for row in readiness_rows:
        readiness_counts[row["readiness"]] = readiness_counts.get(row["readiness"], 0) + 1

    manifest = {
        "bundle_version": "frozen_case_bundle_v1",
        "generated_at": utc_now_iso(),
        "corpus_root": rel_repo_path(repo_root, output_root),
        "cases": {
            row["case_id"]: {
                "readiness": row["readiness"],
                "bundle_dir": rel_repo_path(repo_root, Path(row["bundle_dir"])),
            }
            for row in build_rows
        },
        "summary": {
            "num_cases": len(build_rows),
            "readiness_counts": readiness_counts,
        },
    }
    dump_json(output_root / "manifest.json", manifest)

    audit_payload = {
        "generated_at": utc_now_iso(),
        "corpus_root": rel_repo_path(repo_root, output_root),
        "benchmark_only_global": benchmark_only_global,
        "online_smoke_candidates": _scan_online_stage4_candidates(repo_root),
        "cases": audit_rows,
    }
    dump_json(output_root / "frozen_corpus_audit.json", audit_payload)

    validation = validate_corpus(output_root)
    readiness_payload = {
        "generated_at": utc_now_iso(),
        "corpus_root": rel_repo_path(repo_root, output_root),
        "summary": {
            "num_cases": len(readiness_rows),
            "readiness_counts": readiness_counts,
            "validation_passed": bool(validation.get("passed")),
        },
        "cases": readiness_rows,
        "validation": validation,
    }
    dump_json(output_root / "frozen_corpus_readiness.json", readiness_payload)

    build_report = {
        "generated_at": utc_now_iso(),
        "corpus_root": rel_repo_path(repo_root, output_root),
        "summary": {
            "num_cases": len(build_rows),
            "readiness_counts": readiness_counts,
        },
        "cases": [
            {
                "case_id": row["case_id"],
                "bundle_dir": rel_repo_path(repo_root, Path(row["bundle_dir"])),
                "readiness": row["readiness"],
                "stage1_infer_ready_samples": row["stage1_info"].get("infer_ready_samples", 0),
                "stage_reference_rows": row["stage_reference_rows"],
                "semantic_rows": row["semantic_rows"],
            }
            for row in build_rows
        ],
    }
    dump_json(output_root / "frozen_corpus_build_report.json", build_report)

    mapping_payload = {
        "generated_at": utc_now_iso(),
        "bundle_version": "frozen_case_bundle_v1",
        "corpus_root": rel_repo_path(repo_root, output_root),
        "cases": mapping_cases,
    }
    dump_json(repo_root / "benchmark" / "mapping_benchmark_to_frozen_corpus.json", mapping_payload)

    return {
        "manifest": manifest,
        "audit": audit_payload,
        "readiness": readiness_payload,
        "build_report": build_report,
        "mapping": mapping_payload,
    }
