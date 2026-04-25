from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.run_system_benchmark_e2e import run_e2e_benchmark

from .io import dump_json, load_json, load_yaml, resolve_repo_path


STOP_ALIGNMENT_CASES = ["stop_follow_ambiguity_core"]


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _materialize_runtime_config(repo_root: Path, *, root: Path, case_ids: list[str]) -> Path:
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    runtime_config = deepcopy(benchmark_config)
    cases_dir = resolve_repo_path(repo_root, benchmark_config.get("cases_dir")) or (repo_root / "benchmark" / "cases")
    runtime_cases_dir = root / "cases"
    runtime_cases_dir.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        case_payload = load_yaml(cases_dir / f"{case_id}.yaml")
        bundle_session_dir = (
            repo_root
            / "benchmark"
            / "frozen_corpus"
            / "v1"
            / "cases"
            / case_id
            / "stage1_frozen"
            / "session"
        )
        case_payload["stage_profiles"] = dict(case_payload.get("stage_profiles") or {})
        case_payload["stage_profiles"]["stage4"] = "stage4_online_external_watch_stop_aligned"
        case_payload["scenario_recipe"] = dict(case_payload.get("scenario_recipe") or {})
        case_payload["scenario_recipe"]["stage1_session_dir"] = str(bundle_session_dir)
        case_payload["source"] = dict(case_payload.get("source") or {})
        case_payload["source"]["stage1_session_dir"] = str(bundle_session_dir)
        _write_yaml(runtime_cases_dir / f"{case_id}.yaml", case_payload)

    set_name = "stage4_stop_alignment_subset"
    set_path = root / "sets" / f"{set_name}.yaml"
    _write_yaml(
        set_path,
        {
            "set_name": set_name,
            "description": "Stop-focused Stage 4 online subset with explicit stop-scene alignment.",
            "cases": case_ids,
        },
    )
    runtime_config["cases_dir"] = str(runtime_cases_dir)
    runtime_config["sets"] = dict(runtime_config.get("sets") or {})
    runtime_config["sets"][set_name] = str(set_path)
    config_path = root / "benchmark_runtime_config.yaml"
    _write_yaml(config_path, runtime_config)
    return config_path


def run_stage4_stop_alignment_revalidation(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    report_root = repo_root / "benchmark" / "reports" / f"stage4_stop_alignment_revalidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_root.mkdir(parents=True, exist_ok=True)
    config_path = _materialize_runtime_config(repo_root, root=report_root, case_ids=STOP_ALIGNMENT_CASES)

    online_report_dir = report_root / "online_e2e"
    online_result = run_e2e_benchmark(
        config_path=config_path,
        set_name="stage4_stop_alignment_subset",
        benchmark_mode="online_e2e",
        execute_stages=True,
        report_dir=online_report_dir,
    )

    stage4_dir = online_report_dir / "case_runs" / "stop_follow_ambiguity_core" / "stage4"
    stop_target_payload = load_json(stage4_dir / "stop_target.json", default={}) or {}
    stop_binding_summary = load_json(stage4_dir / "stop_binding_summary.json", default={}) or {}
    stop_scene_alignment = load_json(stage4_dir / "stop_scene_alignment.json", default={}) or {}
    materialization_payload = load_json(stage4_dir / "critical_actor_materialization.json", default={}) or {}
    mode_resolution = load_json(online_report_dir / "mode_resolution.json", default=[]) or []
    mode_row = next((row for row in mode_resolution if str(row.get("scenario_id")) == "stop_follow_ambiguity_core"), {})

    stop_support = dict(stop_target_payload.get("measurement_support") or {})
    blocker_binding = dict((materialization_payload.get("actor_bindings") or {}).get("blocker") or {})
    binding_status = str(blocker_binding.get("binding_status") or "missing")
    stop_final_error_present = str(stop_support.get("stop_final_error_m")) == "present"
    alignment_ready = bool(stop_scene_alignment.get("alignment_ready"))
    status = "ready"
    remaining_blockers: list[str] = []
    if binding_status != "exact":
        status = "partial"
        remaining_blockers.append("blocker_actor_not_exactly_materialized_online")
    if not stop_final_error_present:
        status = "partial"
        remaining_blockers.append("stop_final_error_m_not_directly_available")
    if not alignment_ready:
        status = "partial"
        remaining_blockers.append("online_stop_scene_not_aligned_with_frozen_stop_state")

    readiness = {
        "schema_version": "stage6_stop_target_readiness_v2",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "required_subset": STOP_ALIGNMENT_CASES,
        "online_report_dir": str(online_report_dir),
        "online_runtime_status": mode_row.get("status"),
        "cases_exact": ["stop_follow_ambiguity_core"] if binding_status == "exact" else [],
        "cases_heuristic": [] if binding_status == "exact" else ["stop_follow_ambiguity_core"],
        "can_promote_stop_final_error_m": bool(stop_final_error_present and alignment_ready),
        "alignment_consistent": bool(stop_scene_alignment.get("alignment_consistent")),
        "alignment_ready": alignment_ready,
        "remaining_blockers": remaining_blockers,
        "stop_binding_summary": stop_binding_summary,
        "stop_scene_alignment": stop_scene_alignment,
        "blocker_binding": blocker_binding,
        "online_e2e_overall_status": online_result.get("overall_status"),
    }
    dump_json(repo_root / "benchmark" / "stage6_stop_target_readiness.json", readiness)
    return {
        "report_root": str(report_root),
        "online_result": online_result,
        "readiness": readiness,
    }
