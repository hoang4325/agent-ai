from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.run_system_benchmark_e2e import run_e2e_benchmark

from .io import dump_json, load_json, load_jsonl, load_yaml, resolve_repo_path
from .runner_core import run_benchmark
from .stage6_shadow_contract_audit import run_stage6_shadow_contract_audit
from .stage6_shadow_metric_pack import build_stage6_shadow_metric_pack


SHADOW_CASES = [
    "ml_right_positive_core",
    "stop_follow_ambiguity_core",
]

STAGE4_PROFILE_OVERRIDES = {
    "ml_right_positive_core": "stage4_online_external_watch_shadow_smoke",
    "stop_follow_ambiguity_core": "stage4_online_external_watch_stop_aligned_shadow_smoke",
}

ONLINE_SMOKE_CASE_OVERRIDES = {
    "ml_right_positive_core": {
        "mandatory_artifacts_add": [
            "stage3c_mpc_shadow_proposal",
            "stage3c_mpc_shadow_summary",
            "stage3c_shadow_baseline_comparison",
            "stage3c_shadow_disagreement_events",
            "stage3c_shadow_realized_outcome_summary",
        ],
        "primary_metrics": [
            "artifact_completeness",
            "shadow_output_contract_completeness",
            "shadow_feasibility_rate",
        ],
        "secondary_metrics": [
            "shadow_solver_timeout_rate",
            "shadow_planner_execution_latency_ms",
            "shadow_baseline_disagreement_rate",
            "shadow_fallback_recommendation_rate",
            "shadow_realized_support_rate",
            "shadow_realized_behavior_match_rate",
            "shadow_lateral_oscillation_rate",
            "shadow_trajectory_smoothness_proxy",
            "shadow_control_jerk_proxy",
        ],
        "metric_thresholds": {
            "shadow_output_contract_completeness": {"min": 1.0},
            "shadow_feasibility_rate": {"min": 1.0},
            "shadow_solver_timeout_rate": {"max": 0.0},
            "shadow_planner_execution_latency_ms": {"max": 6.0},
            "shadow_baseline_disagreement_rate": {"max": 0.35},
            "shadow_fallback_recommendation_rate": {"max": 0.2},
            "shadow_realized_support_rate": {"min": 0.9},
            "shadow_realized_behavior_match_rate": {"min": 0.8},
        },
    },
    "stop_follow_ambiguity_core": {
        "mandatory_artifacts_add": [
            "stage3c_mpc_shadow_proposal",
            "stage3c_mpc_shadow_summary",
            "stage3c_shadow_baseline_comparison",
            "stage3c_shadow_disagreement_events",
            "stage3c_shadow_realized_outcome_summary",
        ],
        "primary_metrics": [
            "artifact_completeness",
            "shadow_output_contract_completeness",
            "shadow_feasibility_rate",
        ],
        "secondary_metrics": [
            "shadow_solver_timeout_rate",
            "shadow_planner_execution_latency_ms",
            "shadow_baseline_disagreement_rate",
            "shadow_fallback_recommendation_rate",
            "shadow_realized_support_rate",
            "shadow_realized_behavior_match_rate",
            "shadow_stop_final_error_m",
            "shadow_stop_overshoot_distance_m",
        ],
        "metric_thresholds": {
            "shadow_output_contract_completeness": {"min": 1.0},
            "shadow_feasibility_rate": {"min": 1.0},
            "shadow_solver_timeout_rate": {"max": 0.0},
            "shadow_planner_execution_latency_ms": {"max": 4.5},
            "shadow_baseline_disagreement_rate": {"max": 0.8},
            "shadow_fallback_recommendation_rate": {"max": 0.5},
            "shadow_realized_support_rate": {"min": 0.9},
            "shadow_realized_behavior_match_rate": {"min": 0.8},
        },
    },
}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _materialize_runtime_config(repo_root: Path, *, root: Path) -> Path:
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    runtime_config = deepcopy(benchmark_config)
    cases_dir = resolve_repo_path(repo_root, benchmark_config.get("cases_dir")) or (repo_root / "benchmark" / "cases")
    runtime_cases_dir = root / "cases"
    runtime_sets_dir = root / "sets"
    runtime_cases_dir.mkdir(parents=True, exist_ok=True)
    runtime_sets_dir.mkdir(parents=True, exist_ok=True)

    for case_id in SHADOW_CASES:
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
        case_payload["stage_profiles"]["stage4"] = STAGE4_PROFILE_OVERRIDES[case_id]
        case_payload["scenario_recipe"] = dict(case_payload.get("scenario_recipe") or {})
        case_payload["scenario_recipe"]["stage1_session_dir"] = str(bundle_session_dir)
        case_payload["source"] = dict(case_payload.get("source") or {})
        case_payload["source"]["stage1_session_dir"] = str(bundle_session_dir)
        case_payload["notes"] = (
            str(case_payload.get("notes") or "").rstrip()
            + "\nStage 6 online shadow smoke override: external_watch frozen Stage1 session plus proposal-only MPC shadow runtime."
        ).strip()
        _write_yaml(runtime_cases_dir / f"{case_id}.yaml", case_payload)

    set_name = "stage6_online_shadow_smoke"
    set_path = runtime_sets_dir / f"{set_name}.yaml"
    _write_yaml(
        set_path,
        {
            "set_name": set_name,
            "description": "Minimal Stage 6 online shadow smoke subset on the promoted two-case rollout.",
            "cases": SHADOW_CASES,
        },
    )

    runtime_config["cases_dir"] = str(runtime_cases_dir)
    runtime_config["sets"] = dict(runtime_config.get("sets") or {})
    runtime_config["sets"][set_name] = str(set_path)
    config_path = root / "benchmark_runtime_config.yaml"
    _write_yaml(config_path, runtime_config)
    return config_path


def _apply_shadow_metric_overlay(case_spec: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    updated = dict(case_spec)
    updated["mandatory_artifacts"] = list(
        dict.fromkeys([*(updated.get("mandatory_artifacts") or []), *(override.get("mandatory_artifacts_add") or [])])
    )
    if override.get("primary_metrics"):
        updated["primary_metrics"] = list(override["primary_metrics"])
    if override.get("secondary_metrics"):
        updated["secondary_metrics"] = list(override["secondary_metrics"])
    thresholds = dict(updated.get("metric_thresholds") or {})
    thresholds.update(override.get("metric_thresholds") or {})
    updated["metric_thresholds"] = thresholds
    updated["stage6_online_shadow_runtime_overlay"] = {
        "enabled": True,
        "subset_case": updated.get("scenario_id"),
    }
    return updated


def _materialize_shadow_eval_config(
    *,
    online_report_root: Path,
    metric_pack: dict[str, Any],
) -> Path:
    base_runtime_cfg = load_yaml(online_report_root / "benchmark_runtime_config.yaml")
    base_cases_dir = online_report_root / "cases_runtime"
    overlay_cases_dir = online_report_root / "stage6_online_shadow_cases_runtime"
    overlay_sets_dir = online_report_root / "stage6_online_shadow_sets_runtime"
    overlay_cases_dir.mkdir(parents=True, exist_ok=True)
    overlay_sets_dir.mkdir(parents=True, exist_ok=True)

    for case_id in SHADOW_CASES:
        base_case_path = base_cases_dir / f"{case_id}.yaml"
        if not base_case_path.exists():
            base_case_path = online_report_root.parent / "cases" / f"{case_id}.yaml"
        case_spec = load_yaml(base_case_path)
        case_spec = _apply_shadow_metric_overlay(case_spec, ONLINE_SMOKE_CASE_OVERRIDES[case_id])
        _write_yaml(overlay_cases_dir / f"{case_id}.yaml", case_spec)

    overlay_set_path = overlay_sets_dir / "stage6_online_shadow_smoke.yaml"
    _write_yaml(
        overlay_set_path,
        {
            "set_name": "stage6_online_shadow_smoke",
            "description": "Stage 6 online shadow smoke evaluation subset.",
            "cases": SHADOW_CASES,
        },
    )

    runtime_cfg = dict(base_runtime_cfg)
    runtime_cfg["cases_dir"] = str(overlay_cases_dir)
    runtime_cfg["sets"] = dict(runtime_cfg.get("sets") or {})
    runtime_cfg["sets"]["stage6_online_shadow_smoke"] = str(overlay_set_path)
    runtime_cfg["stage6_online_shadow_runtime_overlay"] = True
    runtime_cfg_path = online_report_root / "stage6_online_shadow_runtime_config.yaml"
    _write_yaml(runtime_cfg_path, runtime_cfg)
    return runtime_cfg_path


def _build_online_shadow_report(
    *,
    repo_root: Path,
    online_result: dict[str, Any],
    eval_result: dict[str, Any],
    contract_audit: dict[str, Any],
    metric_pack: dict[str, Any],
    online_report_root: Path,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case_id in SHADOW_CASES:
        stage3c_dir = online_report_root / "case_runs" / case_id / "_current_run_artifacts" / "stage3c"
        cases.append(
            {
                "case_id": case_id,
                "shadow_summary": load_json(stage3c_dir / "mpc_shadow_summary.json", default={}) or {},
                "comparison": load_json(stage3c_dir / "shadow_baseline_comparison.json", default={}) or {},
                "proposal_rows": len(load_jsonl(stage3c_dir / "mpc_shadow_proposal.jsonl")),
                "stage3c_dir": str(stage3c_dir),
            }
        )
    report = {
        "schema_version": "stage6_online_shadow_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "6_online_shadow_smoke",
        "benchmark_mode": "online_e2e",
        "description": "Proposal-only Stage 6 online shadow smoke report on the promoted two-case subset.",
        "subset_cases": SHADOW_CASES,
        "online_report_dir": str(online_report_root),
        "shadow_evaluation_report_dir": str(eval_result.get("report_dir")),
        "contract_audit": contract_audit,
        "metric_pack_path": str(repo_root / "benchmark" / "stage6_online_shadow_metric_pack.json"),
        "cases": cases,
        "notes": [
            "Baseline remains the only execution authority.",
            "Stage 4 emits proposal-only MPC shadow artifacts during the online loop; no takeover is allowed.",
            "Online shadow smoke stays pinned to the promoted two-case subset until runtime stability is demonstrated repeatedly.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_online_shadow_report.json", report)
    return report


def run_stage6_online_shadow_smoke(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    readiness = load_json(repo_root / "benchmark" / "stage6_shadow_readiness.json", default={}) or {}
    if str(readiness.get("status")) != "ready":
        raise RuntimeError("Stage 6 online shadow smoke is blocked because benchmark/stage6_shadow_readiness.json is not ready.")

    report_root = repo_root / "benchmark" / "reports" / f"stage6_online_shadow_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_root.mkdir(parents=True, exist_ok=True)
    config_path = _materialize_runtime_config(repo_root, root=report_root)
    online_report_root = report_root / "online_e2e"

    online_result = run_e2e_benchmark(
        config_path=config_path,
        set_name="stage6_online_shadow_smoke",
        benchmark_mode="online_e2e",
        execute_stages=True,
        report_dir=online_report_root,
    )

    metric_pack = build_stage6_shadow_metric_pack(repo_root)
    metric_pack["generated_for_mode"] = "online_e2e_smoke"
    dump_json(repo_root / "benchmark" / "stage6_online_shadow_metric_pack.json", metric_pack)

    case_stage3c_dirs = {
        case_id: online_report_root / "case_runs" / case_id / "_current_run_artifacts" / "stage3c"
        for case_id in SHADOW_CASES
    }
    contract_audit = run_stage6_shadow_contract_audit(
        repo_root,
        case_stage3c_dirs=case_stage3c_dirs,
    )
    dump_json(repo_root / "benchmark" / "stage6_online_shadow_contract_audit.json", contract_audit)

    shadow_eval_config = _materialize_shadow_eval_config(
        online_report_root=online_report_root,
        metric_pack=metric_pack,
    )
    eval_result = run_benchmark(
        repo_root=repo_root,
        config_path=shadow_eval_config,
        set_name="stage6_online_shadow_smoke",
        report_dir=online_report_root / "stage6_online_shadow_evaluation",
    )
    shadow_report = _build_online_shadow_report(
        repo_root=repo_root,
        online_result=online_result,
        eval_result=eval_result,
        contract_audit=contract_audit,
        metric_pack=metric_pack,
        online_report_root=online_report_root,
    )
    gate_result = load_json(Path(eval_result["report_dir"]) / "gate_result.json", default={}) or {}
    mode_resolution = load_json(online_report_root / "mode_resolution.json", default=[]) or []
    runtime_failed = [
        str(row.get("scenario_id"))
        for row in mode_resolution
        if str(row.get("status")) not in {"success"}
    ]
    warning_cases = sorted(set(gate_result.get("warning_cases") or []))
    failed_cases = sorted(set((gate_result.get("failed_cases") or []) + runtime_failed))
    overall_status = "pass"
    if failed_cases:
        overall_status = "fail"
    elif warning_cases or str(online_result.get("overall_status")) == "partial":
        overall_status = "partial"
    payload = {
        "schema_version": "stage6_online_shadow_smoke_result_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": "6_online_shadow_smoke",
        "benchmark_mode": "online_e2e",
        "gate_driver": "frozen_corpus + online_e2e + stage4 proposal-only shadow runtime",
        "overall_status": overall_status,
        "failed_cases": failed_cases,
        "warnings": warning_cases,
        "subset_cases": SHADOW_CASES,
        "online_report_dir": str(online_report_root),
        "shadow_evaluation_report_dir": str(eval_result.get("report_dir")),
        "shadow_contract_audit_path": str(repo_root / "benchmark" / "stage6_online_shadow_contract_audit.json"),
        "shadow_metric_pack_path": str(repo_root / "benchmark" / "stage6_online_shadow_metric_pack.json"),
        "shadow_report_path": str(repo_root / "benchmark" / "stage6_online_shadow_report.json"),
        "online_e2e_overall_status": online_result.get("overall_status"),
        "shadow_contract_status": contract_audit.get("shadow_output_contract"),
        "baseline_comparison_ready": contract_audit.get("baseline_comparison_ready"),
    }
    dump_json(repo_root / "benchmark" / "stage6_online_shadow_smoke_result.json", payload)
    return {
        "online_result": online_result,
        "shadow_evaluation": eval_result,
        "shadow_report": shadow_report,
        "contract_audit": contract_audit,
        "metric_pack": metric_pack,
        "gate_result": payload,
    }
