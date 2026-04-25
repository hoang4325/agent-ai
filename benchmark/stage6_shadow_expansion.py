from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml

from scripts.run_system_benchmark_e2e import run_e2e_benchmark

from .io import dump_json, load_json, load_jsonl, load_yaml, resolve_repo_path
from .runner_core import run_benchmark
from .stage6_online_shadow_stability import _ensure_carla_ready, _restart_carla
from .stage6_shadow_contract_audit import run_stage6_shadow_contract_audit


EXPANSION_CASES = [
    "ml_right_positive_core",
    "stop_follow_ambiguity_core",
    "ml_left_positive_core",
    "junction_straight_core",
    "arbitration_stop_during_prepare_right",
    "fr_lc_commit_no_permission",
]

STOP_FOCUSED_CASES = {
    "stop_follow_ambiguity_core",
    "arbitration_stop_during_prepare_right",
}

ARBITRATION_OR_FAILURE_CASES = {
    "arbitration_stop_during_prepare_right",
    "fr_lc_commit_no_permission",
}

SHADOW_ARTIFACTS = [
    "stage3c_mpc_shadow_proposal",
    "stage3c_mpc_shadow_summary",
    "stage3c_shadow_baseline_comparison",
    "stage3c_shadow_disagreement_events",
    "stage3c_shadow_realized_outcome_summary",
]

STAGE4_PROFILE_OVERRIDES = {
    "ml_right_positive_core": "stage4_online_external_watch_shadow_smoke",
    "stop_follow_ambiguity_core": "stage4_online_external_watch_stop_aligned_shadow_smoke",
    "ml_left_positive_core": "stage4_online_external_watch_shadow_smoke_extended",
    "junction_straight_core": "stage4_online_external_watch_shadow_smoke_extended",
    "arbitration_stop_during_prepare_right": "stage4_online_external_watch_stop_aligned_shadow_smoke_arbitration",
    "fr_lc_commit_no_permission": "stage4_online_external_watch_shadow_smoke_extended",
}

CASE_METRIC_OVERRIDES: dict[str, dict[str, Any]] = {
    "ml_right_positive_core": {
        "mandatory_artifacts_add": SHADOW_ARTIFACTS,
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
            "shadow_planner_execution_latency_ms": {"max": 4.5},
            "shadow_baseline_disagreement_rate": {"max": 0.25},
            "shadow_fallback_recommendation_rate": {"max": 0.2},
            "shadow_realized_support_rate": {"min": 0.95},
            "shadow_realized_behavior_match_rate": {"min": 0.95},
        },
    },
    "stop_follow_ambiguity_core": {
        "mandatory_artifacts_add": SHADOW_ARTIFACTS,
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
            "shadow_realized_stop_distance_mae_m",
        ],
        "metric_thresholds": {
            "shadow_output_contract_completeness": {"min": 1.0},
            "shadow_feasibility_rate": {"min": 1.0},
            "shadow_solver_timeout_rate": {"max": 0.0},
            "shadow_planner_execution_latency_ms": {"max": 4.5},
            "shadow_baseline_disagreement_rate": {"max": 0.25},
            "shadow_fallback_recommendation_rate": {"max": 0.5},
            "shadow_realized_support_rate": {"min": 0.95},
            "shadow_realized_behavior_match_rate": {"min": 0.95},
        },
    },
    "ml_left_positive_core": {
        "mandatory_artifacts_add": SHADOW_ARTIFACTS,
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
        ],
        "metric_thresholds": {
            "shadow_output_contract_completeness": {"min": 1.0},
            "shadow_feasibility_rate": {"min": 1.0},
            "shadow_solver_timeout_rate": {"max": 0.0},
            "shadow_planner_execution_latency_ms": {"max": 4.5},
            "shadow_baseline_disagreement_rate": {"max": 0.25},
            "shadow_fallback_recommendation_rate": {"max": 0.2},
            "shadow_realized_support_rate": {"min": 0.95},
            "shadow_realized_behavior_match_rate": {"min": 0.95},
        },
    },
    "junction_straight_core": {
        "mandatory_artifacts_add": SHADOW_ARTIFACTS,
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
            "shadow_trajectory_smoothness_proxy",
        ],
        "metric_thresholds": {
            "shadow_output_contract_completeness": {"min": 1.0},
            "shadow_feasibility_rate": {"min": 1.0},
            "shadow_solver_timeout_rate": {"max": 0.0},
            "shadow_planner_execution_latency_ms": {"max": 4.5},
            "shadow_baseline_disagreement_rate": {"max": 0.25},
            "shadow_fallback_recommendation_rate": {"max": 0.2},
            "shadow_realized_support_rate": {"min": 0.95},
            "shadow_realized_behavior_match_rate": {"min": 0.95},
        },
    },
    "arbitration_stop_during_prepare_right": {
        "mandatory_artifacts_add": SHADOW_ARTIFACTS,
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
            "shadow_realized_stop_distance_mae_m",
        ],
        "metric_thresholds": {
            "shadow_output_contract_completeness": {"min": 1.0},
            "shadow_feasibility_rate": {"min": 1.0},
            "shadow_solver_timeout_rate": {"max": 0.0},
            "shadow_planner_execution_latency_ms": {"max": 4.5},
            "shadow_baseline_disagreement_rate": {"max": 0.35},
            "shadow_fallback_recommendation_rate": {"max": 0.5},
            "shadow_realized_support_rate": {"min": 0.95},
            "shadow_realized_behavior_match_rate": {"min": 0.95},
        },
    },
    "fr_lc_commit_no_permission": {
        "mandatory_artifacts_add": SHADOW_ARTIFACTS,
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
            "shadow_planner_execution_latency_ms": {"max": 4.5},
            "shadow_baseline_disagreement_rate": {"max": 0.35},
            "shadow_fallback_recommendation_rate": {"max": 0.5},
            "shadow_realized_support_rate": {"min": 0.95},
            "shadow_realized_behavior_match_rate": {"min": 0.95},
        },
    },
}

BASELINE_DIR = Path("benchmark") / "baselines" / "system_benchmark_v1_stage6_shadow_6case"
BASELINE_METRICS = BASELINE_DIR / "baseline_metrics.json"
BASELINE_COMMIT = BASELINE_DIR / "baseline_commit.json"
BASELINE_DECISION = Path("benchmark") / "stage6_shadow_expansion_baseline_decision.json"


@dataclass
class ExpansionRun:
    index: int
    overall_status: str
    online_report_dir: Path
    shadow_eval_report_dir: Path
    shadow_report: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _load_metric_registry(repo_root: Path) -> dict[str, Any]:
    benchmark_cfg = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    metric_registry_path = resolve_repo_path(repo_root, benchmark_cfg["metric_registry"])
    return load_yaml(metric_registry_path)


def _stability_envelope_value(values: list[Any], trend_preference: str | None) -> Any:
    if not values:
        return None
    normalized = [value for value in values if value is not None]
    if not normalized:
        return None
    if trend_preference == "lower_better":
        return max(normalized)
    if trend_preference == "higher_better":
        return min(normalized)
    if trend_preference == "exact_match":
        return normalized[0]
    return float(mean(float(value) for value in normalized))


def _tracked_case_metrics(case_id: str) -> list[str]:
    overrides = CASE_METRIC_OVERRIDES.get(case_id) or {}
    return list(
        dict.fromkeys(
            list(overrides.get("primary_metrics") or [])
            + list(overrides.get("secondary_metrics") or [])
        )
    )


def _build_stability_envelope_baseline(repo_root: Path) -> dict[str, Any] | None:
    stability_report_path = repo_root / "benchmark" / "stage6_shadow_stability_6case_report.json"
    stability_report = load_json(stability_report_path, default={}) or {}
    if str(stability_report.get("readiness")) != "ready":
        return None

    metric_registry = _load_metric_registry(repo_root)
    metric_meta = (metric_registry.get("metrics") or {}) if isinstance(metric_registry, dict) else {}
    case_metric_values: dict[str, dict[str, list[Any]]] = {
        case_id: {metric_name: [] for metric_name in _tracked_case_metrics(case_id)}
        for case_id in EXPANSION_CASES
    }

    run_dirs = [
        Path(str(run.get("shadow_eval_report_dir")))
        for run in (stability_report.get("runs") or [])
        if run.get("shadow_eval_report_dir")
    ]

    def _collect_from_eval_dir(case_id: str, run_dir: Path) -> None:
        benchmark_report = load_json(run_dir / "benchmark_report.json", default={}) or {}
        case_reports = {
            str(case.get("scenario_id")): case
            for case in (benchmark_report.get("cases") or [])
        }
        metrics = (case_reports.get(case_id) or {}).get("metrics") or {}
        for metric_name in _tracked_case_metrics(case_id):
            value = (metrics.get(metric_name) or {}).get("value")
            if value is not None:
                case_metric_values[case_id][metric_name].append(value)

    if run_dirs:
        for run_dir in run_dirs:
            for case_id in EXPANSION_CASES:
                _collect_from_eval_dir(case_id, run_dir)
    else:
        for run in (stability_report.get("runs") or []):
            for case in (run.get("cases") or []):
                case_id = str(case.get("case_id") or "")
                if case_id not in case_metric_values:
                    continue
                attempts = list(case.get("attempts") or [])
                if not attempts:
                    continue
                eval_dir = Path(str((attempts[-1]).get("shadow_evaluation_report_dir") or ""))
                if eval_dir:
                    _collect_from_eval_dir(case_id, eval_dir)

    cases_payload: dict[str, Any] = {}
    for case_id in EXPANSION_CASES:
        metrics_payload: dict[str, Any] = {}
        for metric_name, values in (case_metric_values.get(case_id) or {}).items():
            trend_preference = (metric_meta.get(metric_name) or {}).get("trend_preference")
            envelope_value = _stability_envelope_value(values, trend_preference)
            if envelope_value is not None:
                metrics_payload[metric_name] = envelope_value
        if metrics_payload:
            cases_payload[case_id] = {
                "status": "pass",
                "case_status": "ready",
                "metrics": metrics_payload,
            }

    if not cases_payload:
        return None

    return {
        "benchmark_version": "system_benchmark_v1",
        "generated_at_utc": _utc_now_iso(),
        "set": "stage6_shadow_golden_6case",
        "baseline_source": "stage6_shadow_stability_6case_envelope",
        "baseline_source_report": str(stability_report_path),
        "cases": cases_payload,
    }


def _bundle_paths(repo_root: Path, case_id: str) -> dict[str, Path]:
    bundle_dir = repo_root / "benchmark" / "frozen_corpus" / "v1" / "cases" / case_id
    return {
        "bundle_dir": bundle_dir,
        "runner_manifest": bundle_dir / "scenario_artifacts" / "runner_manifest.json",
        "stage1_session_dir": bundle_dir / "stage1_frozen" / "session",
        "scenario_manifest": bundle_dir / "scenario_manifest.json",
    }


def _load_frozen_readiness(repo_root: Path) -> dict[str, Any]:
    payload = load_json(repo_root / "benchmark" / "frozen_corpus" / "v1" / "frozen_corpus_readiness.json", default={}) or {}
    case_rows = list(payload.get("cases") or [])
    return {str(row.get("case_id")): row for row in case_rows}


def audit_stage6_shadow_expansion(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    cases_dir = repo_root / "benchmark" / "cases"
    frozen_readiness = _load_frozen_readiness(repo_root)
    cases: list[dict[str, Any]] = []
    blockers: list[str] = []
    for case_id in EXPANSION_CASES:
        case_spec = load_yaml(cases_dir / f"{case_id}.yaml")
        supported_modes = set(case_spec.get("supported_benchmark_modes") or [case_spec.get("benchmark_mode", "replay_regression")])
        bundle = _bundle_paths(repo_root, case_id)
        readiness_row = frozen_readiness.get(case_id) or {}
        current_stage4_profile = str((case_spec.get("stage_profiles") or {}).get("stage4") or "")
        notes: list[str] = []
        if "online_e2e" not in supported_modes:
            notes.append("needs_online_e2e_support_overlay")
        if not current_stage4_profile:
            notes.append("needs_stage4_profile_overlay")
        if not bundle["runner_manifest"].exists():
            notes.append("missing_frozen_runner_manifest")
        if not bundle["stage1_session_dir"].exists():
            notes.append("missing_frozen_stage1_session")
        shadow_ready = not any(
            note in {
                "missing_frozen_runner_manifest",
                "missing_frozen_stage1_session",
            }
            for note in notes
        )
        if case_id in STOP_FOCUSED_CASES:
            notes.append(f"uses_stage4_profile_overlay:{STAGE4_PROFILE_OVERRIDES[case_id]}")
        if not shadow_ready:
            blockers.append(case_id)
        cases.append(
            {
                "case_id": case_id,
                "current_supported_modes": sorted(supported_modes),
                "current_stage4_profile": current_stage4_profile or None,
                "recommended_stage4_profile": STAGE4_PROFILE_OVERRIDES[case_id],
                "frozen_runner_manifest_available": bundle["runner_manifest"].exists(),
                "frozen_stage1_session_available": bundle["stage1_session_dir"].exists(),
                "frozen_corpus_readiness": readiness_row.get("readiness"),
                "stop_focused": case_id in STOP_FOCUSED_CASES,
                "shadow_ready": shadow_ready,
                "notes": notes,
            }
        )
    payload = {
        "schema_version": "stage6_shadow_expansion_audit_v1",
        "generated_at_utc": _utc_now_iso(),
        "subset_cases": list(EXPANSION_CASES),
        "new_cases": [
            "ml_left_positive_core",
            "junction_straight_core",
            "arbitration_stop_during_prepare_right",
            "fr_lc_commit_no_permission",
        ],
        "cases": cases,
        "blocking_cases": sorted(set(blockers)),
        "notes": [
            "Frozen runner manifests exist for all six cases, so online shadow can attach through frozen-corpus online_e2e overlays.",
            "The arbitration stop-preemption case now uses a dedicated stop-aligned Stage 4 profile anchored to sample_000020/frame 60021 from the frozen corpus.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_shadow_expansion_audit.json", payload)
    return payload


def _build_metric_pack(repo_root: Path, *, case_ids: list[str] | None = None) -> dict[str, Any]:
    selected_cases = list(case_ids or EXPANSION_CASES)
    case_overrides = {
        case_id: deepcopy(CASE_METRIC_OVERRIDES[case_id])
        for case_id in selected_cases
    }
    metric_pack = {
        "schema_version": "stage6_shadow_metric_pack_v1",
        "generated_at_utc": _utc_now_iso(),
        "stage": "6_shadow",
        "description": "Stage 6 MPC shadow six-case expansion ring. Proposal-only, no takeover.",
        "hard_gate_metrics": [
            {"metric": "artifact_completeness"},
            {"metric": "shadow_output_contract_completeness"},
            {"metric": "shadow_feasibility_rate"},
        ],
        "soft_guardrail_metrics": [
            {"metric": "shadow_solver_timeout_rate"},
            {"metric": "shadow_planner_execution_latency_ms"},
            {"metric": "shadow_baseline_disagreement_rate"},
            {"metric": "shadow_fallback_recommendation_rate"},
            {"metric": "shadow_realized_support_rate"},
            {"metric": "shadow_realized_behavior_match_rate"},
            {"metric": "shadow_stop_final_error_m"},
            {"metric": "shadow_stop_overshoot_distance_m"},
            {"metric": "shadow_lateral_oscillation_rate"},
            {"metric": "shadow_trajectory_smoothness_proxy"},
            {"metric": "shadow_control_jerk_proxy"},
        ],
        "diagnostic_metrics": [
            {"metric": "shadow_lane_change_completion_time_s"},
            {"metric": "shadow_heading_settle_time_s"},
            {"metric": "shadow_realized_speed_mae_mps"},
            {"metric": "shadow_realized_progress_mae_m"},
            {"metric": "shadow_realized_stop_distance_mae_m"},
        ],
        "subset_cases": selected_cases,
        "case_requirements": {
            "stage6_shadow_golden_6case": {
                "description": "Stage 6 six-case shadow expansion ring.",
                "required_pass": True,
                "cases": selected_cases,
                "case_metric_overrides": case_overrides,
            }
        },
        "notes": [
            "Baseline remains the only execution authority.",
            "Shadow proposals are emitted and benchmarked without takeover.",
            "Six-case expansion keeps the ring fixed to the requested cases only.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_shadow_metric_pack.json", metric_pack)
    return metric_pack


def _materialize_runtime_config(repo_root: Path, *, root: Path, case_ids: list[str] | None = None) -> Path:
    selected_cases = list(case_ids or EXPANSION_CASES)
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    runtime_config = deepcopy(benchmark_config)
    cases_dir = resolve_repo_path(repo_root, benchmark_config.get("cases_dir")) or (repo_root / "benchmark" / "cases")
    runtime_cases_dir = root / "cases"
    runtime_sets_dir = root / "sets"
    runtime_cases_dir.mkdir(parents=True, exist_ok=True)
    runtime_sets_dir.mkdir(parents=True, exist_ok=True)

    for case_id in selected_cases:
        case_payload = _build_runtime_case_payload(repo_root, case_id, cases_dir=cases_dir)
        _write_yaml(runtime_cases_dir / f"{case_id}.yaml", case_payload)

    set_name = "stage6_shadow_golden_6case"
    set_path = runtime_sets_dir / f"{set_name}.yaml"
    _write_yaml(
        set_path,
        {
            "set_name": set_name,
            "description": "Stage 6 MPC shadow six-case expansion ring.",
            "cases": selected_cases,
        },
    )
    runtime_config["cases_dir"] = str(runtime_cases_dir)
    runtime_config["sets"] = dict(runtime_config.get("sets") or {})
    runtime_config["sets"][set_name] = str(set_path)
    config_path = root / "benchmark_runtime_config.yaml"
    _write_yaml(config_path, runtime_config)
    return config_path


def _build_runtime_case_payload(
    repo_root: Path,
    case_id: str,
    *,
    cases_dir: Path | None = None,
) -> dict[str, Any]:
    if cases_dir is None:
        benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
        cases_dir = resolve_repo_path(repo_root, benchmark_config.get("cases_dir")) or (repo_root / "benchmark" / "cases")
    case_payload = load_yaml(cases_dir / f"{case_id}.yaml")
    bundle = _bundle_paths(repo_root, case_id)
    case_payload["supported_benchmark_modes"] = list(
        dict.fromkeys([*(case_payload.get("supported_benchmark_modes") or []), "scenario_replay", "online_e2e"])
    )
    case_payload["stage_profiles"] = dict(case_payload.get("stage_profiles") or {})
    case_payload["stage_profiles"]["stage4"] = STAGE4_PROFILE_OVERRIDES[case_id]
    case_payload["scenario_recipe"] = dict(case_payload.get("scenario_recipe") or {})
    case_payload["scenario_recipe"]["stage1_session_dir"] = str(bundle["stage1_session_dir"])
    if bundle["runner_manifest"].exists():
        case_payload["scenario_recipe"]["scenario_manifest"] = str(bundle["runner_manifest"])
    case_payload["source"] = dict(case_payload.get("source") or {})
    case_payload["source"]["stage1_session_dir"] = str(bundle["stage1_session_dir"])
    if bundle["runner_manifest"].exists():
        case_payload["source"]["scenario_manifest"] = str(bundle["runner_manifest"])
    case_payload["artifacts"] = dict(case_payload.get("artifacts") or {})
    if bundle["runner_manifest"].exists():
        case_payload["artifacts"]["scenario_manifest"] = str(bundle["runner_manifest"])
    case_payload["notes"] = (
        str(case_payload.get("notes") or "").rstrip()
        + "\nStage 6 six-case shadow expansion override: frozen Stage1 session plus proposal-only online MPC shadow runtime."
    ).strip()
    return case_payload


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
    updated["stage6_shadow_runtime_overlay"] = {
        "enabled": True,
        "subset_case": updated.get("scenario_id"),
    }
    return updated


def _materialize_shadow_eval_config(
    *,
    online_report_root: Path,
    metric_pack: dict[str, Any],
    case_ids: list[str] | None = None,
) -> Path:
    _ = metric_pack
    selected_cases = list(case_ids or EXPANSION_CASES)
    base_runtime_cfg = load_yaml(online_report_root / "benchmark_runtime_config.yaml")
    base_cases_dir = online_report_root / "cases_runtime"
    overlay_cases_dir = online_report_root / "stage6_shadow_cases_runtime"
    overlay_sets_dir = online_report_root / "stage6_shadow_sets_runtime"
    overlay_cases_dir.mkdir(parents=True, exist_ok=True)
    overlay_sets_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    for case_id in selected_cases:
        base_case_path = base_cases_dir / f"{case_id}.yaml"
        if base_case_path.exists():
            case_spec = load_yaml(base_case_path)
        else:
            case_spec = _build_runtime_case_payload(repo_root, case_id)
        case_spec = _apply_shadow_metric_overlay(case_spec, CASE_METRIC_OVERRIDES[case_id])
        _write_yaml(overlay_cases_dir / f"{case_id}.yaml", case_spec)

    overlay_set_path = overlay_sets_dir / "stage6_shadow_golden_6case.yaml"
    _write_yaml(
        overlay_set_path,
        {
            "set_name": "stage6_shadow_golden_6case",
            "description": "Stage 6 shadow evaluation set for six-case expansion.",
            "cases": selected_cases,
        },
    )

    runtime_cfg = dict(base_runtime_cfg)
    runtime_cfg["cases_dir"] = str(overlay_cases_dir)
    runtime_cfg["sets"] = dict(runtime_cfg.get("sets") or {})
    runtime_cfg["sets"]["stage6_shadow_golden_6case"] = str(overlay_set_path)
    runtime_cfg["stage6_shadow_runtime_overlay"] = True
    runtime_cfg_path = online_report_root / "stage6_shadow_runtime_config_6case.yaml"
    _write_yaml(runtime_cfg_path, runtime_cfg)
    return runtime_cfg_path


def _collect_failed_metrics(benchmark_report: dict[str, Any], diagnostic_metrics: list[str]) -> list[str]:
    diagnostic_set = set(diagnostic_metrics)
    failed_metrics: list[str] = []
    for case in benchmark_report.get("cases") or []:
        for metric_name, metric_result in (case.get("metrics") or {}).items():
            if metric_name in diagnostic_set:
                continue
            if metric_result.get("threshold_passed") is False:
                failed_metrics.append(metric_name)
    return sorted(set(failed_metrics))


def _build_run_report(
    *,
    metric_pack: dict[str, Any],
    contract_audit: dict[str, Any],
    online_report_root: Path,
    eval_result: dict[str, Any],
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    selected_cases = list(case_ids or EXPANSION_CASES)
    cases: list[dict[str, Any]] = []
    for case_id in selected_cases:
        stage3c_dir = online_report_root / "case_runs" / case_id / "_current_run_artifacts" / "stage3c"
        stage4_dir = online_report_root / "case_runs" / case_id / "stage4"
        cases.append(
            {
                "case_id": case_id,
                "stage3c_dir": str(stage3c_dir),
                "stage4_dir": str(stage4_dir),
                "shadow_summary": load_json(stage3c_dir / "mpc_shadow_summary.json", default={}) or {},
                "comparison": load_json(stage3c_dir / "shadow_baseline_comparison.json", default={}) or {},
                "proposal_rows": len(load_jsonl(stage3c_dir / "mpc_shadow_proposal.jsonl")),
                "stop_scene_alignment": load_json(stage3c_dir / "stop_scene_alignment.json", default={}) or {},
                "stop_target": load_json(stage3c_dir / "stop_target.json", default={}) or {},
                "stop_binding_summary": load_json(stage4_dir / "stop_binding_summary.json", default={}) or {},
                "planner_quality_summary": load_json(stage4_dir / "planner_quality_summary.json", default={}) or {},
            }
        )
    return {
        "schema_version": "stage6_shadow_expansion_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "subset_cases": selected_cases,
        "shadow_evaluation_report_dir": str(eval_result.get("report_dir")),
        "contract_audit": contract_audit,
        "metric_pack_subset": list(metric_pack.get("subset_cases") or []),
        "cases": cases,
        "notes": [
            "Baseline remains the only execution authority.",
            "MPC shadow remains proposal-only on the six-case ring.",
        ],
    }


def _run_online_shadow_ring(
    repo_root: Path,
    *,
    report_prefix: str,
    compare_baseline_path: Path | None = None,
    carla_root: Path,
    carla_port: int,
    case_ids: list[str] | None = None,
    restart_carla_between_cases: bool = True,
) -> dict[str, Any]:
    selected_cases = list(case_ids or EXPANSION_CASES)
    report_root = repo_root / "benchmark" / "reports" / f"{report_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_root.mkdir(parents=True, exist_ok=True)
    config_path = _materialize_runtime_config(repo_root, root=report_root, case_ids=selected_cases)
    online_report_root = report_root / "online_e2e"

    online_result = run_e2e_benchmark(
        config_path=config_path,
        set_name="stage6_shadow_golden_6case",
        benchmark_mode="online_e2e",
        execute_stages=True,
        report_dir=online_report_root,
        restart_carla_between_online_cases=bool(restart_carla_between_cases),
        carla_root=carla_root,
        carla_port=int(carla_port),
        carla_case_startup_cooldown_seconds=8.0,
    )

    metric_pack = _build_metric_pack(repo_root, case_ids=selected_cases)
    case_stage3c_dirs = {
        case_id: online_report_root / "case_runs" / case_id / "_current_run_artifacts" / "stage3c"
        for case_id in selected_cases
    }
    contract_audit = run_stage6_shadow_contract_audit(repo_root, case_stage3c_dirs=case_stage3c_dirs)
    shadow_eval_config = _materialize_shadow_eval_config(
        online_report_root=online_report_root,
        metric_pack=metric_pack,
        case_ids=selected_cases,
    )
    eval_result = run_benchmark(
        repo_root=repo_root,
        config_path=shadow_eval_config,
        set_name="stage6_shadow_golden_6case",
        report_dir=online_report_root / "stage6_shadow_expansion_evaluation",
        compare_baseline_path=compare_baseline_path,
    )
    run_report = _build_run_report(
        metric_pack=metric_pack,
        contract_audit=contract_audit,
        online_report_root=online_report_root,
        eval_result=eval_result,
        case_ids=selected_cases,
    )
    gate_result = load_json(Path(eval_result["report_dir"]) / "gate_result.json", default={}) or {}
    benchmark_report = load_json(Path(eval_result["report_dir"]) / "benchmark_report.json", default={}) or {}
    mode_resolution = load_json(online_report_root / "mode_resolution.json", default=[]) or []
    runtime_failed = [
        str(row.get("scenario_id"))
        for row in mode_resolution
        if str(row.get("status")) != "success"
    ]
    warning_cases = sorted(set(gate_result.get("warning_cases") or []))
    failed_cases = sorted(set((gate_result.get("failed_cases") or []) + runtime_failed))
    overall_status = "pass"
    if failed_cases:
        overall_status = "fail"
    elif warning_cases or str(online_result.get("overall_status")) == "partial":
        overall_status = "partial"
    payload = {
        "schema_version": "stage6_shadow_expansion_gate_result_v1",
        "generated_at_utc": _utc_now_iso(),
        "stage": "6_shadow_expansion",
        "benchmark_mode": "online_e2e",
        "overall_status": overall_status,
        "failed_cases": failed_cases,
        "warnings": warning_cases,
        "failed_metrics": _collect_failed_metrics(
            benchmark_report,
            [item["metric"] for item in metric_pack.get("diagnostic_metrics") or []],
        ),
        "subset_cases": selected_cases,
        "online_report_dir": str(online_report_root),
        "shadow_evaluation_report_dir": str(eval_result.get("report_dir")),
        "shadow_contract_status": contract_audit.get("shadow_output_contract"),
        "baseline_comparison_ready": contract_audit.get("baseline_comparison_ready"),
        "compare_baseline_path": (str(compare_baseline_path) if compare_baseline_path else None),
    }
    return {
        "online_result": online_result,
        "shadow_evaluation": eval_result,
        "shadow_report": run_report,
        "contract_audit": contract_audit,
        "metric_pack": metric_pack,
        "gate_result": payload,
    }


def _pin_baseline(
    repo_root: Path,
    smoke_result: dict[str, Any] | None = None,
    *,
    prefer_stability_envelope: bool = False,
) -> dict[str, Any]:
    baseline_metrics_path = repo_root / BASELINE_METRICS
    baseline_commit_path = repo_root / BASELINE_COMMIT
    baseline_decision_path = repo_root / BASELINE_DECISION
    baseline_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_source = "smoke_snapshot"
    source_snapshot: str | None = None
    source_report_dir: str | None = None

    payload = _build_stability_envelope_baseline(repo_root) if prefer_stability_envelope else None
    if payload is not None:
        baseline_source = "stability_envelope"
        source_report_dir = str(repo_root / "benchmark" / "stage6_shadow_stability_6case_report.json")
    else:
        gate_result = (smoke_result or {}).get("gate_result") or {}
        eval_dir = Path(str(gate_result.get("shadow_evaluation_report_dir") or ""))
        snapshot_path = eval_dir / "baseline_snapshot.json"
        payload = load_json(snapshot_path, default={}) or {}
        source_snapshot = str(snapshot_path)
        source_report_dir = str(eval_dir)

    dump_json(baseline_metrics_path, payload)
    commit = {
        "schema_version": "stage6_shadow_6case_baseline_commit_v1",
        "generated_at_utc": _utc_now_iso(),
        "source_snapshot": source_snapshot,
        "source_report_dir": source_report_dir,
        "baseline_source": baseline_source,
        "selected_cases": list(EXPANSION_CASES),
    }
    dump_json(baseline_commit_path, commit)
    decision = {
        "schema_version": "stage6_shadow_expansion_baseline_decision_v1",
        "generated_at_utc": _utc_now_iso(),
        "baseline_metrics_path": str(baseline_metrics_path),
        "baseline_commit_path": str(baseline_commit_path),
        "baseline_source": baseline_source,
        "source_snapshot": source_snapshot,
        "source_report_dir": source_report_dir,
        "used_stability_envelope": bool(baseline_source == "stability_envelope"),
    }
    dump_json(baseline_decision_path, decision)
    return {
        "baseline_metrics_path": str(baseline_metrics_path),
        "baseline_commit_path": str(baseline_commit_path),
        "baseline_decision_path": str(baseline_decision_path),
        "baseline_source": baseline_source,
    }


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0, min(len(ordered) - 1, ceil(q * len(ordered)) - 1))
    return float(ordered[position])


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "p50": None, "p95": None}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
    }


def _load_case_runtime_artifacts(run: ExpansionRun, case_id: str) -> dict[str, Any]:
    stage4_dir = run.online_report_dir / "case_runs" / case_id / "stage4"
    stage3c_dir = run.online_report_dir / "case_runs" / case_id / "_current_run_artifacts" / "stage3c"
    stop_alignment = load_json(stage4_dir / "stop_scene_alignment.json", default={}) or load_json(stage3c_dir / "stop_scene_alignment.json", default={}) or {}
    stop_binding_summary = load_json(stage4_dir / "stop_binding_summary.json", default={}) or {}
    fallback_events = load_jsonl(stage4_dir / "fallback_events.jsonl")
    activated = [row for row in fallback_events if bool(row.get("fallback_activated"))]
    takeovers = [float(row.get("takeover_latency_ms")) for row in activated if row.get("takeover_latency_ms") is not None]
    stop_binding_quality = "missing"
    if stop_binding_summary:
        if float(stop_binding_summary.get("exact_rate") or 0.0) > 0.0:
            stop_binding_quality = "exact"
        elif float(stop_binding_summary.get("defined_rate") or 0.0) > 0.0:
            stop_binding_quality = "clean_enough"
    return {
        "alignment_ready": bool(stop_alignment.get("alignment_ready")),
        "alignment_consistent": bool(stop_alignment.get("alignment_consistent")),
        "stop_binding_quality": stop_binding_quality,
        "fallback_activation_evidence": bool(activated),
        "fallback_takeover_latency_ms_values": takeovers,
    }


def _analyze_stability(runs: list[ExpansionRun]) -> dict[str, Any]:
    run_records: list[dict[str, Any]] = []
    case_rollup: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in EXPANSION_CASES}
    pass_runs = 0
    for run in runs:
        gate_result = load_json(run.shadow_eval_report_dir / "gate_result.json", default={}) or {}
        benchmark_report = load_json(run.shadow_eval_report_dir / "benchmark_report.json", default={}) or {}
        case_reports = {
            str(case.get("scenario_id")): case
            for case in (benchmark_report.get("cases") or [])
        }
        mode_resolution = {
            str(row.get("scenario_id")): str(row.get("status"))
            for row in (load_json(run.online_report_dir / "mode_resolution.json", default=[]) or [])
        }
        if str(gate_result.get("overall_status")) == "pass":
            pass_runs += 1
        run_cases: list[dict[str, Any]] = []
        shadow_cases = {
            str(case.get("case_id")): case
            for case in (run.shadow_report.get("cases") or [])
        }
        for case_id in EXPANSION_CASES:
            shadow_case = shadow_cases.get(case_id) or {}
            summary = shadow_case.get("shadow_summary") or {}
            metrics = (case_reports.get(case_id) or {}).get("metrics") or {}
            runtime = _load_case_runtime_artifacts(run, case_id)
            row = {
                "case_id": case_id,
                "runtime_status": mode_resolution.get(case_id, "missing"),
                "gate_status": (case_reports.get(case_id) or {}).get("status", "missing"),
                "shadow_feasibility_rate": (metrics.get("shadow_feasibility_rate") or {}).get("value"),
                "shadow_solver_timeout_rate": summary.get("shadow_solver_timeout_rate"),
                "shadow_planner_execution_latency_ms": (metrics.get("shadow_planner_execution_latency_ms") or {}).get("value"),
                "shadow_realized_support_rate": (metrics.get("shadow_realized_support_rate") or {}).get("value") or summary.get("shadow_realized_support_rate"),
                "shadow_realized_behavior_match_rate": (metrics.get("shadow_realized_behavior_match_rate") or {}).get("value") or summary.get("shadow_realized_behavior_match_rate"),
                "shadow_realized_stop_distance_mae_m": summary.get("shadow_realized_stop_distance_mae_m"),
                "shadow_baseline_disagreement_rate": (metrics.get("shadow_baseline_disagreement_rate") or {}).get("value"),
                "alignment_ready": runtime.get("alignment_ready"),
                "alignment_consistent": runtime.get("alignment_consistent"),
                "stop_binding_quality": runtime.get("stop_binding_quality"),
                "fallback_activation_evidence": runtime.get("fallback_activation_evidence"),
                "fallback_takeover_latency_ms_values": runtime.get("fallback_takeover_latency_ms_values") or [],
            }
            case_rollup[case_id].append(row)
            run_cases.append(row)
        run_records.append(
            {
                "run_index": run.index,
                "overall_status": gate_result.get("overall_status"),
                "online_report_dir": str(run.online_report_dir),
                "shadow_eval_report_dir": str(run.shadow_eval_report_dir),
                "cases": run_cases,
            }
        )

    pass_rate = float(pass_runs / len(runs)) if runs else 0.0
    total_case_rows = sum(len(rows) for rows in case_rollup.values())
    runtime_success_count = sum(1 for rows in case_rollup.values() for row in rows if row["runtime_status"] == "success")
    runtime_success_rate = float(runtime_success_count / total_case_rows) if total_case_rows else 0.0
    remaining_blockers: list[str] = []
    cases: list[dict[str, Any]] = []

    for case_id, rows in case_rollup.items():
        latency_values = [float(row["shadow_planner_execution_latency_ms"]) for row in rows if row["shadow_planner_execution_latency_ms"] is not None]
        feasibility_values = [float(row["shadow_feasibility_rate"]) for row in rows if row["shadow_feasibility_rate"] is not None]
        timeout_values = [float(row["shadow_solver_timeout_rate"]) for row in rows if row["shadow_solver_timeout_rate"] is not None]
        support_values = [float(row["shadow_realized_support_rate"]) for row in rows if row["shadow_realized_support_rate"] is not None]
        behavior_values = [float(row["shadow_realized_behavior_match_rate"]) for row in rows if row["shadow_realized_behavior_match_rate"] is not None]
        stop_mae_values = [float(row["shadow_realized_stop_distance_mae_m"]) for row in rows if row["shadow_realized_stop_distance_mae_m"] is not None]
        disagreement_values = [float(row["shadow_baseline_disagreement_rate"]) for row in rows if row["shadow_baseline_disagreement_rate"] is not None]
        alignment_ready_rate = float(sum(1 for row in rows if row["alignment_ready"]) / len(rows)) if rows else 0.0
        alignment_consistent_rate = float(sum(1 for row in rows if row["alignment_consistent"]) / len(rows)) if rows else 0.0
        runtime_success = float(sum(1 for row in rows if row["runtime_status"] == "success") / len(rows)) if rows else 0.0
        gate_pass = float(sum(1 for row in rows if row["gate_status"] == "pass") / len(rows)) if rows else 0.0
        binding_qualities = sorted(set(str(row["stop_binding_quality"]) for row in rows if row.get("stop_binding_quality")))
        takeover_values = [value for row in rows for value in row.get("fallback_takeover_latency_ms_values") or []]

        if runtime_success < 1.0:
            remaining_blockers.append(f"{case_id}:runtime")
        if gate_pass < 1.0:
            remaining_blockers.append(f"{case_id}:shadow_gate")
        if feasibility_values and min(feasibility_values) < 1.0:
            remaining_blockers.append(f"{case_id}:solver_quality_feasibility")
        if timeout_values and max(timeout_values) > 0.0:
            remaining_blockers.append(f"{case_id}:solver_timeout")
        if support_values and min(support_values) < 0.95:
            remaining_blockers.append(f"{case_id}:solver_quality_support")
        if behavior_values and min(behavior_values) < 0.95:
            remaining_blockers.append(f"{case_id}:solver_quality_behavior_match")
        disagreement_limit = 0.35 if case_id in ARBITRATION_OR_FAILURE_CASES else 0.25
        if disagreement_values and (_quantile(disagreement_values, 0.95) or 0.0) > disagreement_limit:
            remaining_blockers.append(f"{case_id}:solver_quality_disagreement")
        if case_id in STOP_FOCUSED_CASES:
            if alignment_ready_rate < 1.0:
                remaining_blockers.append(f"{case_id}:contract_alignment")
            if not binding_qualities or any(value not in {"exact", "clean_enough"} for value in binding_qualities):
                remaining_blockers.append(f"{case_id}:contract_stop_target")
            p95_stop_mae = _quantile(stop_mae_values, 0.95) or 0.0
            max_stop_mae = max(stop_mae_values) if stop_mae_values else None
            if stop_mae_values and (p95_stop_mae > 0.25 or (max_stop_mae is not None and max_stop_mae > 0.35)):
                remaining_blockers.append(f"{case_id}:solver_quality_stop_mae")

        cases.append(
            {
                "case_id": case_id,
                "runtime_success_rate": runtime_success,
                "gate_pass_rate": gate_pass,
                "shadow_feasibility_rate": _stats(feasibility_values),
                "shadow_solver_timeout_rate": _stats(timeout_values),
                "shadow_planner_execution_latency_ms": _stats(latency_values),
                "shadow_realized_support_rate": _stats(support_values),
                "shadow_realized_behavior_match_rate": _stats(behavior_values),
                "shadow_realized_stop_distance_mae_m": _stats(stop_mae_values),
                "shadow_baseline_disagreement_rate": _stats(disagreement_values),
                "alignment_ready_rate": alignment_ready_rate if case_id in STOP_FOCUSED_CASES else None,
                "alignment_consistent_rate": alignment_consistent_rate if case_id in STOP_FOCUSED_CASES else None,
                "stop_target_quality": binding_qualities if case_id in STOP_FOCUSED_CASES else [],
                "fallback_takeover_latency_ms": _stats(takeover_values),
                "fallback_takeover_latency_evidence_ready": bool(takeover_values),
            }
        )

    readiness = "ready"
    if pass_rate < 1.0 or runtime_success_rate < 1.0 or remaining_blockers:
        readiness = "not_ready"

    aggregate_latency = [float(row["shadow_planner_execution_latency_ms"]) for rows in case_rollup.values() for row in rows if row["shadow_planner_execution_latency_ms"] is not None]
    aggregate_support = [float(row["shadow_realized_support_rate"]) for rows in case_rollup.values() for row in rows if row["shadow_realized_support_rate"] is not None]
    aggregate_behavior = [float(row["shadow_realized_behavior_match_rate"]) for rows in case_rollup.values() for row in rows if row["shadow_realized_behavior_match_rate"] is not None]
    aggregate_stop_mae = [float(row["shadow_realized_stop_distance_mae_m"]) for rows in case_rollup.values() for row in rows if row["shadow_realized_stop_distance_mae_m"] is not None]
    aggregate_disagreement = [float(row["shadow_baseline_disagreement_rate"]) for rows in case_rollup.values() for row in rows if row["shadow_baseline_disagreement_rate"] is not None]
    aggregate_feasibility = [float(row["shadow_feasibility_rate"]) for rows in case_rollup.values() for row in rows if row["shadow_feasibility_rate"] is not None]
    aggregate_timeout = [float(row["shadow_solver_timeout_rate"]) for rows in case_rollup.values() for row in rows if row["shadow_solver_timeout_rate"] is not None]

    return {
        "schema_version": "stage6_shadow_stability_6case_v1",
        "generated_at_utc": _utc_now_iso(),
        "subset_cases": list(EXPANSION_CASES),
        "num_runs": len(runs),
        "pass_rate": pass_rate,
        "runtime_success_rate": runtime_success_rate,
        "readiness": readiness,
        "aggregate_metrics": {
            "shadow_feasibility_rate": _stats(aggregate_feasibility),
            "shadow_solver_timeout_rate": _stats(aggregate_timeout),
            "shadow_planner_execution_latency_ms": _stats(aggregate_latency),
            "shadow_realized_support_rate": _stats(aggregate_support),
            "shadow_realized_behavior_match_rate": _stats(aggregate_behavior),
            "shadow_realized_stop_distance_mae_m": _stats(aggregate_stop_mae),
            "shadow_baseline_disagreement_rate": _stats(aggregate_disagreement),
        },
        "runs": run_records,
        "cases": cases,
        "remaining_blockers": sorted(set(remaining_blockers)),
        "notes": [
            "Fallback takeover latency is reported only when direct fallback activation evidence exists.",
            "Stop-focused criteria follow the requested p95/max bounds on realized stop-distance MAE.",
        ],
    }


def run_stage6_shadow_expansion(
    repo_root: str | Path,
    *,
    repeats: int = 5,
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
    restart_between_runs: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    carla_root = Path(carla_root)

    audit = audit_stage6_shadow_expansion(repo_root)
    _ensure_carla_ready(carla_root=carla_root, port=carla_port)

    smoke = _run_online_shadow_ring(
        repo_root,
        report_prefix="stage6_shadow_expansion_smoke",
        carla_root=carla_root,
        carla_port=int(carla_port),
    )
    baseline_info = _pin_baseline(repo_root, smoke) if str((smoke.get("gate_result") or {}).get("overall_status")) == "pass" else {
        "baseline_metrics_path": None,
        "baseline_commit_path": None,
    }
    if restart_between_runs:
        _restart_carla(carla_root=carla_root, port=carla_port)
    golden = (
        _run_online_shadow_ring(
            repo_root,
            report_prefix="stage6_shadow_expansion_golden",
            compare_baseline_path=Path(str(baseline_info["baseline_metrics_path"])),
            carla_root=carla_root,
            carla_port=int(carla_port),
        )
        if baseline_info.get("baseline_metrics_path")
        else None
    )

    runs: list[ExpansionRun] = []
    if restart_between_runs:
        _restart_carla(carla_root=carla_root, port=carla_port)
    for index in range(1, repeats + 1):
        if restart_between_runs and index > 1:
            _restart_carla(carla_root=carla_root, port=carla_port)
        result = _run_online_shadow_ring(
            repo_root,
            report_prefix="stage6_shadow_stability_6case",
            carla_root=carla_root,
            carla_port=int(carla_port),
        )
        gate_result = result.get("gate_result") or {}
        runs.append(
            ExpansionRun(
                index=index,
                overall_status=str(gate_result.get("overall_status")),
                online_report_dir=Path(str(gate_result.get("online_report_dir"))),
                shadow_eval_report_dir=Path(str(gate_result.get("shadow_evaluation_report_dir"))),
                shadow_report=(result.get("shadow_report") or {}),
            )
        )

    stability = _analyze_stability(runs)
    expansion_report = {
        "schema_version": "stage6_shadow_expansion_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "subset_cases": list(EXPANSION_CASES),
        "audit_path": str(benchmark_root / "stage6_shadow_expansion_audit.json"),
        "smoke": smoke.get("gate_result"),
        "golden": (golden.get("gate_result") if golden else {"overall_status": "fail", "notes": ["smoke_did_not_pass_no_baseline_pin"]}),
        "baseline_metrics_path": baseline_info.get("baseline_metrics_path"),
        "baseline_commit_path": baseline_info.get("baseline_commit_path"),
        "stability_report_path": str(benchmark_root / "stage6_shadow_stability_6case_report.json"),
        "notes": [
            "Six-case expansion keeps proposal-only execution and does not enable takeover.",
            "Smoke and golden are online-e2e shadow checks on the exact requested six-case ring.",
            "CARLA is restarted between online-e2e cases inside the six-case ring to isolate independent benchmark cases and suppress cross-case simulator contamination.",
        ],
    }
    dump_json(benchmark_root / "stage6_shadow_expansion_report.json", expansion_report)
    dump_json(
        benchmark_root / "stage6_shadow_expansion_result.json",
        {
            "schema_version": "stage6_shadow_expansion_result_v1",
            "generated_at_utc": _utc_now_iso(),
            "subset_cases": list(EXPANSION_CASES),
            "smoke_status": (smoke.get("gate_result") or {}).get("overall_status"),
            "golden_status": ((golden.get("gate_result") or {}).get("overall_status") if golden else "fail"),
            "report_path": str(benchmark_root / "stage6_shadow_expansion_report.json"),
            "stability_result_path": str(benchmark_root / "stage6_shadow_stability_6case_result.json"),
        },
    )
    dump_json(benchmark_root / "stage6_shadow_stability_6case_report.json", stability)
    dump_json(
        benchmark_root / "stage6_shadow_stability_6case_result.json",
        {
            "schema_version": "stage6_shadow_stability_6case_result_v1",
            "generated_at_utc": _utc_now_iso(),
            "subset_cases": list(EXPANSION_CASES),
            "overall_status": stability.get("readiness"),
            "num_runs": stability.get("num_runs"),
            "pass_rate": stability.get("pass_rate"),
            "runtime_success_rate": stability.get("runtime_success_rate"),
            "report_path": str(benchmark_root / "stage6_shadow_stability_6case_report.json"),
            "remaining_blockers": stability.get("remaining_blockers") or [],
        },
    )
    return {
        "audit": audit,
        "smoke": smoke,
        "golden": golden,
        "stability": stability,
    }


def rerun_stage6_shadow_expansion_golden_from_stability(
    repo_root: str | Path,
    *,
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    carla_root = Path(carla_root)

    baseline_info = _pin_baseline(repo_root, prefer_stability_envelope=True)
    _restart_carla(carla_root=carla_root, port=int(carla_port))
    golden = _run_online_shadow_ring(
        repo_root,
        report_prefix="stage6_shadow_expansion_golden_repin",
        compare_baseline_path=Path(str(baseline_info["baseline_metrics_path"])),
        carla_root=carla_root,
        carla_port=int(carla_port),
    )

    expansion_report = load_json(benchmark_root / "stage6_shadow_expansion_report.json", default={}) or {}
    expansion_report.update(
        {
            "generated_at_utc": _utc_now_iso(),
            "baseline_metrics_path": baseline_info.get("baseline_metrics_path"),
            "baseline_commit_path": baseline_info.get("baseline_commit_path"),
            "baseline_decision_path": baseline_info.get("baseline_decision_path"),
            "golden": golden.get("gate_result"),
        }
    )
    notes = list(expansion_report.get("notes") or [])
    if "Six-case golden baseline is repinned from the ready stability envelope when requested." not in notes:
        notes.append("Six-case golden baseline is repinned from the ready stability envelope when requested.")
    expansion_report["notes"] = notes
    dump_json(benchmark_root / "stage6_shadow_expansion_report.json", expansion_report)

    result = load_json(benchmark_root / "stage6_shadow_expansion_result.json", default={}) or {}
    result.update(
        {
            "generated_at_utc": _utc_now_iso(),
            "golden_status": (golden.get("gate_result") or {}).get("overall_status"),
            "report_path": str(benchmark_root / "stage6_shadow_expansion_report.json"),
            "stability_result_path": str(benchmark_root / "stage6_shadow_stability_6case_result.json"),
        }
    )
    dump_json(benchmark_root / "stage6_shadow_expansion_result.json", result)
    return {
        "baseline_info": baseline_info,
        "golden": golden,
        "report_path": str(benchmark_root / "stage6_shadow_expansion_report.json"),
        "result_path": str(benchmark_root / "stage6_shadow_expansion_result.json"),
    }
