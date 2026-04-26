from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_yaml, resolve_repo_path
from .metrics import evaluate_case_metrics
from .frozen_corpus_support import overlay_case_with_frozen_corpus


DRIVING_TABLE_METRICS = (
    "route_completion_rate",
    "collision_count",
    "collision_rate_per_km",
    "scenario_success_rate",
)


@dataclass
class ThresholdEvaluation:
    passed: bool
    reason: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _evaluate_threshold(value: Any, threshold: dict[str, Any]) -> ThresholdEvaluation:
    if value is None:
        return ThresholdEvaluation(False, "metric_unavailable")
    if "min" in threshold and float(value) < float(threshold["min"]):
        return ThresholdEvaluation(False, f"value {value} < min {threshold['min']}")
    if "max" in threshold and float(value) > float(threshold["max"]):
        return ThresholdEvaluation(False, f"value {value} > max {threshold['max']}")
    if "equals" in threshold and value != threshold["equals"]:
        return ThresholdEvaluation(False, f"value {value} != equals {threshold['equals']}")
    if threshold.get("nonempty") and value in (None, "", [], {}):
        return ThresholdEvaluation(False, "value_empty")
    return ThresholdEvaluation(True, "ok")


def _compare_metric(
    *,
    current_value: Any,
    baseline_value: Any,
    trend_preference: str | None,
    abs_tolerance: float | None = None,
) -> dict[str, Any]:
    if current_value is None or baseline_value is None:
        return {
            "available": False,
            "baseline_value": baseline_value,
            "current_value": current_value,
            "delta": None,
            "regression": False,
            "reason": "missing_current_or_baseline_value",
        }
    delta = None
    if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
        delta = float(current_value) - float(baseline_value)
    tolerance = abs(float(abs_tolerance)) if abs_tolerance is not None else 0.0
    regression = False
    if trend_preference == "higher_better":
        regression = bool(float(current_value) < float(baseline_value) - tolerance)
    elif trend_preference == "lower_better":
        regression = bool(float(current_value) > float(baseline_value) + tolerance)
    elif trend_preference == "exact_match":
        regression = bool(current_value != baseline_value and (delta is None or abs(delta) > tolerance))
    return {
        "available": True,
        "baseline_value": baseline_value,
        "current_value": current_value,
        "delta": delta,
        "regression": regression,
        "reason": ("within_tolerance" if not regression and tolerance > 0.0 and delta is not None and abs(delta) <= tolerance else "ok"),
    }


def _load_case_spec(case_id: str, cases_dir: Path) -> dict[str, Any]:
    case_path = cases_dir / f"{case_id}.yaml"
    payload = load_yaml(case_path)
    payload["_case_path"] = str(case_path)
    return payload


def _load_set_cases(set_name: str, benchmark_config: dict[str, Any], repo_root: Path) -> list[str]:
    set_path = resolve_repo_path(repo_root, benchmark_config["sets"][set_name])
    payload = load_yaml(set_path)
    return list(payload.get("cases", []))


def _case_metric_thresholds(case_spec: dict[str, Any], benchmark_config: dict[str, Any], metric_name: str) -> dict[str, Any] | None:
    case_thresholds = case_spec.get("metric_thresholds") or {}
    if metric_name in case_thresholds:
        return case_thresholds[metric_name]
    return (benchmark_config.get("default_metric_thresholds") or {}).get(metric_name)


def _safe_metric_value(metrics: dict[str, Any], metric_name: str) -> Any:
    return (metrics.get(metric_name) or {}).get("value")


def _case_semantic_quality(metrics: dict[str, Any], availability: dict[str, Any]) -> dict[str, Any]:
    binding_metric_names = [
        "blocker_binding_accuracy",
        "critical_actor_detected_rate",
        "critical_actor_track_continuity_rate",
        "lane_assignment_consistency",
    ]
    arbitration_metric_names = ["arbitration_conflict_count", "behavior_execution_mismatch_rate"]
    route_metric_names = ["route_context_consistency", "route_influence_correctness"]

    metric_states = {
        "ok": 0,
        "metric_skipped": 0,
        "metric_degraded": 0,
    }
    for metric in metrics.values():
        state = str(metric.get("evaluation_state", "ok"))
        if state in metric_states:
            metric_states[state] += 1

    def _available_fraction(names: list[str]) -> float | None:
        values = [_safe_metric_value(metrics, name) for name in names]
        available = [value for value in values if value is not None]
        if not values:
            return None
        return float(len(available) / len(values))

    return {
        "binding_quality": {
            "metric_values": {name: _safe_metric_value(metrics, name) for name in binding_metric_names},
            "available_fraction": _available_fraction(binding_metric_names),
            "ground_truth_available": bool((availability.get("ground_truth_actor_roles") or {}).get("available")),
            "critical_binding_available": bool((availability.get("critical_actor_binding") or {}).get("available")),
        },
        "arbitration_quality": {
            "metric_values": {name: _safe_metric_value(metrics, name) for name in arbitration_metric_names},
            "available_fraction": _available_fraction(arbitration_metric_names),
            "arbitration_events_available": bool((availability.get("behavior_arbitration_events") or {}).get("available")),
        },
        "route_session_quality": {
            "metric_values": {name: _safe_metric_value(metrics, name) for name in route_metric_names},
            "available_fraction": _available_fraction(route_metric_names),
            "route_session_binding_available": bool((availability.get("route_session_binding") or {}).get("available")),
        },
        "metric_state_counts": metric_states,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _round_optional(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _build_simulation_table_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    values_by_metric: dict[str, list[float]] = {name: [] for name in DRIVING_TABLE_METRICS}

    for case_result in case_results:
        driving_metrics = case_result.get("driving_metrics") or {}
        compact_metrics: dict[str, Any] = {}
        has_any_value = False
        for metric_name in DRIVING_TABLE_METRICS:
            metric_result = driving_metrics.get(metric_name) or {}
            value = metric_result.get("value")
            compact_metrics[metric_name] = value
            if value is not None:
                has_any_value = True
                values_by_metric[metric_name].append(float(value))
        per_case.append(
            {
                "scenario_id": case_result["scenario_id"],
                "status": case_result["status"],
                "has_driving_metrics": has_any_value,
                "metrics": compact_metrics,
            }
        )

    rc_mean = _mean(values_by_metric["route_completion_rate"])
    collision_count_values = values_by_metric["collision_count"]
    collision_rate_mean = _mean(values_by_metric["collision_rate_per_km"])
    success_mean = _mean(values_by_metric["scenario_success_rate"])

    return {
        "schema_version": "simulation_table_summary_v1",
        "num_cases": len(case_results),
        "cases_with_driving_metrics": sum(1 for row in per_case if row["has_driving_metrics"]),
        "cases_missing_driving_metrics": [
            row["scenario_id"] for row in per_case if not row["has_driving_metrics"]
        ],
        "table_metrics": {
            "route_completion_rate_mean": _round_optional(rc_mean),
            "route_completion_pct_mean": _round_optional(None if rc_mean is None else rc_mean * 100.0, 3),
            "collision_count_total": (
                None if not collision_count_values else int(sum(collision_count_values))
            ),
            "collision_count_mean_per_run": _round_optional(_mean(collision_count_values)),
            "collision_rate_per_km_mean": _round_optional(collision_rate_mean),
            "success_rate": _round_optional(success_mean),
            "success_rate_pct": _round_optional(None if success_mean is None else success_mean * 100.0, 3),
        },
        "per_case": per_case,
    }


def _build_case_result(
    *,
    case_spec: dict[str, Any],
    benchmark_config: dict[str, Any],
    metric_registry: dict[str, Any],
    repo_root: Path,
    baseline_cases: dict[str, Any],
    enable_baseline_compare: bool,
    gate_name: str,
    generated_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = case_spec["scenario_id"]
    case_status = case_spec.get("status", "planned")
    if case_status == "planned":
        result = {
            "scenario_id": case_id,
            "status": "skipped",
            "case_status": case_status,
            "gate_status": "skipped_planned_case",
            "hard_failures": [],
            "soft_warnings": [],
            "notes": [str(case_spec.get("notes", ""))] if case_spec.get("notes") else [],
            "artifact_availability": {},
            "generated_artifacts": {},
            "missing_artifact_warnings": [],
            "metrics": {},
            "baseline_regressions": [],
        }
        return result, {"hard_fail": False, "warning": True, "skipped": True}

    metrics, availability, resolved_artifacts, _metric_warnings = evaluate_case_metrics(
        case_spec=case_spec,
        metric_registry=metric_registry,
        repo_root=repo_root,
        generated_root=generated_root,
    )
    primary_metrics = set(case_spec.get("primary_metrics") or [])
    secondary_metrics = set(case_spec.get("secondary_metrics") or [])
    hard_failures: list[str] = []
    soft_warnings: list[str] = []
    baseline_regressions: list[str] = []

    for metric_name in sorted(primary_metrics | secondary_metrics):
        metric_result = metrics.get(metric_name) or {}
        degraded_reason = metric_result.get("degraded_reason")
        if degraded_reason and metric_result.get("gate_usage") != "diagnostic_only":
            soft_warnings.append(f"{metric_name}:{degraded_reason}")

    for artifact_name in case_spec.get("mandatory_artifacts") or []:
        if not (availability.get(artifact_name) or {}).get("available", False):
            hard_failures.append(f"missing_mandatory_artifact:{artifact_name}")

    missing_artifact_warnings = [
        f"{artifact_name}:{','.join((artifact_info.get('notes') or []) or ['missing'])}"
        for artifact_name, artifact_info in availability.items()
        if not artifact_info.get("available")
    ]

    registry_metrics = metric_registry.get("metrics", {})
    baseline_case = baseline_cases.get(case_id, {}) if enable_baseline_compare else {}
    baseline_metric_values = (baseline_case.get("metrics") or {}) if isinstance(baseline_case, dict) else {}

    for metric_name in sorted(primary_metrics | secondary_metrics):
        metadata = registry_metrics.get(metric_name, {})
        metric_result = metrics.get(metric_name, {})
        threshold = _case_metric_thresholds(case_spec, benchmark_config, metric_name)
        if threshold:
            threshold_result = _evaluate_threshold(metric_result.get("value"), threshold)
            metric_result["threshold"] = threshold
            metric_result["threshold_passed"] = threshold_result.passed
            metric_result["threshold_reason"] = threshold_result.reason
            if not threshold_result.passed:
                if metric_result.get("gate_usage") == "diagnostic_only":
                    pass
                elif metric_result.get("gate_usage") == "hard_gate" and metric_name in primary_metrics:
                    hard_failures.append(f"{metric_name}:{threshold_result.reason}")
                else:
                    soft_warnings.append(f"{metric_name}:{threshold_result.reason}")
        else:
            metric_result["threshold"] = None
            metric_result["threshold_passed"] = None
            metric_result["threshold_reason"] = "no_threshold_defined"

        if enable_baseline_compare:
            comparison = _compare_metric(
                current_value=metric_result.get("value"),
                baseline_value=baseline_metric_values.get(metric_name),
                trend_preference=metadata.get("trend_preference"),
                abs_tolerance=metadata.get("baseline_compare_abs_tolerance"),
            )
            metric_result["baseline_comparison"] = comparison
            if comparison.get("regression") and metric_result.get("gate_usage") == "diagnostic_only":
                pass
            elif comparison.get("regression") and metric_result.get("maturity") == "hard_gate_ready" and metric_name in primary_metrics:
                hard_failures.append(f"{metric_name}:baseline_regression")
                baseline_regressions.append(metric_name)
            elif comparison.get("regression"):
                soft_warnings.append(f"{metric_name}:baseline_regression")
                baseline_regressions.append(metric_name)
        else:
            metric_result["baseline_comparison"] = None

    case_report_status = "pass"
    if hard_failures:
        case_report_status = "fail"
    elif soft_warnings:
        case_report_status = "warning"

    result = {
        "scenario_id": case_id,
        "status": case_report_status,
        "case_status": case_status,
        "gate_status": gate_name,
        "hard_failures": hard_failures,
        "soft_warnings": soft_warnings,
        "notes": [str(case_spec.get("notes", ""))] if case_spec.get("notes") else [],
        "artifact_availability": availability,
        "generated_artifacts": {
            name: info
            for name, info in availability.items()
            if info.get("source") in {"generated", "existing"}
            and name in {"ground_truth_actor_roles", "critical_actor_binding", "behavior_arbitration_events", "route_session_binding"}
        },
        "resolved_artifact_paths": {
            name: (str(path) if path is not None else None)
            for name, path in resolved_artifacts.items()
        },
        "missing_artifact_warnings": missing_artifact_warnings,
        "metrics": {name: metrics[name] for name in sorted(primary_metrics | secondary_metrics)},
        "driving_metrics": {
            name: metrics[name]
            for name in DRIVING_TABLE_METRICS
            if name in metrics
        },
        "semantic_quality": _case_semantic_quality(metrics, availability),
        "baseline_regressions": baseline_regressions,
    }
    summary_flags = {
        "hard_fail": bool(hard_failures),
        "warning": bool(soft_warnings),
        "skipped": False,
    }
    return result, summary_flags


def run_benchmark(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    set_name: str,
    compare_baseline_path: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_config = load_yaml(resolve_repo_path(repo_root, config_path))
    metric_registry = load_yaml(resolve_repo_path(repo_root, benchmark_config["metric_registry"]))
    cases_dir = resolve_repo_path(repo_root, benchmark_config["cases_dir"])

    case_ids = _load_set_cases(set_name, benchmark_config, repo_root)
    if not case_ids:
        raise ValueError(f"benchmark set '{set_name}' is empty or missing")

    baseline_path = compare_baseline_path or ((benchmark_config.get("baseline", {}) or {}).get("default_metrics_path"))
    baseline_payload = load_yaml(resolve_repo_path(repo_root, baseline_path)) if baseline_path else {}
    baseline_cases = baseline_payload.get("cases", {}) if isinstance(baseline_payload, dict) else {}
    enable_baseline_compare = bool(compare_baseline_path)

    default_report_dir = Path(benchmark_config["reports_dir"]) / f"{set_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_root = resolve_repo_path(repo_root, report_dir or default_report_dir)
    report_root.mkdir(parents=True, exist_ok=True)

    case_results: list[dict[str, Any]] = []
    failed_cases: list[str] = []
    warning_cases: list[str] = []
    skipped_cases: list[str] = []
    metric_baseline: dict[str, Any] = {}

    for case_id in case_ids:
        case_spec = _load_case_spec(case_id, cases_dir)
        case_spec, overlay_status = overlay_case_with_frozen_corpus(
            repo_root=repo_root,
            benchmark_config=benchmark_config,
            case_spec=case_spec,
            benchmark_mode="replay_regression",
        )
        if overlay_status:
            case_spec["frozen_corpus_overlay_status"] = overlay_status
        case_result, flags = _build_case_result(
            case_spec=case_spec,
            benchmark_config=benchmark_config,
            metric_registry=metric_registry,
            repo_root=repo_root,
            baseline_cases=baseline_cases,
            enable_baseline_compare=enable_baseline_compare,
            gate_name=set_name,
            generated_root=report_root / "_generated_semantics",
        )
        case_results.append(case_result)
        metric_baseline[case_id] = {
            "status": case_result["status"],
            "case_status": case_result["case_status"],
            "metrics": {
                metric_name: metric_result.get("value")
                for metric_name, metric_result in case_result["metrics"].items()
            },
        }
        if flags["hard_fail"]:
            failed_cases.append(case_id)
        elif flags["warning"]:
            warning_cases.append(case_id)
        if flags["skipped"]:
            skipped_cases.append(case_id)

    overall_status = "pass"
    if failed_cases:
        overall_status = "fail"
    elif warning_cases or skipped_cases:
        overall_status = "partial"

    benchmark_report = {
        "benchmark_version": benchmark_config["version"],
        "benchmark_name": benchmark_config["name"],
        "generated_at_utc": _utc_now_iso(),
        "set": set_name,
        "report_dir": str(report_root),
        "compare_baseline": str(resolve_repo_path(repo_root, baseline_path)) if enable_baseline_compare and baseline_path else None,
        "semantic_hardening_enabled": True,
        "semantic_hardening_version": "v1.1",
        "simulation_table_summary": _build_simulation_table_summary(case_results),
        "cases": case_results,
    }

    semantic_degraded_cases = [
        case_result["scenario_id"]
        for case_result in case_results
        if ((case_result.get("semantic_quality") or {}).get("metric_state_counts") or {}).get("metric_degraded", 0) > 0
    ]
    semantic_skipped_cases = [
        case_result["scenario_id"]
        for case_result in case_results
        if ((case_result.get("semantic_quality") or {}).get("metric_state_counts") or {}).get("metric_skipped", 0) > 0
    ]

    def _low_quality(case_result: dict[str, Any], key: str, threshold: float = 0.5) -> bool:
        quality = (case_result.get("semantic_quality") or {}).get(key) or {}
        fraction = quality.get("available_fraction")
        if fraction is None:
            return True
        return float(fraction) < threshold

    gate_result = {
        "benchmark_version": benchmark_config["version"],
        "set": set_name,
        "overall_status": overall_status,
        "failed_cases": failed_cases,
        "warning_cases": sorted(set(warning_cases + skipped_cases)),
        "missing_artifact_warning_cases": [
            case_result["scenario_id"]
            for case_result in case_results
            if case_result.get("missing_artifact_warnings")
        ],
        "semantic_degraded_cases": semantic_degraded_cases,
        "semantic_skipped_cases": semantic_skipped_cases,
        "binding_quality_warning_cases": [
            case_result["scenario_id"]
            for case_result in case_results
            if _low_quality(case_result, "binding_quality")
        ],
        "arbitration_quality_warning_cases": [
            case_result["scenario_id"]
            for case_result in case_results
            if _low_quality(case_result, "arbitration_quality")
        ],
        "route_session_quality_warning_cases": [
            case_result["scenario_id"]
            for case_result in case_results
            if _low_quality(case_result, "route_session_quality")
        ],
        "notes": [
            "planned cases are skipped rather than hard-failed in v1",
            "diagnostic_only metrics do not affect gate outcome",
            "online runtime metrics remain soft_guardrail until Stage 4 artifacts are stable",
            "semantic/binding artifacts may be generated on the fly into benchmark/reports/_generated_semantics",
        ],
    }

    regression_diff = {
        "benchmark_version": benchmark_config["version"],
        "set": set_name,
        "baseline_source": (str(resolve_repo_path(repo_root, baseline_path)) if enable_baseline_compare and baseline_path else None),
        "cases": {
            case_result["scenario_id"]: {
                metric_name: metric_result.get("baseline_comparison")
                for metric_name, metric_result in case_result["metrics"].items()
                if metric_result.get("baseline_comparison") is not None
            }
            for case_result in case_results
        },
    }

    baseline_snapshot = {
        "benchmark_version": benchmark_config["version"],
        "generated_at_utc": _utc_now_iso(),
        "set": set_name,
        "cases": metric_baseline,
    }

    dump_json(report_root / "benchmark_report.json", benchmark_report)
    dump_json(report_root / "simulation_table_summary.json", benchmark_report["simulation_table_summary"])
    dump_json(report_root / "gate_result.json", gate_result)
    dump_json(report_root / "regression_diff.json", regression_diff)
    dump_json(report_root / "baseline_snapshot.json", baseline_snapshot)

    return {
        "report_dir": str(report_root),
        "benchmark_report": benchmark_report,
        "simulation_table_summary": benchmark_report["simulation_table_summary"],
        "gate_result": gate_result,
        "regression_diff": regression_diff,
        "baseline_snapshot": baseline_snapshot,
    }
