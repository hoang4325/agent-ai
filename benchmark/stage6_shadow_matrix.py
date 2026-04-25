from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from .carla_runtime import ensure_carla_ready, restart_carla
from .io import dump_json, load_json
from .stage6_shadow_contract_audit import run_stage6_shadow_contract_audit
from . import stage6_shadow_expansion as shadow_expansion
from .stage6_shadow_expansion import (
    BASELINE_METRICS,
    EXPANSION_CASES,
    _build_metric_pack,
    _run_online_shadow_ring,
)


@dataclass
class MatrixAttempt:
    case_id: str
    attempt_index: int
    overall_status: str
    runtime_status: str
    gate_status: str
    failure_class: str | None
    retryable_infra: bool
    failure_reason: str | None
    warning_only: bool
    soft_warnings: list[str]
    online_report_dir: Path
    shadow_eval_report_dir: Path
    stage3c_dir: Path


LOW_MEMORY_STAGE4_PROFILE_OVERRIDES = {
    "ml_right_positive_core": "stage4_online_external_watch_shadow_smoke_low_memory",
    "stop_follow_ambiguity_core": "stage4_online_external_watch_stop_aligned_shadow_smoke_low_memory",
    "ml_left_positive_core": "stage4_online_external_watch_shadow_smoke_extended_low_memory",
    "junction_straight_core": "stage4_online_external_watch_shadow_smoke_extended_low_memory",
    "arbitration_stop_during_prepare_right": "stage4_online_external_watch_stop_aligned_shadow_smoke_arbitration_low_memory",
    "fr_lc_commit_no_permission": "stage4_online_external_watch_shadow_smoke_extended_low_memory",
}


INFRA_ERROR_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "infra_carla_rpc_timeout",
        (
            "client.get_world() not ready",
            "time-out of 10000ms while waiting for the simulator",
            "did not become ready",
        ),
    ),
    (
        "infra_carla_native_crash",
        (
            "exit_code: 3221226505",
            "exit_code: -1073740791",
            "exception_access_violation",
        ),
    ),
    (
        "infra_carla_connection_failure",
        (
            "connection refused",
            "failed to connect",
            "no process is listening on port",
        ),
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _case_report(benchmark_report: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in benchmark_report.get("cases") or []:
        if str(case.get("scenario_id")) == case_id:
            return case
    return {}


def _shadow_case(shadow_report: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in shadow_report.get("cases") or []:
        if str(case.get("case_id")) == case_id:
            return case
    return {}


def _stop_target_quality(stage4_dir: Path, stage3c_dir: Path) -> str | None:
    binding_summary = load_json(stage4_dir / "stop_binding_summary.json", default={}) or {}
    if not binding_summary:
        binding_summary = load_json(stage3c_dir / "stop_binding_summary.json", default={}) or {}
    if not binding_summary:
        return None
    if float(binding_summary.get("exact_rate") or 0.0) > 0.0:
        return "exact"
    if float(binding_summary.get("defined_rate") or 0.0) > 0.0:
        return "clean_enough"
    return "missing"


def _read_stage4_log_excerpt(stage4_log: Path) -> tuple[str | None, str | None]:
    if not stage4_log.exists():
        return None, None
    text = stage4_log.read_text(encoding="utf-8", errors="ignore")
    lower = text.lower()
    for failure_class, patterns in INFRA_ERROR_PATTERNS:
        for pattern in patterns:
            if pattern in lower:
                return failure_class, pattern
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return None, (lines[-1] if lines else None)


def _summarize_attempt(
    *,
    case_id: str,
    attempt_index: int,
    result: dict[str, Any],
) -> MatrixAttempt:
    gate_result = result.get("gate_result") or {}
    online_report_dir = Path(str(gate_result.get("online_report_dir") or ""))
    shadow_eval_report_dir = Path(str(gate_result.get("shadow_evaluation_report_dir") or ""))
    benchmark_report = load_json(shadow_eval_report_dir / "benchmark_report.json", default={}) or {}
    mode_resolution = load_json(online_report_dir / "mode_resolution.json", default=[]) or []
    runtime_status = "missing"
    for row in mode_resolution:
        if str(row.get("scenario_id")) == case_id:
            runtime_status = str(row.get("status") or "missing")
            break

    case_report = _case_report(benchmark_report, case_id)
    gate_status = str(case_report.get("status") or "missing")
    overall_status = str(gate_result.get("overall_status") or "fail")
    stage4_log = online_report_dir / "case_runs" / case_id / "logs" / "stage4.log"
    stage3c_dir = online_report_dir / "case_runs" / case_id / "_current_run_artifacts" / "stage3c"

    failure_class: str | None = None
    retryable_infra = False
    failure_reason: str | None = None
    warning_only = False
    soft_warnings = list(case_report.get("soft_warnings") or [])
    if runtime_status != "success":
        failure_class, failure_reason = _read_stage4_log_excerpt(stage4_log)
        if failure_class is None:
            failure_class = "infra_runtime_failure"
            failure_reason = failure_reason or runtime_status
        retryable_infra = True
    elif overall_status == "fail" or gate_status == "fail":
        failure_class = "shadow_gate_fail"
        failure_reason = overall_status if overall_status != "pass" else gate_status
    elif overall_status == "partial" or gate_status == "warning":
        warning_only = True
        failure_reason = overall_status if overall_status != "pass" else gate_status

    return MatrixAttempt(
        case_id=case_id,
        attempt_index=int(attempt_index),
        overall_status=overall_status,
        runtime_status=runtime_status,
        gate_status=gate_status,
        failure_class=failure_class,
        retryable_infra=retryable_infra,
        failure_reason=failure_reason,
        warning_only=warning_only,
        soft_warnings=soft_warnings,
        online_report_dir=online_report_dir,
        shadow_eval_report_dir=shadow_eval_report_dir,
        stage3c_dir=stage3c_dir,
    )


def _extract_case_metrics(case_id: str, result: dict[str, Any]) -> dict[str, Any]:
    gate_result = result.get("gate_result") or {}
    online_report_dir = Path(str(gate_result.get("online_report_dir") or ""))
    shadow_eval_report_dir = Path(str(gate_result.get("shadow_evaluation_report_dir") or ""))
    benchmark_report = load_json(shadow_eval_report_dir / "benchmark_report.json", default={}) or {}
    case_report = _case_report(benchmark_report, case_id)
    shadow_case = _shadow_case(result.get("shadow_report") or {}, case_id)
    shadow_summary = shadow_case.get("shadow_summary") or {}
    stage4_dir = online_report_dir / "case_runs" / case_id / "stage4"
    stage3c_dir = online_report_dir / "case_runs" / case_id / "_current_run_artifacts" / "stage3c"
    stop_alignment = load_json(stage4_dir / "stop_scene_alignment.json", default={}) or {}
    if not stop_alignment:
        stop_alignment = load_json(stage3c_dir / "stop_scene_alignment.json", default={}) or {}
    metrics = case_report.get("metrics") or {}
    return {
        "shadow_feasibility_rate": (metrics.get("shadow_feasibility_rate") or {}).get("value"),
        "shadow_solver_timeout_rate": shadow_summary.get("shadow_solver_timeout_rate"),
        "shadow_planner_execution_latency_ms": (metrics.get("shadow_planner_execution_latency_ms") or {}).get("value"),
        "shadow_realized_support_rate": (metrics.get("shadow_realized_support_rate") or {}).get("value")
        or shadow_summary.get("shadow_realized_support_rate"),
        "shadow_realized_behavior_match_rate": (metrics.get("shadow_realized_behavior_match_rate") or {}).get("value")
        or shadow_summary.get("shadow_realized_behavior_match_rate"),
        "shadow_realized_stop_distance_mae_m": shadow_summary.get("shadow_realized_stop_distance_mae_m"),
        "shadow_baseline_disagreement_rate": (metrics.get("shadow_baseline_disagreement_rate") or {}).get("value"),
        "alignment_ready": stop_alignment.get("alignment_ready"),
        "alignment_consistent": stop_alignment.get("alignment_consistent"),
        "stop_target_quality": _stop_target_quality(stage4_dir, stage3c_dir),
        "proposal_rows": shadow_case.get("proposal_rows"),
    }


def run_stage6_shadow_matrix(
    repo_root: str | Path,
    *,
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
    retry_attempts: int = 1,
    case_startup_cooldown_seconds: float = 8.0,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    carla_root = Path(carla_root)
    baseline_path = repo_root / BASELINE_METRICS
    compare_baseline_path = baseline_path if baseline_path.exists() else None

    ensure_carla_ready(carla_root=carla_root, port=int(carla_port))
    original_stage4_profile_overrides = {
        case_id: shadow_expansion.STAGE4_PROFILE_OVERRIDES.get(case_id)
        for case_id in EXPANSION_CASES
    }
    shadow_expansion.STAGE4_PROFILE_OVERRIDES.update(LOW_MEMORY_STAGE4_PROFILE_OVERRIDES)
    final_stage3c_dirs: dict[str, Path] = {}
    case_reports: list[dict[str, Any]] = []
    observed_failure_classes: set[str] = set()
    recovered_cases: list[str] = []

    try:
        for case_id in EXPANSION_CASES:
            attempts_payload: list[dict[str, Any]] = []
            final_result: dict[str, Any] | None = None
            final_attempt: MatrixAttempt | None = None
            recovered_after_retry = False

            for attempt_index in range(1, int(retry_attempts) + 2):
                restart_carla(carla_root=carla_root, port=int(carla_port))
                time.sleep(float(case_startup_cooldown_seconds))
                result = _run_online_shadow_ring(
                    repo_root,
                    report_prefix=f"stage6_shadow_matrix_{case_id}_attempt{attempt_index}",
                    compare_baseline_path=compare_baseline_path,
                    carla_root=carla_root,
                    carla_port=int(carla_port),
                    case_ids=[case_id],
                    restart_carla_between_cases=False,
                )
                attempt = _summarize_attempt(case_id=case_id, attempt_index=attempt_index, result=result)
                attempts_payload.append(
                    {
                        "attempt_index": attempt.attempt_index,
                        "overall_status": attempt.overall_status,
                        "runtime_status": attempt.runtime_status,
                        "gate_status": attempt.gate_status,
                        "failure_class": attempt.failure_class,
                        "retryable_infra": attempt.retryable_infra,
                        "failure_reason": attempt.failure_reason,
                        "warning_only": attempt.warning_only,
                        "soft_warnings": attempt.soft_warnings,
                        "online_report_dir": str(attempt.online_report_dir),
                        "shadow_evaluation_report_dir": str(attempt.shadow_eval_report_dir),
                    }
                )
                if attempt.failure_class:
                    observed_failure_classes.add(str(attempt.failure_class))
                final_result = result
                final_attempt = attempt
                if attempt.failure_class is None:
                    recovered_after_retry = attempt_index > 1
                    break
                if not attempt.retryable_infra or attempt_index > int(retry_attempts):
                    break

            assert final_result is not None
            assert final_attempt is not None
            if recovered_after_retry:
                recovered_cases.append(case_id)

            final_stage3c_dirs[case_id] = final_attempt.stage3c_dir
            case_metrics = _extract_case_metrics(case_id, final_result)
            final_status = "pass"
            if final_attempt.failure_class is not None:
                final_status = str(final_attempt.failure_class)
            elif final_attempt.warning_only:
                final_status = "pass_with_warnings"
            case_reports.append(
                {
                    "case_id": case_id,
                    "final_status": final_status,
                    "recovered_after_retry": recovered_after_retry,
                    "attempt_count": len(attempts_payload),
                    "attempts": attempts_payload,
                    "final_metrics": case_metrics,
                    "notes": list(final_attempt.soft_warnings),
                }
            )

    finally:
        for case_id, profile_name in original_stage4_profile_overrides.items():
            if profile_name is None:
                shadow_expansion.STAGE4_PROFILE_OVERRIDES.pop(case_id, None)
            else:
                shadow_expansion.STAGE4_PROFILE_OVERRIDES[case_id] = profile_name

    run_stage6_shadow_contract_audit(repo_root, case_stage3c_dirs=final_stage3c_dirs)
    _build_metric_pack(repo_root, case_ids=list(EXPANSION_CASES))

    failed_cases = [
        case["case_id"]
        for case in case_reports
        if case["final_status"] not in {"pass", "pass_with_warnings"}
    ]
    warning_cases = [case["case_id"] for case in case_reports if case["final_status"] == "pass_with_warnings"]
    runtime_success_cases = [
        case["case_id"]
        for case in case_reports
        if str((case.get("attempts") or [{}])[-1].get("runtime_status")) == "success"
    ]
    overall_status = "pass"
    if failed_cases:
        overall_status = "fail"
    elif recovered_cases or warning_cases:
        overall_status = "partial"

    report = {
        "schema_version": "stage6_shadow_matrix_v1",
        "generated_at_utc": _utc_now_iso(),
        "subset_cases": list(EXPANSION_CASES),
        "retry_attempts": int(retry_attempts),
        "compare_baseline_path": str(compare_baseline_path) if compare_baseline_path else None,
        "overall_status": overall_status,
        "case_pass_rate": float((len(case_reports) - len(failed_cases)) / len(case_reports)) if case_reports else 0.0,
        "runtime_success_rate": float(len(runtime_success_cases) / len(case_reports)) if case_reports else 0.0,
        "recovered_cases": recovered_cases,
        "warning_cases": warning_cases,
        "observed_failure_classes": sorted(observed_failure_classes),
        "cases": case_reports,
        "notes": [
            "Each case runs in its own CARLA lifecycle so simulator bring-up failures do not contaminate the rest of the six-case ring.",
            "Retry is reserved for infrastructure failures only; shadow gate failures are reported directly and are not retried.",
            "A warning-only shadow evaluation remains a valid bring-up pass; the matrix runner distinguishes soft drift from runtime or contract failure.",
            "The matrix runner forces low-memory Stage 4 profiles so the six-case bring-up path trades image/point density for lower simulator memory pressure.",
            "Baseline remains the execution authority. This runner validates proposal-only six-case bring-up.",
        ],
    }
    result = {
        "schema_version": "stage6_shadow_matrix_result_v1",
        "generated_at_utc": _utc_now_iso(),
        "subset_cases": list(EXPANSION_CASES),
        "overall_status": overall_status,
        "passed_cases": [
            case["case_id"]
            for case in case_reports
            if case["final_status"] in {"pass", "pass_with_warnings"}
        ],
        "failed_cases": failed_cases,
        "recovered_cases": recovered_cases,
        "warning_cases": warning_cases,
        "report_path": str(benchmark_root / "stage6_shadow_matrix_report.json"),
    }
    dump_json(benchmark_root / "stage6_shadow_matrix_report.json", report)
    dump_json(benchmark_root / "stage6_shadow_matrix_result.json", result)
    return {"report": report, "result": result}
