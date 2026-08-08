from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_json


PROMOTED_RING_CASES = [
    "ml_right_positive_core",
    "ml_left_positive_core",
    "junction_straight_core",
    "fr_lc_commit_no_permission",
    "stop_follow_ambiguity_core",
    "arbitration_stop_during_prepare_right",
]
STOP_CASES = {
    "stop_follow_ambiguity_core",
    "arbitration_stop_during_prepare_right",
}
MAX_HANDOFF_DURATION_P95_TICKS = 3.0
MAX_ROLLBACK_DELAY_P95_TICKS = 1.0
MAX_HANDOFF_DISTANCE_P95_M = 0.10
MIN_HANDOFF_FALLBACK_CASES = 2


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _case_rows_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in report.get("cases") or []:
        case_id = str(row.get("case_id") or "")
        if case_id:
            rows[case_id] = dict(row)
    return rows


def _float_value(payload: dict[str, Any], key: str, field: str) -> float | None:
    stats = dict(payload.get(key) or {})
    value = stats.get(field)
    if value is None:
        return None
    return float(value)


def run_stage6_takeover_enable_gate(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"

    pre_takeover_result = load_json(benchmark_root / "stage6_pre_takeover_sandbox_result.json", default={}) or {}
    ring_result = load_json(benchmark_root / "stage6_takeover_ring_6case_stability_result.json", default={}) or {}
    ring_report = load_json(benchmark_root / "stage6_takeover_ring_6case_stability_report.json", default={}) or {}
    timeout_fallback_report = load_json(benchmark_root / "stage6_fallback_authority_ring_report.json", default={}) or {}
    non_timeout_fallback_report = load_json(benchmark_root / "stage6_non_timeout_fallback_ring_report.json", default={}) or {}
    organic_fallback_report = load_json(benchmark_root / "stage6_organic_fallback_ring_report.json", default={}) or {}

    ring_cases = _case_rows_by_id(ring_report)
    aggregate_metrics = dict(ring_report.get("aggregate_metrics") or {})
    handoff_duration_p95 = _float_value(aggregate_metrics, "handoff_duration_ticks_observed", "p95")
    rollback_delay_p95 = _float_value(aggregate_metrics, "rollback_delay_ticks", "p95")
    handoff_distance_p95 = _float_value(aggregate_metrics, "handoff_distance_estimate_m", "p95")

    stop_case_support: dict[str, dict[str, Any]] = {}
    for case_id in STOP_CASES:
        case_row = dict(ring_cases.get(case_id) or {})
        stop_case_support[case_id] = {
            "stop_alignment_ready_rate": case_row.get("stop_alignment_ready_rate"),
            "stop_alignment_consistent_rate": case_row.get("stop_alignment_consistent_rate"),
            "stop_target_exact_rate": case_row.get("stop_target_exact_rate"),
            "stop_final_error_supported_rate": case_row.get("stop_final_error_supported_rate"),
        }

    handoff_fallback_case_ids = [
        str(row.get("case_id"))
        for row in ring_report.get("cases") or []
        if float(((row.get("fallback_activated_during_handoff_count") or {}).get("mean") or 0.0)) > 0.0
    ]

    timeout_aggregate = dict(timeout_fallback_report.get("aggregate") or {})
    non_timeout_aggregate = dict(non_timeout_fallback_report.get("aggregate") or {})
    organic_aggregate = dict(organic_fallback_report.get("aggregate") or {})

    criteria = {
        "pre_takeover_gate_pass": (
            str(pre_takeover_result.get("sandbox_gate_status")) == "pass"
            and not list(pre_takeover_result.get("remaining_blockers") or [])
        ),
        "six_case_ring_repeatable": (
            str(ring_result.get("overall_status")) == "ready"
            and float(ring_result.get("pass_rate") or 0.0) == 1.0
            and float(ring_result.get("runtime_success_rate") or 0.0) == 1.0
            and not list(ring_result.get("remaining_blockers") or [])
        ),
        "handoff_duration_p95_within_bound": handoff_duration_p95 is not None and handoff_duration_p95 <= MAX_HANDOFF_DURATION_P95_TICKS,
        "rollback_delay_p95_within_bound": rollback_delay_p95 is not None and rollback_delay_p95 <= MAX_ROLLBACK_DELAY_P95_TICKS,
        "handoff_distance_p95_within_bound": handoff_distance_p95 is not None and handoff_distance_p95 <= MAX_HANDOFF_DISTANCE_P95_M,
        "stop_case_alignment_support_clean": all(
            float(stop_case_support[case_id]["stop_alignment_ready_rate"] or 0.0) == 1.0
            and float(stop_case_support[case_id]["stop_alignment_consistent_rate"] or 0.0) == 1.0
            and float(stop_case_support[case_id]["stop_target_exact_rate"] or 0.0) == 1.0
            and float(stop_case_support[case_id]["stop_final_error_supported_rate"] or 0.0) == 1.0
            for case_id in STOP_CASES
        ),
        "timeout_fallback_ring_clean": (
            bool(timeout_aggregate.get("ready"))
            and int(timeout_aggregate.get("activated_case_count") or 0) == int(timeout_aggregate.get("probe_case_count") or 0)
        ),
        "non_timeout_fallback_ring_clean": (
            bool(non_timeout_aggregate.get("ready"))
            and int(non_timeout_aggregate.get("activated_case_count") or 0) == int(non_timeout_aggregate.get("fallback_required_case_count") or 0)
        ),
        "organic_fallback_candidate_coverage_clean": (
            bool(organic_aggregate.get("ready"))
            and str(organic_aggregate.get("coverage_scope")) == "promoted_ring_organic_candidate_observation"
            and int(organic_aggregate.get("activated_case_count") or 0) == int(organic_aggregate.get("fallback_required_case_count") or 0)
        ),
        "handoff_window_fallback_observed_on_multiple_cases": len(handoff_fallback_case_ids) >= MIN_HANDOFF_FALLBACK_CASES,
    }

    remaining_blockers = [key for key, passed in criteria.items() if not bool(passed)]
    gate_status = "pass" if not remaining_blockers else "fail"
    next_scope_status = "ready_for_explicit_takeover_rollout_design" if gate_status == "pass" else "not_ready"

    report = {
        "schema_version": "stage6_takeover_enable_gate_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "promoted_ring_cases": list(PROMOTED_RING_CASES),
        "dependencies": {
            "pre_takeover_result_path": str(benchmark_root / "stage6_pre_takeover_sandbox_result.json"),
            "ring_stability_result_path": str(benchmark_root / "stage6_takeover_ring_6case_stability_result.json"),
            "timeout_fallback_report_path": str(benchmark_root / "stage6_fallback_authority_ring_report.json"),
            "non_timeout_fallback_report_path": str(benchmark_root / "stage6_non_timeout_fallback_ring_report.json"),
            "organic_fallback_report_path": str(benchmark_root / "stage6_organic_fallback_ring_report.json"),
        },
        "derived_metrics": {
            "handoff_duration_p95_ticks": handoff_duration_p95,
            "rollback_delay_p95_ticks": rollback_delay_p95,
            "handoff_distance_p95_m": handoff_distance_p95,
            "handoff_fallback_case_ids": handoff_fallback_case_ids,
            "stop_case_support": stop_case_support,
            "timeout_fallback_aggregate": timeout_aggregate,
            "non_timeout_fallback_aggregate": non_timeout_aggregate,
            "organic_fallback_aggregate": organic_aggregate,
        },
        "gating_criteria": criteria,
        "readiness_summary": {
            "takeover_enable_gate_status": gate_status,
            "next_scope_status": next_scope_status,
            "takeover_enable_status": "not_ready",
            "remaining_blockers": remaining_blockers,
        },
        "notes": [
            "This gate consolidates bounded authority-transfer, rollback, and fallback evidence on the promoted six-case ring.",
            "Passing this gate still does not enable takeover automatically. It only clears explicit rollout design work with separate operational approval.",
        ],
    }
    result = {
        "schema_version": "stage6_takeover_enable_gate_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "takeover_enable_gate_status": gate_status,
        "next_scope_status": next_scope_status,
        "takeover_enable_status": "not_ready",
        "remaining_blockers": remaining_blockers,
        "report_path": str(benchmark_root / "stage6_takeover_enable_gate_report.json"),
    }
    dump_json(benchmark_root / "stage6_takeover_enable_gate_report.json", report)
    dump_json(benchmark_root / "stage6_takeover_enable_gate_result.json", result)
    return {
        "report": report,
        "result": result,
    }
