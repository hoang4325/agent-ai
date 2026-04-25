from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .io import dump_json, load_json, load_jsonl
from .stage6_control_authority_handoff_sandbox import run_stage6_control_authority_transfer_sandbox


HARD_HEALTH_FLAGS = {
    "perception_hard_stale",
    "stage1_worker_failed",
}
MAX_ROLLBACK_DELAY_TICKS = 1
MIN_POST_ROLLBACK_BASELINE_TICKS = 2


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _positive_step_seconds(rows: list[dict[str, Any]]) -> float:
    timestamps = [float(row["timestamp"]) for row in rows if row.get("timestamp") is not None]
    deltas = [
        timestamps[index + 1] - timestamps[index]
        for index in range(len(timestamps) - 1)
        if (timestamps[index + 1] - timestamps[index]) > 0.0
    ]
    if deltas:
        return float(median(deltas))
    return 0.10


def _estimate_handoff_distance_m(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    rows = sorted(rows, key=lambda row: int(row.get("tick_id", 0)))
    default_step = _positive_step_seconds(rows)
    total_distance = 0.0
    for index, row in enumerate(rows):
        speed_mps = float(row.get("speed_mps") or 0.0)
        if index + 1 < len(rows):
            current_ts = float(row.get("timestamp") or 0.0)
            next_ts = float(rows[index + 1].get("timestamp") or current_ts)
            dt = next_ts - current_ts
            if dt <= 0.0:
                dt = default_step
        else:
            dt = default_step
        total_distance += max(0.0, speed_mps) * float(dt)
    return float(total_distance)


def _takeover_gate_report(
    *,
    repo_root: Path,
    pre_takeover_result: dict[str, Any],
    transfer_report: dict[str, Any],
    transfer_result: dict[str, Any],
) -> dict[str, Any]:
    online_report_dir = Path(str(transfer_report.get("online_report_dir") or ""))
    stage4_dir = online_report_dir / "case_runs" / str(transfer_report.get("case_id") or "") / "stage4"
    trace_rows = load_jsonl(stage4_dir / "control_authority_sandbox_trace.jsonl")
    tick_rows = load_jsonl(stage4_dir / "online_tick_record.jsonl")
    tick_rows_by_tick = {int(row.get("tick_id", -1)): row for row in tick_rows if row.get("tick_id") is not None}
    summary = dict(transfer_report.get("sandbox_summary") or {})

    handoff_rows = [
        row
        for row in trace_rows
        if bool(row.get("handoff_active"))
    ]
    handoff_tick_ids = sorted(int(row.get("tick_id", -1)) for row in handoff_rows if row.get("tick_id") is not None)
    handoff_tick_rows = [
        tick_rows_by_tick[tick_id]
        for tick_id in handoff_tick_ids
        if tick_id in tick_rows_by_tick
    ]
    fallback_rows_during_handoff = [
        row
        for row in handoff_tick_rows
        if bool(row.get("fallback_activated"))
    ]
    hard_health_flags_during_handoff = sorted(
        {
            str(flag)
            for row in handoff_tick_rows
            for flag in (row.get("health_flags") or [])
            if str(flag) in HARD_HEALTH_FLAGS
        }
    )
    over_budget_during_handoff_count = sum(1 for row in handoff_tick_rows if bool(row.get("over_budget")))
    fallback_success_during_handoff = all(bool(row.get("fallback_success")) for row in fallback_rows_during_handoff)
    handoff_distance_estimate_m = _estimate_handoff_distance_m(handoff_rows)
    handoff_elapsed_seconds = max(
        0.0,
        len(handoff_rows) * _positive_step_seconds(handoff_rows),
    )
    max_speed_limit_mps = float(summary.get("max_speed_limit_mps") or 0.0)
    distance_bound_threshold_m = float(max_speed_limit_mps * handoff_elapsed_seconds * 1.25 + 0.05)

    configured_duration_ticks = int(summary.get("handoff_duration_ticks_configured") or -1)
    observed_duration_ticks = int(summary.get("handoff_duration_ticks_observed") or 0)
    rollback_delay_ticks = summary.get("rollback_delay_ticks")
    post_handoff_baseline_tick_count = int(summary.get("post_handoff_baseline_tick_count") or 0)

    criteria = {
        "pre_takeover_dependency_pass": (
            str(pre_takeover_result.get("sandbox_gate_status")) == "pass"
            and not list(pre_takeover_result.get("remaining_blockers") or [])
        ),
        "transfer_sandbox_ready": str(transfer_result.get("overall_status")) == "ready",
        "shadow_authority_applied": bool(summary.get("actual_shadow_takeover_applied")),
        "executed_control_owner_shadow": str(summary.get("executed_control_owner")) == "mpc_shadow_candidate_sandbox",
        "bounded_speed_limit_respected": bool(summary.get("max_speed_within_limit")),
        "bounded_handoff_duration_configured": configured_duration_ticks > 0,
        "bounded_handoff_duration_observed": configured_duration_ticks > 0 and observed_duration_ticks <= configured_duration_ticks,
        "rollback_to_baseline_observed": bool(summary.get("rollback_to_baseline_observed")),
        "rollback_delay_within_limit": rollback_delay_ticks is not None and int(rollback_delay_ticks) <= MAX_ROLLBACK_DELAY_TICKS,
        "post_handoff_baseline_tail_present": post_handoff_baseline_tick_count >= MIN_POST_ROLLBACK_BASELINE_TICKS,
        "no_over_budget_during_handoff": over_budget_during_handoff_count == 0,
        "no_hard_health_flags_during_handoff": not hard_health_flags_during_handoff,
        "fallback_success_during_handoff": fallback_success_during_handoff,
        "bounded_distance_respected": handoff_distance_estimate_m <= distance_bound_threshold_m,
    }

    remaining_blockers: list[str] = []
    for key, passed in criteria.items():
        if not bool(passed):
            remaining_blockers.append(key)

    report = {
        "schema_version": "stage6_takeover_sandbox_gate_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "pre_takeover_dependency": {
            "result_path": str(repo_root / "benchmark" / "stage6_pre_takeover_sandbox_result.json"),
            "sandbox_gate_status": pre_takeover_result.get("sandbox_gate_status"),
            "takeover_discussion_status": pre_takeover_result.get("takeover_discussion_status"),
            "takeover_enable_status": pre_takeover_result.get("takeover_enable_status"),
            "remaining_blockers": list(pre_takeover_result.get("remaining_blockers") or []),
        },
        "transfer_sandbox": {
            "result_path": str(repo_root / "benchmark" / "stage6_control_authority_transfer_sandbox_result.json"),
            "report_path": str(repo_root / "benchmark" / "stage6_control_authority_transfer_sandbox_report.json"),
            "overall_status": transfer_result.get("overall_status"),
            "case_id": transfer_result.get("case_id"),
            "expected_mode": transfer_result.get("expected_mode"),
            "online_report_dir": str(online_report_dir),
            "summary_path": str(stage4_dir / "control_authority_sandbox_summary.json"),
            "trace_path": str(stage4_dir / "control_authority_sandbox_trace.jsonl"),
            "sandbox_summary": summary,
        },
        "derived_metrics": {
            "handoff_tick_ids": handoff_tick_ids,
            "handoff_duration_ticks_configured": configured_duration_ticks,
            "handoff_duration_ticks_observed": observed_duration_ticks,
            "handoff_elapsed_seconds_estimate": handoff_elapsed_seconds,
            "handoff_distance_estimate_m": handoff_distance_estimate_m,
            "distance_bound_threshold_m": distance_bound_threshold_m,
            "fallback_activated_during_handoff_count": len(fallback_rows_during_handoff),
            "fallback_success_during_handoff": fallback_success_during_handoff,
            "over_budget_during_handoff_count": over_budget_during_handoff_count,
            "hard_health_flags_during_handoff": hard_health_flags_during_handoff,
            "rollback_delay_ticks": rollback_delay_ticks,
            "post_handoff_baseline_tick_count": post_handoff_baseline_tick_count,
        },
        "gating_criteria": criteria,
        "readiness_summary": {
            "takeover_sandbox_gate_status": "pass" if not remaining_blockers else "fail",
            "next_scope_status": "ready_for_multi_case_takeover_sandbox_design" if not remaining_blockers else "not_ready",
            "takeover_enable_status": "not_ready",
            "remaining_blockers": remaining_blockers,
        },
        "notes": [
            "This gate is intentionally narrower than takeover enablement. It validates a single bounded authority-transfer sandbox with explicit rollback criteria.",
            "Passing this gate does not authorize takeover rollout. It only clears the system for multi-case takeover sandbox design work.",
        ],
    }
    return report


def run_stage6_takeover_sandbox_gate(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    pre_takeover_result = load_json(benchmark_root / "stage6_pre_takeover_sandbox_result.json", default={}) or {}
    transfer = run_stage6_control_authority_transfer_sandbox(repo_root)
    transfer_report = dict(transfer.get("report") or {})
    transfer_result = dict(transfer.get("result") or {})
    report = _takeover_gate_report(
        repo_root=repo_root,
        pre_takeover_result=pre_takeover_result,
        transfer_report=transfer_report,
        transfer_result=transfer_result,
    )
    result = {
        "schema_version": "stage6_takeover_sandbox_gate_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "takeover_sandbox_gate_status": report["readiness_summary"]["takeover_sandbox_gate_status"],
        "next_scope_status": report["readiness_summary"]["next_scope_status"],
        "takeover_enable_status": report["readiness_summary"]["takeover_enable_status"],
        "remaining_blockers": list(report["readiness_summary"]["remaining_blockers"]),
        "report_path": str(benchmark_root / "stage6_takeover_sandbox_gate_report.json"),
    }
    dump_json(benchmark_root / "stage6_takeover_sandbox_gate_report.json", report)
    dump_json(benchmark_root / "stage6_takeover_sandbox_gate_result.json", result)
    return {
        "report": report,
        "result": result,
    }
