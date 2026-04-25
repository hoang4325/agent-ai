from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .io import dump_json, load_json, load_jsonl
from .stage6_control_authority_handoff_sandbox import run_stage6_control_authority_transfer_sandbox


HARD_HEALTH_FLAGS = {
    "perception_hard_stale",
    "stage1_worker_failed",
}
MAX_ROLLBACK_DELAY_TICKS = 1
MIN_POST_ROLLBACK_BASELINE_TICKS = 2
CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "ml_right_positive_core",
        "kind": "nominal",
        "stage4_profile": "stage4_online_external_watch_shadow_authority_transfer_low_memory",
        "report_stem": "stage6_multi_case_takeover_nominal_report",
        "result_stem": "stage6_multi_case_takeover_nominal_result",
        "set_name": "stage6_multi_case_takeover_nominal",
        "note_suffix": "Stage 6 multi-case takeover sandbox override: bounded-motion shadow authority transfer nominal case.",
        "description": "Nominal bounded-motion shadow authority transfer sandbox.",
        "stop_alignment_required": False,
    },
    {
        "case_id": "arbitration_stop_during_prepare_right",
        "kind": "stop_arbitration",
        "stage4_profile": "stage4_online_external_watch_stop_aligned_shadow_authority_transfer_arbitration_low_memory",
        "report_stem": "stage6_multi_case_takeover_arbitration_report",
        "result_stem": "stage6_multi_case_takeover_arbitration_result",
        "set_name": "stage6_multi_case_takeover_arbitration",
        "note_suffix": "Stage 6 multi-case takeover sandbox override: bounded-motion shadow authority transfer on stop/arbitration case.",
        "description": "Stop/arbitration bounded-motion shadow authority transfer sandbox.",
        "stop_alignment_required": True,
    },
)


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
        return float(mean(deltas))
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


def _case_gate_payload(
    *,
    repo_root: Path,
    spec: dict[str, Any],
    transfer_report: dict[str, Any],
    transfer_result: dict[str, Any],
) -> dict[str, Any]:
    online_report_dir = Path(str(transfer_report.get("online_report_dir") or ""))
    case_id = str(spec["case_id"])
    stage4_dir = online_report_dir / "case_runs" / case_id / "stage4"
    trace_rows = load_jsonl(stage4_dir / "control_authority_sandbox_trace.jsonl")
    tick_rows = load_jsonl(stage4_dir / "online_tick_record.jsonl")
    tick_rows_by_tick = {int(row.get("tick_id", -1)): row for row in tick_rows if row.get("tick_id") is not None}
    summary = dict(transfer_report.get("sandbox_summary") or {})

    handoff_rows = [row for row in trace_rows if bool(row.get("handoff_active"))]
    handoff_tick_ids = sorted(int(row.get("tick_id", -1)) for row in handoff_rows if row.get("tick_id") is not None)
    handoff_tick_rows = [tick_rows_by_tick[tick_id] for tick_id in handoff_tick_ids if tick_id in tick_rows_by_tick]
    fallback_rows_during_handoff = [row for row in handoff_tick_rows if bool(row.get("fallback_activated"))]
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
    handoff_elapsed_seconds = max(0.0, len(handoff_rows) * _positive_step_seconds(handoff_rows))
    max_speed_limit_mps = float(summary.get("max_speed_limit_mps") or 0.0)
    distance_bound_threshold_m = float(max_speed_limit_mps * handoff_elapsed_seconds * 1.25 + 0.05)

    configured_duration_ticks = int(summary.get("handoff_duration_ticks_configured") or -1)
    observed_duration_ticks = int(summary.get("handoff_duration_ticks_observed") or 0)
    rollback_delay_ticks = summary.get("rollback_delay_ticks")
    post_handoff_baseline_tick_count = int(summary.get("post_handoff_baseline_tick_count") or 0)

    stop_alignment = load_json(stage4_dir / "stop_scene_alignment.json", default={}) or {}
    stop_target = load_json(stage4_dir / "stop_target.json", default={}) or {}
    stop_binding_summary = load_json(stage4_dir / "stop_binding_summary.json", default={}) or {}
    stop_events = list(stop_target.get("stop_events") or [])
    stop_binding_exact = any(str(row.get("binding_status") or "") == "exact" for row in stop_events)
    stop_final_error_supported = bool(
        (stop_target.get("summary") or {}).get("stop_final_error_supported")
        or stop_binding_summary.get("stop_final_error_available")
    )

    criteria = {
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
    if bool(spec.get("stop_alignment_required")):
        criteria["stop_alignment_ready"] = bool(stop_alignment.get("alignment_ready"))
        criteria["stop_alignment_consistent"] = bool(stop_alignment.get("alignment_consistent"))
        criteria["stop_target_exact"] = bool(stop_binding_exact)
        criteria["stop_final_error_supported"] = bool(stop_final_error_supported)

    remaining_blockers = [key for key, passed in criteria.items() if not bool(passed)]
    return {
        "case_id": case_id,
        "kind": spec.get("kind"),
        "transfer_report_path": str(repo_root / "benchmark" / f"{spec['report_stem']}.json"),
        "transfer_result_path": str(repo_root / "benchmark" / f"{spec['result_stem']}.json"),
        "online_report_dir": str(online_report_dir),
        "stage4_dir": str(stage4_dir),
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
            "stop_alignment_ready": stop_alignment.get("alignment_ready"),
            "stop_alignment_consistent": stop_alignment.get("alignment_consistent"),
            "stop_target_exact": stop_binding_exact,
            "stop_final_error_supported": stop_final_error_supported,
        },
        "gating_criteria": criteria,
        "overall_status": "pass" if not remaining_blockers else "fail",
        "remaining_blockers": remaining_blockers,
    }


def run_stage6_multi_case_takeover_sandbox(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    pre_takeover_result = load_json(benchmark_root / "stage6_pre_takeover_sandbox_result.json", default={}) or {}
    pre_takeover_ready = (
        str(pre_takeover_result.get("sandbox_gate_status")) == "pass"
        and not list(pre_takeover_result.get("remaining_blockers") or [])
    )

    case_payloads: list[dict[str, Any]] = []
    transfer_runs: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        transfer = run_stage6_control_authority_transfer_sandbox(
            repo_root,
            case_id=str(spec["case_id"]),
            stage4_profile=str(spec["stage4_profile"]),
            set_name=str(spec["set_name"]),
            report_stem=str(spec["report_stem"]),
            result_stem=str(spec["result_stem"]),
            note_suffix=str(spec["note_suffix"]),
            description=str(spec["description"]),
        )
        transfer_runs.append(transfer)
        case_payloads.append(
            _case_gate_payload(
                repo_root=repo_root,
                spec=spec,
                transfer_report=dict(transfer.get("report") or {}),
                transfer_result=dict(transfer.get("result") or {}),
            )
        )

    remaining_blockers: list[str] = []
    if not pre_takeover_ready:
        remaining_blockers.append("pre_takeover_dependency_not_clean")
    for payload in case_payloads:
        for blocker in list(payload.get("remaining_blockers") or []):
            remaining_blockers.append(f"{payload['case_id']}:{blocker}")

    report = {
        "schema_version": "stage6_multi_case_takeover_sandbox_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "pre_takeover_dependency": {
            "result_path": str(benchmark_root / "stage6_pre_takeover_sandbox_result.json"),
            "sandbox_gate_status": pre_takeover_result.get("sandbox_gate_status"),
            "takeover_discussion_status": pre_takeover_result.get("takeover_discussion_status"),
            "takeover_enable_status": pre_takeover_result.get("takeover_enable_status"),
            "remaining_blockers": list(pre_takeover_result.get("remaining_blockers") or []),
            "ready": pre_takeover_ready,
        },
        "cases": case_payloads,
        "aggregate": {
            "case_count": len(case_payloads),
            "passed_case_count": sum(1 for payload in case_payloads if str(payload.get("overall_status")) == "pass"),
            "overall_status": "pass" if not remaining_blockers else "fail",
            "next_scope_status": "ready_for_takeover_ring_sandbox_expansion" if not remaining_blockers else "not_ready",
            "takeover_enable_status": "not_ready",
            "remaining_blockers": remaining_blockers,
        },
        "notes": [
            "This gate validates a narrow multi-case takeover sandbox ring with one nominal case and one stop/arbitration case.",
            "Passing this gate does not authorize takeover rollout. It only clears the next step of expanding bounded takeover sandbox coverage beyond a single case.",
        ],
    }
    result = {
        "schema_version": "stage6_multi_case_takeover_sandbox_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "overall_status": report["aggregate"]["overall_status"],
        "next_scope_status": report["aggregate"]["next_scope_status"],
        "takeover_enable_status": report["aggregate"]["takeover_enable_status"],
        "remaining_blockers": remaining_blockers,
        "report_path": str(benchmark_root / "stage6_multi_case_takeover_sandbox_report.json"),
    }
    dump_json(benchmark_root / "stage6_multi_case_takeover_sandbox_report.json", report)
    dump_json(benchmark_root / "stage6_multi_case_takeover_sandbox_result.json", result)
    return {
        "report": report,
        "result": result,
        "transfers": transfer_runs,
    }
