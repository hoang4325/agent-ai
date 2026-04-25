from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_json, load_jsonl
from .stage6_control_authority_handoff_sandbox import (
    FULL_AUTHORITY_STAGE4_PROFILE,
    FULL_AUTHORITY_STOP_STAGE4_PROFILE,
    run_stage6_control_authority_full_authority_sandbox,
)


HARD_HEALTH_FLAGS = {
    "perception_hard_stale",
    "stage1_worker_failed",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "ml_right_positive_core",
        "kind": "nominal_right",
        "stage4_profile": FULL_AUTHORITY_STAGE4_PROFILE,
        "report_stem": "stage6_full_authority_ring_nominal_right_report",
        "result_stem": "stage6_full_authority_ring_nominal_right_result",
        "set_name": "stage6_full_authority_ring_nominal_right",
        "note_suffix": "Stage 6 full-authority ring override: full-runtime shadow actuator authority on ml_right_positive_core.",
        "description": "Stage 6 full-authority ring on nominal-right case.",
        "stop_alignment_required": False,
    },
    {
        "case_id": "ml_left_positive_core",
        "kind": "nominal_left",
        "stage4_profile": FULL_AUTHORITY_STAGE4_PROFILE,
        "report_stem": "stage6_full_authority_ring_nominal_left_report",
        "result_stem": "stage6_full_authority_ring_nominal_left_result",
        "set_name": "stage6_full_authority_ring_nominal_left",
        "note_suffix": "Stage 6 full-authority ring override: full-runtime shadow actuator authority on ml_left_positive_core.",
        "description": "Stage 6 full-authority ring on nominal-left case.",
        "stop_alignment_required": False,
    },
    {
        "case_id": "junction_straight_core",
        "kind": "junction_straight",
        "stage4_profile": FULL_AUTHORITY_STAGE4_PROFILE,
        "report_stem": "stage6_full_authority_ring_junction_report",
        "result_stem": "stage6_full_authority_ring_junction_result",
        "set_name": "stage6_full_authority_ring_junction",
        "note_suffix": "Stage 6 full-authority ring override: full-runtime shadow actuator authority on junction_straight_core.",
        "description": "Stage 6 full-authority ring on junction-straight case.",
        "stop_alignment_required": False,
    },
    {
        "case_id": "fr_lc_commit_no_permission",
        "kind": "failure_no_permission",
        "stage4_profile": FULL_AUTHORITY_STAGE4_PROFILE,
        "report_stem": "stage6_full_authority_ring_no_permission_report",
        "result_stem": "stage6_full_authority_ring_no_permission_result",
        "set_name": "stage6_full_authority_ring_no_permission",
        "note_suffix": "Stage 6 full-authority ring override: full-runtime shadow actuator authority on fr_lc_commit_no_permission.",
        "description": "Stage 6 full-authority ring on no-permission failure-replay case.",
        "stop_alignment_required": False,
    },
    {
        "case_id": "stop_follow_ambiguity_core",
        "kind": "stop_follow",
        "stage4_profile": FULL_AUTHORITY_STOP_STAGE4_PROFILE,
        "report_stem": "stage6_full_authority_ring_stop_follow_report",
        "result_stem": "stage6_full_authority_ring_stop_follow_result",
        "set_name": "stage6_full_authority_ring_stop_follow",
        "note_suffix": "Stage 6 full-authority ring override: full-runtime shadow actuator authority on stop_follow_ambiguity_core.",
        "description": "Stage 6 full-authority ring on stop-follow case.",
        "stop_alignment_required": True,
    },
    {
        "case_id": "arbitration_stop_during_prepare_right",
        "kind": "stop_arbitration",
        "stage4_profile": "stage4_online_external_watch_stop_aligned_shadow_authority_full_arbitration_low_memory",
        "report_stem": "stage6_full_authority_ring_arbitration_report",
        "result_stem": "stage6_full_authority_ring_arbitration_result",
        "set_name": "stage6_full_authority_ring_arbitration",
        "note_suffix": "Stage 6 full-authority ring override: full-runtime shadow actuator authority on arbitration_stop_during_prepare_right.",
        "description": "Stage 6 full-authority ring on stop/arbitration case.",
        "stop_alignment_required": True,
    },
)


def _build_case_payload(
    *,
    repo_root: Path,
    spec: dict[str, Any],
    transfer_report: dict[str, Any],
    transfer_result: dict[str, Any],
) -> dict[str, Any]:
    online_report_dir = Path(str(transfer_report.get("online_report_dir") or ""))
    case_id = str(spec["case_id"])
    stage4_dir = online_report_dir / "case_runs" / case_id / "stage4"
    summary = dict(transfer_report.get("sandbox_summary") or {})
    trace_rows = load_jsonl(stage4_dir / "control_authority_sandbox_trace.jsonl")
    tick_rows = load_jsonl(stage4_dir / "online_tick_record.jsonl")
    tick_rows_by_tick = {int(row.get("tick_id", -1)): row for row in tick_rows if row.get("tick_id") is not None}

    active_rows = [row for row in trace_rows if bool(row.get("handoff_active"))]
    authority_start_tick = (
        min(int(row.get("tick_id")) for row in active_rows if row.get("tick_id") is not None)
        if active_rows
        else None
    )
    rows_after_authority_start = (
        [row for row in trace_rows if authority_start_tick is not None and int(row.get("tick_id", -1)) >= authority_start_tick]
        if authority_start_tick is not None
        else []
    )
    shadow_authority_rows = [
        row
        for row in rows_after_authority_start
        if bool(row.get("actual_shadow_takeover_applied"))
        and str(row.get("executed_control_owner") or "") == "mpc_shadow_candidate_sandbox"
    ]
    full_authority_until_end_observed = bool(
        rows_after_authority_start and len(shadow_authority_rows) == len(rows_after_authority_start)
    )
    authority_tick_ids = [
        int(row["tick_id"])
        for row in shadow_authority_rows
        if row.get("tick_id") is not None
    ]
    authority_tick_records = [tick_rows_by_tick[tick_id] for tick_id in authority_tick_ids if tick_id in tick_rows_by_tick]
    over_budget_count = sum(1 for row in authority_tick_records if bool(row.get("over_budget")))
    hard_health_flags = sorted(
        {
            str(flag)
            for row in authority_tick_records
            for flag in (row.get("health_flags") or [])
            if str(flag) in HARD_HEALTH_FLAGS
        }
    )
    fallback_activated_count = sum(1 for row in authority_tick_records if bool(row.get("fallback_activated")))

    stop_alignment = load_json(stage4_dir / "stop_scene_alignment.json", default={}) or {}
    stop_target = load_json(stage4_dir / "stop_target.json", default={}) or {}
    stop_binding_summary = load_json(stage4_dir / "stop_binding_summary.json", default={}) or {}
    stop_events = list(stop_target.get("stop_events") or [])
    stop_target_exact = any(str(row.get("binding_status") or "") == "exact" for row in stop_events)
    stop_final_error_supported = bool(
        (stop_target.get("summary") or {}).get("stop_final_error_supported")
        or stop_binding_summary.get("stop_final_error_available")
    )

    criteria = {
        "full_authority_sandbox_ready": str(transfer_result.get("overall_status")) == "ready",
        "shadow_authority_applied": bool(summary.get("actual_shadow_takeover_applied")),
        "full_authority_until_end_observed": bool(full_authority_until_end_observed),
        "no_retry_recovery": not bool(transfer_report.get("recovered_after_retry")),
        "no_baseline_rollback_inside_run": not bool(summary.get("rollback_to_baseline_observed")),
        "no_over_budget_under_full_authority": over_budget_count == 0,
        "no_hard_health_flags_under_full_authority": not hard_health_flags,
    }
    if bool(spec.get("stop_alignment_required")):
        criteria["stop_alignment_ready"] = bool(stop_alignment.get("alignment_ready"))
        criteria["stop_alignment_consistent"] = bool(stop_alignment.get("alignment_consistent"))
        criteria["stop_target_exact"] = bool(stop_target_exact)
        criteria["stop_final_error_supported"] = bool(stop_final_error_supported)

    remaining_blockers = [key for key, passed in criteria.items() if not bool(passed)]
    return {
        "case_id": case_id,
        "kind": spec.get("kind"),
        "recovered_after_retry": bool(transfer_report.get("recovered_after_retry")),
        "transfer_attempt_count": len(list(transfer_report.get("attempts") or [])),
        "transfer_report_path": str(repo_root / "benchmark" / f"{spec['report_stem']}.json"),
        "transfer_result_path": str(repo_root / "benchmark" / f"{spec['result_stem']}.json"),
        "online_report_dir": str(online_report_dir),
        "stage4_dir": str(stage4_dir),
        "derived_metrics": {
            "authority_start_tick": authority_start_tick,
            "full_authority_tick_ids": authority_tick_ids,
            "full_authority_tail_tick_count": len(shadow_authority_rows),
            "full_authority_until_end_observed": bool(full_authority_until_end_observed),
            "fallback_activated_under_full_authority_count": int(fallback_activated_count),
            "over_budget_under_full_authority_count": int(over_budget_count),
            "hard_health_flags_under_full_authority": hard_health_flags,
            "stop_alignment_ready": stop_alignment.get("alignment_ready"),
            "stop_alignment_consistent": stop_alignment.get("alignment_consistent"),
            "stop_target_exact": stop_target_exact,
            "stop_final_error_supported": stop_final_error_supported,
        },
        "gating_criteria": criteria,
        "overall_status": "pass" if not remaining_blockers else "fail",
        "remaining_blockers": remaining_blockers,
    }


def run_stage6_full_authority_ring(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    phase0_result = load_json(benchmark_root / "stage6_full_authority_canary_phase0_result.json", default={}) or {}
    phase1_result = load_json(benchmark_root / "stage6_full_authority_canary_phase1_result.json", default={}) or {}
    prerequisites_ready = (
        str(phase0_result.get("overall_status")) == "pass"
        and str(phase1_result.get("overall_status")) == "pass"
    )

    case_payloads: list[dict[str, Any]] = []
    transfer_runs: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        transfer = run_stage6_control_authority_full_authority_sandbox(
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
            _build_case_payload(
                repo_root=repo_root,
                spec=spec,
                transfer_report=dict(transfer.get("report") or {}),
                transfer_result=dict(transfer.get("result") or {}),
            )
        )

    remaining_blockers: list[str] = []
    if not prerequisites_ready:
        remaining_blockers.append("full_authority_canary_dependency_not_clean")
    for payload in case_payloads:
        for blocker in list(payload.get("remaining_blockers") or []):
            remaining_blockers.append(f"{payload['case_id']}:{blocker}")

    fallback_case_ids = [
        str(payload["case_id"])
        for payload in case_payloads
        if int((payload.get("derived_metrics") or {}).get("fallback_activated_under_full_authority_count") or 0) > 0
    ]
    report = {
        "schema_version": "stage6_full_authority_ring_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "dependencies": {
            "phase0_result_path": str(benchmark_root / "stage6_full_authority_canary_phase0_result.json"),
            "phase0_status": phase0_result.get("overall_status"),
            "phase1_result_path": str(benchmark_root / "stage6_full_authority_canary_phase1_result.json"),
            "phase1_status": phase1_result.get("overall_status"),
            "prerequisites_ready": prerequisites_ready,
        },
        "cases": case_payloads,
        "aggregate": {
            "case_count": len(case_payloads),
            "passed_case_count": sum(1 for payload in case_payloads if str(payload.get("overall_status")) == "pass"),
            "fallback_case_ids": fallback_case_ids,
            "fallback_activated_case_count": len(fallback_case_ids),
            "overall_status": "pass" if not remaining_blockers else "fail",
            "next_scope_status": "ready_for_full_authority_ring_stability_campaign" if not remaining_blockers else "not_ready",
            "remaining_blockers": remaining_blockers,
        },
        "notes": [
            "This gate expands full-authority MPC sandbox coverage from nominal and stop canaries to the promoted six-case ring.",
            "It is still sandbox authority evidence, not production rollout authorization.",
        ],
    }
    result = {
        "schema_version": "stage6_full_authority_ring_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "overall_status": report["aggregate"]["overall_status"],
        "next_scope_status": report["aggregate"]["next_scope_status"],
        "remaining_blockers": remaining_blockers,
        "report_path": str(benchmark_root / "stage6_full_authority_ring_report.json"),
    }
    dump_json(benchmark_root / "stage6_full_authority_ring_report.json", report)
    dump_json(benchmark_root / "stage6_full_authority_ring_result.json", result)
    return {
        "report": report,
        "result": result,
        "transfers": transfer_runs,
    }
