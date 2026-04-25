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


PHASE_SPECS: dict[str, dict[str, Any]] = {
    "phase0": {
        "phase_id": "full_authority_phase0_nominal_canary",
        "case_id": "ml_left_positive_core",
        "kind": "full_authority_nominal_canary",
        "stage4_profile": FULL_AUTHORITY_STAGE4_PROFILE,
        "report_stem": "stage6_full_authority_canary_phase0_report",
        "result_stem": "stage6_full_authority_canary_phase0_result",
        "set_name": "stage6_full_authority_canary_phase0",
        "note_suffix": "Stage 6 full-authority MPC canary override: nominal full-runtime shadow actuator authority on ml_left_positive_core.",
        "description": "Stage 6 full-authority MPC phase0 nominal canary.",
        "stop_alignment_required": False,
        "next_scope_status_on_pass": "ready_for_full_authority_phase1_stop_canary_review",
        "notes": [
            "This is the first full-authority MPC canary beyond the bounded takeover path.",
            "It requires MPC shadow control to remain the executed owner through the end of the runtime window.",
        ],
    },
    "phase1": {
        "phase_id": "full_authority_phase1_stop_canary",
        "case_id": "stop_follow_ambiguity_core",
        "kind": "full_authority_stop_canary",
        "stage4_profile": FULL_AUTHORITY_STOP_STAGE4_PROFILE,
        "report_stem": "stage6_full_authority_canary_phase1_report",
        "result_stem": "stage6_full_authority_canary_phase1_result",
        "set_name": "stage6_full_authority_canary_phase1",
        "note_suffix": "Stage 6 full-authority MPC canary override: stop-focused full-runtime shadow actuator authority on stop_follow_ambiguity_core.",
        "description": "Stage 6 full-authority MPC phase1 stop canary.",
        "stop_alignment_required": True,
        "next_scope_status_on_pass": "ready_for_full_authority_ring_design",
        "notes": [
            "This is the first stop-focused full-authority MPC canary beyond the bounded takeover path.",
            "It requires clean stop alignment and exact stop-target support while MPC shadow remains the executed owner through the end of the runtime window.",
        ],
    },
}


def run_stage6_full_authority_canary(
    repo_root: str | Path,
    *,
    phase: str = "phase0",
    operator_signoff_source: str = "interactive_user_request",
    kill_switch_verification_source: str = "simulation_abort_path_verified",
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    phase_key = str(phase).strip().lower()
    if phase_key not in PHASE_SPECS:
        raise ValueError(f"Unsupported full-authority canary phase: {phase}")
    spec = dict(PHASE_SPECS[phase_key])

    completion_result = load_json(benchmark_root / "stage6_mpc_e2e_completion_result.json", default={}) or {}
    rollout_result = load_json(benchmark_root / "stage6_takeover_rollout_result.json", default={}) or {}
    phase0_result = load_json(benchmark_root / "stage6_full_authority_canary_phase0_result.json", default={}) or {}

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
    transfer_report = dict(transfer.get("report") or {})
    transfer_result = dict(transfer.get("result") or {})
    summary = dict(transfer_report.get("sandbox_summary") or {})
    online_report_dir = Path(str(transfer_report.get("online_report_dir") or ""))
    stage4_dir = online_report_dir / "case_runs" / str(spec["case_id"]) / "stage4"
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
        rows_after_authority_start
        and len(shadow_authority_rows) == len(rows_after_authority_start)
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
    fallback_activated_under_full_authority_count = sum(
        1 for row in authority_tick_records if bool(row.get("fallback_activated"))
    )

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
        "bounded_takeover_path_complete": str(completion_result.get("bounded_takeover_status")) == "complete"
        or (
            str(rollout_result.get("rollout_design_status")) == "ready"
            and str(rollout_result.get("next_scope_status")) == "ready_for_mpc_e2e_completion_gate_review"
        ),
        "operator_signoff_recorded": bool(operator_signoff_source.strip()),
        "kill_switch_verified_before_run": bool(kill_switch_verification_source.strip()),
        "full_authority_sandbox_ready": str(transfer_result.get("overall_status")) == "ready",
        "shadow_authority_applied": bool(summary.get("actual_shadow_takeover_applied")),
        "full_authority_until_end_observed": bool(full_authority_until_end_observed),
        "no_retry_recovery": not bool(transfer_report.get("recovered_after_retry")),
        "no_baseline_rollback_inside_full_authority_run": not bool(summary.get("rollback_to_baseline_observed")),
        "no_over_budget_under_full_authority": over_budget_count == 0,
        "no_hard_health_flags_under_full_authority": not hard_health_flags,
    }
    if phase_key == "phase1":
        criteria["phase0_full_authority_nominal_canary_pass"] = str(phase0_result.get("overall_status")) == "pass"
        criteria["stop_alignment_ready"] = bool(stop_alignment.get("alignment_ready"))
        criteria["stop_alignment_consistent"] = bool(stop_alignment.get("alignment_consistent"))
        criteria["stop_target_exact"] = bool(stop_target_exact)
        criteria["stop_final_error_supported"] = bool(stop_final_error_supported)

    remaining_blockers = [key for key, passed in criteria.items() if not bool(passed)]
    overall_status = "pass" if not remaining_blockers else "fail"

    report = {
        "schema_version": "stage6_full_authority_canary_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "phase_id": str(spec["phase_id"]),
        "case_id": str(spec["case_id"]),
        "dependencies": {
            "completion_result_path": str(benchmark_root / "stage6_mpc_e2e_completion_result.json"),
            "bounded_takeover_status": completion_result.get("bounded_takeover_status"),
            "rollout_result_path": str(benchmark_root / "stage6_takeover_rollout_result.json"),
            "rollout_next_scope_status": rollout_result.get("next_scope_status"),
            "phase0_result_path": str(benchmark_root / "stage6_full_authority_canary_phase0_result.json"),
            "phase0_status": phase0_result.get("overall_status"),
        },
        "operator_signoff": {
            "recorded": bool(operator_signoff_source.strip()),
            "source": operator_signoff_source,
        },
        "kill_switch_verification": {
            "recorded": bool(kill_switch_verification_source.strip()),
            "source": kill_switch_verification_source,
        },
        "transfer_report_path": str(benchmark_root / f"{spec['report_stem']}.json"),
        "online_report_dir": str(online_report_dir),
        "stage4_dir": str(stage4_dir),
        "derived_metrics": {
            "authority_start_tick": authority_start_tick,
            "full_authority_tick_ids": authority_tick_ids,
            "full_authority_tail_tick_count": len(shadow_authority_rows),
            "full_authority_until_end_observed": bool(full_authority_until_end_observed),
            "fallback_activated_under_full_authority_count": int(fallback_activated_under_full_authority_count),
            "over_budget_under_full_authority_count": int(over_budget_count),
            "hard_health_flags_under_full_authority": hard_health_flags,
            "max_speed_mps": summary.get("max_speed_mps"),
            "max_speed_limit_mps": summary.get("max_speed_limit_mps"),
            "rollback_to_baseline_observed": summary.get("rollback_to_baseline_observed"),
            "rollback_tick": summary.get("rollback_tick"),
            "stop_alignment_ready": stop_alignment.get("alignment_ready"),
            "stop_alignment_consistent": stop_alignment.get("alignment_consistent"),
            "stop_target_exact": stop_target_exact,
            "stop_final_error_supported": stop_final_error_supported,
        },
        "sandbox_summary": summary,
        "gating_criteria": criteria,
        "readiness_summary": {
            "overall_status": overall_status,
            "next_scope_status": str(spec["next_scope_status_on_pass"]) if overall_status == "pass" else "not_ready",
            "remaining_blockers": remaining_blockers,
        },
        "notes": list(spec["notes"]),
    }
    result = {
        "schema_version": "stage6_full_authority_canary_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "overall_status": overall_status,
        "next_scope_status": report["readiness_summary"]["next_scope_status"],
        "remaining_blockers": remaining_blockers,
        "report_path": str(benchmark_root / f"{spec['report_stem']}.json"),
    }
    dump_json(benchmark_root / f"{spec['report_stem']}.json", report)
    dump_json(benchmark_root / f"{spec['result_stem']}.json", result)
    return {
        "report": report,
        "result": result,
        "transfer": transfer,
    }
