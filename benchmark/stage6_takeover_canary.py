from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .io import dump_json, load_json, load_jsonl
from .stage6_control_authority_handoff_sandbox import run_stage6_control_authority_transfer_sandbox
from .stage6_takeover_ring_sandbox import _case_gate_payload


PHASE_SPECS: dict[str, dict[str, Any]] = {
    "phase0": {
        "phase_id": "phase0_nominal_canary",
        "case_id": "ml_left_positive_core",
        "kind": "phase0_nominal_canary",
        "stage4_profile": "stage4_online_external_watch_shadow_authority_transfer_low_memory",
        "report_stem": "stage6_takeover_canary_phase0_report",
        "result_stem": "stage6_takeover_canary_phase0_result",
        "set_name": "stage6_takeover_canary_phase0",
        "note_suffix": "Stage 6 explicit takeover canary override: phase0 nominal canary on ml_left_positive_core with bounded shadow authority transfer.",
        "description": "Stage 6 explicit takeover phase0 nominal canary.",
        "stop_alignment_required": False,
        "next_scope_status_on_pass": "ready_for_phase1_stop_canary_review",
        "notes": [
            "This is the first explicit takeover canary on the cleanest nominal case from the rollout design package.",
            "It is still bounded to the sandbox authority window and does not authorize broader takeover rollout.",
        ],
    },
    "phase1": {
        "phase_id": "phase1_stop_canary",
        "case_id": "stop_follow_ambiguity_core",
        "kind": "phase1_stop_canary",
        "stage4_profile": "stage4_online_external_watch_stop_aligned_shadow_authority_transfer_low_memory",
        "report_stem": "stage6_takeover_canary_phase1_report",
        "result_stem": "stage6_takeover_canary_phase1_result",
        "set_name": "stage6_takeover_canary_phase1",
        "note_suffix": "Stage 6 explicit takeover canary override: phase1 stop canary on stop_follow_ambiguity_core with bounded shadow authority transfer.",
        "description": "Stage 6 explicit takeover phase1 stop canary.",
        "stop_alignment_required": True,
        "next_scope_status_on_pass": "ready_for_phase2_nominal_ring_extension_review",
        "notes": [
            "This is the first explicit stop-focused takeover canary from the rollout design package.",
            "It remains bounded to the sandbox authority window and does not authorize broader takeover rollout.",
        ],
    },
}
HARD_HEALTH_FLAGS = {
    "perception_hard_stale",
    "stage1_worker_failed",
}


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


def run_stage6_takeover_canary(
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
        raise ValueError(f"Unsupported takeover canary phase: {phase}")
    spec = dict(PHASE_SPECS[phase_key])

    rollout_design_result = load_json(benchmark_root / "stage6_takeover_rollout_result.json", default={}) or {}
    takeover_enable_result = load_json(benchmark_root / "stage6_takeover_enable_gate_result.json", default={}) or {}

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
    transfer_report = dict(transfer.get("report") or {})
    transfer_result = dict(transfer.get("result") or {})

    case_payload = _case_gate_payload(
        repo_root=repo_root,
        spec={
            "case_id": str(spec["case_id"]),
            "kind": str(spec["kind"]),
            "report_stem": str(spec["report_stem"]),
            "result_stem": str(spec["result_stem"]),
            "stop_alignment_required": bool(spec["stop_alignment_required"]),
        },
        transfer_report=transfer_report,
        transfer_result=transfer_result,
    )

    online_report_dir = Path(str(transfer_report.get("online_report_dir") or ""))
    stage4_dir = online_report_dir / "case_runs" / str(spec["case_id"]) / "stage4"
    trace_rows = load_jsonl(stage4_dir / "control_authority_sandbox_trace.jsonl")
    tick_rows = load_jsonl(stage4_dir / "online_tick_record.jsonl")
    tick_rows_by_tick = {int(row.get("tick_id", -1)): row for row in tick_rows if row.get("tick_id") is not None}
    handoff_tick_ids = list((case_payload.get("derived_metrics") or {}).get("handoff_tick_ids") or [])
    handoff_tick_rows = [tick_rows_by_tick[tick_id] for tick_id in handoff_tick_ids if tick_id in tick_rows_by_tick]
    handoff_rows = [row for row in trace_rows if bool(row.get("handoff_active"))]
    hard_health_flags_during_handoff = sorted(
        {
            str(flag)
            for row in handoff_tick_rows
            for flag in (row.get("health_flags") or [])
            if str(flag) in HARD_HEALTH_FLAGS
        }
    )
    planner_latency_values = [
        float(row.get("planner_execution_latency_ms") or 0.0)
        for row in tick_rows
        if row.get("planner_execution_latency_ms") is not None
    ]

    criteria = {
        "rollout_design_ready": str(rollout_design_result.get("rollout_design_status")) == "ready",
        "takeover_enable_gate_pass": str(takeover_enable_result.get("takeover_enable_gate_status")) == "pass",
        "operator_signoff_recorded": bool(operator_signoff_source.strip()),
        "kill_switch_verified_before_run": bool(kill_switch_verification_source.strip()),
        "transfer_case_gate_pass": str(case_payload.get("overall_status")) == "pass",
        "no_retry_recovery": not bool(transfer_report.get("recovered_after_retry")),
        "no_fallback_during_handoff": int((case_payload.get("derived_metrics") or {}).get("fallback_activated_during_handoff_count") or 0) == 0,
        "no_hard_health_flags_during_handoff": not hard_health_flags_during_handoff,
    }
    if bool(spec["stop_alignment_required"]):
        criteria["stop_alignment_ready"] = bool((case_payload.get("derived_metrics") or {}).get("stop_alignment_ready"))
        criteria["stop_alignment_consistent"] = bool((case_payload.get("derived_metrics") or {}).get("stop_alignment_consistent"))
        criteria["stop_target_exact"] = bool((case_payload.get("derived_metrics") or {}).get("stop_target_exact"))
        criteria["stop_final_error_supported"] = bool((case_payload.get("derived_metrics") or {}).get("stop_final_error_supported"))
    remaining_blockers = [key for key, passed in criteria.items() if not bool(passed)]

    report = {
        "schema_version": "stage6_takeover_canary_report_v2",
        "generated_at_utc": _utc_now_iso(),
        "phase_id": str(spec["phase_id"]),
        "case_id": str(spec["case_id"]),
        "rollout_design_dependency": {
            "result_path": str(benchmark_root / "stage6_takeover_rollout_result.json"),
            "rollout_design_status": rollout_design_result.get("rollout_design_status"),
            "next_scope_status": rollout_design_result.get("next_scope_status"),
            "takeover_enable_status": rollout_design_result.get("takeover_enable_status"),
        },
        "takeover_enable_dependency": {
            "result_path": str(benchmark_root / "stage6_takeover_enable_gate_result.json"),
            "takeover_enable_gate_status": takeover_enable_result.get("takeover_enable_gate_status"),
            "next_scope_status": takeover_enable_result.get("next_scope_status"),
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
            "handoff_tick_ids": handoff_tick_ids,
            "handoff_duration_ticks_observed": (case_payload.get("derived_metrics") or {}).get("handoff_duration_ticks_observed"),
            "rollback_delay_ticks": (case_payload.get("derived_metrics") or {}).get("rollback_delay_ticks"),
            "handoff_distance_estimate_m": _estimate_handoff_distance_m(handoff_rows),
            "fallback_activated_during_handoff_count": (case_payload.get("derived_metrics") or {}).get("fallback_activated_during_handoff_count"),
            "fallback_success_during_handoff": (case_payload.get("derived_metrics") or {}).get("fallback_success_during_handoff"),
            "over_budget_during_handoff_count": (case_payload.get("derived_metrics") or {}).get("over_budget_during_handoff_count"),
            "post_handoff_baseline_tick_count": (case_payload.get("derived_metrics") or {}).get("post_handoff_baseline_tick_count"),
            "hard_health_flags_during_handoff": hard_health_flags_during_handoff,
            "planner_execution_latency_ms_min": min(planner_latency_values) if planner_latency_values else None,
            "planner_execution_latency_ms_max": max(planner_latency_values) if planner_latency_values else None,
        },
        "base_case_gate": case_payload,
        "gating_criteria": criteria,
        "readiness_summary": {
            "overall_status": "pass" if not remaining_blockers else "fail",
            "next_scope_status": str(spec["next_scope_status_on_pass"]) if not remaining_blockers else "not_ready",
            "takeover_enable_status": "not_ready",
            "remaining_blockers": remaining_blockers,
        },
        "notes": list(spec["notes"]),
    }
    result = {
        "schema_version": "stage6_takeover_canary_result_v2",
        "generated_at_utc": report["generated_at_utc"],
        "overall_status": report["readiness_summary"]["overall_status"],
        "next_scope_status": report["readiness_summary"]["next_scope_status"],
        "takeover_enable_status": report["readiness_summary"]["takeover_enable_status"],
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
