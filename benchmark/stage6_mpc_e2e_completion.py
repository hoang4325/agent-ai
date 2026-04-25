from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_json


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_stage6_mpc_e2e_completion(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"

    rollout_design = load_json(benchmark_root / "stage6_takeover_rollout_design.json", default={}) or {}
    rollout_result = load_json(benchmark_root / "stage6_takeover_rollout_result.json", default={}) or {}
    enable_gate_result = load_json(benchmark_root / "stage6_takeover_enable_gate_result.json", default={}) or {}
    phase0_result = load_json(benchmark_root / "stage6_takeover_canary_phase0_result.json", default={}) or {}
    phase1_result = load_json(benchmark_root / "stage6_takeover_canary_phase1_result.json", default={}) or {}
    phase2_result = load_json(benchmark_root / "stage6_takeover_phase2_result.json", default={}) or {}
    phase3_result = load_json(benchmark_root / "stage6_takeover_phase3_result.json", default={}) or {}
    full_authority_phase0_result = load_json(benchmark_root / "stage6_full_authority_canary_phase0_result.json", default={}) or {}
    full_authority_phase0_report = load_json(benchmark_root / "stage6_full_authority_canary_phase0_report.json", default={}) or {}
    full_authority_phase1_result = load_json(benchmark_root / "stage6_full_authority_canary_phase1_result.json", default={}) or {}
    full_authority_phase1_report = load_json(benchmark_root / "stage6_full_authority_canary_phase1_report.json", default={}) or {}
    full_authority_ring_result = load_json(benchmark_root / "stage6_full_authority_ring_result.json", default={}) or {}
    full_authority_ring_report = load_json(benchmark_root / "stage6_full_authority_ring_report.json", default={}) or {}
    full_authority_ring_stability_result = load_json(benchmark_root / "stage6_full_authority_ring_stability_result.json", default={}) or {}
    full_authority_ring_stability_report = load_json(benchmark_root / "stage6_full_authority_ring_stability_report.json", default={}) or {}

    rollout_policy = dict(rollout_design.get("rollout_policy") or {})
    authority_window = dict(rollout_policy.get("authority_window") or {})
    authority_transition = [str(item) for item in (rollout_policy.get("authority_transition") or [])]

    bounded_takeover_path_complete = (
        str(enable_gate_result.get("takeover_enable_gate_status")) == "pass"
        and str(rollout_result.get("rollout_design_status")) == "ready"
        and str(phase0_result.get("overall_status")) == "pass"
        and str(phase1_result.get("overall_status")) == "pass"
        and str(phase2_result.get("overall_status")) == "pass"
        and str(phase3_result.get("overall_status")) == "pass"
    )

    bounded_authority_window_active = (
        int(authority_window.get("handoff_duration_ticks") or 0) > 0
        and "shadow_authority_window_bounded_to_3_ticks" in authority_transition
    )
    mandatory_baseline_rollback_active = (
        int(authority_window.get("rollback_delay_ticks_max") or 0) >= 0
        and "immediate_rollback_to_baseline" in authority_transition
    )
    full_authority_phase0_pass = str(full_authority_phase0_result.get("overall_status")) == "pass"
    full_authority_phase1_pass = str(full_authority_phase1_result.get("overall_status")) == "pass"
    fallback_under_full_authority_count = int(
        ((full_authority_phase0_report.get("derived_metrics") or {}).get("fallback_activated_under_full_authority_count") or 0)
    ) + int(
        ((full_authority_phase1_report.get("derived_metrics") or {}).get("fallback_activated_under_full_authority_count") or 0)
    ) + int(((full_authority_ring_report.get("aggregate") or {}).get("fallback_activated_case_count") or 0))

    criteria = {
        "bounded_takeover_path_complete": bounded_takeover_path_complete,
        "full_authority_nominal_canary_pass": full_authority_phase0_pass,
        "full_authority_stop_canary_pass": full_authority_phase1_pass,
        "full_scenario_authority_window_present": bool(full_authority_phase0_pass and full_authority_phase1_pass),
        "mandatory_baseline_rollback_disabled": bool(full_authority_phase0_pass and full_authority_phase1_pass),
        "full_scenario_mpc_control_executed": bool(full_authority_phase0_pass and full_authority_phase1_pass),
        "full_authority_promoted_ring_executed": str(full_authority_ring_result.get("overall_status")) == "pass",
        "full_authority_promoted_ring_repeatable": str(full_authority_ring_stability_result.get("overall_status")) == "ready",
        "fallback_under_full_authority_evidenced": fallback_under_full_authority_count > 0,
    }
    remaining_blockers = [key for key, passed in criteria.items() if not bool(passed)]
    if not remaining_blockers:
        next_scope_status = "complete"
    elif bounded_takeover_path_complete and full_authority_phase0_pass and full_authority_phase1_pass:
        next_scope_status = "ready_for_full_authority_fallback_or_ring_design"
    elif bounded_takeover_path_complete:
        next_scope_status = "ready_for_full_authority_canary_review"
    else:
        next_scope_status = "not_ready"

    report = {
        "schema_version": "stage6_mpc_e2e_completion_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "dependencies": {
            "takeover_enable_gate_result_path": str(benchmark_root / "stage6_takeover_enable_gate_result.json"),
            "takeover_rollout_result_path": str(benchmark_root / "stage6_takeover_rollout_result.json"),
            "phase0_result_path": str(benchmark_root / "stage6_takeover_canary_phase0_result.json"),
            "phase1_result_path": str(benchmark_root / "stage6_takeover_canary_phase1_result.json"),
            "phase2_result_path": str(benchmark_root / "stage6_takeover_phase2_result.json"),
            "phase3_result_path": str(benchmark_root / "stage6_takeover_phase3_result.json"),
        },
        "phase_statuses": {
            "phase0_nominal_canary": phase0_result.get("overall_status"),
            "phase1_stop_canary": phase1_result.get("overall_status"),
            "phase2_nominal_ring_extension": phase2_result.get("overall_status"),
            "phase3_full_ring_candidate": phase3_result.get("overall_status"),
            "full_authority_phase0_nominal_canary": full_authority_phase0_result.get("overall_status"),
            "full_authority_phase1_stop_canary": full_authority_phase1_result.get("overall_status"),
            "full_authority_ring": full_authority_ring_result.get("overall_status"),
            "full_authority_ring_stability": full_authority_ring_stability_result.get("overall_status"),
        },
        "rollout_policy_snapshot": {
            "takeover_enable_status": rollout_policy.get("takeover_enable_status"),
            "authority_window": authority_window,
            "authority_transition": authority_transition,
        },
        "full_authority_snapshot": {
            "bounded_authority_window_active_in_rollout_design": bool(bounded_authority_window_active),
            "mandatory_baseline_rollback_active_in_rollout_design": bool(mandatory_baseline_rollback_active),
            "full_authority_phase0_result_path": str(benchmark_root / "stage6_full_authority_canary_phase0_result.json"),
            "full_authority_phase1_result_path": str(benchmark_root / "stage6_full_authority_canary_phase1_result.json"),
            "full_authority_ring_result_path": str(benchmark_root / "stage6_full_authority_ring_result.json"),
            "full_authority_ring_stability_result_path": str(benchmark_root / "stage6_full_authority_ring_stability_result.json"),
            "fallback_under_full_authority_count": int(fallback_under_full_authority_count),
            "full_authority_ring_fallback_case_ids": list((full_authority_ring_report.get("aggregate") or {}).get("fallback_case_ids") or []),
            "full_authority_ring_stability_pass_rate": full_authority_ring_stability_result.get("pass_rate"),
        },
        "completion_criteria": criteria,
        "completion_summary": {
            "bounded_takeover_status": "complete" if bounded_takeover_path_complete else "not_complete",
            "mpc_end_to_end_status": "complete" if not remaining_blockers else "not_complete",
            "next_scope_status": next_scope_status,
            "remaining_blockers": remaining_blockers,
        },
        "notes": [
            "This gate distinguishes bounded takeover readiness from full end-to-end MPC authority completion.",
            "A clean bounded takeover path is necessary but not sufficient for full end-to-end MPC completion.",
            "Full-authority canaries can clear the short-window and rollback blockers, but they still do not replace promoted-ring execution and repeatability evidence.",
        ],
    }
    result = {
        "schema_version": "stage6_mpc_e2e_completion_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "bounded_takeover_status": report["completion_summary"]["bounded_takeover_status"],
        "mpc_end_to_end_status": report["completion_summary"]["mpc_end_to_end_status"],
        "next_scope_status": report["completion_summary"]["next_scope_status"],
        "remaining_blockers": remaining_blockers,
        "report_path": str(benchmark_root / "stage6_mpc_e2e_completion_report.json"),
    }
    dump_json(benchmark_root / "stage6_mpc_e2e_completion_report.json", report)
    dump_json(benchmark_root / "stage6_mpc_e2e_completion_result.json", result)
    return {
        "report": report,
        "result": result,
    }
