from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_json
from .stage6_control_authority_handoff_sandbox import run_stage6_control_authority_transfer_sandbox
from .stage6_takeover_ring_sandbox import _case_gate_payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


PHASE_SPECS: dict[str, dict[str, Any]] = {
    "phase2": {
        "phase_id": "phase2_nominal_ring_extension",
        "report_stem": "stage6_takeover_phase2_report",
        "result_stem": "stage6_takeover_phase2_result",
        "next_scope_status_on_pass": "ready_for_phase3_full_ring_candidate_review",
        "cases": (
            {
                "case_id": "ml_right_positive_core",
                "kind": "nominal_right",
                "stage4_profile": "stage4_online_external_watch_shadow_authority_transfer_low_memory",
                "report_stem": "stage6_takeover_phase2_nominal_right_report",
                "result_stem": "stage6_takeover_phase2_nominal_right_result",
                "set_name": "stage6_takeover_phase2_nominal_right",
                "note_suffix": "Stage 6 explicit takeover rollout override: phase2 nominal ring extension on ml_right_positive_core with bounded shadow authority transfer.",
                "description": "Stage 6 explicit takeover phase2 nominal ring extension on nominal-right case.",
                "stop_alignment_required": False,
            },
            {
                "case_id": "junction_straight_core",
                "kind": "junction_straight",
                "stage4_profile": "stage4_online_external_watch_shadow_authority_transfer_low_memory",
                "report_stem": "stage6_takeover_phase2_junction_report",
                "result_stem": "stage6_takeover_phase2_junction_result",
                "set_name": "stage6_takeover_phase2_junction",
                "note_suffix": "Stage 6 explicit takeover rollout override: phase2 nominal ring extension on junction_straight_core with bounded shadow authority transfer.",
                "description": "Stage 6 explicit takeover phase2 nominal ring extension on junction-straight case.",
                "stop_alignment_required": False,
            },
            {
                "case_id": "fr_lc_commit_no_permission",
                "kind": "failure_no_permission",
                "stage4_profile": "stage4_online_external_watch_shadow_authority_transfer_low_memory",
                "report_stem": "stage6_takeover_phase2_no_permission_report",
                "result_stem": "stage6_takeover_phase2_no_permission_result",
                "set_name": "stage6_takeover_phase2_no_permission",
                "note_suffix": "Stage 6 explicit takeover rollout override: phase2 nominal ring extension on fr_lc_commit_no_permission with bounded shadow authority transfer.",
                "description": "Stage 6 explicit takeover phase2 nominal ring extension on no-permission failure-replay case.",
                "stop_alignment_required": False,
            },
        ),
        "notes": [
            "This phase expands explicit bounded takeover from single-case canaries to the remaining nominal, junction, and failure-replay cases.",
            "It still uses the bounded authority window and does not authorize broader takeover rollout.",
        ],
    },
    "phase3": {
        "phase_id": "phase3_full_ring_candidate",
        "report_stem": "stage6_takeover_phase3_report",
        "result_stem": "stage6_takeover_phase3_result",
        "next_scope_status_on_pass": "ready_for_mpc_e2e_completion_gate_review",
        "cases": (
            {
                "case_id": "arbitration_stop_during_prepare_right",
                "kind": "stop_arbitration",
                "stage4_profile": "stage4_online_external_watch_stop_aligned_shadow_authority_transfer_arbitration_low_memory",
                "report_stem": "stage6_takeover_phase3_arbitration_report",
                "result_stem": "stage6_takeover_phase3_arbitration_result",
                "set_name": "stage6_takeover_phase3_arbitration",
                "note_suffix": "Stage 6 explicit takeover rollout override: phase3 full-ring candidate on arbitration_stop_during_prepare_right with bounded shadow authority transfer.",
                "description": "Stage 6 explicit takeover phase3 full-ring candidate on stop/arbitration case.",
                "stop_alignment_required": True,
            },
        ),
        "notes": [
            "This phase adds the highest-complexity promoted-ring case last, after the narrower phases have already passed.",
            "It still remains bounded to the sandbox authority window with mandatory rollback to baseline.",
        ],
    },
}


def run_stage6_takeover_rollout_phase(
    repo_root: str | Path,
    *,
    phase: str,
    operator_signoff_source: str = "interactive_user_request",
    kill_switch_verification_source: str = "simulation_abort_path_verified",
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    phase_key = str(phase).strip().lower()
    if phase_key not in PHASE_SPECS:
        raise ValueError(f"Unsupported takeover rollout phase: {phase}")

    spec = dict(PHASE_SPECS[phase_key])
    rollout_result = load_json(benchmark_root / "stage6_takeover_rollout_result.json", default={}) or {}
    enable_gate_result = load_json(benchmark_root / "stage6_takeover_enable_gate_result.json", default={}) or {}
    phase0_result = load_json(benchmark_root / "stage6_takeover_canary_phase0_result.json", default={}) or {}
    phase1_result = load_json(benchmark_root / "stage6_takeover_canary_phase1_result.json", default={}) or {}
    phase2_result = load_json(benchmark_root / "stage6_takeover_phase2_result.json", default={}) or {}

    case_payloads: list[dict[str, Any]] = []
    transfer_runs: list[dict[str, Any]] = []
    recovered_case_ids: list[str] = []
    fallback_case_ids: list[str] = []
    max_handoff_distance_estimate_m = 0.0
    max_rollback_delay_ticks = 0
    for case_spec in spec["cases"]:
        transfer = run_stage6_control_authority_transfer_sandbox(
            repo_root,
            case_id=str(case_spec["case_id"]),
            stage4_profile=str(case_spec["stage4_profile"]),
            set_name=str(case_spec["set_name"]),
            report_stem=str(case_spec["report_stem"]),
            result_stem=str(case_spec["result_stem"]),
            note_suffix=str(case_spec["note_suffix"]),
            description=str(case_spec["description"]),
        )
        transfer_runs.append(transfer)
        transfer_report = dict(transfer.get("report") or {})
        transfer_result = dict(transfer.get("result") or {})
        case_payload = _case_gate_payload(
            repo_root=repo_root,
            spec=case_spec,
            transfer_report=transfer_report,
            transfer_result=transfer_result,
        )
        case_payload["recovered_after_retry"] = bool(transfer_report.get("recovered_after_retry"))
        if bool(transfer_report.get("recovered_after_retry")):
            recovered_case_ids.append(str(case_spec["case_id"]))
        if int((case_payload.get("derived_metrics") or {}).get("fallback_activated_during_handoff_count") or 0) > 0:
            fallback_case_ids.append(str(case_spec["case_id"]))
        max_handoff_distance_estimate_m = max(
            max_handoff_distance_estimate_m,
            float((case_payload.get("derived_metrics") or {}).get("handoff_distance_estimate_m") or 0.0),
        )
        max_rollback_delay_ticks = max(
            max_rollback_delay_ticks,
            int((case_payload.get("derived_metrics") or {}).get("rollback_delay_ticks") or 0),
        )
        case_payloads.append(case_payload)

    criteria = {
        "rollout_design_ready": str(rollout_result.get("rollout_design_status")) == "ready",
        "takeover_enable_gate_pass": str(enable_gate_result.get("takeover_enable_gate_status")) == "pass",
        "operator_signoff_recorded": bool(operator_signoff_source.strip()),
        "kill_switch_verified_before_run": bool(kill_switch_verification_source.strip()),
        "all_cases_pass": all(str(payload.get("overall_status")) == "pass" for payload in case_payloads),
        "no_retry_recovery": not recovered_case_ids,
    }
    if phase_key == "phase2":
        criteria["phase0_nominal_canary_pass"] = str(phase0_result.get("overall_status")) == "pass"
        criteria["phase1_stop_canary_pass"] = str(phase1_result.get("overall_status")) == "pass"
    elif phase_key == "phase3":
        criteria["phase2_nominal_ring_extension_pass"] = str(phase2_result.get("overall_status")) == "pass"

    remaining_blockers = [key for key, passed in criteria.items() if not bool(passed)]
    for payload in case_payloads:
        for blocker in list(payload.get("remaining_blockers") or []):
            remaining_blockers.append(f"{payload['case_id']}:{blocker}")
    remaining_blockers = sorted(set(remaining_blockers))
    overall_status = "pass" if not remaining_blockers else "fail"

    report = {
        "schema_version": "stage6_takeover_rollout_phase_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "phase_id": str(spec["phase_id"]),
        "dependencies": {
            "rollout_result_path": str(benchmark_root / "stage6_takeover_rollout_result.json"),
            "rollout_design_status": rollout_result.get("rollout_design_status"),
            "takeover_enable_gate_result_path": str(benchmark_root / "stage6_takeover_enable_gate_result.json"),
            "takeover_enable_gate_status": enable_gate_result.get("takeover_enable_gate_status"),
            "phase0_result_path": str(benchmark_root / "stage6_takeover_canary_phase0_result.json"),
            "phase0_status": phase0_result.get("overall_status"),
            "phase1_result_path": str(benchmark_root / "stage6_takeover_canary_phase1_result.json"),
            "phase1_status": phase1_result.get("overall_status"),
            "phase2_result_path": str(benchmark_root / "stage6_takeover_phase2_result.json"),
            "phase2_status": phase2_result.get("overall_status"),
        },
        "operator_signoff": {
            "recorded": bool(operator_signoff_source.strip()),
            "source": operator_signoff_source,
        },
        "kill_switch_verification": {
            "recorded": bool(kill_switch_verification_source.strip()),
            "source": kill_switch_verification_source,
        },
        "cases": case_payloads,
        "aggregate": {
            "case_count": len(case_payloads),
            "passed_case_count": sum(1 for payload in case_payloads if str(payload.get("overall_status")) == "pass"),
            "recovered_case_ids": recovered_case_ids,
            "fallback_case_ids": fallback_case_ids,
            "max_handoff_distance_estimate_m": max_handoff_distance_estimate_m,
            "max_rollback_delay_ticks": max_rollback_delay_ticks,
        },
        "gating_criteria": criteria,
        "readiness_summary": {
            "overall_status": overall_status,
            "next_scope_status": str(spec["next_scope_status_on_pass"]) if overall_status == "pass" else "not_ready",
            "takeover_enable_status": "not_ready",
            "remaining_blockers": remaining_blockers,
        },
        "notes": list(spec["notes"]),
    }
    result = {
        "schema_version": "stage6_takeover_rollout_phase_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "overall_status": overall_status,
        "next_scope_status": report["readiness_summary"]["next_scope_status"],
        "takeover_enable_status": "not_ready",
        "remaining_blockers": remaining_blockers,
        "report_path": str(benchmark_root / f"{spec['report_stem']}.json"),
    }
    dump_json(benchmark_root / f"{spec['report_stem']}.json", report)
    dump_json(benchmark_root / f"{spec['result_stem']}.json", result)
    return {
        "report": report,
        "result": result,
        "transfers": transfer_runs,
    }
