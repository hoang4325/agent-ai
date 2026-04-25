from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .stage6_organic_fallback_ring import run_stage6_organic_fallback_ring
from .stage6_non_timeout_fallback_ring import run_stage6_non_timeout_fallback_ring
from .stage6_control_authority_handoff_sandbox import run_stage6_control_authority_handoff_sandbox
from .stage6_control_authority_handoff_sandbox import run_stage6_control_authority_bounded_motion_sandbox
from .stage6_control_authority_handoff_sandbox import run_stage6_control_authority_transfer_sandbox
from .stage6_fallback_authority_ring import run_stage6_fallback_authority_ring
from .io import dump_json, load_json
from .stage6_shadow_matrix_campaign import sync_stage6_shadow_matrix_campaign_to_stability


PROMOTED_RING_CASES = [
    "ml_right_positive_core",
    "stop_follow_ambiguity_core",
    "ml_left_positive_core",
    "junction_straight_core",
    "arbitration_stop_during_prepare_right",
    "fr_lc_commit_no_permission",
]

ALLOWED_AUTHORITY_VALUES = {
    "proposal_only_non_executed",
    "proposal_vs_realized_baseline_short_horizon",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _latest_matrix_case_attempts(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for run in report.get("runs") or []:
        for case in run.get("cases") or []:
            case_id = str(case.get("case_id") or "")
            if case_id:
                latest[case_id] = dict(case)
    return latest


def _read_case_sandbox_audit(*, repo_root: Path, case_id: str, case_row: dict[str, Any]) -> dict[str, Any]:
    attempts = list(case_row.get("attempts") or [])
    latest_attempt = attempts[-1] if attempts else {}
    online_report_dir = Path(str(latest_attempt.get("online_report_dir") or ""))
    stage3c_dir = online_report_dir / "case_runs" / case_id / "_current_run_artifacts" / "stage3c"
    summary = load_json(stage3c_dir / "mpc_shadow_summary.json", default={}) or {}
    comparison = load_json(stage3c_dir / "shadow_baseline_comparison.json", default={}) or {}

    proposal_mode = str(summary.get("proposal_mode") or "")
    support_map = dict(summary.get("shadow_metric_support") or {})
    authority_values = sorted(
        {
            str((payload or {}).get("authority") or "")
            for payload in support_map.values()
            if str((payload or {}).get("authority") or "").strip()
        }
    )
    summary_warnings = [str(value) for value in (summary.get("warnings") or [])]
    comparison_notes = [str(value) for value in (comparison.get("notes") or [])]

    proposal_only_warning_present = any(
        "proposal_only" in value or "not_executed_control" in value
        for value in summary_warnings
    )
    no_takeover_warning_present = any(
        "not_shadow_takeover" in value or proposal_mode == "shadow_only_no_takeover"
        for value in summary_warnings
    )
    baseline_authority_note_present = any(
        "baseline is still the only execution authority" in value.lower()
        for value in comparison_notes
    )
    authority_values_allowed = bool(authority_values) and all(
        value in ALLOWED_AUTHORITY_VALUES for value in authority_values
    )

    notes: list[str] = []
    if proposal_mode != "shadow_only_no_takeover":
        notes.append(f"unexpected_proposal_mode:{proposal_mode or 'missing'}")
    if not authority_values_allowed:
        notes.append("authority_values_outside_allowed_shadow_only_set")
    if not proposal_only_warning_present:
        notes.append("proposal_only_warning_missing")
    if not no_takeover_warning_present:
        notes.append("no_takeover_warning_missing")
    if not baseline_authority_note_present:
        notes.append("baseline_authority_note_missing")

    return {
        "case_id": case_id,
        "online_report_dir": str(online_report_dir) if str(online_report_dir) else None,
        "stage3c_dir": str(stage3c_dir),
        "proposal_mode": proposal_mode or None,
        "authority_values": authority_values,
        "proposal_only_warning_present": proposal_only_warning_present,
        "no_takeover_warning_present": no_takeover_warning_present,
        "baseline_authority_note_present": baseline_authority_note_present,
        "authority_values_allowed": authority_values_allowed,
        "sandbox_clean": not notes,
        "notes": notes,
    }


def _build_control_authority_sandbox_audit(repo_root: Path, matrix_report: dict[str, Any]) -> dict[str, Any]:
    latest_cases = _latest_matrix_case_attempts(matrix_report)
    case_audits: list[dict[str, Any]] = []
    remaining_blockers: list[str] = []

    for case_id in PROMOTED_RING_CASES:
        case_row = latest_cases.get(case_id) or {}
        case_audit = _read_case_sandbox_audit(repo_root=repo_root, case_id=case_id, case_row=case_row)
        case_audits.append(case_audit)
        if not bool(case_audit.get("sandbox_clean")):
            remaining_blockers.append(f"{case_id}:control_authority_sandbox")

    payload = {
        "schema_version": "stage6_control_authority_sandbox_audit_v1",
        "generated_at_utc": _utc_now_iso(),
        "source_matrix_campaign_report": str(repo_root / "benchmark" / "stage6_shadow_matrix_campaign_report.json"),
        "cases": case_audits,
        "overall_status": "pass" if not remaining_blockers else "fail",
        "remaining_blockers": remaining_blockers,
        "notes": [
            "This audit verifies that promoted-ring shadow artifacts still declare proposal-only authority and never claim executed shadow control.",
            "Allowed authority values are limited to proposal_only_non_executed and proposal_vs_realized_baseline_short_horizon.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_control_authority_sandbox_audit.json", payload)
    return payload


def run_stage6_pre_takeover_sandbox_gate(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"

    sync_stage6_shadow_matrix_campaign_to_stability(repo_root)
    expansion_result = load_json(benchmark_root / "stage6_shadow_expansion_result.json", default={}) or {}
    stability_result = load_json(benchmark_root / "stage6_shadow_stability_6case_result.json", default={}) or {}
    matrix_report = load_json(benchmark_root / "stage6_shadow_matrix_campaign_report.json", default={}) or {}
    fallback_ring_result = run_stage6_fallback_authority_ring(repo_root)
    fallback_ring_report = dict(fallback_ring_result.get("report") or {})
    fallback_ring_summary = dict(fallback_ring_report.get("aggregate") or {})
    non_timeout_fallback_ring_result = run_stage6_non_timeout_fallback_ring(repo_root)
    non_timeout_fallback_ring_report = dict(non_timeout_fallback_ring_result.get("report") or {})
    non_timeout_fallback_ring_summary = dict(non_timeout_fallback_ring_report.get("aggregate") or {})
    organic_fallback_ring_result = run_stage6_organic_fallback_ring(repo_root)
    organic_fallback_ring_report = dict(organic_fallback_ring_result.get("report") or {})
    organic_fallback_ring_summary = dict(organic_fallback_ring_report.get("aggregate") or {})
    handoff_sandbox_result = run_stage6_control_authority_handoff_sandbox(repo_root)
    handoff_sandbox_report = dict(handoff_sandbox_result.get("report") or {})
    handoff_sandbox_summary = dict(handoff_sandbox_report.get("sandbox_summary") or {})
    bounded_motion_sandbox_result = run_stage6_control_authority_bounded_motion_sandbox(repo_root)
    bounded_motion_sandbox_report = dict(bounded_motion_sandbox_result.get("report") or {})
    bounded_motion_sandbox_summary = dict(bounded_motion_sandbox_report.get("sandbox_summary") or {})
    transfer_sandbox_result = run_stage6_control_authority_transfer_sandbox(repo_root)
    transfer_sandbox_report = dict(transfer_sandbox_result.get("report") or {})
    transfer_sandbox_summary = dict(transfer_sandbox_report.get("sandbox_summary") or {})
    control_authority_audit = _build_control_authority_sandbox_audit(repo_root, matrix_report)

    shadow_ring_ready = (
        str(expansion_result.get("smoke_status")) == "pass"
        and str(expansion_result.get("golden_status")) == "pass"
        and str(stability_result.get("overall_status")) == "ready"
    )
    fallback_probe_ready = str((fallback_ring_result.get("result") or {}).get("overall_status")) == "ready"
    organic_fallback_ring_ready = str((organic_fallback_ring_result.get("result") or {}).get("overall_status")) == "ready"
    control_authority_sandbox_ready = str(control_authority_audit.get("overall_status")) == "pass"

    fallback_evidence_scope = str(fallback_ring_summary.get("coverage_scope") or "single_case_fault_injection")
    fallback_ring_coverage_ready = fallback_probe_ready and str(fallback_evidence_scope) == "multi_case_fault_injection"
    non_timeout_fallback_ring_ready = str((non_timeout_fallback_ring_result.get("result") or {}).get("overall_status")) == "ready"
    control_authority_handoff_executed = bool(handoff_sandbox_summary.get("handoff_executed"))
    control_authority_handoff_sandbox_ready = str((handoff_sandbox_result.get("result") or {}).get("overall_status")) == "ready"
    bounded_motion_handoff_executed = bool(bounded_motion_sandbox_summary.get("handoff_executed"))
    bounded_motion_sandbox_ready = str((bounded_motion_sandbox_result.get("result") or {}).get("overall_status")) == "ready"
    transfer_handoff_executed = bool(transfer_sandbox_summary.get("handoff_executed"))
    transfer_sandbox_ready = str((transfer_sandbox_result.get("result") or {}).get("overall_status")) == "ready"

    remaining_blockers: list[str] = []
    if not shadow_ring_ready:
        remaining_blockers.append("promoted_ring_shadow_readiness_not_clean")
    if not fallback_probe_ready:
        remaining_blockers.append("fallback_authority_ring_not_clean")
    if not non_timeout_fallback_ring_ready:
        remaining_blockers.append("non_timeout_fallback_ring_not_clean")
    if not control_authority_sandbox_ready:
        remaining_blockers.append("control_authority_sandbox_audit_failed")
    if not control_authority_handoff_sandbox_ready:
        remaining_blockers.append("control_authority_handoff_sandbox_not_clean")
    if not bounded_motion_sandbox_ready:
        remaining_blockers.append("control_authority_bounded_motion_sandbox_not_clean")
    if not transfer_sandbox_ready:
        remaining_blockers.append("control_authority_transfer_sandbox_not_clean")

    sandbox_gate_status = (
        "pass"
        if shadow_ring_ready
        and fallback_probe_ready
        and control_authority_sandbox_ready
        and control_authority_handoff_sandbox_ready
        and bounded_motion_sandbox_ready
        else "fail"
    )
    takeover_discussion_status = "ready" if sandbox_gate_status == "pass" else "not_ready"
    takeover_enable_status = "not_ready"

    if fallback_ring_coverage_ready and not non_timeout_fallback_ring_ready:
        remaining_blockers.append("fallback_evidence_still_fault_injection_only")
    elif non_timeout_fallback_ring_ready and not organic_fallback_ring_ready:
        remaining_blockers.append("fallback_evidence_non_timeout_perturbation_only")
    if not control_authority_handoff_executed:
        remaining_blockers.append("no_control_authority_handoff_sandbox_executed")
    elif not bounded_motion_handoff_executed:
        remaining_blockers.append("only_no_motion_handoff_sandbox_executed")
    elif not transfer_handoff_executed:
        remaining_blockers.append("only_baseline_locked_bounded_motion_handoff_sandbox_executed")

    report = {
        "schema_version": "stage6_pre_takeover_sandbox_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "promoted_ring_cases": list(PROMOTED_RING_CASES),
        "shadow_ring": {
            "expansion_result_path": str(benchmark_root / "stage6_shadow_expansion_result.json"),
            "stability_result_path": str(benchmark_root / "stage6_shadow_stability_6case_result.json"),
            "smoke_status": expansion_result.get("smoke_status"),
            "golden_status": expansion_result.get("golden_status"),
            "stability_status": stability_result.get("overall_status"),
            "pass_rate": stability_result.get("pass_rate"),
            "runtime_success_rate": stability_result.get("runtime_success_rate"),
            "ready": shadow_ring_ready,
        },
        "fallback_probe": {
            "artifact_path": str(benchmark_root / "stage6_fallback_authority_ring_report.json"),
            "trace_path": None,
            "case_id": None,
            "trigger_type": "hard_timeout_fault_injection",
            "fallback_activated": None,
            "fallback_reason": None,
            "takeover_latency_ms": fallback_ring_summary.get("mean_takeover_latency_ms"),
            "outcome": "success" if fallback_probe_ready else "failed_or_missing",
            "scope": fallback_evidence_scope,
            "ready": fallback_probe_ready,
            "report_root": None,
            "cases": list(fallback_ring_report.get("cases") or []),
        },
        "non_timeout_fallback_probe": {
            "artifact_path": str(benchmark_root / "stage6_non_timeout_fallback_ring_report.json"),
            "scope": non_timeout_fallback_ring_summary.get("coverage_scope"),
            "ready": non_timeout_fallback_ring_ready,
            "activated_case_count": non_timeout_fallback_ring_summary.get("activated_case_count"),
            "fallback_required_case_count": non_timeout_fallback_ring_summary.get("fallback_required_case_count"),
            "cases": list(non_timeout_fallback_ring_report.get("cases") or []),
        },
        "control_authority_handoff_sandbox": {
            "artifact_path": str(benchmark_root / "stage6_control_authority_handoff_sandbox_report.json"),
            "overall_status": (handoff_sandbox_result.get("result") or {}).get("overall_status"),
            "ready": control_authority_handoff_sandbox_ready,
            "case_id": handoff_sandbox_report.get("case_id"),
            "sandbox_summary": handoff_sandbox_summary,
        },
        "control_authority_bounded_motion_sandbox": {
            "artifact_path": str(benchmark_root / "stage6_control_authority_bounded_motion_sandbox_report.json"),
            "overall_status": (bounded_motion_sandbox_result.get("result") or {}).get("overall_status"),
            "ready": bounded_motion_sandbox_ready,
            "case_id": bounded_motion_sandbox_report.get("case_id"),
            "sandbox_summary": bounded_motion_sandbox_summary,
        },
        "control_authority_transfer_sandbox": {
            "artifact_path": str(benchmark_root / "stage6_control_authority_transfer_sandbox_report.json"),
            "overall_status": (transfer_sandbox_result.get("result") or {}).get("overall_status"),
            "ready": transfer_sandbox_ready,
            "case_id": transfer_sandbox_report.get("case_id"),
            "sandbox_summary": transfer_sandbox_summary,
        },
        "organic_fallback_probe": {
            "artifact_path": str(benchmark_root / "stage6_organic_fallback_ring_report.json"),
            "scope": organic_fallback_ring_summary.get("coverage_scope"),
            "ready": organic_fallback_ring_ready,
            "activated_case_count": organic_fallback_ring_summary.get("activated_case_count"),
            "fallback_required_case_count": organic_fallback_ring_summary.get("fallback_required_case_count"),
            "cases": list(organic_fallback_ring_report.get("cases") or []),
        },
        "control_authority_sandbox": {
            "artifact_path": str(benchmark_root / "stage6_control_authority_sandbox_audit.json"),
            "overall_status": control_authority_audit.get("overall_status"),
            "ready": control_authority_sandbox_ready,
            "remaining_blockers": control_authority_audit.get("remaining_blockers") or [],
        },
        "readiness_summary": {
            "sandbox_gate_status": sandbox_gate_status,
            "takeover_discussion_status": takeover_discussion_status,
            "takeover_enable_status": takeover_enable_status,
            "fallback_evidence_scope": fallback_evidence_scope,
            "fallback_ring_coverage_ready": fallback_ring_coverage_ready,
            "non_timeout_fallback_ring_ready": non_timeout_fallback_ring_ready,
            "organic_fallback_ring_ready": organic_fallback_ring_ready,
            "control_authority_handoff_executed": control_authority_handoff_executed,
            "bounded_motion_handoff_executed": bounded_motion_handoff_executed,
            "transfer_handoff_executed": transfer_handoff_executed,
            "remaining_blockers": remaining_blockers,
        },
        "notes": [
            "This gate does not enable takeover. It only determines whether the promoted shadow ring is clean enough to start explicit takeover sandbox design work.",
            "Multi-case fallback fault injection is accepted as sandbox evidence, but it is still not equivalent to organic runtime failure coverage across the full promoted ring.",
            "Non-timeout route-uncertainty perturbation is stronger than timeout-only fault injection, but it still does not count as organic fallback coverage.",
            "No-motion and baseline-locked bounded-motion authority sandboxes keep baseline as the only controller that drives the vehicle.",
            "The transfer sandbox is still a tightly bounded single-case authority experiment, not a takeover rollout.",
        ],
    }
    result = {
        "schema_version": "stage6_pre_takeover_sandbox_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "sandbox_gate_status": sandbox_gate_status,
        "takeover_discussion_status": takeover_discussion_status,
        "takeover_enable_status": takeover_enable_status,
        "remaining_blockers": remaining_blockers,
        "report_path": str(benchmark_root / "stage6_pre_takeover_sandbox_report.json"),
    }
    dump_json(benchmark_root / "stage6_pre_takeover_sandbox_report.json", report)
    dump_json(benchmark_root / "stage6_pre_takeover_sandbox_result.json", result)
    return {
        "report": report,
        "result": result,
        "fallback_result": fallback_ring_result,
        "handoff_sandbox_result": handoff_sandbox_result,
        "control_authority_audit": control_authority_audit,
    }
