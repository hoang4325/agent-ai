from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import dump_json, load_json
from .stage6_control_authority_handoff_sandbox import run_stage6_control_authority_transfer_sandbox
from .stage6_takeover_ring_sandbox import _case_gate_payload


CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "ml_right_positive_core",
        "kind": "nominal_right",
        "stage4_profile": "stage4_online_external_watch_shadow_authority_transfer_low_memory",
        "report_stem": "stage6_takeover_ring6_nominal_right_report",
        "result_stem": "stage6_takeover_ring6_nominal_right_result",
        "set_name": "stage6_takeover_ring6_nominal_right",
        "note_suffix": "Stage 6 takeover ring 6-case override: bounded-motion shadow authority transfer nominal right case.",
        "description": "Full takeover ring nominal right bounded-motion shadow authority transfer sandbox.",
        "stop_alignment_required": False,
    },
    {
        "case_id": "ml_left_positive_core",
        "kind": "nominal_left",
        "stage4_profile": "stage4_online_external_watch_shadow_authority_transfer_low_memory",
        "report_stem": "stage6_takeover_ring6_nominal_left_report",
        "result_stem": "stage6_takeover_ring6_nominal_left_result",
        "set_name": "stage6_takeover_ring6_nominal_left",
        "note_suffix": "Stage 6 takeover ring 6-case override: bounded-motion shadow authority transfer nominal left case.",
        "description": "Full takeover ring nominal left bounded-motion shadow authority transfer sandbox.",
        "stop_alignment_required": False,
    },
    {
        "case_id": "junction_straight_core",
        "kind": "junction_straight",
        "stage4_profile": "stage4_online_external_watch_shadow_authority_transfer_low_memory",
        "report_stem": "stage6_takeover_ring6_junction_report",
        "result_stem": "stage6_takeover_ring6_junction_result",
        "set_name": "stage6_takeover_ring6_junction",
        "note_suffix": "Stage 6 takeover ring 6-case override: bounded-motion shadow authority transfer junction straight case.",
        "description": "Full takeover ring junction straight bounded-motion shadow authority transfer sandbox.",
        "stop_alignment_required": False,
    },
    {
        "case_id": "fr_lc_commit_no_permission",
        "kind": "failure_no_permission",
        "stage4_profile": "stage4_online_external_watch_shadow_authority_transfer_low_memory",
        "report_stem": "stage6_takeover_ring6_no_permission_report",
        "result_stem": "stage6_takeover_ring6_no_permission_result",
        "set_name": "stage6_takeover_ring6_no_permission",
        "note_suffix": "Stage 6 takeover ring 6-case override: bounded-motion shadow authority transfer failure replay no-permission case.",
        "description": "Full takeover ring failure replay no-permission bounded-motion shadow authority transfer sandbox.",
        "stop_alignment_required": False,
    },
    {
        "case_id": "stop_follow_ambiguity_core",
        "kind": "stop_follow",
        "stage4_profile": "stage4_online_external_watch_stop_aligned_shadow_authority_transfer_low_memory",
        "report_stem": "stage6_takeover_ring6_stop_follow_report",
        "result_stem": "stage6_takeover_ring6_stop_follow_result",
        "set_name": "stage6_takeover_ring6_stop_follow",
        "note_suffix": "Stage 6 takeover ring 6-case override: bounded-motion shadow authority transfer stop-follow case.",
        "description": "Full takeover ring stop-follow bounded-motion shadow authority transfer sandbox.",
        "stop_alignment_required": True,
    },
    {
        "case_id": "arbitration_stop_during_prepare_right",
        "kind": "stop_arbitration",
        "stage4_profile": "stage4_online_external_watch_stop_aligned_shadow_authority_transfer_arbitration_low_memory",
        "report_stem": "stage6_takeover_ring6_arbitration_report",
        "result_stem": "stage6_takeover_ring6_arbitration_result",
        "set_name": "stage6_takeover_ring6_arbitration",
        "note_suffix": "Stage 6 takeover ring 6-case override: bounded-motion shadow authority transfer stop/arbitration case.",
        "description": "Full takeover ring stop/arbitration bounded-motion shadow authority transfer sandbox.",
        "stop_alignment_required": True,
    },
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_stage6_takeover_ring_6case_sandbox(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    prior_result = load_json(benchmark_root / "stage6_takeover_ring_stability_result.json", default={}) or {}
    prior_ready = (
        str(prior_result.get("overall_status")) == "ready"
        and not list(prior_result.get("remaining_blockers") or [])
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
    if not prior_ready:
        remaining_blockers.append("takeover_ring_stability_dependency_not_clean")
    for payload in case_payloads:
        for blocker in list(payload.get("remaining_blockers") or []):
            remaining_blockers.append(f"{payload['case_id']}:{blocker}")

    report = {
        "schema_version": "stage6_takeover_ring_6case_sandbox_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "prior_dependency": {
            "result_path": str(benchmark_root / "stage6_takeover_ring_stability_result.json"),
            "overall_status": prior_result.get("overall_status"),
            "pass_rate": prior_result.get("pass_rate"),
            "runtime_success_rate": prior_result.get("runtime_success_rate"),
            "remaining_blockers": list(prior_result.get("remaining_blockers") or []),
            "ready": prior_ready,
        },
        "cases": case_payloads,
        "aggregate": {
            "case_count": len(case_payloads),
            "passed_case_count": sum(1 for payload in case_payloads if str(payload.get("overall_status")) == "pass"),
            "overall_status": "pass" if not remaining_blockers else "fail",
            "next_scope_status": "ready_for_takeover_ring_6case_stability_campaign" if not remaining_blockers else "not_ready",
            "takeover_enable_status": "not_ready",
            "remaining_blockers": remaining_blockers,
        },
        "notes": [
            "This gate expands bounded takeover sandbox coverage from the repeatable four-case ring to the promoted six-case shadow ring.",
            "Passing this gate still does not authorize takeover rollout. It only clears repeatability testing on the six-case bounded takeover sandbox ring.",
        ],
    }
    result = {
        "schema_version": "stage6_takeover_ring_6case_sandbox_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "overall_status": report["aggregate"]["overall_status"],
        "next_scope_status": report["aggregate"]["next_scope_status"],
        "takeover_enable_status": report["aggregate"]["takeover_enable_status"],
        "remaining_blockers": remaining_blockers,
        "report_path": str(benchmark_root / "stage6_takeover_ring_6case_sandbox_report.json"),
    }
    dump_json(benchmark_root / "stage6_takeover_ring_6case_sandbox_report.json", report)
    dump_json(benchmark_root / "stage6_takeover_ring_6case_sandbox_result.json", result)
    return {
        "report": report,
        "result": result,
        "transfers": transfer_runs,
    }
