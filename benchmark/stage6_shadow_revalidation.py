from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fallback_activation_probe import run_fallback_activation_probe
from .io import dump_json, load_jsonl
from .stage4_stop_alignment_revalidation import run_stage4_stop_alignment_revalidation
from .stage6_shadow_closeout_audit import run_stage6_shadow_closeout_audit


def _latency_contract_status(*report_dirs: Path) -> str:
    statuses: list[str] = []
    for report_dir in report_dirs:
        rows = load_jsonl(report_dir / "planner_control_latency.jsonl")
        if not rows:
            statuses.append("missing")
            continue
        has_planner = any(row.get("planner_execution_latency_ms") is not None for row in rows)
        has_decision = any(row.get("decision_to_execution_latency_ms") is not None for row in rows)
        has_total = any(row.get("total_loop_latency_ms") is not None for row in rows)
        statuses.append("present" if has_planner and has_decision and has_total else "approximate")
    if statuses and all(status == "present" for status in statuses):
        return "clean_enough"
    if any(status != "missing" for status in statuses):
        return "approximate"
    return "missing"


def _build_metric_pack(
    *,
    stop_target_contract: str,
    fallback_contract: str,
) -> dict[str, Any]:
    stop_final_error_maturity = "soft_guardrail" if stop_target_contract == "clean_enough" else "diagnostic_only"
    fallback_takeover_maturity = "diagnostic_only"
    return {
        "schema_version": "stage6_metric_pack_v1",
        "hard_gate_metrics": [
            "stop_target_defined_rate",
            "fallback_reason_coverage",
        ],
        "soft_guardrail_metrics": [
            "planner_execution_latency_ms",
            "decision_to_execution_latency_ms",
            "over_budget_rate",
            "stop_overshoot_distance_m",
            *(["stop_final_error_m"] if stop_final_error_maturity == "soft_guardrail" else []),
        ],
        "diagnostic_metrics": [
            *(["stop_final_error_m"] if stop_final_error_maturity != "soft_guardrail" else []),
            "fallback_takeover_latency_ms",
            "fallback_activation_rate",
            "fallback_missing_when_required_count",
        ],
        "metric_maturity_updates": {
            "stop_target_defined_rate": "hard_gate_ready",
            "fallback_reason_coverage": "hard_gate_ready",
            "stop_final_error_m": stop_final_error_maturity,
            "stop_overshoot_distance_m": "soft_guardrail",
            "planner_execution_latency_ms": "soft_guardrail",
            "decision_to_execution_latency_ms": "soft_guardrail",
            "over_budget_rate": "soft_guardrail",
            "fallback_takeover_latency_ms": fallback_takeover_maturity,
        },
        "notes": [
            "stop_final_error_m only promotes when stop-scene alignment is ready on the stop-focused online subset.",
            "fallback_takeover_latency_ms stays diagnostic until more than one activation probe case is available.",
            "online subset remains a shadow-opening guardrail, not the main regression backbone.",
        ],
    }


def run_stage6_shadow_revalidation(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    pre_closeout_audit = run_stage6_shadow_closeout_audit(repo_root)
    stop_result = run_stage4_stop_alignment_revalidation(repo_root)
    fallback_result = run_fallback_activation_probe(repo_root)

    stop_readiness = stop_result.get("readiness") or {}
    stop_alignment = dict(stop_readiness.get("stop_scene_alignment") or {})
    stop_binding_summary = dict(stop_readiness.get("stop_binding_summary") or {})
    stop_target_contract = (
        "clean_enough"
        if bool(stop_readiness.get("can_promote_stop_final_error_m"))
        and bool(stop_alignment.get("alignment_ready"))
        and bool(stop_binding_summary.get("stop_final_error_available"))
        else "approximate"
    )

    stop_stage4_dir = Path(str(stop_readiness.get("online_report_dir") or "")) / "case_runs" / "stop_follow_ambiguity_core" / "stage4"
    fallback_stage4_dir = Path(str(fallback_result.get("report_root") or "")) / "online_e2e" / "case_runs" / "ml_right_positive_core" / "stage4"
    latency_contract = _latency_contract_status(stop_stage4_dir, fallback_stage4_dir)

    probe_payload = dict(fallback_result.get("probe") or {})
    fallback_contract = (
        "clean_enough"
        if bool(probe_payload.get("fallback_activated"))
        and bool(str(probe_payload.get("fallback_reason") or "").strip())
        and probe_payload.get("takeover_latency_ms") is not None
        else "approximate"
    )

    metric_pack = _build_metric_pack(
        stop_target_contract=stop_target_contract,
        fallback_contract=fallback_contract,
    )
    dump_json(repo_root / "benchmark" / "stage6_metric_pack.json", metric_pack)

    readiness = "ready"
    remaining_blockers: list[str] = []
    if stop_target_contract != "clean_enough":
        readiness = "partial"
        remaining_blockers.append("stop_scene_alignment_not_clean_enough")
    if latency_contract != "clean_enough":
        readiness = "partial"
        remaining_blockers.append("latency_contract_not_clean_enough")
    if fallback_contract != "clean_enough":
        readiness = "partial"
        remaining_blockers.append("fallback_activation_takeover_evidence_missing")

    preflight_revalidation = {
        "schema_version": "stage6_preflight_revalidation_v2",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "subset": ["ml_right_positive_core", "stop_follow_ambiguity_core"],
        "stop_target_contract": stop_target_contract,
        "latency_contract": latency_contract,
        "fallback_contract": fallback_contract,
        "metric_maturity_updates": metric_pack.get("metric_maturity_updates"),
        "readiness": readiness,
        "remaining_blockers": remaining_blockers,
        "stop_alignment_report_dir": stop_readiness.get("online_report_dir"),
        "fallback_probe_report_dir": probe_payload.get("report_root"),
    }
    shadow_readiness = {
        "schema_version": "stage6_shadow_readiness_v2",
        "status": readiness,
        "stop_target_contract": stop_target_contract,
        "latency_contract": latency_contract,
        "fallback_contract": fallback_contract,
        "allowed_shadow_subset": (
            ["ml_right_positive_core", "stop_follow_ambiguity_core"] if readiness == "ready" else []
        ),
        "hard_gate_metrics": metric_pack.get("hard_gate_metrics"),
        "soft_guardrail_metrics": metric_pack.get("soft_guardrail_metrics"),
        "remaining_blockers": remaining_blockers,
    }
    dump_json(repo_root / "benchmark" / "stage6_preflight_revalidation.json", preflight_revalidation)
    dump_json(repo_root / "benchmark" / "stage6_shadow_readiness.json", shadow_readiness)
    closeout_audit = run_stage6_shadow_closeout_audit(repo_root)

    return {
        "pre_closeout_audit": pre_closeout_audit,
        "closeout_audit": closeout_audit,
        "stop_result": stop_result,
        "fallback_result": fallback_result,
        "preflight_revalidation": preflight_revalidation,
        "shadow_readiness": shadow_readiness,
        "metric_pack": metric_pack,
    }
