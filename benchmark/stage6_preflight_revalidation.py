from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.run_system_benchmark_e2e import run_e2e_benchmark

from .io import dump_json, load_json, load_jsonl, load_yaml, resolve_repo_path
from .stage6_preflight_contract_audit import run_stage6_preflight_contract_audit
from .stage6_preflight_metric_pack import build_stage6_preflight_metric_pack
from .stop_target_binding_audit import run_stop_target_binding_audit


SHADOW_READY_SUBSET = [
    "ml_right_positive_core",
    "stop_follow_ambiguity_core",
]


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _materialize_subset_config(repo_root: Path, *, case_ids: list[str], root: Path) -> Path:
    benchmark_config = load_yaml(repo_root / "benchmark" / "benchmark_v1.yaml")
    runtime_config = deepcopy(benchmark_config)
    cases_dir = resolve_repo_path(repo_root, benchmark_config.get("cases_dir")) or (repo_root / "benchmark" / "cases")
    runtime_cases_dir = root / "cases"
    runtime_cases_dir.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        case_payload = load_yaml(cases_dir / f"{case_id}.yaml")
        bundle_session_dir = (
            repo_root
            / "benchmark"
            / "frozen_corpus"
            / "v1"
            / "cases"
            / case_id
            / "stage1_frozen"
            / "session"
        )
        case_payload["stage_profiles"] = dict(case_payload.get("stage_profiles") or {})
        case_payload["stage_profiles"]["stage4"] = "stage4_online_external_watch"
        case_payload["scenario_recipe"] = dict(case_payload.get("scenario_recipe") or {})
        case_payload["scenario_recipe"]["stage1_session_dir"] = str(bundle_session_dir)
        case_payload["source"] = dict(case_payload.get("source") or {})
        case_payload["source"]["stage1_session_dir"] = str(bundle_session_dir)
        _write_yaml(runtime_cases_dir / f"{case_id}.yaml", case_payload)

    set_name = "stage6_shadow_subset"
    set_path = root / "sets" / f"{set_name}.yaml"
    _write_yaml(
        set_path,
        {
            "set_name": set_name,
            "description": "Minimal Stage 6 preflight subset with one multilane and one stop-focused case.",
            "cases": case_ids,
        },
    )
    runtime_config["cases_dir"] = str(runtime_cases_dir)
    runtime_config["sets"] = dict(runtime_config.get("sets") or {})
    runtime_config["sets"][set_name] = str(set_path)
    config_path = root / "benchmark_runtime_config.yaml"
    _write_yaml(config_path, runtime_config)
    return config_path


def _stop_contract_status(summary: dict[str, Any], online_stop_target_paths: list[Path]) -> str:
    defined_rate = summary.get("defined_rate")
    exact_rate = float(summary.get("exact_rate") or 0.0)
    heuristic_or_exact = float(exact_rate + (summary.get("heuristic_rate") or 0.0))
    online_present = False
    online_exact = False
    for path in online_stop_target_paths:
        payload = load_json(path, default={}) or {}
        support = dict(payload.get("measurement_support") or {})
        summary_payload = dict(payload.get("summary") or {})
        if str(support.get("stop_target_defined_rate")) == "present" and str(support.get("stop_final_error_m")) in {"present", "approximate"}:
            online_present = True
            break
        if int(summary_payload.get("explicit_target_events") or 0) > 0 and str(support.get("stop_final_error_m")) == "present":
            online_exact = True
    if defined_rate == 1.0 and exact_rate > 0.0 and online_exact:
        return "clean_enough"
    if defined_rate or online_present:
        return "approximate"
    return "missing"


def _latency_contract_status(online_report_dir: Path) -> str:
    status_rows = []
    for case_id in SHADOW_READY_SUBSET:
        rows = load_jsonl(online_report_dir / "case_runs" / case_id / "stage4" / "planner_control_latency.jsonl")
        if not rows:
            status_rows.append("missing")
            continue
        has_planner = any(row.get("planner_execution_latency_ms") is not None for row in rows)
        has_decision = any(row.get("decision_to_execution_latency_ms") is not None for row in rows)
        has_total = any(row.get("total_loop_latency_ms") is not None for row in rows)
        status_rows.append("present" if has_planner and has_decision and has_total else "approximate")
    if status_rows and all(status == "present" for status in status_rows):
        return "clean_enough"
    if any(status != "missing" for status in status_rows):
        return "approximate"
    return "missing"


def _fallback_contract_status(online_report_dir: Path) -> str:
    status_rows = []
    for case_id in SHADOW_READY_SUBSET:
        rows = load_jsonl(online_report_dir / "case_runs" / case_id / "stage4" / "fallback_events.jsonl")
        if not rows:
            status_rows.append("missing")
            continue
        activated = [row for row in rows if row.get("fallback_activated")]
        if not activated:
            status_rows.append("approximate")
            continue
        reason_covered = all(str(row.get("fallback_reason") or "").strip() for row in activated)
        takeover_present = all(row.get("takeover_latency_ms") is not None for row in activated)
        status_rows.append("present" if reason_covered and takeover_present else "approximate")
    if status_rows and all(status == "present" for status in status_rows):
        return "clean_enough"
    if any(status != "missing" for status in status_rows):
        return "approximate"
    return "missing"


def _revalidated_metric_pack(repo_root: Path, stop_target_contract: str, latency_contract: str) -> dict[str, Any]:
    pack = build_stage6_preflight_metric_pack(repo_root)
    if stop_target_contract == "clean_enough":
        diagnostic = [row for row in pack.get("diagnostic_metrics") or [] if row.get("metric") != "stop_final_error_m"]
        soft = list(pack.get("soft_guardrail_metrics") or [])
        soft.append(
            {
                "metric": "stop_final_error_m",
                "rationale": "Promoted to soft guardrail after explicit stop-target binding and online stop-target evidence landed on the Stage 6 preflight subset.",
            }
        )
        pack["diagnostic_metrics"] = diagnostic
        pack["soft_guardrail_metrics"] = soft
    notes = list(pack.get("notes") or [])
    if latency_contract == "clean_enough":
        notes.append("online latency evidence now present on the Stage 6 shadow subset.")
    pack["notes"] = notes
    dump_json(repo_root / "benchmark" / "stage6_preflight_metric_pack.json", pack)
    return pack


def run_stage6_preflight_revalidation(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    report_root = repo_root / "benchmark" / "reports" / f"stage6_preflight_revalidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_root.mkdir(parents=True, exist_ok=True)
    config_path = _materialize_subset_config(repo_root, case_ids=SHADOW_READY_SUBSET, root=report_root)

    scenario_report_dir = report_root / "scenario_replay"
    online_report_dir = report_root / "online_e2e"
    scenario_result = run_e2e_benchmark(
        config_path=config_path,
        set_name="stage6_shadow_subset",
        benchmark_mode="scenario_replay",
        execute_stages=True,
        report_dir=scenario_report_dir,
    )
    online_result = run_e2e_benchmark(
        config_path=config_path,
        set_name="stage6_shadow_subset",
        benchmark_mode="online_e2e",
        execute_stages=True,
        report_dir=online_report_dir,
    )

    audit = run_stage6_preflight_contract_audit(repo_root)
    stop_audit = run_stop_target_binding_audit(
        repo_root,
        report_roots=[scenario_report_dir, online_report_dir],
        case_ids=SHADOW_READY_SUBSET,
    )

    online_stop_target_paths = [
        online_report_dir / "case_runs" / case_id / "stage4" / "stop_target.json"
        for case_id in SHADOW_READY_SUBSET
    ]
    stop_target_contract = _stop_contract_status(stop_audit["summary"], online_stop_target_paths)
    latency_contract = _latency_contract_status(online_report_dir)
    fallback_contract = _fallback_contract_status(online_report_dir)
    revalidated_pack = _revalidated_metric_pack(repo_root, stop_target_contract, latency_contract)

    metric_maturity_updates = {
        "stop_target_defined_rate": "hard_gate_ready",
        "fallback_reason_coverage": "hard_gate_ready",
        "stop_final_error_m": "soft_guardrail" if stop_target_contract == "clean_enough" else "diagnostic_only",
        "stop_overshoot_distance_m": "soft_guardrail",
        "planner_execution_latency_ms": "soft_guardrail" if latency_contract == "clean_enough" else "soft_guardrail",
        "decision_to_execution_latency_ms": "soft_guardrail" if latency_contract == "clean_enough" else "soft_guardrail",
        "over_budget_rate": "soft_guardrail" if latency_contract == "clean_enough" else "soft_guardrail",
        "fallback_takeover_latency_ms": "diagnostic_only",
    }

    readiness = "ready"
    remaining_blockers: list[str] = []
    if stop_target_contract != "clean_enough":
        readiness = "partial"
        remaining_blockers.append("stop_target_contract_not_clean_enough")
    if latency_contract != "clean_enough":
        readiness = "partial"
        remaining_blockers.append("latency_contract_missing_online_clean_breakdown")
    if fallback_contract != "clean_enough":
        readiness = "partial"
        remaining_blockers.append("fallback_contract_missing_online_takeover_evidence")

    revalidation = {
        "schema_version": "stage6_preflight_revalidation_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "subset": SHADOW_READY_SUBSET,
        "scenario_replay_report_dir": str(scenario_report_dir),
        "online_e2e_report_dir": str(online_report_dir),
        "scenario_replay_overall_status": scenario_result.get("overall_status"),
        "online_e2e_overall_status": online_result.get("overall_status"),
        "stop_target_contract": stop_target_contract,
        "latency_contract": latency_contract,
        "fallback_contract": fallback_contract,
        "metric_maturity_updates": metric_maturity_updates,
        "readiness": readiness,
        "remaining_blockers": remaining_blockers,
        "raw_contract_audit": {
            "stop_target_contract": audit.get("stop_target_contract"),
            "latency_contract": audit.get("latency_contract"),
            "fallback_contract": audit.get("fallback_contract"),
        },
    }
    shadow_readiness = {
        "schema_version": "stage6_shadow_readiness_v1",
        "status": readiness,
        "required_subset": SHADOW_READY_SUBSET,
        "hard_gate_metrics": ["stop_target_defined_rate", "fallback_reason_coverage"],
        "soft_guardrail_metrics": [
            "stop_final_error_m" if metric_maturity_updates["stop_final_error_m"] == "soft_guardrail" else "stop_overshoot_distance_m",
            "planner_execution_latency_ms",
            "decision_to_execution_latency_ms",
            "over_budget_rate",
        ],
        "remaining_blockers": remaining_blockers,
    }
    dump_json(repo_root / "benchmark" / "stage6_preflight_revalidation.json", revalidation)
    dump_json(repo_root / "benchmark" / "stage6_shadow_readiness.json", shadow_readiness)
    return {
        "revalidation": revalidation,
        "shadow_readiness": shadow_readiness,
        "metric_pack": revalidated_pack,
        "stop_target_audit": stop_audit,
    }
