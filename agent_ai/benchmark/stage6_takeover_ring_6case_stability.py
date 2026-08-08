from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .io import dump_json
from .stage6_takeover_ring_6case_sandbox import CASE_SPECS, run_stage6_takeover_ring_6case_sandbox


@dataclass
class CampaignRun:
    index: int
    overall_status: str
    report: dict[str, Any]
    result: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * float(q))))
    return float(ordered[position])


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
        }
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
    }


def _bool_rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if bool(value)) / len(values))


def _analyze_campaign(runs: list[CampaignRun]) -> dict[str, Any]:
    expected_cases = [str(spec["case_id"]) for spec in CASE_SPECS]
    run_records: list[dict[str, Any]] = []
    case_rollup: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in expected_cases}
    remaining_blockers: list[str] = []

    pass_runs = 0
    for run in runs:
        if str(run.result.get("overall_status")) == "pass":
            pass_runs += 1
        run_cases: list[dict[str, Any]] = []
        for payload in run.report.get("cases") or []:
            case_id = str(payload.get("case_id") or "")
            if not case_id:
                continue
            row = {
                "case_id": case_id,
                "overall_status": str(payload.get("overall_status") or "fail"),
                "remaining_blockers": list(payload.get("remaining_blockers") or []),
                "handoff_duration_ticks_observed": (payload.get("derived_metrics") or {}).get("handoff_duration_ticks_observed"),
                "handoff_distance_estimate_m": (payload.get("derived_metrics") or {}).get("handoff_distance_estimate_m"),
                "distance_bound_threshold_m": (payload.get("derived_metrics") or {}).get("distance_bound_threshold_m"),
                "rollback_delay_ticks": (payload.get("derived_metrics") or {}).get("rollback_delay_ticks"),
                "post_handoff_baseline_tick_count": (payload.get("derived_metrics") or {}).get("post_handoff_baseline_tick_count"),
                "fallback_activated_during_handoff_count": (payload.get("derived_metrics") or {}).get("fallback_activated_during_handoff_count"),
                "fallback_success_during_handoff": (payload.get("derived_metrics") or {}).get("fallback_success_during_handoff"),
                "over_budget_during_handoff_count": (payload.get("derived_metrics") or {}).get("over_budget_during_handoff_count"),
                "stop_alignment_ready": (payload.get("derived_metrics") or {}).get("stop_alignment_ready"),
                "stop_alignment_consistent": (payload.get("derived_metrics") or {}).get("stop_alignment_consistent"),
                "stop_target_exact": (payload.get("derived_metrics") or {}).get("stop_target_exact"),
                "stop_final_error_supported": (payload.get("derived_metrics") or {}).get("stop_final_error_supported"),
                "gating_criteria": dict(payload.get("gating_criteria") or {}),
            }
            case_rollup.setdefault(case_id, []).append(row)
            run_cases.append(row)
        run_records.append(
            {
                "run_index": int(run.index),
                "overall_status": str(run.result.get("overall_status") or "fail"),
                "report_path": str(run.result.get("report_path") or ""),
                "cases": run_cases,
            }
        )

    case_payloads: list[dict[str, Any]] = []
    runtime_success_count = 0
    total_case_rows = 0
    aggregate_distance_values: list[float] = []
    aggregate_rollback_values: list[float] = []
    aggregate_duration_values: list[float] = []

    for spec in CASE_SPECS:
        case_id = str(spec["case_id"])
        rows = list(case_rollup.get(case_id) or [])
        total_case_rows += len(rows)
        runtime_success_count += sum(1 for row in rows if str(row.get("overall_status")) == "pass")
        pass_rate = float(sum(1 for row in rows if str(row.get("overall_status")) == "pass") / len(rows)) if rows else 0.0
        duration_values = [
            float(row["handoff_duration_ticks_observed"])
            for row in rows
            if row.get("handoff_duration_ticks_observed") is not None
        ]
        distance_values = [
            float(row["handoff_distance_estimate_m"])
            for row in rows
            if row.get("handoff_distance_estimate_m") is not None
        ]
        rollback_values = [
            float(row["rollback_delay_ticks"])
            for row in rows
            if row.get("rollback_delay_ticks") is not None
        ]
        post_handoff_tail_values = [
            float(row["post_handoff_baseline_tick_count"])
            for row in rows
            if row.get("post_handoff_baseline_tick_count") is not None
        ]
        fallback_activation_counts = [
            float(row["fallback_activated_during_handoff_count"])
            for row in rows
            if row.get("fallback_activated_during_handoff_count") is not None
        ]
        over_budget_counts = [
            float(row["over_budget_during_handoff_count"])
            for row in rows
            if row.get("over_budget_during_handoff_count") is not None
        ]
        fallback_success_values = [bool(row.get("fallback_success_during_handoff")) for row in rows if row.get("fallback_success_during_handoff") is not None]
        stop_alignment_ready_values = [bool(row.get("stop_alignment_ready")) for row in rows if row.get("stop_alignment_ready") is not None]
        stop_alignment_consistent_values = [bool(row.get("stop_alignment_consistent")) for row in rows if row.get("stop_alignment_consistent") is not None]
        stop_target_exact_values = [bool(row.get("stop_target_exact")) for row in rows if row.get("stop_target_exact") is not None]
        stop_final_error_supported_values = [bool(row.get("stop_final_error_supported")) for row in rows if row.get("stop_final_error_supported") is not None]

        aggregate_distance_values.extend(distance_values)
        aggregate_rollback_values.extend(rollback_values)
        aggregate_duration_values.extend(duration_values)

        if pass_rate < 1.0:
            remaining_blockers.append(f"{case_id}:pass_rate<1.0")
        if any(list(row.get("remaining_blockers") or []) for row in rows):
            remaining_blockers.append(f"{case_id}:case_gate_regression")

        case_payload = {
            "case_id": case_id,
            "kind": spec.get("kind"),
            "pass_rate": pass_rate,
            "handoff_duration_ticks_observed": _stats(duration_values),
            "handoff_distance_estimate_m": _stats(distance_values),
            "rollback_delay_ticks": _stats(rollback_values),
            "post_handoff_baseline_tick_count": _stats(post_handoff_tail_values),
            "fallback_activated_during_handoff_count": _stats(fallback_activation_counts),
            "fallback_success_during_handoff_rate": _bool_rate(fallback_success_values),
            "over_budget_during_handoff_count": _stats(over_budget_counts),
            "stop_alignment_ready_rate": _bool_rate(stop_alignment_ready_values),
            "stop_alignment_consistent_rate": _bool_rate(stop_alignment_consistent_values),
            "stop_target_exact_rate": _bool_rate(stop_target_exact_values),
            "stop_final_error_supported_rate": _bool_rate(stop_final_error_supported_values),
            "notes": [],
        }
        if bool(spec.get("stop_alignment_required")):
            if case_payload["stop_alignment_ready_rate"] != 1.0:
                remaining_blockers.append(f"{case_id}:stop_alignment_ready_rate<1.0")
            if case_payload["stop_alignment_consistent_rate"] != 1.0:
                remaining_blockers.append(f"{case_id}:stop_alignment_consistent_rate<1.0")
            if case_payload["stop_target_exact_rate"] != 1.0:
                remaining_blockers.append(f"{case_id}:stop_target_exact_rate<1.0")
            if case_payload["stop_final_error_supported_rate"] != 1.0:
                remaining_blockers.append(f"{case_id}:stop_final_error_supported_rate<1.0")
        case_payloads.append(case_payload)

    campaign_pass_rate = float(pass_runs / len(runs)) if runs else 0.0
    runtime_success_rate = float(runtime_success_count / total_case_rows) if total_case_rows else 0.0
    readiness = "ready"
    if campaign_pass_rate < 1.0 or runtime_success_rate < 1.0 or remaining_blockers:
        readiness = "not_ready"
        if campaign_pass_rate < 1.0:
            remaining_blockers.append("campaign_pass_rate<1.0")
        if runtime_success_rate < 1.0:
            remaining_blockers.append("campaign_runtime_success_rate<1.0")

    return {
        "schema_version": "stage6_takeover_ring_6case_stability_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "subset_cases": expected_cases,
        "num_runs": len(runs),
        "pass_rate": campaign_pass_rate,
        "runtime_success_rate": runtime_success_rate,
        "readiness": readiness,
        "aggregate_metrics": {
            "handoff_duration_ticks_observed": _stats(aggregate_duration_values),
            "handoff_distance_estimate_m": _stats(aggregate_distance_values),
            "rollback_delay_ticks": _stats(aggregate_rollback_values),
        },
        "runs": run_records,
        "cases": case_payloads,
        "remaining_blockers": sorted(set(remaining_blockers)),
        "notes": [
            "This campaign measures repeatability of the six-case bounded takeover sandbox ring, not takeover rollout readiness.",
            "Passing this campaign only clears design work for a stricter takeover-enable gate. It does not authorize authority rollout.",
        ],
    }


def run_stage6_takeover_ring_6case_stability(
    repo_root: str | Path,
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"

    runs: list[CampaignRun] = []
    for index in range(1, int(repeats) + 1):
        payload = run_stage6_takeover_ring_6case_sandbox(repo_root)
        runs.append(
            CampaignRun(
                index=int(index),
                overall_status=str((payload.get("result") or {}).get("overall_status") or "fail"),
                report=dict(payload.get("report") or {}),
                result=dict(payload.get("result") or {}),
            )
        )

    stability = _analyze_campaign(runs)
    dump_json(benchmark_root / "stage6_takeover_ring_6case_stability_report.json", stability)
    dump_json(
        benchmark_root / "stage6_takeover_ring_6case_stability_result.json",
        {
            "schema_version": "stage6_takeover_ring_6case_stability_result_v1",
            "generated_at_utc": _utc_now_iso(),
            "overall_status": stability.get("readiness"),
            "num_runs": stability.get("num_runs"),
            "pass_rate": stability.get("pass_rate"),
            "runtime_success_rate": stability.get("runtime_success_rate"),
            "subset_cases": stability.get("subset_cases") or [],
            "report_path": str(benchmark_root / "stage6_takeover_ring_6case_stability_report.json"),
            "remaining_blockers": stability.get("remaining_blockers") or [],
        },
    )
    return stability
