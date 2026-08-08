from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .io import dump_json
from .stage6_full_authority_ring import CASE_SPECS, run_stage6_full_authority_ring


@dataclass
class CampaignRun:
    index: int
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
        return {"mean": None, "std": None, "min": None, "max": None, "p50": None, "p95": None}
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


def run_stage6_full_authority_ring_stability(
    repo_root: str | Path,
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    runs: list[CampaignRun] = []
    for index in range(1, int(repeats) + 1):
        payload = run_stage6_full_authority_ring(repo_root)
        runs.append(CampaignRun(index=index, report=dict(payload.get("report") or {}), result=dict(payload.get("result") or {})))

    expected_cases = [str(spec["case_id"]) for spec in CASE_SPECS]
    pass_rate = float(sum(1 for run in runs if str(run.result.get("overall_status")) == "pass") / len(runs)) if runs else 0.0
    runtime_success_rate = pass_rate
    per_case_rows: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in expected_cases}
    run_records: list[dict[str, Any]] = []
    for run in runs:
        case_rows = list(run.report.get("cases") or [])
        for row in case_rows:
            case_id = str(row.get("case_id") or "")
            if case_id:
                per_case_rows.setdefault(case_id, []).append(dict(row))
        run_records.append(
            {
                "run_index": int(run.index),
                "overall_status": str(run.result.get("overall_status") or "fail"),
                "report_path": str(run.result.get("report_path") or ""),
                "fallback_activated_case_count": int(((run.report.get("aggregate") or {}).get("fallback_activated_case_count") or 0)),
                "case_overview": [
                    {
                        "case_id": str(row.get("case_id") or ""),
                        "overall_status": str(row.get("overall_status") or "fail"),
                        "recovered_after_retry": bool(row.get("recovered_after_retry")),
                        "transfer_attempt_count": int(row.get("transfer_attempt_count") or 0),
                        "remaining_blockers": list(row.get("remaining_blockers") or []),
                    }
                    for row in case_rows
                ],
            }
        )

    remaining_blockers: list[str] = []
    aggregate_fallback_counts: list[float] = []
    case_payloads: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        case_id = str(spec["case_id"])
        rows = list(per_case_rows.get(case_id) or [])
        case_pass_rate = float(sum(1 for row in rows if str(row.get("overall_status")) == "pass") / len(rows)) if rows else 0.0
        fallback_counts = [
            float((row.get("derived_metrics") or {}).get("fallback_activated_under_full_authority_count") or 0.0)
            for row in rows
        ]
        aggregate_fallback_counts.extend(fallback_counts)
        full_authority_tail_counts = [
            float((row.get("derived_metrics") or {}).get("full_authority_tail_tick_count") or 0.0)
            for row in rows
        ]
        stop_alignment_ready_values = [
            bool((row.get("derived_metrics") or {}).get("stop_alignment_ready"))
            for row in rows
            if (row.get("derived_metrics") or {}).get("stop_alignment_ready") is not None
        ]
        stop_alignment_consistent_values = [
            bool((row.get("derived_metrics") or {}).get("stop_alignment_consistent"))
            for row in rows
            if (row.get("derived_metrics") or {}).get("stop_alignment_consistent") is not None
        ]
        stop_target_exact_values = [
            bool((row.get("derived_metrics") or {}).get("stop_target_exact"))
            for row in rows
            if (row.get("derived_metrics") or {}).get("stop_target_exact") is not None
        ]
        stop_final_error_supported_values = [
            bool((row.get("derived_metrics") or {}).get("stop_final_error_supported"))
            for row in rows
            if (row.get("derived_metrics") or {}).get("stop_final_error_supported") is not None
        ]

        if case_pass_rate < 1.0:
            remaining_blockers.append(f"{case_id}:pass_rate<1.0")

        case_payload = {
            "case_id": case_id,
            "kind": spec.get("kind"),
            "pass_rate": case_pass_rate,
            "full_authority_tail_tick_count": _stats(full_authority_tail_counts),
            "fallback_activated_under_full_authority_count": _stats(fallback_counts),
            "stop_alignment_ready_rate": _bool_rate(stop_alignment_ready_values),
            "stop_alignment_consistent_rate": _bool_rate(stop_alignment_consistent_values),
            "stop_target_exact_rate": _bool_rate(stop_target_exact_values),
            "stop_final_error_supported_rate": _bool_rate(stop_final_error_supported_values),
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

    readiness = "ready"
    if pass_rate < 1.0 or runtime_success_rate < 1.0 or remaining_blockers:
        readiness = "not_ready"
        if pass_rate < 1.0:
            remaining_blockers.append("campaign_pass_rate<1.0")
        if runtime_success_rate < 1.0:
            remaining_blockers.append("campaign_runtime_success_rate<1.0")

    report = {
        "schema_version": "stage6_full_authority_ring_stability_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "subset_cases": expected_cases,
        "num_runs": len(runs),
        "pass_rate": pass_rate,
        "runtime_success_rate": runtime_success_rate,
        "readiness": readiness,
        "aggregate_metrics": {
            "fallback_activated_under_full_authority_count": _stats(aggregate_fallback_counts),
        },
        "runs": run_records,
        "cases": case_payloads,
        "remaining_blockers": sorted(set(remaining_blockers)),
        "notes": [
            "This campaign measures repeatability of full-authority MPC sandbox execution on the promoted six-case ring.",
            "It is still sandbox evidence, not production authority rollout approval.",
        ],
    }
    result = {
        "schema_version": "stage6_full_authority_ring_stability_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "overall_status": readiness,
        "num_runs": len(runs),
        "pass_rate": pass_rate,
        "runtime_success_rate": runtime_success_rate,
        "subset_cases": expected_cases,
        "report_path": str(benchmark_root / "stage6_full_authority_ring_stability_report.json"),
        "remaining_blockers": report["remaining_blockers"],
    }
    dump_json(benchmark_root / "stage6_full_authority_ring_stability_report.json", report)
    dump_json(benchmark_root / "stage6_full_authority_ring_stability_result.json", result)
    return {
        "report": report,
        "result": result,
    }
