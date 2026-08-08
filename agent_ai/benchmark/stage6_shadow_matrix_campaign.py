from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .io import dump_json, load_json
from .stage6_shadow_expansion import EXPANSION_CASES
from .stage6_shadow_matrix import run_stage6_shadow_matrix


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


def _build_matrix_stability_payload(report: dict[str, Any], benchmark_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runs = list(report.get("runs") or [])
    total_case_observations = sum(len(run.get("cases") or []) for run in runs)
    runtime_success_count = 0
    aggregate_latency: list[float] = []
    aggregate_support: list[float] = []
    aggregate_behavior: list[float] = []
    aggregate_stop_mae: list[float] = []
    aggregate_disagreement: list[float] = []
    aggregate_feasibility: list[float] = []
    aggregate_timeout: list[float] = []
    remaining_blockers: list[str] = []
    cases_payload: list[dict[str, Any]] = []

    for case_id in EXPANSION_CASES:
        case_rows: list[dict[str, Any]] = []
        for run in runs:
            for case in run.get("cases") or []:
                if str(case.get("case_id")) == case_id:
                    case_rows.append(case)
                    break

        runtime_success_rate = (
            float(
                sum(
                    1
                    for row in case_rows
                    if str(((row.get("attempts") or [{}])[-1]).get("runtime_status")) == "success"
                )
                / len(case_rows)
            )
            if case_rows
            else 0.0
        )
        gate_pass_rate = (
            float(
                sum(
                    1
                    for row in case_rows
                    if str(((row.get("attempts") or [{}])[-1]).get("gate_status")) == "pass"
                )
                / len(case_rows)
            )
            if case_rows
            else 0.0
        )
        runtime_success_count += sum(
            1
            for row in case_rows
            if str(((row.get("attempts") or [{}])[-1]).get("runtime_status")) == "success"
        )

        feasibility_values = [
            float((row.get("final_metrics") or {}).get("shadow_feasibility_rate"))
            for row in case_rows
            if (row.get("final_metrics") or {}).get("shadow_feasibility_rate") is not None
        ]
        timeout_values = [
            float((row.get("final_metrics") or {}).get("shadow_solver_timeout_rate"))
            for row in case_rows
            if (row.get("final_metrics") or {}).get("shadow_solver_timeout_rate") is not None
        ]
        latency_values = [
            float((row.get("final_metrics") or {}).get("shadow_planner_execution_latency_ms"))
            for row in case_rows
            if (row.get("final_metrics") or {}).get("shadow_planner_execution_latency_ms") is not None
        ]
        support_values = [
            float((row.get("final_metrics") or {}).get("shadow_realized_support_rate"))
            for row in case_rows
            if (row.get("final_metrics") or {}).get("shadow_realized_support_rate") is not None
        ]
        behavior_values = [
            float((row.get("final_metrics") or {}).get("shadow_realized_behavior_match_rate"))
            for row in case_rows
            if (row.get("final_metrics") or {}).get("shadow_realized_behavior_match_rate") is not None
        ]
        stop_mae_values = [
            float((row.get("final_metrics") or {}).get("shadow_realized_stop_distance_mae_m"))
            for row in case_rows
            if (row.get("final_metrics") or {}).get("shadow_realized_stop_distance_mae_m") is not None
        ]
        disagreement_values = [
            float((row.get("final_metrics") or {}).get("shadow_baseline_disagreement_rate"))
            for row in case_rows
            if (row.get("final_metrics") or {}).get("shadow_baseline_disagreement_rate") is not None
        ]
        alignment_ready_values = [
            bool((row.get("final_metrics") or {}).get("alignment_ready"))
            for row in case_rows
            if (row.get("final_metrics") or {}).get("alignment_ready") is not None
        ]
        alignment_consistent_values = [
            bool((row.get("final_metrics") or {}).get("alignment_consistent"))
            for row in case_rows
            if (row.get("final_metrics") or {}).get("alignment_consistent") is not None
        ]
        stop_target_qualities = sorted(
            set(
                str((row.get("final_metrics") or {}).get("stop_target_quality"))
                for row in case_rows
                if (row.get("final_metrics") or {}).get("stop_target_quality")
            )
        )

        aggregate_feasibility.extend(feasibility_values)
        aggregate_timeout.extend(timeout_values)
        aggregate_latency.extend(latency_values)
        aggregate_support.extend(support_values)
        aggregate_behavior.extend(behavior_values)
        aggregate_stop_mae.extend(stop_mae_values)
        aggregate_disagreement.extend(disagreement_values)

        if runtime_success_rate < 1.0:
            remaining_blockers.append(f"{case_id}:runtime")
        if gate_pass_rate < 1.0:
            remaining_blockers.append(f"{case_id}:shadow_gate")
        if any(str(row.get("final_status")) == "pass_with_warnings" for row in case_rows):
            remaining_blockers.append(f"{case_id}:soft_warning")
        if any(bool(row.get("recovered_after_retry")) for row in case_rows):
            remaining_blockers.append(f"{case_id}:infra_recovery")
        if feasibility_values and min(feasibility_values) < 1.0:
            remaining_blockers.append(f"{case_id}:solver_quality_feasibility")
        if timeout_values and max(timeout_values) > 0.0:
            remaining_blockers.append(f"{case_id}:solver_timeout")
        if support_values and min(support_values) < 0.95:
            remaining_blockers.append(f"{case_id}:solver_quality_support")
        if behavior_values and min(behavior_values) < 0.95:
            remaining_blockers.append(f"{case_id}:solver_quality_behavior_match")

        disagreement_limit = 0.35 if case_id in {"arbitration_stop_during_prepare_right", "fr_lc_commit_no_permission"} else 0.25
        if disagreement_values and (_quantile(disagreement_values, 0.95) or 0.0) > disagreement_limit:
            remaining_blockers.append(f"{case_id}:solver_quality_disagreement")
        if case_id in {"stop_follow_ambiguity_core", "arbitration_stop_during_prepare_right"}:
            alignment_ready_rate = (
                float(sum(1 for value in alignment_ready_values if value) / len(alignment_ready_values))
                if alignment_ready_values
                else 0.0
            )
            alignment_consistent_rate = (
                float(sum(1 for value in alignment_consistent_values if value) / len(alignment_consistent_values))
                if alignment_consistent_values
                else 0.0
            )
            if alignment_ready_rate < 1.0:
                remaining_blockers.append(f"{case_id}:contract_alignment")
            if not stop_target_qualities or any(value not in {"exact", "clean_enough"} for value in stop_target_qualities):
                remaining_blockers.append(f"{case_id}:contract_stop_target")
            p95_stop_mae = _quantile(stop_mae_values, 0.95) or 0.0
            max_stop_mae = max(stop_mae_values) if stop_mae_values else None
            if stop_mae_values and (p95_stop_mae > 0.25 or (max_stop_mae is not None and max_stop_mae > 0.35)):
                remaining_blockers.append(f"{case_id}:solver_quality_stop_mae")
        else:
            alignment_ready_rate = None
            alignment_consistent_rate = None

        cases_payload.append(
            {
                "case_id": case_id,
                "runtime_success_rate": runtime_success_rate,
                "gate_pass_rate": gate_pass_rate,
                "shadow_feasibility_rate": _stats(feasibility_values),
                "shadow_solver_timeout_rate": _stats(timeout_values),
                "shadow_planner_execution_latency_ms": _stats(latency_values),
                "shadow_realized_support_rate": _stats(support_values),
                "shadow_realized_behavior_match_rate": _stats(behavior_values),
                "shadow_realized_stop_distance_mae_m": _stats(stop_mae_values),
                "shadow_baseline_disagreement_rate": _stats(disagreement_values),
                "alignment_ready_rate": alignment_ready_rate,
                "alignment_consistent_rate": alignment_consistent_rate,
                "stop_target_quality": stop_target_qualities if case_id in {"stop_follow_ambiguity_core", "arbitration_stop_during_prepare_right"} else [],
                "fallback_takeover_latency_ms": _stats([]),
                "fallback_takeover_latency_evidence_ready": False,
            }
        )

    pass_rate = float(report.get("pass_rate") or 0.0)
    runtime_success_rate = float(runtime_success_count / total_case_observations) if total_case_observations else 0.0
    readiness = "ready"
    if pass_rate < 1.0 or runtime_success_rate < 1.0 or remaining_blockers:
        readiness = "not_ready"

    stability_report = {
        "schema_version": "stage6_shadow_stability_6case_v2_matrix_campaign",
        "generated_at_utc": _utc_now_iso(),
        "subset_cases": list(EXPANSION_CASES),
        "num_runs": int(report.get("repeats") or len(runs)),
        "pass_rate": pass_rate,
        "runtime_success_rate": runtime_success_rate,
        "readiness": readiness,
        "aggregate_metrics": {
            "shadow_feasibility_rate": _stats(aggregate_feasibility),
            "shadow_solver_timeout_rate": _stats(aggregate_timeout),
            "shadow_planner_execution_latency_ms": _stats(aggregate_latency),
            "shadow_realized_support_rate": _stats(aggregate_support),
            "shadow_realized_behavior_match_rate": _stats(aggregate_behavior),
            "shadow_realized_stop_distance_mae_m": _stats(aggregate_stop_mae),
            "shadow_baseline_disagreement_rate": _stats(aggregate_disagreement),
        },
        "runs": runs,
        "cases": cases_payload,
        "remaining_blockers": sorted(set(remaining_blockers)),
        "notes": [
            "Matrix campaign is the readiness authority for the six-case low-memory bring-up path.",
            "Each run executes one case per CARLA lifecycle with process-exit wait, memory drain, and runtime stabilization before Stage 4 starts.",
            "Fallback takeover latency remains non-authoritative until direct activation evidence exists on the promoted ring.",
        ],
        "campaign_report_path": str(benchmark_root / "stage6_shadow_matrix_campaign_report.json"),
    }
    stability_result = {
        "schema_version": "stage6_shadow_stability_6case_result_v2_matrix_campaign",
        "generated_at_utc": stability_report["generated_at_utc"],
        "subset_cases": list(EXPANSION_CASES),
        "overall_status": readiness,
        "num_runs": stability_report["num_runs"],
        "pass_rate": pass_rate,
        "runtime_success_rate": runtime_success_rate,
        "report_path": str(benchmark_root / "stage6_shadow_stability_6case_report.json"),
        "remaining_blockers": stability_report["remaining_blockers"],
        "campaign_report_path": stability_report["campaign_report_path"],
    }
    return stability_report, stability_result


def run_stage6_shadow_matrix_campaign(
    repo_root: str | Path,
    *,
    repeats: int = 5,
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
    retry_attempts: int = 1,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    runs: list[dict[str, Any]] = []

    for run_index in range(1, int(repeats) + 1):
        matrix = run_stage6_shadow_matrix(
            repo_root,
            carla_root=carla_root,
            carla_port=int(carla_port),
            retry_attempts=int(retry_attempts),
        )
        result = (matrix.get("result") or {}).copy()
        report = (matrix.get("report") or {}).copy()
        runs.append(
            {
                "run_index": int(run_index),
                "overall_status": result.get("overall_status"),
                "failed_cases": list(result.get("failed_cases") or []),
                "warning_cases": list(result.get("warning_cases") or []),
                "recovered_cases": list(result.get("recovered_cases") or []),
                "report_generated_at_utc": report.get("generated_at_utc"),
                "cases": list(report.get("cases") or []),
            }
        )

    pass_runs = sum(1 for row in runs if str(row.get("overall_status")) == "pass")
    partial_runs = sum(1 for row in runs if str(row.get("overall_status")) == "partial")
    fail_runs = sum(1 for row in runs if str(row.get("overall_status")) == "fail")

    case_rollup: list[dict[str, Any]] = []
    for case_id in EXPANSION_CASES:
        case_rows = []
        for run in runs:
            for case in run.get("cases") or []:
                if str(case.get("case_id")) == case_id:
                    case_rows.append(case)
                    break
        latency_values = [
            float((row.get("final_metrics") or {}).get("shadow_planner_execution_latency_ms"))
            for row in case_rows
            if (row.get("final_metrics") or {}).get("shadow_planner_execution_latency_ms") is not None
        ]
        stop_mae_values = [
            float((row.get("final_metrics") or {}).get("shadow_realized_stop_distance_mae_m"))
            for row in case_rows
            if (row.get("final_metrics") or {}).get("shadow_realized_stop_distance_mae_m") is not None
        ]
        case_rollup.append(
            {
                "case_id": case_id,
                "pass_rate": float(
                    sum(1 for row in case_rows if str(row.get("final_status")) == "pass") / len(case_rows)
                )
                if case_rows
                else 0.0,
                "warning_rate": float(
                    sum(1 for row in case_rows if str(row.get("final_status")) == "pass_with_warnings") / len(case_rows)
                )
                if case_rows
                else 0.0,
                "attempt_count_max": max((int(row.get("attempt_count") or 0) for row in case_rows), default=0),
                "shadow_planner_execution_latency_ms": _stats(latency_values),
                "shadow_realized_stop_distance_mae_m": _stats(stop_mae_values),
            }
        )

    report = {
        "schema_version": "stage6_shadow_matrix_campaign_v1",
        "generated_at_utc": _utc_now_iso(),
        "repeats": int(repeats),
        "pass_rate": float(pass_runs / len(runs)) if runs else 0.0,
        "partial_rate": float(partial_runs / len(runs)) if runs else 0.0,
        "fail_rate": float(fail_runs / len(runs)) if runs else 0.0,
        "runs": runs,
        "cases": case_rollup,
        "notes": [
            "Each run executes the per-case isolated matrix runner with CARLA process-exit wait and memory drain before the next case.",
            "This campaign validates repeatability of the low-memory six-case bring-up path, not takeover behavior.",
        ],
    }
    result = {
        "schema_version": "stage6_shadow_matrix_campaign_result_v1",
        "generated_at_utc": report["generated_at_utc"],
        "repeats": int(repeats),
        "overall_status": "pass"
        if pass_runs == len(runs)
        else ("partial" if pass_runs + partial_runs == len(runs) else "fail"),
        "pass_rate": report["pass_rate"],
        "partial_rate": report["partial_rate"],
        "fail_rate": report["fail_rate"],
        "report_path": str(benchmark_root / "stage6_shadow_matrix_campaign_report.json"),
    }
    dump_json(benchmark_root / "stage6_shadow_matrix_campaign_report.json", report)
    dump_json(benchmark_root / "stage6_shadow_matrix_campaign_result.json", result)
    stability_report, stability_result = _build_matrix_stability_payload(report, benchmark_root)
    dump_json(benchmark_root / "stage6_shadow_stability_6case_report.json", stability_report)
    dump_json(benchmark_root / "stage6_shadow_stability_6case_result.json", stability_result)
    return {"report": report, "result": result}


def sync_stage6_shadow_matrix_campaign_to_stability(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    report = load_json(benchmark_root / "stage6_shadow_matrix_campaign_report.json", default={}) or {}
    if not report:
        raise FileNotFoundError("stage6_shadow_matrix_campaign_report.json not found")
    stability_report, stability_result = _build_matrix_stability_payload(report, benchmark_root)
    dump_json(benchmark_root / "stage6_shadow_stability_6case_report.json", stability_report)
    dump_json(benchmark_root / "stage6_shadow_stability_6case_result.json", stability_result)
    return {"report": stability_report, "result": stability_result}
