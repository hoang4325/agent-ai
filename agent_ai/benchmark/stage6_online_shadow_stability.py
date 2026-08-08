from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from math import ceil
from statistics import mean, pstdev
from typing import Any

from .io import dump_json, load_json
from .stage6_online_shadow_smoke import SHADOW_CASES, run_stage6_online_shadow_smoke
from .kinematic_mpc_shadow import DEFAULT_SHADOW_MPC_CONFIG_OVERRIDE_PATH
from .carla_runtime import ensure_carla_ready as _ensure_carla_ready
from .carla_runtime import restart_carla as _restart_carla


@dataclass
class CampaignRun:
    index: int
    overall_status: str
    online_report_dir: Path
    shadow_eval_report_dir: Path
    shadow_report: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _case_status_by_id(benchmark_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get("scenario_id")): case
        for case in (benchmark_report.get("cases") or [])
    }


def _metric_values(case_report: dict[str, Any]) -> dict[str, float | None]:
    metrics = case_report.get("metrics") or {}
    return {
        "shadow_feasibility_rate": (metrics.get("shadow_feasibility_rate") or {}).get("value"),
        "shadow_planner_execution_latency_ms": (metrics.get("shadow_planner_execution_latency_ms") or {}).get("value"),
        "shadow_baseline_disagreement_rate": (metrics.get("shadow_baseline_disagreement_rate") or {}).get("value"),
        "shadow_fallback_recommendation_rate": (metrics.get("shadow_fallback_recommendation_rate") or {}).get("value"),
        "shadow_realized_support_rate": (metrics.get("shadow_realized_support_rate") or {}).get("value"),
        "shadow_realized_behavior_match_rate": (metrics.get("shadow_realized_behavior_match_rate") or {}).get("value"),
    }


def _summary_values(shadow_report: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in shadow_report.get("cases") or []:
        if str(case.get("case_id")) != case_id:
            continue
        summary = case.get("shadow_summary") or {}
        comparison = case.get("comparison") or {}
        return {
            "proposal_rows": case.get("proposal_rows"),
            "mpc_tuning_config_ids": summary.get("mpc_tuning_config_ids") or [],
            "shadow_planner_execution_latency_ms": summary.get("shadow_planner_execution_latency_ms"),
            "shadow_feasibility_rate": summary.get("shadow_feasibility_rate"),
            "shadow_solver_timeout_rate": summary.get("shadow_solver_timeout_rate"),
            "shadow_baseline_disagreement_rate": summary.get("shadow_baseline_disagreement_rate"),
            "shadow_fallback_recommendation_rate": summary.get("shadow_fallback_recommendation_rate"),
            "shadow_realized_support_rate": summary.get("shadow_realized_support_rate"),
            "shadow_realized_behavior_match_rate": summary.get("shadow_realized_behavior_match_rate"),
            "shadow_realized_stop_distance_mae_m": summary.get("shadow_realized_stop_distance_mae_m"),
            "better_metrics": comparison.get("better_metrics") or [],
            "worse_metrics": comparison.get("worse_metrics") or [],
        }
    return {
        "proposal_rows": 0,
        "mpc_tuning_config_ids": [],
        "shadow_planner_execution_latency_ms": None,
        "shadow_feasibility_rate": None,
        "shadow_solver_timeout_rate": None,
        "shadow_baseline_disagreement_rate": None,
        "shadow_fallback_recommendation_rate": None,
        "shadow_realized_support_rate": None,
        "shadow_realized_behavior_match_rate": None,
        "shadow_realized_stop_distance_mae_m": None,
        "better_metrics": [],
        "worse_metrics": [],
    }


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0, min(len(ordered) - 1, ceil(q * len(ordered)) - 1))
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


def _analyze_campaign(runs: list[CampaignRun]) -> dict[str, Any]:
    run_records: list[dict[str, Any]] = []
    case_rollup: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in SHADOW_CASES}
    all_latency_values: list[float] = []
    all_stop_mae_values: list[float] = []
    all_disagreement_values: list[float] = []
    all_support_values: list[float] = []
    all_behavior_match_values: list[float] = []
    all_feasibility_values: list[float] = []
    all_timeout_values: list[float] = []
    observed_config_ids: set[str] = set()

    pass_runs = 0
    for run in runs:
        gate_result = load_json(run.shadow_eval_report_dir / "gate_result.json", default={}) or {}
        benchmark_report = load_json(run.shadow_eval_report_dir / "benchmark_report.json", default={}) or {}
        mode_resolution = load_json(run.online_report_dir / "mode_resolution.json", default=[]) or []
        shadow_report = run.shadow_report or {}

        if str(gate_result.get("overall_status")) == "pass":
            pass_runs += 1

        case_reports = _case_status_by_id(benchmark_report)
        runtime_status = {
            str(row.get("scenario_id")): str(row.get("status"))
            for row in mode_resolution
        }
        run_cases: list[dict[str, Any]] = []
        for case_id in SHADOW_CASES:
            case_report = case_reports.get(case_id, {})
            summary = _summary_values(shadow_report, case_id)
            metrics = _metric_values(case_report)
            row = {
                "case_id": case_id,
                "runtime_status": runtime_status.get(case_id, "missing"),
                "gate_status": case_report.get("status", "missing"),
                "soft_warnings": case_report.get("soft_warnings") or [],
                "proposal_rows": summary.get("proposal_rows"),
                "mpc_tuning_config_ids": summary.get("mpc_tuning_config_ids") or [],
                "shadow_feasibility_rate": metrics.get("shadow_feasibility_rate"),
                "shadow_solver_timeout_rate": summary.get("shadow_solver_timeout_rate"),
                "shadow_planner_execution_latency_ms": metrics.get("shadow_planner_execution_latency_ms"),
                "shadow_baseline_disagreement_rate": metrics.get("shadow_baseline_disagreement_rate"),
                "shadow_fallback_recommendation_rate": metrics.get("shadow_fallback_recommendation_rate"),
                "shadow_realized_support_rate": metrics.get("shadow_realized_support_rate") or summary.get("shadow_realized_support_rate"),
                "shadow_realized_behavior_match_rate": metrics.get("shadow_realized_behavior_match_rate") or summary.get("shadow_realized_behavior_match_rate"),
                "shadow_realized_stop_distance_mae_m": summary.get("shadow_realized_stop_distance_mae_m"),
                "better_metrics": summary.get("better_metrics") or [],
                "worse_metrics": summary.get("worse_metrics") or [],
            }
            observed_config_ids.update(str(item) for item in (row.get("mpc_tuning_config_ids") or []) if str(item).strip())
            if row["shadow_planner_execution_latency_ms"] is not None:
                all_latency_values.append(float(row["shadow_planner_execution_latency_ms"]))
            if row["shadow_baseline_disagreement_rate"] is not None:
                all_disagreement_values.append(float(row["shadow_baseline_disagreement_rate"]))
            if row["shadow_realized_support_rate"] is not None:
                all_support_values.append(float(row["shadow_realized_support_rate"]))
            if row["shadow_realized_behavior_match_rate"] is not None:
                all_behavior_match_values.append(float(row["shadow_realized_behavior_match_rate"]))
            if row["shadow_realized_stop_distance_mae_m"] is not None:
                all_stop_mae_values.append(float(row["shadow_realized_stop_distance_mae_m"]))
            if row["shadow_feasibility_rate"] is not None:
                all_feasibility_values.append(float(row["shadow_feasibility_rate"]))
            if row["shadow_solver_timeout_rate"] is not None:
                all_timeout_values.append(float(row["shadow_solver_timeout_rate"]))
            case_rollup[case_id].append(row)
            run_cases.append(row)

        run_records.append(
            {
                "run_index": run.index,
                "overall_status": gate_result.get("overall_status"),
                "online_report_dir": str(run.online_report_dir),
                "shadow_eval_report_dir": str(run.shadow_eval_report_dir),
                "cases": run_cases,
            }
        )

    pass_rate = float(pass_runs / len(runs)) if runs else 0.0
    total_case_rows = sum(len(rows) for rows in case_rollup.values())
    runtime_success_count = sum(
        1 for rows in case_rollup.values() for row in rows if row["runtime_status"] == "success"
    )
    runtime_success_rate = float(runtime_success_count / total_case_rows) if total_case_rows else 0.0
    readiness = "ready" if runs and pass_rate >= 1.0 and runtime_success_rate >= 1.0 else "not_ready"

    cases: list[dict[str, Any]] = []
    remaining_blockers: list[str] = []
    for case_id, rows in case_rollup.items():
        runtime_success_rate = float(sum(1 for row in rows if row["runtime_status"] == "success") / len(rows)) if rows else 0.0
        gate_pass_rate = float(sum(1 for row in rows if row["gate_status"] == "pass") / len(rows)) if rows else 0.0
        latency_values = [float(row["shadow_planner_execution_latency_ms"]) for row in rows if row["shadow_planner_execution_latency_ms"] is not None]
        disagreement_values = [float(row["shadow_baseline_disagreement_rate"]) for row in rows if row["shadow_baseline_disagreement_rate"] is not None]
        fallback_values = [float(row["shadow_fallback_recommendation_rate"]) for row in rows if row["shadow_fallback_recommendation_rate"] is not None]
        feasibility_values = [float(row["shadow_feasibility_rate"]) for row in rows if row["shadow_feasibility_rate"] is not None]
        timeout_values = [float(row["shadow_solver_timeout_rate"]) for row in rows if row["shadow_solver_timeout_rate"] is not None]
        support_values = [float(row["shadow_realized_support_rate"]) for row in rows if row["shadow_realized_support_rate"] is not None]
        behavior_match_values = [float(row["shadow_realized_behavior_match_rate"]) for row in rows if row["shadow_realized_behavior_match_rate"] is not None]
        stop_mae_values = [float(row["shadow_realized_stop_distance_mae_m"]) for row in rows if row["shadow_realized_stop_distance_mae_m"] is not None]
        proposal_rows = [int(row["proposal_rows"]) for row in rows if row["proposal_rows"] is not None]
        config_ids = sorted(set(str(item) for row in rows for item in (row.get("mpc_tuning_config_ids") or []) if str(item).strip()))
        if runtime_success_rate < 1.0:
            remaining_blockers.append(f"{case_id}:runtime_success_rate<{1.0}")
        if gate_pass_rate < 1.0:
            remaining_blockers.append(f"{case_id}:gate_pass_rate<{1.0}")
        cases.append(
            {
                "case_id": case_id,
                "observed_mpc_tuning_config_ids": config_ids,
                "runtime_success_rate": runtime_success_rate,
                "gate_pass_rate": gate_pass_rate,
                "proposal_rows": _stats([float(value) for value in proposal_rows]),
                "shadow_feasibility_rate": _stats(feasibility_values),
                "shadow_solver_timeout_rate": _stats(timeout_values),
                "shadow_planner_execution_latency_ms": _stats(latency_values),
                "shadow_realized_support_rate": _stats(support_values),
                "shadow_realized_behavior_match_rate": _stats(behavior_match_values),
                "shadow_realized_stop_distance_mae_m": _stats(stop_mae_values),
                "shadow_baseline_disagreement_rate": _stats(disagreement_values),
                "shadow_fallback_recommendation_rate": _stats(fallback_values),
                "notes": [],
            }
        )

    default_override = load_json(DEFAULT_SHADOW_MPC_CONFIG_OVERRIDE_PATH, default={}) or {}
    expected_config_id = str(default_override.get("config_id") or "baseline_v1")
    if observed_config_ids and observed_config_ids != {expected_config_id}:
        remaining_blockers.append(f"observed_config_ids_mismatch:{','.join(sorted(observed_config_ids))}")
    if readiness != "ready":
        remaining_blockers.append("online_shadow_smoke_repeatability_not_clean")

    return {
        "schema_version": "stage6_online_shadow_stability_v1",
        "generated_at_utc": _utc_now_iso(),
        "stage": "6_online_shadow_stability",
        "expected_mpc_tuning_config_id": expected_config_id,
        "observed_mpc_tuning_config_ids": sorted(observed_config_ids),
        "default_config_override_path": str(DEFAULT_SHADOW_MPC_CONFIG_OVERRIDE_PATH),
        "subset_cases": SHADOW_CASES,
        "num_runs": len(runs),
        "pass_rate": pass_rate,
        "runtime_success_rate": runtime_success_rate,
        "readiness": readiness,
        "aggregate_metrics": {
            "shadow_feasibility_rate": _stats(all_feasibility_values),
            "shadow_solver_timeout_rate": _stats(all_timeout_values),
            "shadow_planner_execution_latency_ms": _stats(all_latency_values),
            "shadow_realized_support_rate": _stats(all_support_values),
            "shadow_realized_behavior_match_rate": _stats(all_behavior_match_values),
            "shadow_realized_stop_distance_mae_m": _stats(all_stop_mae_values),
            "shadow_baseline_disagreement_rate": _stats(all_disagreement_values),
        },
        "hard_gate_metrics": [
            "artifact_completeness",
            "shadow_output_contract_completeness",
            "shadow_feasibility_rate",
        ],
        "soft_guardrail_metrics": [
            "shadow_solver_timeout_rate",
            "shadow_planner_execution_latency_ms",
            "shadow_baseline_disagreement_rate",
            "shadow_fallback_recommendation_rate",
        ],
        "runs": run_records,
        "cases": cases,
        "remaining_blockers": sorted(set(remaining_blockers)),
        "notes": [
            "Baseline remains the only execution authority during online shadow smoke.",
            "This campaign measures repeatability of the promoted 2-case online shadow subset before any expansion beyond the subset.",
            "When restart_between_runs is enabled, each campaign iteration starts from a fresh CARLA process so runtime reuse crashes do not masquerade as shadow instability.",
        ],
    }


def run_stage6_online_shadow_stability(
    repo_root: str | Path,
    *,
    repeats: int = 3,
    carla_root: str | Path = "D:/carla",
    carla_port: int = 2000,
    restart_between_runs: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    benchmark_root = repo_root / "benchmark"
    carla_root = Path(carla_root)
    if restart_between_runs:
        _restart_carla(carla_root=carla_root, port=carla_port)
    else:
        _ensure_carla_ready(carla_root=carla_root, port=carla_port)

    runs: list[CampaignRun] = []
    for index in range(1, repeats + 1):
        if restart_between_runs and index > 1:
            _restart_carla(carla_root=carla_root, port=carla_port)
        result = run_stage6_online_shadow_smoke(repo_root)
        gate_result = result.get("gate_result") or {}
        runs.append(
            CampaignRun(
                index=index,
                overall_status=str(gate_result.get("overall_status")),
                online_report_dir=Path(str(gate_result.get("online_report_dir"))),
                shadow_eval_report_dir=Path(str(gate_result.get("shadow_evaluation_report_dir"))),
                shadow_report=(result.get("shadow_report") or {}),
            )
        )

    stability = _analyze_campaign(runs)
    dump_json(benchmark_root / "stage6_online_shadow_stability_report.json", stability)
    dump_json(
        benchmark_root / "stage6_online_shadow_stability_result.json",
        {
            "schema_version": "stage6_online_shadow_stability_result_v1",
            "generated_at_utc": _utc_now_iso(),
            "stage": "6_online_shadow_stability",
            "overall_status": stability.get("readiness"),
            "num_runs": stability.get("num_runs"),
            "pass_rate": stability.get("pass_rate"),
            "subset_cases": SHADOW_CASES,
            "report_path": str(benchmark_root / "stage6_online_shadow_stability_report.json"),
            "remaining_blockers": stability.get("remaining_blockers") or [],
        },
    )
    return stability
