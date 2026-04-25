from __future__ import annotations

import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from benchmark.kinematic_mpc_shadow import (
    DEFAULT_SHADOW_MPC_CONFIG,
    DEFAULT_SHADOW_MPC_CONFIG_OVERRIDE_PATH,
    get_shadow_tuning_config,
    reset_shadow_solver_caches,
    reset_shadow_tuning_config,
    set_shadow_tuning_config,
)
from benchmark.runner_core import run_benchmark
from benchmark.shadow_artifacts import generate_mpc_shadow_artifacts
from benchmark.stage6_online_shadow_smoke import SHADOW_CASES, run_stage6_online_shadow_smoke
from benchmark.stage6_online_shadow_stability import _restart_carla

from .io import dump_json, load_json, load_yaml


TRACKED_METRICS = [
    "shadow_feasibility_rate",
    "shadow_planner_execution_latency_ms",
    "shadow_baseline_disagreement_rate",
    "shadow_realized_support_rate",
    "shadow_realized_behavior_match_rate",
    "shadow_realized_stop_distance_mae_m",
]

BASELINE_CONFIG_ID = "baseline_v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _mean_or_none(values: list[float | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(sum(cleaned) / len(cleaned))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_nested_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _deep_delta(default_payload: dict[str, Any], candidate_payload: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in candidate_payload.items():
        if key == "config_id":
            continue
        default_value = default_payload.get(key)
        if isinstance(value, dict) and isinstance(default_value, dict):
            nested = _deep_delta(default_value, value)
            if nested:
                delta[key] = nested
        elif value != default_value:
            delta[key] = value
    return delta


def _parameter_space() -> list[dict[str, Any]]:
    return [
        {
            "path": "longitudinal.stop.horizon_steps",
            "default": 12,
            "expected_effect": "Longer stop horizon can reduce stop-distance bias on stop-focused cases.",
            "risk": "Higher compute cost and more conservative stop profiles.",
            "affected_metrics": ["shadow_realized_stop_distance_mae_m", "shadow_planner_execution_latency_ms"],
        },
        {
            "path": "longitudinal.stop.q_position",
            "default": 0.12,
            "expected_effect": "Higher running position weight should pull the stop profile closer to the target window earlier.",
            "risk": "Can create overly stiff stop trajectories if too large.",
            "affected_metrics": ["shadow_realized_stop_distance_mae_m", "shadow_baseline_disagreement_rate"],
        },
        {
            "path": "longitudinal.stop.q_position_terminal",
            "default": 18.0,
            "expected_effect": "Higher terminal stop weight should reduce residual stop-distance error at the end of the horizon.",
            "risk": "May increase disagreement or fallback pressure if the stop target is noisy.",
            "affected_metrics": ["shadow_realized_stop_distance_mae_m", "shadow_fallback_recommendation_rate"],
        },
        {
            "path": "longitudinal.stop.q_speed_terminal",
            "default": 5.5,
            "expected_effect": "Stronger terminal speed penalty encourages cleaner hold near zero speed.",
            "risk": "Can over-damp approach speed and hurt behavior agreement.",
            "affected_metrics": ["shadow_realized_behavior_match_rate", "shadow_realized_stop_distance_mae_m"],
        },
        {
            "path": "lateral.r_control_delta",
            "default": 0.25,
            "expected_effect": "Higher steering-rate penalty should smooth lane-change proposals.",
            "risk": "Too much smoothing can slow lateral convergence.",
            "affected_metrics": ["shadow_baseline_disagreement_rate", "shadow_planner_execution_latency_ms"],
        },
        {
            "path": "lane_change.target_speed_cap_high_error_mps",
            "default": 4.5,
            "expected_effect": "Lower cap can calm aggressive lane-change entries on the multi-lane case.",
            "risk": "May diverge from baseline behavior if too restrictive.",
            "affected_metrics": ["shadow_realized_behavior_match_rate", "shadow_baseline_disagreement_rate"],
        },
        {
            "path": "osqp.max_iter",
            "default": 4000,
            "expected_effect": "Lower iteration cap can trim latency when the problem is already well-conditioned.",
            "risk": "Premature termination can reduce feasibility or force fallback recommendations.",
            "affected_metrics": ["shadow_planner_execution_latency_ms", "shadow_feasibility_rate"],
        },
        {
            "path": "osqp.eps_abs / osqp.eps_rel",
            "default": 1e-4,
            "expected_effect": "Tighter tolerance can improve tracking precision if solver time budget allows it.",
            "risk": "Solver latency can jump without meaningful quality gains.",
            "affected_metrics": ["shadow_realized_stop_distance_mae_m", "shadow_planner_execution_latency_ms"],
        },
    ]


def _candidate_configs() -> list[dict[str, Any]]:
    return [
        {"config_id": BASELINE_CONFIG_ID},
        {
            "config_id": "stop_precision_v1",
            "longitudinal": {
                "stop": {
                    "horizon_steps": 14,
                    "q_position": 0.18,
                    "q_position_terminal": 26.0,
                    "q_speed_terminal": 6.8,
                }
            },
        },
        {
            "config_id": "balanced_stop_lane_v1",
            "longitudinal": {
                "stop": {
                    "horizon_steps": 14,
                    "q_position": 0.16,
                    "q_position_terminal": 24.0,
                    "q_speed_terminal": 6.2,
                }
            },
            "lateral": {
                "r_control": 0.16,
                "r_control_delta": 0.38,
                "q_offset_terminal": 9.5,
            },
            "lane_change": {
                "target_speed_cap_high_error_mps": 4.2,
            },
        },
        {
            "config_id": "fast_reactive_v1",
            "longitudinal": {
                "follow": {"horizon_steps": 8},
                "stop": {
                    "horizon_steps": 10,
                    "q_position": 0.1,
                    "q_position_terminal": 16.0,
                },
            },
            "lane_change": {
                "min_horizon_steps": 7,
                "max_horizon_steps": 12,
            },
            "osqp": {
                "max_iter": 2500,
            },
        },
    ]


def _load_online_shadow_source(repo_root: Path) -> Path:
    result = load_json(repo_root / "benchmark" / "stage6_online_shadow_smoke_result.json", default={}) or {}
    online_report_dir = Path(str(result.get("online_report_dir") or "")).resolve()
    if not online_report_dir.exists():
        raise RuntimeError("stage6_online_shadow_smoke_result.json does not point to a valid online_report_dir.")
    return online_report_dir


def _base_case_spec_dir(online_report_dir: Path) -> Path:
    candidate = online_report_dir / "stage6_online_shadow_cases_runtime"
    if candidate.exists():
        return candidate
    return online_report_dir / "cases_runtime"


def _patch_case_spec_for_tuning(case_spec: dict[str, Any], stage3c_dir: Path) -> dict[str, Any]:
    updated = deepcopy(case_spec)
    updated["source"] = dict(updated.get("source") or {})
    updated["source"]["stage3c_dir"] = str(stage3c_dir)
    updated["artifacts"] = dict(updated.get("artifacts") or {})
    for artifact_name, artifact_path in list(updated["artifacts"].items()):
        if not str(artifact_name).startswith("stage3c_"):
            continue
        filename = Path(str(artifact_path)).name
        candidate = stage3c_dir / filename
        if candidate.exists():
            updated["artifacts"][artifact_name] = str(candidate)
    return updated


def _extract_case_metrics(stage3c_dir: Path) -> dict[str, Any]:
    summary = load_json(stage3c_dir / "mpc_shadow_summary.json", default={}) or {}
    realized = load_json(stage3c_dir / "shadow_realized_outcome_summary.json", default={}) or {}
    return {
        "shadow_feasibility_rate": summary.get("shadow_feasibility_rate"),
        "shadow_planner_execution_latency_ms": summary.get("shadow_planner_execution_latency_ms"),
        "shadow_baseline_disagreement_rate": summary.get("shadow_baseline_disagreement_rate"),
        "shadow_fallback_recommendation_rate": summary.get("shadow_fallback_recommendation_rate"),
        "shadow_realized_support_rate": realized.get("shadow_realized_support_rate"),
        "shadow_realized_behavior_match_rate": realized.get("shadow_realized_behavior_match_rate"),
        "shadow_realized_stop_distance_mae_m": realized.get("shadow_realized_stop_distance_mae_m"),
        "shadow_realized_speed_mae_mps": realized.get("shadow_realized_speed_mae_mps"),
        "shadow_realized_progress_mae_m": realized.get("shadow_realized_progress_mae_m"),
        "proposal_source": summary.get("proposal_source"),
        "mpc_tuning_config_ids": list(summary.get("mpc_tuning_config_ids") or []),
        "mpc_tuning_config_sources": list(summary.get("mpc_tuning_config_sources") or []),
    }


def _aggregate_case_metrics(case_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "shadow_feasibility_rate": _mean_or_none([_safe_float(row.get("shadow_feasibility_rate")) for row in case_metrics.values()]),
        "shadow_planner_execution_latency_ms": _mean_or_none([_safe_float(row.get("shadow_planner_execution_latency_ms")) for row in case_metrics.values()]),
        "shadow_baseline_disagreement_rate": _mean_or_none([_safe_float(row.get("shadow_baseline_disagreement_rate")) for row in case_metrics.values()]),
        "shadow_realized_support_rate": _mean_or_none([_safe_float(row.get("shadow_realized_support_rate")) for row in case_metrics.values()]),
        "shadow_realized_behavior_match_rate": _mean_or_none([_safe_float(row.get("shadow_realized_behavior_match_rate")) for row in case_metrics.values()]),
        "shadow_realized_stop_distance_mae_m": _mean_or_none([
            _safe_float(row.get("shadow_realized_stop_distance_mae_m"))
            for row in case_metrics.values()
        ]),
    }


def _materialize_tuning_eval_config(
    *,
    repo_root: Path,
    tuning_root: Path,
    source_online_report_dir: Path,
    config_id: str,
    case_stage3c_dirs: dict[str, Path],
) -> Path:
    base_runtime_cfg = load_yaml(source_online_report_dir / "stage6_online_shadow_runtime_config.yaml")
    base_case_dir = _base_case_spec_dir(source_online_report_dir)
    candidate_root = tuning_root / "configs" / config_id
    cases_dir = candidate_root / "cases_runtime"
    sets_dir = candidate_root / "sets_runtime"
    cases_dir.mkdir(parents=True, exist_ok=True)
    sets_dir.mkdir(parents=True, exist_ok=True)

    for case_id in SHADOW_CASES:
        case_spec = load_yaml(base_case_dir / f"{case_id}.yaml")
        case_spec = _patch_case_spec_for_tuning(case_spec, case_stage3c_dirs[case_id])
        _write_yaml(cases_dir / f"{case_id}.yaml", case_spec)

    set_name = f"stage6_mpc_tuning_{config_id}"
    set_path = sets_dir / f"{set_name}.yaml"
    _write_yaml(
        set_path,
        {
            "set_name": set_name,
            "description": f"Stage 6 MPC tuning evaluation set for {config_id}",
            "cases": list(SHADOW_CASES),
        },
    )
    runtime_cfg = deepcopy(base_runtime_cfg)
    runtime_cfg["cases_dir"] = str(cases_dir)
    runtime_cfg["sets"] = dict(runtime_cfg.get("sets") or {})
    runtime_cfg["sets"][set_name] = str(set_path)
    config_path = candidate_root / "benchmark_runtime_config.yaml"
    _write_yaml(config_path, runtime_cfg)
    return config_path


def _run_single_candidate(
    *,
    repo_root: Path,
    source_online_report_dir: Path,
    tuning_root: Path,
    candidate_config: dict[str, Any],
) -> dict[str, Any]:
    config_id = str(candidate_config["config_id"])
    selected_config = set_shadow_tuning_config(candidate_config, source=f"stage6_mpc_tuning:{config_id}")
    reset_shadow_solver_caches()

    candidate_root = tuning_root / "configs" / config_id
    case_stage3c_dirs: dict[str, Path] = {}
    case_metrics: dict[str, dict[str, Any]] = {}
    for case_id in SHADOW_CASES:
        src_stage3c = source_online_report_dir / "case_runs" / case_id / "_current_run_artifacts" / "stage3c"
        dst_stage3c = candidate_root / "case_runs" / case_id / "stage3c"
        dst_stage3c.parent.mkdir(parents=True, exist_ok=True)
        if dst_stage3c.exists():
            shutil.rmtree(dst_stage3c)
        shutil.copytree(src_stage3c, dst_stage3c)
        generate_mpc_shadow_artifacts(dst_stage3c, case_id=case_id)
        case_stage3c_dirs[case_id] = dst_stage3c
        case_metrics[case_id] = _extract_case_metrics(dst_stage3c)

    config_path = _materialize_tuning_eval_config(
        repo_root=repo_root,
        tuning_root=tuning_root,
        source_online_report_dir=source_online_report_dir,
        config_id=config_id,
        case_stage3c_dirs=case_stage3c_dirs,
    )
    set_name = f"stage6_mpc_tuning_{config_id}"
    eval_result = run_benchmark(
        repo_root=repo_root,
        config_path=config_path,
        set_name=set_name,
        report_dir=candidate_root / "evaluation",
    )
    benchmark_report = load_json(Path(str(eval_result["report_dir"])) / "benchmark_report.json", default={}) or {}
    gate_result = load_json(Path(str(eval_result["report_dir"])) / "gate_result.json", default={}) or {}
    aggregate_metrics = _aggregate_case_metrics(case_metrics)
    return {
        "config_id": config_id,
        "selected_config": selected_config,
        "changed_params": _deep_delta(DEFAULT_SHADOW_MPC_CONFIG, selected_config),
        "report_dir": str(eval_result.get("report_dir")),
        "benchmark_report_path": str(Path(str(eval_result["report_dir"])) / "benchmark_report.json"),
        "gate_result_path": str(Path(str(eval_result["report_dir"])) / "gate_result.json"),
        "gate_overall_status": gate_result.get("overall_status"),
        "failed_cases": list(gate_result.get("failed_cases") or []),
        "warning_cases": list(gate_result.get("warning_cases") or []),
        "case_metrics": case_metrics,
        "aggregate_metrics": aggregate_metrics,
        "notes": [
            "candidate evaluated on a fixed online shadow current-run snapshot to isolate backend changes from CARLA runtime noise"
        ],
        "benchmark_summary": {
            "set": benchmark_report.get("set"),
            "overall_status": gate_result.get("overall_status"),
        },
    }


def _build_baseline_audit(run_payload: dict[str, Any], source_online_report_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "stage6_mpc_tuning_baseline_v1",
        "generated_at_utc": _utc_now_iso(),
        "source_online_report_dir": str(source_online_report_dir),
        "config_id": run_payload.get("config_id"),
        "aggregate_metrics": run_payload.get("aggregate_metrics"),
        "cases": [
            {
                "case_id": case_id,
                **metrics,
            }
            for case_id, metrics in (run_payload.get("case_metrics") or {}).items()
        ],
        "gate_overall_status": run_payload.get("gate_overall_status"),
        "notes": [
            "Baseline audit is regenerated through the same fixed-snapshot tuning path as every candidate.",
        ],
    }


def _selection_inputs(run_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_by_id = {str(run["config_id"]): run for run in run_rows}
    baseline = run_by_id[BASELINE_CONFIG_ID]
    candidates = [run for run in run_rows if str(run["config_id"]) != BASELINE_CONFIG_ID]
    return baseline, candidates


def run_stage6_mpc_tuning_compare(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    runs_payload = load_json(repo_root / "benchmark" / "stage6_mpc_tuning_runs.json", default={}) or {}
    run_rows = list(runs_payload.get("runs") or [])
    if not run_rows:
        raise RuntimeError("stage6_mpc_tuning_runs.json does not contain any tuning runs.")

    baseline, candidates = _selection_inputs(run_rows)
    baseline_stop_mae = _safe_float(
        ((baseline.get("case_metrics") or {}).get("stop_follow_ambiguity_core") or {}).get("shadow_realized_stop_distance_mae_m")
    )
    baseline_latency = _safe_float((baseline.get("aggregate_metrics") or {}).get("shadow_planner_execution_latency_ms"))
    baseline_disagreement = _safe_float((baseline.get("aggregate_metrics") or {}).get("shadow_baseline_disagreement_rate"))

    comparison_rows: list[dict[str, Any]] = []
    retained_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        aggregate = candidate.get("aggregate_metrics") or {}
        stop_case = ((candidate.get("case_metrics") or {}).get("stop_follow_ambiguity_core") or {})
        candidate_stop_mae = _safe_float(stop_case.get("shadow_realized_stop_distance_mae_m"))
        candidate_latency = _safe_float(aggregate.get("shadow_planner_execution_latency_ms"))
        candidate_disagreement = _safe_float(aggregate.get("shadow_baseline_disagreement_rate"))
        candidate_support = _safe_float(aggregate.get("shadow_realized_support_rate"))
        candidate_behavior = _safe_float(aggregate.get("shadow_realized_behavior_match_rate"))
        baseline_support = _safe_float((baseline.get("aggregate_metrics") or {}).get("shadow_realized_support_rate"))
        baseline_behavior = _safe_float((baseline.get("aggregate_metrics") or {}).get("shadow_realized_behavior_match_rate"))

        retained = (
            str(candidate.get("gate_overall_status")) == "pass"
            and (candidate_support is None or baseline_support is None or candidate_support >= baseline_support - 1e-6)
            and (candidate_behavior is None or baseline_behavior is None or candidate_behavior >= baseline_behavior - 1e-6)
            and (candidate_disagreement is None or baseline_disagreement is None or candidate_disagreement <= baseline_disagreement + 0.05)
            and (candidate_stop_mae is None or baseline_stop_mae is None or candidate_stop_mae <= baseline_stop_mae + 0.02)
        )
        row = {
            "config_id": candidate.get("config_id"),
            "gate_overall_status": candidate.get("gate_overall_status"),
            "retained_for_selection": retained,
            "delta_vs_baseline": {
                "shadow_realized_stop_distance_mae_m": (
                    None if candidate_stop_mae is None or baseline_stop_mae is None else float(candidate_stop_mae - baseline_stop_mae)
                ),
                "shadow_planner_execution_latency_ms": (
                    None if candidate_latency is None or baseline_latency is None else float(candidate_latency - baseline_latency)
                ),
                "shadow_baseline_disagreement_rate": (
                    None if candidate_disagreement is None or baseline_disagreement is None else float(candidate_disagreement - baseline_disagreement)
                ),
            },
            "aggregate_metrics": aggregate,
            "case_metrics": candidate.get("case_metrics"),
        }
        comparison_rows.append(row)
        if retained:
            retained_candidates.append(candidate)

    def _selection_key(run: dict[str, Any]) -> tuple[float, float, float]:
        stop_case = ((run.get("case_metrics") or {}).get("stop_follow_ambiguity_core") or {})
        stop_mae = _safe_float(stop_case.get("shadow_realized_stop_distance_mae_m"))
        aggregate = run.get("aggregate_metrics") or {}
        latency = _safe_float(aggregate.get("shadow_planner_execution_latency_ms"))
        disagreement = _safe_float(aggregate.get("shadow_baseline_disagreement_rate"))
        return (
            float("inf") if stop_mae is None else float(stop_mae),
            float("inf") if latency is None else float(latency),
            float("inf") if disagreement is None else float(disagreement),
        )

    selected = baseline
    selection_reason = "keep_baseline_no_candidate_showed_a_clean_enough_improvement"
    if retained_candidates:
        best_candidate = min(retained_candidates, key=_selection_key)
        best_key = _selection_key(best_candidate)
        baseline_key = _selection_key(baseline)
        if best_key < baseline_key:
            selected = best_candidate
            selection_reason = "selected_candidate_improves_stop_mae_without_regressing_support_match_or_gate"

    comparison_payload = {
        "schema_version": "stage6_mpc_tuning_comparison_v1",
        "generated_at_utc": _utc_now_iso(),
        "baseline_config_id": BASELINE_CONFIG_ID,
        "comparison_rows": comparison_rows,
    }
    dump_json(repo_root / "benchmark" / "stage6_mpc_tuning_comparison.json", comparison_payload)

    decision_payload = {
        "schema_version": "stage6_mpc_tuning_decision_v1",
        "generated_at_utc": _utc_now_iso(),
        "baseline_config_id": BASELINE_CONFIG_ID,
        "selected_config_id": selected.get("config_id"),
        "selection_reason": selection_reason,
        "patch_default": str(selected.get("config_id")) != BASELINE_CONFIG_ID,
        "selected_run_report_dir": selected.get("report_dir"),
        "selected_aggregate_metrics": selected.get("aggregate_metrics"),
        "selected_case_metrics": selected.get("case_metrics"),
        "notes": [
            "Selection is lexicographic: keep support/match/gate clean, then minimize stop_follow stop-distance MAE, then latency, then disagreement.",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_mpc_tuning_decision.json", decision_payload)
    return {
        "comparison": comparison_payload,
        "decision": decision_payload,
    }


def _write_selected_default_override(selected_config: dict[str, Any] | None) -> str:
    if not selected_config or str(selected_config.get("config_id")) == BASELINE_CONFIG_ID:
        DEFAULT_SHADOW_MPC_CONFIG_OVERRIDE_PATH.unlink(missing_ok=True)
        return "baseline_default_reused"
    dump_json(DEFAULT_SHADOW_MPC_CONFIG_OVERRIDE_PATH, selected_config)
    return "selected_default_override_written"


def run_stage6_mpc_tuning(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    source_online_report_dir = _load_online_shadow_source(repo_root)
    parameter_space_payload = {
        "schema_version": "stage6_mpc_tuning_parameter_space_v1",
        "generated_at_utc": _utc_now_iso(),
        "default_config": get_shadow_tuning_config(),
        "parameters": _parameter_space(),
    }
    dump_json(repo_root / "benchmark" / "stage6_mpc_tuning_parameter_space.json", parameter_space_payload)

    candidate_configs = _candidate_configs()
    plan_payload = {
        "schema_version": "stage6_mpc_tuning_plan_v1",
        "generated_at_utc": _utc_now_iso(),
        "source_online_report_dir": str(source_online_report_dir),
        "evaluation_mode": "fixed_online_shadow_snapshot_then_single_online_confirmation",
        "baseline_config_id": BASELINE_CONFIG_ID,
        "candidate_configs": [
            {
                "config_id": config["config_id"],
                "changed_params": _deep_delta(DEFAULT_SHADOW_MPC_CONFIG, _merge_nested_dict(DEFAULT_SHADOW_MPC_CONFIG, config)),
            }
            for config in candidate_configs
        ],
        "selection_criteria": [
            "candidate must keep gate overall_status=pass",
            "candidate must not regress realized support or behavior match below baseline",
            "candidate must keep disagreement close to baseline",
            "winner is chosen by lower stop_follow stop-distance MAE, then lower latency, then lower disagreement",
        ],
    }
    dump_json(repo_root / "benchmark" / "stage6_mpc_tuning_plan.json", plan_payload)

    tuning_root = repo_root / "benchmark" / "reports" / f"stage6_mpc_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tuning_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    try:
        for candidate in candidate_configs:
            runs.append(
                _run_single_candidate(
                    repo_root=repo_root,
                    source_online_report_dir=source_online_report_dir,
                    tuning_root=tuning_root,
                    candidate_config=candidate,
                )
            )
    finally:
        reset_shadow_tuning_config()

    runs_payload = {
        "schema_version": "stage6_mpc_tuning_runs_v1",
        "generated_at_utc": _utc_now_iso(),
        "source_online_report_dir": str(source_online_report_dir),
        "report_root": str(tuning_root),
        "runs": runs,
    }
    dump_json(repo_root / "benchmark" / "stage6_mpc_tuning_runs.json", runs_payload)
    dump_json(repo_root / "benchmark" / "stage6_mpc_tuning_baseline.json", _build_baseline_audit(runs[0], source_online_report_dir))

    comparison_outputs = run_stage6_mpc_tuning_compare(repo_root)
    selected_config_id = str((comparison_outputs.get("decision") or {}).get("selected_config_id") or BASELINE_CONFIG_ID)
    selected_run = next((run for run in runs if str(run.get("config_id")) == selected_config_id), runs[0])
    override_status = _write_selected_default_override(selected_run.get("selected_config"))

    _restart_carla(carla_root=repo_root / ".." / "carla", port=2000)
    reset_shadow_tuning_config()
    confirmation_result = run_stage6_online_shadow_smoke(repo_root)
    decision_payload = load_json(repo_root / "benchmark" / "stage6_mpc_tuning_decision.json", default={}) or {}
    decision_payload["default_override_status"] = override_status
    decision_payload["final_confirmation"] = {
        "overall_status": ((confirmation_result.get("gate_result") or {}).get("overall_status")),
        "online_report_dir": ((confirmation_result.get("gate_result") or {}).get("online_report_dir")),
        "shadow_report_path": str(repo_root / "benchmark" / "stage6_online_shadow_report.json"),
    }
    dump_json(repo_root / "benchmark" / "stage6_mpc_tuning_decision.json", decision_payload)

    return {
        "baseline": load_json(repo_root / "benchmark" / "stage6_mpc_tuning_baseline.json", default={}) or {},
        "parameter_space": parameter_space_payload,
        "plan": plan_payload,
        "runs": runs_payload,
        "comparison": comparison_outputs.get("comparison"),
        "decision": decision_payload,
        "confirmation_result": confirmation_result,
    }
